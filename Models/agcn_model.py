"""
AGCN — Two-Stream Adaptive Graph Convolutional Network for Skeleton-Based Action Recognition.
Reference: Lei Shi et al., CVPR 2019.

Key difference from ST-GCN:
  - The adjacency matrix A is NOT fixed; instead it is learned adaptively via a
    data-driven attention mechanism (A_learned = softmax(theta(x) @ phi(x)^T)).
  - Three graph components: A_natural (skeleton), A_global (learned jointly), A_local (per-sample).
  - This allows the model to capture implicit joint correlations beyond the physical skeleton.

Compatible with the same 17-joint COCO layout used by ST-GCN in this project.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.stgcn_model import (
    NUM_JOINTS, COCO_EDGES, CENTER_JOINT,
    build_adjacency_matrix, build_spatial_partition, normalize_adjacency
)


# ==============================================================================
# Adaptive Graph Convolution
# ==============================================================================
class AdaptiveGraphConv(nn.Module):
    """
    Adaptive GCN layer from Shi et al. CVPR 2019.

    Three-component adjacency:
      A_combined = A_natural (normalized fixed) + A_global (learnable param) + A_local (per-sample attention)
    """

    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray):
        super().__init__()
        K = A.shape[0]  # 3 partitions
        self.K = K
        self.out_channels = out_channels
        V = A.shape[1]

        self.register_buffer("A", torch.from_numpy(A).float())
        self.A_global = nn.Parameter(torch.zeros_like(torch.from_numpy(A).float()))
        nn.init.uniform_(self.A_global, -1e-6, 1e-6)

        self.theta = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)

        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=True)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        A_nat_global = self.A + self.A_global

        x_pool = x.mean(dim=2)
        theta   = self.theta(x_pool.unsqueeze(-1)).squeeze(-1)
        phi     = self.phi(x_pool.unsqueeze(-1)).squeeze(-1)
        A_local = torch.einsum("nci,ncj->nij", theta, phi) / (theta.size(1) ** 0.5)
        A_local = torch.softmax(A_local, dim=-1)

        x_conv = self.conv(x)
        x_conv = x_conv.reshape(N, self.K, self.out_channels, T, V)

        out = torch.zeros(N, self.out_channels, T, V, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            o_nat   = torch.einsum("nctv,vw->nctw", x_conv[:, k], A_nat_global[k])
            o_local = torch.einsum("nctv,nvw->nctw", x_conv[:, k], A_local)
            out = out + o_nat + self.alpha * o_local

        return self.bn(out)


# ==============================================================================
# AGCN Block
# ==============================================================================
class AGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0, temporal_kernel_size=9):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)

        pad = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Module()
        self.tcn.conv = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=(temporal_kernel_size, 1),
                                   stride=(stride, 1), padding=(pad, 0), bias=True)
        self.tcn.bn = nn.BatchNorm2d(out_channels)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Module()
            self.residual.conv = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=(stride, 1), bias=True)
            self.residual.bn = nn.BatchNorm2d(out_channels)
            self._has_residual_transform = True
        else:
            self.residual = nn.Identity()
            self._has_residual_transform = False

    def forward(self, x):
        if self._has_residual_transform:
            res = self.residual.bn(self.residual.conv(x))
        else:
            res = self.residual(x)
        out = self.relu(self.gcn(x))
        out = self.dropout(self.tcn.bn(self.tcn.conv(out)))
        return self.relu(out + res)


# ==============================================================================
# Full AGCN Model
# ==============================================================================
class AGCN(nn.Module):
    CHANNEL_CONFIG = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]

    def __init__(self, in_channels=3, num_classes=4, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes

        adj   = build_adjacency_matrix()
        A     = build_spatial_partition(adj, CENTER_JOINT)
        A_norm = np.stack([normalize_adjacency(A[k]) for k in range(A.shape[0])], axis=0)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        channels = [in_channels] + self.CHANNEL_CONFIG
        self.gcn = nn.ModuleList()
        for i in range(len(self.CHANNEL_CONFIG)):
            c_in  = channels[i]; c_out = channels[i + 1]
            stride = 2 if (c_in != c_out and c_in < c_out) else 1
            self.gcn.append(AGCNBlock(c_in, c_out, A_norm, stride=stride, dropout=dropout))

        self.fc = nn.Linear(self.CHANNEL_CONFIG[-1], num_classes)

    def forward(self, x):
        n, c, t, v = x.shape
        x_bn = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x_bn = self.data_bn(x_bn)
        x    = x_bn.reshape(n, v, c, t).permute(0, 2, 3, 1)
        for block in self.gcn:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


if __name__ == "__main__":
    model = AGCN(in_channels=3, num_classes=4, dropout=0.3)
    print(f"AGCN parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(4, 3, 100, 17)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (4, 4)
    print("OK -- AGCN forward pass works.")
