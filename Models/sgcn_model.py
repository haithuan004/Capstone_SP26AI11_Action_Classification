"""
S-GCN — Sparse Graph Convolutional Network for Skeleton-Based Action Recognition.

Key distinction: globally shared learnable adjacency A_sparse is trained end-to-end
with an L1 sparsity penalty, causing it to prune weak joint connections to near-zero.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Models.stgcn_model import (
        NUM_JOINTS, CENTER_JOINT,
        build_adjacency_matrix, build_spatial_partition, normalize_adjacency
    )
except ImportError:
    from stgcn_model import (
        NUM_JOINTS, CENTER_JOINT,
        build_adjacency_matrix, build_spatial_partition, normalize_adjacency
    )


class SparseGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, sparsity_lambda=1e-4):
        super().__init__()
        K = A.shape[0]
        self.K = K
        self.out_channels = out_channels
        self.sparsity_lambda = sparsity_lambda

        self.register_buffer("A_fixed", torch.from_numpy(A).float())
        self.A_learnable = nn.Parameter(torch.zeros_like(torch.from_numpy(A).float()))
        nn.init.uniform_(self.A_learnable, -0.01, 0.01)

        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=True)
        self.bn   = nn.BatchNorm2d(out_channels)

    def get_sparse_adjacency(self):
        return F.relu(self.A_fixed + self.A_learnable)

    def sparsity_loss(self):
        return self.sparsity_lambda * self.A_learnable.abs().mean()

    def forward(self, x):
        N, C, T, V = x.shape
        A_sparse = self.get_sparse_adjacency()
        x_conv = self.conv(x).reshape(N, self.K, self.out_channels, T, V)
        out = torch.zeros(N, self.out_channels, T, V, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            out = out + torch.einsum("nctv,vw->nctw", x_conv[:, k], A_sparse[k])
        return self.bn(out)


class SGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0,
                 temporal_kernel_size=9, sparsity_lambda=1e-4):
        super().__init__()
        self.gcn = SparseGraphConv(in_channels, out_channels, A, sparsity_lambda)

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
        sp_loss = self.gcn.sparsity_loss()
        out = self.dropout(self.tcn.bn(self.tcn.conv(out)))
        return self.relu(out + res), sp_loss


class SGCN(nn.Module):
    CHANNEL_CONFIG = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]

    def __init__(self, in_channels=3, num_classes=4, dropout=0.3, sparsity_lambda=1e-4):
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
            self.gcn.append(SGCNBlock(c_in, c_out, A_norm, stride=stride,
                                       dropout=dropout, sparsity_lambda=sparsity_lambda))

        self.fc = nn.Linear(self.CHANNEL_CONFIG[-1], num_classes)

    def forward(self, x):
        n, c, t, v = x.shape
        x_bn = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x_bn = self.data_bn(x_bn)
        x    = x_bn.reshape(n, v, c, t).permute(0, 2, 3, 1)
        total_sp = torch.tensor(0.0, device=x.device)
        for block in self.gcn:
            x, sp = block(x)
            total_sp = total_sp + sp
        x = x.mean(dim=[2, 3])
        return self.fc(x), total_sp


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    model = SGCN(in_channels=3, num_classes=4)
    print(f"S-GCN parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(4, 3, 100, 17)
    logits, sp = model(x)
    print(f"Input: {x.shape} -> Output: {logits.shape} | Sparsity loss: {sp.item():.6f}")
    assert logits.shape == (4, 4)
    print("OK -- S-GCN forward pass works.")
