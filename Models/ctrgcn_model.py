"""
CTR-GCN — Channel-wise Topology Refinement Graph Convolutional Network.
Reference: Chen et al., ICCV 2021 (https://arxiv.org/abs/2107.12213)

Key innovations over AGCN:
  1. Channel-wise topology: instead of ONE shared A for all channels,
     CTR-GCN learns a DIFFERENT refined A for EACH output channel.
  2. Pairwise-difference refinement: tanh(theta_i - phi_j) instead of dot-product.
  3. Multi-Scale Temporal Convolution (MS-TCN) with 4 branches (k=3,5,dilated,maxpool).

Compatible with the 17-joint COCO skeleton used by ST-GCN / AGCN in this project.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from Models.stgcn_model import (
    NUM_JOINTS, COCO_EDGES, CENTER_JOINT,
    build_adjacency_matrix, build_spatial_partition, normalize_adjacency,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core CTR module
# ──────────────────────────────────────────────────────────────────────────────
class CTRGC(nn.Module):
    """
    Single partition CTR-GC layer.

    Topology per output-channel:
        A_local[c, i, j] = tanh(theta_c[i] - phi_c[j])   (N, rel_C, V, V)
        A_local → conv4 → (N, out_C, V, V)
        A_combined = alpha * A_local + A_base              (N, out_C, V, V)
        out = einsum('nctv, ncvw -> nctw', conv3(x), A_combined)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        rel_c = 8 if in_channels <= 16 else in_channels // 8

        self.conv1 = nn.Conv2d(in_channels, rel_c,      1)  # theta
        self.conv2 = nn.Conv2d(in_channels, rel_c,      1)  # phi
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)  # feature
        self.conv4 = nn.Conv2d(rel_c,       out_channels, 1)  # topology embed
        self.tanh  = nn.Tanh()

    def forward(self, x: torch.Tensor, A: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) | A: (V, V)
        theta = self.conv1(x).mean(dim=2)   # (N, rel_C, V)
        phi   = self.conv2(x).mean(dim=2)   # (N, rel_C, V)

        # pairwise difference → (N, rel_C, V, V)
        A_local = self.tanh(theta.unsqueeze(-1) - phi.unsqueeze(-2))
        A_local = self.conv4(A_local)        # (N, out_C, V, V)

        A_combined = alpha * A_local + A.unsqueeze(0).unsqueeze(0)  # broadcast
        return torch.einsum("nctv,ncvw->nctw", self.conv3(x), A_combined)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Temporal Convolution
# ──────────────────────────────────────────────────────────────────────────────
class MultiScaleTCN(nn.Module):
    """
    4-branch MS-TCN (kernel 3, kernel 5, dilated-3, max-pool).
    Each branch: out_channels // 4 → concat → out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        assert out_channels % 4 == 0
        bc = out_channels // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(bc),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, bc, (5, 1), stride=(stride, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(bc),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, bc, (3, 1), stride=(stride, 1), padding=(2, 0), dilation=(2, 1), bias=False),
            nn.BatchNorm2d(bc),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, bc, 1, bias=False),
            nn.BatchNorm2d(bc),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# CTR-GCN Block
# ──────────────────────────────────────────────────────────────────────────────
class CTRGCNBlock(nn.Module):
    """
    One CTR-GCN residual block:
        Σ_k CTRGC_k(x, A_k) → BN → ReLU → MS-TCN → BN → ReLU + residual
    """

    def __init__(self, in_channels: int, out_channels: int, A_norm: np.ndarray,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        K = A_norm.shape[0]
        self.K = K

        self.ctrgc = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(K)])
        for k in range(K):
            self.register_buffer(f"A{k}", torch.from_numpy(A_norm[k]).float())

        self.alpha   = nn.Parameter(torch.zeros(1))
        self.bn_gcn  = nn.BatchNorm2d(out_channels)
        self.mstcn   = MultiScaleTCN(out_channels, out_channels, stride=stride, dropout=dropout)
        self.bn_tcn  = nn.BatchNorm2d(out_channels)
        self.relu    = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        gcn_out = sum(
            self.ctrgc[k](x, getattr(self, f"A{k}"), self.alpha)
            for k in range(self.K)
        )
        gcn_out = self.relu(self.bn_gcn(gcn_out))
        tcn_out = self.bn_tcn(self.mstcn(gcn_out))
        return self.relu(tcn_out + res)


# ──────────────────────────────────────────────────────────────────────────────
# Full CTR-GCN Model
# ──────────────────────────────────────────────────────────────────────────────
class CTRGCN(nn.Module):
    """
    10-block CTR-GCN for 4-class skeleton action recognition (COCO 17 joints).
    Channel config mirrors ST-GCN / AGCN for fair comparison.
    """

    CHANNEL_CONFIG = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]

    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        adj    = build_adjacency_matrix()
        A      = build_spatial_partition(adj, CENTER_JOINT)
        A_norm = np.stack([normalize_adjacency(A[k]) for k in range(A.shape[0])], axis=0)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        channels = [in_channels] + self.CHANNEL_CONFIG
        self.blocks = nn.ModuleList()
        for i, c_out in enumerate(self.CHANNEL_CONFIG):
            c_in   = channels[i]
            stride = 2 if (c_in != c_out and c_in < c_out) else 1
            self.blocks.append(CTRGCNBlock(c_in, c_out, A_norm, stride=stride, dropout=dropout))

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(self.CHANNEL_CONFIG[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N, V, C, T).permute(0, 2, 3, 1)   # (N, C, T, V)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=[2, 3])   # global average pool → (N, C)
        return self.fc(self.drop(x))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CTRGCN(in_channels=3, num_classes=4, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CTR-GCN parameters: {n_params:,}")
    x = torch.randn(4, 3, 100, 17)
    out = model(x)
    print(f"Input: {x.shape}  →  Output: {out.shape}")
    assert out.shape == (4, 4)
    print("OK — CTR-GCN forward pass verified.")
