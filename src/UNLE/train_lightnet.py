# train_lightnet.py
#!/usr/bin/env python3
"""
Train LightNet to predict light positions from (I, A, N),
with a physics-guided constraint based on the UNLPS near-light model:

    I_pk ≈ ρ_p * N_p^T (L_k - X_p) / ||L_k - X_p||^q * e_k

We:
  1) Run classical UNLPS once to get N_unc, A_unc, depth_unc (saved by unle.py).
  2) Build (I, A, N) -> L_gt_cam dataset.
  3) Train LightNet in two phases:
       - Phase 1: supervised MSE only.
       - Phase 2: supervised MSE + physics energy F(L, e*(L)) as regularizer.

Assumes all inputs are (128,128).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2  # for resizing

from UNLE.lightnet import LightNet
from UNLE.geometry_util import GeometryUtils
from util import load_dataset


# -------------------------------------------------------------------------
# Physics loss: F(L_k, e_k*) using current geometry
# -------------------------------------------------------------------------

def physical_energy_and_intensity_torch(
    I_k_t: torch.Tensor,
    X_t: torch.Tensor,
    B_t: torch.Tensor,
    mask_t: torch.Tensor,
    L_pred: torch.Tensor,
    q: int = 2,
    eps: float = 1e-6,
):
    """
    Compute physics-guided energy F(L_k, e_k*) + closed-form e_k (torch).

    I_k_t : (H, W)  torch float32 image (normalized)
    X_t   : (H, W, 3) torch float32 3D points
    B_t   : (H, W, 3) torch float32 pseudo-normals (albedo * normal)
    mask_t: (H, W) bool or 0/1 tensor
    L_pred: (B, 3) or (3,) predicted light position in camera coords
    q     : falloff exponent (2 or 3)

    Returns:
        F_mean: scalar tensor (mean residual^2)
        e_k   : scalar tensor (best LS intensity)
    """

    # We assume batch size 1 here; if (1,3) → take first
    if L_pred.ndim == 2:
        L = L_pred[0]  # (3,)
    else:
        L = L_pred     # (3,)
    L = L.view(1, 3)   # (1,3)

    # Ensure mask is boolean
    mask_flat = (mask_t.view(-1) > 0)

    I_flat = I_k_t.view(-1)[mask_flat]          # (N,)
    X_flat = X_t.view(-1, 3)[mask_flat]         # (N,3)
    B_flat = B_t.view(-1, 3)[mask_flat]         # (N,3)

    # v_p = L - X_p
    v = L - X_flat                              # (N,3)
    r = torch.norm(v, dim=1, keepdim=True).clamp_min(eps)  # (N,1)
    B_dot_v = (B_flat * v).sum(dim=1, keepdim=True)        # (N,1)

    # φ_p(L) = B_p^T v_p / ||v_p||^q
    phi = B_dot_v / (r ** q)                    # (N,1)

    # Closed-form e_k*(L)
    denom = (phi ** 2).sum() + eps
    num = (I_flat.view(-1, 1) * phi).sum()
    e_k = num / denom

    # Residual
    residual = I_flat.view(-1, 1) - e_k * phi   # (N,1)
    F_mean = (residual ** 2).mean()

    return F_mean, e_k
