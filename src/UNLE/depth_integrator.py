"""
Depth integration from normals using Poisson reconstruction.
"""

import numpy as np
from numpy.linalg import norm

from UNLE.geometry_util import GeometryUtils
from Poisson_Depth_Recovery.main import PoissonOperator


class DepthIntegrator:
    def __init__(self, intrinsics, is_perspective=True):
        self.intrinsics = intrinsics
        self.is_perspective = is_perspective

    def depth_from_normals(self, B, valid_mask, depth_info=None, depth_weight=0.1):
        H, W = B.shape[:2]
        
        # Convert to unit normals
        normB = norm(B, axis=-1, keepdims=True) + 1e-8
        N = B / normB
        
        # Compute surface gradients from normals
        p, q, grad_mask = GeometryUtils.compute_surface_gradients_from_normals(
            N, self.intrinsics, self.is_perspective
        )
        
        # Combine masks
        final_mask = valid_mask & grad_mask
        
        # Prepare depth info for perspective case (log depth)
        depth_log = None
        if depth_info is not None:
            if self.is_perspective:
                # Perspective: convert to log depth
                depth_log = depth_info.copy()
                depth_log[depth_log <= 0] = 1  # Avoid log(0)
                depth_log = np.log(depth_log)
                depth_log[~final_mask] = 0
            else:
                # Orthographic: use depth directly
                depth_log = depth_info

        # Create PoissonOperator and solve
        mask_int = final_mask.astype(np.int8)
        poisson_op = PoissonOperator(
            np.dstack([p, q]), 
            mask_int, 
            depth_info=depth_log,
            depth_weight=depth_weight if depth_log is not None else 0.0
        )
        
        z = poisson_op.run()
        
        if self.is_perspective and depth_log is not None:
            # If we used log depth, exponentiate the result
            z = np.exp(z)
        
        # Clamp the depth update to avoid explosion
        default_depth = 1.0
        z = np.nan_to_num(z, nan=default_depth, posinf=default_depth, neginf=default_depth)
        # Cap depth range
        z = np.clip(z, 0.2, 5.0)
        
        # Zero out invalid regions
        z = np.where(final_mask, z, 0)
        
        # Center the depth
        if np.any(final_mask):
            z_mean = np.mean(z[final_mask])
            z = z - z_mean
        # --- Ensure returned depth matches full image resolution (H_full, W_full) ---
        # B and valid_mask are always full-res, so use their shape
        H_full, W_full = valid_mask.shape

        if z.shape[0] != H_full or z.shape[1] != W_full:
            import cv2
            z = cv2.resize(z.astype(np.float32), (W_full, H_full),
                        interpolation=cv2.INTER_CUBIC)
        return z

    def center_depth(self, z, mask):
        if np.any(mask):
            z_mean = np.mean(z[mask])
            z_centered = z - z_mean
        else:
            z_centered = z.copy()
        return z_centered