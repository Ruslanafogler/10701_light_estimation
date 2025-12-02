"""
Geometry utilities for 3D coordinate transformations and normal handling.
"""

import numpy as np


class GeometryUtils:
    """Static utility methods for geometry operations."""
    
    @staticmethod
    def make_X_from_depth(z, intrinsics):
        """
        Convert depth map to 3D camera-space coordinates.
        
        Args:
            z: (H, W) depth map
            intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            
        Returns:
            X: (H, W, 3) camera-space coordinates
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        H, W = z.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        X = np.zeros((H, W, 3), np.float32)
        X[..., 2] = z
        X[..., 0] = (u - cx) * z / fx
        X[..., 1] = (v - cy) * z / fy
        return X

    @staticmethod
    def normalize_normals(B):
        """
        Convert scaled normals B to unit normals and albedo.
        
        Args:
            B: (H, W, 3) scaled normals (B = albedo * N)
            
        Returns:
            N: (H, W, 3) unit normals
            A: (H, W) albedo (magnitude of B)
        """
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        mag = np.linalg.norm(B, axis=-1, keepdims=True) + 1e-8
        N = B / mag
        A = mag[..., 0]  # albedo = magnitude
        return N, A

    @staticmethod
    def compute_surface_gradients_from_normals(N, intrinsics, is_perspective=True):
        """
        Compute surface gradients (p, q) from normals.
        
        Args:
            N: (H, W, 3) unit normals
            intrinsics: dict with camera intrinsics
            is_perspective: bool, True for perspective camera
            
        Returns:
            p, q: (H, W) surface gradients
            mask: (H, W) valid pixel mask
        """
        H, W = N.shape[:2]
        Nx, Ny, Nz = N[..., 0], N[..., 1], N[..., 2]
        
        # Create validity mask
        valid_mask = ~(np.isnan(Nx) | np.isnan(Ny) | np.isnan(Nz) |
                      np.isinf(Nx) | np.isinf(Ny) | np.isinf(Nz))
        valid_mask = valid_mask & (Nz != 0)
        
        if is_perspective:
            # Perspective camera case
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            u = x - cx
            v = np.flipud(y) - cy  # Note: flipud for image coordinate convention
            
            # Fix 1: Mask out NaN/INF in normals before computing denominators
            bad = (~valid_mask) | np.isnan(Nx) | np.isnan(Ny) | np.isnan(Nz) | np.isinf(Nx) | np.isinf(Ny) | np.isinf(Nz)
            Nx = np.where(bad, 0, Nx)
            Ny = np.where(bad, 0, Ny)
            Nz = np.where(bad, 1, Nz)
            
            # Compute p, q for perspective case
            denom_p = u * Nx + v * Ny + fx * Nz
            denom_q = u * Nx + v * Ny + fy * Nz
            
            # Handle near-zero denominators
            denom_p_safe = np.where(np.abs(denom_p) < 1e-8,
                                   np.sign(denom_p) * 1e-8, denom_p)
            denom_q_safe = np.where(np.abs(denom_q) < 1e-8,
                                   np.sign(denom_q) * 1e-8, denom_q)
            
            # Set bad pixels to safe denominator
            denom_p_safe[bad] = 1e-8
            denom_q_safe[bad] = 1e-8
            
            p = -Nx / denom_p_safe
            p[bad] = 0
            
            q = -Ny / denom_q_safe
            q[bad] = 0
            
        else:
            # Orthographic camera case
            Nz_safe = np.where(np.abs(Nz) < 1e-4, np.sign(Nz) * 1e-4, Nz)
            Zc = 1.0  # Default orthographic depth
            
            p = -Nx / Nz_safe / Zc
            q = -Ny / Nz_safe / Zc
        
        # Set invalid pixels to zero
        p = np.where(valid_mask, p, 0)
        q = np.where(valid_mask, q, 0)
        
        return p, q, valid_mask