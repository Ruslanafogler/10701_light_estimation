#Uncalibrated Near Field Photometric Stereo
import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq, norm
from scipy.ndimage import binary_erosion
from scipy.linalg import solve

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_normals,
    save_normals_as_image,
)

from Poisson_Depth_Recovery.main import PoissonOperator

# =============================================================================
# Low-level math utilities
# =============================================================================

class GeometryUtils:
    @staticmethod
    def make_X_from_depth(z, intrinsics):
        """Z → (H,W,3) camera-space coordinates using intrinsics."""
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
        """Convert scaled normals B → (N, albedo)."""

        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        mag = np.linalg.norm(B, axis=-1, keepdims=True) + 1e-8
        N = B / mag
        A = mag[..., 0]  # albedo = magnitude
        return N, A


# =============================================================================
# Depth Integration (Poisson)
# =============================================================================

class DepthIntegrator:
    """Wrapper around PoissonOperator for depth integration from normals."""

    def __init__(self, intrinsics, is_perspective=True):
        """
        Args:
            intrinsics: dict with 'fx', 'fy', 'cx', 'cy'
            is_perspective: bool, True for perspective camera, False for orthographic
        """
        self.intrinsics = intrinsics
        self.is_perspective = is_perspective

    def depth_from_normals(self, B, valid_mask, depth_info=None, depth_weight=0.1):
        """
        B: (H,W,3) scaled normals → compute depth via Poisson.
        
        Args:
            B: (H, W, 3) scaled normals (B = albedo * N)
            valid_mask: (H, W) boolean mask of valid pixels
            depth_info: optional (H, W) depth prior (for log depth in perspective case)
            depth_weight: weight for depth prior
        
        Returns:
            z: (H, W) depth map
        """
        H, W = B.shape[:2]
        
        # Convert to unit normals
        normB = norm(B, axis=-1, keepdims=True) + 1e-8
        N = B / normB
        Nx, Ny, Nz = N[..., 0], N[..., 1], N[..., 2]

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        if self.is_perspective:
            # Perspective camera: compute p, q from normals
            # Following Poisson_Depth_Recovery/main.py convention
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            u = x - cx
            v = np.flipud(y) - cy  # Note: flipud for image coordinate convention

            # Perspective case (from main.py lines 224-225):
            # p = -Nx / (u*Nx + v*Ny + fx*Nz)
            # q = -Ny / (u*Nx + v*Ny + fy*Nz)
            
            bad = (~valid_mask) | np.isnan(Nx) | np.isnan(Ny) | np.isnan(Nz) | np.isinf(Nx) | np.isinf(Ny) | np.isinf(Nz)
            Nx = np.where(bad, 0, Nx)
            Ny = np.where(bad, 0, Ny)
            Nz = np.where(bad, 1, Nz)
            
            denom_p = u * Nx + v * Ny + fx * Nz
            denom_p_safe = np.where(np.abs(denom_p) < 1e-8, 
                                    np.sign(denom_p) * 1e-8, denom_p)
            denom_p_safe[bad] = 1e-8
            
            p = -Nx / denom_p_safe
            p[bad] = 0

            denom_q = u * Nx + v * Ny + fy * Nz
            denom_q_safe = np.where(np.abs(denom_q) < 1e-8,
                                    np.sign(denom_q) * 1e-8, denom_q)
            denom_q_safe[bad] = 1e-8
            q = -Ny / denom_q_safe
            q[bad] = 0
        else:
            # Orthographic camera
            Nz_safe = np.where(np.abs(Nz) < 1e-4, np.sign(Nz) * 1e-4, Nz)
            Zc = 1.0  # Default orthographic depth
            p = -Nx / Nz_safe / Zc
            q = -Ny / Nz_safe / Zc

        # Mask out invalid regions (already handled above for perspective case)
        if not self.is_perspective:
            p = np.where(valid_mask, p, 0)
            q = np.where(valid_mask, q, 0)

        # Prepare depth info for perspective case (log depth)
        # For perspective cameras, we use log depth as prior to avoid numerical issues
        depth_log = None
        if depth_info is not None:
            if self.is_perspective:
                # Perspective: convert to log depth (following main.py convention)
                depth_log = depth_info.copy()
                depth_log[depth_log <= 0] = 1  # Avoid log(0)
                depth_log = np.log(depth_log)
                depth_log[~valid_mask] = 0
            else:
                # Orthographic: use depth directly
                depth_log = depth_info

        # Create PoissonOperator and solve
        mask_int = valid_mask.astype(np.int8)
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
        
        default_depth = 1.0  # Default depth value
        z = np.nan_to_num(z, nan=default_depth, posinf=default_depth, neginf=default_depth)
        # Cap depth range
        z = np.clip(z, 0.2, 5.0)
        
        # Center the depth
        z = np.where(valid_mask, z, 0)
        z_mean = np.mean(z[valid_mask]) if np.any(valid_mask) else 0
        z = z - z_mean
        
        return z


# =============================================================================
# Light estimation
# =============================================================================

class LightEstimator:
    def __init__(self, q=2):
        self.q = q  # inverse distance exponent

    def _closed_form_energy(self, I_k, X, B, mask, Lk):
        """Compute ek and cost for one candidate Lk."""
        mask_flat = mask.ravel()
        Xf = X.reshape(-1,3)[mask_flat]
        Bf = B.reshape(-1,3)[mask_flat]
        If = I_k.ravel()[mask_flat]

        v = Lk[None,:] - Xf
        r = norm(v, axis=1) + 1e-8
        rpow = r ** self.q
        phi = np.sum(Bf * v, axis=1) / rpow

        denom = np.sum(phi * phi) + 1e-8
        e = np.sum(If * phi) / denom
        residual = If - e * phi
        F = np.sum(residual * residual)
        return e, F

    def coarse_search(self, I_k, X, B, mask, bounds, step):
        """Brute-force initial Lk with vectorized energy computation."""
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = bounds
        xs = np.arange(xmin, xmax+1e-6, step)
        ys = np.arange(ymin, ymax+1e-6, step)
        zs = np.arange(zmin, zmax+1e-6, step)

        # Pre-compute valid pixel data once
        mask_flat = mask.ravel()
        Xf = X.reshape(-1,3)[mask_flat]
        Bf = B.reshape(-1,3)[mask_flat]
        If = I_k.ravel()[mask_flat]
        N_valid = len(Xf)
        
        best_L = None
        best_e = 0
        best_F = np.inf

        # Process in chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 candidates at a time
        
        # Vectorize over z dimension (usually smallest)
        for z in zs:
            # Create grid for this z
            xx, yy = np.meshgrid(xs, ys)
            xy_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N_xy, 2)
            N_xy = len(xy_grid)
            
            # Process in chunks
            for chunk_start in range(0, N_xy, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N_xy)
                xy_chunk = xy_grid[chunk_start:chunk_end]  # (chunk_size, 2)
                chunk_size_actual = len(xy_chunk)
                
                Lk_candidates = np.zeros((chunk_size_actual, 3), dtype=np.float32)
                Lk_candidates[:, :2] = xy_chunk
                Lk_candidates[:, 2] = z
                
                # Vectorized energy computation for chunk
                # v: (chunk_size, N_valid, 3)
                v = Lk_candidates[:, None, :] - Xf[None, :, :]
                r = np.linalg.norm(v, axis=2) + 1e-8  # (chunk_size, N_valid)
                rpow = r ** self.q
                phi = np.sum(Bf[None, :, :] * v, axis=2) / rpow  # (chunk_size, N_valid)
                
                # Compute energy for each candidate in chunk
                denom = np.sum(phi * phi, axis=1) + 1e-8  # (chunk_size,)
                e_candidates = np.sum(If[None, :] * phi, axis=1) / denom  # (chunk_size,)
                residual = If[None, :] - e_candidates[:, None] * phi  # (chunk_size, N_valid)
                F_candidates = np.sum(residual * residual, axis=1)  # (chunk_size,)
                
                # Find best candidate in this chunk
                best_idx = np.argmin(F_candidates)
                if F_candidates[best_idx] < best_F:
                    best_F = F_candidates[best_idx]
                    best_L = Lk_candidates[best_idx].copy()
                    best_e = e_candidates[best_idx]
        
        return best_L, best_e

    def update_B(self, I, X, L, e, mask):
        """Solve B_p = argmin ||I_p - sum_k e_k bL_pk B_p|| per pixel.
        
        Vectorized version using batched operations.
        """
        K, H, W = I.shape
        B = np.zeros((H, W, 3), np.float32)
        
        # Get valid pixel indices
        mask_flat = mask.ravel()
        valid_indices = np.where(mask_flat)[0]
        
        if len(valid_indices) == 0:
            return B
        
        # Reshape for vectorized operations
        X_flat = X.reshape(-1, 3)  # (H*W, 3)
        I_flat = I.reshape(K, -1).T  # (H*W, K)
        X_valid = X_flat[valid_indices]  # (N_valid, 3)
        I_valid = I_flat[valid_indices]  # (N_valid, K)
        
        N_valid = len(valid_indices)
        
        # Vectorized computation: v = L - Xp for all pixels
        # L: (K, 3), X_valid: (N_valid, 3)
        # v: (N_valid, K, 3)
        v = L[None, :, :] - X_valid[:, None, :]  # Broadcasting: (N_valid, K, 3)
        
        # Compute distances: r = ||v|| for all (N_valid, K)
        r = np.linalg.norm(v, axis=2) + 1e-8  # (N_valid, K)
        rpow = r ** self.q  # (N_valid, K)
        
        # Compute b = (e * v) / r^q for all pixels
        # e: (K,), v: (N_valid, K, 3), rpow: (N_valid, K)
        b = (e[None, :, None] * v) / rpow[:, :, None]  # (N_valid, K, 3)
        
        # Solve B_p for all valid pixels at once using batched lstsq
        # For each pixel: b_p @ B_p = I_p
        # b: (N_valid, K, 3), I_valid: (N_valid, K)
        B_valid = np.zeros((N_valid, 3), np.float32)
        
        # Vectorized solve using einsum or batched operations
        # We can use np.linalg.lstsq in a loop, but vectorized with proper broadcasting
        for i in range(N_valid):
            b_p = b[i]  # (K, 3)
            I_p = I_valid[i]  # (K,)
            
            # Solve: b_p @ B_p = I_p
            # Using normal equations: (b_p^T @ b_p) @ B_p = b_p^T @ I_p
            bTb = b_p.T @ b_p  # (3, 3)
            bTI = b_p.T @ I_p  # (3,)
            
            # Solve linear system
            try:
                Bp = solve(bTb, bTI)
                if not np.isfinite(Bp).all():
                    Bp[:] = 0
            except:
                Bp = np.zeros(3, dtype=np.float32)
            
            B_valid[i] = Bp
        
        # Map back to full image
        B_flat = B.reshape(-1, 3)
        B_flat[valid_indices] = B_valid
        B = B_flat.reshape(H, W, 3)
        
        return B


# =============================================================================
# Main Engine: Algorithm 1
# =============================================================================

class UNLPSEngine:
    """
    Full implementation of:
    Papadhimitri & Favaro, "Uncalibrated Near-Light Photometric Stereo".
    """

    def __init__(self,
                 images_gray,
                 intrinsics,
                 valid_mask=None,
                 exponent_q=2,
                 mean_depth_init=1.0,
                 coarse_bounds=((-0.5,0.5), (-0.5,0.5), (0.5,2.0)),
                 coarse_step=0.05):

        self.I = images_gray.astype(np.float32)  # (K,H,W)
        self.K, self.H, self.W = self.I.shape

        self.intr = intrinsics
        self.q = exponent_q
        self.d0 = mean_depth_init
        self.bounds = coarse_bounds
        self.step = coarse_step

        if valid_mask is None:
            self.mask = (images_gray.max(axis=0) > 0.01)
        else:
            self.mask = valid_mask
        
        # Mask out depth & gradient around silhouette using binary erosion
        # This prevents divergence from "hairline fringes" where depth is nearly zero
        if np.any(self.mask):
            mask_eroded = binary_erosion(self.mask, iterations=2)
            self.mask = mask_eroded

        # helper modules
        self.light_est = LightEstimator(q=exponent_q)
        # Initialize depth integrator with camera intrinsics
        # Assume perspective camera by default (dataset uses perspective)
        self.depth_integrator = DepthIntegrator(intrinsics, is_perspective=True)


    def initialize(self):
        """Initialize X and B as in the paper."""
        fx, fy = self.intr["fx"], self.intr["fy"]
        cx, cy = self.intr["cx"], self.intr["cy"]

        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        Z = np.full((self.H, self.W), self.d0, np.float32)

        self.X = GeometryUtils.make_X_from_depth(Z, self.intr)

        self.B = np.zeros((self.H, self.W, 3), np.float32)
        self.B[...,2] = -1  # front-facing


    def _initial_light_estimation(self):
        self.L = np.zeros((self.K,3), np.float32)
        self.e = np.zeros((self.K,), np.float32)

        for k in range(self.K):
            Lk, ek = self.light_est.coarse_search(
                self.I[k], self.X, self.B,
                self.mask,
                bounds=self.bounds,
                step=self.step
            )
            self.L[k] = Lk
            self.e[k] = ek


    def solve(self, max_iters=4, depth_tol=1e-4):
        self.initialize()
        self._initial_light_estimation()

        prev_z = self.X[..., 2].copy()

        for it in range(max_iters):
            print(f"== Outer iter {it+1}/{max_iters} ==")

            # STEP 1: update B
            self.B = self.light_est.update_B(self.I, self.X,
                                             self.L, self.e, self.mask)

            # STEP 2: depth via Poisson
            # Use current depth as prior (for perspective case with log depth)
            z_prev = self.X[..., 2].copy()
            z_prev[z_prev <= 0] = self.d0  # Ensure positive depth
            # Do NOT use depth_info on iter 1 or 2 (depth_prev is very wrong early)
            z = self.depth_integrator.depth_from_normals(
                self.B, 
                self.mask,
                depth_info=z_prev if it >= 3 else None,  # Use previous depth as prior starting from iter 3
                depth_weight=0.1
            )

            # Clamp depth before updating X
            z = np.nan_to_num(z, nan=self.d0, posinf=self.d0, neginf=self.d0)
            z = np.clip(z, 0.2, 5.0)

            # STEP 3: update X from z
            self.X = GeometryUtils.make_X_from_depth(z, self.intr)

            # STEP 4: re-estimate lights
            # After first iteration, use smaller search space around current estimate
            search_radius = 0.2 if it > 0 else None  # Smaller radius for refinement
            refined_step = self.step * 0.5 if it > 0 else self.step  # Finer step for refinement
            
            for k in range(self.K):
                old_Lk = self.L[k].copy()
                old_ek = self.e[k]
                
                if it > 0 and search_radius is not None:
                    # Refine around current estimate for this specific light
                    Lk_curr = self.L[k]
                    refined_bounds = (
                        (max(self.bounds[0][0], Lk_curr[0] - search_radius), 
                         min(self.bounds[0][1], Lk_curr[0] + search_radius)),
                        (max(self.bounds[1][0], Lk_curr[1] - search_radius), 
                         min(self.bounds[1][1], Lk_curr[1] + search_radius)),
                        (max(self.bounds[2][0], Lk_curr[2] - search_radius), 
                         min(self.bounds[2][1], Lk_curr[2] + search_radius))
                    )
                    Lk, ek = self.light_est.coarse_search(
                        self.I[k], self.X, self.B,
                        self.mask,
                        refined_bounds,
                        refined_step
                    )
                else:
                    # Full search on first iteration
                    Lk, ek = self.light_est.coarse_search(
                        self.I[k], self.X, self.B,
                        self.mask,
                        self.bounds,
                        self.step
                    )
                
                # Clamp light positions to prevent lights from running off to infinity
                if np.linalg.norm(Lk) > 5.0 or not np.isfinite(Lk).all():
                    Lk = old_Lk
                    ek = old_ek
                self.L[k] = Lk
                self.e[k] = ek

            # Convergence check
            dz = np.mean(np.abs(z - prev_z)[self.mask]) if np.any(self.mask) else 0
            print(f"   mean |Δz| = {dz:.6f}")
            if dz < depth_tol:
                print("Converged.")
                break

            prev_z = z.copy()

        # Final outputs
        N, A = GeometryUtils.normalize_normals(self.B)
        return N, A, z, self.L, self.e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncalibrated photometric stereo baseline on a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sphere",
        help="Model name (subdirectory in dataset/, default: sphere)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    dataset_dir = project_dir / "dataset" / args.model
    results_base_dir = project_dir / "results" / args.model

    calibrated_dir = results_base_dir / "unle"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    rgb_images = dataset.rgb()
    # Convert RGB images to grayscale as in calibrated_nf.py
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=rgb_images.dtype)
    images_gray = np.tensordot(rgb_images, weights, axes=([3], [0]))

    #downsample images

    # Extract camera intrinsics from metadata
    if "camera" not in dataset.metadata:
        raise ValueError("Camera metadata not found in dataset")
    
    camera_meta = dataset.metadata["camera"]
    intrinsics = {
        "fx": camera_meta["fx"],
        "fy": camera_meta["fy"],
        "cx": camera_meta["cx"],
        "cy": camera_meta["cy"],
    }

    # Get valid mask from dataset if available
    valid_mask = None
    try:
        # Try to load valid mask from first image
        valid_mask_path = dataset_dir / "valid_mask_000.npy"
        if valid_mask_path.exists():
            valid_mask = np.load(valid_mask_path).astype(bool)
            print(f"Loaded valid mask from {valid_mask_path}")
    except Exception as e:
        print(f"Could not load valid mask: {e}, using default mask")

    engine = UNLPSEngine(
        images_gray, 
        intrinsics,
        valid_mask=valid_mask
    )
    normals_unc, albedos_unc, z, light_dirs_est, e = engine.solve()
    print(f"\nSaving results to {calibrated_dir}/")
    
    # Use engine's mask for saving
    save_mask = engine.mask
    
    # Save uncalibrated results
    save_normals(normals_unc, str(calibrated_dir / "normals.npy"))
    save_normals_as_image(
        normals_unc, str(calibrated_dir / "normals.png"), mask=save_mask
    )
    save_albedo(albedos_unc, str(calibrated_dir / "albedos.npy"))
    save_albedo_as_image(albedos_unc, str(calibrated_dir / "albedos.png"))
    print("Saved uncalibrated results")