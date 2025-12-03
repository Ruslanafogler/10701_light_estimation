# unle.py
# Uncalibrated Near-Light Photometric Stereo with physics-guided ML light estimation

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
import cv2  

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from UNLE.light_estimation import LightEstimator
from UNLE.depth_integrator import DepthIntegrator
from UNLE.geometry_util import GeometryUtils
from UNLE.lightnet import LightNet

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_normals,
    save_normals_as_image,
)

# -------------------------------------------------------------------------
# Helper: closed-form intensity given L_k (Eq. (8) in the paper)
# -------------------------------------------------------------------------

def estimate_intensity_from_L(I_k, X, B, mask, L_k, q):
    """
    Compute light intensity e_k via closed-form least squares (Eq. 8).

    I_k:  (H, W) image for light k
    X:    (H, W, 3) 3D points in camera coords
    B:    (H, W, 3) pseudonormals (albedo * normal)
    mask: (H, W) bool
    L_k:  (3,) light position
    q:    falloff exponent (2 or 3)
    """
    mask_flat = mask.ravel()
    Xf = X.reshape(-1, 3)[mask_flat]
    Bf = B.reshape(-1, 3)[mask_flat]
    If = I_k.ravel()[mask_flat].astype(np.float64)

    v = L_k[None, :] - Xf            # (N,3)
    r = np.linalg.norm(v, axis=1) + 1e-8
    phi = np.sum(Bf * v, axis=1) / (r ** q)  # (N,)

    denom = np.sum(phi * phi) + 1e-8
    e_k = np.sum(If * phi) / denom
    return float(np.clip(e_k, 0.0, 100.0))


# -------------------------------------------------------------------------
# Physics-guided energy F(L_k, e_k*) + closed-form e_k in torch
# -------------------------------------------------------------------------

def physical_energy_and_intensity_torch(
    I_k_t, X_t, B_t, mask_t, L_pred, q, eps=1e-6
):
    """
    Physics-guided energy F(L_k, e_k*) + closed-form e_k in torch.

    Implements Eq. (7) with closed-form Eq. (8), given a predicted L_k.

    Args:
        I_k_t : (H, W) torch float32        (normalized or raw image)
        X_t   : (H, W, 3) torch float32     (3D points in camera coords)
        B_t   : (H, W, 3) torch float32     (pseudonormals)
        mask_t: (H, W) torch bool/byte
        L_pred: (1,3) or (3,) torch float32 (predicted light position)
        q     : exponent (2 or 3)
    Returns:
        F_mean : scalar tensor, mean squared residual (physics loss)
        e_k    : scalar tensor, closed-form intensity
    """

    if I_k_t.ndim == 3:
        # in case it's (1,H,W)
        I_k_t = I_k_t[0]

    if L_pred.ndim == 2:
        L = L_pred[0]  # (3,)
    else:
        L = L_pred
    L = L.view(1, 3)  # (1,3)

    mask_flat = mask_t.view(-1) > 0
    I_flat = I_k_t.view(-1)[mask_flat]       # (N,)
    X_flat = X_t.view(-1, 3)[mask_flat]      # (N,3)
    B_flat = B_t.view(-1, 3)[mask_flat]      # (N,3)

    v = L - X_flat                           # (N,3)
    r = torch.norm(v, dim=1, keepdim=True).clamp_min(eps)  # (N,1)

    B_dot_v = (B_flat * v).sum(dim=1, keepdim=True)        # (N,1)
    phi = B_dot_v / (r ** q)                               # (N,1)

    denom = (phi ** 2).sum() + eps
    num = (I_flat.view(-1,1) * phi).sum()
    e_k = num / denom

    residual = I_flat.view(-1,1) - e_k * phi              # (N,1)
    F_mean = (residual ** 2).mean()                       # mean, not sum

    return F_mean, e_k


# -------------------------------------------------------------------------
# LightNet-based estimator wrapper (I, A, N → L)
# -------------------------------------------------------------------------

class MLLightEstimator:
    """
    Wraps a trained LightNet model to predict light positions from (I, A, N).
    """

    def __init__(self, model, A_unc, N_unc, device="cpu"):
        """
        A_unc: (H,W) numpy albedo from first UNLPS pass
        N_unc: (H,W,3) numpy normals from first UNLPS pass
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.A_t = torch.from_numpy(A_unc.astype(np.float32)).to(device)      # (H,W)
        self.N_t = torch.from_numpy(N_unc.astype(np.float32)).to(device)      # (H,W,3)

    def predict_L(self, I_k):
        """
        I_k: (H, W) numpy array (grayscale image under light k)
        Returns:
            L_k: (3,) numpy array in camera coordinates
        """
        img = I_k.astype(np.float32)
        if img.max() > 0:
            img = img / (img.max() + 1e-6)

        I_t = torch.from_numpy(img).to(self.device)               # (H,W)
        x = torch.stack(
            [
                I_t,
                self.A_t,
                self.N_t[..., 0],
                self.N_t[..., 1],
                self.N_t[..., 2],
            ],
            dim=0
        ).unsqueeze(0)                                            # (1,5,H,W)

        with torch.no_grad():
            pred = self.model(x)[0]  # (3,)

        return pred.cpu().numpy().astype(np.float32)


# -------------------------------------------------------------------------
# Core UNLPS Engine (Algorithm 1)
# -------------------------------------------------------------------------

class UNLPSEngine:
    """
    Implementation of Algorithm 1 from:
      Papadhimitri & Favaro,
      "Uncalibrated Near-Light Photometric Stereo"

    Modes:
      - Without ml_light_estimator: classic UNLPS (coarse search + GD refine).
      - With ml_light_estimator: use ML-predicted lights, only update e_k, B, z.
    """

    def __init__(
        self,
        images_gray,
        intrinsics,
        valid_mask=None,
        exponent_q=2,
        mean_depth_init=1.0,
        ml_light_estimator=None,
    ):
        """
        Args:
            images_gray: (K, H, W) grayscale images
            intrinsics: dict with 'fx', 'fy', 'cx', 'cy'
            valid_mask: optional (H, W) boolean mask
            exponent_q: near-light falloff exponent (2 or 3)
            mean_depth_init: initial mean depth d
            ml_light_estimator: optional MLLightEstimator instance
        """
        self.I = images_gray.astype(np.float32)
        self.K, self.H, self.W = self.I.shape

        self.intr = intrinsics
        self.q = exponent_q
        self.d0 = float(mean_depth_init)

        # Mask (where we have object)
        if valid_mask is None:
            self.mask = (self.I.max(axis=0) > 0.01)
        else:
            self.mask = valid_mask

        # Erode to avoid silhouette artifacts
        if np.any(self.mask):
            self.mask = binary_erosion(self.mask, iterations=2)

        self.light_est = LightEstimator(q=self.q)
        self.depth_integrator = DepthIntegrator(intrinsics, is_perspective=True)

        # to be filled
        self.X = None  # 3D points
        self.B = None  # scaled normals
        self.L = None  # light positions
        self.e = None  # intensities

        # coarse-search volume (for non-ML mode)
        self.bounds = ((-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0))
        self.step = 0.5

        self.ml_light_est = ml_light_estimator

    # ------------------------------------------------------------------
    def _initialize_X_B(self):
        """Initialize depth as constant plane and normals as front-facing."""
        z0 = np.full((self.H, self.W), self.d0, np.float32)
        self.X = GeometryUtils.make_X_from_depth(z0, self.intr)

        self.B = np.zeros((self.H, self.W, 3), np.float32)
        self.B[..., 2] = -1.0  # facing camera

    # ------------------------------------------------------------------
    def _classical_unlps(self, max_iters=8, depth_tol=1e-4, gd_steps=30, gd_lr=0.01):
        """
        Classic Algorithm 1: coarse search + gradient descent refinement for lights.
        """

        self._initialize_X_B()

        print("Coarse initialization of lights (classical UNLPS) ...")
        self.L = np.zeros((self.K, 3), np.float32)
        self.e = np.zeros((self.K,), np.float32)

        # Initial coarse search for each light
        for k in range(self.K):
            Lk, ek = self.light_est.coarse_search(
                self.I[k], self.X, self.B, self.mask,
                self.bounds, self.step
            )
            self.L[k] = Lk
            self.e[k] = ek
            print(f"  Light {k}: coarse L={Lk}, e={ek:.3f}")

        prev_z = self.X[..., 2].copy()

        for it in range(max_iters):
            print(f"\n== [Classical UNLPS] Outer iteration {it+1}/{max_iters} ==")

            # 1) Update B
            print("  Updating B ...")
            self.B = self.light_est.update_B(
                self.I, self.X, self.L, self.e, self.mask
            )

            # 2) Integrate depth from normals
            print("  Integrating depth from normals ...")
            z_raw = self.depth_integrator.depth_from_normals(self.B, self.mask)

            # Fix mean depth ambiguity
            z = z_raw.copy()
            z[~self.mask] = np.nan
            if np.any(self.mask):
                mean_z = np.nanmean(z)
                if mean_z > 0:
                    z *= (self.d0 / mean_z)
                else:
                    z[:] = self.d0
            z = np.nan_to_num(z, nan=self.d0, posinf=self.d0, neginf=self.d0)

            # 3) Update X
            print("  Updating 3D points ...")
            self.X = GeometryUtils.make_X_from_depth(z, self.intr)

            # 4) Re-optimize lights (gradient descent refine)
            print("  Refining lights ...")
            for k in range(self.K):
                Lk, ek = self.light_est.gradient_descent_refine(
                    self.I[k], self.X, self.B, self.mask,
                    self.L[k], self.e[k],
                    num_steps=gd_steps,
                    lr=gd_lr
                )
                if np.linalg.norm(Lk) <= 10.0 and np.isfinite(Lk).all():
                    self.L[k] = Lk
                    self.e[k] = ek

            # 5) Convergence check on depth
            if np.any(self.mask):
                dz = np.mean(np.abs(z - prev_z)[self.mask])
            else:
                dz = 0.0
            print(f"  mean |Δz| = {dz:.6f}")

            if dz < depth_tol:
                print("  Depth converged — stopping.")
                break

            prev_z = z.copy()

        return z

    # ------------------------------------------------------------------
    def _ml_unlps(self, max_iters=8, depth_tol=1e-4):
        """
        UNLPS variant using ML-predicted lights (L fixed, only e_k, B, z updated).
        """

        self._initialize_X_B()

        print("ML-based initialization of lights ...")
        self.L = np.zeros((self.K, 3), np.float32)
        self.e = np.zeros((self.K,), np.float32)

        # Use ML to get L_k, then LS for e_k
        for k in range(self.K):
            I_k = self.I[k]  # (H, W)
            Lk = self.ml_light_est.predict_L(I_k)
            ek = estimate_intensity_from_L(
                I_k, self.X, self.B, self.mask, Lk, self.q
            )
            self.L[k] = Lk
            self.e[k] = ek
            print(f"  [ML] Light {k}: L={Lk}, e={ek:.3f}")

        prev_z = self.X[..., 2].copy()

        for it in range(max_iters):
            print(f"\n== [ML UNLPS] Outer iteration {it+1}/{max_iters} ==")

            # 1) Update B
            print("  Updating B ...")
            self.B = self.light_est.update_B(
                self.I, self.X, self.L, self.e, self.mask
            )

            # 2) Integrate depth from normals
            print("  Integrating depth from normals ...")
            z_raw = self.depth_integrator.depth_from_normals(self.B, self.mask)

            # Fix mean depth ambiguity
            z = z_raw.copy()
            z[~self.mask] = np.nan
            if np.any(self.mask):
                mean_z = np.nanmean(z)
                if mean_z > 0:
                    z *= (self.d0 / mean_z)
                else:
                    z[:] = self.d0
            z = np.nan_to_num(z, nan=self.d0, posinf=self.d0, neginf=self.d0)

            # 3) Update X
            print("  Updating 3D points ...")
            self.X = GeometryUtils.make_X_from_depth(z, self.intr)

            # 4) Recompute intensities (L fixed)
            print("  Updating intensities e_k ...")
            for k in range(self.K):
                self.e[k] = estimate_intensity_from_L(
                    self.I[k], self.X, self.B, self.mask, self.L[k], self.q
                )

            # 5) Convergence check on depth
            if np.any(self.mask):
                dz = np.mean(np.abs(z - prev_z)[self.mask])
            else:
                dz = 0.0
            print(f"  mean |Δz| = {dz:.6f}")

            if dz < depth_tol:
                print("  Depth converged — stopping.")
                break

            prev_z = z.copy()

        return z

    # ------------------------------------------------------------------
    def solve(self, max_iters=8, depth_tol=1e-4, gd_steps=30, gd_lr=0.01):
        """
        Run UNLPS in classical mode (no ML) or ML mode if ml_light_estimator is given.

        Returns:
            N: (H, W, 3) normals
            A: (H, W) albedo
            z: (H, W) depth
            L: (K, 3) light positions
            e: (K,) intensities
        """

        if self.ml_light_est is None:
            z = self._classical_unlps(
                max_iters=max_iters,
                depth_tol=depth_tol,
                gd_steps=gd_steps,
                gd_lr=gd_lr,
            )
        else:
            z = self._ml_unlps(
                max_iters=max_iters,
                depth_tol=depth_tol,
            )

        # Final normal/albedo output from pseudonormals B
        N, A = GeometryUtils.extract_normal_and_albedo(self.B)
        return N, A, z, self.L, self.e


# -------------------------------------------------------------------------
# LightDataset for (I, A, N) → L_gt_cam
# -------------------------------------------------------------------------

class LightDatasetIAN(Dataset):
    """
    Dataset mapping (I, A, N) --> ground-truth light position in camera coords.

    images_gray: (K, H, W)
    albedo     : (H, W)
    normals    : (H, W, 3)
    lights_cam : (K, 3)
    """

    def __init__(self, images_gray, albedo, normals, lights_cam):
        self.images = images_gray.astype(np.float32)
        self.A = albedo.astype(np.float32)
        self.N = normals.astype(np.float32)
        self.lights = lights_cam.astype(np.float32)

        self.K, self.H, self.W = self.images.shape

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
        I_k = self.images[idx]  # (H,W)
        if I_k.max() > 0:
            I_norm = I_k / (I_k.max() + 1e-6)
        else:
            I_norm = I_k

        # Channels: [I, A, Nx, Ny, Nz]
        x = np.stack(
            [
                I_norm,
                self.A,
                self.N[..., 0],
                self.N[..., 1],
                self.N[..., 2],
            ],
            axis=0,
        )  # (5,H,W)

        L = self.lights[idx]  # (3,)

        return torch.from_numpy(x), torch.from_numpy(L)

# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightNet with physics-guided loss.")
    parser.add_argument("--model", type=str, default="sphere",
                        help="Dataset subdirectory name (e.g. 'sphere', 'bunny').")
    parser.add_argument("--epochs_phase1", type=int, default=20,
                        help="Supervised-only epochs.")
    parser.add_argument("--epochs_phase2", type=int, default=40,
                        help="Supervised + physics epochs.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for supervised training.")
    parser.add_argument("--q", type=int, default=2, choices=[2, 3],
                        help="Falloff exponent in near-light model.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths and dataset loading
    # ------------------------------------------------------------------
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    dataset_dir = project_dir / "dataset" / args.model
    results_dir = project_dir / "results" / args.model / "unle"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_dataset(str(dataset_dir))

    # RGB -> grayscale
    rgb_images = dataset.rgb()           # (K, H_full, W_full, 3)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=rgb_images.dtype)
    images_gray_full = np.tensordot(rgb_images, weights, axes=([3], [0]))  # (K,H_full,W_full)

    # Target resolution (128x128)
    RES = 128
    K, Hf, Wf = images_gray_full.shape
    images_gray = np.zeros((K, RES, RES), np.float32)
    for k in range(K):
        images_gray[k] = cv2.resize(
            images_gray_full[k].astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------
    # Load UNLPS results (from classical pass: run unle.py first)
    # ------------------------------------------------------------------
    normals_unc = np.load(results_dir / "normals_unc.npy")   # (H_unc,W_unc,3)
    albedos_unc = np.load(results_dir / "albedos_unc.npy")   # (H_unc,W_unc)
    depth_unc   = np.load(results_dir / "depth_unc.npy")     # (H_unc,W_unc)

    # Resize UNLPS outputs to 128x128 if needed
    if normals_unc.shape[0] != RES or normals_unc.shape[1] != RES:
        normals_res = np.zeros((RES, RES, 3), np.float32)
        for c in range(3):
            normals_res[..., c] = cv2.resize(
                normals_unc[..., c], (RES, RES), interpolation=cv2.INTER_CUBIC
            )
        normals_unc = normals_res

    if albedos_unc.shape != (RES, RES):
        albedos_unc = cv2.resize(
            albedos_unc.astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_AREA,
        )

    if depth_unc.shape != (RES, RES):
        depth_unc = cv2.resize(
            depth_unc.astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_CUBIC,
        )

    # Simple object mask: where albedo > 0
    mask = (albedos_unc > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Geometry for physics term: X_unc, B_unc
    # ------------------------------------------------------------------
    cam = dataset.metadata["camera"]
    intrinsics = {
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
    }

    # Build 3D points from depth (same intrinsics used in UNLPS)
    X_unc = GeometryUtils.make_X_from_depth(depth_unc, intrinsics)        # (H,W,3)
    # Pseudonormals B = ρ N
    B_unc = albedos_unc[..., None] * normals_unc                          # (H,W,3)

    # ------------------------------------------------------------------
    # Ground-truth light positions, converted to camera coordinates
    # ------------------------------------------------------------------
    light_positions_world = dataset.light_positions()   # (K,3)
    R_cw = np.array(cam["R_cw"])                       # world -> cam, shape (3,3) or (1,3,3)
    C = np.array(cam["location_world"])                # camera center (3,)

    if R_cw.ndim == 3:
        R_cw = R_cw[0]

    # L_cam = R_cw @ (L_world - C)
    L_gt_cam = (R_cw @ (light_positions_world - C).T).T   # (K,3)

    # ------------------------------------------------------------------
    # Build dataset and dataloaders
    # ------------------------------------------------------------------
    ds = LightDatasetIAN(images_gray, albedos_unc, normals_unc, L_gt_cam)
    loader_sup = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # For physics loss, we'll use batch size 1 (per-light physics)
    loader_phys = DataLoader(ds, batch_size=1, shuffle=True)

    # ------------------------------------------------------------------
    # Model, optimizer, losses
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LightNet(in_channels=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn_mse = nn.MSELoss()

    # Physics geometry as torch tensors (fixed during training)
    X_t = torch.from_numpy(X_unc.astype(np.float32)).to(device)      # (H,W,3)
    B_t = torch.from_numpy(B_unc.astype(np.float32)).to(device)      # (H,W,3)
    mask_t = torch.from_numpy(mask.astype(np.uint8)).to(device)      # (H,W)

    q = args.q

    # ------------------------------------------------------------------
    # Phase 1: Supervised-only training (helps get into right basin)
    # ------------------------------------------------------------------
    print("\n=== Phase 1: Supervised-only training ===")
    for epoch in range(args.epochs_phase1):
        model.train()
        total_loss = 0.0

        for x, y in loader_sup:
            # x: (B,5,H,W), y: (B,3)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            L_pred = model(x)           # (B,3)
            loss = loss_fn_mse(L_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(ds)
        print(f"[Phase1] Epoch {epoch+1}/{args.epochs_phase1} "
              f"- Supervised Loss: {avg_loss:.6f}")

    # ------------------------------------------------------------------
    # Phase 2: Supervised + physics-guided constraint
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Supervised + Physics-guided training ===")

    lambda_mse = 1.0
    lambda_phys = 0.5   # strength of physics constraint

    for epoch in range(args.epochs_phase2):
        model.train()
        total_loss = 0.0

        for (x, y) in loader_phys:
            x = x.to(device)            # (1,5,H,W)
            y = y.to(device)            # (1,3)

            # Normalized intensity for this light (channel 0)
            I_k_t = x[0, 0]            # (H,W)

            optimizer.zero_grad()
            L_pred = model(x)          # (1,3)

            # Physics term (using fixed geometry X_unc, B_unc)
            F_phys, e_k = physical_energy_and_intensity_torch(
                I_k_t, X_t, B_t, mask_t, L_pred, q
            )

            # Supervised term
            L_mse = loss_fn_mse(L_pred, y)

            # Total loss: physics as constraint, not as main target
            loss = lambda_mse * L_mse + lambda_phys * F_phys
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ds)
        print(f"[Phase2] Epoch {epoch+1}/{args.epochs_phase2} "
              f"- Total Loss: {avg_loss:.6f}")

    # ------------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------------
    out_path = results_dir / "lightnet_ian_physics.pth"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved LightNet weights to: {out_path}")


if __name__ == "__main__":
    main()
