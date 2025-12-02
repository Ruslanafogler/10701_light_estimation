#Uncalibrated Near Field Photometric Stereo
import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq, norm
from scipy.ndimage import binary_erosion
from scipy.linalg import solve

from UNLE.light_estimation import LightEstimator
from UNLE.depth_integrator import DepthIntegrator
from UNLE.geometry_util import GeometryUtils

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_normals,
    save_normals_as_image,
)


class UNLPSEngine:
    """
    Implementation of Algorithm 1 from:
      Papadhimitri & Favaro,
      "Uncalibrated Near-Light Photometric Stereo"
    """

    def __init__(
        self,
        images_gray,
        intrinsics,
        valid_mask=None,
        exponent_q=2,
        mean_depth_init=1.0,
    ):
        """
        Args:
            images_gray: (K, H, W) grayscale images
            intrinsics: dict with 'fx', 'fy', 'cx', 'cy'
            valid_mask: optional (H, W) boolean mask
            exponent_q: near-light falloff exponent (2 or 3)
            mean_depth_init: initial mean depth d
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

    def _initialize_X_B(self):
        """Initialize depth as constant plane and normals as front-facing."""
        z0 = np.full((self.H, self.W), self.d0, np.float32)
        self.X = GeometryUtils.make_X_from_depth(z0, self.intr)

        self.B = np.zeros((self.H, self.W, 3), np.float32)
        self.B[..., 2] = -1.0  # facing camera

    def _initialize_lights(self):
        """
        Simple uncalibrated initialization for light positions and intensities:

        - Place lights on a hemisphere above the object center.
        - Intensities all start at 1.0
        """
        # Object center in camera space
        X_flat = self.X.reshape(-1, 3)
        mask_flat = self.mask.ravel()
        if np.any(mask_flat):
            center = np.mean(X_flat[mask_flat], axis=0)
        else:
            center = np.array([0.0, 0.0, self.d0], dtype=np.float32)

        # Put lights on a hemisphere above center with fixed radius
        radius = 1.5 * self.d0
        self.L = np.zeros((self.K, 3), np.float32)
        self.e = np.ones((self.K,), np.float32)

        golden = (1.0 + 5.0 ** 0.5) / 2.0
        n = self.K
        count = 0
        k = 0
        while count < n:
            z = 1 - (2 * k + 1) / (2 * n)
            k += 1
            if z <= 0:
                continue
            theta = 2 * np.pi * k / golden
            r_xy = (1 - z * z) ** 0.5
            x = r_xy * np.cos(theta)
            y = r_xy * np.sin(theta)
            self.L[count] = center + radius * np.array([x, y, z], dtype=np.float32)
            count += 1

    def solve(self, max_iters=8, depth_tol=1e-4, gd_steps=30, gd_lr=0.02):
        """
        Run alternating minimization.

        Args:
            max_iters: number of outer iterations
            depth_tol: stopping criterion on mean |Δz|
            gd_steps: steps of gradient descent per light per outer iteration
            gd_lr: learning rate for gradient descent

        Returns:
            N: (H, W, 3) normals
            A: (H, W) albedo
            z: (H, W) depth
            L: (K, 3) light positions
            e: (K,) light intensities
        """
        self._initialize_X_B()
        self._initialize_lights()

        prev_z = self.X[..., 2].copy()

        for it in range(max_iters):
            print(f"== Outer iteration {it+1}/{max_iters} ==")

            # 1) Update B (scaled normals)
            print("  Updating B ...")
            self.B = self.light_est.update_B(
                self.I, self.X, self.L, self.e, self.mask
            )

            # 2) Integrate depth from normals (Poisson)
            print("  Integrating depth from normals ...")
            z_raw = self.depth_integrator.depth_from_normals(self.B, self.mask)

            # Fix mean depth ambiguity: scale so that mean depth = d0
            z = z_raw.copy()
            z[~self.mask] = np.nan
            if np.any(self.mask):
                mean_z = np.nanmean(z)
                if mean_z > 0:
                    z *= (self.d0 / mean_z)
                else:
                    z[:] = self.d0

            z = np.nan_to_num(z, nan=self.d0, posinf=self.d0, neginf=self.d0)

            # 3) Update 3D points
            print("  Updating 3D points ...")
            self.X = GeometryUtils.make_X_from_depth(z, self.intr)

            # 4) Refine lights via gradient descent
            print("  Refining lights ...")
            for k in range(self.K):
                Lk, ek = self.light_est.gradient_descent_refine(
                    self.I[k], self.X, self.B, self.mask,
                    self.L[k], self.e[k],
                    num_steps=gd_steps,
                    lr=gd_lr
                )
                # Clamp to avoid crazy explosions
                if np.linalg.norm(Lk) <= 10.0 and np.isfinite(Lk).all():
                    self.L[k] = Lk
                    self.e[k] = ek

            # Convergence check
            if np.any(self.mask):
                dz = np.mean(np.abs(z - prev_z)[self.mask])
            else:
                dz = 0.0
            print(f"  mean |Δz| = {dz:.6f}")
            if dz < depth_tol:
                print("  Converged.")
                break

            prev_z = z.copy()

        N, A = GeometryUtils.normalize_normals(self.B)
        return N, A, z, self.L, self.e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncalibrated near-light photometric stereo (UNLPS)"
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
    out_dir = results_base_dir / "unle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    rgb_images = dataset.rgb()
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=rgb_images.dtype)
    images_gray = np.tensordot(rgb_images, weights, axes=([3], [0]))  # (K,H,W)

    # Intrinsics from metadata (from your Mitsuba dataset generator)
    if "camera" not in dataset.metadata:
        raise ValueError("Camera metadata not found in dataset")
    cam = dataset.metadata["camera"]
    intrinsics = {
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
    }

    # Optional valid mask
    valid_mask = None
    try:
        vm_path = dataset_dir / "valid_mask_000.npy"
        if vm_path.exists():
            valid_mask = np.load(vm_path).astype(bool)
            print(f"Loaded valid mask from {vm_path}")
    except Exception as e:
        print(f"Could not load valid mask: {e}, using default mask")

    engine = UNLPSEngine(
        images_gray,
        intrinsics,
        valid_mask=valid_mask,
        exponent_q=2,
        mean_depth_init=1.0,
    )

    N_unc, A_unc, z_unc, L_est, e_est = engine.solve(
        max_iters=8,
        depth_tol=1e-4,
        gd_steps=30,
        gd_lr=0.02,
    )

    print(f"\nSaving results to {out_dir}/")

    save_mask = engine.mask

    save_normals(N_unc, str(out_dir / "normals.npy"))
    save_normals_as_image(N_unc, str(out_dir / "normals.png"), mask=save_mask)
    save_albedo(A_unc, str(out_dir / "albedos.npy"))
    save_albedo_as_image(A_unc, str(out_dir / "albedos.png"))

    print("Saved UNLPS results.")