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

        self.bounds=((-3.0, 3.0), (-3.0, 3.0), (0.5, 3.0))
        self.step=0.5


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

    def solve(self, max_iters=8, depth_tol=1e-4, gd_steps=30, gd_lr=0.2):
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
        # X_p = d [x, y, 1]^T  (mean depth = d0, same for all pixels)
        Z0 = np.full((self.H, self.W), self.d0, np.float32)
        self.X = GeometryUtils.make_X_from_depth(Z0, self.intr)

        # B_p = (0, 0, -1)^T  (front-facing)
        self.B = np.zeros((self.H, self.W, 3), np.float32)
        self.B[..., 2] = -1

        # ----------------------------------------------------------------------
        # Initial coarse light search (as in the paper)
        # ----------------------------------------------------------------------
        print("Coarse initialization of lights ...")
        self.L = np.zeros((self.K, 3), np.float32)
        self.e = np.zeros((self.K,), np.float32)

        self.bounds=((-3.0, 3.0), (-3.0, 3.0), (0.5, 3.0))
        self.step=1.0

        for k in range(self.K):
            Lk, ek = self.light_est.coarse_search(
                self.I[k], self.X, self.B, self.mask,
                self.bounds, self.step
            )
            self.L[k] = Lk
            self.e[k] = ek

            print("Light", k, "position:", Lk)

        # Keep previous depth for convergence test
        prev_z = self.X[..., 2].copy()

        # ----------------------------------------------------------------------
        # Main alternating loop
        # ----------------------------------------------------------------------
        for it in range(max_iters):
            print(f"== Outer iteration {it+1}/{max_iters} ==")

            # ---------------------------------------------------------
            # 1) Update B
            # ---------------------------------------------------------
            print("  Updating B ...")
            self.B = self.light_est.update_B(
                self.I, self.X, self.L, self.e, self.mask
            )

            # ---------------------------------------------------------
            # 2) Integrate depth from normals
            # ---------------------------------------------------------
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


            # ---------------------------------------------------------
            # 3) Update X
            # ---------------------------------------------------------
            print("  Updating 3D points ...")
            self.X = GeometryUtils.make_X_from_depth(z, self.intr)

            # ---------------------------------------------------------
            # Convergence check on DEPTH only
            # ---------------------------------------------------------
            if np.any(self.mask):
                dz = np.mean(np.abs(z - prev_z)[self.mask])
            else:
                dz = 0.0
            print(f"  mean |Δz| = {dz:.6f}")

            if dz < depth_tol:
                print("  Depth converged — will stop outer iterations.")
                converged = True
                break

            prev_z = z.copy()

            # ---------------------------------------------------------
            # 4) Light refinement (inner loop per iteration)
            # ---------------------------------------------------------
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

        # -------------------------------------------------------------
        # FINAL STEP: If depth converged early → run light refinement
        # UNTIL light GD itself converges per-light.
        # -------------------------------------------------------------

        if converged:
            print("\n=== Final Light Refinement (post-convergence) ===")

            for k in range(self.K):
                print(f"  refining light {k} ...")
                Lk, ek = self.light_est.gradient_descent_refine(
                    self.I[k], self.X, self.B, self.mask,
                    self.L[k], self.e[k],
                    num_steps=200,      # longer, or even auto-stop inside refine()
                    lr=gd_lr * 0.5      # safer small step
                )

                print(f"light {k}: |ΔL| = {np.linalg.norm(Lk - self.L[k]):.4f}")

                if np.linalg.norm(Lk) <= 10.0 and np.isfinite(Lk).all():
                    self.L[k] = Lk
                    self.e[k] = ek

        # ---------------------------------------------------------
        # Final normal/albedo output
        # ---------------------------------------------------------
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

    #plot light positions and compare with ground truth
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import (required for 3D plot)
    import numpy as np
    from util import load_dataset

    # dataset = load_dataset(str(dataset_dir))
    light_positions_world = dataset.light_positions()    # shape (K, 3)

    camera_meta = dataset.metadata["camera"]
    R_cw = np.array(camera_meta["R_cw"])                 # world -> cam
    R_wc = R_cw.T                                        # cam -> world
    C = np.array(camera_meta["location_world"])          # camera center (world space)


    # -----------------------------
    # Convert estimated lights to world space
    # -----------------------------
    light_positions_est_cam = L_est                      # (K,3) from your algorithm

    # L_world = R_wc @ L_cam + C
    light_positions_est_world = (R_wc[:,:,0] @ light_positions_est_cam.T).T + C

    #Normalize with respect to object center
    object_center = np.mean(light_positions_world, axis=0)
    light_positions_est_world = light_positions_est_world - object_center
    light_positions_est_world = light_positions_est_world / np.linalg.norm(light_positions_est_world, axis=1, keepdims=True)
    # light_positions_est_world[:, 2] = -light_positions_est_world[:, 2]

    light_positions_world = light_positions_world - object_center
    light_positions_world = light_positions_world / np.linalg.norm(light_positions_world, axis=1, keepdims=True)

        print("Saved UNLPS results.")

    # -----------------------------------------------------------
    # Compute GT and estimated light positions in world coords
    # -----------------------------------------------------------
    print("\nConverting light estimates to world coordinates...")

    light_positions_world = dataset.light_positions()    # (K,3)

    camera_meta = dataset.metadata["camera"]
    R_cw = np.array(camera_meta["R_cw"])                 # world -> cam
    if R_cw.ndim == 3:
        R_cw = R_cw[0]
    R_wc = R_cw.T                                        # cam -> world
    C = np.array(camera_meta["location_world"])          # camera center (world)

    L_est_cam = L_est                                    # (K,3) estimated
    L_est_world = (R_wc @ L_est_cam.T).T + C             # convert to world

    np.save(out_dir / "lights_est_cam.npy", L_est_cam)
    np.save(out_dir / "lights_est_world.npy", L_est_world)

    # -----------------------------------------------------------
    # Normalize both to unit vectors (direction only)
    # -----------------------------------------------------------
    L_gt_norm = light_positions_world - light_positions_world.mean(axis=0)
    L_gt_norm = L_gt_norm / np.linalg.norm(L_gt_norm, axis=1, keepdims=True)

    L_est_norm = L_est_world - L_est_world.mean(axis=0)
    L_est_norm = L_est_norm / np.linalg.norm(L_est_norm, axis=1, keepdims=True)

    # -----------------------------------------------------------
    # ALIGN estimated lights to GT using Kabsch (rotation only)
    # -----------------------------------------------------------
    print("Aligning estimated lights to ground truth via Procrustes...")

    H = L_est_norm.T @ L_gt_norm
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    # Fix improper reflection
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # Apply rotation
    L_est_aligned = (R_opt @ L_est_norm.T).T

    np.save(out_dir / "lights_est_world_aligned.npy", L_est_aligned)

    # -----------------------------------------------------------
    # Plot GT vs Aligned Estimated Lights
    # -----------------------------------------------------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        L_gt_norm[:, 0], L_gt_norm[:, 1], L_gt_norm[:, 2],
        c='green', s=60, label='Ground Truth'
    )

    ax.scatter(
        L_est_aligned[:, 0], L_est_aligned[:, 1], L_est_aligned[:, 2],
        c='red', s=60, label='Estimated (Aligned)'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Light Directions: GT vs Estimated (Aligned)")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()

    print("\nSaved aligned and unaligned light positions.")

    # -----------------------------
    # Plot them together
    # -----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        light_positions_world[:, 0],
        light_positions_world[:, 1],
        light_positions_world[:, 2],
        color='green', s=50, label='Ground Truth (World)'
    )

    ax.scatter(
        light_positions_est_world[:, 0],
        light_positions_est_world[:, 1],
        light_positions_est_world[:, 2],
        color='red', s=50, label='Estimated (Converted to World)'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_box_aspect([1,1,1])  # equal aspect ratio

    plt.title("Light Positions: Ground Truth vs Estimated")
    plt.show()

