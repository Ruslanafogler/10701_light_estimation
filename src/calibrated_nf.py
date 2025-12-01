import argparse
from pathlib import Path

import numpy as np

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_normals,
    save_normals_as_image,
)

def calibrated_near_light_photometric_stereo(
    images: np.ndarray,
    light_positions: np.ndarray,    # (n, 3), 3D positions of lights L_k
    light_intensities: np.ndarray,  # (n,), scalar intensities e_k
    X: np.ndarray,                  # (h, w, 3), per-pixel 3D positions X_p
    q: int = 3,                     # attenuation exponent (q = 2 or 3)
) -> tuple:
    """
    Calibrated near-light photometric stereo under Lambertian model:
        I_pk = rho_p * N_p^T (L_k - X_p) / ||L_k - X_p||^q * e_k

    Args:
        images:           (n, h, w, 3) RGB images
        light_positions:  (n, 3) 3D positions of each light L_k
        light_intensities:(n,)   scalar intensity e_k for each light
        X:                (h, w, 3) 3D position of each pixel X_p
        q:                attenuation exponent (2 or 3)

    Returns:
        normals:   (h, w, 3) unit normals
        albedos:   (h, w)    diffuse albedo
        valid_mask:(h, w)    boolean mask of valid pixels
    """
    # --- shape checks ---
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected images shape (n, h, w, 3), got {images.shape}")
    if light_positions.ndim != 2 or light_positions.shape[1] != 3:
        raise ValueError(
            f"Expected light_positions shape (n, 3), got {light_positions.shape}"
        )
    if light_intensities.ndim != 1 or light_intensities.shape[0] != light_positions.shape[0]:
        raise ValueError(
            f"Expected light_intensities shape (n,), got {light_intensities.shape}"
        )
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"Expected X shape (h, w, 3), got {X.shape}")

    n, h, w, _ = images.shape
    assert X.shape[0] == h and X.shape[1] == w, "X must match image resolution"

    # --- RGB to grayscale ---
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=images.dtype)
    I = np.tensordot(images, weights, axes=([3], [0]))  # (n, h, w)
    I = I.reshape(n, h * w).T                            # (h*w, n)

    # --- remove dark pixels ---
    valid_mask_flat = (I.max(axis=1) > 0.01)
    I_valid = I[valid_mask_flat, :]                      # (Nv, n)

    # --- flatten 3D positions ---
    X_flat = X.reshape(-1, 3)                            # (h*w, 3)
    X_valid = X_flat[valid_mask_flat, :]                 # (Nv, 3)

    Nv = I_valid.shape[0]
    M_valid = np.zeros((Nv, 3), dtype=np.float32)        # scaled normals B_p

    light_positions = light_positions.astype(np.float32)
    light_intensities = light_intensities.astype(np.float32)

    # --- per-pixel least squares ---
    for i in range(Nv):
        Xp = X_valid[i]              # (3,)
        Ip = I_valid[i]              # (n,)

        # V_k = L_k - X_p, shape (n, 3)
        V = light_positions - Xp[None, :]                # broadcast subtract

        # distance^q
        dist = np.linalg.norm(V, axis=1) + 1e-8          # (n,)
        dist_q = dist ** q                               # (n,)

        # bL_pk = (L_k - X_p) / ||L_k - X_p||^q * e_k
        # shape: (n, 3)
        bL_p = (V / dist_q[:, None]) * light_intensities[:, None]

        # Solve bL_p @ B_p ≈ I_p for B_p (3,)
        # lstsq: minimize ||bL_p B_p - I_p||_2
        B_p, _, _, _ = np.linalg.lstsq(bL_p, Ip, rcond=None)
        M_valid[i, :] = B_p

    # --- extract albedo and normals from scaled normals ---
    albedos_valid = np.linalg.norm(M_valid, axis=1)              # (Nv,)
    normals_valid = M_valid / (albedos_valid[:, None] + 1e-6)    # (Nv, 3)

    # --- reconstruct full images ---
    normals_full = np.zeros((h * w, 3), dtype=np.float32)
    albedos_full = np.zeros(h * w, dtype=np.float32)

    normals_full[valid_mask_flat, :] = normals_valid
    albedos_full[valid_mask_flat] = albedos_valid

    # default for invalid pixels
    normals_full[~valid_mask_flat, :] = np.array([0, 0, 1], dtype=np.float32)
    albedos_full[~valid_mask_flat] = 0.0

    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)
    valid_mask = valid_mask_flat.reshape(h, w)

    # normalize normals numerically
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, normals)

    # optional: clip albedo
    albedos = np.clip(albedos, 0, 1)

    return normals, albedos, valid_mask


def main():
    parser = argparse.ArgumentParser(
        description="Run calibrated photometric stereo baseline on a dataset"
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

    calibrated_dir = results_base_dir / "calibrated_nf"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    rgb_images = dataset.rgb()
    light_dirs_gt = dataset.light_directions()

    # Run calibrated photometric stereo
    print("\nRunning calibrated photometric stereo...")
    normals_cal, albedos_cal, valid_mask = calibrated_near_light_photometric_stereo(
        rgb_images, 
        light_positions=dataset.light_positions(),
        light_intensities=dataset.light_intensities(),
        X=dataset.xyz(),
        q=3
    )

    X=dataset.xyz()
    H, W = X.shape[:2]
    mask = np.isfinite(X[..., 0])

    # check a center pixel
    print("Center X:", X[H//2, W//2], "mask:", mask[H//2, W//2])

    print(f"\nSaving results to {results_base_dir}/")

    # Save calibrated results
    save_normals(normals_cal, str(calibrated_dir / "normals.npy"))
    save_normals_as_image(
        normals_cal, str(calibrated_dir / "normals.png"), mask=valid_mask
    )
    save_albedo(albedos_cal, str(calibrated_dir / "albedos.npy"))
    save_albedo_as_image(albedos_cal, str(calibrated_dir / "albedos.png"))

    print("Saved calibrated results")


if __name__ == "__main__":
    main()
