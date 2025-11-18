import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_light_directions,
    save_normals,
    save_normals_as_image,
)


def uncalibrated_photometric_stereo(
    images: np.ndarray,
    num_lights: int,
) -> tuple:
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected images shape (n, h, w, 3), got {images.shape}")

    n, h, w, _ = images.shape
    assert n == num_lights, "Number of images must match num_lights"

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=images.dtype)
    I = np.tensordot(images, weights, axes=([3], [0]))
    I = I.reshape(n, h * w).T
    valid_mask_flat = (I.max(axis=1) > 0.01).astype(bool)
    I_valid = I[valid_mask_flat, :]

    U, S, Vt = svd(I_valid, full_matrices=False)

    # rank-3 factorization for Lambertian surface
    rank = 3
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    # measurement + estimated light sources prenorm
    M = U_r @ np.diag(np.sqrt(S_r))
    L_est = (np.diag(np.sqrt(S_r)) @ Vt_r).T


    normals_valid = M / (
        np.linalg.norm(M, axis=1, keepdims=True) + 1e-6
    )

    # estimate albedos
    L_n = L_est @ normals_valid.T

    albedos_valid = np.zeros(I_valid.shape[0])
    for i in range(I_valid.shape[0]):
        denom = np.sum(L_n[:, i] ** 2) + 1e-6
        albedos_valid[i] = np.sum(I_valid[i, :] * L_n[:, i]) / denom
        albedos_valid[i] = max(0, albedos_valid[i])

    # normalise lights
    light_norms = np.linalg.norm(L_est, axis=1)
    L_normalized = L_est / (light_norms[:, np.newaxis] + 1e-6)

    # Rescale albedos
    mean_light_norm = light_norms.mean()
    albedos_valid *= mean_light_norm

    # Convert from image coordinates (y-down) to camera coordinates (y-up).
    image_to_camera = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=normals_valid.dtype,
    )
    normals_valid = normals_valid @ image_to_camera.T
    L_normalized = L_normalized @ image_to_camera.T

    # remap
    normals_full = np.zeros((h * w, 3))
    albedos_full = np.zeros(h * w)
    normals_full[valid_mask_flat, :] = normals_valid
    albedos_full[valid_mask_flat] = albedos_valid

    # Set invalid pixels to black (0, 0, 1)
    # this is y-up, btw, which is perp to camera ray
    normals_full[~valid_mask_flat, :] = np.array([0, 0, 1])
    albedos_full[~valid_mask_flat] = 0

    # Reshape to image format
    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)
    valid_mask = valid_mask_flat.reshape(h, w)

    # Normalize normals
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, normals)

    albedos = np.clip(albedos, 0, 1)

    return normals, albedos, L_normalized, valid_mask


def main():
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

    uncalibrated_dir = results_base_dir / "uncalibrated_ff"
    uncalibrated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    rgb_images = dataset.rgb()
    num_lights = len(dataset)

    # Run uncalibrated photometric stereo
    print("\nRunning uncalibrated photometric stereo...")
    normals_unc, albedos_unc, light_dirs_est, valid_mask = (
        uncalibrated_photometric_stereo(rgb_images, num_lights=num_lights)
    )

    print(f"\nSaving results to {results_base_dir}/")

    # Save uncalibrated results
    save_normals(normals_unc, str(uncalibrated_dir / "normals.npy"))
    save_normals_as_image(
        normals_unc, str(uncalibrated_dir / "normals.png"), mask=valid_mask
    )
    save_albedo(albedos_unc, str(uncalibrated_dir / "albedos.npy"))
    save_albedo_as_image(albedos_unc, str(uncalibrated_dir / "albedos.png"))
    save_light_directions(
        light_dirs_est, str(uncalibrated_dir / "light_directions_estimated.npy")
    )

    print("Saved uncalibrated results")


if __name__ == "__main__":
    main()
