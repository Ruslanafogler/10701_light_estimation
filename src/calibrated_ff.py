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


def calibrated_photometric_stereo(
    images: np.ndarray,
    light_directions: np.ndarray,
) -> tuple:
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected images shape (n, h, w, 3), got {images.shape}")
    if light_directions.ndim != 2 or light_directions.shape[1] != 3:
        raise ValueError(
            f"Expected light_directions shape (n, 3), got {light_directions.shape}"
        )

    n, h, w, _ = images.shape
    assert n == light_directions.shape[0], "Number of images must match lights"

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=images.dtype)
    I = np.tensordot(images, weights, axes=([3], [0]))
    I = I.reshape(n, h * w).T

    # remove dark pixels
    valid_mask_flat = (I.max(axis=1) > 0.01).astype(bool)
    I_valid = I[valid_mask_flat, :]

    L = light_directions

    M_valid = np.linalg.lstsq(L, I_valid.T, rcond=None)[0].T

    albedos_valid = np.linalg.norm(M_valid, axis=1)
    normals_valid = M_valid / (albedos_valid[:, np.newaxis] + 1e-6)

    # reconstruct
    normals_full = np.zeros((h * w, 3))
    albedos_full = np.zeros(h * w)
    normals_full[valid_mask_flat, :] = normals_valid
    albedos_full[valid_mask_flat] = albedos_valid
    normals_full[~valid_mask_flat, :] = np.array([0, 0, 1])
    albedos_full[~valid_mask_flat] = 0

    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)
    valid_mask = valid_mask_flat.reshape(h, w)

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, normals)
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

    calibrated_dir = results_base_dir / "calibrated_ff"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    rgb_images = dataset.rgb()
    light_dirs_gt = dataset.light_directions()

    # Run calibrated photometric stereo
    print("\nRunning calibrated photometric stereo...")
    normals_cal, albedos_cal, valid_mask = calibrated_photometric_stereo(
        rgb_images, light_dirs_gt
    )

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
