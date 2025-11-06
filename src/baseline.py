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
    if len(images.shape) == 4:
        h, w, _, n = images.shape
        # Average across channels if multi-channel
        images = images.mean(axis=2)
    else:
        h, w, n = images.shape

    assert n == num_lights, "Number of images must match num_lights"

    img_matrix = images.reshape(h * w, n)

    # Remove dark pixels
    valid_mask = (img_matrix.max(axis=1) > 0.01).astype(bool)
    img_valid = img_matrix[valid_mask, :]

    # SVD decomposition
    U, S, Vt = svd(img_valid, full_matrices=False)

    # rank-3 factorization for Lambertian surface
    rank = 3
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    # measurement + estimated light sources prenorm
    M = U_r @ np.diag(np.sqrt(S_r))
    L_est = (np.diag(np.sqrt(S_r)) @ Vt_r).T

    # Normalize using the integrability constraint
    # Compute initial normals from M
    normals_valid = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-6)

    # estimate albedos
    L_n = L_est @ normals_valid.T

    albedos_valid = np.zeros(img_valid.shape[0])
    for i in range(img_valid.shape[0]):
        # solve for albedo for this pixel
        denom = np.sum(L_n[:, i] ** 2) + 1e-6
        albedos_valid[i] = np.sum(img_valid[i, :] * L_n[:, i]) / denom
        albedos_valid[i] = max(0, albedos_valid[i])

    # normalise lights
    light_norms = np.linalg.norm(L_est, axis=1)
    L_normalized = L_est / (light_norms[:, np.newaxis] + 1e-6)

    # Rescale albedos
    mean_light_norm = light_norms.mean()
    albedos_valid *= mean_light_norm

    # remap
    normals_full = np.zeros((h * w, 3))
    albedos_full = np.zeros(h * w)
    normals_full[valid_mask, :] = normals_valid
    albedos_full[valid_mask] = albedos_valid

    # Set invalid pixels to default (0, 0, 1)
    normals_full[~valid_mask, :] = np.array([0, 0, 1])
    albedos_full[~valid_mask] = 0

    # Reshape to image format
    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)

    # Normalize normals
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norms + 1e-6)

    albedos = np.clip(albedos, 0, 1)
    light_directions = L_normalized

    return normals, albedos, light_directions


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    dataset_dir = project_dir / "dataset"
    output_dir = project_dir / "results" / "baseline"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...")
    images, metadata = load_dataset(str(dataset_dir))

    # run photometric
    num_lights = images.shape[2]
    normals, albedos, light_dirs = uncalibrated_photometric_stereo(
        images, num_lights=num_lights
    )

    print(f"\nSaving results to {output_dir}/")
    save_normals(normals, str(output_dir / "normals.npy"))
    save_normals_as_image(normals, str(output_dir / "normals.png"))
    save_albedo(albedos, str(output_dir / "albedos.npy"))
    save_albedo_as_image(albedos, str(output_dir / "albedos.png"))
    save_light_directions(light_dirs, str(output_dir / "light_directions.npy"))


if __name__ == "__main__":
    main()
