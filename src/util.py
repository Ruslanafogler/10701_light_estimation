import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, dict]:
    """
    Load TIFF images from the dataset directory.

    Args:
        dataset_dir: Path to the dataset directory containing TIFF images.

    Returns:
        images: Shape (height, width, num_images) containing grayscale image data.
        metadata: Dictionary containing camera and light information if available.
    """
    dataset_path = Path(dataset_dir)

    # Get all TIFF files sorted by name
    tiff_files = sorted(dataset_path.glob("img_*.tiff"))

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF images found in {dataset_dir}")

    # Load images
    images_list = []
    for tiff_file in tiff_files:
        img = Image.open(tiff_file)
        img_array = np.array(img, dtype=np.float32)

        # Convert to grayscale if multi-channel
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                # Convert RGBA to grayscale using luminance formula
                img_array = (
                    0.299 * img_array[:, :, 0]
                    + 0.587 * img_array[:, :, 1]
                    + 0.114 * img_array[:, :, 2]
                )
            elif img_array.shape[2] == 3:  # RGB
                # Convert RGB to grayscale using luminance formula
                img_array = (
                    0.299 * img_array[:, :, 0]
                    + 0.587 * img_array[:, :, 1]
                    + 0.114 * img_array[:, :, 2]
                )
            # else: assume already grayscale

        # Normalize to [0, 1] range
        max_val = img_array.max()
        if max_val > 1.0:
            img_array = img_array / max_val

        images_list.append(img_array)

    # Stack into (H, W, N) format
    images = np.stack(images_list, axis=-1)

    # Try to load metadata if available
    metadata = {}
    meta_file = dataset_path / "lights_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)

    return images, metadata


def save_normals(normals: np.ndarray, output_path: str) -> None:
    """
    Save normal map as NPY file.

    Args:
        normals: Normal map of shape (height, width, 3).
        output_path: Path to save the normals array.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, normals)


def save_normals_as_image(normals: np.ndarray, output_path: str) -> None:
    """
    Save normal map as PNG image (for visualization).
    Normals are visualized as RGB where each component is mapped to [0, 255].

    Args:
        normals: Normal map of shape (height, width, 3), values in [-1, 1].
        output_path: Path to save the image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert [-1, 1] to [0, 1] then to [0, 255]
    vis_normals = (normals + 1.0) / 2.0
    vis_normals = np.clip(vis_normals, 0, 1)
    vis_normals = (vis_normals * 255).astype(np.uint8)

    img = Image.fromarray(vis_normals)
    img.save(output_path)


def save_albedo(albedo: np.ndarray, output_path: str) -> None:
    """
    Save albedo map as NPY file.

    Args:
        albedo: Albedo map of shape (height, width).
        output_path: Path to save the albedo array.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, albedo)


def save_albedo_as_image(albedo: np.ndarray, output_path: str) -> None:
    """
    Save albedo map as PNG image (for visualization).

    Args:
        albedo: Albedo map of shape (height, width), values in [0, 1].
        output_path: Path to save the image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert [0, 1] to [0, 255]
    vis_albedo = np.clip(albedo, 0, 1) * 255
    vis_albedo = vis_albedo.astype(np.uint8)

    img = Image.fromarray(vis_albedo, mode="L")
    img.save(output_path)


def save_light_directions(light_dirs: np.ndarray, output_path: str) -> None:
    """
    Save estimated light directions as NPY file.

    Args:
        light_dirs: Array of shape (num_lights, 3) containing light directions.
        output_path: Path to save the light directions.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, light_dirs)
