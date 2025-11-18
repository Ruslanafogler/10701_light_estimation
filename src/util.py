import json
import os
from pathlib import Path

import mitsuba as mi
import numpy as np
from PIL import Image


class ImageDataset:
    def __init__(self, data: np.ndarray, metadata: dict):
        if data.ndim != 4 or data.shape[3] != 10:
            raise ValueError(
                f"Expected data shape (N, H, W, 10), got {data.shape}"
            )
        self.data = data
        self.metadata = metadata

    def rgb(self) -> np.ndarray:
        return self.data[:, :, :, 0:3]

    def depth(self) -> np.ndarray:
        return self.data[:, :, :, 3]

    def normals(self) -> np.ndarray:
        normals_world = self.data[:, :, :, 4:7]

        # transform the camera space
        if "camera" in self.metadata and "R_cw" in self.metadata["camera"]:
            R_cw = np.array(self.metadata["camera"]["R_cw"], dtype=np.float32)
            n, h, w, _ = normals_world.shape
            normals_flat = normals_world.reshape(-1, 3)
            normals_cam_flat = (R_cw @ normals_flat.T).T
            normals_cam = normals_cam_flat.reshape(n, h, w, 3)
            return normals_cam

        return normals_world


    def albedos(self) -> np.ndarray:
        return self.data[:, :, :, 7:10]

    def light_directions(self) -> np.ndarray:
        if "lights" not in self.metadata:
            raise ValueError("No light metadata available")

        lights = self.metadata["lights"]
        directions = np.array(
            [light["direction_to_object_camera"] for light in lights],
            dtype=np.float32,
        )

        if directions.ndim == 3:
            directions = directions.squeeze(axis=1)

        return -directions

    def light_power(self) -> np.ndarray:
        if "lights" not in self.metadata:
            raise ValueError("No light metadata available")

        lights = self.metadata["lights"]
        power = np.array([light["energy_W"] for light in lights], dtype=np.float32)

        return power

    def light_positions(self) -> np.ndarray:
        if "lights" not in self.metadata:
            raise ValueError("No light metadata available")
        if "camera" not in self.metadata:
            raise ValueError("No camera metadata available")

        lights = self.metadata["lights"]
        positions_world = np.array(
            [light["position_world"] for light in lights], dtype=np.float32
        )
        
        R_cw = np.array(self.metadata["camera"]["R_cw"], dtype=np.float32).squeeze(0)
        t_cw = np.array(self.metadata["camera"]["t_cw"], dtype=np.float32).squeeze(0)

        positions_camera = (R_cw @ positions_world.T).T + t_cw        

        return positions_camera

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.shape[0]


def load_dataset(dataset_dir: str) -> ImageDataset:
    """
    Load EXR images from the dataset directory with all AOV channels.

    The EXR files are expected to contain the following channels:
    - my_image.RGB (3 channels): rendered image
    - dd.Y (1 channel): depth
    - nn.RGB (3 channels): shading normals
    - al.RGB (3 channels): albedo

    Args:
        dataset_dir: Path to the dataset directory containing EXR images.

    Returns:
        ImageDataset containing all channels and metadata
    """
    dataset_path = Path(dataset_dir)

    exr_files = sorted(dataset_path.glob("img_*.exr"))

    if not exr_files:
        raise FileNotFoundError(f"No EXR images found in {dataset_dir}")

    if not mi.variant():
        available_backends = mi.variants()
        backends = ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"]
        for backend in backends:
            if backend in available_backends:
                mi.set_variant(backend)
                break

    first_bitmap = mi.Bitmap(str(exr_files[0]))
    h, w = first_bitmap.height(), first_bitmap.width()
    num_images = len(exr_files)
    all_data = np.zeros((num_images, h, w, 10), dtype=np.float32)

    for i, exr_file in enumerate(exr_files):
        bitmap = mi.Bitmap(str(exr_file))

        img_array = np.array(bitmap, dtype=np.float32)

        if img_array.shape[2] != 10:
            raise ValueError(
                f"Expected 10 channels in {exr_file}, got {img_array.shape[2]}"
            )

        all_data[i] = img_array

    metadata = {}
    meta_file = dataset_path / "lights_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)

    return ImageDataset(all_data, metadata)


def save_normals(normals: np.ndarray, output_path: str) -> None:
    """
    Save normal map as NPY file.

    Args:
        normals: Normal map of shape (height, width, 3).
        output_path: Path to save the normals array.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, normals)


def save_normals_as_image(
    normals: np.ndarray, output_path: str, mask: np.ndarray = None
) -> None:
    """
    Save normal map as PNG image (for visualization).
    Normals are visualized as RGB where each component is mapped to [0, 255].

    Args:
        normals: Normal map of shape (height, width, 3), values in [-1, 1].
        output_path: Path to save the image.
        mask: Optional binary mask of shape (height, width). Masked regions (where mask==0) will be set to black.
              If None, creates a mask based on normal magnitude < 0.5.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if mask is None:
        norms = np.linalg.norm(normals, axis=2)
        mask = (norms >= 0.5).astype(bool)

    # Convert [-1, 1] to [0, 1] then to [0, 255]
    vis_normals = (normals + 1.0) / 2.0
    vis_normals = np.clip(vis_normals, 0, 1)
    vis_normals = (vis_normals * 255).astype(np.uint8)

    if mask.ndim != 2 or mask.shape[:2] != vis_normals.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match normals shape {vis_normals.shape[:2]}"
        )
    vis_normals[mask == 0] = 0

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