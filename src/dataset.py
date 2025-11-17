import argparse
import json
import math
import os
from pathlib import Path

import mitsuba as mi
import numpy as np

from util import (
    load_dataset,
    save_albedo,
    save_albedo_as_image,
    save_light_directions,
    save_normals,
    save_normals_as_image,
)


def fibonacci_hemisphere(n, radius=1.0):
    """Generate n evenly spaced points on the UPPER hemisphere (z > 0)."""
    pts = []
    golden = (1 + 5**0.5) / 2
    for k in range(n * 2):  # oversample to ensure enough z>0
        z = 1 - (2 * k + 1) / (n * 2)
        if z <= 0:
            continue
        theta = 2 * math.pi * k / golden
        r = (1 - z * z) ** 0.5
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        pts.append([x * radius, y * radius, z * radius])
        if len(pts) >= n:
            break
    return pts


def get_mesh_centering_transform(obj_path, target_size=2.0):
    # find bbox
    min_coords = np.array([float("inf"), float("inf"), float("inf")])
    max_coords = np.array([float("-inf"), float("-inf"), float("-inf")])

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                min_coords = np.minimum(min_coords, vertex)
                max_coords = np.maximum(max_coords, vertex)

    # center of bbox
    center = (min_coords + max_coords) / 2.0

    # bbox dimensions
    bbox_size = max_coords - min_coords
    max_dimension = np.max(bbox_size)

    # scale factor to fit within target_size
    scale_factor = target_size / max_dimension if max_dimension > 0 else 1.0

    return mi.ScalarTransform4f.scale(
        mi.ScalarVector3f([scale_factor, scale_factor, scale_factor])
    ) @ mi.ScalarTransform4f.translate([-center[0], -center[1], -center[2]])


def generate_ground_truth(dataset_dir, model_name):
    dataset = load_dataset(dataset_dir)

    gt_normals = dataset.normals()[0]
    gt_albedos_rgb = dataset.albedos()[0]

    # luminance 
    gt_albedos = (
        0.2126 * gt_albedos_rgb[:, :, 0]
        + 0.7152 * gt_albedos_rgb[:, :, 1]
        + 0.0722 * gt_albedos_rgb[:, :, 2]
    )

    light_dirs_gt = dataset.light_directions()

    project_dir = Path(dataset_dir).parent.parent
    gt_dir = project_dir / "results" / model_name / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving ground truth to {gt_dir}/")

    # Save ground truth
    save_normals(gt_normals, str(gt_dir / "normals.npy"))
    save_normals_as_image(gt_normals, str(gt_dir / "normals.png"))
    save_albedo(gt_albedos, str(gt_dir / "albedos.npy"))
    save_albedo_as_image(gt_albedos, str(gt_dir / "albedos.png"))
    save_light_directions(light_dirs_gt, str(gt_dir / "light_directions.npy"))

    print("Ground truth saved successfully")


def generate_dataset(
    output_dir,
    num_lights=30,
    image_res=512,
    samples=64,
    light_radius=2.0,
    light_energy=30.0,
    mesh=None,
):
    """
    Generate a synthetic light estimation dataset using Mitsuba 3.

    Args:
        output_dir: Directory to save rendered images and metadata
        num_lights: Number of lighting directions to sample
        image_res: Image resolution (square, in pixels)
        samples: Number of samples per pixel
        light_radius: Distance from object center to light source
        light_energy: Light intensity in Watts
        mesh: Path to OBJ mesh file (default: None, uses sphere)
    """
    if mesh is not None:
        model_name = os.path.splitext(os.path.basename(mesh))[0]
    else:
        model_name = "sphere"

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    BACKENDS = ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"]

    available_backends = mi.variants()
    for i, backend in enumerate(BACKENDS):
        if backend in available_backends:
            mi.set_variant(backend)
            print(f"Using {backend} [{i}]")
            break

    # init scene
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "aov",
            "aovs": "dd.y:depth,nn:sh_normal,al:albedo",
            "my_image": {
                "type": "path",
            },
        },
        "sensor": {
            "type": "perspective",
            "fov": 50.0,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0.0, 0.0, 3.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0],
            ),
            "sampler": {
                "type": "independent",
                "sample_count": samples,
            },
            "film": {
                "type": "hdrfilm",
                "width": image_res,
                "height": image_res,
                # "rfilter": {"type": "gaussian"},
            },
        },
    }

    # add mesh
    if mesh is not None:
        centering_transform = get_mesh_centering_transform(mesh)
        scene_dict["geometry"] = {
            "type": "obj",
            "filename": mesh,
            "to_world": centering_transform,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": 0.8,
                },
            },
        }
    else:
        # Default to sphere
        scene_dict["geometry"] = {
            "type": "sphere",
            "radius": 1.0,
            "center": [0.0, 0.0, 0.0],
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": 0.8,
                },
            },
        }

    # Camera
    camera_to_world = mi.Transform4f.look_at(
        origin=[0.0, 0.0, 3.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
    )
    camera_pos = np.array([0.0, 0.0, 3.0])

    # Extract rotation matrix from transformation
    m = camera_to_world.matrix
    R_wc = np.array([m[i, :3] for i in range(3)])
    R_cw = R_wc.T

    meta = {"camera": {}, "lights": []}
    positions = fibonacci_hemisphere(num_lights, radius=light_radius)

    for i, pos in enumerate(positions):
        pos_array = np.array(pos)

        # Update light position in scene
        scene_dict["light"] = {
            "type": "point",
            "position": pos,
            "intensity": {
                "type": "uniform",
                "value": light_energy / (4 * math.pi),
            },
        }

        # Load and render scene
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene, spp=samples)

        # Save rendered image
        output_path = os.path.join(output_dir, f"img_{i:03d}.exr")
        mi.Bitmap(img).write(output_path)

        # Calculate light direction
        l_dir_world = -pos_array / np.linalg.norm(pos_array)
        l_dir_cam = (R_cw @ l_dir_world).tolist()

        meta["lights"].append(
            {
                "index": i,
                "position_world": pos,
                "direction_to_object_world": l_dir_world.tolist(),
                "direction_to_object_camera": l_dir_cam,
                "energy_W": light_energy,
            }
        )

        print(f"Rendering image {i + 1}/{num_lights} ...")

    meta["camera"] = {
        "location_world": camera_pos.tolist(),
        "R_cw": R_cw.tolist(),
        "t_cw": (-(R_cw @ camera_pos)).tolist(),
    }

    with open(os.path.join(output_dir, "lights_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Rendered {num_lights} images to {output_dir}")

    # save ground truth
    generate_ground_truth(output_dir, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic light estimation dataset using Mitsuba 3"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset"),
        help="Output directory for rendered images and metadata (default: ./dataset)",
    )
    parser.add_argument(
        "--num-lights",
        type=int,
        default=30,
        help="Number of lighting directions to sample (default: 30)",
    )
    parser.add_argument(
        "--image-res",
        type=int,
        default=512,
        help="Image resolution in pixels (square, default: 512)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Samples per pixel (default: 64)",
    )
    parser.add_argument(
        "--light-radius",
        type=float,
        default=2.0,
        help="Distance from object center to light source (default: 2.0)",
    )
    parser.add_argument(
        "--light-energy",
        type=float,
        default=30,
        help="Light intensity in Watts (default: 30.0)",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to OBJ mesh file (default: None, uses sphere)",
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_lights=args.num_lights,
        image_res=args.image_res,
        samples=args.samples,
        light_radius=args.light_radius,
        light_energy=args.light_energy,
        mesh=args.mesh,
    )
