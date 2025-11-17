import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

from util import load_dataset


def visualize_lights(model: str, method: str = "uncalibrated_ff"):
    """
    Visualize estimated and ground truth light positions using polyscope.

    Args:
        model: Model name (subdirectory in dataset/)
        method: Method name (uncalibrated_ff, calibrated_ff, or gt), default: uncalibrated_ff
    """
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    dataset_dir = project_dir / "dataset" / model
    results_dir = project_dir / "results" / model / method

    # Load dataset
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(str(dataset_dir))

    # Load ground truth light positions (in camera space)
    light_positions_cam = dataset.light_positions()
    print(light_positions_cam.shape)

    # Load estimated light directions based on method
    if method == "gt":
        light_dirs_file = results_dir / "light_directions.npy"
    else:
        light_dirs_file = results_dir / "light_directions_estimated.npy"

    if not light_dirs_file.exists():
        raise FileNotFoundError(
            f"Light directions file not found: {light_dirs_file}\n"
            f"Make sure results exist for model='{model}' and method='{method}'"
        )

    light_dirs_est_cam = np.load(light_dirs_file)

    R_cw = np.array(dataset.metadata["camera"]["R_cw"], dtype=np.float32)
    t_cw = np.array(dataset.metadata["camera"]["t_cw"], dtype=np.float32)

    if R_cw.ndim == 3:
        R_cw = R_cw.squeeze(axis=0)
    if t_cw.ndim == 2:
        t_cw = t_cw.squeeze(axis=0)
    R_wc = R_cw.T

    def wtc(p):
        return (R_cw @ p.T).T + t_cw

    def ctw(p):
        return (R_cw.T @ (p - t_cw).T).T

    # Convert light positions from camera space to world space
    light_positions_world = ctw(light_positions_cam)

    light_distance = 2.0
    surface_center_cam = t_cw
    light_positions_est_cam = surface_center_cam + light_dirs_est_cam * light_distance
    light_positions_est_world = ctw(light_positions_est_cam)

    # Initialize polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.0, 0.0, 0.0))

    # Add ground truth light positions
    ps_lights_gt = ps.register_point_cloud(
        "Ground Truth Lights",
        light_positions_world,
        enabled=True
    )
    ps_lights_gt.add_color_quantity(
        "color",
        np.tile([0.0, 1.0, 0.0], (len(light_positions_world), 1)),  # Green
        enabled=True
    )
    ps_lights_gt.set_radius(0.01)

    # Add estimated light positions
    ps_lights_est = ps.register_point_cloud(
        "Estimated Lights",
        light_positions_est_world,
        enabled=True
    )
    ps_lights_est.add_color_quantity(
        "color",
        np.tile([1.0, 0.0, 0.0], (len(light_positions_est_world), 1)),  # Red
        enabled=True
    )
    ps_lights_est.set_radius(0.01)

    # Add coordinate frame at origin (world space object center)
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    ps.register_point_cloud("Origin", origin).set_radius(0.05)

    # Load and display the mesh if available
    if model != "sphere":
        mesh_path = project_dir / "meshes" / f"{model}.obj"
        if mesh_path.exists():
            mesh = trimesh.load(mesh_path)
            # Apply the same centering transform as used in dataset generation
            bbox = mesh.bounds
            center = (bbox[0] + bbox[1]) / 2.0
            max_dim = np.max(bbox[1] - bbox[0])
            scale = 2.0 / max_dim if max_dim > 0 else 1.0

            # Transform vertices
            vertices = (mesh.vertices - center) * scale

            ps.register_surface_mesh(
                "Object Mesh",
                vertices,
                mesh.faces,
                enabled=True,
                color=(0.7, 0.7, 0.7)
            )
        else:
            print(f"Warning: Mesh file not found at {mesh_path}")
    else:
        # Create a sphere mesh for visualization
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        ps.register_surface_mesh(
            "Object Mesh",
            sphere.vertices,
            sphere.faces,
            enabled=True,
            color=(0.7, 0.7, 0.7)
        )

    # Add camera position in world space
    camera_pos_cam = np.array([0.0, 0.0, 0.0])
    camera_pos_world = ctw(camera_pos_cam)
    ps_camera = ps.register_point_cloud(
        "Camera",
        camera_pos_world.reshape(1, 3),
        enabled=True
    )
    ps_camera.add_color_quantity(
        "color",
        np.array([[0.0, 0.0, 1.0]]),  # Blue
        enabled=True
    )
    ps_camera.set_radius(0.05)

    # Add light direction vectors (from light to object)
    # Ground truth directions in world space
    light_dirs_cam_gt = dataset.light_directions()
    light_dirs_world_gt = (R_wc @ light_dirs_cam_gt.T).T

    # Create vector visualization: arrows from light position toward origin
    # Vectors point from light TO object (opposite of light direction)
    n_lights = len(light_positions_world)
    gt_nodes = np.vstack([
        light_positions_world,
        light_positions_world - light_dirs_world_gt * 0.5
    ])
    gt_edges = np.column_stack([
        np.arange(n_lights),
        np.arange(n_lights) + n_lights
    ])

    ps.register_curve_network(
        "GT Light Directions",
        gt_nodes,
        gt_edges,
        enabled=True,
        color=(0.0, 1.0, 0.0)
    )

    # Estimated directions in world space
    # Transform directions: directions transform like vectors, not points
    light_dirs_world_est = (R_wc @ light_dirs_est_cam.T).T
    est_nodes = np.vstack([
        light_positions_est_world,
        light_positions_est_world - light_dirs_world_est * 0.5
    ])
    est_edges = np.column_stack([
        np.arange(n_lights),
        np.arange(n_lights) + n_lights
    ])

    ps.register_curve_network(
        "Est Light Directions",
        est_nodes,
        est_edges,
        enabled=True,
        color=(1.0, 0.0, 0.0)
    )

    # Add text overlay with instructions
    def callback():
        psim.TextUnformatted("Light Visualization")
        psim.Separator()
        psim.TextUnformatted("Green: Ground truth light positions")
        psim.TextUnformatted("Red: Estimated light positions (from uncalibrated PS)")
        psim.TextUnformatted("Blue: Camera position")
        psim.Separator()
        psim.TextUnformatted(f"Model: {model}")
        psim.TextUnformatted(f"Number of lights: {len(light_positions_world)}")

    ps.set_user_callback(callback)

    # Show
    ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize estimated and ground truth light positions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sphere",
        help="Model name (subdirectory in dataset/, default: sphere)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="uncalibrated_ff",
        choices=["uncalibrated_ff", "calibrated_ff", "gt"],
        help="Method name (default: uncalibrated_ff)",
    )

    args = parser.parse_args()
    visualize_lights(args.model, args.method)
