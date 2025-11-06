import argparse
import json
import math
import os

import bpy
import mathutils as mu


# ============================================================
# HELPERS
# ============================================================
def safe_last_object():
    obj = bpy.context.view_layer.objects.active
    if not obj:
        obj = bpy.data.objects[-1]
    return obj


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
        pts.append(mu.Vector((x * radius, y * radius, z * radius)))
        if len(pts) >= n:
            break
    return pts


# ============================================================
# MAIN DATASET GENERATION FUNCTION
# ============================================================
def generate_dataset(
    output_dir,
    num_lights=30,
    image_res=512,
    samples=64,
    use_gpu=True,
    light_radius=2.0,
    light_energy=30.0,
):
    """
    Generate a synthetic light estimation dataset using Blender.

    Args:
        output_dir: Directory to save rendered images and metadata
        num_lights: Number of lighting directions to sample
        image_res: Image resolution (square, in pixels)
        samples: Cycles samples per render
        use_gpu: Whether to enable GPU rendering
        light_radius: Distance from object center to light source
        light_energy: Light intensity in Watts
    """
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # SCENE INITIALIZATION
    # ============================================================
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = "GPU" if use_gpu else "CPU"
    scene.render.resolution_x = image_res
    scene.render.resolution_y = image_res
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.render.film_transparent = False

    # Disable default world light (black background)
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[1].default_value = 0.0

    # ============================================================
    # GEOMETRY
    # ============================================================
    # Sphere at origin
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    sphere = safe_last_object()

    # Diffuse Lambertian material
    mat = bpy.data.materials.new("Lambert")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    diff = nt.nodes.new(type="ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    out = nt.nodes["Material Output"]
    nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
    sphere.data.materials.append(mat)

    # ============================================================
    # CAMERA (placed along +Z, looking down −Z)
    # ============================================================
    bpy.ops.object.camera_add(location=(0.0, 0.0, 3.0))
    camera = safe_last_object()
    scene.camera = camera

    # ============================================================
    # LIGHT (single point light reused per render)
    # ============================================================
    bpy.ops.object.light_add(type="POINT", location=(0, 0, 2))
    light = safe_last_object()
    light.data.energy = light_energy

    # ============================================================
    # RENDER LOOP
    # ============================================================
    meta = {"camera": {}, "lights": []}
    positions = fibonacci_hemisphere(num_lights, radius=light_radius)
    R_wc = camera.matrix_world.to_3x3()
    R_cw = R_wc.transposed()
    t_wc = mu.Vector(camera.location)

    for i, pos in enumerate(positions):
        light.location = pos
        l_dir_world = (-pos).normalized()
        l_dir_cam = (R_cw @ l_dir_world).normalized()

        scene.render.image_settings.file_format = "TIFF"
        scene.render.image_settings.color_depth = "16"
        scene.render.filepath = os.path.join(output_dir, f"img_{i:03d}.tiff")

        print(f"Rendering image {i + 1}/{num_lights} ...")
        bpy.ops.render.render(write_still=True)

        meta["lights"].append(
            {
                "index": i,
                "position_world": [pos.x, pos.y, pos.z],
                "direction_to_object_world": [
                    l_dir_world.x,
                    l_dir_world.y,
                    l_dir_world.z,
                ],
                "direction_to_object_camera": [l_dir_cam.x, l_dir_cam.y, l_dir_cam.z],
                "energy_W": light.data.energy,
            }
        )

    # ============================================================
    # SAVE METADATA
    # ============================================================
    meta["camera"] = {
        "location_world": list(camera.location),
        "R_cw": [[R_cw[row][col] for col in range(3)] for row in range(3)],
        "t_cw": list(-(R_cw @ t_wc)),
    }

    with open(os.path.join(output_dir, "lights_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Done! Rendered {num_lights} images to {output_dir}")


# ============================================================
# CLI INTERFACE
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic light estimation dataset using Blender"
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
        help="Cycles samples per render (default: 64)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU rendering (use CPU instead)"
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
        default=30.0,
        help="Light intensity in Watts (default: 30.0)",
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_lights=args.num_lights,
        image_res=args.image_res,
        samples=args.samples,
        use_gpu=not args.no_gpu,
        light_radius=args.light_radius,
        light_energy=args.light_energy,
    )
