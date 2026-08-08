# make_scene.py — builds the light-footprint benchmark scene and renders the dataset.
# v2: uses Blender's data API only (no context-dependent ops), so it runs the same
#     from the Scripting tab or headless (blender -b -P make_scene.py).
#
# Output (under OUT_DIR):
#   with/rgb/0001.png ...               orbit rendered WITH the object
#   with/depth/0001.exr ...             per-frame depth (Z pass)
#   with/mask/0001.png ...              per-frame object silhouette (index pass)
#   without/rgb/0001.png ...            identical orbit WITHOUT the object = clean plate
#   transforms.json                     camera poses, NeRF-synthetic style
#
# Edit the CONFIG block, nothing else, on first use.

import bpy
import bmesh
import math
import json
import os

# ----------------------------- CONFIG ---------------------------------------
OUT_DIR   = os.path.join(os.path.expanduser("~"), "footprint_dataset")
N_FRAMES  = 81          # matches typical video-model clip length
RES_X     = 832         # ~480p, a size video models accept
RES_Y     = 480
SAMPLES   = 64          # Cycles samples; raise to 128+ for final data
RADIUS    = 5.5         # camera distance from the pivot
CAM_Z     = 1.5         # camera height; lower = more grazing view = stronger reflection
ARC_DEG   = 150         # orbit arc in degrees, centred on the front of the scene
QUICK_TEST = True       # True -> renders only 3 frames at low quality, to check setup
# -----------------------------------------------------------------------------

if QUICK_TEST:
    N_FRAMES, SAMPLES = 3, 8

SCENE_CENTRE = (0.0, 1.2, 0.0)   # the object sits here; camera & light aim near it


def clean_scene():
    """Remove every object in the current file (works in GUI and headless)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_material(name, base_color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def new_plane(name, size, location, rotation=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    return link(obj)


def new_sphere(name, radius, location):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    return link(obj)


def new_empty(name, location):
    obj = bpy.data.objects.new(name, None)
    obj.location = location
    return link(obj)


def aim_at(obj, target):
    """Track-To constraint with the right axes for cameras and lights (-Z forward)."""
    c = obj.constraints.new(type="TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return c


def build_scene():
    scn = bpy.context.scene

    # --- renderer -------------------------------------------------------------
    scn.render.engine = "CYCLES"
    scn.cycles.samples = SAMPLES
    scn.cycles.use_denoising = True
    scn.render.resolution_x = RES_X
    scn.render.resolution_y = RES_Y
    scn.render.resolution_percentage = 100
    scn.frame_start, scn.frame_end = 1, N_FRAMES
    try:  # use GPU if one is configured; fall back to CPU silently
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        for d in prefs.devices:
            d.use = True
        scn.cycles.device = "GPU"
    except Exception:
        scn.cycles.device = "CPU"

    # --- world (dim ambient so the shadow stays crisp) ------------------------
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    bg.inputs[1].default_value = 1.0
    scn.world = world

    # --- polished floor (REFLECTION via Fresnel + diffuse base shows SHADOW) --
    # NOTE: not metallic! A pure-metal mirror has no diffuse response, so
    # shadows are invisible on it. A polished dielectric shows both effects.
    floor = new_plane("Floor", 20, (0, 0, 0))
    floor.data.materials.append(
        make_material("GlossyFloor", (0.38, 0.38, 0.42), metallic=0.0, roughness=0.04))

    # --- white back wall, close to the object (COLOUR BLEED lands here) -------
    wall = new_plane("Wall", 14, (0, 2.4, 3.0), rotation=(math.radians(90), 0, 0))
    wall.data.materials.append(
        make_material("WhiteWall", (0.9, 0.9, 0.9), roughness=0.9))

    # --- the target object: a saturated red sphere ("the vase") ---------------
    target = new_sphere("TargetObject", 0.55, (SCENE_CENTRE[0], SCENE_CENTRE[1], 0.55))
    target.data.materials.append(
        make_material("RedObject", (0.85, 0.04, 0.04), roughness=0.35))
    target.pass_index = 1          # lets the renderer mark its pixels (mask pass)

    # --- look-at target for camera and light ----------------------------------
    lookat = new_empty("LookAt", (SCENE_CENTRE[0], SCENE_CENTRE[1], 0.6))

    # --- one area light from the side (this is what makes the SHADOW) ---------
    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 800
    light_data.size = 1.0          # smaller light -> harder, more visible shadow
    light = link(bpy.data.objects.new("KeyLight", light_data))
    light.location = (-3.0, -1.5, 3.5)
    aim_at(light, lookat)

    # --- camera on an orbiting pivot ------------------------------------------
    pivot = new_empty("OrbitPivot", SCENE_CENTRE)

    cam_data = bpy.data.cameras.new("OrbitCam")
    cam = link(bpy.data.objects.new("OrbitCam", cam_data))
    cam.parent = pivot
    cam.location = (0, -RADIUS, CAM_Z)   # local to the pivot -> true orbit radius
    aim_at(cam, lookat)
    scn.camera = cam

    # animate the pivot: arc from -ARC/2 to +ARC/2.
    # One exact keyframe per frame -> interpolation mode is irrelevant,
    # which keeps this working across Blender API versions.
    half = math.radians(ARC_DEG / 2.0)
    for f in range(1, N_FRAMES + 1):
        t = (f - 1) / max(N_FRAMES - 1, 1)
        pivot.rotation_euler = (0, 0, -half + t * 2 * half)
        pivot.keyframe_insert("rotation_euler", frame=f)

    return target, cam


def enable_passes():
    """Depth + object-index passes; written into the multilayer EXRs."""
    vl = bpy.context.scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True


def render_rgb(mode_dir):
    """Normal colour render -> PNG sequence."""
    scn = bpy.context.scene
    scn.cycles.samples = SAMPLES
    scn.cycles.use_denoising = True
    scn.render.filepath = os.path.join(mode_dir, "rgb", "")
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(animation=True)


def render_mask(mode_dir):
    """Object-silhouette masks without any special passes: hide everything
    except the target, render on a transparent background, and the alpha
    channel of each PNG is the mask. 1 sample is enough."""
    scn = bpy.context.scene
    floor = bpy.data.objects["Floor"]
    wall = bpy.data.objects["Wall"]

    floor.hide_render = True
    wall.hide_render = True
    scn.render.film_transparent = True
    scn.cycles.samples = 1
    scn.cycles.use_denoising = False
    scn.render.filepath = os.path.join(mode_dir, "mask_raw", "")
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(animation=True)

    floor.hide_render = False
    wall.hide_render = False
    scn.render.film_transparent = False


def export_transforms(cam, path):
    """NeRF-synthetic style transforms.json (poses are identical for both passes)."""
    scn = bpy.context.scene
    frames = []
    for f in range(1, N_FRAMES + 1):
        scn.frame_set(f)
        m = cam.matrix_world
        frames.append({
            "file_path": f"rgb/{f:04d}",
            "transform_matrix": [list(row) for row in m],
        })
    data = {"camera_angle_x": cam.data.angle_x, "frames": frames}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def main():
    clean_scene()
    target, cam = build_scene()

    os.makedirs(OUT_DIR, exist_ok=True)
    export_transforms(cam, os.path.join(OUT_DIR, "transforms.json"))

    enable_passes()
    for mode in ["with", "without"]:
        target.hide_render = (mode == "without")
        mode_dir = os.path.join(OUT_DIR, mode)
        os.makedirs(os.path.join(mode_dir, "rgb"), exist_ok=True)
        print(f"--- rendering '{mode}' RGB ({N_FRAMES} frames) ---")
        render_rgb(mode_dir)
        if mode == "with":  # object mask only needed for the with-pass
            os.makedirs(os.path.join(mode_dir, "mask_raw"), exist_ok=True)
            print(f"--- rendering '{mode}' mask pass (fast) ---")
            render_mask(mode_dir)

    print("Done. Dataset written to", OUT_DIR)


main()
