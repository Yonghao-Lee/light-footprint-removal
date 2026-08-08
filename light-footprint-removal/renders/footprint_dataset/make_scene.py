# make_scene.py — Blender/Cycles scene + renders for the light-footprint benchmark.
# Data API only (no context-dependent ops), so it behaves the same in the
# Scripting tab and headless: blender -b -P make_scene.py
#
# Writes under OUT_DIR:
#   with/rgb/*.png       orbit with the object
#   with/mask_raw/*.png  object silhouette in the alpha channel (see extract_aux.py)
#   without/rgb/*.png    same orbit, object hidden -> exact clean plates
#   transforms.json      camera poses, NeRF-synthetic format

import bpy
import bmesh
import math
import json
import os

# ----------------------------- CONFIG ---------------------------------------
OUT_DIR   = os.path.join(os.path.expanduser("~"), "footprint_dataset")
N_FRAMES  = 81          # video models want 16n+1 frames
RES_X     = 832         # native VACE/ROSE resolution
RES_Y     = 480
SAMPLES   = 64          # raise to 128+ for final data
RADIUS    = 5.5
CAM_Z     = 1.5         # lower camera = more grazing view = stronger floor reflection
ARC_DEG   = 120
QUICK_TEST = False      # 3 frames at low quality, to check the setup
# -----------------------------------------------------------------------------

if QUICK_TEST:
    N_FRAMES, SAMPLES = 3, 8

SCENE_CENTRE = (0.0, 1.2, 0.0)


def clean_scene():
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
    # cameras and lights both point down -Z
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

    # dim ambient so the shadow stays crisp
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.12, 0.12, 0.13, 1.0)
    bg.inputs[1].default_value = 1.0
    scn.world = world

    # floor must be a polished dielectric, not metal: a pure-specular floor has
    # no diffuse response, so a shadow would be invisible on it
    floor = new_plane("Floor", 20, (0, 0, 0))
    floor.data.materials.append(
        make_material("GlossyFloor", (0.38, 0.38, 0.42), metallic=0.0, roughness=0.04))

    # white wall close behind the object catches the colour bleed
    wall = new_plane("Wall", 24, (0, 2.4, 3.0), rotation=(math.radians(90), 0, 0))
    wall.data.materials.append(
        make_material("WhiteWall", (0.9, 0.9, 0.9), roughness=0.9))

    # mirror panel angled so sphere and mirror image are in frame together;
    # floor Fresnel alone is too weak at these angles to carry the reflection
    mirror = new_plane("Mirror", 2.8, (1.4, 2.0, 1.2),
                       rotation=(math.radians(90), 0, math.radians(-15)))
    mirror.data.materials.append(
        make_material("Mirror", (0.92, 0.92, 0.92), metallic=1.0, roughness=0.01))

    target = new_sphere("TargetObject", 0.55, (SCENE_CENTRE[0], SCENE_CENTRE[1], 0.55))
    target.data.materials.append(
        make_material("RedObject", (0.85, 0.04, 0.04), roughness=0.35))
    target.pass_index = 1

    lookat = new_empty("LookAt", (SCENE_CENTRE[0], SCENE_CENTRE[1], 0.6))

    # small key light -> hard shadow
    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 800
    light_data.size = 1.0
    light = link(bpy.data.objects.new("KeyLight", light_data))
    light.location = (-3.0, -1.5, 3.5)
    aim_at(light, lookat)

    # fill from the mirror's side: the mirror sees the sphere's unlit face,
    # which reads as a black blob without this
    fill_data = bpy.data.lights.new("FillLight", type="AREA")
    fill_data.energy = 250
    fill_data.size = 2.0
    fill = link(bpy.data.objects.new("FillLight", fill_data))
    fill.location = (3.5, -0.5, 2.5)
    aim_at(fill, lookat)

    pivot = new_empty("OrbitPivot", SCENE_CENTRE)

    cam_data = bpy.data.cameras.new("OrbitCam")
    cam = link(bpy.data.objects.new("OrbitCam", cam_data))
    cam.parent = pivot
    cam.location = (0, -RADIUS, CAM_Z)   # local to pivot -> true orbit radius
    aim_at(cam, lookat)
    scn.camera = cam

    # one keyframe per frame, so the interpolation mode (which changed across
    # Blender API versions) never matters
    half = math.radians(ARC_DEG / 2.0)
    for f in range(1, N_FRAMES + 1):
        t = (f - 1) / max(N_FRAMES - 1, 1)
        pivot.rotation_euler = (0, 0, -half + t * 2 * half)
        pivot.keyframe_insert("rotation_euler", frame=f)

    return target, cam


def enable_passes():
    vl = bpy.context.scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True


def render_rgb(mode_dir):
    scn = bpy.context.scene
    scn.cycles.samples = SAMPLES
    scn.cycles.use_denoising = True
    scn.render.filepath = os.path.join(mode_dir, "rgb", "")
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(animation=True)


def render_mask(mode_dir):
    # silhouette without compositor passes: hide everything but the target and
    # render on film_transparent -- the alpha channel is the mask
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
    # poses are shared by the with/without passes
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


SCRIPT_VERSION = "v13-full"


def main():
    print(f"=== make_scene {SCRIPT_VERSION} running ===")
    clean_scene()
    target, cam = build_scene()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "script_version.txt"), "w") as fh:
        fh.write(SCRIPT_VERSION + "\n")
    export_transforms(cam, os.path.join(OUT_DIR, "transforms.json"))

    enable_passes()
    for mode in ["with", "without"]:
        target.hide_render = (mode == "without")
        mode_dir = os.path.join(OUT_DIR, mode)
        os.makedirs(os.path.join(mode_dir, "rgb"), exist_ok=True)
        print(f"--- rendering '{mode}' RGB ({N_FRAMES} frames) ---")
        render_rgb(mode_dir)
        if mode == "with":
            os.makedirs(os.path.join(mode_dir, "mask_raw"), exist_ok=True)
            print(f"--- rendering '{mode}' mask pass ---")
            render_mask(mode_dir)

    print("Done. Dataset written to", OUT_DIR)


main()
