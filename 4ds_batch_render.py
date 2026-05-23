import bpy
import os
import gc
import math
import struct
import traceback
from mathutils import Vector

# The 4ds.py plugin natively supports both Mafia (v29) and HD2 (v41) formats.
# No conversion step is needed — files are passed directly to the importer.


def _check_version(filepath: str) -> int:
    """Read and return the 4DS version number. Raises ValueError for unsupported files."""
    with open(filepath, 'rb') as f:
        f.seek(4)
        version = struct.unpack('<H', f.read(2))[0]
    if version not in (29, 41):
        raise ValueError(f"Unsupported 4DS version {version} (expected 29 or 41)")
    return version


bl_info = {
    "name": "4DS Batch Renderer",
    "author": "Generated for LS3D workflow",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > 4DS Batch",
    "description": "Batch render .4ds files: front + back 512×512, then delete",
    "category": "Render",
}

# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

_BATCH_CAM_DATA  = "_BatchCamData"
_BATCH_CAM_OBJ   = "_BatchCamObj"
_BATCH_SUN_KEY   = "_BatchSunKey"
_BATCH_SUN_FILL  = "_BatchSunFill"
_BATCH_KEEP = {_BATCH_CAM_OBJ, _BATCH_SUN_KEY, _BATCH_SUN_FILL}


def _purge_scene_objects():
    """
    Remove every object and data-block that is NOT part of the batch rig.

    Thorough multi-pass cleanup to prevent memory leaks across thousands of files:
      1. Remove all non-batch scene objects (unlinks meshes, mats, armatures…)
      2. Free GPU texture memory BEFORE removing image data-blocks
      3. Three passes over all data types — needed because A→B→C references
         only become orphans after the parent is removed in a prior pass
      4. Final built-in orphan purge catches anything still missed
    """
    # ── 1. Remove scene objects ───────────────────────────────────────────────
    for obj in list(bpy.data.objects):
        if obj.name not in _BATCH_KEEP:
            bpy.data.objects.remove(obj, do_unlink=True)

    # ── 2. Free GPU memory for ALL images before removing data-blocks ─────────
    #       Without gl_free() Blender keeps the VRAM copy even after the
    #       Python data-block is gone, causing silent GPU memory growth.
    for img in list(bpy.data.images):
        if not img.use_fake_user:
            try:
                img.gl_free()
            except Exception:
                pass

    # ── 3. Multi-pass orphan removal (3 passes handles deep reference chains) ──
    for _pass in range(3):
        for mesh in list(bpy.data.meshes):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            if mat.users == 0:
                bpy.data.materials.remove(mat)
        for img in list(bpy.data.images):
            if img.users == 0 and not img.use_fake_user:
                bpy.data.images.remove(img)
        for arm in list(bpy.data.armatures):
            if arm.users == 0:
                bpy.data.armatures.remove(arm)
        for col in list(bpy.data.collections):
            if col.users == 0:
                bpy.data.collections.remove(col)
        for curve in list(bpy.data.curves):
            if curve.users == 0:
                bpy.data.curves.remove(curve)
        for ng in list(bpy.data.node_groups):
            if ng.users == 0:
                bpy.data.node_groups.remove(ng)
        for tex in list(bpy.data.textures):
            if tex.users == 0:
                bpy.data.textures.remove(tex)
        for act in list(bpy.data.actions):
            if act.users == 0:
                bpy.data.actions.remove(act)
        for lgt in list(bpy.data.lights):
            if lgt.users == 0 and lgt.name not in _BATCH_KEEP:
                bpy.data.lights.remove(lgt)
        for cam in list(bpy.data.cameras):
            if cam.users == 0 and cam.name != _BATCH_CAM_DATA:
                bpy.data.cameras.remove(cam)

    # ── 4. Blender's built-in recursive orphan purge (catches anything missed) ─
    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass

    # ── 5. Python garbage collection ──────────────────────────────────────────
    #       Blender Python objects may hold C-side references even after
    #       bpy.data removal; gc.collect() breaks those cycles immediately.
    gc.collect()


def _get_visible_mesh_objects():
    """
    Return mesh objects intended for rendering.

    Priority: objects that have at least one material assigned — these are the
    actual visual models (bottles, cars, props…).  Helper geometry like sectors,
    occluders, or collision meshes typically has no materials and is excluded.

    Falls back to ALL mesh objects only when no object with a material is found
    (e.g. untextured debug imports).
    """
    all_meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == 'MESH' and obj.name not in _BATCH_KEEP
    ]
    # Objects that actually have a material in at least one slot
    with_mats = [
        obj for obj in all_meshes
        if any(slot.material is not None for slot in obj.material_slots)
    ]
    return with_mats if with_mats else all_meshes


def _scene_bounds(objects):
    """Return (center Vector, max_dimension float) in world space."""
    coords = []
    for obj in objects:
        for corner in obj.bound_box:
            coords.append(obj.matrix_world @ Vector(corner))

    if not coords:
        return Vector((0.0, 0.0, 0.0)), 1.0

    xs = [v.x for v in coords]
    ys = [v.y for v in coords]
    zs = [v.z for v in coords]
    center = Vector((
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        (min(zs) + max(zs)) * 0.5,
    ))
    size = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        0.01,
    )
    return center, size


def _check_in_frame(objects, center, size, padding=1.15):
    """
    Verify that the computed ortho scale covers all object bounds.
    Returns (ok: bool, message: str).
    The padding factor must match what's used in _setup_camera.
    """
    required = size * padding
    # Bounds are derived from the same objects — they always fit by construction.
    # This check guards against degenerate (zero-size) geometry.
    if size < 1e-4:
        return False, "Object bounding box is near-zero — geometry may be empty"
    return True, f"All {len(objects)} mesh object(s) fit in frame (ortho scale {required:.4f})"


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _ensure_batch_camera():
    """Get or create the persistent batch camera data + object."""
    cam_data = bpy.data.cameras.get(_BATCH_CAM_DATA)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(_BATCH_CAM_DATA)

    cam_obj = bpy.data.objects.get(_BATCH_CAM_OBJ)
    if cam_obj is None:
        cam_obj = bpy.data.objects.new(_BATCH_CAM_OBJ, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)

    return cam_obj, cam_data


def _fit_ortho_scale(cam_obj, objects, padding: float = 1.15) -> float:
    """
    Project every bounding-box corner into camera space and return the
    ortho_scale (= viewport width/height for a 1:1 render) needed to
    fit all objects, including padding.

    Two-pass approach:
      Pass 1 — use only objects WITH materials (visual props).
      Pass 2 — fallback to all objects if pass 1 gives zero extent
               (shouldn't happen after _get_visible_mesh_objects filtering,
               but kept as a safety net).
    """
    bpy.context.view_layer.update()          # ensure matrix_world is current
    cam_inv = cam_obj.matrix_world.inverted()

    def _max_extent(obj_list):
        mx = my = 0.0
        for obj in obj_list:
            for corner in obj.bound_box:
                cam_co = cam_inv @ (obj.matrix_world @ Vector(corner))
                mx = max(mx, abs(cam_co.x))
                my = max(my, abs(cam_co.y))
        return mx, my

    # Pass 1: prefer objects with at least one material
    with_mats = [o for o in objects if any(s.material for s in o.material_slots)]
    if with_mats:
        max_x, max_y = _max_extent(with_mats)
    else:
        max_x, max_y = _max_extent(objects)

    return max(2.0 * max(max_x, max_y) * padding, 0.01)


def _setup_camera(direction: str, center: Vector, size: float,
                  objects: list,
                  elev_deg: float = 25.0, horiz_deg: float = 30.0,
                  padding: float = 1.15):
    """
    Place camera at a 3/4 angle, then compute ortho_scale by projecting
    the actual bounding box into camera space so nothing is clipped.

    direction : 'FRONT'  – base direction along -Y axis, offset toward +X
                'BACK'   – base direction along +Y axis, offset toward -X (symmetric)
    """
    cam_obj, cam_data = _ensure_batch_camera()

    dist = size * 10.0
    cam_data.type       = 'ORTHO'
    cam_data.clip_start = 0.001
    cam_data.clip_end   = dist * 3.0

    elev  = math.radians(elev_deg)
    horiz = math.radians(horiz_deg)
    cos_e = math.cos(elev)
    sin_e = math.sin(elev)

    if direction == 'FRONT':
        offset = Vector((
             dist * math.sin(horiz) * cos_e,
            -dist * math.cos(horiz) * cos_e,
             dist * sin_e,
        ))
    else:  # BACK
        offset = Vector((
            -dist * math.sin(horiz) * cos_e,
             dist * math.cos(horiz) * cos_e,
             dist * sin_e,
        ))

    cam_obj.location = center + offset

    look = (center - cam_obj.location).normalized()
    cam_obj.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()

    # Recompute ortho_scale from projected bbox so every object fits
    cam_data.ortho_scale = _fit_ortho_scale(cam_obj, objects, padding)

    bpy.context.scene.camera = cam_obj


def _remove_batch_camera():
    cam_obj = bpy.data.objects.get(_BATCH_CAM_OBJ)
    if cam_obj:
        bpy.data.objects.remove(cam_obj, do_unlink=True)
    cam_data = bpy.data.cameras.get(_BATCH_CAM_DATA)
    if cam_data:
        bpy.data.cameras.remove(cam_data)


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

_BATCH_WORLD = "_BatchWorld"


def _configure_eevee():
    """Set up EEVEE with a flat white ambient world so textures are visible."""
    scene = bpy.context.scene

    # Pick EEVEE engine (Blender 4.2+ uses EEVEE_NEXT)
    for engine in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = engine
            break
        except Exception:
            pass

    # Minimal EEVEE settings for speed
    eevee = getattr(scene, 'eevee', None)
    if eevee is not None:
        for attr, val in (
            ('taa_render_samples', 1),
            ('use_bloom',          False),
            ('use_ssr',            False),
            ('use_gtao',           False),
            ('use_shadows',        False),
        ):
            if hasattr(eevee, attr):
                setattr(eevee, attr, val)

    # Gray ambient world — matches background color, flat lighting
    # Linear value 0.467 ≈ sRGB #b2b2b2
    BG_GRAY = 0.467
    world = bpy.data.worlds.get(_BATCH_WORLD)
    if world is None:
        world = bpy.data.worlds.new(_BATCH_WORLD)
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new('ShaderNodeBackground')
    bg.inputs[0].default_value = (BG_GRAY, BG_GRAY, BG_GRAY, 1.0)
    bg.inputs[1].default_value = 0.25   # low ambient — lets sun shading show
    scene.world = world


def _cleanup_batch_world():
    world = bpy.data.worlds.get(_BATCH_WORLD)
    if world:
        bpy.data.worlds.remove(world)


def _add_scene_lights(elev_deg: float, horiz_deg: float):
    """
    Add two sun lamps that match the camera angle:
    - Key light  : strong, from the camera side
    - Fill light : weaker, from the opposite side to soften shadows
    """
    scene = bpy.context.scene
    elev  = math.radians(elev_deg)
    horiz = math.radians(horiz_deg)

    def _make_sun(name, energy, rx, ry, rz):
        light = bpy.data.lights.get(name)
        if light is None:
            light = bpy.data.lights.new(name, type='SUN')
        light.energy = energy
        light.angle  = math.radians(10)   # slightly soft sun
        obj = bpy.data.objects.get(name)
        if obj is None:
            obj = bpy.data.objects.new(name, light)
            scene.collection.objects.link(obj)
        obj.rotation_euler = (rx, ry, rz)
        return obj

    # Key light — from the FRONT camera direction (top-right-front)
    _make_sun(
        _BATCH_SUN_KEY, energy=4.0,
        rx=math.pi / 2 - elev,
        ry=0.0,
        rz=-(math.pi / 2 + horiz),
    )

    # Fill light — opposite side, roughly symmetric, softer
    _make_sun(
        _BATCH_SUN_FILL, energy=1.2,
        rx=math.pi / 2 - elev * 0.5,
        ry=0.0,
        rz=math.pi / 2 - horiz,
    )


def _remove_scene_lights():
    for name in (_BATCH_SUN_KEY, _BATCH_SUN_FILL):
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        light = bpy.data.lights.get(name)
        if light:
            bpy.data.lights.remove(light)


def _render_to(output_path: str):
    scene = bpy.context.scene
    scene.render.resolution_x               = 768
    scene.render.resolution_y               = 768
    scene.render.resolution_percentage      = 100
    scene.render.filepath                    = output_path
    scene.render.image_settings.file_format  = 'JPEG'
    scene.render.image_settings.color_mode   = 'RGB'
    scene.render.image_settings.quality      = 92
    scene.render.film_transparent            = False
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------

def _save_render_settings(scene):
    return {
        'engine':      scene.render.engine,
        'res_x':       scene.render.resolution_x,
        'res_y':       scene.render.resolution_y,
        'res_pct':     scene.render.resolution_percentage,
        'filepath':    scene.render.filepath,
        'transparent': scene.render.film_transparent,
        'camera_name': scene.camera.name if scene.camera else None,
    }


def _restore_render_settings(scene, saved):
    scene.render.engine                 = saved['engine']
    scene.render.resolution_x          = saved['res_x']
    scene.render.resolution_y          = saved['res_y']
    scene.render.resolution_percentage = saved['res_pct']
    scene.render.filepath               = saved['filepath']
    scene.render.film_transparent = saved['transparent']
    cam_name = saved.get('camera_name')
    scene.camera = bpy.data.objects.get(cam_name) if cam_name else None
    world_name = saved.get('world_name')
    scene.world = bpy.data.worlds.get(world_name) if world_name else None


def _collect_files(input_dir: str):
    """
    Build the render queue from all .4ds files in input_dir.

    Rules
    -----
    - Files ending in _ps2.4ds are skipped when a base file exists
      (they are legacy pre-converted copies; we auto-convert HD2 on the fly).
    - Standalone _ps2.4ds files (no base file) are processed directly.
    - All other .4ds files (v29 Mafia or v41 HD2) are queued for import;
      HD2 files are auto-converted in memory before import.

    Returns list of (source_path, output_name, [paths_to_delete])
    """
    files = {
        f.lower(): os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.4ds')
    }

    visited = set()
    queue   = []

    for fname_lower in sorted(files):
        if fname_lower in visited:
            continue

        full_path   = files[fname_lower]
        stem        = fname_lower[:-4]   # strip ".4ds"

        if stem.endswith('_ps2'):
            # Only process standalone PS2 file if no base file exists
            base_lower = stem[:-4] + '.4ds'
            if base_lower in files:
                visited.add(fname_lower)
                continue
            output_name = stem[:-4]   # strip "_ps2" for output name
            queue.append((full_path, output_name, [full_path]))
            visited.add(fname_lower)
        else:
            output_name = stem
            # If a companion _ps2.4ds exists, skip it (base HD2 file is preferred).
            # Include the PS2 file in files_to_delete so it gets cleaned up too.
            ps2_lower = stem + '_ps2.4ds'
            to_delete = [full_path]
            if ps2_lower in files:
                visited.add(ps2_lower)
                to_delete.append(files[ps2_lower])
            queue.append((full_path, output_name, to_delete))
            visited.add(fname_lower)

    return queue


def process_batch(input_dir: str, output_dir: str, delete_after: bool,
                  elev_deg: float, horiz_deg: float, operator):
    """
    Process every .4ds file in input_dir.
    Returns (processed_count, error_count).
    """
    scene  = bpy.context.scene
    saved  = _save_render_settings(scene)
    saved['world_name'] = scene.world.name if scene.world else None

    # ── Disable undo for the entire batch ─────────────────────────────────────
    # Blender's undo stack keeps a full copy of the scene for every operation.
    # Across thousands of imports this grows to tens of GB.
    # undo_steps = 0 disables undo entirely; we restore it after the batch.
    prefs = bpy.context.preferences.edit
    saved_undo_steps    = prefs.undo_steps
    saved_undo_memory   = prefs.undo_memory_limit
    prefs.undo_steps    = 0
    prefs.undo_memory_limit = 0   # 0 = no memory limit guard needed (undo off)

    _configure_eevee()
    _add_scene_lights(elev_deg, horiz_deg)

    os.makedirs(output_dir, exist_ok=True)

    queue = _collect_files(input_dir)

    if not queue:
        operator.report({'WARNING'}, "No .4ds files found in input folder")
        _restore_render_settings(scene, saved)
        return 0, 0

    processed = 0
    errors    = 0

    for import_path, output_name, files_to_delete in queue:
        filename = os.path.basename(import_path)
        print(f"\n[4DS Batch] ── Processing: {filename}  →  {output_name}")

        try:
            # ── 1. Clear scene ────────────────────────────────────────────
            _purge_scene_objects()

            # ── 2. Validate version, then import directly ─────────────────
            try:
                _check_version(import_path)
            except ValueError as e:
                operator.report({'WARNING'}, f"Skipping {filename}: {e}")
                errors += 1
                continue

            import_op = getattr(bpy.ops.import_scene, '4ds')
            result    = import_op(filepath=import_path)

            if 'CANCELLED' in result:
                operator.report({'WARNING'}, f"Import cancelled: {filename}")
                errors += 1
                continue

            bpy.context.view_layer.update()

            # ── 3. Check objects in frame ─────────────────────────────────
            mesh_objects = _get_visible_mesh_objects()
            if not mesh_objects:
                operator.report({'WARNING'}, f"No mesh objects after import: {filename}")
                errors += 1
                continue

            center, size = _scene_bounds(mesh_objects)
            ok, msg      = _check_in_frame(mesh_objects, center, size)

            if ok:
                print(f"[4DS Batch]   Frame check: {msg}")
            else:
                operator.report({'WARNING'}, f"{filename}: {msg}")
                errors += 1
                continue

            # ── 4. Render FRONT ───────────────────────────────────────────
            _setup_camera('FRONT', center, size, mesh_objects, elev_deg, horiz_deg)
            front_path = os.path.join(output_dir, f"{output_name}_front")
            _render_to(front_path)
            print(f"[4DS Batch]   Rendered front → {output_name}_front.jpg")

            # ── 5. Render BACK ────────────────────────────────────────────
            _setup_camera('BACK', center, size, mesh_objects, elev_deg, horiz_deg)
            back_path = os.path.join(output_dir, f"{output_name}_back")
            _render_to(back_path)
            print(f"[4DS Batch]   Rendered back  → {output_name}_back.jpg")

            processed += 1

            # ── 6. Delete files ───────────────────────────────────────────
            if delete_after:
                for path in files_to_delete:
                    try:
                        os.remove(path)
                        print(f"[4DS Batch]   Deleted: {path}")
                    except Exception as e:
                        print(f"[4DS Batch]   Could not delete {path}: {e}")

        except Exception as exc:
            operator.report({'WARNING'}, f"Error processing {filename}: {exc}")
            traceback.print_exc()
            errors += 1

    # ── Cleanup ────────────────────────────────────────────────────────────
    _purge_scene_objects()
    _remove_batch_camera()
    _remove_scene_lights()
    _cleanup_batch_world()
    _restore_render_settings(scene, saved)

    # Restore undo system
    prefs.undo_steps        = saved_undo_steps
    prefs.undo_memory_limit = saved_undo_memory

    return processed, errors


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class BATCHRENDER_Props(bpy.types.PropertyGroup):
    input_folder: bpy.props.StringProperty(
        name="Input Folder",
        description="Folder containing .4ds files to process",
        default="",
        subtype='DIR_PATH',
    )
    output_folder: bpy.props.StringProperty(
        name="Output Folder",
        description="Where rendered PNG files will be saved",
        default="",
        subtype='DIR_PATH',
    )
    delete_after: bpy.props.BoolProperty(
        name="Delete After Render",
        description="Delete the source .4ds file after both renders are saved",
        default=True,
    )
    elev_angle: bpy.props.FloatProperty(
        name="Elevation",
        description="Camera elevation above the horizontal plane (degrees)",
        default=25.0, min=0.0, max=89.0, step=100, precision=1,
    )
    horiz_angle: bpy.props.FloatProperty(
        name="Horizontal offset",
        description="Horizontal rotation from the pure front/back axis (degrees)",
        default=30.0, min=0.0, max=89.0, step=100, precision=1,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class BATCHRENDER_OT_Start(bpy.types.Operator):
    bl_idname  = "batchrender.start"
    bl_label   = "Start Batch Render"
    bl_description = (
        "Import each .4ds file, render 512×512 front and back views, "
        "save PNGs named after the file, then optionally delete the source"
    )

    def execute(self, context):
        props      = context.scene.batch_render_props
        input_dir  = bpy.path.abspath(props.input_folder).rstrip(os.sep)
        output_dir = bpy.path.abspath(props.output_folder).rstrip(os.sep)

        if not input_dir or not os.path.isdir(input_dir):
            self.report({'ERROR'}, "Input folder does not exist")
            return {'CANCELLED'}

        if not output_dir:
            self.report({'ERROR'}, "Output folder must be set")
            return {'CANCELLED'}

        # Check that the 4DS importer is registered
        if not hasattr(bpy.ops.import_scene, '4ds'):
            self.report({'ERROR'},
                "4DS importer not found — make sure the LS3D 4DS plugin is enabled")
            return {'CANCELLED'}

        ok, err = process_batch(
            input_dir, output_dir, props.delete_after,
            props.elev_angle, props.horiz_angle, self)

        if err == 0:
            self.report({'INFO'}, f"Batch render done: {ok} file(s) rendered")
        else:
            self.report({'WARNING'},
                f"Batch render done: {ok} ok, {err} error(s) — see console")

        return {'FINISHED'}


class BATCHRENDER_OT_OpenOutput(bpy.types.Operator):
    bl_idname  = "batchrender.open_output"
    bl_label   = "Open Output Folder"
    bl_description = "Open the output folder in the system file manager"

    def execute(self, context):
        import subprocess, sys
        output_dir = bpy.path.abspath(
            context.scene.batch_render_props.output_folder)
        if not os.path.isdir(output_dir):
            self.report({'ERROR'}, "Output folder does not exist yet")
            return {'CANCELLED'}
        if sys.platform == 'darwin':
            subprocess.Popen(['open', output_dir])
        elif sys.platform == 'win32':
            subprocess.Popen(['explorer', output_dir])
        else:
            subprocess.Popen(['xdg-open', output_dir])
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class BATCHRENDER_PT_Panel(bpy.types.Panel):
    bl_label      = "4DS Batch Renderer"
    bl_idname     = "BATCHRENDER_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category   = "4DS Batch"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.batch_render_props

        layout.label(text="Folders", icon='FILE_FOLDER')
        col = layout.column(align=True)
        col.prop(props, "input_folder",  text="Input")
        col.prop(props, "output_folder", text="Output")

        layout.separator()
        layout.prop(props, "delete_after")

        layout.separator()
        layout.label(text="Camera Angles", icon='CAMERA_DATA')
        col = layout.column(align=True)
        col.prop(props, "elev_angle",  text="Elevation °")
        col.prop(props, "horiz_angle", text="Horizontal °")

        layout.separator()

        # Show file count for feedback
        input_dir = bpy.path.abspath(props.input_folder)
        if os.path.isdir(input_dir):
            count = sum(1 for f in os.listdir(input_dir) if f.lower().endswith('.4ds'))
            layout.label(text=f"{count} .4ds file(s) found", icon='INFO')
        else:
            layout.label(text="Set input folder above", icon='ERROR')

        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("batchrender.start", icon='RENDER_ANIMATION')

        layout.operator("batchrender.open_output", icon='FILE_FOLDER', text="Open Output Folder")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    BATCHRENDER_Props,
    BATCHRENDER_OT_Start,
    BATCHRENDER_OT_OpenOutput,
    BATCHRENDER_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.batch_render_props = bpy.props.PointerProperty(
        type=BATCHRENDER_Props)


def unregister():
    del bpy.types.Scene.batch_render_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
