import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG


def get_position(frame, frequency):
    return -0.5 * np.cos(frequency * frame)


# --- create scene and attach a renderer and simulator
num_frames = 480
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = num_frames  # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
renderer = KubricBlender(scene)

# --- populate the scene with objects, lights, cameras
rng = np.random.RandomState(0)
wall_material = kb.FlatMaterial(
    color=kb.get_color("white"), indirect_visibility=True
)

cb1 = kb.Cube(
    name="cube-1",
    scale=(0.25, 0.25, 0.5),
    position=(0, -1, -0.5),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)
cb2 = kb.Cube(
    name="cube-2",
    scale=(0.25, 0.25, 0.5),
    position=(0, 0, -0.5),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)
cb3 = kb.Cube(
    name="cube-3",
    scale=(0.25, 0.25, 0.5),
    position=(0, 1, -0.5),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)
scene += kb.Cube(
    name="floor",
    scale=(3, 3, 0.001),
    position=(0, 0, -1),
    static=True,
    background=True,
    material=wall_material,
)
scene += cb1
scene += cb2
scene += cb3


scene += kb.Cube(
    scale=(0.1, 100, 100),
    position=(-4, 0, 0),
    material=wall_material,
    static=True,
    background=True,
)

scene += kb.DirectionalLight(
    name="sun", position=(4, 0, 3), look_at=(0, 0, 0), intensity=1.5
)

scene.camera = kb.PerspectiveCamera(
    name="camera", position=(4, 0, 0), look_at=(0, 0, 0)
)


xs = np.linspace(-np.pi, np.pi, num_frames)


for frame in range(1, num_frames + 1):
    cb1.position = np.array([0, -1, get_position(xs[frame - 1], 2.11)])
    cb1.keyframe_insert("position", frame)

    cb2.position = np.array([0, 0, get_position(xs[frame - 1], 3.17)])
    cb2.keyframe_insert("position", frame)

    cb3.position = np.array([0, 1, get_position(xs[frame - 1], 1.5)])
    cb3.keyframe_insert("position", frame)


# --- renders the output
kb.as_path("output").mkdir(exist_ok=True)
renderer.save_state("output/cubes.blend")
frames_dict = renderer.render()
kb.write_image_dict(frames_dict, "output")
