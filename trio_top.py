import logging

import numpy as np

import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

world_matrix = {
    "bunny": np.array(
        (
            (-1.0, 3.2584136988589307e-07, 0.0, 0.7087775468826294),
            (-3.2584136988589307e-07, -1.0, 0.0, -1.2878063917160034),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    "suzanne": np.array(
        (
            (1.0, 0.0, 0.0, -0.8567398190498352),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    ),
    "teapot": np.array(
        (
            (1.0, 0.0, 0.0, -0.9078792333602905),
            (0.0, 1.0, 0.0, 1.2115877866744995),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    ),
}


points = {
    "bunny": np.array(
        (
            (
                0.044713765382766724,
                -1.0193415880203247,
                0.8044384121894836,
                1.0,
            ),
            (
                0.056191492825746536,
                -0.31232786178588867,
                0.8044384121894836,
                1.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ),
    ),
    "suzanne": np.array(
        (
            (-1.0, 0.0, 0.0, 1.0),
            (-0.2928931713104248, 2.9802322387695312e-08, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        )
    ),
    "teapot": np.array(
        (
            (
                0.044713765382766724,
                -1.0193415880203247,
                0.8044384121894836,
                1.0,
            ),
            (
                0.056191492825746536,
                -0.31232786178588867,
                0.8044384121894836,
                1.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ),
    ),
}


def interpolate_position(
    t: np.ndarray, handles: np.ndarray, world_matrix: np.ndarray
) -> np.ndarray:
    p0, p1, p2, p3 = handles[:, np.newaxis]
    t = t[..., np.newaxis]
    r = 1 - t
    out = r ** 3 * p0 + 3 * r ** 2 * t * p1 + 3 * r * t ** 2 * p2 + t ** 3 * p3
    out = out / out[..., [-1]]
    return (world_matrix @ out.T).T[..., :-1]


# --- create scene and attach a renderer and simulator
num_frames = 480
scene = kb.Scene(resolution=(256, 256), background=kb.get_color("white"))
scene.frame_end = num_frames  # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
renderer = KubricBlender(scene)

# --- populate the scene with objects, lights, cameras
rng = np.random.RandomState(0)
wall_material = kb.FlatMaterial(
    color=kb.get_color("white"), indirect_visibility=True
)

bunny = kb.FileBasedObject(
    render_filename="objects/bunny.obj",
    name="bunny",
    scale=(4.89, 4.89, 4.89),
    position=(0, -1, -0.47044),
    quaternion=(0.0, 0.0, 0.707, 0.707),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)
suzanne = kb.FileBasedObject(
    render_filename="objects/suzanne.obj",
    name="suzanne",
    scale=(0.316, 0.316, 0.316),
    position=(0, 0, 0.001821),
    quaternion=(0.5, 0.5, 0.5, 0.5),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)
teapot = kb.FileBasedObject(
    render_filename="objects/teapot.obj",
    name="teapot",
    scale=(0.19, 0.19, 0.19),
    position=(0, 1, -0.28363),
    quaternion=(0.707, 0.70, 0.0, 0.0),
    material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)),
)

scene += bunny
scene += suzanne
scene += teapot


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

camera = kb.PerspectiveCamera(
    name="camera",
    position=(0, 0.0, 6.0),
    quaternion=(1.0, 0.0, 0.0, 1.0),
)
scene.camera = camera


xs = np.linspace(-np.pi / 2, np.pi / 2, num_frames)


positions = {
    "bunny": interpolate_position(
        np.abs(np.cos(xs * 8.33)), points["bunny"], world_matrix["bunny"]
    ),
    "teapot": interpolate_position(
        np.abs(np.cos(xs * 5.13)), points["teapot"], world_matrix["teapot"]
    ),
    "suzanne": interpolate_position(
        np.abs(np.cos(xs * 7.11)), points["suzanne"], world_matrix["suzanne"]
    ),
}

for frame in range(1, num_frames + 1):
    bunny.position = positions["bunny"][frame - 1]
    bunny.keyframe_insert("position", frame)

    teapot.position = positions["teapot"][frame - 1]
    teapot.keyframe_insert("position", frame)

    suzanne.position = positions["suzanne"][frame - 1]
    suzanne.keyframe_insert("position", frame)


# --- renders the output
kb.as_path("output_top").mkdir(exist_ok=True)
np.save("output_top/suzanne.npy", positions["suzanne"])
np.save("output_top/teapot.npy", positions["teapot"])
np.save("output_top/bunny.npy", positions["bunny"])
np.save("output_top/camera_pos.npy", np.array(camera.position))
renderer.save_state("output_top/trio_top.blend")

frames_dict = renderer.render()
kb.write_image_dict(frames_dict, "output_top")
