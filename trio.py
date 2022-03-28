import logging

import bpy
import numpy as np

import kubric as kb
from kubric.core import materials
from kubric.renderer import Blender
from kubric.renderer.blender import Blender as KubricBlender


def get_proper_time(nums: int, times: np.ndarray) -> np.ndarray:
    return np.concatenate([times] * nums)


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
    "camera_train": np.array(
        (
            (1.0, 0.0, 0.0, 5.299691677093506),
            (0.0, 1.0, 0.0, 2.9463558197021484),
            (0.0, 0.0, 1.0, 4.46529483795166),
            (0.0, 0.0, 0.0, 1.0),
        )
    ),
    "camera_valid": np.array(
        (
            (1.0, 0.0, 0.0, 5.299691677093506),
            (0.0, 1.0, 0.0, 2.946355819702148),
            (0.0, 0.0, 1.0, 3.7338485717773438),
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
    "camera": np.array(
        (
            (
                0.11302351206541061,
                -2.022498846054077,
                -0.03925067186355591,
                1.0,
            ),
            (0.2999248206615448, -2.991917133331299, 0.536824643611908, 1.0),
            (0.1303337812423706, -5.388640403747559, 0.32903361320495605, 1.0),
            (0.3103395700454712, -6.060530662536621, -0.3085057735443115, 1.0),
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
cam_frames = 20
obj_frames = 45
num_fake_cameras = 100


obj_xs = np.linspace(-np.pi / 2, np.pi / 2, obj_frames)
cam_xs = np.linspace(-np.pi / 2, np.pi / 2, cam_frames)
fake_xs = np.linspace(-np.pi / 2, np.pi / 2, num_fake_cameras)


train_times = {
    "bunny": np.abs(np.cos(obj_xs * 3.33)),
    "teapot": np.abs(np.cos(obj_xs * 3.33)),
    "suzanne": np.abs(np.cos(obj_xs * 3.33)),
    "camera": np.abs(np.cos(cam_xs) ** 2),
}
val_times = {
    "bunny": np.abs(np.cos(obj_xs * 3.33)),
    "teapot": np.abs(np.cos(obj_xs * 2.13)),
    "suzanne": np.abs(np.cos(obj_xs * 4.11)),
    "camera": np.abs(np.cos(cam_xs) ** 2),
}


def generate_single_dataset(times: np.ndarray, is_train: bool):
    scene = kb.Scene(resolution=(256, 256))
    scene.frame_end = cam_frames * obj_frames  # < numbers of frames to render
    scene.frame_rate = 24  # < rendering framerate
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
    renderer = KubricBlender(scene)

    # --- populate the scene with objects, lights, cameras
    rng = np.random.RandomState(0)
    wall_material = kb.FlatMaterial(color=kb.get_color("silver"))
    floor_material = kb.FlatMaterial(color=kb.get_color("gray"))

    bunny_material = kb.PrincipledBSDFMaterial(
        color=kb.random_hue_color(rng=rng)
    )
    bunny = kb.FileBasedObject(
        render_filename="objects/bunny.obj",
        name="bunny",
        scale=(4.89, 4.89, 4.89),
        position=(0, -1, -0.47044),
        quaternion=(0.0, 0.0, 0.707, 0.707),
        material=bunny_material,
    )

    suzanne_material = kb.PrincipledBSDFMaterial(
        color=kb.random_hue_color(rng=rng)
    )
    suzanne = kb.FileBasedObject(
        render_filename="objects/suzanne.obj",
        name="suzanne",
        scale=(0.316, 0.316, 0.316),
        position=(0, 0, 0.001821),
        quaternion=(0.5, 0.5, 0.5, 0.5),
        material=suzanne_material,
    )

    teapot_material = kb.PrincipledBSDFMaterial(
        color=kb.random_hue_color(rng=rng)
    )
    teapot = kb.FileBasedObject(
        render_filename="objects/teapot.obj",
        name="teapot",
        scale=(0.19, 0.19, 0.19),
        position=(0, 1, -0.28363),
        quaternion=(0.707, 0.70, 0.0, 0.0),
        material=teapot_material,
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

    scene += kb.Cube(
        scale=(100, 100, 0.01),
        position=(0, 0, -0.7),
        material=floor_material,
        static=True,
        background=True,
    )

    scene += kb.DirectionalLight(
        name="sun", position=(4, 0, 3), look_at=(0, 0, 0), intensity=1.5
    )

    camera = kb.PerspectiveCamera(
        name="camera", position=(0.64658, 0.81138, 0.50452), look_at=(0, 0, 0)
    )
    fake_camera = kb.PerspectiveCamera(
        name="fake_camera",
        position=(0.64658, 0.81138, 0.50452),
        look_at=(0, 0, 0),
    )
    scene.camera = camera
    suffix = "_train" if is_train else "_valid"

    positions = {
        "bunny": interpolate_position(
            times["bunny"],
            points["bunny"],
            world_matrix["bunny"],
        ),
        "teapot": interpolate_position(
            times["teapot"],
            points["teapot"],
            world_matrix["teapot"],
        ),
        "suzanne": interpolate_position(
            times["suzanne"],
            points["suzanne"],
            world_matrix["suzanne"],
        ),
        "camera": interpolate_position(
            times["camera"],
            points["camera"],
            world_matrix[f"camera{suffix}"],
        ),
        "fake_camera": interpolate_position(
            np.abs(np.cos(fake_xs) ** 2),
            points["camera"],
            world_matrix[f"camera{suffix}"],
        ),
    }

    min_z = 10
    for obj in ["bunny", "teapot", "suzanne"]:
        min_z = positions[obj][..., -1].min()

    for obj in ["bunny", "teapot", "suzanne"]:
        if obj == "bunny":
            positions[obj][..., -1] = min_z - 0.42538
        elif obj == "teapot":
            positions[obj][..., -1] = min_z - 0.22788
    for obj in ["bunny", "suzanne"]:
        positions[obj][..., 1] = positions[obj][..., 1].min()
    for obj in ["teapot"]:
        positions[obj][..., 1] = positions[obj][..., 1].max()

    positions = {
        "bunny": (
            np.zeros_like(positions["bunny"])
            + np.mean(positions["bunny"], axis=0)
        ),
        "teapot": (
            np.zeros_like(positions["teapot"])
            + np.mean(positions["teapot"], axis=0)
        ),
        "suzanne": (
            np.zeros_like(positions["suzanne"])
            + np.mean(positions["suzanne"], axis=0)
        ),
        "camera": positions["camera"],
        "fake_camera": positions["fake_camera"],
    }

    camera_matrices = []

    total_frame = 1
    for cam_frame in range(1, cam_frames + 1):
        for obj_frame in range(1, obj_frames + 1):
            bunny.position = positions["bunny"][obj_frame - 1]
            bunny_material.color = kb.Color.from_hsv(
                times["bunny"][obj_frame - 1] / 2, 1, 1
            )
            bunny.keyframe_insert("position", total_frame)
            bunny_material.keyframe_insert("color", total_frame)

            teapot.position = positions["teapot"][obj_frame - 1]
            teapot_material.color = kb.Color.from_hsv(
                times["teapot"][obj_frame - 1] / 2, 1, 1
            )
            teapot.keyframe_insert("position", total_frame)
            teapot_material.keyframe_insert("color", total_frame)

            suzanne.position = positions["suzanne"][obj_frame - 1]
            suzanne_material.color = kb.Color.from_hsv(
                times["suzanne"][obj_frame - 1] / 2, 1, 1
            )
            suzanne.keyframe_insert("position", total_frame)
            suzanne_material.keyframe_insert("color", total_frame)

            camera.position = positions["camera"][cam_frame - 1]
            camera.look_at((0, 0, 0))
            camera.keyframe_insert("position", total_frame)
            camera.keyframe_insert("quaternion", total_frame)

            camera_matrices.append(camera.matrix_world)
            total_frame += 1

    fake_camera_matrices = []
    for frame in range(1, num_fake_cameras + 1):
        fake_camera.position = positions["fake_camera"][frame - 1]
        fake_camera.look_at((0, 0, 0))
        fake_camera.keyframe_insert("position", frame)
        fake_camera.keyframe_insert("quaternion", frame)

        fake_camera_matrices.append(fake_camera.matrix_world)

    # --- renders the output
    kb.as_path(f"output{suffix}").mkdir(exist_ok=True)
    np.save(f"output{suffix}/suzanne.npy", positions["suzanne"])
    np.save(f"output{suffix}/teapot.npy", positions["teapot"])
    np.save(f"output{suffix}/bunny.npy", positions["bunny"])

    np.save(
        f"output{suffix}/bunny_time.npy",
        get_proper_time(cam_frames, times["bunny"]),
    )
    np.save(
        f"output{suffix}/teapot_time.npy",
        get_proper_time(cam_frames, times["teapot"]),
    )
    np.save(
        f"output{suffix}/suzanne_time.npy",
        get_proper_time(cam_frames, times["suzanne"]),
    )
    np.save(
        f"output{suffix}/camera_time.npy",
        get_proper_time(cam_frames, times["camera"]),
    )

    np.save(f"output{suffix}/camera.npy", np.array(camera_matrices))
    np.save(f"output{suffix}/fake_camera.npy", np.array(fake_camera_matrices))
    renderer.save_state(f"output{suffix}/trio.blend")

    frames_dict = renderer.render()
    kb.write_image_dict(frames_dict, f"output{suffix}")


# generate_single_dataset(train_times, True)
generate_single_dataset(val_times, False)
