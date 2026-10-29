"""The three cameras render offscreen and actually see the robot."""
import mujoco
import numpy as np
import pytest

from pandawork.core.robot import CAMERAS, reset_home

SIZE = (320, 240)


def _robot_pixel_fraction(model, seg, bodies=None):
    obj_id, obj_type = seg[..., 0], seg[..., 1]
    is_geom = (obj_type == mujoco.mjtObj.mjOBJ_GEOM) & (obj_id >= 0)
    body = np.where(is_geom, model.geom_bodyid[np.clip(obj_id, 0, None)], 0)
    wanted = body != 0 if bodies is None else np.isin(body, [model.body(b).id for b in bodies])
    return float(np.mean(is_geom & wanted))


@pytest.mark.parametrize("camera", list(CAMERAS))
def test_camera_modes(models, cameras, camera):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    rgb = cameras.render(data, camera, "rgb", SIZE)
    depth = cameras.render(data, camera, "depth", SIZE)
    seg = cameras.render(data, camera, "seg", SIZE)
    assert rgb.shape == (SIZE[1], SIZE[0], 3) and rgb.dtype == np.uint8
    assert rgb.std() > 5, "image is blank"
    assert depth.shape == SIZE[::-1] and np.all(np.isfinite(depth)) and depth.min() > 0
    assert seg.shape == (SIZE[1], SIZE[0], 2)


@pytest.mark.parametrize("camera", ["front", "top"])
def test_fixed_cameras_see_the_arm(models, cameras, camera):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    seg = cameras.render(data, camera, "seg", SIZE)
    assert _robot_pixel_fraction(plant, seg) > 0.01


def test_wrist_camera_sees_the_fingers(models, cameras):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    seg = cameras.render(data, "wrist", "seg", SIZE)
    assert _robot_pixel_fraction(plant, seg, ["left_finger", "right_finger"]) > 0.005


def test_images_do_not_depend_on_render_history(models, cameras):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    first = cameras.render(data, "front", "rgb", SIZE)
    cameras.render(data, "top", "rgb", SIZE)
    again = cameras.render(data, "front", "rgb", SIZE)
    assert np.array_equal(first, again)


def test_corrupted_segmentation_render_rebuilds_the_renderers_and_retries(monkeypatch):
    """A segmentation image whose pixel ids are not scene geoms (a lost GL context) must not
    poison the episode: the renderers are rebuilt once and the render retried."""
    import mujoco
    from pandawork.tasks import make_episode
    episode = make_episode("pick_place", seed=0)
    cams = episode.world.cameras
    good = cams.render(episode.world.data, "top", "seg", (320, 240))
    real_render = mujoco.Renderer.render
    state = {"fail": 1}

    def flaky(self, *a, **k):
        if state["fail"]:
            state["fail"] -= 1
            raise IndexError("index 16777215 is out of bounds for axis 0 with size 69")
        return real_render(self, *a, **k)
    monkeypatch.setattr(mujoco.Renderer, "render", flaky)
    again = cams.render(episode.world.data, "top", "seg", (320, 240))
    assert cams.rebuilds == 1 and (again == good).all()
    with monkeypatch.context() as m:  # a second failure in a row is a real error
        state["fail"] = 2
        import pytest
        with pytest.raises(IndexError):
            cams.render(episode.world.data, "top", "seg", (320, 240))
    episode.world.close()


def test_package_defaults_offscreen_rendering_to_egl_on_linux():
    import os, sys
    import pandawork  # noqa: F401
    if sys.platform == "linux":
        assert os.environ.get("MUJOCO_GL")  # set by pandawork/__init__.py unless the user chose one
