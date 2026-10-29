"""Offscreen rendering of the model's named cameras (RGB, depth, segmentation).

Uses whichever GL backend MuJoCo picks; the package defaults to EGL on Linux (see
pandawork/__init__.py). If a segmentation render comes back corrupted (pixel ids that are not
scene geoms: the sign of a lost GL context, seen with the desktop backend), the renderers are
rebuilt once and the render retried, so one bad context does not poison the rest of an episode.
"""
import sys

import mujoco
import numpy as np

MODES = ("rgb", "depth", "seg")


class Cameras:
    """Renders named cameras. One renderer per image size, created on first use."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.rebuilds = 0  # times the renderers were thrown away after a corrupted render

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render(self, data: mujoco.MjData, camera: str, mode: str = "rgb",
               size: tuple[int, int] = (320, 240)) -> np.ndarray:
        """RGB -> (H, W, 3) uint8; depth -> (H, W) metres; seg -> (H, W, 2) [object id, type]."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        try:
            return self._render(data, camera, mode, size)
        except IndexError:
            # mujoco.Renderer maps segmentation pixels through a table of scene geoms; an id
            # outside it means the pixels were not a segmentation image at all.
            self.rebuild()
            print(f"pandawork: corrupted {mode} render from camera {camera!r}; GL renderers rebuilt "
                  f"(rebuild {self.rebuilds})", file=sys.stderr)
            return self._render(data, camera, mode, size)

    def _render(self, data, camera, mode, size) -> np.ndarray:
        r = self._renderer(*size)
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        if mode == "depth":
            r.enable_depth_rendering()
        elif mode == "seg":
            r.enable_segmentation_rendering()
        # Twice on purpose: MuJoCo places the headlight from the camera of the *previous* scene
        # update, so a single update makes an image depend on which camera was rendered before.
        r.update_scene(data, camera=camera)
        r.update_scene(data, camera=camera)
        return r.render()

    def rebuild(self) -> None:
        """Throw the renderers (and their GL contexts) away; the next render creates new ones."""
        self.close()
        self.rebuilds += 1

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
