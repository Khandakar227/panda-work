"""Finding objects: exact poses (simulator truth) and camera-based detection.

Detection stands in for a real perception model: it segments the image, back-projects the
depth of each object's pixels into 3D with the camera's intrinsics and pose, and summarises the
visible surface. Objects with too few visible pixels are missed, as a real detector would.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..core.scene import OBJECT_PREFIX
from ..world import World
from .sensing import latest

DETECT_SIZE = (640, 480)
MIN_PIXELS = 40  # fewer visible pixels than this: not detected


@dataclass(frozen=True)
class ObjectPose:
    name: str
    position: np.ndarray    # centre, m
    yaw: float              # rad, rotation about vertical
    dimensions: tuple[float, float, float]  # width, depth, height (m)


@dataclass(frozen=True)
class Detection:
    name: str
    center_xy: np.ndarray   # m, centre of the visible top surface
    top_z: float            # m, height of the visible top surface
    yaw: float              # rad, long axis of the footprint (ambiguous by 90 deg for squares)
    footprint: tuple[float, float]  # visible extent along / across that axis (m)
    pixels: int


def true_object_poses(world: World) -> list[ObjectPose]:
    """Simulator truth."""
    out = []
    for name in world.objects:
        pos, quat = world.object_pose(name)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis = mat.reshape(3, 3)[:, 0]
        geom = world.plant.geom(OBJECT_PREFIX + name)
        out.append(ObjectPose(name, pos, float(np.arctan2(x_axis[1], x_axis[0])),
                              _dimensions(int(geom.type[0]), geom.size)))
    return out


def _dimensions(geom_type: int, size: np.ndarray) -> tuple[float, float, float]:
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return tuple(float(2 * s) for s in size)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (float(2 * size[0]), float(2 * size[0]), float(2 * size[1]))
    return (float(2 * size[0]),) * 3


def detect_objects(world: World, camera: str, noise: tuple[float, float] | None = None,
                   rng: np.random.Generator | None = None) -> tuple[list[Detection], list[str]]:
    """Detections of visible objects, and names of objects that were not detected.
    noise = (position std in m, yaw std in rad), drawn from `rng`."""
    seg = world.render(camera, "seg", DETECT_SIZE)
    depth = world.render(camera, "depth", DETECT_SIZE)
    points = _backproject(world, camera, depth)
    found, missed = [], []
    for name in world.objects:
        gid = world.plant.geom(OBJECT_PREFIX + name).id
        mask = (seg[..., 0] == gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        if mask.sum() < MIN_PIXELS:
            missed.append(name)
            continue
        found.append(_summarise(name, points[mask], int(mask.sum()), noise, rng))
    return found, missed


def _backproject(world: World, camera: str, depth: np.ndarray) -> np.ndarray:
    """World coordinates (H, W, 3) of every pixel. MuJoCo cameras look along -z, y up.
    The image is real, but where the camera is comes from the robot's self-model at its
    encoder readings, as on a real robot: a camera mounted or an arm positioned differently
    from what the robot believes skews the estimate."""
    pos, rot = believed_camera_pose(world, camera)
    fovy = world.belief.model.camera(camera).fovy[0]
    height, width = depth.shape
    focal = (height / 2) / np.tan(np.radians(fovy) / 2)
    v, u = np.mgrid[0:height, 0:width]
    x = (u - (width - 1) / 2) * depth / focal
    y = -(v - (height - 1) / 2) * depth / focal
    local = np.stack([x, y, -depth], axis=-1)
    return local @ rot.T + pos


def believed_camera_pose(world: World, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, 3x3 rotation) of a camera according to the self-model at the current
    encoder readings (fixed cameras: where the robot was told they are)."""
    belief = world.belief
    belief.fk(latest(world, "/joints").data["q"])
    mujoco.mj_camlight(belief.model, belief.data)
    cid = belief.model.camera(camera).id
    return belief.data.cam_xpos[cid].copy(), belief.data.cam_xmat[cid].reshape(3, 3).copy()


def _summarise(name, pts, pixels, noise, rng) -> Detection:
    top_z = float(np.percentile(pts[:, 2], 98))
    top = pts[pts[:, 2] > top_z - 0.01][:, :2]  # the upper surface
    center = np.median(top, axis=0)
    yaw, footprint = _min_area_rectangle(top - center)
    if noise is not None:
        pos_std, yaw_std = noise
        center = center + rng.normal(0.0, pos_std, 2)
        top_z += float(rng.normal(0.0, pos_std))
        yaw += float(rng.normal(0.0, yaw_std))
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)  # a footprint axis has no direction
    return Detection(name, center, top_z, yaw, footprint, pixels)


def _min_area_rectangle(xy: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Orientation (rad, in [0, 90 deg)) and side lengths of the smallest rectangle around the
    points. Unlike principal axes, this also finds the orientation of square footprints."""
    angles = np.radians(np.arange(0.0, 90.0, 1.0))
    best = (np.inf, 0.0, (0.0, 0.0))
    for a in angles:
        axis, normal = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        along, across = np.ptp(xy @ axis), np.ptp(xy @ normal)
        if along * across < best[0]:
            best = (along * across, float(a), (float(along), float(across)))
    return best[1], best[2]
