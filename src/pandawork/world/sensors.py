"""Sensors: read the plant, add the tier's noise, and publish stamped messages on the bus.

A sensor is either periodic (published by World.step at `rate_hz`) or on demand (rate_hz=None,
published by World.capture). Sensors hold no per-world state, so world copies can share them.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import mujoco
import numpy as np

from ..core.robot import ARM_JOINTS, CAMERAS, FINGER_JOINTS, TOUCH_SENSORS, WRIST_FORCE, WRIST_TORQUE
from ..tiers import Tier
from .noise import Gaussian, PixelNoise

if TYPE_CHECKING:
    from .world import World

DEFAULT_RATE_HZ = 500.0  # every physics step at the default 2 ms timestep


class Sensor(ABC):
    def __init__(self, name: str, topic: str, frame_id: str, rate_hz: float | None,
                 noise: Mapping[Tier, Any] | None = None, retention: float | None = None):
        self.name = name
        self.topic = topic
        self.frame_id = frame_id
        self.rate_hz = rate_hz
        self.noise = dict(noise or {})
        self.retention = retention  # seconds of history the bus keeps; None = bus default

    def bind(self, model: mujoco.MjModel) -> None:
        """Precompute model indices. Called once by World.add_sensor."""

    @abstractmethod
    def read(self, world: "World") -> dict[str, Any]:
        """Exact (noise-free) reading of the current state."""

    def sample(self, world: "World", rng: np.random.Generator) -> dict[str, Any]:
        data = self.read(world)
        noise = self.noise.get(world.tier)
        return noise.apply(data, rng) if noise else data

    def period_steps(self, timestep: float) -> int | None:
        return None if self.rate_hz is None else max(1, round(1.0 / (self.rate_hz * timestep)))


def _sensor_slices(model: mujoco.MjModel, names) -> list[slice]:
    out = []
    for name in names:
        s = model.sensor(name)
        out.append(slice(s.adr[0], s.adr[0] + s.dim[0]))
    return out


def _read(data: mujoco.MjData, slices: list[slice]) -> np.ndarray:
    return np.concatenate([data.sensordata[s] for s in slices])


class JointStateSensor(Sensor):
    """Arm joint angles, velocities and torques, plus gripper opening (sum of both fingers)."""

    def __init__(self, rate_hz: float | None = DEFAULT_RATE_HZ):
        noisy = Gaussian({"q": 1e-4, "dq": 1e-3, "tau": 0.05, "gripper_width": 1e-4})
        super().__init__("joints", "/joints", "base", rate_hz,
                         noise={Tier.REALISTIC: noisy, Tier.VISION_ONLY: noisy})

    def bind(self, model):
        self._q = _sensor_slices(model, [f"{j}_pos" for j in ARM_JOINTS])
        self._dq = _sensor_slices(model, [f"{j}_vel" for j in ARM_JOINTS])
        self._tau = _sensor_slices(model, [f"{j}_tau" for j in ARM_JOINTS])
        self._fingers = _sensor_slices(model, [f"{j}_pos" for j in FINGER_JOINTS])
        self._dof = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])

    def read(self, world):
        d = world.data
        # Motor torque = servo torque + the gravity-compensation torque the controller added.
        tau = _read(d, self._tau) + d.qfrc_applied[self._dof]
        return {"q": world.fault.encoders(_read(d, self._q)), "dq": _read(d, self._dq),
                "tau": tau, "gripper_width": float(_read(d, self._fingers).sum())}


class WristFTSensor(Sensor):
    """Force (N) and torque (Nm) between link7 and the hand, in the sensor ("ft") frame."""

    def __init__(self, rate_hz: float | None = DEFAULT_RATE_HZ):
        noisy = Gaussian({"force": 0.1, "torque": 0.005})
        super().__init__("wrist_ft", "/wrist_ft", "ft", rate_hz,
                         noise={Tier.REALISTIC: noisy})

    def bind(self, model):
        self._force, self._torque = _sensor_slices(model, [WRIST_FORCE, WRIST_TORQUE])

    def read(self, world):
        d = world.data
        return {"force": d.sensordata[self._force].copy(), "torque": d.sensordata[self._torque].copy()}


class TouchSensor(Sensor):
    """Normal force (N) on the left and right fingertip pads."""

    def __init__(self, rate_hz: float | None = DEFAULT_RATE_HZ):
        noisy = Gaussian({"left": 0.05, "right": 0.05}, nonnegative=("left", "right"))
        super().__init__("touch", "/touch", "fingers", rate_hz, noise={Tier.REALISTIC: noisy})

    def bind(self, model):
        self._slices = _sensor_slices(model, TOUCH_SENSORS)

    def read(self, world):
        left, right = (float(world.data.sensordata[s][0]) for s in self._slices)
        return {"left": left, "right": right}


class CameraSensor(Sensor):
    """On-demand camera image. RGB -> uint8 (H, W, 3); depth -> metres (H, W); seg -> (H, W, 2)."""

    def __init__(self, camera: str, mode: str = "rgb", size: tuple[int, int] = (640, 480),
                 rate_hz: float | None = None):
        noise = {"rgb": {Tier.REALISTIC: PixelNoise(3.0), Tier.VISION_ONLY: PixelNoise(3.0)},
                 "depth": {Tier.REALISTIC: Gaussian({"image": 0.002})},
                 "seg": {}}[mode]
        super().__init__(f"{camera}_{mode}", f"/cam/{camera}/{mode}", f"{camera}_camera",
                         rate_hz, noise=noise, retention=0.0)
        self.camera, self.mode, self.size = camera, mode, size

    def bind(self, model):
        self._fovy = float(model.camera(self.camera).fovy[0])

    def read(self, world):
        image = world.render(self.camera, self.mode, self.size)
        return {"image": image, "fovy": self._fovy}


def default_sensors() -> list[Sensor]:
    cams = [CameraSensor(cam, mode) for cam in CAMERAS for mode in ("rgb", "depth", "seg")]
    return [JointStateSensor(), WristFTSensor(), TouchSensor(), *cams]
