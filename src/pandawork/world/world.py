"""The simulated world: plant + self-model, a sim clock, an event bus, and sensors.

Time only moves inside step(). Between calls the world is paused, e.g. while the agent thinks.

Timing convention: after a physics step, MuJoCo's sensor values describe the state the step
started from. Periodic sensor messages are therefore stamped with that state's (sim_time, step),
so a message always describes the moment in its header.
"""
import copy
import time
import zlib
from typing import Callable

import mujoco
import numpy as np

from ..core.control import apply_control
from ..core.ik import SelfModel
from ..core.render import Cameras
from ..core.robot import ARM_JOINTS, FINGER_JOINTS, GRIPPER_OPEN, arm_indices, build_models
from ..core.scene import OBJECT_PREFIX, object_names
from ..tiers import Tier
from .bus import Clock, EventBus
from .messages import Message
from .sensors import Sensor, default_sensors


class World:
    def __init__(self, plant: mujoco.MjModel | None = None, self_model: mujoco.MjModel | None = None,
                 *, tier: Tier = Tier.PRIVILEGED, seed: int = 0,
                 sensors: list[Sensor] | None = None, wall_clock: Callable[[], float] = time.time):
        if plant is None or self_model is None:
            self_model, plant = build_models(with_target=False)
        self.plant = plant
        # How the body differs from the belief: shapes the encoder readings and the servo
        self.belief = SelfModel(self_model)
        self.data = mujoco.MjData(plant)
        self.qadr, self.dof = arm_indices(plant)
        self.tier = Tier(tier)
        self.clock = Clock(wall_clock)
        self.bus = EventBus(self.clock)
        self.sensors: dict[str, Sensor] = {}
        self._periods: dict[str, int] = {}
        self._cameras: Cameras | None = None
        # Called after every physics step, e.g. to drive a live viewer. Not copied by fork().
        self.step_hooks: list[Callable[["World"], None]] = []
        self.objects = object_names(plant)
        for sensor in default_sensors() if sensors is None else sensors:
            self.add_sensor(sensor)
        self.reset(seed)

    @property
    def timestep(self) -> float:
        return self.plant.opt.timestep

    @property
    def cameras(self) -> Cameras:
        if self._cameras is None:  # created on first use so GL is only needed for images
            self._cameras = Cameras(self.plant)
        return self._cameras

    def add_sensor(self, sensor: Sensor) -> None:
        if sensor.name in self.sensors:
            raise ValueError(f"duplicate sensor name {sensor.name!r}")
        sensor.bind(self.plant)
        self.sensors[sensor.name] = sensor
        period = sensor.period_steps(self.timestep)
        if period is not None:
            self._periods[sensor.name] = period
        if sensor.retention is not None:
            self.bus.set_retention(sensor.topic, sensor.retention)
        if hasattr(self, "seed"):
            self._rngs[sensor.name] = self._sensor_rng(sensor.name)

    def reset(self, seed: int | None = None, keyframe: str = "home") -> None:
        """Robot back to the keyframe, objects to their scene poses, t=0, fresh noise streams.
        Subscribers stay attached."""
        if seed is not None:
            self.seed = seed
        # The keyframe only knows the robot (it zeroes any object's free joint), so start from
        # the scene's default pose and copy just the robot's joints and controls from it.
        key = self.plant.key(keyframe)
        mujoco.mj_resetData(self.plant, self.data)
        robot_qpos = [self.plant.joint(j).qposadr[0] for j in [*ARM_JOINTS, *FINGER_JOINTS]]
        self.data.qpos[robot_qpos] = key.qpos[robot_qpos]
        self.data.ctrl[:] = key.ctrl
        mujoco.mj_forward(self.plant, self.data)
        self.q_cmd = self.data.qpos[self.qadr].copy()
        self.gripper = GRIPPER_OPEN
        self.clock.reset()
        self.bus.clear()
        # One noise stream per sensor, keyed by name, so adding a sensor never shifts the others.
        self._rngs = {name: self._sensor_rng(name) for name in self.sensors}

    def set_arm(self, q: np.ndarray) -> None:
        """Put the arm at rest at joint angles q (rad) and hold it there: a start posture other
        than the keyframe. Call before stepping (the sensors report it from the next step)."""
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0.0
        self.q_cmd = np.asarray(q, dtype=float).copy()
        mujoco.mj_forward(self.plant, self.data)

    def command(self, q: np.ndarray | None = None, gripper: float | None = None) -> None:
        """Set the joint-servo target (7 arm joints, radians) and/or the gripper command."""
        if q is not None:
            self.q_cmd = np.asarray(q, dtype=float).copy()
        if gripper is not None:
            self.gripper = float(gripper)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            stamp = (self.clock.sim_time, self.clock.step)
            apply_control(self.plant, self.data, self.belief, self.q_cmd,
                          self.gripper, q_measured=self.encoders())
            mujoco.mj_step(self.plant, self.data)
            self.clock.advance(self.data.time)
            for name, period in self._periods.items():
                if stamp[1] % period == 0:
                    self._publish(self.sensors[name], stamp)
            for hook in self.step_hooks:
                hook(self)

    def encoders(self) -> np.ndarray:
        """The arm joint angles as the robot's encoders report them. Everything the robot computes
        about itself starts here."""
        return self.data.qpos[self.qadr].copy()

    def capture(self, name: str) -> Message:
        """Publish an on-demand reading of the current state (e.g. a camera image)."""
        return self._publish(self.sensors[name], (self.clock.sim_time, self.clock.step))

    def render(self, camera: str, mode: str = "rgb", size: tuple[int, int] = (320, 240)) -> np.ndarray:
        """Exact image of the current state (no noise; sensors add it)."""
        mujoco.mj_kinematics(self.plant, self.data)  # poses after mj_step lag qpos by one step
        return self.cameras.render(self.data, camera, mode, size)

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Exact world position and quaternion of object `name` (simulator truth)."""
        body = self.data.body(OBJECT_PREFIX + name)
        return body.xpos.copy(), body.xquat.copy()

    def set_object_pose(self, name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport an object (scene setup only) and zero its velocity."""
        joint = self.plant.joint(self.plant.body(OBJECT_PREFIX + name).jntadr[0])
        qadr, dadr = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[qadr:qadr + 7] = [*pos, *quat]
        self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.plant, self.data)

    def fork(self) -> "World":
        """An independent copy at the current instant: same models, copied state, history and
        noise streams, no subscribers. Used to predict "what would happen if"."""
        twin = object.__new__(World)
        twin.qadr, twin.dof = self.qadr, self.dof
        twin.belief = SelfModel(self.belief.model)
        twin.plant, twin.data = self.plant, copy.copy(self.data)
        twin.tier, twin.seed = self.tier, self.seed
        twin.clock = self.clock.copy()
        twin.bus = self.bus.fork(twin.clock)
        twin.sensors, twin._periods = self.sensors, self._periods
        twin._cameras = None
        twin.step_hooks = []
        twin.objects = self.objects
        twin._rngs = copy.deepcopy(self._rngs)
        twin.q_cmd, twin.gripper = self.q_cmd.copy(), self.gripper
        return twin

    def close(self) -> None:
        if self._cameras is not None:
            self._cameras.close()
            self._cameras = None

    def rng(self, name: str) -> np.random.Generator:
        """A named random stream derived from the world seed (e.g. for detection noise).
        Streams restart on reset() and are copied by fork(), so runs stay reproducible."""
        if name not in self._rngs:
            self._rngs[name] = self._sensor_rng(name)
        return self._rngs[name]

    def _sensor_rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(name.encode())])

    def _publish(self, sensor: Sensor, stamp: tuple[float, int]) -> Message:
        data = sensor.sample(self, self._rngs[sensor.name])
        return self.bus.publish(sensor.topic, data, sensor.frame_id, stamp)
