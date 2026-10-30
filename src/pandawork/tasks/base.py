"""Tasks: a scene, a layout randomised per seed, an instruction, and a success check.

Success is judged from the state of the world, never from what the agent claims. Every task
also has an oracle: a scripted solution that uses the same tool calls the agent has (with
privileged perception). It proves the task is solvable with the tools, and is a baseline.
"""
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

import mujoco
import numpy as np

from ..core.robot import panda_spec
from ..tools import Dispatcher, ToolCall
from ..world import World

WORKSPACE_X = (0.35, 0.65)
WORKSPACE_Y = (-0.25, 0.25)


@dataclass(frozen=True)
class Evaluation:
    success: bool
    summary: str                       # one human-readable line
    metrics: dict[str, Any] = field(default_factory=dict)
    # How far the task got, 0..1 (1 = done), for grading partial outcomes: an episode that
    # dropped the cube 6 cm from the pad scores higher than one that never lifted it. None:
    # the task has no partial scale, and progress is 1.0 on success, 0.0 otherwise.
    progress: float | None = None

    @property
    def score(self) -> float:
        return float(self.success) if self.progress is None else float(min(1.0, max(0.0, self.progress)))


class Task(ABC):
    name: ClassVar[str]
    level: ClassVar[int] = 1             # 1: one step; 2: several steps; 3: long horizon
    max_seconds: ClassVar[float] = 60.0  # suggested simulated-time budget for an episode
    # How a human viewer (live console, oracle --watch) should look at the scene: MuJoCo free
    # camera settings {lookat, distance, azimuth, elevation}; None: the default view.
    view: ClassVar[dict[str, Any] | None] = None
    # Arm joint angles (deg) to start from instead of the "home" keyframe; None: home.
    start_deg: ClassVar[tuple[float, ...] | None] = None

    def __init__(self):
        self.instruction = ""

    @classmethod
    @abstractmethod
    def build_spec(cls) -> mujoco.MjSpec:
        """The scene shared by every episode of this task (layout is randomised in reset)."""

    @classmethod
    def models(cls) -> tuple[mujoco.MjModel, mujoco.MjModel]:
        """(self_model, plant), compiled once per task class."""
        return _compile(cls)

    @abstractmethod
    def reset(self, world: World, rng: np.random.Generator) -> None:
        """Lay out the scene for this episode and set self.instruction. Called after world.reset."""

    def monitor(self, world: World) -> None:
        """Called after every physics step, for events that must be caught as they happen."""

    def fixture_states(self, world: World) -> dict[str, Any]:
        """Exact state of the scene's mechanisms (drawer opening, switches, ...): simulator
        truth, shown only in the privileged tier."""
        return {}

    def involved(self) -> set[str]:
        """Names of the objects the task is about. Anything else the robot touches or moves is
        a side effect (recorded as such; it does not change the verdict). Valid after reset."""
        target = getattr(self, "target", None)
        return {str(target)} if target is not None else set()

    @abstractmethod
    def evaluate(self, world: World) -> Evaluation:
        ...

    @abstractmethod
    def oracle(self, d: Dispatcher) -> None:
        """Solve the task with tool calls (privileged tier)."""

    def seed_rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng([seed, zlib.crc32(self.name.encode())])


@lru_cache(maxsize=None)
def _compile(task_cls) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    plant_spec = task_cls.build_spec()
    return panda_spec().compile(), plant_spec.compile()


def sample_xy(rng: np.random.Generator, n: int, min_dist: float, x=WORKSPACE_X, y=WORKSPACE_Y,
              avoid: list[np.ndarray] = (), avoid_dist: float | None = None) -> list[np.ndarray]:
    """n random table positions, at least min_dist apart from each other and avoid_dist
    (default min_dist) from each point in `avoid`."""
    avoid_dist = min_dist if avoid_dist is None else avoid_dist
    placed: list[np.ndarray] = []
    for _ in range(10_000):
        p = np.array([rng.uniform(*x), rng.uniform(*y)])
        if (all(np.linalg.norm(p - q) >= min_dist for q in placed)
                and all(np.linalg.norm(p - q) >= avoid_dist for q in avoid)):
            placed.append(p)
            if len(placed) == n:
                return placed
    raise RuntimeError("could not place objects; the workspace is too crowded")


def yaw_quat_z(yaw: float) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
    return quat


def grasp_yaw(object_yaw: float, near: float = 0.0) -> float:
    """Gripper yaw (rad) aligned with a square object's faces, choosing among the four aligned
    yaws the one closest to `near` (e.g. pi/2 to keep the hand's long side along y)."""
    base = (object_yaw - near + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(near + base)


def call(d: Dispatcher, *turn: tuple[str, dict]) -> list[dict]:
    """Run one turn of tool calls (for oracles); returns the tool_result blocks."""
    return d.run_batch([ToolCall(f"oracle_t{d.turn}_{i}", name, args)
                        for i, (name, args) in enumerate(turn)])
