"""An episode: one task, one seed, one tier — a ready world with its tools."""
from dataclasses import dataclass

import numpy as np

from ..tiers import Tier
from ..tools import Budget, BudgetLimits, Dispatcher, ToolContext, default_registry
from ..world import World
from .base import Evaluation, Task
from .library import TASK_CLASSES

SETTLE_SECONDS = 0.3  # let objects settle and fill sensor buffers before the agent starts


@dataclass
class Episode:
    task: Task
    world: World
    dispatcher: Dispatcher
    seed: int

    def evaluate(self) -> Evaluation:
        return self.task.evaluate(self.world)

    def reopen(self) -> None:
        """Start a new instruction in an interactive session: done() ended only the last one."""
        for key in ("done", "claimed_success", "explanation"):
            self.dispatcher.ctx.state.pop(key, None)


def make_episode(task: str | Task, tier: Tier = Tier.PRIVILEGED, seed: int = 0,
                 limits: BudgetLimits | None = None) -> Episode:
    task = TASK_CLASSES[task]() if isinstance(task, str) else task
    self_model, plant = task.models()
    world = World(plant, self_model, tier=tier, seed=seed)
    if task.start_deg is not None:
        world.set_arm(np.radians(task.start_deg))
    task.reset(world, task.seed_rng(seed))
    world.step_hooks.append(task.monitor)
    world.step(round(SETTLE_SECONDS / world.timestep))
    limits = limits or BudgetLimits(max_calls=40, max_sim_seconds=task.max_seconds)
    ctx = ToolContext(world=world, budget=Budget(limits, world.clock), task=task)
    return Episode(task, world, Dispatcher(default_registry(), ctx), seed)
