"""Tool base class, results and errors.

A tool does one thing for the agent. run() returns typed data (machine-checkable, logged in
full); render() turns it into the short text and images the model actually sees.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..tiers import Tier
from .rendering import clean_schema, render_result

if TYPE_CHECKING:
    from ..world import World
    from .budget import Budget

ALL_TIERS = frozenset(Tier)


class Group(str, Enum):
    PERCEIVE = "perceive"  # read sensors; no effect on the world
    ACT = "act"            # moves the robot; runs one at a time, in order
    PREDICT = "predict"    # "what if" on a world copy or the self-model; no effect on the world
    META = "meta"          # task info, ending the episode


class ToolArgs(BaseModel):
    """Base for tool arguments. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")


class NoArgs(ToolArgs):
    pass


@dataclass(frozen=True)
class ToolErrorInfo:
    code: str         # stable, machine-readable: "invalid_arguments", "timeout", ...
    message: str      # what went wrong, for the model
    hint: str | None = None  # how to recover


class ToolError(Exception):
    """Raised by a tool for an expected failure. The agent receives it as an error result."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.info = ToolErrorInfo(code, message, hint)


@dataclass
class ToolResult:
    data: dict[str, Any] = field(default_factory=dict)
    text: str | None = None  # model-facing text; when None, render() formats `data`
    images: list[np.ndarray] = field(default_factory=list)  # uint8 RGB/gray, shown to the model
    error: ToolErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def halts_batch(self) -> bool:
        """For ACT tools: should the remaining actions of the same turn be skipped?"""
        return not self.ok or self.data.get("status") in ("failed", "partial")

    @classmethod
    def failure(cls, code: str, message: str, hint: str | None = None,
                data: dict[str, Any] | None = None) -> "ToolResult":
        return cls(data=dict(data or {}), error=ToolErrorInfo(code, message, hint))


@dataclass
class ToolContext:
    """Everything a tool may touch while it runs."""
    world: "World"
    budget: "Budget"
    task: Any = None                                       # set by the episode (M4)
    state: dict[str, Any] = field(default_factory=dict)    # per-episode scratch (e.g. "done")
    deadline: float | None = None                          # simulated-time deadline (deterministic)
    wall_deadline: float | None = None                     # monotonic wall-clock hang guard

    @property
    def tier(self) -> Tier:
        return self.world.tier

    def check_deadline(self) -> None:
        """Long-running tools call this inside their loops; past the deadline it raises.

        The deadline is in simulated time, so an episode's outcome does not depend on how fast
        the machine ran (a wall-clock timeout once broke exact replay when a watched, paced run
        on a loaded machine stopped a motion that replay then completed). The wall-clock guard
        is much longer and only catches a tool that stops advancing the simulation at all."""
        if self.deadline is not None and self.world.clock.sim_time > self.deadline:
            raise ToolError("timeout", "the tool ran out of time and was stopped",
                            hint="try a shorter or simpler request")
        if self.wall_deadline is not None and time.monotonic() > self.wall_deadline:
            raise ToolError("timeout", "the tool stopped making progress and was stopped",
                            hint="try a shorter or simpler request")


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    group: ClassVar[Group]
    tiers: ClassVar[frozenset[Tier]] = ALL_TIERS
    Args: ClassVar[type[ToolArgs]] = NoArgs
    timeout_s: ClassVar[float] = 30.0  # simulated seconds, enforced via ctx.check_deadline()

    @abstractmethod
    def run(self, ctx: ToolContext, args: ToolArgs) -> ToolResult:
        ...

    def render(self, result: ToolResult) -> list[dict]:
        """Anthropic content blocks (text, then images) the model sees for this result."""
        return render_result(result)

    def schema(self) -> dict:
        """Tool definition for the Anthropic API."""
        return {"name": self.name, "description": self.description,
                "input_schema": clean_schema(self.Args.model_json_schema())}


class FunctionTool(Tool):
    """A tool defined by a plain function; see ToolRegistry.tool()."""

    def __init__(self, fn, name: str, description: str, group: Group,
                 tiers: frozenset[Tier], args: type[ToolArgs], timeout_s: float,
                 takes_args: bool):
        self._fn = fn
        self._takes_args = takes_args  # fn(ctx, args), or fn(ctx) for argument-free tools
        self.name, self.description, self.group = name, description, group
        self.tiers, self.Args, self.timeout_s = frozenset(tiers), args, timeout_s

    def run(self, ctx, args):
        return self._fn(ctx, args) if self._takes_args else self._fn(ctx)
