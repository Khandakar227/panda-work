"""The tool registry: every tool the harness offers, filtered per access tier."""
import inspect
from typing import Callable, Iterable

from ..tiers import Tier
from .base import ALL_TIERS, FunctionTool, Group, NoArgs, Tool, ToolArgs

_GROUP_ORDER = {g: i for i, g in enumerate(Group)}


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, *, name: str, description: str, group: Group,
             tiers: Iterable[Tier] = ALL_TIERS, timeout_s: float = 30.0) -> Callable:
        """Decorator: register `fn(ctx, args) -> ToolResult` as a tool. The argument model is
        taken from the type annotation of `args` (no annotation: no arguments)."""
        def decorate(fn):
            params = list(inspect.signature(fn).parameters.values())
            annotation = params[1].annotation if len(params) > 1 else inspect.Parameter.empty
            args = annotation if isinstance(annotation, type) and issubclass(annotation, ToolArgs) \
                else NoArgs
            return self.register(FunctionTool(fn, name, description, group, frozenset(tiers),
                                              args, timeout_s, takes_args=len(params) > 1))
        return decorate

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def available(self, tier: Tier) -> list[Tool]:
        """Tools usable in `tier`, in a fixed order (by group, then name) so the tool list the
        model sees is byte-identical every time, which keeps prompt caching effective."""
        tools = [t for t in self._tools.values() if tier in t.tiers]
        return sorted(tools, key=lambda t: (_GROUP_ORDER[t.group], t.name))

    def schemas(self, tier: Tier) -> list[dict]:
        return [t.schema() for t in self.available(tier)]

    def __len__(self) -> int:
        return len(self._tools)
