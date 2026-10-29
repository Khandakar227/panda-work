"""Per-episode budgets: tool calls, simulated time, LLM tokens and dollars."""
from dataclasses import dataclass

from ..world.bus import Clock


@dataclass(frozen=True)
class BudgetLimits:
    max_calls: int | None = 40
    max_sim_seconds: float | None = 120.0
    max_tokens: int | None = None     # input + output, charged by the agent runner (M6)
    max_cost_usd: float | None = None  # charged by the agent runner (M6)


class Budget:
    def __init__(self, limits: BudgetLimits, clock: Clock):
        self.limits = limits
        self.clock = clock
        self.start_sim_time = clock.sim_time
        self.calls = 0
        self.tokens = 0
        self.cost_usd = 0.0

    @property
    def sim_seconds(self) -> float:
        return self.clock.sim_time - self.start_sim_time

    def charge_call(self) -> None:
        self.calls += 1

    def charge_llm(self, tokens: int, cost_usd: float) -> None:
        self.tokens += tokens
        self.cost_usd += cost_usd

    def exhausted(self) -> str | None:
        """Why the episode must stop, or None while there is budget left."""
        lim = self.limits
        if lim.max_calls is not None and self.calls >= lim.max_calls:
            return f"tool-call budget used up ({self.calls}/{lim.max_calls} calls)"
        if lim.max_sim_seconds is not None and self.sim_seconds >= lim.max_sim_seconds:
            return f"time budget used up ({self.sim_seconds:.1f}/{lim.max_sim_seconds:g} s)"
        if lim.max_tokens is not None and self.tokens >= lim.max_tokens:
            return f"token budget used up ({self.tokens}/{lim.max_tokens} tokens)"
        if lim.max_cost_usd is not None and self.cost_usd >= lim.max_cost_usd:
            return f"cost budget used up (${self.cost_usd:.2f}/${lim.max_cost_usd:.2f})"
        return None

    def footer(self) -> str:
        """One line appended to each turn's results, e.g. [t=1.20 s · calls 3/40 · ...]."""
        lim = self.limits
        parts = [f"t={self.clock.sim_time:.2f} s", f"step {self.clock.step}"]
        parts.append(f"calls {self.calls}" + (f"/{lim.max_calls}" if lim.max_calls else ""))
        if lim.max_sim_seconds:
            parts.append(f"time {self.sim_seconds:.1f}/{lim.max_sim_seconds:g} s")
        if lim.max_tokens:
            parts.append(f"tokens {self.tokens}/{lim.max_tokens}")
        if lim.max_cost_usd:
            parts.append(f"${self.cost_usd:.2f}/${lim.max_cost_usd:.2f}")
        return "[" + " · ".join(parts) + "]"
