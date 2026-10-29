"""Cost ledger: turns the token usage of every LLM call into dollars, logs it, and charges the
episode budget. The Claude agent (M6) calls CostLedger.record() after each API response.

Claude Code reports what each model cost (costUSD), priced with its own, current price list;
when that is given, the ledger uses it, so the two always agree. PRICES is the fallback for
callers without such a report. A model in neither is recorded with its tokens at $0 and
flagged "unpriced" (a bookkeeping gap must never end an episode).

Prices are USD per million tokens, from the Claude API reference as of late September 2026.
Cache writes cost 1.25x the input price (5-minute cache) or 2x (1-hour cache, which Claude
Code uses); cache reads are listed per model.
"""
from dataclasses import dataclass, field

from ..tools import Budget
from ..world import EventBus

USAGE_TOPIC = "/agent/llm_usage"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float


PRICES = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-haiku-4-5": Price(1.00, 5.00, 0.10),  # cache read assumed 10% of input; verify
}
CACHE_WRITE_FACTOR = {"5m": 1.25, "1h": 2.0}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0                  # uncached input
    output_tokens: int = 0                 # includes thinking
    cache_creation_input_tokens: int = 0   # all cache writes...
    cache_read_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0  # ...of which written to the 1-hour cache

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens + self.cache_creation_input_tokens
                + self.cache_read_input_tokens)


def cost_usd(model: str, usage: Usage) -> float:
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to PRICES")
    p = PRICES[model]
    write_1h = usage.cache_creation_1h_input_tokens
    write_5m = usage.cache_creation_input_tokens - write_1h
    return (usage.input_tokens * p.input + usage.output_tokens * p.output
            + write_5m * p.input * CACHE_WRITE_FACTOR["5m"]
            + write_1h * p.input * CACHE_WRITE_FACTOR["1h"]
            + usage.cache_read_input_tokens * p.cache_read) / 1e6


@dataclass
class CostLedger:
    bus: EventBus
    budget: Budget
    calls: list[tuple[str, Usage, float]] = field(default_factory=list)
    unpriced: set[str] = field(default_factory=set)  # models recorded at $0: no price known

    def record(self, model: str, usage: Usage, request_id: str = "", latency_s: float = 0.0,
               reported_usd: float | None = None) -> float:
        """Price one LLM call (with the provider's reported cost if given, else PRICES),
        publish it on the bus (so it is logged), and charge the budget."""
        if reported_usd is not None:
            cost, source = max(float(reported_usd), 0.0), "reported"
        elif model in PRICES:
            cost, source = cost_usd(model, usage), "price_table"
        else:
            cost, source = 0.0, "unpriced"
            self.unpriced.add(model)
        self.calls.append((model, usage, cost))
        self.budget.charge_llm(usage.total, cost)
        self.bus.publish(USAGE_TOPIC, {
            "model": model, "request_id": request_id, "latency_s": float(latency_s),
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cache_creation_1h_input_tokens": usage.cache_creation_1h_input_tokens, "cost_usd": cost,
            "cost_source": source,
        }, frame_id="agent")
        return cost

    @property
    def total_usd(self) -> float:
        return sum(c for _, _, c in self.calls)

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from the cache: should be high from turn 2 on."""
        read = sum(u.cache_read_input_tokens for _, u, _ in self.calls)
        all_input = sum(u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens
                        for _, u, _ in self.calls)
        return read / all_input if all_input else 0.0
