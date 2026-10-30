"""Aggregate a batch of episodes into a report: success and progress with confidence intervals,
how well the agent's claims match the world, effort, cost, side effects, and which tools it
used. Also episodes.csv, one row per episode, for plotting."""
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

# The headline task metric for each task (name in result.json "metrics", label, unit)
KEY_METRIC = {
    "reach": ("error_cm", "fingertip error", "cm"),
    "pick_place": ("distance_cm", "cube-to-pad distance", "cm"),
    "press_button": (None, "target travel", "mm"),
    "open_drawer": ("opening_cm", "drawer opening", "cm"),
    "stack": ("max_offset_cm", "worst cube offset in the tower", "cm"),
    "sort": ("sorted", "cubes in their bin", "of 4"),
    "stow": ("opening_cm", "drawer opening at the end", "cm"),
    "press_key": ("target_travel_mm", "target key travel", "mm"),
    "type_code": ("correct_digits", "code digits typed correctly", "of 4"),
}


@dataclass
class EpisodeRow:
    task: str
    tier: str
    seed: int
    success: bool
    claimed: bool | None      # None: the agent never called done
    turns: int
    calls: int
    sim_s: float
    wall_s: float | None
    cost: float
    metric: float | None
    tools: Counter = field(default_factory=Counter)
    images: int = 0
    budget_exhausted: bool = False
    progress: float = 0.0          # the task's partial score, 0..1 (= success when it has none)
    disturbed: int = 0             # bystander objects moved by more than 1 cm
    bystander_contact_s: float = 0.0
    peak_force: float | None = None


def load_rows(run_dir: Path) -> list[EpisodeRow]:
    rows = []
    for result_file in sorted(Path(run_dir).glob("*/result.json")):
        folder = result_file.parent
        if "." in folder.name:  # set aside (see batch.set_aside)
            continue
        meta = json.loads((folder / "meta.json").read_text())
        r = json.loads(result_file.read_text())
        tools, images = Counter(), 0
        for line in (folder / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["topic"] == "/agent/tool_call":
                tools[e["data"]["tool"]] += 1
            elif e["topic"] == "/agent/tool_result":
                images += e["data"].get("n_images", 0)
        sdk = r.get("sdk_result") or {}
        side = r.get("side_effects") or {}
        rows.append(EpisodeRow(
            task=meta["task"], tier=meta["tier"], seed=meta["seed"], success=r["success"],
            claimed=r.get("agent_claimed_success"), turns=r["usage"]["turns"],
            calls=r["usage"]["tool_calls"], sim_s=r["usage"]["sim_seconds"],
            wall_s=(sdk.get("duration_ms") or 0) / 1000 or None, cost=r["usage"]["cost_usd"],
            metric=_metric(meta["task"], r["metrics"], meta), tools=tools, images=images,
            budget_exhausted=bool(r.get("budget_exhausted")),
            progress=float(r.get("progress", float(r["success"]))),
            disturbed=len(side.get("disturbed") or {}),
            bystander_contact_s=sum((side.get("bystander_contact_s") or {}).values()),
            peak_force=side.get("peak_wrist_force_N")))
    return rows


def _metric(task: str, metrics: dict, meta: dict) -> float | None:
    if task == "press_button":  # travel of the button the instruction named
        target = next((o for o in meta["objects"] if o in meta["instruction"]), None)
        return metrics.get(f"{target}_travel_mm") if target else None
    key = KEY_METRIC.get(task, (None,))[0]
    return metrics.get(key) if key else None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate (sensible for small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Group:
    label: str
    rows: list[EpisodeRow]

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def successes(self) -> int:
        return sum(r.success for r in self.rows)

    def summary(self) -> dict:
        rows, n = self.rows, self.n
        lo, hi = wilson(self.successes, n)
        claimed = [r for r in rows if r.claimed is not None]
        over = sum(1 for r in rows if r.claimed and not r.success)
        under = sum(1 for r in rows if r.claimed is False and r.success)
        cost = sum(r.cost for r in rows)
        metrics = [r.metric for r in rows if r.metric is not None]
        return {
            "group": self.label, "n": n, "successes": self.successes,
            "success_rate": self.successes / n if n else 0.0, "ci95": (lo, hi),
            "mean_progress": mean(r.progress for r in rows) if rows else 0.0,
            "claimed_success": sum(1 for r in claimed if r.claimed),
            "over_claims": over, "under_claims": under,
            "no_done": sum(1 for r in rows if r.claimed is None),
            "claim_accuracy": (sum(1 for r in claimed if r.claimed == r.success) / len(claimed)
                               if claimed else None),
            "mean_turns": mean(r.turns for r in rows), "mean_calls": mean(r.calls for r in rows),
            "mean_sim_s": mean(r.sim_s for r in rows),
            "mean_wall_s": mean(r.wall_s for r in rows if r.wall_s) if any(r.wall_s for r in rows) else None,
            "mean_images": mean(r.images for r in rows),
            "total_cost": cost, "mean_cost": cost / n if n else 0.0,
            "cost_per_success": cost / self.successes if self.successes else None,
            "mean_metric": mean(metrics) if metrics else None,
            "budget_exhausted": sum(r.budget_exhausted for r in rows),
            "disturbed_episodes": sum(r.disturbed > 0 for r in rows),
            "mean_bystander_contact_s": mean(r.bystander_contact_s for r in rows) if rows else 0.0,
            "mean_peak_force": (mean(r.peak_force for r in rows if r.peak_force is not None)
                                if any(r.peak_force is not None for r in rows) else None),
        }


def groups(rows: list[EpisodeRow]) -> dict[str, list[Group]]:
    by = lambda key: [Group(k, v) for k, v in sorted(_bucket(rows, key).items())]
    return {"task_tier": by(lambda r: f"{r.task} / {r.tier}"), "tier": by(lambda r: r.tier),
            "task": by(lambda r: r.task), "all": [Group("all", rows)]}


def _bucket(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def render_markdown(run_dir: Path, rows: list[EpisodeRow]) -> str:
    g = groups(rows)
    batch = json.loads((Path(run_dir) / "batch.json").read_text()) if (Path(run_dir) / "batch.json").exists() else {}
    lines = [f"# Batch report: {Path(run_dir).name}", "",
             f"Agent: {batch.get('agent', '?')} · model {batch.get('model', '?')} · effort "
             f"{batch.get('effort', '?')} · {len(rows)} episodes", ""]

    def table(title, groups_):
        lines.extend([f"## {title}", "",
                      "| group | n | success | 95% CI | progress | claimed | over-claims | no done | turns | calls | images | cost/ep | cost/success | key metric | disturbed |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for grp in groups_:
            s = grp.summary()
            lo, hi = s["ci95"]
            cps = f"${s['cost_per_success']:.3f}" if s["cost_per_success"] else "-"
            metric = f"{s['mean_metric']:.2f}" if s["mean_metric"] is not None else "-"
            lines.append(
                f"| {s['group']} | {s['n']} | {s['successes']}/{s['n']} ({s['success_rate']:.0%}) | "
                f"{lo:.0%}-{hi:.0%} | {s['mean_progress']:.2f} | {s['claimed_success']} | {s['over_claims']} | {s['no_done']} | "
                f"{s['mean_turns']:.1f} | {s['mean_calls']:.1f} | {s['mean_images']:.1f} | "
                f"${s['mean_cost']:.3f} | {cps} | {metric} | {s['disturbed_episodes']} |")
        lines.append("")

    table("By tier", g["tier"])
    table("By task", g["task"])
    table("By task and tier", g["task_tier"])
    table("Overall", g["all"])
    tool_use = defaultdict(Counter)
    for r in rows:
        tool_use[r.tier].update(r.tools)
    lines.extend(["## Tool calls per episode, by tier", ""])
    tools = sorted({t for c in tool_use.values() for t in c})
    lines.append("| tool | " + " | ".join(sorted(tool_use)) + " |")
    lines.append("|---|" + "---|" * len(tool_use))
    n_by_tier = Counter(r.tier for r in rows)
    for t in tools:
        lines.append(f"| {t} | " + " | ".join(f"{tool_use[tier][t] / n_by_tier[tier]:.2f}"
                                               for tier in sorted(tool_use)) + " |")
    lines.extend(["", "Key metrics: " + "; ".join(f"{t}: {v[1]} ({v[2]})" for t, v in KEY_METRIC.items()),
                  "Over-claim: the agent called done claiming success, but the world check failed.",
                  "Progress: the task's partial score (0..1). Disturbed: episodes in which an object "
                  "the task is not about moved by over 1 cm.", ""])
    return "\n".join(lines)


CSV_FIELDS = ["episode", "task", "tier", "seed", "success", "progress", "claimed",
              "turns", "calls", "images", "sim_s", "wall_s", "cost", "metric", "budget_exhausted",
              "disturbed", "bystander_contact_s", "peak_force"]


def write_csv(run_dir: Path, rows: list[EpisodeRow]) -> Path:
    """episodes.csv: one row per episode."""
    path = Path(run_dir) / "episodes.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            row = {k: getattr(r, k) for k in CSV_FIELDS if hasattr(r, k)}
            row["episode"] = f"{r.task}-{r.tier}-s{r.seed}"
            w.writerow(row)
    return path


def write_report(run_dir: Path) -> tuple[Path, str]:
    rows = load_rows(run_dir)
    text = render_markdown(run_dir, rows)
    path = Path(run_dir) / "report.md"
    path.write_text(text)
    summary = {k: [grp.summary() for grp in v] for k, v in groups(rows).items()}
    (Path(run_dir) / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    write_csv(run_dir, rows)
    return path, text
