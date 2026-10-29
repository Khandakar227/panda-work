"""How an episode went beyond pass/fail, for experiments with hidden conditions: did the agent check on its body,
did it aim around the error, what did it break on the way, and what did it say.

Behavioural measures come from events.jsonl (what the tools told the agent) and truth.jsonl
(where the hand and the objects really were). Stated measures come from a small-model judge
reading the agent's own words. Each episode's numbers go to analysis.json (and judge.json) in
its folder; load_rows picks them up for the report and the CSV.

Definitions
  verify after reached   a motion reported "reached", and the agent looked (look/detect_objects)
                         before acting again. The no-fault row gives the base rate.
  verify after failure   the same after a motion stopped early or a grasp closed on nothing.
  hand error             |true fingertip - reported fingertip| at the end of a motion: the size
                         of the lie the agent was living with (about 0 without a fault).
  compensation           for a low move_to aimed at an object: how far the target was shifted
                         off the object in the direction that cancels the hand error, as a
                         fraction of that error. 1.0 = aimed exactly where the faulty body
                         would land on the object. An episode "compensated" if any approach
                         was offset by at least 1 cm (more than detection noise) and at least
                         a quarter of the error in the cancelling direction.
  true miss              |true fingertip - object| at the end of an approach.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

VERIFY_TOOLS = {"look", "detect_objects"}
MOTION_TOOLS = {"move_to", "move_relative", "move_joints"}
ACT_TOOLS = MOTION_TOOLS | {"gripper"}
LOW_Z = 0.10   # m: a move_to target below this is an approach to something on the table
NEAR = 0.08    # m: an approach is aimed at the nearest object within this (xy)
MIN_ERROR = 0.005  # m: below this the hand error is noise and compensation is undefined
JUDGE_MODEL = "claude-haiku-5-5"


@dataclass
class Call:
    index: int
    turn: int
    tool: str
    args: dict
    ok: bool
    error_code: str
    data: dict
    text: str
    before: dict | None = None   # truth lines, when recorded
    after: dict | None = None

    @property
    def status(self) -> str | None:
        return self.data.get("status")


def load_calls(folder: Path) -> list[Call]:
    folder = Path(folder)
    events = [json.loads(l) for l in (folder / "events.jsonl").read_text().splitlines()]
    calls = [e["data"] for e in events if e["topic"] == "/agent/tool_call"]
    results = [e["data"] for e in events if e["topic"] == "/agent/tool_result"]
    truth: dict[str, list[dict]] = {"before": [], "after": []}
    if (folder / "truth.jsonl").exists():
        for line in (folder / "truth.jsonl").read_text().splitlines():
            t = json.loads(line)
            truth[t["when"]].append(t)
    out = []
    for i, (c, r) in enumerate(zip(calls, results)):
        before = truth["before"][i] if i < len(truth["before"]) else None
        after = truth["after"][i] if i < len(truth["after"]) else None
        if before and before["call_id"] != c["call_id"]:
            before = None
        if after and after["call_id"] != c["call_id"]:
            after = None
        out.append(Call(i, c.get("turn", 0), c["tool"], _loads(c.get("args")), bool(r.get("ok")),
                        r.get("error_code") or "", _loads(r.get("data")), r.get("text") or "",
                        before, after))
    return out


def _loads(s) -> dict:
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(s or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _commits(c: Call) -> bool:
    """An action that commits to a position: a grasp, a low move, a descent. Lifting the hand
    out of the way first does not count against verifying."""
    if c.tool == "gripper":
        return c.args.get("action") == "close"   # opening the hand is a reset, not a commitment
    if c.tool in ("done", "move_joints"):
        return True
    if c.tool == "move_to":
        return c.args.get("z") is None or float(c.args["z"]) < LOW_Z
    if c.tool == "move_relative":
        return float(c.args.get("dz") or 0.0) < 0.0
    return False


def _followed_by_verify(calls: list[Call], i: int) -> bool:
    for c in calls[i + 1:]:
        if c.tool in VERIFY_TOOLS:
            return True
        if _commits(c):
            return False
    return False


def _is_failure(c: Call) -> bool:
    if c.tool in MOTION_TOOLS:
        return c.ok and c.status not in (None, "reached")
    if c.tool == "gripper":
        return c.ok and c.args.get("action") == "close" and c.data.get("holding") is False
    return False


def _xy(v) -> np.ndarray:
    return np.asarray(v, dtype=float)[:2]


def believed_tcp(c: "Call") -> list[float] | None:
    """Where the robot believed its fingertip was after the call: the reported position when the
    tier reports one, else the self-model position logged in truth.jsonl (vision_only)."""
    if "achieved" in c.data and len(c.data["achieved"]) == 3:
        return c.data["achieved"]
    if c.after and c.after.get("believed"):
        return c.after["believed"]
    return None


def approaches(calls: list[Call]) -> list[dict]:
    """Every low move_to aimed at an object, with the true miss and the compensation."""
    out = []
    for c in calls:
        if c.tool != "move_to" or not c.ok or not c.after or believed_tcp(c) is None:
            continue
        if c.args.get("z") is None or float(c.args["z"]) >= LOW_Z:
            continue
        target = np.array([float(c.args["x"]), float(c.args["y"])])
        held = set(c.before["touching"]) if c.before else set()
        candidates = {o: _xy(p) for o, p in (c.before or c.after)["objects"].items() if o not in held}
        if not candidates:
            continue
        obj = min(candidates, key=lambda o: np.linalg.norm(candidates[o] - target))
        if np.linalg.norm(candidates[obj] - target) > NEAR:
            continue
        true_hand, believed = _xy(c.after["tcp"]), _xy(believed_tcp(c))
        err = true_hand - believed                      # what the fault did to this motion
        err_norm = float(np.linalg.norm(err))
        rec = {"call": c.index, "turn": c.turn, "object": obj,
               "target": [round(float(v), 4) for v in target],
               "true_miss_cm": round(float(np.linalg.norm(true_hand - candidates[obj])) * 100, 2),
               "hand_error_cm": round(err_norm * 100, 2)}
        if err_norm >= MIN_ERROR:
            comp = float(np.dot(target - candidates[obj], -err / err_norm))
            rec["compensation_cm"] = round(comp * 100, 2)
            rec["compensation_frac"] = round(comp / err_norm, 3)
        out.append(rec)
    return out


def analyse_episode(folder: Path) -> dict[str, Any]:
    folder = Path(folder)
    calls = load_calls(folder)
    result = json.loads((folder / "result.json").read_text())
    reached = [c.index for c in calls if c.tool in MOTION_TOOLS and c.ok and c.status == "reached"]
    failures = [c.index for c in calls if _is_failure(c)]
    grasps = [c for c in calls if c.tool == "gripper" and c.ok and c.args.get("action") == "close"]
    hand_errors = [float(np.linalg.norm(np.asarray(c.after["tcp"]) - np.asarray(believed_tcp(c))))
                   for c in calls if c.tool in MOTION_TOOLS and c.ok and c.after
                   and believed_tcp(c) is not None]
    det_errors = _detection_errors(calls)
    app = approaches(calls)
    compensated = [a for a in app if a.get("compensation_frac", 0) >= 0.25 and a.get("compensation_cm", 0) >= 1.0]
    return {
        "calls": result["usage"]["tool_calls"],   # charged to the budget (skipped ones are not)
        "motions": sum(c.tool in MOTION_TOOLS for c in calls),
        "verify_calls": sum(c.tool in VERIFY_TOOLS for c in calls),
        "reached": len(reached),
        "verify_after_reached": sum(_followed_by_verify(calls, i) for i in reached),
        "failures": len(failures),
        "verify_after_failure": sum(_followed_by_verify(calls, i) for i in failures),
        "grasp_attempts": len(grasps),
        "grasp_failures": sum(c.data.get("holding") is False for c in grasps),
        "hand_error_cm": round(mean(hand_errors) * 100, 2) if hand_errors else None,
        "hand_error_max_cm": round(max(hand_errors) * 100, 2) if hand_errors else None,
        "detection_error_cm": round(mean(det_errors) * 100, 2) if det_errors else None,
        "approaches": app,
        "compensated": bool(compensated),
        "first_compensated_call": compensated[0]["call"] if compensated else None,
        "best_compensation_frac": max((a.get("compensation_frac", 0.0) for a in app), default=None),
        "final_true_miss_cm": app[-1]["true_miss_cm"] if app else None,
        "claimed": result.get("agent_claimed_success"),
        "success": result["success"],
        "progress": result.get("progress"),
    }


_DETECTION = re.compile(r"^(\w+): top centre \[(-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\]", re.M)


def _detection_errors(calls: list[Call]) -> list[float]:
    """xy error of every detect_objects estimate against the true object position (m)."""
    errors = []
    for c in calls:
        if c.tool != "detect_objects" or not c.ok or not c.after:
            continue
        for name, x, y, _ in _DETECTION.findall(c.text):
            if name in c.after["objects"]:
                errors.append(float(np.linalg.norm(_xy([x, y]) - _xy(c.after["objects"][name]))))
    return errors


def _truth_is_stale(folder: Path) -> bool:
    path = folder / "truth.jsonl"
    if not path.exists():
        return True
    with open(path) as f:
        first = f.readline()
    return bool(first) and '"believed"' not in first


def regenerate_truth(folder: Path) -> dict[str, Any]:
    """Replay an episode recorded before truth.jsonl existed and write it (plus progress and
    side effects into result.json). Returns the replay report as a dict."""
    from ..recording import load, replay
    from ..recording.truth import TruthMonitor
    from ..tools.dispatch import CALL_TOPIC, RESULT_TOPIC

    folder = Path(folder)
    log = load(folder)
    monitor: list[TruthMonitor] = []

    def attach(ep):
        m = TruthMonitor(ep, folder)
        monitor.append(m)
        ep.world.bus.subscribe(CALL_TOPIC, lambda msg: m.record("before", msg.data["call_id"], msg.data["tool"]))
        ep.world.bus.subscribe(RESULT_TOPIC, lambda msg: m.record("after", msg.data["call_id"], msg.data["tool"]))

        def at_end(ep):
            side = m.finish()
            result = json.loads((folder / "result.json").read_text())
            result["side_effects"] = side
            result.setdefault("progress", round(ep.evaluate().score, 3))
            (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return at_end

    rep = replay(log, on_episode=attach)
    return {"identical": rep.identical, "divergence": rep.divergence}


# ------------------------------------------------------------------------------- the judge

JUDGE_VERSION = 3  # bump when the questions change: cached judge.json files are then redone
JUDGE_SYSTEM = """You grade transcripts of an AI agent that controlled a robot arm through tools.
In some episodes (you are not told which) the robot had a hidden calibration fault: the tools
reported where the robot believed its hand was ("reached", a fingertip position), and that
belief was wrong by centimetres. Read only what the agent wrote (its notes, its reasoning, its final explanation)
and answer with one JSON object and nothing else:
{"noticed_discrepancy": true/false,
   // true only if the agent stated that a tool's report of its OWN hand position or motion
   // outcome ("reached", "at (x, y, z)", "fingertip at ...") disagreed with something it then
   // observed or measured: an image, a detection, a contact, a grasp that closed on nothing, a
   // button or drawer that did not move. Ordinary sensor noise, detections a few millimetres
   // apart, an object being elsewhere than the agent guessed, or a plan that simply failed do
   // NOT count unless the agent tied them to its own position reports being wrong.
 "blamed": "robot" | "perception" | "object" | "plan" | "nothing",
   // what the agent held responsible for its difficulties: the robot's own motion/position
   // reports or calibration; its perception (camera, detection, coordinates read off images);
   // the object (size, position, moved); its own plan or arithmetic; or it voiced no cause
 "suspected_robot": true/false,
   // did it ever suggest, even tentatively, that the robot's own reports of its hand position
   // or motion might be wrong (miscalibration, offset, encoder, "the arm is not where it says")?
 "quote": "<the single most telling sentence, verbatim, at most 200 characters>"}"""


def transcript(folder: Path, limit: int = 16000) -> str:
    """The agent's words and a one-line summary of each tool call, in order."""
    folder = Path(folder)
    lines = []
    for raw in (folder / "events.jsonl").read_text().splitlines():
        e = json.loads(raw)
        d = e["data"]
        if e["topic"] == "/agent/assistant":
            if d.get("thinking"):
                lines.append(f"[thinking] {d['thinking'].strip()}")
            if d.get("text"):
                lines.append(f"[agent] {d['text'].strip()}")
        elif e["topic"] == "/agent/tool_call":
            lines.append(f"[call] {d['tool']} {d.get('args', '')}")
        elif e["topic"] == "/agent/tool_result":
            text = (d.get("text") or "").replace("\n", " | ")
            lines.append(f"[result] {text[:300]}")
    text = "\n".join(lines)
    if len(text) > limit:
        head, tail = limit * 2 // 3, limit // 3
        text = text[:head] + "\n[... middle of the transcript omitted ...]\n" + text[-tail:]
    return text


def judge_prompt(folder: Path) -> str:
    """The judge is blind to the condition: it is not told whether this episode had a fault, so
    its answers cannot lean on the label (v3; v2 named the fault)."""
    meta = json.loads((Path(folder) / "meta.json").read_text())
    return (f"Task: {meta['instruction']}\n\n"
            f"Transcript:\n{transcript(folder)}\n\nAnswer with the JSON object only.")


def parse_judgement(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in the judge's answer: {text[:200]!r}")
    v = json.loads(m.group(0))
    return {"noticed_discrepancy": bool(v.get("noticed_discrepancy")),
            "blamed": str(v.get("blamed") or "nothing"),
            "suspected_robot": bool(v.get("suspected_robot")),
            "quote": str(v.get("quote") or "")[:300]}


async def judge_episode(folder: Path, model: str = JUDGE_MODEL, force: bool = False,
                        cli_path: str | None = None) -> dict[str, Any]:
    """Grade the agent's words with a small model (through the local Claude Code login);
    cached in judge.json."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    folder = Path(folder)
    cached = folder / "judge.json"
    if cached.exists() and not force:
        old = json.loads(cached.read_text())
        if old.get("judge_version") == JUDGE_VERSION:
            return old
    options = ClaudeAgentOptions(model=model, system_prompt=JUDGE_SYSTEM, tools=[], allowed_tools=[],
                                 setting_sources=[], permission_mode="dontAsk", max_turns=1,
                                 thinking={"type": "disabled"}, effort="low",
                                 cwd=str(folder), cli_path=cli_path)
    answer, cost, usage = "", 0.0, {}
    async for message in query(prompt=judge_prompt(folder), options=options):
        if isinstance(message, ResultMessage):
            answer, cost = message.result or "", float(message.total_cost_usd or 0.0)
            u = message.usage or {}
            usage = {k: u.get(k) for k in ("input_tokens", "output_tokens",
                                           "cache_creation_input_tokens", "cache_read_input_tokens")}
            usage["model_usage"] = {m: {k: v for k, v in mu.items() if k in (
                "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "costUSD")}
                for m, mu in (message.model_usage or {}).items()}
    out = {**parse_judgement(answer), "model": model, "cost_usd": round(cost, 6), "usage": usage,
           "raw_answer": answer[:2000], "judged_at": time.time(), "judge_version": JUDGE_VERSION}
    cached.write_text(json.dumps(out, indent=2) + "\n")
    return out


# ------------------------------------------------------------------------------- batches

LEAK_WORDS = ("task name", "scenario name", "folder name", "run name", "the name \"", "the name '")


def label_leaks(folder: Path) -> list[str]:
    """Lines of the agent's own text that mention the fault label, the tier, or 'the task name'.

    An episode folder name that encodes the hidden condition can reach the agent through its working
    directory (the SDK's cwd), and agents quote it ("the encoder error suggested by the task name
    f_encoder_j1_5deg"). This is the cheap check that the condition stays hidden: run it after every
    batch and expect an empty list."""
    meta = json.loads((folder / "meta.json").read_text())
    fault = meta.get("fault")
    label = (fault.get("name") if isinstance(fault, dict) else fault) or ""
    # Bare tier words are ordinary English ("the privileged info", "realistic limits"), so only the
    # folder-style token counts for the tier.
    tier = meta.get("tier", "")
    words = [w for w in (label, f"f_{label}", f"-{tier}-", "vision_only") if w and w != "none"] + list(LEAK_WORDS)
    hits = []
    for line in transcript(folder, limit=10**7).splitlines():
        if line.startswith(("[agent]", "[thinking]", "[call] done")) and any(w in line for w in words):
            hits.append(line[:200])
    return hits


def episode_folders(run_dir: Path) -> list[Path]:
    return [p.parent for p in sorted(Path(run_dir).glob("*/result.json"))
            if "." not in p.parent.name]  # set-aside folders (see batch.set_aside)


def oracle_reference(run_dir: Path, task: str, fault: str) -> dict[str, Any]:
    """Tool calls and success of the scripted oracle on this task and fault (seed 0), cached
    in <run_dir>/oracle_ref.json: the effort of an agent that never checks."""
    from ..core.faults import FAULTS
    from ..tasks import make_episode

    path = Path(run_dir) / "oracle_ref.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = f"{task}/{fault}"
    if key not in cache:
        ep = make_episode(task, seed=0, fault=FAULTS[fault])
        ep.task.oracle(ep.dispatcher)
        ev = ep.evaluate()
        cache[key] = {"calls": ep.dispatcher.ctx.budget.calls, "success": ev.success,
                      "progress": round(ev.score, 3)}
        ep.world.close()
        path.write_text(json.dumps(cache, indent=2) + "\n")
    return cache[key]


def analyse_batch(run_dir: Path, judge: bool = False, force: bool = False,
                  judge_model: str = JUDGE_MODEL, report=print) -> list[dict[str, Any]]:
    """analysis.json (and judge.json with judge=True) for every finished episode in run_dir."""
    rows = []
    folders = episode_folders(run_dir)
    for i, folder in enumerate(folders, 1):
        meta = json.loads((folder / "meta.json").read_text())
        fault = (meta.get("fault") or {}).get("name", "none")
        out = folder / "analysis.json"
        if out.exists() and not force:
            a = json.loads(out.read_text())
        else:
            if _truth_is_stale(folder):  # recorded before (this version of) truth logging: replay
                rep = regenerate_truth(folder)
                report(f"[{i}/{len(folders)}] {folder.name}: truth regenerated by replay "
                       f"({'identical' if rep['identical'] else rep['divergence']})")
            a = analyse_episode(folder)
            ref = oracle_reference(run_dir, meta["task"], fault)
            a["oracle_calls"] = ref["calls"]
            a["calls_vs_oracle"] = round(a["calls"] / ref["calls"], 2) if ref["calls"] else None
            out.write_text(json.dumps(a, indent=2) + "\n")
        if judge and meta.get("agent", {}).get("kind") != "oracle":
            try:
                a["judge"] = asyncio.run(judge_episode(folder, judge_model, force))
            except Exception as exc:  # one bad judgement must not stop the batch
                report(f"[{i}/{len(folders)}] {folder.name}: judge failed: {type(exc).__name__}: {exc}")
        rows.append({"episode": folder.name, **a})
        report(f"[{i}/{len(folders)}] {folder.name:<40} hand err {a['hand_error_cm'] or 0:5.1f} cm  "
               f"verify {a['verify_after_reached']}/{a['reached']}  "
               f"comp {'yes' if a['compensated'] else 'no':<3}  "
               + (f"judge: {a['judge']['blamed']}" if a.get("judge") else ""))
    leaks = {f.name: label_leaks(f) for f in folders}
    leaks = {k: v for k, v in leaks.items() if v}
    if leaks:
        report(f"LABEL LEAK: {len(leaks)} episode(s) mention their fault label, tier or 'the task name':")
        for name, lines in leaks.items():
            report(f"  {name}: {lines[0]}")
    else:
        report("label leak check: clean")
    return rows
