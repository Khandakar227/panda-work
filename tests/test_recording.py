"""Episode logs: structure, exact replay, tamper detection, costs, video."""
import json
import shutil

import numpy as np
import pytest

from pandawork.recording import (
    CostLedger, EpisodeRecorder, Usage, cost_usd, load, render_video, replay, summarise,
)
from pandawork.tasks import make_episode
from pandawork.tasks.base import call
from pandawork.tiers import Tier


def record(tmp_path, task="reach", seed=0, tier=Tier.PRIVILEGED, script=None, name="ep"):
    ep = make_episode(task, tier=tier, seed=seed)
    rec = EpisodeRecorder(ep, tmp_path / name, agent={"kind": "test"})
    (script or (lambda e: e.task.oracle(e.dispatcher)))(ep)
    return ep, rec.finish()


def messy(ep):
    """An episode with failures: unreachable target, skipped action, bad arguments, a look."""
    call(ep.dispatcher, ("move_to", dict(x=1.3, y=0, z=0.3)), ("gripper", {"action": "close"}))
    call(ep.dispatcher, ("move_to", dict(x="far", y=0, z=0.3)))
    call(ep.dispatcher, ("look", {"camera": "side"}), ("move_to", dict(x=0.45, y=0.1, z=0.2)))


def test_log_structure(tmp_path):
    ep, d = record(tmp_path, "pick_place", seed=1)
    meta, result = json.loads((d / "meta.json").read_text()), json.loads((d / "result.json").read_text())
    assert meta["task"] == "pick_place" and meta["seed"] == 1 and meta["instruction"]
    assert meta["budget_limits"]["max_calls"] == 40
    assert result["success"] and result["usage"]["turns"] == 4
    log = load(d)
    turns = log.turns()
    assert [len(t) for t in turns] == [2, 3, 2, 2]
    assert set(log.snapshots) == {"start", "turn0", "turn1", "turn2", "turn3", "end"}
    # every physics step since recording started has its /joints reading in the stream file
    steps = ep.world.clock.step - meta["start_step"]
    assert len(log.streams["joints.sim_time"]) == steps
    assert log.streams["wrist_ft.force"].shape == (steps, 3)


def test_images_are_stored_once(tmp_path):
    def look_twice(ep):
        call(ep.dispatcher, ("look", {"camera": "front"}))
        call(ep.dispatcher, ("look", {"camera": "front"}))
    _, d = record(tmp_path, script=look_twice)
    # one camera frame + one rendered tool image, each written once despite two looks
    assert len(list((d / "blobs").iterdir())) == 2
    log = load(d)
    event = log.events_on("/agent/tool_result")[0]
    assert log.blob(event["data"]["image_0"]).shape == (240, 320, 3)


@pytest.mark.parametrize("task", ["reach", "open_drawer"])
def test_oracle_episode_replays_exactly(tmp_path, task):
    _, d = record(tmp_path, task, seed=3)
    report = replay(load(d))
    assert report.identical, report
    assert report.max_state_diff == 0.0 and report.final_state_identical


def test_noisy_episode_with_failures_replays_exactly(tmp_path):
    _, d = record(tmp_path, "reach", seed=5, tier=Tier.REALISTIC, script=messy)
    log = load(d)
    assert any(e["data"]["error_code"] == "skipped" for e in log.events_on("/agent/tool_result"))
    report = replay(log)
    assert report.identical, report


def test_tampered_log_is_detected(tmp_path):
    _, d = record(tmp_path, "reach", seed=0, script=messy)
    copy = tmp_path / "tampered"
    shutil.copytree(d, copy)
    events = [json.loads(l) for l in (copy / "events.jsonl").read_text().splitlines()]
    tampered = 0
    for e in events:  # nudge the reachable move_to target by 1 cm
        if e["topic"] == "/agent/tool_call" and e["data"]["args"] == '{"x":0.45,"y":0.1,"z":0.2}':
            e["data"]["args"] = '{"x":0.46,"y":0.1,"z":0.2}'
            tampered += 1
    assert tampered == 1
    (copy / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report = replay(load(copy))
    assert not report.identical and report.divergence


def test_recorder_never_overwrites(tmp_path):
    record(tmp_path)
    with pytest.raises(FileExistsError):
        EpisodeRecorder(make_episode("reach"), tmp_path / "ep")


def test_summary_reads_like_a_timeline(tmp_path):
    _, d = record(tmp_path, "press_button", seed=1)
    text = summarise(load(d))
    assert "verdict: SUCCESS" in text and "move_to(" in text and "turn  2" in text


def test_cost_arithmetic():
    usage = Usage(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=4000,
                  cache_read_input_tokens=20000)
    expected = (1000 * 4 + 500 * 20 + 4000 * 4 * 1.25 + 20000 * 0.20) / 1e6
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(expected)
    assert cost_usd("claude-sonnet-5-5", usage) < cost_usd("claude-opus-5-5", usage)
    with pytest.raises(KeyError):
        cost_usd("gpt-9", usage)


def test_ledger_charges_budget_and_is_logged(tmp_path):
    def spend(ep):
        ledger = CostLedger(ep.world.bus, ep.dispatcher.ctx.budget)
        ledger.record("claude-sonnet-5-5", Usage(4000, 300, 4000, 0), request_id="req_1")
        ledger.record("claude-sonnet-5-5", Usage(600, 300, 0, 8000), request_id="req_2")
        assert ledger.cache_hit_rate == pytest.approx(8000 / 16600)
        ep.task.oracle(ep.dispatcher)
    _, d = record(tmp_path, script=spend)
    log = load(d)
    usage = log.events_on("/agent/llm_usage")
    assert [u["data"]["request_id"] for u in usage] == ["req_1", "req_2"]
    total = sum(u["data"]["cost_usd"] for u in usage)
    assert log.result["usage"]["cost_usd"] == pytest.approx(total, abs=1e-6)
    assert log.result["usage"]["tokens"] == 4000 + 300 + 4000 + 600 + 300 + 8000


def test_video(tmp_path, cameras):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    _, d = record(tmp_path, "reach", seed=1)
    path, report = render_video(load(d), camera="side")
    assert path.exists() and path.stat().st_size > 10_000 and report.identical
