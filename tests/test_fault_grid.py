"""Batch plumbing for fault experiments: faults and the warning as grid axes, reports by fault, and
pausing a batch when the subscription's usage limit is hit."""
import json
import time

from pandawork.agent.prompt import system_prompt
from pandawork.core.faults import NO_FAULT
from pandawork.evaluation import batch, load_rows, write_report
from pandawork.evaluation.batch import Job, grid, run_grid
from pandawork.tiers import Tier


def test_job_names_and_explicit_exclusions():
    assert Job("reach", Tier.PRIVILEGED, 3).name == "reach-privileged-s3"  # as before faults
    assert Job("reach", Tier.REALISTIC, 3, "stiff_j4", True).name == "reach-realistic-f_stiff_j4-warn-s3"
    jobs = grid(["press_key", "reach"], [Tier.REALISTIC], [0], faults=("none", "stiff_j4"))
    assert [j.name for j in jobs] == ["press_key-realistic-s0", "reach-realistic-s0",
                                      "press_key-realistic-f_stiff_j4-s0", "reach-realistic-f_stiff_j4-s0"]
    jobs = grid(["press_key", "reach"], [Tier.REALISTIC], [0], faults=("none", "stiff_j4"),
                exclude={("press_key", "stiff_j4")})
    assert [j.name for j in jobs] == ["press_key-realistic-s0", "reach-realistic-s0", "reach-realistic-f_stiff_j4-s0"]


def test_exclude_option_parses_task_fault_pairs_and_rejects_bad_shapes():
    import argparse, importlib.util, pathlib
    import pytest
    spec = importlib.util.spec_from_file_location("batch_cli", pathlib.Path(__file__).parents[1] / "scripts" / "batch.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    assert cli.parse_exclude(["stack:stiff_j4", "open_drawer:stiff_j4"]) == {("stack", "stiff_j4"), ("open_drawer", "stiff_j4")}
    for bad in (["stack"], ["stack:nope"], ["nope:stiff_j4"]):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_exclude(bad)


def test_oracle_batch_records_the_fault_and_the_report_groups_by_it(tmp_path):
    jobs = grid(["reach"], [Tier.PRIVILEGED], [0, 1], faults=("none", "encoder_j1_5deg"))
    run_grid(tmp_path, jobs, agent="oracle", report=lambda s: None)
    rows = load_rows(tmp_path)
    assert sorted(r.fault for r in rows) == ["encoder_j1_5deg"] * 2 + ["none"] * 2
    by_fault = {r.fault: r.success for r in rows}
    assert by_fault["none"] and not by_fault["encoder_j1_5deg"]  # a trusting operator misses by 4 cm
    meta = json.loads((tmp_path / "reach-privileged-f_encoder_j1_5deg-s0" / "meta.json").read_text())
    assert meta["fault"]["name"] == "encoder_j1_5deg"
    _, text = write_report(tmp_path)
    assert "## By fault" in text and "encoder_j1_5deg" in text
    assert json.loads((tmp_path / "batch.json").read_text())["faults"] == ["none", "encoder_j1_5deg"]


def test_warned_prompt_mentions_miscalibration():
    assert "miscalibrated" in system_prompt(Tier.REALISTIC, warn=True)
    assert "miscalibrated" not in system_prompt(Tier.REALISTIC)


def test_batch_pauses_and_retries_when_the_usage_limit_is_hit(tmp_path, monkeypatch):
    from pandawork.agent.session import RateLimited
    calls, slept = [], []

    async def fake_run(task, tier, seed, folder, config, limits=None, fault=NO_FAULT, **_):
        calls.append(seed)
        if len(calls) == 1:  # first attempt: the limit hits mid-episode (a result was still written)
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps({"usage": {"cost_usd": 0.01}}))
            raise RateLimited({"resetsAt": time.time() + 100, "rateLimitType": "five_hour"})
        return batch.run_oracle(Job(task, tier, seed), folder)

    monkeypatch.setattr(batch, "run_agent_episode", fake_run)
    done = run_grid(tmp_path, grid(["reach"], [Tier.PRIVILEGED], [0]), sleep=slept.append,
                    report=lambda s: None)
    assert calls == [0, 0] and len(done) == 1
    assert 100 <= slept[0] <= 200  # until the reset, plus a margin
    assert any(".incomplete-ratelimit" in p.name for p in tmp_path.iterdir())
    assert (tmp_path / "reach-privileged-s0" / "result.json").exists()


def test_warning_arms_are_three_levels_and_old_booleans_still_work():
    from pandawork.agent.prompt import WARNINGS, system_prompt, warn_level
    base = system_prompt(Tier.REALISTIC)
    mild, full = system_prompt(Tier.REALISTIC, "mild"), system_prompt(Tier.REALISTIC, "full")
    assert base not in (mild, full) and "not ideal" in mild and "Verify" not in mild
    assert "miscalibrated" in full and "Verify" in full and WARNINGS["none"] == ""
    assert warn_level(False) == "none" and warn_level(True) == "full" and warn_level("mild") == "mild"
    assert Job("reach", Tier.REALISTIC, 0, "none", "mild").name == "reach-realistic-warnmild-s0"
    assert Job("reach", Tier.REALISTIC, 0, "none", True).name == "reach-realistic-warn-s0"
    names = [j.name for j in grid(["reach"], [Tier.REALISTIC], [0], warn=("none", "mild", "full"))]
    assert names == ["reach-realistic-s0", "reach-realistic-warnmild-s0", "reach-realistic-warn-s0"]
