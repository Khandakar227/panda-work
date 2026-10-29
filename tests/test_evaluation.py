"""Batch runs and reports, exercised with the free oracle agent."""
import json

import pytest

from pandawork.evaluation import grid, load_rows, run_grid, wilson, write_report
from pandawork.evaluation.batch import pending
from pandawork.tiers import Tier


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.722, abs=0.01)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.53, abs=0.02)


def test_grid_is_tier_major():
    jobs = grid(["reach", "pick_place"], [Tier.REALISTIC, Tier.PRIVILEGED], [0, 1])
    assert [j.tier for j in jobs[:4]] == [Tier.REALISTIC] * 4
    assert jobs[0].name == "reach-realistic-s0" and jobs[2].name == "pick_place-realistic-s0"


def test_batch_runs_resumes_and_reports(tmp_path):
    jobs = grid(["reach", "press_button"], [Tier.PRIVILEGED], [0, 1])
    lines = []
    assert len(run_grid(tmp_path, jobs, "oracle", report=lines.append)) == 4
    assert run_grid(tmp_path, jobs, "oracle", report=lines.append) == []  # all finished: skipped
    assert "4 of 4 episodes already finished" in lines[-1]

    rows = load_rows(tmp_path)
    assert len(rows) == 4 and all(r.success for r in rows)
    press = next(r for r in rows if r.task == "press_button")
    assert press.metric >= 8  # travel of the named button, in mm
    path, text = write_report(tmp_path)
    assert "| reach / privileged | 2 | 2/2 (100%) |" in text
    summary = json.loads((tmp_path / "report.json").read_text())
    assert summary["all"][0]["successes"] == 4


def test_interrupted_episode_is_set_aside_and_rerun(tmp_path):
    jobs = grid(["reach"], [Tier.PRIVILEGED], [0])
    (tmp_path / jobs[0].name).mkdir()  # a folder without result.json: interrupted
    assert [j.name for j in pending(tmp_path, jobs)] == [jobs[0].name]
    assert len(list(tmp_path.glob("*.incomplete-*"))) == 1


def test_spending_cap_stops_the_batch(tmp_path):
    lines = []
    assert run_grid(tmp_path, grid(["reach"], [Tier.PRIVILEGED], [0]), "oracle",
                    max_total_usd=0.0, report=lines.append) == []
    assert "stopping" in lines[-1]


def test_folders_set_aside_by_hand_are_ignored_everywhere(tmp_path):
    """Renaming an episode folder to <name>.<anything> takes it out of resume, cost and reports,
    so a bad episode can be re-run without deleting its data."""
    import json
    from pandawork.evaluation.batch import grid, pending, run_oracle, spent_usd
    from pandawork.evaluation.report import load_rows
    from pandawork.tiers import Tier
    jobs = grid(["reach"], [Tier.PRIVILEGED], [0, 1])
    for job in jobs:
        run_oracle(job, tmp_path / job.name)
    assert len(list(pending(tmp_path, jobs))) == 0 and len(load_rows(tmp_path)) == 2
    (tmp_path / jobs[0].name).rename(tmp_path / f"{jobs[0].name}.bad")
    assert [j.name for j in pending(tmp_path, jobs)] == [jobs[0].name]
    assert len(load_rows(tmp_path)) == 1
    assert spent_usd(tmp_path) == json.loads((tmp_path / jobs[1].name / "result.json").read_text())["usage"]["cost_usd"]
