"""Batch plumbing: job names and grid order, an oracle batch's report, and pausing a batch when
the subscription's usage limit is hit."""
import json
import time

from pandawork.evaluation import batch, load_rows, write_report
from pandawork.evaluation.batch import Job, grid, run_grid
from pandawork.tiers import Tier


def test_job_names_and_grid_order_are_tier_major():
    assert Job("reach", Tier.PRIVILEGED, 3).name == "reach-privileged-s3"
    names = [j.name for j in grid(["reach", "pick_place"], [Tier.PRIVILEGED, Tier.REALISTIC], [0, 1])]
    assert names == ["reach-privileged-s0", "reach-privileged-s1", "pick_place-privileged-s0",
                     "pick_place-privileged-s1", "reach-realistic-s0", "reach-realistic-s1",
                     "pick_place-realistic-s0", "pick_place-realistic-s1"]


def test_oracle_batch_writes_episodes_and_a_report(tmp_path):
    jobs = grid(["reach"], [Tier.PRIVILEGED], [0, 1])
    run_grid(tmp_path, jobs, agent="oracle", report=lambda s: None)
    rows = load_rows(tmp_path)
    assert [(r.task, r.tier, r.seed, r.success) for r in rows] == [("reach", "privileged", 0, True),
                                                                   ("reach", "privileged", 1, True)]
    _, text = write_report(tmp_path)
    assert "## By tier" in text and "reach / privileged" in text
    assert json.loads((tmp_path / "batch.json").read_text())["jobs"] == ["reach-privileged-s0", "reach-privileged-s1"]
    assert (tmp_path / "episodes.csv").read_text().splitlines()[1].startswith("reach-privileged-s0,reach,privileged,0,True")


def test_batch_pauses_and_retries_when_the_usage_limit_is_hit(tmp_path, monkeypatch):
    from pandawork.agent.session import RateLimited
    calls, slept = [], []

    async def fake_run(task, tier, seed, folder, config, limits=None, **_):
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
