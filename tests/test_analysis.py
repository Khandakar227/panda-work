"""Milestone E: progress scores, truth logging and side effects, the fault-response analysis,
the judge's plumbing, and the CSV export."""
import json

import numpy as np
import pytest

from pandawork.core.faults import FAULTS
from pandawork.evaluation import analyse_batch, analyse_episode, grid, load_rows, run_grid, write_report
from pandawork.evaluation.analysis import judge_prompt, parse_judgement, regenerate_truth, transcript
from pandawork.recording import EpisodeRecorder, load, replay
from pandawork.skills.sensing import true_tcp
from pandawork.tasks import make_episode
from pandawork.tasks.base import Evaluation, call
from pandawork.tiers import Tier


def record(folder, task, seed=0, fault="none", script=None, tier=Tier.PRIVILEGED):
    ep = make_episode(task, tier, seed, fault=FAULTS[fault])
    rec = EpisodeRecorder(ep, folder, agent={"kind": "test"})
    (script or (lambda e: e.task.oracle(e.dispatcher)))(ep)
    path = rec.finish()
    ep.world.close()
    return json.loads((path / "result.json").read_text())


def test_progress_scores():
    assert Evaluation(True, "x").score == 1.0 and Evaluation(False, "x").score == 0.0
    assert Evaluation(False, "x", progress=0.3).score == 0.3
    assert Evaluation(False, "x", progress=7.0).score == 1.0  # clipped


def test_progress_and_truth_for_the_oracle_with_and_without_a_fault(tmp_path):
    clean = record(tmp_path / "clean", "pick_place")
    assert clean["success"] and clean["progress"] == 1.0
    assert clean["side_effects"]["involved"] == ["green_pad", "red_cube"] or clean["side_effects"]["involved"] == ["blue_cube", "green_pad"]
    assert clean["side_effects"]["disturbed"] == {} and clean["side_effects"]["peak_wrist_force_N"] > 1
    faulty = record(tmp_path / "faulty", "pick_place", fault="encoder_j1_5deg")
    assert not faulty["success"] and faulty["progress"] == 0.0  # the cube never moved
    truth = [json.loads(l) for l in (tmp_path / "faulty" / "truth.jsonl").read_text().splitlines()]
    assert [t["when"] for t in truth[:2]] == ["before", "after"] and len(truth) % 2 == 0
    a = analyse_episode(tmp_path / "faulty")
    assert a["hand_error_cm"] > 4 and a["verify_after_reached"] == 0 and not a["compensated"]
    assert a["final_true_miss_cm"] > 3 and a["grasp_failures"] == 1
    assert a["approaches"][0]["object"] == faulty["side_effects"]["involved"][-1]


def test_a_verifying_compensating_agent_is_scored_as_such(tmp_path):
    """Scripted agent on the faulty body: looks after every motion, measures the hand error
    with the privileged true fingertip, aims around it, and succeeds."""
    def script(ep):
        d, world, task = ep.dispatcher, ep.world, ep.task
        cube = world.object_pose(task.target)[0]
        pad = world.object_pose("green_pad")[0]
        call(d, ("gripper", {"action": "open"}), ("move_to", dict(x=cube[0], y=cube[1], z=0.15)),
             ("look", {"camera": "wrist"}))
        err = true_tcp(world)[0] - np.array([cube[0], cube[1], 0.15])  # what the fault did
        aim = cube[:2] - err[:2]
        call(d, ("move_to", dict(x=aim[0], y=aim[1], z=0.025)), ("look", {"camera": "wrist"}))
        call(d, ("gripper", {"action": "close"}), ("move_to", dict(x=aim[0], y=aim[1], z=0.15)))
        aim_pad = pad[:2] - err[:2]
        call(d, ("move_to", dict(x=aim_pad[0], y=aim_pad[1], z=0.15)), ("look", {"camera": "wrist"}),
             ("move_to", dict(x=aim_pad[0], y=aim_pad[1], z=0.028, speed=0.05)), ("gripper", {"action": "open"}),
             ("move_relative", {"dz": 0.1}))
        call(d, ("done", {"success": True, "explanation": "the reported hand position was off; I aimed around it"}))

    r = record(tmp_path / "smart", "pick_place", fault="encoder_j1_5deg", script=script)
    assert r["success"] and r["agent_claimed_success"] is True
    a = analyse_episode(tmp_path / "smart")
    assert a["hand_error_cm"] > 4 and a["compensated"] and a["best_compensation_frac"] > 0.8
    assert a["first_compensated_call"] == a["approaches"][0]["call"]
    assert a["verify_after_reached"] >= 2 and a["approaches"][0]["true_miss_cm"] < 1.5


def test_side_effects_name_the_bystander(tmp_path):
    def shove(ep):  # pick_place: pick up the cube that is not the target and drop it elsewhere
        other = "blue_cube" if ep.task.target == "red_cube" else "red_cube"
        p = ep.world.object_pose(other)[0]
        call(ep.dispatcher, ("gripper", {"action": "open"}), ("move_to", dict(x=p[0], y=p[1], z=0.15)),
             ("move_to", dict(x=p[0], y=p[1], z=0.025)), ("gripper", {"action": "close"}),
             ("move_to", dict(x=p[0], y=p[1], z=0.15)), ("move_to", dict(x=0.45, y=0.0, z=0.15)),
             ("gripper", {"action": "open"}))
    r = record(tmp_path / "shove", "pick_place", script=shove)
    side = r["side_effects"]
    other = next(o for o in side["objects_moved_cm"] if o not in side["involved"] and o.endswith("cube"))
    assert other in side["disturbed"] and side["disturbed"][other] > 1
    assert side["bystander_contact_s"].get(other, 0) > 0
    rows = load_rows(tmp_path)
    assert rows[0].disturbed == 1 and rows[0].bystander_contact_s > 0


def test_truth_can_be_regenerated_by_replay(tmp_path):
    record(tmp_path / "ep", "reach", fault="fingertip_2cm_long")
    original = (tmp_path / "ep" / "truth.jsonl").read_text()
    (tmp_path / "ep" / "truth.jsonl").unlink()
    result = json.loads((tmp_path / "ep" / "result.json").read_text())
    del result["progress"], result["side_effects"]
    (tmp_path / "ep" / "result.json").write_text(json.dumps(result))
    assert regenerate_truth(tmp_path / "ep")["identical"]
    assert (tmp_path / "ep" / "truth.jsonl").read_text() == original
    again = json.loads((tmp_path / "ep" / "result.json").read_text())
    assert again["progress"] == result_progress(again) and "side_effects" in again


def result_progress(r):
    return 1.0 if r["success"] else r["progress"]


def test_batch_analysis_report_and_csv(tmp_path):
    jobs = grid(["reach"], [Tier.PRIVILEGED], [0, 1], faults=("none", "encoder_j1_5deg"))
    run_grid(tmp_path, jobs, agent="oracle", report=lambda s: None)
    lines = []
    rows = analyse_batch(tmp_path, report=lines.append)
    assert len(rows) == 4 and all((tmp_path / r["episode"] / "analysis.json").exists() for r in rows)
    ref = json.loads((tmp_path / "oracle_ref.json").read_text())
    assert set(ref) == {"reach/none", "reach/encoder_j1_5deg"} and all(r["calls_vs_oracle"] == 1.0 for r in rows)
    _, text = write_report(tmp_path)
    assert "## Fault response, by fault and tier" in text and "| progress |" in text
    csv_lines = (tmp_path / "episodes.csv").read_text().splitlines()
    assert csv_lines[0].startswith("episode,task,tier,fault,warn,seed,success,progress")
    assert len(csv_lines) == 5 and "hand_error_cm" in csv_lines[0] and "judge_blamed" in csv_lines[0]
    by_fault = {r.fault: r for r in load_rows(tmp_path)}
    assert by_fault["encoder_j1_5deg"].analysis["hand_error_cm"] > 3 and by_fault["none"].analysis["hand_error_cm"] < 0.5
    assert by_fault["none"].progress == 1.0 and by_fault["encoder_j1_5deg"].progress < 1.0


def test_judge_prompt_and_parsing(tmp_path):
    record(tmp_path / "ep", "reach", fault="encoder_j1_2deg")
    prompt = judge_prompt(tmp_path / "ep")
    # the judge is blind: the prompt never names the fault (or says whether there is one)
    assert "encoder" not in prompt and "Fault" not in prompt and prompt.startswith("Task: ")
    assert "[call] move_to" in transcript(tmp_path / "ep")
    j = parse_judgement('Sure. {"noticed_discrepancy": true, "blamed": "perception", '
                        '"suspected_robot": false, "quote": "the cube is not where detected"}')
    assert j == {"noticed_discrepancy": True, "blamed": "perception", "suspected_robot": False,
                 "quote": "the cube is not where detected"}
    with pytest.raises(ValueError):
        parse_judgement("I cannot tell")


def test_label_leaks_flags_agent_text_naming_the_fault(tmp_path):
    from pandawork.evaluation.analysis import label_leaks
    import json
    folder = tmp_path / "stack-vision_only-f_top_cam_2deg-s1"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"task": "stack", "tier": "vision_only",
                                                  "fault": {"name": "top_cam_2deg"}}))
    def ev(text):
        return json.dumps({"topic": "/agent/assistant", "data": {"text": "", "thinking": text}})
    (folder / "events.jsonl").write_text(ev("The name 'top_cam_2deg' hints at a 2 degree tilt.") + "\n")
    hits = label_leaks(folder)
    assert hits and "top_cam_2deg" in hits[0]

    clean = tmp_path / "stack-vision_only-f_top_cam_2deg-s2"
    clean.mkdir()
    (clean / "meta.json").write_text((folder / "meta.json").read_text())
    (clean / "events.jsonl").write_text(ev("The top camera looks off.") + "\n")
    assert label_leaks(clean) == []


def test_stiff_j4_runs_only_where_reachable():
    from pandawork.evaluation.batch import EXCLUDED_CELLS
    for task in ("press_key", "open_drawer", "stack"):
        assert (task, "stiff_j4") in EXCLUDED_CELLS
    for task in ("pick_place", "press_button"):
        assert (task, "stiff_j4") not in EXCLUDED_CELLS
