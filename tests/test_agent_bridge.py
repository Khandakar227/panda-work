"""The Claude Agent SDK bridge, exercised offline: no CLI, no model, no cost.

Claude Code's behaviour is imitated with the SDK's own message types: for each tool call it
runs the PreToolUse hook, then the MCP handler; the assistant message that issued the calls
arrives on the message stream, possibly after the handler has started.
"""
import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import AssistantMessage, ToolUseBlock  # noqa: E402

from pandawork.agent import ToolBridge, system_prompt, to_mcp  # noqa: E402
from pandawork.agent.bridge import PREFIX  # noqa: E402
from pandawork.recording import EpisodeRecorder, load, replay  # noqa: E402
from pandawork.tasks import make_episode  # noqa: E402
from pandawork.tiers import Tier  # noqa: E402


class FakeClaudeCode:
    """Drives a ToolBridge the way Claude Code does."""

    def __init__(self, bridge: ToolBridge):
        self.bridge = bridge
        self.handlers = {t.name: bridge._handler(t.name) for t in bridge.tools}
        self.n = 0

    async def message(self, calls: list[tuple[str, dict]], announce_late: bool = False) -> list[dict]:
        self.n += 1
        ids = [f"toolu_{self.n}_{i}" for i in range(len(calls))]
        msg = AssistantMessage(content=[ToolUseBlock(id=i, name=PREFIX + name, input=args)
                                        for i, (name, args) in zip(ids, calls)],
                               model="fake", message_id=f"msg_{self.n}")
        if not announce_late:
            self.bridge.observe(msg)
        results = []
        for tool_use_id, (name, args) in zip(ids, calls):
            await self.bridge.pre_tool_use({"tool_name": PREFIX + name, "tool_input": args,
                                            "tool_use_id": tool_use_id}, tool_use_id, None)
            task = asyncio.ensure_future(self.handlers[name](args))
            if announce_late:  # the stream delivers the message only after the handler started
                await asyncio.sleep(0.05)
                self.bridge.observe(msg)
                announce_late = False
            results.append(await task)
        return results


def text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if c["type"] == "text")


async def finish(bridge, recorder=None):
    await bridge.on_sim(bridge.end_turn)
    path = await bridge.on_sim(recorder.finish) if recorder else None
    bridge.shutdown()
    return path


def test_mcp_conversion():
    blocks = [{"type": "text", "text": "hi"},
              {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}]
    assert to_mcp(blocks) == [{"type": "text", "text": "hi"},
                              {"type": "image", "data": "QUJD", "mimeType": "image/png"}]


def test_system_prompt_is_fixed_and_tier_specific():
    assert system_prompt(Tier.REALISTIC) == system_prompt(Tier.REALISTIC)
    assert "get_object_poses" in system_prompt(Tier.PRIVILEGED)
    assert "only have camera images" in system_prompt(Tier.VISION_ONLY)
    assert "(1.25, 0, 0.55)" in system_prompt(Tier.REALISTIC)  # camera placements included


def test_turns_skips_and_footers():
    async def scenario():
        bridge = ToolBridge(make_episode("reach", seed=0))
        claude = FakeClaudeCode(bridge)
        first = await claude.message([("move_to", {"x": 1.3, "y": 0, "z": 0.3}),
                                      ("gripper", {"action": "close"})])
        second = await claude.message([("get_robot_state", {})])
        await finish(bridge)
        return bridge, first, second

    bridge, (move, grip), (state,) = asyncio.run(scenario())
    assert "status: failed" in text(move) and not move["is_error"]
    assert grip["is_error"] and "[skipped]" in text(grip)
    assert not state["is_error"] and "joints_deg" in text(state)
    assert all("[t=" in text(r) for r in (move, grip, state))  # each result carries the footer
    calls = bridge.episode.world.bus.window("/agent/tool_call", 1e9)
    assert [(m.data["call_id"], m.data["turn"]) for m in calls] == [
        ("toolu_1_0", 0), ("toolu_1_1", 0), ("toolu_2_0", 1)]
    assert bridge.dispatcher.turn == 2 and bridge.unmatched_calls == 0


def test_handler_waits_for_its_message():
    async def scenario():
        bridge = ToolBridge(make_episode("reach", seed=0))
        claude = FakeClaudeCode(bridge)
        await claude.message([("get_task", {})])
        await claude.message([("get_robot_state", {})], announce_late=True)
        await finish(bridge)
        return bridge

    bridge = asyncio.run(scenario())
    turns = [m.data["turn"] for m in bridge.episode.world.bus.window("/agent/tool_call", 1e9)]
    assert turns == [0, 1]


def test_unpaired_call_still_runs():
    async def scenario():
        bridge = ToolBridge(make_episode("reach", seed=0))
        result = await bridge._handler("get_task")({})
        await finish(bridge)
        return bridge, result

    bridge, result = asyncio.run(scenario())
    assert not result["is_error"] and bridge.unmatched_calls == 1


def test_images_reach_claude_code_as_mcp_images(cameras):
    async def scenario():
        bridge = ToolBridge(make_episode("reach", seed=0))
        (result,) = await FakeClaudeCode(bridge).message([("look", {"camera": "side"})])
        await finish(bridge)
        return result

    result = asyncio.run(scenario())
    image = next(c for c in result["content"] if c["type"] == "image")
    assert image["mimeType"] == "image/png" and len(image["data"]) > 1000


def test_bridge_driven_episode_replays_exactly(tmp_path):
    async def scenario():
        episode = make_episode("pick_place", tier=Tier.REALISTIC, seed=2)
        bridge = ToolBridge(episode)
        recorder = EpisodeRecorder(episode, tmp_path / "ep")
        claude = FakeClaudeCode(bridge)
        cube = episode.world.object_pose(episode.task.target)[0]
        await claude.message([("detect_objects", {"camera": "top"}),
                              ("move_to", {"x": float(cube[0]), "y": float(cube[1]), "z": 0.15})])
        await claude.message([("move_to", {"x": 1.3, "y": 0, "z": 0.3}),
                              ("gripper", {"action": "open"})])
        await claude.message([("done", {"success": False, "explanation": "test"})])
        return await finish(bridge, recorder)

    path = asyncio.run(scenario())
    log = load(path)
    assert [len(t) for t in log.turns()] == [2, 2, 1]
    report = replay(log)
    assert report.identical, report
    assert json.loads((path / "result.json").read_text())["ended_by_agent"] is True


def test_usage_accounting_matches_claude_code():
    """From a real episode's ResultMessage: the ledger reproduces Claude Code's reported cost."""
    from claude_agent_sdk import ResultMessage
    from pandawork.agent.session import record_usage as _record_usage
    from pandawork.recording import CostLedger

    message = ResultMessage(
        subtype="success", duration_ms=24000, duration_api_ms=20000, is_error=False, num_turns=12,
        session_id="s", total_cost_usd=0.0546672,
        usage={"input_tokens": 14, "output_tokens": 1380, "cache_read_input_tokens": 36066,
               "cache_creation_input_tokens": 8160,
               "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 8160}},
        model_usage={"claude-haiku-4-5-20251001": {"inputTokens": 916, "outputTokens": 14,
                                                   "canonicalModel": "claude-haiku-4-5"},
                     "claude-sonnet-5-5": {"canonicalModel": "claude-sonnet-5-5"}})
    episode = make_episode("reach", seed=0)
    ledger = CostLedger(episode.world.bus, episode.dispatcher.ctx.budget)
    _record_usage(ledger, "claude-sonnet-5-5", message)
    assert [m for m, _, _ in ledger.calls] == ["claude-sonnet-5-5", "claude-haiku-4-5"]
    assert ledger.total_usd == pytest.approx(message.total_cost_usd, rel=1e-6)


# ---- live console ---------------------------------------------------------------------------

def test_overlay_text_columns_stay_aligned():
    from pandawork.agent.live import text_rows
    labels, values = text_rows([("You", "pick up the red cube"), ("Claude", "word " * 40)], width=30)
    assert len(labels.split("\n")) == len(values.split("\n"))
    assert labels.split("\n")[0] == "You" and labels.split("\n")[1] == "Claude"
    assert values.split("\n")[-1].endswith("...")  # long text is cut to max_lines


class FakeViewer:
    def __init__(self, model):
        import mujoco
        self.user_scn = mujoco.MjvScene(model, maxgeom=10)
        self.texts, self.syncs = None, 0

    def is_running(self):
        return True

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        self.syncs += 1


def test_overlay_shows_instruction_action_and_target():
    from pandawork.agent.live import ViewerOverlay
    from pandawork.tasks import PlaygroundTask
    from pandawork.tasks.base import call
    episode = make_episode(PlaygroundTask(), seed=0)
    viewer = FakeViewer(episode.world.plant)
    overlay = ViewerOverlay(viewer, episode, realtime=1000.0)
    overlay.set(you="pick up the red cube")
    call(episode.dispatcher, ("move_to", {"x": 0.5, "y": 0.1, "z": 0.2}))
    overlay.draw()
    _, _, labels, values = viewer.texts
    assert "You" in labels and "pick up the red cube" in values
    assert "move_to: status: reached" in values
    assert viewer.user_scn.ngeom == 1 and list(viewer.user_scn.geoms[0].pos) == pytest.approx([0.5, 0.1, 0.2])
    assert viewer.syncs > 10  # the viewer was refreshed while the arm moved


def test_session_with_several_instructions_replays_exactly(tmp_path):
    from pandawork.tasks import PlaygroundTask
    from pandawork.tools import BudgetLimits

    async def scenario():
        episode = make_episode(PlaygroundTask(), seed=1, limits=BudgetLimits(None, None))
        bridge = ToolBridge(episode)
        recorder = EpisodeRecorder(episode, tmp_path / "live")
        claude = FakeClaudeCode(bridge)
        for instruction, calls in [
            ("move above the red cube", [("move_to", {"x": 0.5, "y": 0.0, "z": 0.2}),
                                         ("done", {"success": True, "explanation": "ok"})]),
            ("open the gripper", [("gripper", {"action": "open"}),
                                  ("done", {"success": True, "explanation": "ok"})]),
        ]:
            await bridge.on_sim(lambda: episode.world.bus.publish(
                "/agent/user_message", {"text": instruction}, "agent"))
            results = await claude.message(calls)
            assert not any(r["is_error"] for r in results), [text(r) for r in results]
            await bridge.on_sim(episode.reopen)
        return await finish(bridge, recorder)

    log = load(asyncio.run(scenario()))
    assert log.instruction_starts() == {0, 1}
    assert replay(log).identical


def test_helper_model_is_counted_once_across_requests():
    from claude_agent_sdk import ResultMessage
    from pandawork.agent.session import record_usage
    from pandawork.recording import CostLedger

    def request(sonnet_usage, haiku_total):
        return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                             num_turns=1, session_id="s", usage=sonnet_usage,
                             model_usage={"claude-haiku-4-5": {"inputTokens": haiku_total,
                                                               "canonicalModel": "claude-haiku-4-5"}})
    episode = make_episode("reach", seed=0)
    ledger = CostLedger(episode.world.bus, episode.dispatcher.ctx.budget)
    seen = {}
    record_usage(ledger, "claude-sonnet-5-5", request({"output_tokens": 100}, 900), seen)
    record_usage(ledger, "claude-sonnet-5-5", request({"output_tokens": 100}, 900), seen)  # no new Haiku use
    assert [m for m, _, _ in ledger.calls] == ["claude-sonnet-5-5", "claude-haiku-4-5", "claude-sonnet-5-5"]


def test_reported_costs_are_used_and_unknown_models_never_crash():
    """From a real live session: Claude Code had switched its helper to a model missing from
    PRICES (the ledger raised and ended the session) and priced Sonnet's cache reads below
    PRICES. The ledger now takes each model's reported costUSD, as deltas across requests."""
    from claude_agent_sdk import ResultMessage
    from pandawork.agent.session import record_usage
    from pandawork.recording import CostLedger

    def request(sonnet_out, sonnet_cost, haiku_in, haiku_cost):
        return ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
            session_id="s", total_cost_usd=sonnet_cost + haiku_cost,
            usage={"input_tokens": 24, "output_tokens": sonnet_out, "cache_read_input_tokens": 120024,
                   "cache_creation_input_tokens": 17694,
                   "cache_creation": {"ephemeral_1h_input_tokens": 17694}},
            model_usage={"claude-haiku-5-5": {"inputTokens": haiku_in, "outputTokens": 17,
                                              "costUSD": haiku_cost, "canonicalModel": "claude-haiku-5-5"},
                         "claude-sonnet-5-5": {"outputTokens": sonnet_out, "costUSD": sonnet_cost,
                                               "canonicalModel": "claude-sonnet-5-5"}})
    episode = make_episode("reach", seed=0)
    ledger = CostLedger(episode.world.bus, episode.dispatcher.ctx.budget)
    seen = {}
    record_usage(ledger, "claude-sonnet-5-5", request(4791, 0.1307364, 1180, 0.0001265), seen)
    assert ledger.total_usd == pytest.approx(0.1308629, rel=1e-9)  # what Claude Code reported
    record_usage(ledger, "claude-sonnet-5-5", request(6000, 0.2000000, 1180, 0.0001265), seen)
    assert ledger.total_usd == pytest.approx(0.2001265, rel=1e-9)  # no helper use the 2nd time
    assert [m for m, _, _ in ledger.calls] == ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-sonnet-5-5"]

    from pandawork.recording.costs import Usage
    assert ledger.record("claude-unknown-9", Usage(input_tokens=10)) == 0.0  # no report, no price
    assert ledger.unpriced == {"claude-unknown-9"}


def test_overlay_names_the_fault_and_marks_believed_and_true_fingertips():
    from pandawork.core.faults import FAULTS
    from pandawork.tasks.base import call
    from pandawork.watch import ViewerOverlay
    episode = make_episode("pick_place", seed=0, fault=FAULTS["encoder_j1_5deg"])
    viewer = FakeViewer(episode.world.plant)
    overlay = ViewerOverlay(viewer, episode, realtime=0)
    call(episode.dispatcher, ("move_to", {"x": 0.5, "y": 0.1, "z": 0.2}))
    overlay.draw()
    _, _, labels, values = viewer.texts
    assert "Fault" in labels and "encoder_j1_5deg (hidden)" in values and "Hand" in labels
    scn = viewer.user_scn
    assert scn.ngeom == 3 and [g.label for g in scn.geoms[:3]] == ["", "believed", "true"]
    believed, true = overlay.fingertips()
    assert 0.02 < np.linalg.norm(true - believed) < 0.08  # the 5 deg offset, a few cm at the hand
    assert list(scn.geoms[2].pos) == pytest.approx(list(true))
    assert "4." in values or "5." in values  # the live distance is on the panel
    episode.world.close()


def test_run_oracle_calls_the_episode_hook_and_its_cleanup(tmp_path):
    from pandawork.evaluation.batch import Job, run_oracle
    seen = []
    def on_episode(episode):
        seen.append(("start", episode.task.name, episode.world.fault.name))
        return lambda: seen.append(("end",))
    run_oracle(Job("reach", Tier.PRIVILEGED, 0, "fingertip_1cm_long"), tmp_path / "ep", on_episode=on_episode)
    assert seen == [("start", "reach", "fingertip_1cm_long"), ("end",)]
    assert (tmp_path / "ep" / "result.json").exists()


def test_overlay_shows_what_claude_says_from_the_bus():
    """Batch runs do not pass SDK messages to the viewer; the panel must still show Claude's
    words, which the message log publishes on /agent/assistant."""
    from pandawork.watch import ViewerOverlay
    episode = make_episode("reach", seed=0)
    viewer = FakeViewer(episode.world.plant)
    overlay = ViewerOverlay(viewer, episode, realtime=0)
    overlay.set(you=episode.task.instruction)
    episode.world.bus.publish("/agent/assistant", {"message_id": "m1", "model": "x", "text": "",
                                                   "thinking": "The yellow cube is at x=0.5.",
                                                   "tool_uses": "[]", "stop_reason": ""}, "agent")
    assert "yellow cube" in viewer.texts[3]
    episode.world.bus.publish("/agent/assistant", {"message_id": "m2", "model": "x", "text": "Moving above it now.",
                                                   "thinking": "", "tool_uses": "[]", "stop_reason": ""}, "agent")
    labels, values = viewer.texts[2], viewer.texts[3]
    assert "Moving above it now." in values and "Episode" in labels and "reach · privileged · seed 0" in values
    episode.world.close()


def test_agent_cwd_never_names_the_episode(tmp_path):
    """Claude Code shows the model its working directory, so the episode folder (whose name
    carries the fault label) must never be the cwd."""
    from pandawork.agent.session import AgentConfig, claude_options, neutral_cwd
    episode_dir = tmp_path / "stack-vision_only-f_top_cam_2deg-s1"
    episode_dir.mkdir()
    from types import SimpleNamespace
    bridge = SimpleNamespace(allowed_tools=[], mcp_server=lambda: {}, pre_tool_use=lambda *a: None)
    opts = claude_options(bridge, AgentConfig(), "prompt", episode_dir, 10, [])
    for word in ("top_cam", "vision_only", "stack", "f_", "grid", "pandawork", "panda"):
        assert word not in opts.cwd
    assert opts.cwd == str(neutral_cwd())
