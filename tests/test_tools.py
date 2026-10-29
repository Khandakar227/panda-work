"""Tool system: schemas, tier gate, guards, budget, validation, errors, batching, logging."""
import base64
import io
import json
import time
from typing import Literal

import numpy as np
import pytest
from PIL import Image
from pydantic import Field

from pandawork.tiers import Tier
from pandawork.tools import (
    Budget, BudgetLimits, Dispatcher, Group, ToolArgs, ToolCall, ToolContext, ToolError,
    ToolRegistry, ToolResult,
)
from pandawork.world import JointStateSensor, StreamDigest, World


class WaitArgs(ToolArgs):
    seconds: float = Field(gt=0, le=10, description="how long to let the simulation run")


class NudgeArgs(ToolArgs):
    status: Literal["ok", "partial", "failed"]


def make_registry(calls: list[str] | None = None) -> ToolRegistry:
    """Demo tools exercising each pipeline path. `calls` records which tools actually ran."""
    reg = ToolRegistry()
    log = calls if calls is not None else []

    @reg.tool(name="wait", description="Let time pass.", group=Group.ACT)
    def wait(ctx, args: WaitArgs):
        log.append("wait")
        for _ in range(round(args.seconds / ctx.world.timestep)):
            ctx.world.step()
        return ToolResult(data={"status": "ok", "t": ctx.world.clock.sim_time})

    @reg.tool(name="nudge", description="Pretend action with a chosen outcome.", group=Group.ACT)
    def nudge(ctx, args: NudgeArgs):
        log.append(f"nudge:{args.status}")
        return ToolResult(data={"status": args.status})

    @reg.tool(name="read_joints", description="Joint angles.", group=Group.PERCEIVE)
    def read_joints(ctx):
        log.append("read_joints")
        return ToolResult(data={"q": ctx.world.data.qpos[ctx.world.qadr]})

    @reg.tool(name="secret_pose", description="Exact pose.", group=Group.PERCEIVE,
              tiers=[Tier.PRIVILEGED])
    def secret_pose(ctx):
        log.append("secret_pose")
        return ToolResult(data={"pose": [0.5, 0.0, 0.2]})

    @reg.tool(name="boom", description="Harness bug.", group=Group.META)
    def boom(ctx):
        raise RuntimeError("kaboom")

    @reg.tool(name="refuse", description="Expected failure.", group=Group.META)
    def refuse(ctx):
        raise ToolError("not_reachable", "target is out of reach", hint="pick a closer point")

    @reg.tool(name="spin", description="Never finishes.", group=Group.META, timeout_s=0.01)
    def spin(ctx):
        while True:
            ctx.check_deadline()
            time.sleep(0.001)

    @reg.tool(name="snap", description="Tiny image.", group=Group.PERCEIVE)
    def snap(ctx):
        return ToolResult(data={"camera": "front"}, images=[np.full((24, 32, 3), 200, np.uint8)])

    @reg.tool(name="done", description="End the episode.", group=Group.META)
    def done(ctx):
        ctx.state["done"] = True
        return ToolResult(text="episode ended")

    return reg


def make_dispatcher(tier=Tier.PRIVILEGED, limits=BudgetLimits(), calls=None):
    world = World(tier=tier, seed=0, sensors=[JointStateSensor()])
    ctx = ToolContext(world=world, budget=Budget(limits, world.clock))
    return Dispatcher(make_registry(calls), ctx)


def text_of(block: dict) -> str:
    return "\n".join(c["text"] for c in block["content"] if c["type"] == "text")


def one(d: Dispatcher, name: str, args=None, call_id="c1") -> dict:
    return d.run_batch([ToolCall(call_id, name, args or {})])[0]


# ---- schemas -------------------------------------------------------------------------------

def test_schemas_are_ordered_filtered_and_lean():
    reg = make_registry()
    names = [s["name"] for s in reg.schemas(Tier.REALISTIC)]
    assert names == ["read_joints", "snap", "nudge", "wait", "boom", "done", "refuse", "spin"]
    assert "secret_pose" in [s["name"] for s in reg.schemas(Tier.PRIVILEGED)]
    wait = next(s for s in reg.schemas(Tier.REALISTIC) if s["name"] == "wait")
    assert wait["input_schema"] == {
        "additionalProperties": False, "type": "object", "required": ["seconds"],
        "properties": {"seconds": {"description": "how long to let the simulation run",
                                   "exclusiveMinimum": 0, "maximum": 10, "type": "number"}},
    }
    no_args = next(s for s in reg.schemas(Tier.REALISTIC) if s["name"] == "done")
    assert no_args["input_schema"] == {"additionalProperties": False, "properties": {},
                                       "type": "object"}


def test_schemas_are_byte_identical_across_registries():
    dump = lambda: json.dumps(make_registry().schemas(Tier.PRIVILEGED))
    assert dump() == dump()


def test_duplicate_tool_names_are_rejected():
    reg = make_registry()
    with pytest.raises(ValueError):
        reg.register(reg.get("wait"))


# ---- gate, guards, validation, errors ------------------------------------------------------

def test_tier_gate_denies_without_running_the_tool():
    calls = []
    d = make_dispatcher(Tier.REALISTIC, calls=calls)
    block = one(d, "secret_pose")
    assert block["is_error"] and "[tool_not_available]" in text_of(block)
    assert calls == []


def test_guard_denies_after_done():
    calls = []
    d = make_dispatcher(calls=calls)
    one(d, "done")
    block = one(d, "read_joints")
    assert block["is_error"] and "[denied]" in text_of(block) and calls == []


def test_unknown_tool_lists_alternatives():
    block = one(make_dispatcher(Tier.REALISTIC), "fly")
    assert "[unknown_tool]" in text_of(block) and "read_joints" in text_of(block)
    assert "secret_pose" not in text_of(block)  # hints never leak tools outside the tier


@pytest.mark.parametrize("args, field", [({}, "seconds"), ({"seconds": "soon"}, "seconds"),
                                         ({"seconds": 50}, "seconds"),
                                         ({"seconds": 1, "speed": 2}, "speed")])
def test_invalid_arguments_name_the_field(args, field):
    block = one(make_dispatcher(), "wait", args)
    assert block["is_error"] and "[invalid_arguments]" in text_of(block) and field in text_of(block)


def test_expected_failure_carries_code_and_hint():
    text = text_of(one(make_dispatcher(), "refuse"))
    assert "error [not_reachable]: target is out of reach" in text and "hint: pick a closer point" in text


def test_tool_bug_becomes_internal_error_and_the_episode_continues():
    d = make_dispatcher()
    block = one(d, "boom")
    assert block["is_error"] and "RuntimeError: kaboom" in text_of(block)
    event = d.ctx.world.bus.latest("/agent/tool_result")
    assert "Traceback" in event.data["traceback"]
    assert "is_error" not in one(d, "read_joints", call_id="c2")


def test_timeout_stops_a_runaway_tool():
    assert "[timeout]" in text_of(one(make_dispatcher(), "spin"))


# ---- budget --------------------------------------------------------------------------------

def test_call_budget():
    d = make_dispatcher(limits=BudgetLimits(max_calls=2))
    for i in range(2):
        assert "is_error" not in one(d, "read_joints", call_id=f"c{i}")
    block = one(d, "read_joints", call_id="c3")
    assert "[budget_exhausted]" in text_of(block) and "2/2" in text_of(block)


def test_sim_time_budget():
    d = make_dispatcher(limits=BudgetLimits(max_sim_seconds=0.5))
    one(d, "wait", {"seconds": 0.6})
    assert "[budget_exhausted]" in text_of(one(d, "read_joints", call_id="c2"))


# ---- batches -------------------------------------------------------------------------------

def test_batch_runs_in_order_and_failed_action_skips_later_actions():
    calls = []
    d = make_dispatcher(calls=calls)
    blocks = d.run_batch([
        ToolCall("a", "nudge", {"status": "ok"}),
        ToolCall("b", "nudge", {"status": "partial"}),
        ToolCall("c", "wait", {"seconds": 0.1}),
        ToolCall("d", "read_joints", {}),
    ])
    assert [b["tool_use_id"] for b in blocks] == ["a", "b", "c", "d"]
    assert calls == ["nudge:ok", "nudge:partial", "read_joints"]  # "wait" was skipped
    assert "[skipped]" in text_of(blocks[2]) and blocks[2]["is_error"]
    assert "is_error" not in blocks[3]


def test_footer_appears_once_at_the_end_of_the_turn():
    blocks = make_dispatcher().run_batch([ToolCall("a", "read_joints", {}),
                                          ToolCall("b", "wait", {"seconds": 0.1})])
    assert "[t=" not in text_of(blocks[0])
    assert text_of(blocks[1]).endswith("[t=0.10 s · step 50 · calls 2/40 · time 0.1/120 s]")


def test_repeat_reminder_on_third_identical_call():
    d = make_dispatcher()
    texts = [text_of(one(d, "read_joints", call_id=f"c{i}")) for i in range(3)]
    assert "note: this is call 3" in texts[2] and "note:" not in texts[1]
    assert "note:" not in text_of(one(d, "wait", {"seconds": 0.1}, call_id="c4"))


# ---- rendering and logging -----------------------------------------------------------------

def test_images_are_sent_as_png_blocks_and_logged():
    d = make_dispatcher()
    block = one(d, "snap")
    image = next(c for c in block["content"] if c["type"] == "image")
    decoded = Image.open(io.BytesIO(base64.b64decode(image["source"]["data"])))
    assert decoded.size == (32, 24)
    assert d.ctx.world.bus.latest("/agent/tool_result").data["image_0"].shape == (24, 32, 3)


def test_every_call_is_logged_on_the_bus_with_matching_ids():
    d = make_dispatcher()
    d.run_batch([ToolCall("a", "wait", {"seconds": 0.1}), ToolCall("b", "fly", {})])
    calls = d.ctx.world.bus.window("/agent/tool_call", 1e9)
    results = d.ctx.world.bus.window("/agent/tool_result", 1e9)
    assert [m.data["call_id"] for m in calls] == ["a", "b"]
    assert [(m.data["call_id"], m.data["ok"]) for m in results] == [("a", True), ("b", False)]
    assert json.loads(calls[0].data["args"]) == {"seconds": 0.1}
    # The call is stamped before the action ran, the result after.
    assert calls[0].header.sim_time == 0.0
    assert results[0].header.sim_time == pytest.approx(0.1)


def test_tool_logging_is_deterministic():
    def run():
        d = make_dispatcher(Tier.REALISTIC)
        digest = StreamDigest()
        d.ctx.world.bus.subscribe("*", digest)
        d.run_batch([ToolCall("a", "wait", {"seconds": 0.2}), ToolCall("b", "read_joints", {}),
                     ToolCall("c", "refuse", {})])
        return digest.hexdigest()
    assert run() == run()


def test_timeout_is_measured_in_simulated_time():
    """A tool that keeps stepping the world past its budget stops at the same simulated time
    on every machine, so replay stays exact; wall-clock speed does not matter."""
    reg = ToolRegistry()

    @reg.tool(name="crawl", description="Steps the world until stopped.", group=Group.META, timeout_s=0.05)
    def crawl(ctx):
        while True:
            ctx.world.step(5)
            ctx.check_deadline()

    world = World(tier=Tier.PRIVILEGED, seed=0, sensors=[JointStateSensor()])
    d = Dispatcher(reg, ToolContext(world=world, budget=Budget(BudgetLimits(), world.clock)))
    t0 = d.ctx.world.clock.sim_time
    text = text_of(one(d, "crawl"))
    assert "[timeout]" in text and "ran out of time" in text
    assert 0.05 <= d.ctx.world.clock.sim_time - t0 < 0.07
