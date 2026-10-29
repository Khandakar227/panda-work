"""Tool console: call the robot tools the way Claude will, and see exactly what Claude gets back.

  python scripts/tool_console.py                    # interactive, privileged tier, demo scene
  python scripts/tool_console.py --viewer           # ...and watch the robot in the MuJoCo viewer
  python scripts/tool_console.py --tier realistic   # noisy sensors, no exact poses
  python scripts/tool_console.py --demo             # guided tour: pick-and-place + failure cases
  python scripts/tool_console.py --schemas          # the tool definitions Claude would receive

One turn per line. A call is `name {json args}`; separate several calls with ` ; `, e.g.
  move_to {"x": 0.5, "y": 0.12, "z": 0.15} ; gripper {"action": "open"}
Console commands: :tools  :budget  :quit
Demo scene: red_cube (0.50, 0.12), blue_cube (0.45, -0.15, yawed 30 deg),
            green_cylinder (0.62, 0.00). Heights: cubes 5 cm, cylinder 8 cm.
"""
import argparse
import base64
import json
import time

from pandawork.core.robot import PACKAGE_ROOT
from pandawork.core.scene import DEMO_OBJECTS, build_scene
from pandawork.core.viewer import exit_after
from pandawork.tiers import Tier
from pandawork.tools import (
    Budget, BudgetLimits, Dispatcher, ToolCall, ToolContext, default_registry,
)
from pandawork.world import World

OUT_DIR = PACKAGE_ROOT / "outputs" / "console"

DEMO = [
    "get_task",
    "get_object_poses",
    'detect_objects {"camera": "front"}',
    'look {"camera": "front"}',
    'move_to {"x": 0.5, "y": 0.12, "z": 0.15} ; gripper {"action": "open"}',
    'move_to {"x": 0.5, "y": 0.12, "z": 0.025} ; gripper {"action": "close"}',
    'move_to {"x": 0.5, "y": 0.12, "z": 0.15} ; read_wrist_ft',
    'move_to {"x": 0.4, "y": 0.25, "z": 0.15, "yaw_deg": 45} ; '
    'move_to {"x": 0.4, "y": 0.25, "z": 0.0, "yaw_deg": 45, "speed": 0.05}',
    'gripper {"action": "open"} ; move_relative {"dz": 0.1}',
    "get_object_poses",
    'move_to {"x": 1.3, "y": 0.0, "z": 0.3} ; gripper {"action": "close"}',
    'check_reachable {"x": 0.45, "y": -0.15, "z": 0.025, "yaw_deg": 30}',
    'simulate {"actions": [{"tool": "move_to", "args": {"x": 0.45, "y": -0.15, "z": 0.025, '
    '"yaw_deg": 30}}, {"tool": "gripper", "args": {"action": "close"}}, {"tool": "move_to", '
    '"args": {"x": 0.45, "y": -0.15, "z": 0.2, "yaw_deg": 30}}]}',
    'done {"success": true, "explanation": "moved the red cube to (0.4, 0.25)"}',
]


class Console:
    def __init__(self, tier: Tier, seed: int, limits: BudgetLimits):
        self_model, plant = build_scene(DEMO_OBJECTS)
        self.world = World(plant, self_model, tier=tier, seed=seed)
        self.world.step()  # fill the sensor buffers once
        ctx = ToolContext(world=self.world, budget=Budget(limits, self.world.clock))
        self.dispatcher = Dispatcher(default_registry(), ctx)
        self.n = 0
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def attach_viewer(self, viewer, realtime: float) -> None:
        """Show every physics step in the viewer, paced to `realtime` x real time."""
        state = {"wall": time.perf_counter(), "sim": self.world.clock.sim_time}

        def hook(world):
            if world.clock.step % 10:
                return
            viewer.sync()
            ahead = (world.clock.sim_time - state["sim"]) / realtime - (time.perf_counter() - state["wall"])
            if ahead > 0:
                time.sleep(ahead)
            else:  # we fell behind (e.g. the agent was thinking): restart the pacing clock
                state["wall"], state["sim"] = time.perf_counter(), world.clock.sim_time

        self.world.step_hooks.append(hook)

    def turn(self, line: str) -> None:
        calls = [self._parse(part) for part in line.split(" ; ") if part.strip()]
        print(f"\n>>> {line}")
        for call, block in zip(calls, self.dispatcher.run_batch(calls)):
            print(f"--- {call.id} {call.name} [{'x' if block.get('is_error') else 'ok'}]")
            for i, item in enumerate(block["content"]):
                if item["type"] == "text":
                    print("    " + item["text"].replace("\n", "\n    "))
                else:
                    path = OUT_DIR / f"{call.id}_{i}.png"
                    path.write_bytes(base64.b64decode(item["source"]["data"]))
                    print(f"    [image -> {path.relative_to(PACKAGE_ROOT)}]")

    def _parse(self, text: str) -> ToolCall:
        self.n += 1
        name, _, raw = text.strip().partition(" ")
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args = {"_unparseable": raw}
        return ToolCall(f"c{self.n}", name, args)

    def repl(self, tier: Tier) -> None:
        print(__doc__)
        while True:
            try:
                line = input(f"[{tier.value}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if line in (":quit", ":q"):
                return
            if line == ":tools":
                for tool in self.dispatcher.registry.available(tier):
                    print(f"  {tool.group.value:<9} {tool.name:<17} {tool.description[:70]}")
            elif line == ":budget":
                print("  " + self.dispatcher.ctx.budget.footer())
            elif line:
                self.turn(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", default="privileged", choices=[t.value for t in Tier])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--schemas", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="watch the robot live")
    parser.add_argument("--realtime", type=float, default=1.0, help="viewer speed (x real time)")
    args = parser.parse_args()
    tier = Tier(args.tier)

    if args.schemas:
        schemas = default_registry().schemas(tier)
        print(json.dumps(schemas, indent=2))
        size = len(json.dumps(schemas))
        print(f"\n{len(schemas)} tools for tier {tier.value}; {size} characters, "
              f"very roughly {size // 4} tokens")
        return

    console = Console(tier, args.seed, BudgetLimits(max_calls=args.max_calls))
    run = (lambda: [console.turn(line) for line in DEMO]) if args.demo else (lambda: console.repl(tier))
    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(console.world.plant, console.world.data) as viewer:
            console.attach_viewer(viewer, args.realtime)
            run()
            if args.demo:
                input("\nDemo finished. Press Enter to close the viewer.")
    else:
        run()


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer
