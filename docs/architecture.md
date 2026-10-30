# Architecture

How the pieces fit, written from the code as it is. Module names are under `src/pandawork/`.

## The modules

| module | responsibility |
|---|---|
| `core/robot.py`, `core/scene.py` | the Panda model and the table; scene builders (cubes, pads, buttons, drawer, bins, keypad) |
| `core/ik.py` | `SelfModel`: forward kinematics, damped-least-squares IK with soft joint limits, gravity torques, all on the robot's *belief* |
| `core/control.py` | one physics step of control: joint position servo targets plus gravity compensation from the self-model |
| `core/render.py`, `core/viewer.py` | offscreen cameras (RGB, depth, segmentation); the passive viewer and a clean exit |
| `world/bus.py`, `world/messages.py` | a synchronous in-process publish/subscribe bus with a simulated clock and per-topic history |
| `world/sensors.py`, `world/noise.py` | sensors that read the plant, add the tier's noise, and publish stamped messages |
| `world/world.py` | `World`: plant + self-model + clock + bus + sensors; `step`, `fork`, object access |
| `skills/` | closed-loop motion (`move_tcp`, grasp IK, contact watching), perception (detection from segmentation and depth), sensing helpers; shared by tools and oracles |
| `tools/` | `Tool` base and arg models, the registry with its tier gate, budgets, the dispatcher pipeline, result rendering, and `robot_tools.py` with the 16 tools |
| `tasks/` | `Task` base (scene, reset, monitor, evaluate, oracle), the task library, `make_episode` |
| `agent/` | the Claude Agent SDK client: system prompt per tier, the MCP bridge, the episode runner, the live console, session options and usage accounting |
| `recording/` | the recorder (events, streams, snapshots, blobs, truth), exact replay, video, the cost ledger |
| `evaluation/` | batches (resumable grids) and reports |
| `watch.py` | the viewer panel and markers, hooked into any runner |

## One episode

`tasks/episode.py:make_episode(task, tier, seed, limits)`:

1. The task builds a scene spec; `task.models()` compiles two identical models from it, the
   plant and the self-model.
2. `World(plant, self_model, tier, seed)` creates the clock, the bus, the default sensors
   (joints, wrist F/T, touch, cameras) with the tier's noise, and seeded random streams per sensor.
3. The task places objects from a seeded generator and writes its instruction; the world settles
   for a moment of simulated time.
4. A `ToolRegistry` is filtered to the tier, a `Dispatcher` is built around it with the budget,
   and the `Episode` object carries all of this to whoever drives it: the oracle, the Claude
   runner, the live console, or a test.

Time moves only inside `World.step`. Between tool calls the world is paused, however long the
agent thinks.

## One tool call

The agent's client calls a tool on the MCP server that `agent/bridge.py` exposes. From there
(`tools/dispatch.py`):

```
log call -> tier gate -> guards -> budget -> validate args -> run (with a deadline)
         -> render the result -> log result
```

- **Tier gate.** The registry only offered the tools the tier allows; a call to anything else
  returns an error result the agent can read.
- **Budget.** Tool calls and simulated seconds are counted per episode (`tools/budget.py`);
  tokens and dollars are charged by the agent runner. Exhaustion ends the episode without `done`.
- **Run.** The tool calls a skill. A motion skill (`skills/motion.py:move_tcp`) solves IK on the
  self-model, streams joint targets to `World.command`, and calls `World.step` in a loop while a
  contact watcher checks the wrist force and touch pads; it stops early on unexpected contact and
  returns `reached` or `partial` with the believed fingertip pose.
- **Inside `World.step`** (`world/world.py`): the controller applies the servo targets with
  gravity compensation computed from the self-model at the encoder readings; MuJoCo steps the
  plant; the clock advances; periodic sensors publish; step hooks run (the task's `monitor`, the
  viewer, the truth logger).
- **Render.** The `ToolResult` becomes text and images for the agent (`tools/rendering.py`);
  image blobs are stored once by content hash.
- **Turn rule.** Calls from one model turn run in order; if an acting call fails or is partial,
  the rest of that turn's acting calls are skipped, so the robot never executes a plan that no
  longer fits. The skipped calls are reported back as such.

Every message on the bus, including the tool call and its result, reaches the recorder. The
recorder writes `events.jsonl` (low-rate messages), `streams.npz` (high-rate sensors),
`snapshots.npz` (full simulator state at the start, after every turn, and at the end), `blobs/`
(images), and, around each tool call, `truth.jsonl` with the fingertip, object poses and
contacts. Truth is written by the recorder's hooks, never by a tool.

## Belief and plant

Only `World.step` touches the plant's dynamics, and only through `apply_control`. Everything
the agent reads goes through a sensor or a skill that computes on the self-model from the
encoder readings, so the non-privileged tiers get what a real robot's software would have:
its own model of itself plus its sensors, never ground truth. The privileged tier is the
deliberate exception, for experiments that want a "told the truth" control.

The agent's `simulate` tool uses `World.fork()`: an independent copy at the current instant,
with its own clock, bus and noise streams, on which a plan can be run before committing to it.

## Determinism and replay

The bus delivers synchronously in subscription order, the clock is simulated, and every sensor
draws noise from its own seeded stream. The same `(task, tier, seed)` and the same
sequence of tool calls therefore produce the same bytes. `recording/replay.py` re-executes the
logged calls on a fresh episode and compares the final state hash and every snapshot;
`scripts/episodes.py replay` reports whether the reproduction is identical. Video rendering is
a replay with a camera running.

## The agent side

`agent/prompt.py` holds the system prompt: a base and a paragraph per tier. `agent/bridge.py` turns the registry into an MCP server (named `panda`), pairs
each SDK tool-use event with the call that reaches the server, enforces the turn rule, and ends
the episode when `done` arrives with the agent's claim. `agent/runner.py` runs one episode:
options, prompt, budget, the usage ledger, the result file. `agent/session.py` is the part
specific to the Claude Agent SDK: options, the neutral working directory, message buffer size,
usage extraction. Another MCP-capable client replaces `session.py` and keeps the rest.

## Batches and reports

`evaluation/batch.py` expands a grid into `Job`s, names each episode folder from the job, and
runs the unfinished ones under a total cost cap; it resumes by skipping folders that already have
a `result.json`. `evaluation/report.py` writes the tables and the CSV.
