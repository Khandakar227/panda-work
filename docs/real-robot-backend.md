# A real-robot backend

Nothing in this repo talks to hardware. This page says what a backend would have to provide,
which parts of the harness would stop existing, and a reasonable first step. It is a map, not
an implementation.

## The boundary

The physical world enters the harness through one class, `World` in
`src/pandawork/world/world.py`. Everything above it (skills, tools, tasks, the agent bridge, the
recorder) calls these methods:

| method | what it does today | what hardware must provide |
|---|---|---|
| `timestep`, `clock` | MuJoCo's step size and a simulated clock that advances only in `step` | real time, read from the robot's controller; the clock is wall time and nothing pauses |
| `command(q, gripper)` | set the servo targets for the next step | send joint position targets and a gripper command at the control rate |
| `step(n)` | advance physics `n` steps, publish periodic sensors, run step hooks | wait `n` control periods while the controller runs; publish the readings that arrived |
| `encoders()` | the arm's joint angles as the robot reports them | the controller's joint state |
| `set_arm(q)` | teleport the arm (tests, task start postures) | a slow move to `q` or nothing |
| `capture(name)`, `render(camera, mode, size)` | an offscreen camera image: RGB, depth or segmentation | frames from a real camera by name; depth from an RGB-D sensor; segmentation from a detector or not at all |
| `object_pose(name)`, `set_object_pose(name, ...)` | read or place an object in the simulator | not available (see below) |
| `fork()` | an exact independent twin at this instant | not available |
| `reset(seed)` | restore the home keyframe and reseed the noise streams | move home; the seed means nothing |

The sensors in `world/sensors.py` read the plant directly; on hardware each becomes a
subscriber to a driver topic that publishes the same `Message` on the bus. The bus itself
(`world/bus.py`) is synchronous and in-process; a hardware backend feeds it from a background
thread and the harness consumes it inside `step`.

The self-model (`core/ik.py`) needs no change: it is already a MuJoCo model of the nominal
robot, fed with whatever `encoders()` returns, and real controllers work the same way.

## What stops existing

- **Privileged tools.** `get_object_poses`, `get_fixture_states` and the true fingertip are
  simulator truth. The privileged tier has no hardware meaning; only realistic and vision-only do.
- **Faults as plant patches.** The plant is the robot. A fault experiment on hardware is a
  deliberate miscalibration of the controller or the camera, and `Fault.read_encoders` is the one
  hook that still applies (offset the reported angles before they reach the self-model).
- **`simulate` and `fork`.** The twin can still be a MuJoCo model of the believed world, but it
  must be built from estimated object poses, not copied from the plant. Today the tool is simply
  absent in vision-only; on hardware it would be absent everywhere or rebuilt on estimates.
- **Exact replay.** The log stays useful (every call, result and image is recorded), but
  re-executing it cannot reproduce the world. `recording/replay.py` would report a mismatch and
  `scripts/episodes.py video` would need to render from recorded frames instead of a replay.
- **Truth logging and the behavioural measures.** `truth.jsonl` records the true hand and
  object poses around each call. On hardware those come from a motion-capture system or a
  calibrated external camera, or the hand-error and compensation measures in
  `docs/analysis.md` are unavailable.
- **Task evaluation.** `Task.evaluate` reads the simulator (cube on pad, drawer opening, button
  travel). Each task needs a physical check: a scale, a switch in the button, a camera and a
  marker, or a human.

What survives unchanged: the tool definitions and their schemas, the tier gate, budgets, the
turn rule, the system prompt, the MCP bridge, the recorder's event log, the batch runner, the
judge and the leak check.

## A first step

A `World` subclass that keeps MuJoCo for the self-model and the viewer but proxies the plant:

1. `command` publishes joint targets to the robot driver; `step(n)` sleeps `n` control periods
   and drains the driver's joint-state and camera queues into the bus.
2. `encoders` returns the last joint state; the joint, wrist F/T and touch sensors subscribe to
   the driver instead of reading `data`.
3. `render` returns the last frame of the named camera, resized; depth from the RGB-D stream;
   segmentation raises or is replaced by a detector that returns the same per-object masks
   `skills/perception.py` expects.
4. `object_pose` raises; `Task.reset` for the chosen task is replaced by a prompt to a human to
   lay out the scene; `Task.evaluate` by a physical check or a human verdict entered at the end.

ROS 2 is the obvious transport for 1 to 3 (joint states, `sensor_msgs/Image`, a position
controller), and the bus's topic layout was chosen to make that mapping direct. With that in
place, `scripts/run_agent.py pick_place --tier realistic` would run against the arm with the same
prompt, tools and recording as in simulation.

Safety is the backend's job, not the harness's: joint and workspace limits, a force threshold
that stops motion, and a dead-man switch belong between `command` and the robot. The motion
skills already stop on unexpected contact, but they trust the wrist force reading they are
given.
