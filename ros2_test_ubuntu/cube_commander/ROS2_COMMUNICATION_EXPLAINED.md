# ROS 2 Communication — Cube Commander Pipeline

## The Problem This Architecture Solves

PyDrake (the physics/control engine) and Isaac Sim (the 3D simulator) cannot share
a Python process. They each require their own Python environment:

| Component  | Environment                  | Python | Why isolated                          |
|------------|------------------------------|--------|---------------------------------------|
| Drake      | `conda env_isaacsim` (3.11)  | 3.11   | PyDrake wheels are environment-specific |
| ROS 2      | System Python                | 3.12   | ROS 2 Jazzy built against system Python |
| Isaac Sim  | `conda env_isaacsim` (3.11)  | 3.11   | Isaac Sim owns its Python stack; `SimulationApp` must be the very first import |

ROS 2 is the **message bus** that lets these three isolated processes talk to each other.

---

## Full Pipeline Overview

```
Terminal 1 (run_drake_commander.sh)
┌──────────────────────────────────────────────────────────┐
│  conda Python 3.11                                       │
│  drake_logic.py                                          │
│    CubePositionCommander (Drake LeafSystem)              │
│      – increments x by 1cm every second                 │
│      – prints "CUBE_POS:0.0100,0.0000,0.1000" to stdout │
└──────────────────────────┬───────────────────────────────┘
                           │  OS pipe  (bash |)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  System Python 3.12                                      │
│  ros2_publisher.py                                       │
│    CubeCommanderNode (rclpy Node)                        │
│      – reads CUBE_POS lines from stdin                   │
│      – publishes geometry_msgs/Point on /cube_target_pos │
└──────────────────────────┬───────────────────────────────┘
                           │  ROS 2 DDS topic
                           │  /cube_target_pos
                           │  geometry_msgs/Point
                           ▼
Terminal 2 (run_isaac.sh)
┌──────────────────────────────────────────────────────────┐
│  System Python 3.12                                      │
│  ros2_subscriber.py                                      │
│    CubeListenerNode (rclpy Node)                         │
│      – subscribes to /cube_target_pos                    │
│      – prints "CUBE_POS:x,y,z" to stdout                 │
└──────────────────────────┬───────────────────────────────┘
                           │  OS pipe  (bash |)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  conda Python 3.11  (env_isaacsim)                       │
│  isaac_sim.py                                            │
│    StdinPositionReader (background thread)               │
│      – reads CUBE_POS lines from stdin                   │
│    Main simulation loop                                  │
│      – calls cube.set_world_pose(position)               │
└──────────────────────────────────────────────────────────┘
```

---

## Step-by-Step: What Happens When You Run It

### Step 1 — Drake generates a position (`drake_logic.py`)

Drake's `CubePositionCommander` is a **LeafSystem** with a discrete-time state `[x, y, z]`.

```python
# Fires every period_sec (default: 1.0 s)
def _update_position(self, context, discrete_state):
    current[0] += self.step_size   # +0.01 m along X each tick
    discrete_state.set_value(current)

def _publish_position(self, context):
    x, y, z = state[0], state[1], state[2]
    print(f"CUBE_POS:{x:.4f},{y:.4f},{z:.4f}", flush=True)  # → stdout
```

Two things happen on every tick:
1. The state is updated (x increases by 1 cm)
2. The new position is serialised to a single text line and written to **stdout**

The `flush=True` is critical — without it, Python buffers the output and the
downstream process might receive a burst of old messages all at once instead of
one per second.

Lines that start with `#` (debug info written to **stderr**, not stdout) are
invisible to downstream processes and only appear in the terminal for the operator.

---

### Step 2 — OS `|` pipe carries the text to ros2_publisher.py

The bash launch script connects the two Python environments using nothing more than
the shell pipe operator:

```bash
# run_drake_commander.sh
"$CONDA_PYTHON" "$DRAKE_COMMANDER" "$@" \
    | "$SYSTEM_PYTHON" "$ROS2_COMMANDER_NODE"
```

The shell creates an anonymous **FIFO (pipe)**:
- Drake's `stdout` fd is connected to the write end
- ros2_publisher's `stdin` fd is connected to the read end
- The kernel buffers data between them — no files, no sockets, no network

This is the lowest-latency, zero-configuration IPC mechanism available on Linux.

---

### Step 3 — ros2_publisher.py converts text → ROS 2 message

Because `rclpy.spin()` occupies the main thread (it runs the ROS 2 executor
which handles callbacks, timers, and DDS), the stdin reading must happen in a
**background thread** so the two don't block each other.

```python
class CubeCommanderNode(Node):
    def __init__(self):
        self.publisher_ = self.create_publisher(Point, '/cube_target_pos', 10)

        # Background thread reads Drake's stdout and publishes to ROS 2
        self._stdin_thread = threading.Thread(
            target=self._stdin_loop, daemon=True
        )
        self._stdin_thread.start()

    def _stdin_loop(self):
        for line in sys.stdin:               # blocks until Drake writes a line
            if line.startswith('CUBE_POS:'):
                x, y, z = parse(line)
                msg = Point(x=x, y=y, z=z)
                self.publisher_.publish(msg) # thread-safe in rclpy
```

The `daemon=True` flag means the thread is automatically killed when the main
process exits — no manual cleanup needed.

**Queue depth = 10**: the `create_publisher(..., 10)` QoS depth means ROS 2 will
buffer up to 10 unread messages before dropping the oldest. At 1 msg/s this is
10 seconds of buffer — plenty for any subscriber startup delay.

---

### Step 4 — ROS 2 DDS delivers the message to ros2_subscriber.py

ROS 2 uses **DDS (Data Distribution Service)** as its underlying transport. DDS
is a publish-subscribe middleware that handles:

- **Discovery**: nodes announce themselves on the local network (or loopback)
  automatically. No broker or server is needed.
- **Matching**: `ros2_subscriber.py` advertises it wants `/cube_target_pos` →
  DDS automatically routes messages from the publisher.
- **Serialisation**: the `geometry_msgs/Point` struct (`float64 x, y, z`) is
  serialised to CDR binary format for transport, then deserialised back.

On a single machine the DDS transport uses **shared memory** (or loopback UDP),
so there is negligible latency beyond the serialisation cost.

```python
class CubeListenerNode(Node):
    def __init__(self):
        self.subscription_ = self.create_subscription(
            Point,
            '/cube_target_pos',
            self._callback,   # called by rclpy executor when a message arrives
            10,               # QoS depth
        )

    def _callback(self, msg: Point):
        print(f'CUBE_POS:{msg.x:.4f},{msg.y:.4f},{msg.z:.4f}', flush=True)
```

The callback runs on the **main thread** inside `rclpy.spin()`. Each time a
message arrives, `_callback` serialises it back to text and writes to stdout.

---

### Step 5 — Second OS pipe carries text to isaac_sim.py

```bash
# run_isaac.sh
"$SYSTEM_PYTHON" "$ROS2_LISTENER_NODE" \
    | "$CONDA_PYTHON" "$ISAAC_CUBE_TEST"
```

Same pipe mechanism as Step 2. The subscriber's stdout feeds Isaac Sim's stdin.

---

### Step 6 — isaac_sim.py moves the cube

Isaac Sim's main loop is **synchronous** — `world.step(render=True)` must be
called on the main thread and blocks until the physics step is complete. Reading
stdin on the same thread would deadlock, so a **background thread + queue** is
used again:

```python
class StdinPositionReader:
    def __init__(self):
        self._queue = queue.Queue()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        for line in sys.stdin:              # blocks in background
            x, y, z = parse(line)
            self._queue.put(np.array([x, y, z]))

    def get_latest_position(self):
        latest = None
        while not self._queue.empty():     # drain — always get most recent
            latest = self._queue.get_nowait()
        return latest

# Main loop (main thread only)
while simulation_app.is_running():
    world.step(render=True)               # advances physics + renders frame

    new_pos = reader.get_latest_position()
    if new_pos is not None:
        cube.set_world_pose(position=new_pos, orientation=[1,0,0,0])
```

**Queue draining**: if Drake sends faster than Isaac Sim renders, multiple
positions queue up. `get_latest_position()` discards all but the newest — the
cube jumps to the most recent target rather than replaying stale positions.

**VisualCuboid vs DynamicCuboid**: the cube uses `VisualCuboid` (no PhysX
rigid body). `DynamicCuboid` would have gravity applied every `world.step()`
call, causing the cube to fall between `set_world_pose()` calls — the
"teleport up and fall" bug that was observed initially.

---

## Message Format: Why Plain Text?

The inter-process format `CUBE_POS:x,y,z` is intentionally simple text:

| Property      | Detail                                                |
|---------------|-------------------------------------------------------|
| Human-readable | You can `echo "CUBE_POS:0.5,0.0,0.1"` to test Isaac Sim manually |
| No shared library | Drake and Isaac Sim don't need to agree on a binary format |
| Easy to grep/log | `bash run_drake_commander.sh | tee log.txt | python ros2_publisher.py` |
| Comment lines | Lines starting with `#` are printed to stderr by Drake and silently ignored by all consumers |

The cost is minimal: `float64` → 6-digit ASCII → `float64` introduces ~1e-4 m
rounding error (sub-millimetre), which is acceptable for visualisation.

---

## Coordinate Conventions

All three components share the **Z-up, right-handed** convention:

| Axis | Meaning        |
|------|----------------|
| X    | Forward        |
| Y    | Left           |
| Z    | Up             |

Isaac Sim defaults to **Y-up** (OpenGL/USD convention). The Isaac Sim script
explicitly overrides this:

```python
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
```

Without this line, Drake's X-axis increments appear as Z-axis motion in Isaac Sim.

**Quaternion convention**: both Drake and Isaac Sim use **scalar-first `[w, x, y, z]`**.
No conversion is needed. ROS 2 uses **scalar-last `[x, y, z, w]`** — but since
orientation is not sent over ROS 2 in this pipeline (only `geometry_msgs/Point`),
this difference does not apply here. It will matter if you extend the pipeline to
send full poses (`geometry_msgs/Pose`).

---

## Why Two OS Pipes Instead of One ROS 2 Integration?

Isaac Sim and ROS 2 *can* be integrated natively using `isaacsim.ros2.bridge`,
but that requires the full ROS 2 workspace to be built against Isaac Sim's
Python stack — a significant setup burden. The pipe approach works with any
Isaac Sim installation and any ROS 2 distro without any additional packages.

The trade-off is the extra serialise→publish→subscribe→deserialise round-trip
through ROS 2 that sits between the two pipes. For slow-moving commands (1 Hz)
this is negligible.

---

## Thread and Process Map

```
Terminal 1
├── Process: drake_logic.py          (conda Python 3.11)
│     └── Main thread: Drake Simulator loop
└── Process: ros2_publisher.py       (system Python 3.12)
      ├── Main thread: rclpy.spin()  ← ROS 2 executor
      └── Daemon thread: stdin read loop

        [stdin → stdout via OS FIFO]
        [/cube_target_pos via DDS]

Terminal 2
├── Process: ros2_subscriber.py      (system Python 3.12)
│     └── Main thread: rclpy.spin() ← ROS 2 executor, runs _callback
└── Process: isaac_sim.py            (conda Python 3.11)
      ├── Main thread: world.step() loop (must be main thread for GPU rendering)
      └── Daemon thread: stdin read loop → queue
```

---

## Extending the Pipeline

To add a new robot or sensor to this architecture:

1. **New scenario folder** (e.g., `manipulator_commander/`)
2. **New topic** in `common/topics.py`
3. **drake_logic.py** — compute the new commands as a Drake LeafSystem
4. **ros2_publisher.py** — parse Drake stdout, publish to new topic
5. **ros2_subscriber.py** — subscribe to new topic, forward to Isaac Sim stdin
6. **isaac_sim.py** — read positions and apply to Isaac Sim prim
7. **run_commander.sh / run_isaac.sh** — the same two-terminal launch pattern

The ROS 2 topic layer means future components (logging nodes, visualisers,
other simulators) can subscribe to the same topic with zero changes to the
existing pipeline.
