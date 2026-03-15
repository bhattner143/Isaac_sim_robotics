# Isaac Sim — Cup Manipulator Tendon Implementation

## Overview

The Isaac Sim implementation mirrors the PyDrake `CupManipulatorTendon` class,
providing the same API surface (`prepare_usd`, `load_urdf`, `set_positions_user_order`, etc.)
so that scene-building code is engine-agnostic.

**Key files:**

| File | Role |
|---|---|
| `robots/cup_manipulator_tendon_isaac.py` | Robot class — config, URDF→USD, articulation, IK |
| `test_cup_manipulator_tendon_scene_viz.py` | Interactive scene-viz script (CLI + render loop) |
| `robot_types.py` | Shared `ManipulatorConfig` / `JointConfig` dataclasses |

---

## Startup Sequence

Isaac Sim has strict ordering requirements. The boot sequence is:

```
1. Pre-parse --render flag       ← before any isaacsim import
2. SimulationApp()               ← MUST be the very first Isaac Sim call
3. Enable extensions (WebRTC)    ← optional, for remote streaming
4. Import everything else        ← argparse, numpy, robot class, pxr, etc.
```

`SimulationApp()` initialises the Omniverse Kit runtime, PhysX, the USD stage
context, and the renderer. **Nothing** from `isaacsim.*`, `omni.*`, or `pxr`
can be imported before it is created.

### Scene setup order (from `test_combined_urdf.py`)

```
prepare_usd()          ← URDF→USD conversion + bake colors (before World)
World()                ← creates USD stage, physics scene
load_urdf()            ← add_reference_to_stage (after World)
weld_base_to_world()
set_joint_properties()
add_joint_actuators()
world.reset()          ← initialises PhysX (MUST be before Articulation)
initialize_state()     ← creates Articulation wrapper
set_initial_positions()
warm-up steps          ← propagate positions to renderer
```

---

## Threading in Isaac Sim

Think of Isaac Sim like a **game engine** with one control room.

Only **one person** is allowed to touch the controls:

- start physics
- move robot joints
- update the scene
- render frames

That "person" is the **main thread** — the thread that created:

```python
SimulationApp(...)
```

If another thread also starts touching those controls, the engine can crash.

---

### Why threading is needed at all

In your interactive script, you want two things at once:

1. keep the simulator rendering smoothly
2. wait for the user to type commands like:

```text
j -50 60
e 0.2 0.1
p
q
```

The problem is `input()` is **blocking**:

```python
cmd = input(">> ")   # Python stops here until the user types something
```

If you do that on the main thread, Isaac Sim freezes.

So the fix is:

- run `input()` in a **background thread**
- send typed commands to the **main thread** via a queue
- let the main thread execute all Isaac Sim actions

---

### Bad example: what not to do

```python
import threading

def input_thread_fn(world, robot):
    while True:
        cmd = input(">> ")
        if cmd.startswith("j "):
            robot.set_positions_user_order([0.1, 0.2])  # BAD: Isaac Sim API
            world.step(render=True)                      # BAD: Isaac Sim API

# main thread
while simulation_app.is_running():
    world.step(render=True)
    simulation_app.update()
```

Now both threads are using Isaac Sim simultaneously:

- **main thread** calls `world.step()`
- **background thread** also calls `world.step()` and robot APIs

That creates a race. Possible results: random bugs, deadlocks, mutex assertion
failures, process abort.

The actual crash you see:

```
ASSERTION FAILED: RecursiveSharedMutex.h(357):
void carb::thread::recursive_shared_mutex::unlock()():
Assertion (e.second != 0) failed.
terminate called without an active exception
[exit code 134]  ← SIGABRT
```

---

### Good example: queue-based design

#### Step 1 — background thread only reads input

```python
import threading
import queue

cmd_queue = queue.Queue()

def input_thread_fn():
    while True:
        cmd = input(">> ")
        cmd_queue.put(cmd)   # safe — queue.Queue is thread-safe
```

This thread does only two things:

- waits for keyboard input
- puts text into a thread-safe queue

It does **not** call Isaac Sim.

#### Step 2 — main thread processes commands

```python
def process_pending_commands(robot):
    while True:
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            break

        parts = cmd.strip().split()
        if not parts:
            continue

        if parts[0] == "j" and len(parts) == 3:
            q1 = np.deg2rad(float(parts[1]))
            q2 = np.deg2rad(float(parts[2]))
            robot.set_positions_user_order([q1, q2])  # SAFE: main thread

        elif parts[0] == "p":
            print(robot.get_positions_user_order())    # SAFE: main thread
```

#### Step 3 — main render loop

```python
while simulation_app.is_running():
    process_pending_commands(robot)   # handle input commands
    world.step(render=True)           # physics + render
    simulation_app.update()           # Kit update
```

Now there is only **one thread** touching Isaac Sim: the main thread.

---

### What happens when the user types a command

Suppose the user types `j -50 60`:

```text
User types command
       │
       ▼
Background thread
input(">> ")  →  gets "j -50 60"
cmd_queue.put("j -50 60")
       │
       ▼
Main loop — next frame
process_pending_commands()
  cmd_queue.get_nowait()  →  "j -50 60"
  robot.set_positions_user_order([...])   ← safe, main thread
world.step(render=True)
simulation_app.update()
```

The background thread **reports** the command; the main thread **executes** it.

---

### Cup manipulator — command timeline

#### `j -30 45` (set joints)

| Thread | Action |
|---|---|
| Background | `cmd_queue.put("j -30 45")` |
| Main | `robot.set_positions_user_order([deg2rad(-30), deg2rad(45)])` |

#### `e 0.15 0.20` (IK move)

| Thread | Action |
|---|---|
| Background | `cmd_queue.put("e 0.15 0.20")` |
| Main | solve IK, `robot.set_positions_user_order(q_sol)`, update EE marker (USD) |

Even if IK math is safe in another thread, the moment you apply it to Isaac Sim
it must be on the main thread.

---

### What is safe in another thread?

| Operation | Safe in background? |
|---|---|
| `input(">> ")` | ✓ Yes |
| `queue.Queue.put()` | ✓ Yes |
| Pure Python / numpy math | ✓ Yes |
| `world.step()` | ✗ Main thread only |
| `simulation_app.update()` | ✗ Main thread only |
| `articulation.set_joint_positions()` | ✗ Main thread only |
| Any USD prim read/write | ✗ Main thread only |

---

### Real-world analogy

- The **main thread** is the only driver of a car.
- The **background thread** is a passenger reading messages aloud.

Safe: passenger says "turn left" → driver turns left.

Unsafe: passenger grabs the steering wheel while the driver is driving.

That is what bad threading in Isaac Sim looks like.

---

### Complete reusable template

```python
import threading
import queue
import numpy as np

cmd_queue = queue.Queue()
quit_flag = False

def input_reader():
    while True:
        line = input(">> ")
        cmd_queue.put(line)
        if line.strip() == "q":
            break

def process_pending_commands(robot):
    global quit_flag
    while True:
        try:
            line = cmd_queue.get_nowait()
        except queue.Empty:
            return

        tokens = line.strip().split()
        if not tokens:
            continue
        cmd = tokens[0]

        if cmd == "j" and len(tokens) == 3:
            q1 = np.deg2rad(float(tokens[1]))
            q2 = np.deg2rad(float(tokens[2]))
            robot.set_positions_user_order([q1, q2])

        elif cmd == "e" and len(tokens) == 3:
            x, y = float(tokens[1]), float(tokens[2])
            q_sol = robot.solve_ik(x, y)           # pure math — fine anywhere
            robot.set_positions_user_order(q_sol)  # Isaac Sim — main thread
            robot.update_ee_marker(x, y)           # USD update — main thread

        elif cmd == "p":
            print(robot.get_positions_user_order())

        elif cmd == "q":
            quit_flag = True

        else:
            print("Unknown command")

# Startup
threading.Thread(target=input_reader, daemon=True).start()

while simulation_app.is_running() and not quit_flag:
    process_pending_commands(robot)
    world.step(render=True)
    simulation_app.update()
```

---

### Final takeaway

> **Thread the input, not the simulator.**

Use a background thread only to **collect commands**, then execute all Isaac Sim
actions on the **main thread** through a `queue.Queue`.
This is the **producer–consumer pattern** — the background thread produces
commands, the main thread consumes and applies them.

---

## Adding Colors to the Robot (URDF → USD Color Baking)

### The problem

The `URDFParseAndImportFile` importer converts URDF geometry to USD but
**silently drops all `<material><color>` definitions**. Every mesh appears
as default grey.

### Why the naïve fix doesn't work

The importer creates a multi-file USD structure:

```
manipulator_cable_obj.usd              ← main file, defaultPrim = /manipulator_cable
  └─ sublayers:
       configuration/manipulator_cable_obj_base.usd     ← visual meshes HERE
       configuration/manipulator_cable_obj_robot.usd
       configuration/manipulator_cable_obj_physics.usd
       configuration/manipulator_cable_obj_sensor.usd
```

Visual mesh prims live inside sublayer files at paths like:
```
/visuals/base_mate/tutup_base_1/World/mesh
/visuals/base_mate/pulley_idler/World/mesh
```

When `add_reference_to_stage` loads the main USD, only the `defaultPrim`
hierarchy (`/manipulator_cable`) appears in the live stage. Traversing
`omni.usd.get_context().get_stage()` finds **zero** Mesh prims from the
sublayers — so iterating the live stage and setting colors does nothing.

### The working approach: modify sublayer files on disk

`apply_urdf_colors()` runs **after** `import_urdf_to_usd()` but **before**
`add_reference_to_stage()`:

```python
prepare_usd():
    import_urdf_to_usd(urdf, usd)   # 1. create USD files
    apply_urdf_colors()              # 2. bake colors into files on disk
    # ...later...
load_urdf():
    add_reference_to_stage(usd)      # 3. stage reads already-colored files
```

#### Step-by-step

1. **Parse URDF** — `xml.etree.ElementTree` extracts `<visual><material><color rgba="..."/>`
   for each mesh, keyed by the OBJ filename stem (e.g. `"base"`, `"tutup_base_1"`).

2. **Find sublayer files** — glob `*.usd` from both the main directory and the
   `configuration/` subdirectory.

3. **Open each sublayer directly** — `Usd.Stage.Open(sublayer_path)` gives a
   standalone stage where the Mesh prims are visible to `Traverse()`.

4. **Match part names** — for each `Mesh` prim, walk path components in reverse
   and match against the color map. Three fallbacks handle importer quirks:
   - **Exact match**: `tutup_base_1` → found in color_map
   - **`part_` prefix**: the importer prefixes invalid USD names (e.g.
     `623zz` → `part_623zz`); strip the prefix and retry
   - **Numeric suffix**: Isaac Sim appends `_0`, `_1` for duplicate prims;
     strip the trailing `_N` and retry

5. **Set `primvars:displayColor`** — via `UsdGeom.Gprim`:
   ```python
   gprim = UsdGeom.Gprim(prim)
   gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(r, g, b)]))
   ```

6. **Save** — `sub_stage.Save()` writes modified colors back to the sublayer
   `.usd` file on disk.

### Color map (from URDF)

| Part | RGB | Visual |
|---|---|---|
| `base` | (0.0, 0.6, 0.8) | Blue |
| `tutup_base_1` | (0.698, 0.0, 0.0) | Red |
| `link1` | (0.325, 0.529, 0.753) | Steel blue |
| `link2_tendon` | (0.765, 0.545, 0.804) | Purple |
| `cup` | (0.980, 0.714, 0.004) | Yellow |
| `link1_base_pulley` | (0.867, 0.322, 0.157) | Orange-red |
| `outer_big_stopper` | (0.0, 0.0, 0.0) | Black |
| `623zz` | (0.957, 0.957, 0.957) | Silver |
| `shaft` | (0.655, 0.824, 0.576) | Green |
| `simple_ball` | (0.616, 0.812, 0.929) | Light blue |
| pulleys / gearboxes | (0.749, 0.749, 0.749) | Grey |

---

## Render Modes

| Flag | Behaviour |
|---|---|
| `--render native` | Local Isaac Sim window (default) |
| `--render websocket` | Headless + WebRTC on port 49100 — view with NVIDIA Omniverse Streaming Client |
| `--render headless` | No display (CI / testing) |

`--render` is pre-parsed from `sys.argv` before `SimulationApp` so that
`headless=True/False` is set at construction time. WebRTC streaming is
enabled via `enable_extension("omni.kit.livestream.webrtc")` after
`SimulationApp` creation.

---

## Interactive Commands

| Command | Description |
|---|---|
| `j <q1> <q2>` | Set joint angles in degrees |
| `e <x> <y>` | Move end-effector via analytical IK |
| `p` | Print current joint state + EE position |
| `q` | Quit |

All commands are processed on the main thread via the command queue
(see Threading section above).
