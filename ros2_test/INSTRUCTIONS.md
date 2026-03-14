# ROS 2 + Drake Cable-Manipulator System — Run Instructions

This folder contains a Drake-based cable manipulator control system with two
execution environments:

| Environment | When to use | Backend |
|---|---|---|
| **macOS (conda)** | Development, simulation, Meshcat viz | `drake_ros_bridge` (pure Python) |
| **Docker (ARM64)** | Real `drake_ros` C++ DDS transport | `bazel run` inside container |

The drake-ROS style wires all ROS I/O as `LeafSystem` blocks inside a single
`DiagramBuilder` — no manual timer callbacks. `drake_ros_compat.py` auto-selects
between the real `drake_ros` C++ package (Docker) and the pure-Python
`drake_ros_bridge` (macOS).

---

## 1. Prerequisites

### 1.1 Conda environment

```bash
conda activate pydrake_ros2
```

> All commands below assume this environment is active.
> See [`docs/INSTALL_PYDRAKE_ROS2.md`](docs/INSTALL_PYDRAKE_ROS2.md) for setup instructions.

### 1.2 Check which backend will be used

```bash
python -c "
import sys; sys.path.insert(0,'ros2_test')
from drake_ros_compat import BACKEND
print('Backend:', BACKEND)
"
```

Expected output on **macOS**:
```
[drake_ros_compat] Backend: drake_ros_bridge (pure-Python fallback)
Backend: drake_ros_bridge
```

Expected output in **Docker/Linux** with compiled `drake_ros`:
```
[drake_ros_compat] Backend: drake_ros (native DDS transport)
Backend: drake_ros
```

---

## 2. Quick Start — One Command

```bash
conda activate pydrake_ros2
bash ros2_test/launch/launch_drakeROS_system.sh
```

This starts **both** the plant node and the controller node. The robot moves
from home (0°, 0°) to (60°, −120°) over 3 seconds, then holds.

---

## 3. Launch Script — Full Reference

```
bash ros2_test/launch/launch_drakeROS_system.sh [Q_GOAL] [DURATION] [MODE] [TIMESTEP]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `Q_GOAL` | `q1,q2` (degrees) | `60,-120` | Target joint angles. `q1` = link1_base (shoulder), `q2` = link2_link1 (elbow). Positive = counter-clockwise. |
| `DURATION` | seconds | `3.0` | How long the min-jerk trajectory takes. Ignored in `hold` mode. |
| `MODE` | `min-jerk` \| `hold` | `min-jerk` | `min-jerk`: smooth polynomial motion. `hold`: stay at q-start. |
| `TIMESTEP` | seconds | `0.002` | Plant simulation timestep. Smaller = more accurate (min ≈ 0.001). |

### Examples

```bash
# Default: move to (60°, −120°) over 3 s
bash ros2_test/launch/launch_drakeROS_system.sh

# Custom goal: move to (30°, −60°) over 2 s
bash ros2_test/launch/launch_drakeROS_system.sh 30,-60 2.0

# Hold joint 1 at 45°, joint 2 at 0° (hold mode, q-goal ignored)
bash ros2_test/launch/launch_drakeROS_system.sh 45,0 3.0 hold

# Fast motion (1 s) with fine timestep (1 ms)
bash ros2_test/launch/launch_drakeROS_system.sh 90,-90 1.0 min-jerk 0.001

# Full elbow extension, slow (5 s)
bash ros2_test/launch/launch_drakeROS_system.sh 0,-90 5.0
```

Press **Ctrl-C** to stop both nodes.

---

## 4. Running Nodes Separately

For debugging, run the plant and controller in separate terminals.

### Terminal 1 — Plant node

```bash
conda activate pydrake_ros2
python ros2_test/nodes/ros2_drakeROS_plant_node.py --mode dynamics
```

**Plant node arguments:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `dynamics` | `dynamics`: full physics sim. `scene-viz`: position-driven, no dynamics. |
| `--timestep` | `0.002` | Simulation timestep (s). |
| `--no-meshcat` | off | Disable Meshcat 3D viewer. |
| `--joint-damping D1 D2` | `0.05 0.05` | Joint damping [Nm·s/rad] for link1, link2. |
| `--joint-stiffness K1 K2` | `2.5 2.5` | Passive spring stiffness [Nm/rad]. |
| `--simulation_sec` | `inf` | Stop after N seconds (default: run forever). |

```bash
# With Meshcat disabled, faster timestep
python ros2_test/nodes/ros2_drakeROS_plant_node.py \
    --mode dynamics --timestep 0.001 --no-meshcat

# Scene-viz mode (visualise joint commands, no controller needed)
python ros2_test/nodes/ros2_drakeROS_plant_node.py --mode scene-viz
```

**Meshcat URL** (printed on startup):
```
Meshcat: http://localhost:7001
```
Open in browser → press ▶ to replay animation.

---

### Terminal 2 — Controller node

```bash
conda activate pydrake_ros2
python ros2_test/nodes/ros2_drakeROS_controller_node.py --mode min-jerk --q-goal 60,-120
```

**Controller node arguments:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `min-jerk` | `min-jerk`: polynomial trajectory. `hold`: stay at q-start. |
| `--q-start` | `0,0` | Start angles in **degrees** (comma-separated). |
| `--q-goal` | `60,-120` | Goal angles in **degrees** (comma-separated). |
| `--duration` | `3.0` | Trajectory duration (s). |
| `--kp` | `10000` | Position gain Kp [s⁻²]. ωₙ = √Kp ≈ 100 rad/s. |
| `--kd` | `400` | Velocity gain Kd [s⁻¹]. ζ = Kd/(2√Kp) ≈ 2 (overdamped). |
| `--tau-max` | `10` | Torque saturation [Nm]. |
| `--joint-damping D1 D2` | `0.05 0.05` | Must match plant node values. |
| `--joint-stiffness K1 K2` | `2.5 2.5` | Must match plant node values. |
| `--simulation_sec` | `inf` | Stop after N seconds. |

```bash
# Hold at (45°, 0°) forever
python ros2_test/nodes/ros2_drakeROS_controller_node.py --mode hold --q-start 45,0

# Aggressive gains, fast trajectory
python ros2_test/nodes/ros2_drakeROS_controller_node.py \
    --kp 20000 --kd 600 --tau-max 20 \
    --mode min-jerk --q-goal 90,-90 --duration 1.5

# Convert radians to degrees manually if needed:
#   0.5 rad = 28.6°,  -0.3 rad = -17.2°
python ros2_test/nodes/ros2_drakeROS_controller_node.py --mode min-jerk --q-goal 28.6,-17.2
```

---

## 5. ROS Topics

Monitor in a third terminal:

```bash
# Joint angles and velocities
ros2 topic echo /joint_states

# Torques sent to the plant
ros2 topic echo /torque_command

# End-effector XYZ position
ros2 topic echo /ee_position

# Simulation clock
ros2 topic echo /clock

# Publishing rate
ros2 topic hz /joint_states
ros2 topic hz /torque_command
```

---

## 6. Scene-Viz Mode with Commander

Visualise joint trajectories without running the controller:

```bash
# Terminal 1: plant in scene-viz mode
python ros2_test/nodes/ros2_drakeROS_plant_node.py --mode scene-viz

# Terminal 2: send sine-wave joint commands
bash ros2_test/launch/launch_ros2_scene_viz.sh sine
# or: hold, step
```

---

## 7. File Reference

| File | Purpose |
|---|---|
| [`drake_ros_bridge.py`](drake_ros_bridge.py) | Pure-Python reimplementation of `drake_ros.core` API (macOS) |
| [`drake_ros_compat.py`](drake_ros_compat.py) | Auto-selects `drake_ros` C++ or `drake_ros_bridge` at import time |
| [`nodes/ros2_drakeROS_plant_node.py`](nodes/ros2_drakeROS_plant_node.py) | Drake-diagram plant — physics sim + ROS pub/sub as LeafSystems |
| [`nodes/ros2_drakeROS_controller_node.py`](nodes/ros2_drakeROS_controller_node.py) | Drake-diagram computed-torque controller |
| [`nodes/ros2_drake_plant_node.py`](nodes/ros2_drake_plant_node.py) | Alternative Drake plant node |
| [`nodes/ros2_computed_torque_controller_node.py`](nodes/ros2_computed_torque_controller_node.py) | Computed-torque controller node |
| [`nodes/ros2_scene_viz_joint_position_commander_node.py`](nodes/ros2_scene_viz_joint_position_commander_node.py) | Scene-viz joint position commander |
| [`launch/launch_drakeROS_system.sh`](launch/launch_drakeROS_system.sh) | Launch plant + controller together (macOS) |
| [`launch/launch_ros2_drake_system.sh`](launch/launch_ros2_drake_system.sh) | Alternative launch for drake system |
| [`launch/launch_ros2_scene_viz.sh`](launch/launch_ros2_scene_viz.sh) | Scene-viz + joint commander |
| [`docs/INSTALL_PYDRAKE_ROS2.md`](docs/INSTALL_PYDRAKE_ROS2.md) | macOS conda environment setup instructions |
| [`docs/INSTALL_ROS2.md`](docs/INSTALL_ROS2.md) | ROS 2 installation instructions |
| [`tools/check_ros2_installation.py`](tools/check_ros2_installation.py) | Diagnostic script to verify ROS 2 install |
| [`tools/drake_ros_pubsub_test_bridge.py`](tools/drake_ros_pubsub_test_bridge.py) | macOS bridge pub/sub test |
| [`drake_ros_docker_test/launch_docker_start.sh`](drake_ros_docker_test/launch_docker_start.sh) | Build Docker image, start container, pre-build Bazel target |
| [`drake_ros_docker_test/launch_docker_run_pubsub_test.sh`](drake_ros_docker_test/launch_docker_run_pubsub_test.sh) | Run pub/sub test inside container via `bazel run` |
| [`drake_ros_docker_test/launch_docker_stop.sh`](drake_ros_docker_test/launch_docker_stop.sh) | Stop the container |
| [`drake_ros_docker_test/drake_ros_pubsub_test.py`](drake_ros_docker_test/drake_ros_pubsub_test.py) | Pub/sub round-trip test using real `drake_ros.core` C++ API |

---

## 8. Docker — Real `drake_ros` C++ (ARM64)

Drake has **no pip wheel for ARM64 Linux**. The `drake_ros.core` C++ extension
also requires Bazel's runfiles environment — direct `python3` execution cannot
work. All scripts must be invoked via `bazel run` using the `ros_py_binary` rule,
exactly like the official `rs_flip_flop` and `multirobot` examples.

### 8.1 Folder structure

```
ros2_test/drake_ros_docker_test/       ← YOUR WORKING FOLDER
    launch_docker_start.sh          # Step 1: build image + start container
    launch_docker_run_pubsub_test.sh # Step 2: run pub/sub test via bazel
    launch_docker_stop.sh           # Step 3: stop container
    drake_ros_pubsub_test.py        # ← REAL FILE — edit this

drake-ros/drake_ros_examples/examples/pubsub_test/   ← Bazel workspace (upstream repo)
    BUILD.bazel                     # Bazel ros_py_binary target (9 lines)
    drake_ros_pubsub_test.py        # ← SYMLINK only — points to the file above
```

**Why two locations?**

Bazel requires every file listed in `srcs = [...]` inside `BUILD.bazel` to exist
physically within the Bazel workspace (`drake_ros_examples/`). It cannot reference
files outside that directory. However, you don't want your scripts living inside
the upstream `drake-ros` repo.

The solution is a **symlink**:

```
ros2_test/drake_ros_docker_test/drake_ros_pubsub_test.py   ← REAL FILE (1 copy)
                          ▲
                          │ symlink (89 bytes, not a copy)
                          │
drake_ros_examples/examples/pubsub_test/drake_ros_pubsub_test.py
```

- **Edit** `ros2_test/drake_ros_docker_test/drake_ros_pubsub_test.py` — this is your code
- The file in `drake_ros_examples/` is not a copy — it is a pointer; changes are visible instantly with no sync needed
- If the symlink is deleted or broken, `bazel run` fails with `missing input file`
- There is only **one real file** on disk; the symlink consumes ~89 bytes

> **Rule:** Always open and edit the file in `ros2_test/drake_ros_docker_test/`.
> Never edit the file shown at the `drake_ros_examples/` path directly — that
> path resolves to the same file via the symlink anyway.

### 8.2 First-time setup

```bash
# Step 1: build Docker image and start container
# Mounts drake-ros/ and ros2_test/ into the container.
# Persists Bazel cache at ~/.cache/drake_ros_bazel_cache/ (survives restarts).
bash ros2_test/drake_ros_docker_test/launch_docker_start.sh
# First run: ~15-30 min (fetches Drake via Bazel). Subsequent runs: seconds.
```

### 8.3 Run the pub/sub test

```bash
bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh [DURATION] [PREFIX] [TIMESTEP] [JOBS]
```

| Argument | Default | Description |
|---|---|---|
| `DURATION` | `15.0` | Simulation duration in seconds |
| `PREFIX` | `/drake_test` | ROS topic prefix |
| `TIMESTEP` | `0.1` | Simulation timestep in seconds |
| `JOBS` | `4` | Bazel parallel build jobs |

```bash
# Default run
bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh

# Custom duration and prefix
bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh 30.0 /my_robot

# Monitor topics in a second terminal while the test runs:
docker exec -it drake_ros_container bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /drake_test/echo
```

### 8.4 Stop the container

```bash
bash ros2_test/drake_ros_docker_test/launch_docker_stop.sh
```

### 8.5 Adding a new script

1. Create `ros2_test/drake_ros_docker_test/my_script.py`
2. Symlink it into the Bazel workspace:
   ```bash
   ln -s /Volumes/Data/Isaac_sim_robotics/ros2_test/drake_ros_docker_test/my_script.py \
         /Volumes/Data/Isaac_sim_robotics/drake-ros/drake_ros_examples/examples/pubsub_test/my_script.py
   ```
3. Add a target to `drake-ros/drake_ros_examples/examples/pubsub_test/BUILD.bazel`:
   ```python
   ros_py_binary(
       name = "my_script_py",
       srcs = ["my_script.py"],
       main = "my_script.py",
       rmw_implementation = "rmw_cyclonedds_cpp",
       visibility = ["//visibility:public"],
       deps = [
           "@drake//bindings/pydrake",
           "@drake_ros//:drake_ros_py",
           "@ros2//:rclpy_py",
           "@ros2//:std_msgs_py",
       ],
   )
   ```
4. Run it:
   ```bash
   docker exec -it drake_ros_container bash -c "
       source /opt/ros/jazzy/setup.bash
       cd /drake-ros/drake_ros_examples
       bazel run //examples/pubsub_test:my_script_py
   "
   ```

### 8.6 Bazel cache

The cache is stored on your Mac at `~/.cache/drake_ros_bazel_cache/` and
mounted into the container. It survives `docker stop` / `docker start` cycles.
Rebuild is only triggered if source files or Bazel dependencies change.

### 8.7 drake-ros git branch

Your additions live on a local branch — upstream is untouched:

```bash
cd drake-ros
git branch          # → user/pubsub-test (current)

# Pull upstream updates and rebase your branch on top:
git fetch origin
git rebase origin/main
```

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'rclpy'` | `conda activate pydrake_ros2` |
| `ModuleNotFoundError: No module named 'robots'` | Run from workspace root: `cd /Volumes/Data/Isaac_sim_robotics` |
| Meshcat page blank | Wait 3–5 s for URDF to load, then refresh browser |
| Controller outputs zero torque | Plant not ready yet — wait 5 s or increase `sleep 5` in `.sh` |
| `RuntimeError: GetMyContextFromRoot` | Use `plant.CreateDefaultContext()` instead |
| Simulation runs slower than real-time | Increase `--timestep` (e.g. `0.005`) or add `--no-meshcat` |
| Joint angles wrong direction | Negate the angle: `-60` instead of `60` |
| Docker: `Container is not running` | Run `launch_docker_start.sh` first |
| Docker: `bazel: command not found` | Bazel is at `/usr/bin/bazel` inside the container — should work automatically |
| Docker: `ERROR: Unable to find package` | Make sure you `cd /drake-ros/drake_ros_examples` before `bazel run` |
| Docker: first `bazel run` is slow | Expected — fetching Drake via Bazel takes 15-30 min. Subsequent runs use `~/.cache/drake_ros_bazel_cache/` |
| Docker: `ModuleNotFoundError: No module named 'pydrake'` | Do not use `python3` directly — always use `bazel run` for drake_ros scripts |
