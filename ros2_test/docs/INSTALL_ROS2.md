# Install ROS 2 Humble on macOS (Apple Silicon)

> **Scope: ROS 2 only (standalone)**
> Use this guide if you want a lightweight ROS 2 environment without Drake.
> If you need **Drake + ROS 2 together** (for the cable-manipulator scripts),
> use [`INSTALL_PYDRAKE_ROS2.md`](INSTALL_PYDRAKE_ROS2.md) instead.

Verified on: macOS 26.3.1, Apple Silicon (arm64), 2026-03-13

## Prerequisites

- macOS on Apple Silicon (arm64)
- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) installed

## Step 1: Create a Conda Environment

```bash
conda create -n ros2env python=3.11 -y
```

## Step 2: Activate the Environment

```bash
conda activate ros2env
```

## Step 3: Install Mamba (faster package solver)

```bash
conda install -y -c conda-forge mamba
```

## Step 4: Install ROS 2 Humble Desktop

```bash
mamba install -y -c robostack-staging -c conda-forge ros-humble-desktop
```

This installs ~272 packages including `demo_nodes_cpp`, `demo_nodes_py`, `turtlesim`, `rviz2`, `rqt`, etc.

## Step 5: Install Build Tools

```bash
mamba install -y -c robostack-staging -c conda-forge ros-humble-ros2bag compilers cmake pkg-config make ninja
pip install colcon-common-extensions
```

## Step 5b: Remove Mamba from ros2env After Installation

Mamba is only needed for installing packages, not at runtime. Leaving it in ros2env causes a `libmamba` version conflict when running `conda deactivate`. Remove it:

```bash
conda remove -n ros2env mamba libmamba --force -y
```

## Step 6: Set Up Environment Variables

Every time you open a new terminal, run:

```bash
conda activate ros2env
export AMENT_PREFIX_PATH=$CONDA_PREFIX
```

### Persistent Setup (recommended)

Create an activation script so all env vars are set automatically every time you run `conda activate ros2env`:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/ros2_env.sh << 'EOF'
export AMENT_PREFIX_PATH=$CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Welcome Dipankar to ROS2"
echo "ROS 2 installation location --> $CONDA_PREFIX"
EOF
```

Create a matching deactivation script to clean up when you leave the env:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/ros2_env.sh << 'EOF'
unset AMENT_PREFIX_PATH
EOF
```

After this, `conda activate ros2env` will print your welcome message, set `AMENT_PREFIX_PATH`, and fix the Python PATH automatically.

## Step 7: Verify Installation

```bash
# Check ros2 CLI
ros2 --help

# List installed packages
ros2 pkg list | wc -l

# Check key packages
ros2 pkg list | grep -E "demo_nodes|turtlesim"
```

Expected output:
```
272       (or similar count)
demo_nodes_cpp
demo_nodes_cpp_native
demo_nodes_py
turtlesim
```

## Step 8: Test with a Demo

### Terminal 1 — Run a talker

```bash
conda activate ros2env
ros2 run demo_nodes_py talker
```

### Terminal 2 — Listen to the topic

```bash
conda activate ros2env
ros2 topic echo /chatter
```

You should see messages like:
```
data: 'Hello World: 0'
---
data: 'Hello World: 1'
---
```

### Quick Python Test

```python
import rclpy
from std_msgs.msg import String

rclpy.init()
node = rclpy.create_node('test_node')
pub = node.create_publisher(String, 'test_topic', 10)
msg = String()
msg.data = 'Hello ROS 2!'
pub.publish(msg)
node.destroy_node()
rclpy.shutdown()
print('ROS 2 works!')
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`

The wrong Python is being used (system Python 3.9 instead of conda env's 3.11). Verify:

```bash
which python3
# Should be: /opt/anaconda3/envs/ros2env/bin/python3
python3 --version
# Should be: Python 3.11.x
```

If it shows `/usr/bin/python3`, the conda env's `bin/` isn't first in PATH. Fix:

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
```

This is handled automatically if you set up the persistent activation script in Step 6.

### `AMENT_PREFIX_PATH is not set or empty`

Run manually or set up the persistent activation script in Step 6:

```bash
export AMENT_PREFIX_PATH=$CONDA_PREFIX
```

### `Error while loading conda entry point: conda-content-trust` / `anaconda-auth`

**Symptom:** Errors containing `Symbol not found: _EVP_DigestSqueeze` when running `conda activate ros2env`.

**Cause:** The base Anaconda installation has OpenSSL 3.5.x, but ros2env installs OpenSSL 3.6+ (via conda-forge). The `EVP_DigestSqueeze` function was added in OpenSSL 3.6, so conda's own plugins fail to load when they pick up ros2env's newer `cryptography` library.

**Fix:** Update the base environment's OpenSSL to 3.6+:

```bash
conda deactivate
conda install -n base -c conda-forge "openssl>=3.6" -y
```

### `Error while loading conda entry point: conda-libmamba-solver` (on deactivate)

**Symptom:** Error containing `Symbol not found: __ZN5mamba10validation...` when running `conda deactivate`.

**Cause:** The base env's `libmambapy` finds ros2env's `libmamba` (a different version installed by mamba) and tries to link against it, causing a symbol mismatch.

**Fix:** Remove mamba and libmamba from ros2env — they are only needed at install time:

```bash
conda remove -n ros2env mamba libmamba --force -y
```

You can still use `mamba` from the base env for future installs by specifying `-n ros2env`:

```bash
conda run -n base mamba install -n ros2env -c robostack-staging <package>
```

## Distro Info

- **ROS 2 Distro**: Humble Hawksbill
- **Install method**: RoboStack (conda-forge)
- **Python**: 3.11
- **Package**: `ros-humble-desktop` 0.10.0
