# Install PyDrake + ROS 2 Humble on macOS (Apple Silicon)

> **Scope: Drake + ROS 2 combined** (`pydrake_ros2` environment)
> Use this guide to run the cable-manipulator scripts in `ros2_test/`.
> If you only need **ROS 2 without Drake**, see [`INSTALL_ROS2.md`](INSTALL_ROS2.md) instead.

Verified on: macOS 26.3.1, Apple Silicon (arm64), 2026-03-13

This guide creates a unified `pydrake_ros2` conda environment with both **Drake** (robotics simulation) and **ROS 2 Humble** (middleware) in the same Python 3.11 environment.

> **Why Python 3.11?**
> - Drake 1.40.0 is the latest version with a Python 3.11 wheel for macOS arm64.
> - ROS 2 Humble requires Python 3.10–3.12.
> - Python 3.11 is the best common version for both.

## Prerequisites

- macOS on Apple Silicon (arm64)
- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) installed

---

## Step 1: Create Conda Environment (Python 3.11)

```bash
conda create -n pydrake_ros2 python=3.11 -y -c conda-forge
```

## Step 2: Install Mamba (faster solver)

```bash
conda install -n pydrake_ros2 -y -c conda-forge mamba
```

## Step 3: Install ROS 2 Humble Desktop

```bash
/opt/anaconda3/envs/pydrake_ros2/bin/mamba install -n pydrake_ros2 -y \
  -c robostack-staging -c conda-forge ros-humble-desktop
```

This installs ~272 packages including `demo_nodes_cpp`, `demo_nodes_py`, `turtlesim`, `rviz2`, `rqt`, etc.

## Step 4: Install Build Tools

```bash
/opt/anaconda3/envs/pydrake_ros2/bin/mamba install -n pydrake_ros2 -y \
  -c robostack-staging -c conda-forge \
  ros-humble-ros2bag compilers cmake pkg-config make ninja

/opt/anaconda3/envs/pydrake_ros2/bin/pip install colcon-common-extensions
```

## Step 5: Remove Mamba (avoid runtime conflicts)

```bash
conda remove -n pydrake_ros2 mamba libmamba --force -y
```

## Step 6: Install Scientific / PyDrake Conda Packages

```bash
conda install -n pydrake_ros2 -y -c conda-forge \
  matplotlib scipy ipython qhull
```

## Step 7: Install Drake and PyDrake Pip Packages

```bash
/opt/anaconda3/envs/pydrake_ros2/bin/pip install \
  drake==1.40.0 \
  certifi charset-normalizer colorama commentjson idna jinja2 \
  lark-parser markupsafe mosek mpld3 numpy-stl onshape-to-robot \
  pydot python-dotenv python-utils pyyaml requests termcolor \
  transforms3d trimesh urllib3
```

## Step 8: Set Up Environment Variables (Persistent)

```bash
mkdir -p /opt/anaconda3/envs/pydrake_ros2/etc/conda/activate.d
cat > /opt/anaconda3/envs/pydrake_ros2/etc/conda/activate.d/ros2_env.sh << 'EOF'
export AMENT_PREFIX_PATH=$CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Welcome Dipankar to ROS2 + PyDrake (pydrake_ros2)"
echo "ROS 2 installation location --> $CONDA_PREFIX"
EOF

mkdir -p /opt/anaconda3/envs/pydrake_ros2/etc/conda/deactivate.d
cat > /opt/anaconda3/envs/pydrake_ros2/etc/conda/deactivate.d/ros2_env.sh << 'EOF'
unset AMENT_PREFIX_PATH
EOF
```

---

## Step 9: Verify Installation

```bash
conda activate pydrake_ros2

# Check pydrake
python -c "import pydrake.all; print('pydrake OK')"

# Check rclpy
python -c "import rclpy; print('rclpy OK')"

# Check ros2 CLI
ros2 pkg list | wc -l

# Check demo nodes
ros2 pkg list | grep -E "demo_nodes|turtlesim"
```

Expected output:
```
pydrake OK
rclpy OK
272       (or similar count)
demo_nodes_cpp
demo_nodes_cpp_native
demo_nodes_py
turtlesim
```

---

## Step 10: Test Pub/Sub with Drake + ROS 2

### Terminal 1 — Subscriber

```bash
conda activate pydrake_ros2
python /Volumes/Data/ros2/subscriber_node.py
```

### Terminal 2 — Publisher

```bash
conda activate pydrake_ros2
python /Volumes/Data/ros2/publisher_node.py
```

### Combined Drake + ROS 2 quick test

```python
import pydrake.all
import rclpy
from std_msgs.msg import String

print("pydrake:", pydrake.__file__)

rclpy.init()
node = rclpy.create_node('drake_ros2_test')
pub = node.create_publisher(String, 'test_topic', 10)
msg = String()
msg.data = 'Hello from Drake + ROS 2!'
pub.publish(msg)
node.destroy_node()
rclpy.shutdown()
print('Drake + ROS 2 works!')
```

---

## Package Summary

| Package      | Version | Source         |
|--------------|---------|----------------|
| Python       | 3.11    | conda-forge    |
| drake        | 1.40.0  | pip (PyPI)     |
| ROS 2 Humble | 0.10.0  | robostack-staging |
| matplotlib   | latest  | conda-forge    |
| scipy        | latest  | conda-forge    |
| ipython      | latest  | conda-forge    |
| numpy        | latest  | conda-forge    |

## Troubleshooting

### `ImportError: No module named 'rclpy'`
Make sure `AMENT_PREFIX_PATH` is set:
```bash
export AMENT_PREFIX_PATH=$CONDA_PREFIX
```
Or re-run `conda activate pydrake_ros2` to trigger the activation script.

### Drake version mismatch
Drake 1.40.0 is the latest wheel available for Python 3.11 on macOS arm64.
Newer drake versions (1.41+) only ship Python 3.12+ wheels.
To use the latest drake, create a separate environment with Python 3.12 and install ROS 2 Jazzy instead.

### `conda deactivate` conflict with mamba
Mamba is removed after installation (Step 5) to avoid the `libmamba` version conflict during `conda deactivate`.
