# ROS 2 Jazzy — Installation Instructions (Source Build)

## System Info
| | |
|---|---|
| **OS** | Ubuntu 24.04 LTS (Noble Numbat) |
| **ROS 2 Distro** | Jazzy Jalisco (LTS) |
| **Build Type** | Source build (from `ros2.repos`) |
| **Architecture** | x86_64 |
| **Python** | 3.12 (system) |
| **Date Installed** | March 14, 2026 |
| **Build Location** | `~/ros2_jazzy/` |
| **Build Time** | ~11 minutes (24-core workstation) |
| **Packages Built** | 366 packages, 0 failures |

---

## Why Source Build?
Ubuntu 24.04 (Noble) requires ROS 2 Jazzy. A source build was used instead of
`apt install ros-jazzy-desktop` because of GPG key conflicts from a previously
attempted binary install. The source build also resolved a **Conda/Python
interception issue** (CMake was picking up Anaconda Python instead of system
Python).

---

## Prerequisites

### 1. Set Locale
```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify UTF-8
```

### 2. Enable Universe Repository
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
```

### 3. Add ROS 2 apt Source
```bash
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

> ⚠️ If you previously added a ROS 2 keyring manually, remove it first to avoid conflicts:
> ```bash
> sudo rm -f /usr/share/keyrings/ros-archive-keyring.gpg
> sudo rm -f /etc/apt/sources.list.d/ros2.list
> sudo apt update
> ```

### 4. Install Dev Tools
```bash
sudo apt update && sudo apt install -y \
  python3-flake8-blind-except \
  python3-flake8-class-newline \
  python3-flake8-deprecated \
  python3-mypy \
  python3-pip \
  python3-pytest \
  python3-pytest-cov \
  python3-pytest-mock \
  python3-pytest-repeat \
  python3-pytest-rerunfailures \
  python3-pytest-runner \
  python3-pytest-timeout \
  ros-dev-tools
```

### 5. Fix Missing catkin_pkg (Conda Conflict)
If you have Anaconda installed, CMake will pick up conda's Python and fail
on `catkin_pkg`. Fix with:
```bash
sudo /usr/bin/pip3 install --break-system-packages catkin_pkg lark empy
```

---

## Build ROS 2 from Source

### 1. Create Workspace & Clone Repos
```bash
mkdir -p ~/ros2_jazzy/src
cd ~/ros2_jazzy
vcs import --input https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos src
```
> This clones ~200 repos. Run in background if needed:
> ```bash
> nohup vcs import --input https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos src \
>   > /tmp/vcs_import.log 2>&1 &
> tail -f /tmp/vcs_import.log
> ```

### 2. Install Dependencies via rosdep
```bash
sudo rosdep init
rosdep update
sudo apt upgrade -y
cd ~/ros2_jazzy
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"
```

### 3. Build (CRITICAL: Exclude Conda from PATH)
**If you have Anaconda installed**, CMake will use conda's Python and fail.
You must build with system Python forced:

```bash
cd ~/ros2_jazzy

# Run in background (recommended — takes 10-120 min depending on CPU)
nohup bash -c '
  export PATH=/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
  colcon build --symlink-install --parallel-workers $(nproc) \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
' > /tmp/colcon_build.log 2>&1 &

# Monitor progress
tail -f /tmp/colcon_build.log

# Check summary when done
grep "Summary:" /tmp/colcon_build.log
grep "^Failed"  /tmp/colcon_build.log   # should be empty
```

---

## Setup Environment

### Source in Current Terminal
```bash
. ~/ros2_jazzy/install/local_setup.bash
```

### Make Permanent (added to ~/.bashrc)
```bash
echo ". ~/ros2_jazzy/install/local_setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Verify Installation

```bash
# Check available packages
ros2 pkg list | wc -l   # should be ~351

# Check topics are active
ros2 topic list
# Expected:
#   /parameter_events
#   /rosout

# Run talker/listener demo
# Terminal 1:
. ~/ros2_jazzy/install/local_setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2:
. ~/ros2_jazzy/install/local_setup.bash
ros2 run demo_nodes_py listener
```

---

## Integration with This Project (Isaac Sim + PyDrake)

### Conda + ROS 2 Coexistence
ROS 2 uses **system Python** (`/usr/bin/python3`), not conda. Always source
ROS 2 **after** activating your conda environment:

```bash
# Isaac Sim bridge (workstation)
conda activate env_isaacsim
source ~/ros2_jazzy/install/local_setup.bash
python ros2_isaac_bridge.py

# PyDrake controller (Jetson / local)
conda activate pydrake
source ~/ros2_jazzy/install/local_setup.bash
python jetson_drake_controller.py
```

### Key ROS 2 Topics for SIL Architecture
| Topic | Direction | Message Type |
|---|---|---|
| `/joint_states` | Isaac Sim → Drake | `sensor_msgs/JointState` |
| `/control_torques` | Drake → Isaac Sim | `std_msgs/Float64MultiArray` |
| `/cart_force_cmd` | Drake → Isaac Sim | `std_msgs/Float64MultiArray` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ros2: command not found` | `source ~/ros2_jazzy/install/local_setup.bash` |
| `No module named 'catkin_pkg'` | CMake using conda Python — use `PATH=/usr/bin:$PATH` and `--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3` |
| GPG key conflict on `apt update` | Remove old keyring: `sudo rm /usr/share/keyrings/ros-archive-keyring.gpg /etc/apt/sources.list.d/ros2.list` |
| `ament_cmake_core` fails | Wipe build cache: `rm -rf ~/ros2_jazzy/build ~/ros2_jazzy/install ~/ros2_jazzy/log` then rebuild |
| Conda breaks ROS 2 env vars | `unset PYTHONPATH PYTHONHOME CONDA_PREFIX` before building |

---

## References
- [ROS 2 Jazzy Docs](https://docs.ros.org/en/jazzy/)
- [ROS 2 Jazzy Source Build Guide](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html)
- [Isaac Sim ROS 2 Bridge](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/)
