# ISAAC Sim Robotics Projects

This repository contains robotics simulation projects using NVIDIA Isaac Sim for control systems, manipulation, and dynamics research.

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Python Package Installation](#python-package-installation)
  - [Binary Installation](#binary-installation)
- [Running Examples](#running-examples)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository demonstrates various robotics applications using NVIDIA Isaac Sim, including:
- **Cart-Pendulum Systems**: 2DOF control and dynamics
- **Ball-Plate Manipulator**: Vision-based control systems
- **Dynamic Simulations**: Physics-based robot interactions
- **Custom Robot Models**: URDF and USD model integration

---

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX series or better (RTX 2060 or higher recommended)
- **RAM**: 32GB+ recommended (minimum 16GB)
- **Storage**: 50GB+ free space for Isaac Sim installation
- **CPU**: Multi-core processor (8+ cores recommended)

### Software
- **OS**: Ubuntu 20.04/22.04/24.04 (or compatible Linux distribution)
- **NVIDIA Driver**: Latest stable driver (535+ recommended)
- **Python**: 3.11 (managed by conda)
- **Drake**: 1.51+ (robotics toolkit, pip package name: `drake`)
- **Isaac Sim**: 5.1.0 (local build or pip install)

---

## Installation Guide

### Python Package Installation

#### Why Install `pip install isaacsim`?

The `isaacsim` Python package is **essential for programmatic access** to Isaac Sim. It provides:

**Core Components:**
- **Python API Bindings**: Access to `SimulationApp`, `World`, `Scene`, and all Isaac Sim APIs
- **Standalone Script Support**: Run simulations without the GUI (headless mode)
- **Development Tools**: Utilities for robot control, sensors, and environment setup

**Two Operating Modes:**

1. **Headed Mode (GUI/Interactive)**:
   - Full Isaac Sim GUI application with 3D viewport
   - Interactive scene editing and visualization
   - Real-time parameter tuning and debugging
   - Best for: Development, debugging, visualization

2. **Headless Mode (No GUI/Batch)**:
   - Runs simulations without graphical interface
   - Faster execution and lower resource usage
   - Perfect for: Training AI agents, batch simulations, CI/CD pipelines, remote servers
   - Enabled with: `experience=""` or `headless=True` in `SimulationApp`

**Installation Steps:**

```bash
# Create a conda environment
conda create -n env_isaacsim python=3.11 -y

# Activate the environment
conda activate env_isaacsim

# Install the isaacsim Python package
pip install isaacsim

# Install Drake robotics toolkit (NOT 'pydrake' — that's an unrelated package)
pip install drake pyyaml
```

**What This Package Does:**
- Installs Python bindings to interact with Isaac Sim from code
- Enables writing standalone Python scripts that control simulations
- Provides API access for sensors, robots, physics, and rendering
- Works with **both** binary installation and from-source builds

**Important:** This package **requires** a full Isaac Sim installation (binary or built from source) to be present on your system. The `pip install` alone does **not** include the actual simulator - you must also complete the binary installation below.

### Local Build Setup (Required for SimulationApp)

If you built Isaac Sim from source, you **must** source the setup script before running any Isaac Sim scripts. Without it, `SimulationApp` will be `None`:

```bash
# Activate env and source Isaac Sim local build setup
conda activate env_isaacsim
source ~/Documents/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh

# Or use the alias (if added to ~/.bashrc):
isaacsim-env
```

This sets `PYTHONPATH`, `LD_LIBRARY_PATH`, `CARB_APP_PATH` etc. to point at the local build's extensions.

**VS Code Integration:** The `.vscode/settings.json` and `.env.isaacsim` files inject these environment variables automatically for the integrated terminal and debug configurations.

---

### Binary Installation

Download and install the complete Isaac Sim application for full simulator functionality.

#### Step 1: Check System Compatibility

Before installing, **run the compatibility checker** to ensure your system meets all requirements:

After downloading Isaac Sim, run:
```bash
cd ~/isaacsim
./isaac-sim.compatibility_check.sh
```

This verifies:
- NVIDIA driver version
- GPU compatibility
- Operating system requirements
- Required dependencies

#### Step 2: Download Isaac Sim

Download Isaac Sim standalone binary from the official NVIDIA documentation:

**📥 Download Link:** [Isaac Sim 5.1.0 Download Page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)

#### Step 3: Install Isaac Sim

For Linux (x86_64), execute the following commands:

```bash
# Create installation directory
mkdir ~/isaacsim
cd ~/Downloads

# Extract the downloaded archive
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim

# Navigate to installation directory
cd ~/isaacsim

# Run post-installation script
./post_install.sh

# Launch Isaac Sim selector (GUI mode)
./isaac-sim.selector.sh
```

**Final load message example:**
```
2025-03-31 23:15:34 [105,275ms] [Warning] [omni.isaac.range_sensor.ui.menu] omni.isaac.range_sensor.ui.menu has been deprecated
Please update your code accordingly.
[105.5s][ext: isaacsim.robot.wheeled_robots.ui-2.1.5] startup
```

---

## Running Examples

### Set Environment Variable (Recommended)

Set the `ISAAC_SIM_PATH` environment variable to point to your Isaac Sim installation:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export ISAAC_SIM_PATH="$HOME/isaacsim"

# Reload your shell configuration
source ~/.bashrc
```

### Launch Isaac Sim

```bash
# Navigate to Isaac Sim directory
cd ~/isaacsim

# Launch the simulator
./isaac-sim.sh

# Or launch with a Python script
./isaac-sim.sh --python-script /path/to/your/script.py
```

### Run Project Examples

This repository contains various robotics examples:

For easier access, set the `ISAAC_SIM_PATH` environment variable:

```bash
# Add to your ~/.bashrc or ~/.zshrc
This repository contains various robotics examples:

```bash
# Activate your environment
conda activate env_isaacsim

# Run examples from this repository
cd ~/Documents/isaac_sim_robotics

# Test cart-pendulum 2DOF system
python test_cart_pendulum_2dof.py

# Test ball-plate manipulator
python test_ball_plate_manipulator_so101.py

# Test dynamic simulation
python test_cube_ball_dynamic_simulation.py
```

## Project Structure

```
isaac_sim_robotics/
├── example_interactive/          # Interactive examples
├── examples_selected_from_standalone_examples/
├── model/                        # Robot models (URDF, USD)
│   ├── manipulators/
│   ├── plate/
│   └── plate_dips/
├── notes_ball_plate/            # Documentation and notes
├── standalone_examples/         # Standalone example scripts
├── tests/                       # Test scripts
├── test_cart_pendulum_2dof.py  # 2DOF cart-pendulum simulation
├── test_ball_plate_*.py        # Ball-plate control examples
├── README.md                   # This file
└── .gitignore
```

## Troubleshooting

### Isaac Sim Won't Launch

**Solution**: Verify your system compatibility first:
```bash
cd ~/isaacsim
./isaac-sim.compatibility_check.sh
```

### NVIDIA Driver Issues

**Solution**: Update to latest NVIDIA drivers:
```bash
# Check current driver version
nvidia-smi

# Install latest driver (Ubuntu)
sudo apt-get install nvidia-driver-535
sudo reboot
```

### Python Import Errors

**`SimulationApp` is `None` / `TypeError: 'NoneType' object is not callable`:**  
You need to source the Isaac Sim local build setup script first:
```bash
source ~/Documents/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh
# Or use: isaacsim-env
```
This happens when `PYTHONPATH` doesn't include `exts/isaacsim.simulation_app` from the local build.

**`ModuleNotFoundError: No module named 'pydrake.all'`:**  
You have the wrong `pydrake` package installed (a Riot Games API wrapper). Fix:
```bash
pip uninstall pydrake -y
pip install drake  # correct Drake robotics toolkit
pip install pyyaml  # required dependency
```

**`ModuleNotFoundError: No module named 'yaml'`:**  
```bash
pip install pyyaml
```

### Slow Performance

**Solution**: 
- Ensure you're using an NVIDIA RTX GPU
- Check GPU utilization: `nvidia-smi`
- Close other GPU-intensive applications
- Reduce simulation complexity or rendering quality

## Quick Reference Commands

```bash
# Activate environment
conda activate env_isaacsim

# Source Isaac Sim local build (required for Isaac Sim scripts)
source ~/Documents/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh
# Or use alias: isaacsim-env

# Run PyDrake scripts (no source needed)
python script_cup_manipulator_pydrake.py
python script_cup_manipulator_pendulam_with_spring_damper_pydrake.py

# Run Isaac Sim scripts (source required)
python script_cart_pendulum_manipulator_controller.py
python script_cart_pendulum_manipulator_basic_run.py
```

## Additional Resources

- **Official Documentation**: [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- **Download Page**: [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- **API Reference**: Check the installed package documentation
- **Community Forum**: NVIDIA Developer Forums

## Development Workflow

1. **Activate conda environment**: `conda activate isac_sim`
2. **Make changes to source code** in `isaacsim/source/`
3. **Rebuild**: `cd isaacsim && ./build.sh`
4. **Test**: Run the simulator with your changes
5. **Iterate**: Repeat as needed

## License

This project uses ISAAC Sim which requires acceptance of NVIDIA's license terms. See the [LICENSE](isaacsim/LICENSE) file and [Additional Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/) for details.

---

**Last Updated**: December 19, 2025
**ISAAC Sim Version**: Built from latest source (main branch)
**Python Version**: 3.11.14
**Conda Environment**: isac_sim
