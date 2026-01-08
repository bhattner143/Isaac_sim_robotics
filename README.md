# ISAAC Sim Robotics Projects

This repository contains robotics simulation projects using NVIDIA Isaac Sim for control systems, manipulation, and dynamics research.

## Table of Contents
- [System Requirements](#system-requirements)
- [Isaac Sim Installation](#isaac-sim-installation)
- [Environment Setup](#environment-setup)
- [Running Examples](#running-examples)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX series or better (RTX 2060 or higher recommended)
- **RAM**: 32GB+ recommended (minimum 16GB)
- **Storage**: 50GB+ free space for Isaac Sim installation
- **CPU**: Multi-core processor (8+ cores recommended)

### Software
- **OS**: Ubuntu 20.04/22.04/24.04 (or compatible Linux distribution)
- **NVIDIA Driver**: Latest stable driver (535+ recommended)
- **Python**: 3.10 or 3.11 (managed by conda)

## Isaac Sim Installation

### Step 1: Check System Compatibility

Before installing Isaac Sim, **run the compatibility checker** to ensure your system meets all requirements:

**From binary installation (Workstation or Open-Source repository setup):**

1. Install/build Isaac Sim according to the target setup workflow.
2. Run the `isaac-sim.compatibility_check.sh` script on Linux, or the `isaac-sim.compatibility_check.bat` script on Windows.

This will verify:
- NVIDIA driver version
- GPU compatibility
- Operating system requirements
- Required dependencies

### Step 2: Download Isaac Sim

Download Isaac Sim standalone binary from the official NVIDIA documentation:

**📥 Download Link:** [Isaac Sim 5.1.0 Download Page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)

### Step 3: Install Isaac Sim (Binary Installation)

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

# Launch Isaac Sim selector
./isaac-sim.selector.sh
```

**Final load message example:**
```
2025-03-31 23:15:34 [105,275ms] [Warning] [omni.isaac.range_sensor.ui.menu] omni.isaac.range_sensor.ui.menu has been deprecated
Please update your code accordingly.
[105.5s][ext: isaacsim.robot.wheeled_robots.ui-2.1.5] startup
```

### Step 4: Create Conda Environment for Development

Set up a Python environment for your robotics projects:

```bash
# Create conda environment
conda create -n env_isaacsim python=3.11 -y

# Activate the environment
conda activate env_isaacsim

# Install the isaacsim Python package (for API access)
pip install isaacsim
```

**Note:** The `pip install isaacsim` installs Python bindings and API interfaces to interact with Isaac Sim programmatically from your scripts.

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

**Solution**: Ensure the environment is properly activated and isaacsim package is installed:
```bash
conda activate env_isaacsim
pip install isaacsim
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

# Launch Isaac Sim
cd ~/isaacsim
./isaac-sim.sh

# Run with a Python script
./isaac-sim.sh --python-script /path/to/script.py

# Run examples from this repository
cd ~/Documents/isaac_sim_robotics
python test_cart_pendulum_2dof.py
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
