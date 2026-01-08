# ISAAC Sim Setup and Build Instructions

This repository contains the ISAAC Sim simulator built from source for robotics simulation and development.

## Table of Contents
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Building ISAAC Sim](#building-isaac-sim)
- [Running ISAAC Sim](#running-isaac-sim)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX series or better (RTX 2060 or higher recommended)
- **RAM**: 32GB+ recommended (minimum 16GB)
- **Storage**: 50GB+ free space for build artifacts and cache
- **CPU**: Multi-core processor (8+ cores recommended)

### Software
- **OS**: Ubuntu 20.04/22.04/24.04 (or compatible Linux distribution)
- **GCC/G++**: Version 11 (higher versions not yet supported - see note below)
- **Git**: With Git LFS support
- **Python**: 3.10 or 3.11 (managed by conda)
- **NVIDIA Driver**: Latest stable driver (535+ recommended)
- **build-essential**: Required for make and other build tools

## Environment Setup

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt-get update

# Install build essentials (required for make and build tools)
sudo apt-get install -y build-essential

# Install Git LFS (required for large files)
sudo apt-get install -y git-lfs

# Verify GCC/G++ version (should be 11)
gcc --version
g++ --version
```

**⚠️ Important - GCC/G++ Version 11 Required:**

ISAAC Sim currently supports **GCC/G++ 11 only**. Higher versions are not yet supported. If you have a different version, install GCC 11:

```bash
# Install GCC/G++ 11
sudo apt-get install gcc-11 g++-11

# Set GCC 11 as default using update-alternatives
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

# Verify the version (should show 11.x.x)
gcc --version
g++ --version
```

**Note:** If you want to skip the compiler version check (at your own risk), you can add `--skip-compiler-version-check` to the build command.

### 2. Create Conda Environment

The project uses a conda environment with Python 3.11:

```bash
# Create conda environment
conda create -n isac_sim python=3.11 -y

# Activate the environment
conda activate isac_sim

# Install the isaacsim launcher package (stub/launcher only)
pip install isaacsim
```
### 2. Run the Build Script

The build process will download dependencies and compile ISAAC Sim:

```bash
# Make the build script executable (if needed)
chmod +x build.sh

# Run the build
./build.sh
```

**Alternative: Skip Compiler Version Check** (if you have GCC > 11):
```bash
./build.sh --skip-compiler-version-check
```
⚠️ Use at your own risk - unsupported build environments may cause issues.

**Important Notes:**
- On first run, you'll be prompted to accept the NVIDIA Omniverse License Agreement (type "yes" to accept)
- The build process takes **1-2 hours** depending on your system and internet speed
- Downloads several GB of dependencies from NVIDIA servers
- Creates build artifacts in `_build/` directory
- **Note:** The build script was executed outside VS Code terminal in this setup
```

### 2. Run the Build Script

The build process will download dependencies and compile ISAAC Sim:

```bash
# Make the build script executable (if needed)
chmod +x build.sh

# Run the build
./build.sh
```
### 3. Monitor Build Progress

You can check the build status using the provided helper script:

```bash
bash ~/Documents/isac_sim_pydrake/check_build_status.sh
```

Or manually check:

```bash
# Check if build directory exists
ls -lh _build/linux-x86_64/release/

# Monitor packman cache size (build caches packages here)
du -sh ~/.cache/packman

# Check for running build processes
ps aux | grep build.sh
```

**Build Process Overview:**
1. **License Acceptance**: You'll be prompted to accept NVIDIA's terms (type "yes")
2. **Package Manager Setup**: Installs packman and downloads Python environments
3. **Dependency Download**: Fetches Kit kernel, physics engine, USD libraries (~5-15 GB)
4. **Compilation**: Builds the actual simulator binaries
5. **Finalization**: Creates the executable in `_build/linux-x86_64/release/`

**Note:** Initial build is run in a system terminal (outside VS Code) for better stability with long-running processes.-lh _build/linux-x86_64/release/

# Monitor packman cache size
du -sh ~/.cache/packman

# Check for running build processes
ps aux | grep build.sh
```

## Running ISAAC Sim

### After Successful Build

Once the build completes successfully, navigate to the binary directory and run the executable:

```bash
cd ~/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release
./isaac-sim.sh
```

### Set Environment Variable (Optional but Recommended)

For easier access, set the `ISAAC_SIM_PATH` environment variable:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export ISAAC_SIM_PATH="$HOME/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release"

# Reload your shell configuration
source ~/.bashrc
```

Then you can run from anywhere:

```bash
$ISAAC_SIM_PATH/isaac-sim.sh
```

### Run Example Simulations

```bash
cd ~/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release

# Launch with Python script
./isaac-sim.sh --python-script /path/to/your/script.py

# Or use the test script
./isaac-sim.sh --python-script ~/Documents/isac_sim_pydrake/test_isac_sim.py
```

## Testing the Installation

Use the provided test script to verify the installation:

```bash
# Activate conda environment
conda activate isac_sim

# Run the installation checker
python ~/Documents/isac_sim_pydrake/test_isac_sim.py
```

This script will:
- Check if the isaacsim package is installed
- Verify if SimulationApp is available
- Provide detailed installation instructions if needed
- Optionally test launching the simulator

## Project Structure

```
isac_sim_pydrake/
├── isaacsim/                    # ISAAC Sim source code (cloned repo)
│   ├── build.sh                 # Main build script
│   ├── source/                  # Source code
│   ├── deps/                    # Dependencies
│   ├── _build/                  # Build output (created after build)
│   │   └── linux-x86_64/
│   │       └── release/
│   │           └── isaac-sim.sh # Executable to run ISAAC Sim
│   └── ...
├── test_isac_sim.py            # Installation checker and test script
├── check_build_status.sh       # Build status monitoring script
└── README.md                   # This file
```

## Troubleshooting

### Build Fails with "No such file or directory"

**Solution**: Ensure you're in the correct directory and the build script has execute permissions:
### GCC/G++ Version Issues

**Problem**: ISAAC Sim requires exactly GCC/G++ 11. Higher versions are not yet supported.

**Solution for older versions**: Update to GCC 11:
```bash
sudo apt-get install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
```

**Solution for newer versions**: Either install GCC 11 as above, or use the skip flag:
```bash
./build.sh --skip-compiler-version-check
```
⚠️ Warning: Using unsupported compiler versions may lead to build failures or runtime issues.
**Solution**: Install Git LFS:
```bash
sudo apt-get install git-lfs
git lfs install
cd ~/Documents/isac_sim_pydrake/isaacsim
git lfs pull
```

### GCC/G++ Version Too Old

**Solution**: Update to GCC 11 or higher:
```bash
sudo apt-get install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

### Out of Disk Space During Build

**Solution**: 
- Free up space or use a different drive
- The build requires ~50GB including cache and build artifacts
- Check cache size: `du -sh ~/.cache/packman`

### NVIDIA Driver Issues

**Solution**: Update to latest NVIDIA drivers:
```bash
# Check current driver version
nvidia-smi

# Install latest driver (Ubuntu)
sudo apt-get install nvidia-driver-535
sudo reboot
```

### Slow Download Speeds

**Solution**: 
- The build downloads several GB from NVIDIA servers
- Be patient - first build takes longest
- Subsequent builds reuse cached packages in `~/.cache/packman`

### Python Version Mismatch

**Solution**: 
- ISAAC Sim requires Python 3.10 or 3.11
## Quick Reference Commands

```bash
# Activate environment
conda activate isac_sim

# Build ISAAC Sim (run in system terminal for best results)
cd ~/Documents/isac_sim_pydrake/isaacsim
./build.sh

# Build with compiler version check skip (if using GCC > 11)
./build.sh --skip-compiler-version-check

# Check build status
bash ~/Documents/isac_sim_pydrake/check_build_status.sh

# Run ISAAC Sim
cd ~/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release
./isaac-sim.sh

# Test installation
python ~/Documents/isac_sim_pydrake/test_isac_sim.py
```

**💡 Tip:** For long-running builds, use a system terminal (Ctrl+Alt+T) rather than VS Code's integrated terminal to avoid interruptions.bash
# Activate environment
conda activate isac_sim

# Build ISAAC Sim
cd ~/Documents/isac_sim_pydrake/isaacsim
./build.sh

# Check build status
bash ~/Documents/isac_sim_pydrake/check_build_status.sh

# Run ISAAC Sim
cd ~/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release
./isaac-sim.sh

# Test installation
python ~/Documents/isac_sim_pydrake/test_isac_sim.py
```

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
