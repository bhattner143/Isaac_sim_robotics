# OnShape Robotics Toolkit - Setup and Usage Notes

## 1. Installation

### Install OnShape Robotics Toolkit (Version 0.2.1)

```bash
pip install onshape-robotics-toolkit==0.2.1
```

**Important:** Version 0.2.1 is required as it has the `Robot.from_url()` method that is used in the converter script. Newer versions (like 0.4.0) have breaking API changes.

**Dependencies installed:**
- mujoco>=3.2.7
- optuna>=4.1.0
- pandas>=2.2.3
- plotly>=5.24.1
- pyarrow>=18.0.0
- scikit-learn>=1.6.1
- tqdm>=4.67.0
- And other required packages

---

## 2. OnShape API Setup

### Get API Credentials

1. Go to: https://cad.onshape.com/appstore/dev-portal
2. Create API keys
3. You'll receive:
   - **Access Key** (starts with "on_")
   - **Secret Key** (long alphanumeric string)

### Create `key.env` File

Create a file named `key.env` in your project directory with the following content:

```env
# OnShape API Credentials
# Get these from: https://cad.onshape.com/appstore/dev-portal

ONSHAPE_API=https://cad.onshape.com
ONSHAPE_ACCESS_KEY=your_access_key_here
ONSHAPE_SECRET_KEY=your_secret_key_here
```

**Example (from our setup):**
```env
ONSHAPE_API=https://cad.onshape.com
ONSHAPE_ACCESS_KEY=on_iZkf0qkFogbdbhQ35B6x9
ONSHAPE_SECRET_KEY=KUM35mvdn6i4cAXON6rTce1WTBdIOMEuD2pCISBNprJ8QSMJ
```

**Security Note:** Never commit `key.env` to version control. Add it to `.gitignore`.

---

## 3. Converter Script (`convertor.py`)

### Purpose
Converts an OnShape CAD assembly into a URDF robot model suitable for simulation.

### Script Content

```python
from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.robot import Robot
from onshape_robotics_toolkit.utilities.helpers import save_model_as_json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
key_env_path = os.path.join(script_dir, "key.env")

# Initialize the client
client = Client(env=key_env_path)

# Load the Onshape Assembly
robot = Robot.from_url(
    name="manipulator_2dof",
    url="https://cad.onshape.com/documents/02830e503a1c2626d58e0780/w/62db7934caa01effe231f1a9/e/9d31039ea010abe028749d07",
    client=client,
    max_depth=0,
    use_user_defined_root=True,
)

# Save the Assembly as JSON in the script directory
json_path = os.path.join(script_dir, "manipulator_2dof.json")
save_model_as_json(robot.assembly, json_path)

# Visualize the Assembly Graph
png_path = os.path.join(script_dir, "manipulator_2dof.png")
robot.show_graph(file_name=png_path)

# Save the Robot Object as a URDF File in the script directory
urdf_path = os.path.join(script_dir, "manipulator_2dof.urdf")
robot.save(file_path=urdf_path)
```

### Key Parameters Explained

**`name`**: The name of the robot (used in URDF)

**`url`**: OnShape document URL
- Format: `https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}`
- `did`: Document ID
- `wid`: Workspace ID  
- `eid`: Element ID (Assembly)

**`max_depth`**: Subassembly recursion depth (0 = only top-level parts)

**`use_user_defined_root`**: Whether to use user-defined root link from OnShape
- `True`: Uses fixed part from OnShape as base
- `False`: Auto-selects root

**`matplotlib.use('Agg')`**: Uses non-interactive backend to prevent tkinter threading crashes when generating PNG graph

---

## 4. Generated Outputs

When you run `convertor.py`, it generates the following files:

### 4.1 URDF File (`manipulator_2dof.urdf`)
**Size:** ~3 KB  
**Format:** XML  
**Purpose:** Robot description for simulation

**Contains:**
- Links (robot parts) with visual, collision, and inertial properties
- Joints (connections) with types (revolute, fixed, etc.)
- References to mesh files
- Mass and inertia tensors
- Joint limits and dynamics

**Example structure:**
```xml
<?xml version="1.0" ?>
<robot name="manipulator_2dof">
  <link name="base_mount_manipulator_1">
    <visual>...</visual>
    <collision>...</collision>
    <inertial>...</inertial>
  </link>
  <joint name="dof_jt_mount_link1" type="revolute">
    <parent link="base_mount_manipulator_1"/>
    <child link="link1_1"/>
    <axis xyz="0 0 1"/>
  </joint>
  ...
</robot>
```

### 4.2 JSON File (`manipulator_2dof.json`)
**Size:** ~20 KB  
**Format:** JSON  
**Purpose:** Complete assembly data from OnShape

**Contains:**
- Full assembly hierarchy
- Part metadata
- Mate/joint information
- Occurrences and instances
- Transform matrices
- Part properties

**Use cases:**
- Debugging
- Custom processing
- Analysis of assembly structure

### 4.3 PNG Graph (`manipulator_2dof.png`)
**Size:** ~11 KB  
**Format:** PNG image  
**Purpose:** Visual representation of robot kinematic tree

**Shows:**
- Nodes: Robot links
- Edges: Joints connecting links
- Hierarchy: Parent-child relationships
- Root node identification

### 4.4 Meshes Folder (`meshes/`)
**Contents:** STL mesh files for each link

**Files generated:**
- `base_mount_manipulator_1.stl`
- `link1_1.stl`
- `link2_1.stl`

**Properties:**
- Binary STL format
- Exported from OnShape CAD
- Used for visualization and collision detection
- Referenced in URDF file

### 4.5 Log File (`log_YYYYMMDD_HHMMSS.log`)
**Size:** ~25 KB  
**Format:** Plain text  
**Purpose:** Detailed conversion log

**Contains:**
- API calls and responses
- Mass property fetches
- Graph creation details
- File download progress
- Error messages (if any)

---

## 5. Running the Converter

### Command
```bash
python convertor.py
```

### Expected Output
```
[INFO] Generated joiners - SUBASSEMBLY: _XXXXX_, MATE: _XXXXX_
[INFO] Onshape API initialized with env file: .../key.env
[INFO] Fetching mass properties for part: ...
[INFO] Graph created with 3 nodes and 2 edges with root node: base_mount_manipulator_1
[INFO] Processing root node: base_mount_manipulator_1
[INFO] Creating robot link for base_mount_manipulator_1
[INFO] Processing 2 edges in the graph.
[INFO] Creating robot joint from base_mount_manipulator_1 to link1_1
[INFO] Creating robot link for link1_1
[INFO] Creating robot joint from link1_1 to link2_1
[INFO] Creating robot link for link2_1
[INFO] Starting download for base_mount_manipulator_1.stl
[INFO] Starting download for link1_1.stl
[INFO] Starting download for link2_1.stl
[INFO] Mesh file saved: .../meshes/base_mount_manipulator_1.stl
[INFO] Mesh file saved: .../meshes/link1_1.stl
[INFO] Mesh file saved: .../meshes/link2_1.stl
[INFO] All assets downloaded successfully.
[INFO] Robot model saved to .../manipulator_2dof.urdf
```

---

## 6. Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'onshape_robotics_toolkit'`
**Solution:** Install the package: `pip install onshape-robotics-toolkit==0.2.1`

### Issue: `FileNotFoundError: 'key.env' file not found`
**Solution:** Create `key.env` file with your OnShape API credentials

### Issue: `AttributeError: type object 'Robot' has no attribute 'from_url'`
**Solution:** Wrong version installed. Uninstall and reinstall version 0.2.1:
```bash
pip uninstall onshape-robotics-toolkit
pip install onshape-robotics-toolkit==0.2.1
```

### Issue: Tkinter threading crash when generating PNG
**Solution:** Already fixed with `matplotlib.use('Agg')` in the script

### Issue: Invalid API credentials
**Solution:** 
- Verify credentials in key.env
- Check if keys are still valid on OnShape dev portal
- Ensure no extra spaces in key.env file

---

## 7. File Structure

```
model_using_onshape_robotics_toolkit/
├── convertor.py                    # Main conversion script
├── key.env                         # API credentials (DO NOT COMMIT)
├── check_api_usage.py             # Test script for API
├── config_options_reference.txt   # Reference for onshape-to-robot options
├── manipulator_2dof.urdf          # Generated URDF file
├── manipulator_2dof.json          # Generated assembly JSON
├── manipulator_2dof.png           # Generated kinematic graph
├── log_YYYYMMDD_HHMMSS.log       # Conversion log
├── meshes/                        # Generated mesh files
│   ├── base_mount_manipulator_1.stl
│   ├── link1_1.stl
│   └── link2_1.stl
└── notes/
    └── setup_and_usage.md         # This documentation
```

---

## 8. Next Steps

After generating the URDF and meshes:

1. **Verify URDF**: Check that joint types and limits are correct
2. **Test in Simulator**: Load URDF in Isaac Sim, Gazebo, or PyBullet
3. **Adjust Properties**: Modify mass, inertia, or joint properties if needed
4. **Add Controllers**: Implement control algorithms for your robot
5. **Iterate**: Re-run converter if you modify the OnShape model

---

## 9. Additional Resources

- **OnShape Robotics Toolkit Docs:** https://pypi.org/project/onshape-robotics-toolkit/
- **OnShape API Portal:** https://cad.onshape.com/appstore/dev-portal
- **URDF Format Specification:** http://wiki.ros.org/urdf/XML
- **Alternative Tool (onshape-to-robot):** https://github.com/Rhoban/onshape-to-robot

---

**Last Updated:** January 8, 2026  
**Author:** Generated for 2-DOF Manipulator Project  
**OnShape Robotics Toolkit Version:** 0.2.1
