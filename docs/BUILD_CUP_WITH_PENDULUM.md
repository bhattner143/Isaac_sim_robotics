# Building Cup Manipulator with 3D Pendulum

## Overview

This guide shows how the 3D pendulum is attached to the cup manipulator in Drake, including plant setup and Meshcat visualization.

## Key Components

### 1. **Pendulum3D Class** (lines 716-900)
- Located in `script_cup_manipulator_controller_ofc.py`
- Implements a 2-DOF gimbal pendulum using spherical coordinates
- Attaches to any parent body in the plant

### 2. **build_cup_manipulator_with_pendulum() Function** (lines 1847-2060)
- Complete workflow for building the system
- Handles URDF loading, pendulum attachment, actuators, and visualization

## How the Pendulum is Attached

### Step-by-Step Process:

```python
# 1. Get the parent body (link2 of cup manipulator)
link2_body = plant.GetBodyByName("link2", cup_manipulator.model_instance)

# 2. Create pendulum with configuration
pendulum_config = create_pendulum_config(
    mass=0.5,
    length=0.2,
    radius=0.05,
    damping=0.1,
    attachment_point=(-1.2545, 0.0, -0.188125),  # Point on link2
    initial_pitch=0.0,
    initial_roll=180.0,
)

# 3. Create pendulum instance
pendulum = Pendulum3D(pendulum_config)

# 4. Attach to parent body
pendulum.attach_to_body(plant, link2_body, cup_manipulator.model_instance)
```

### What `attach_to_body()` Does:

1. **Creates Pivot Frame** - Fixed offset frame on parent body at attachment point
2. **Creates Gimbal Body** - Intermediate body for first rotation (pitch/Y-axis)
3. **Adds Pitch Joint** - RevoluteJoint rotating about Y-axis
4. **Creates Pendulum Body** - Main pendulum mass with proper inertia
5. **Adds Roll Joint** - RevoluteJoint rotating about X-axis
6. **Adds Geometry** - Visual (rod + ball) and collision (ball only)

## Complete Build Workflow

### Using the Helper Function:

```python
from script_cup_manipulator_controller_ofc import build_cup_manipulator_with_pendulum

# Build complete system with visualization
diagram, simulator, plant, scene_graph, meshcat, cup_manip, pendulum = \
    build_cup_manipulator_with_pendulum(
        enable_pendulum=True,
        enable_visualization=True,
        initial_joint_angles=(0.0, 0.0),        # Cup manipulator angles
        initial_pendulum_pitch=0.0,              # Hanging down
        initial_pendulum_roll=180.0,             # Default
    )

# Simulate
simulator.AdvanceTo(5.0)  # Run for 5 seconds
```

## System Structure

After building, the system has:

```
MultibodyPlant
├── Cup Manipulator (from URDF)
│   ├── link1_base (actuated revolute joint)
│   ├── link2_link1 (actuated revolute joint)
│   └── link2 (end effector body)
│       
└── 3D Pendulum (programmatic)
    ├── Pivot frame (on link2)
    ├── Gimbal1 body (intermediate)
    ├── pendulum_pitch (passive revolute joint, Y-axis)
    ├── Pendulum ball body
    └── pendulum_roll (passive revolute joint, X-axis)
```

**Total DOF:** 4
- 2 actuated (manipulator joints)
- 2 passive (pendulum pitch & roll)

## Meshcat Visualization Setup

```python
# 1. Start Meshcat
meshcat = StartMeshcat()

# 2. Configure parameters
visualizer_params = MeshcatVisualizerParams()
visualizer_params.show_hydroelastic = True
visualizer_params.show_contact_forces = True

# 3. Add to builder
MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, visualizer_params
)

# 4. Get URL
print(f"Meshcat URL: {meshcat.web_url()}")
```

## Demonstrations

Run the demo script to see different configurations:

```bash
python demo_cup_with_pendulum.py
```

### Available Demos:

1. **Static Visualization** - Build and view the robot
2. **Gravity Simulation** - Let pendulum swing (no control)
3. **Custom Configuration** - Set specific initial pose
4. **Without Pendulum** - Build cup manipulator only

## Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Pendulum3D class | script_cup_manipulator_controller_ofc.py | 716-900 |
| build function | script_cup_manipulator_controller_ofc.py | 1847-2060 |
| CupManipulator class | script_cup_manipulator_controller_ofc.py | 618-680 |
| Demo script | demo_cup_with_pendulum.py | Full file |

## Example: Manual Build

If you want more control, you can build manually:

```python
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser

# Create builder
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

# Load manipulator
config = create_cup_manipulator_config(...)
cup_manipulator = CupManipulator(config)
parser = Parser(plant)
cup_manipulator.load_urdf_to_plant(plant, parser)

# Add pendulum
pendulum_config = create_pendulum_config(...)
pendulum = Pendulum3D(pendulum_config)
link2_body = plant.GetBodyByName("link2", cup_manipulator.model_instance)
pendulum.attach_to_body(plant, link2_body, cup_manipulator.model_instance)

# Add actuators
for joint_name in ["link1_base", "link2_link1"]:
    joint = plant.GetJointByName(joint_name, cup_manipulator.model_instance)
    plant.AddJointActuator(joint_name, joint)

# Set gravity and finalize
plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
plant.Finalize()

# Add visualization
meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Build diagram
diagram = builder.Build()
simulator = Simulator(diagram)
```

## Next Steps

- Add controllers using the linearized system
- Implement task-space control
- Add trajectory tracking
- Visualize forces and frames
