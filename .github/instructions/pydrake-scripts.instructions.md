---
description: "Use when writing or editing PyDrake scripts (script_*drake*.py, script_cup_manipulator*.py, demo_*.py). Covers Drake joint ordering, Linearize, MultibodyPlant, Meshcat, two-system architecture."
applyTo: ["script_*.py", "demo_*.py"]
---

# PyDrake Script Conventions

## Joint Ordering (CRITICAL)
Drake's `GetJointIndices()` returns joints in **URDF parse order**, NOT alphabetical.
For `cup_manipulator.urdf`: joints are `[link2_link1, link1_base]`.

```python
# Drake expects [q2, q1], NOT [q1, q2]
plant.SetPositions(context, model_instance, np.array([q2, q1]))
```

Always verify:
```python
for i, idx in enumerate(plant.GetJointIndices(model_instance)):
    joint = plant.get_joint(idx)
    print(f"[{i}] {joint.name()}")
```

## Linearization
Use Drake's automatic Jacobian, NOT manual formulas:

```python
from pydrake.systems.primitives import Linearize
linearized = Linearize(nonlinear_plant, context,
    input_port_index=plant.get_actuation_input_port().get_index(),
    output_port_index=plant.get_state_output_port().get_index())
A, B, C, D = linearized.A(), linearized.B(), linearized.C(), linearized.D()
```

Set the equilibrium state in `context` BEFORE calling `Linearize()`.

## Two-System Architecture
Scripts build separate plants for physics and control:
```
MultibodyPlant (physics) → [state] → Controller (LeafSystem) → [τ] → Plant
```
This is Drake's standard pattern for custom controllers — not redundant.

## Meshcat
All scripts should print `Meshcat: http://localhost:7000`. Use `meshcat.StartRecording()` / `meshcat.PublishRecording()` for post-sim replay.

## Conda Environment
Run under `conda activate pydrake` (Python 3.11).

## Configuration System
Use `robot_types.py` dataclass configs. Configs are frozen and JSON-serializable:
```python
from robot_types import create_cup_manipulator_config
config = create_cup_manipulator_config(urdf_path=..., joint_angles=(...), damping=(...))
```
