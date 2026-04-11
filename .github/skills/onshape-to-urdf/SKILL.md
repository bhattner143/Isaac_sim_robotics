---
name: onshape-to-urdf
description: "Convert Onshape CAD models to simulation-ready URDF. Use when: importing a new robot from Onshape, fixing mesh references, converting STL to OBJ, updating inertia properties, or debugging URDF loading failures."
argument-hint: "The robot name (e.g. 'manipulator_cable') or describe the conversion issue"
---

# Onshape → URDF Conversion Pipeline

## When to Use
- Importing a new robot design from Onshape
- Re-exporting after CAD changes
- Fixing mesh/inertia issues in URDF

## Prerequisites
- Onshape API keys in `model_using_onshape_to_robot/.env`:
  ```
  ONSHAPE_ACCESS_KEY=...
  ONSHAPE_SECRET_KEY=...
  ```
- `onshape-to-robot` Python package installed

## Full Pipeline

```bash
cd model_using_onshape_to_robot

# Step 1: Export URDF from Onshape (requires API keys)
bash step1_convert_from_onshape.sh <robot_name>
# Example: bash step1_convert_from_onshape.sh manipulator_cable

# Step 2: Convert STL meshes → OBJ (Drake/Isaac Sim need OBJ)
python step2_convert_stl_to_obj.py <robot_name>

# Step 3: Update URDF to reference OBJ instead of STL
python step3_urdf_stl_to_obj.py <robot_name>

# Step 4 (optional): Fix inertia values for tendon manipulator
python step4_urdf_fix_inertia_for_tendon_manip_v1.py
```

## Output Structure
```
model_using_onshape_to_robot/<robot_name>/
  <robot_name>.urdf              # Original (STL refs)
  <robot_name>_obj.urdf          # ← Use this one (OBJ refs)
  meshes/
    *.stl                        # Original Onshape meshes
    *.obj                        # Converted for simulation
```

## Verification

### Drake
```python
from pydrake.all import MultibodyPlant, Parser
plant = MultibodyPlant(0.001)
parser = Parser(plant)
model = parser.AddModels("manipulator_cable_obj.urdf")[0]
plant.Finalize()
for idx in plant.GetJointIndices(model):
    print(plant.get_joint(idx).name())
```

### Isaac Sim
URDF → USD conversion happens automatically via `CupManipulatorTendonIsaac.prepare_usd()`.

## Common Issues

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: *.stl` | Run step 2+3 to convert to OBJ |
| Missing mesh colors in Drake | OBJ files need matching `.mtl` files |
| Inertia warnings | Run step 4 or manually fix `<inertial>` in URDF |
| Joint limits missing | Add `<limit effort="100" velocity="10"/>` manually |
| Wrong joint axis | Check `<axis xyz="0 0 1"/>` matches CAD orientation |

## Available Robots
- `manipulator_cable` — 2-DOF cable-driven tendon manipulator
- `manipulator-hybrid-planar` — Hybrid planar manipulator
- `cup_manipulator` — Original cup manipulator
- `cart_pendulum` — Cart-pendulum system
- `ball` — Ball for pendulum attachment
