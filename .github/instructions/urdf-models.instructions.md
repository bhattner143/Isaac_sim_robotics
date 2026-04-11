---
description: "Use when working on URDF models, Onshape conversion, mesh assets, STL/OBJ conversion, model_using_onshape_to_robot pipeline."
applyTo: "model_using_onshape_to_robot/**"
---

# URDF Model Pipeline

## Conversion Steps (Onshape → URDF → simulation-ready)

1. **Onshape → URDF**: `step1_convert_from_onshape.sh <robot_name>`
   - Requires `.env` with Onshape API keys
2. **STL → OBJ**: `python step2_convert_stl_to_obj.py <robot_name>`
   - Drake and Isaac Sim require OBJ meshes (not STL)
3. **Update URDF refs**: `python step3_urdf_stl_to_obj.py <robot_name>`
   - Rewrites `<mesh filename="...stl">` → `...obj`
4. **Fix inertia** (optional): `step4_urdf_fix_inertia_for_tendon_manip_v1.py`

## URDF Gotchas
- Joints need explicit actuators in Drake: `plant.AddJointActuator("tau_joint", joint)`
- Isaac Sim requires `<limit effort="..." velocity="..."/>` on each joint
- The `_obj.urdf` variant has OBJ mesh references (use this one)
- Check URDF loading: print all joints via `plant.GetJointIndices()` to verify ordering

## Directory Layout
```
model_using_onshape_to_robot/
  manipulator_cable/
    manipulator_cable.urdf          # Original (STL refs)
    manipulator_cable_obj.urdf      # Converted (OBJ refs) ← use this
    meshes/                         # STL + OBJ mesh files
```

## Mesh Formats
- **STL**: Onshape export default, no color, larger
- **OBJ**: Required by Drake and Isaac Sim, supports materials
- Never commit large mesh files without checking `.gitignore`
