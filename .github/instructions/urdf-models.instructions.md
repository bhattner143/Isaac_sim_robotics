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

## MHP (manipulator_hybrid_planar_fusion) Cable Routing

### Spool Position Adjustment
When working with cable routing visualization in `test_mhp_cable_routing_viz.py`, the spool position is **manually adjusted from the URDF physical position** for cable routing purposes.

**Why two positions exist:**
| Aspect | URDF Physical Position | Cable Routing Position |
|--------|------------------------|------------------------|
| **Location** | `[0.225, 0, 0.1268]` m | `[-0.0595, 1.76602e-13, 0.0125]` m |
| **Frame** | upper_arm (origin anchor) | upper_arm (cable exit point) |
| **Purpose** | 3D Meshcat visualization | 2D cable routing visualization |
| **Representation** | Spool drum in housing | Where cable emerges into arm channel |
| **Source** | URDF line ~391 | URDF ball marker line ~664 |

**How Drake renders it:**
- URDF specifies an **attachment origin** at [0.225, 0, 0.1268]
- The mesh object `mhp_arm_00_elbow_spool_v2.obj` contains **both the spool AND cable geometry**
- The mesh extends backward and to the side from this anchor point
- Result: Visual spool appears on the left/back (correct position in Meshcat), but the URDF position is "weird" because it anchors the entire assembly, not just the spool drum

**Cable routing adjustment logic:**
- Extracted the `ball_cable_spool_upper_arm_start` marker position from URDF: `[-0.0795, 0, 0.0155]`
- This marker was placed by the Onshape CAD designer at the cable exit point
- Manually adjusted slightly to `[-0.0595, 1.76602e-13, 0.0125]` for better visualization alignment
- Used this position for 2D visualization because it represents where routing actually begins, not the deep anchor point inside the housing

**When editing cable positions:**
- URDF positions = where mesh assembly is anchored (may contain multiple geometry elements)
- Visualization positions = where features visually appear or where routing matters
- These can differ significantly when mesh objects contain composite geometry
