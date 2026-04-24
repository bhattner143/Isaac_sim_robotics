---
description: "Use when working on exosuit cable routing, co-contraction stiffness, exo spring visualisation, or dual-groove elbow pulley geometry. Covers both Method A (offset pulleys) and Method B (centred elbow pulley)."
applyTo: ["cable/*exo*", "cable/*elbow*", "*exosuit*", "*exo_springs*"]
---

# Exosuit Cable Conventions

## Two Methods

### Method A: Offset Elbow Pulleys (`cable/cable_with_exo_springs.py`)
- **URDF**: `manipulator_cable_exo_springs`
- Two separate elbow pulleys: `ExoElbowPulleyRight` (-Y), `ExoElbowPulleyLeft` (+Y)
- r_elb = 32 mm, offset d_off = 103 mm from joint axis
- Cables stay on own Y-side (no crossing)
- `ExoRouting` enum: `CW_CCW` or `CCW_CW`
- Variable moment arm: l_c'(q₂) ≈ r_elb at small angles
- Springs provide passive decoupling (no encoder needed for zero force)
- k_eff_A = 2·k_exo·[l_c'(q₂)]² ≈ 0.41 N·m/rad

### Method B: Centred Elbow Pulley (`cable/cable_with_exo_springs_elbow_follow.py`)
- **URDF**: `manipulator_cable_exo_springs_elbow_follow`
- One shared pulley: `ExoElbowPulleyBig`, centred on joint axis at x = 0.334804
- r_cp = 30 mm (from mesh), **not** drive pulley radius
- Two grooves: upper (Z = 0.23855, world 286.55 mm), lower (Z = 0.23555, world 283.55 mm), 3 mm apart
- Cables cross Y-sides at elbow via **internal tangent** (mimics drive cable routing)
- Constant moment arm: l_c' = r_cp (exact for all q₂)
- **Encoder tracking essential**: motor must feed cable at r_cp · q̇₂ to maintain zero force
- k_eff_B = 2·k_exo·r_cp² ≈ 0.91 N·m/rad (with r_cp = 47.75 mm)

## Class Inventory (Method B)

| Class | Body | vis_xyz Z | World Z | Role |
|-------|------|-----------|---------|------|
| `ExoStartRight` | q1 (-Y) | 0.23855 | 286.55 mm | Motor anchor (orange) |
| `ExoStartLeft` | q1 (+Y) | 0.23555 | 283.55 mm | Motor anchor (purple) |
| `ExoLink1PulleyRight` | q1 (-Y) | 0.23855 | 286.55 mm | L1 idler, groove r=0.035 |
| `ExoLink1PulleyLeft` | q1 (+Y) | 0.23555 | 283.55 mm | L1 idler, groove r=0.035 |
| `ExoElbowPulleyBig` | q1 (centre) | 0.23705 (midpoint) | — | Shared dual-groove pulley |
| `ExoEndPlusY` | q2 (+Y) | z=0.0165 | 286.55 mm | Link-2 anchor (orange) |
| `ExoEndMinusY` | q2 (-Y) | z=0.0135 | 283.55 mm | Link-2 anchor (purple) |

## Z-Plane Conventions (Method B)
- **Orange cable** (right): entirely on **upper groove** Z = 0.23855 (286.55 mm)
  - StartRight → Link1PulleyRight → ElbowBig upper → EndPlusY
- **Purple cable** (left): entirely on **lower groove** Z = 0.23555 (283.55 mm)
  - StartLeft → Link1PulleyLeft → ElbowBig lower → EndMinusY
- ElbowPulleyBig has `Z_UPPER` / `Z_LOWER` class attributes for per-groove centre

## Routing Direction (Method B)
Cables **cross Y-sides** at the shared elbow pulley:
```
Orange: Start -Y → L1 -Y → [internal tangent, b=-1] → Elbow upper → [external, b=+1] → End +Y
Purple: Start +Y → L1 +Y → [internal tangent, b=+1] → Elbow lower → [external, b=-1] → End -Y
```
Branch signs: `b_link1_elbow = -1 / +1` (internal), `b_elbow_end = +1 / -1` (external opposite).

## ExoCableRig.compute_tangents() (Method B)
- Uses `plant.CalcRelativeTransform()` to FK End points from q2 body → q1 body frame
- Computes elbow→end tangent points using FK'd end positions
- Updates `FixedBodyPoint` wrappers (which store tangent points in native body frames)
- `wrap_arcs` tuple: 6 elements — `(center, radius, start_angle, end_angle, color, center_override)`
  - `center_override` (6th element) gives per-groove Z for elbow arcs (drawn at groove Z, not midpoint)

## Visualisation Functions
- `draw_exo_cables(meshcat, rig)` — 3D Meshcat lines, spheres, wrap arcs
- `visualize_exo_cable_routing_top_view(rig)` — 2D matplotlib, annotated tangent geometry
- Both functions handle the 6-tuple `wrap_arcs` format with `center_override`

## Key Parameters
| Parameter | Method A | Method B |
|-----------|----------|----------|
| L1 pulley tangent_radius | 0.035 m | 0.035 m |
| L1 pulley draw radius | 0.041 m | 0.041 m |
| Elbow pulley radius | 0.032 m (per side) | ~0.048 m (shared, r_cp from mesh) |
| Spring fraction | 0.12 | 0.12 |
| Spring coils | 8 | 8 |
| Spring amplitude | 0.005 m | 0.005 m |

## Common Pitfalls
1. **Z-plane mismatch**: End points are on q2 body; verify world Z after FK matches cable's groove
2. **Wrap arc centre**: Use `center_override` for elbow arcs, not `pulley.centroid` (midpoint Z)
3. **Internal vs external tangent**: Crossing at elbow = internal; exit to end = external (opposite branch signs)
4. **r_cp value**: Use mesh-measured 30 mm for visualisation, not drive pulley 47.75 mm
5. **FK for cross-body tangents**: End anchors are on q2; must FK to q1 frame before computing tangent to elbow pulley
