---
name: exosuit-cable-routing
description: "Work on exosuit cable routing, co-contraction stiffness, dual-groove pulley geometry, or exo spring visualisation. Use when: modifying cable routing, adding exo pulleys, debugging Z-plane alignment, implementing co-contraction, or comparing Method A vs Method B."
argument-hint: "Describe what you need (e.g., 'add a new exo endpoint' or 'debug Z-plane mismatch in elbow wrap arc')"
---

# Exosuit Cable Routing

## When to Use
- Creating or modifying exo cable routing visualisation
- Debugging Z-plane mismatches between pulleys and endpoints
- Adding new exo pulley classes or anchor points
- Implementing co-contraction stiffness control
- Comparing Method A (offset) vs Method B (centred) designs

## Two Architectural Methods

### Method A: Offset Elbow Pulleys
**File**: `cable/cable_with_exo_springs.py`
**URDF**: `manipulator_cable_exo_springs`

- Two separate elbow pulleys (Right -Y, Left +Y), each r_elb = 32 mm
- Offset d_off = 103 mm from joint axis
- Cables stay on own Y-side (no crossing)
- Variable moment arm l_c'(q₂) — nonlinear
- Passive decoupling: springs at rest → zero force (no encoder required)
- `ExoRouting` enum controls wrapping direction: `CW_CCW` or `CCW_CW`

### Method B: Centred Elbow Pulley (Preferred)
**File**: `cable/cable_with_exo_springs_elbow_follow.py`
**URDF**: `manipulator_cable_exo_springs_elbow_follow`

- One shared pulley (`ExoElbowPulleyBig`), centred on joint axis
- Dual grooves: upper Z = 0.23855 (286.55 mm), lower Z = 0.23555 (283.55 mm)
- Cables cross Y-sides at elbow via internal tangent
- Constant moment arm l_c' = r_cp (exact for all q₂)
- Encoder tracking essential for zero-force free rotation

## Adding a New Exo Pulley or Anchor

1. **Define the class** inheriting from appropriate base:
   ```python
   class ExoNewAnchor:
       body_name = "link2_tendon"      # Drake body name
       vis_xyz = np.array([x, y, z])   # body-frame position
       radius = 0.0                     # 0.0 for anchors
       tangent_radius = 0.0
       color = [1.0, 0.5, 0.0, 1.0]   # RGBA
   ```

2. **Assign Z-plane**: Match the cable's groove:
   - Orange cable → Z = 0.23855 (upper groove, body frame of q1)
   - Purple cable → Z = 0.23555 (lower groove, body frame of q1)
   - For q2 body: subtract FK chain offset (0.222049) to get body-local z

3. **Add to routing in ExoCableRig.__init__**:
   - Insert into the ordered waypoint list
   - Set branch signs (internal=-1/+1 for crossing, external=+1/-1 for non-crossing)

4. **Update compute_tangents()**:
   - If anchor is on q2 body: FK it to q1 frame via `plant.CalcRelativeTransform()`
   - Compute tangent to/from adjacent pulleys using `compute_tangent()` helper
   - Store tangent points back in `FixedBodyPoint` wrappers

5. **Update wrap_arcs** if wrapping around new pulley:
   - Use 6-tuple: `(center, radius, start_angle, end_angle, color, center_override)`
   - `center_override` = per-groove Z centre (not midpoint)

## Z-Plane Verification

After modifying routing, verify all waypoints are on the correct Z-plane:
```python
# Trace each cable's world-Z at every waypoint
for name, xyz_world in cable_waypoints:
    print(f"{name}: Z = {xyz_world[2]*1000:.2f} mm")
# Orange: all should be ~286.55 mm
# Purple: all should be ~283.55 mm
```

## Tangent Geometry Key Functions

- `PulleyBase.compute_tangent(c1, r1, c2, r2, branch, kind)` — returns tangent points
  - `kind="external"`: cables on same side of pulleys
  - `kind="internal"`: cables cross between pulleys
  - `branch`: +1 or -1 selects which tangent line

- `ExoCableRig.compute_tangents(plant, plant_context, manipulator)` — full routing solve
  - FK transforms end points from q2 → q1 body frame
  - Computes all tangent segments
  - Populates `self.cable_right_segments` and `self.cable_left_segments`

## Co-Contraction Stiffness Equations

For both methods, effective stiffness with symmetric co-contraction:
```
k_eff = 2 · k_exo · r_eff²
```
where `r_eff = l_c'(q₂)` (Method A, varies) or `r_eff = r_cp` (Method B, constant).

| Method | r_eff | k_eff (k_exo=200 N/m) |
|--------|-------|-----------------------|
| A (offset) | ~32 mm | ~0.41 N·m/rad |
| B (centred) | 47.75 mm | 0.91 N·m/rad |

## Common Issues

1. **Wrap arc at wrong Z**: Use `center_override` (6th tuple element) for elbow arcs
2. **End point on wrong groove**: Verify body-local z + FK chain offset = target world Z
3. **Tangent fails**: Check that `d(pulleys) > r1 + r2` for external, `d > |r1 - r2|` for internal
4. **matplotlib vs Meshcat mismatch**: Both must handle 6-tuple wrap_arcs format
