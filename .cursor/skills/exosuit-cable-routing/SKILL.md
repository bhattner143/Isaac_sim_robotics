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

---

## Isaac Sim Exo Co-Contraction Scripts

### Key files (Isaac Sim side)
| File | Purpose |
|------|---------|
| `script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py` | Main run script: CT + drive SEA + exo co-contraction |
| `actuators/sea_exo_isaacsim.py` | NumPy-only SEA exo model (mirrors `actuators/sea_exo.py`) |
| `robots/cup_manipulator_tendon_with_exo_isaac.py` | Isaac ArticulationView wrapper for the exo URDF |
| `project_utils/viz_cables_isaacsim.py` | USD cylinder cable + helix spring visualisation |

### Run modes (`--mode`)
- `run-sea-exo` (default) — full CT + drive SEA + exo torque injection
- `show-scene` — loads robot, runs cable/spring viz, no control loop

### Important CLI flags
```bash
# Exo co-contraction
--exo-activate              # enable timed activation (requires --exo-activate-time)
--exo-activate-time 4.0     # time [s] to activate exo
--exo-delta-theta 0.1       # pre-tension angle [rad]; torque = k_eff * delta_theta
--exo-ks 8000               # exo spring stiffness [N/m]
--exo-r 0.04775             # exo pulley radius [m]

# Disturbance
--disturbance               # enable disturbance
--disturbance-mode vel|pos|torque|sine
--disturbance-time 7.0      # start time [s]
--disturbance-tau 2.0       # torque amplitude [Nm] (torque/sine modes)
--disturbance-dqdot 60.0    # velocity impulse [deg/s] (vel mode)
--disturbance-dq 15.0       # position jump [deg] (pos mode)
--disturbance-freq 3.0      # frequency [Hz] (sine mode)
--disturbance-cycles 1      # number of sine cycles (overrides --disturbance-dur)

# Trajectory clamping (CRITICAL — prevents IK failures)
# L1, L2 link lengths fed to CircleTrajectory/LineTrajectory via build_trajectory(L1=, L2=)
# All waypoints clamped to reach annulus [|L1-L2|, 0.97*(L1+L2)]
```

### Exo SEA actuator equations (NumPy port)
```python
# actuators/sea_exo_isaacsim.py — torque mode
delta   = r_exo * (theta_m / N - q2)      # cable stretch [m]
delta_d = r_exo * (theta_m_dot / N - q2_dot)
F_cable = k_exo * delta + b_exo * delta_d  # cable force [N]
tau_exo = r_exo * (F_R - F_L)            # net elbow torque [Nm]

# Pre-tension: theta_m offset = N * delta_theta / r_exo
# k_eff = 2 * k_exo * r_exo^2
```

### Disturbance types
| Mode | Mechanism | Best for |
|------|-----------|----------|
| `torque` | Constant τ_ext [Nm] added to joint 2 actuation | Steady-state deflection vs k_eff |
| `sine` | Sinusoidal τ_ext for N cycles | Passive bandwidth above CT cutoff |
| `vel` | Velocity impulse Δq̇₂ injected directly | Transient recovery test |
| `pos` | Position jump Δq₂ injected directly | Stiffness/convergence test |

Startup banner printed when `--disturbance` is armed:
```
💥 External-torque disturbance armed: τ_ext = +2.00 Nm (const)  window=[7.00, 8.50] s
```

### Helix spring visualisation
```python
# project_utils/viz_cables_isaacsim.py — ExoSpringVisualizerIsaac
exo_spring_viz = ExoSpringVisualizerIsaac(stage, exo_viz, k_exo, r_exo, tau_max)
exo_spring_viz.create_prims()   # pre-allocate USD cylinder prims
# in main loop (every 5 steps):
exo_spring_viz.update(delta_R=exo_diag.delta_R, delta_L=exo_diag.delta_L)
# Right spring: orange  Left spring: magenta
# Length fraction ∝ |δ| / δ_max,  clamped to [REST_FRAC=0.20, MAX_FRAC=0.75]
```

### Trajectory clamping (root cause of oscillations)
The `CircleTrajectory` in `controller/trajectory.py` was missing `_clamp_to_reach()`.
Without it, IK fails for unreachable waypoints → q_des freezes → δ oscillation → τ_exo ringing.

Fix already applied: pass `L1=L1, L2=L2` to `build_trajectory()`:
```python
main_traj = build_trajectory(args, L1=L1, L2=L2)
# Reads link lengths AFTER world.reset(): L1, L2 = manip._get_link_lengths()
```
If EE RMS is still high, check IK failure diagnostic printed at sim end.

### Verified results (2026-04-24)
- Drive SEA: k_s=200 N/m, b_c=8 N·s/m, motor=AK60_6_KV80_Config
- Exo: k_exo=8000 N/m, Δθ=0.10 rad, k_eff=36.48 Nm/rad
- EE tracking RMS = **1.76 mm** (circle, 2 laps)
- Both PyDrake and Isaac give identical k_eff = 36.4810 Nm/rad
