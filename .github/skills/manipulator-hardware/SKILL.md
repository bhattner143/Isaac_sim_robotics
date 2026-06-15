---
name: manipulator-hardware
description: >-
  Distinguishes the real physical manipulator (manipulator-hybrid-planar URDF, no
  exo yet) from legacy digital-twin-only exo scripts (cup manipulator with exosuit).
  Real MHP actuation is cable-only (no series springs). Use when choosing URDFs,
  wiring control to hardware, sim-to-real, or when asked what exists on the real
  robot vs simulation only.
---

# Manipulator: Real Hardware vs Digital Twin

## Real Robot (What Exists Physically)

**URDF / CAD folder:** `model_using_onshape_to_robot/manipulator-hybrid-planar/`

| Item | Value |
|------|-------|
| Onshape export name | `manipulator_hybrid_planar` |
| URDF | `manipulator_hybrid_planar.urdf` |
| Config | `manipulator-hybrid-planar/config.json` |
| Convert task | `.vscode/tasks.json` → `🔄 Onshape → URDF: manipulator-hybrid-planar` |

**Current hardware state:**
- **2-DOF hybrid planar manipulator** (shoulder transmission + elbow cable drive)
- **No exosuit** on the real robot yet — exo will be added later
- Cable routing work-in-progress: `cable/test_mhp_cable_routing_actual_viz.py`, `manipulator_hybrid_planar_fusion/` (fusion CAD variant for routing viz)

### Actuation: cable only, no springs

The **real manipulator has cable drive only — no series elastic springs** in the
transmission path (for the time being).

**Hardware topology:**
- **Shoulder (`jt_upper_base`)** — **direct-drive** motor (MIT mode), no cable
- **Elbow (`jt_lower_upper`)** — **one motor**, two **antagonistic** cables
  (lower +Y / upper −Y on the same spool).  Only one cable is taut; the other
  is slack: `T_lower · T_upper = 0`.

| Real MHP | Legacy cup / exo sim |
|----------|----------------------|
| Shoulder direct + elbow antagonistic cable | `SEACableActuator` with `k_s`, `b_c` spring |
| `τ₁` direct; `τ₂ = r_p·(T_lower − T_upper)` | Software spring model `F = k_s·δ + b_c·δ̇` |
| CT → shoulder MIT + elbow tension split → MIT | CT → SEA → plant |
| `script_mhp_manipulator_cable_framework_pydrake.py` | Exo / spring SEA scripts |

When implementing **MHP** control (`script_mhp_manipulator_cable_framework_pydrake.py`):

- **Shoulder**: direct-drive MIT — `τ₁` passes straight through.
- **Elbow**: one motor, antagonistic lower (+Y) / upper (−Y) cables; decompose
  `F_net = τ₂/r_p` into `T_lower = max(F,0)`, `T_upper = max(−F,0)`.
- Use `cable/*_mhp.py` for **routing geometry and viz**, not spring physics.
- **Do not** model two independent cable motors for shoulder and elbow.

When implementing control for **hardware**, target `manipulator-hybrid-planar` URDF and drivers — **not** the legacy exo URDFs below.

---

## Legacy Digital-Twin Only (No Physical Hardware)

These scripts simulate an **older cup-manipulator design with exosuit**. The physical exo version was **never built** — code exists only as a digital twin for research.

| Script | URDF | What it simulates |
|--------|------|-------------------|
| `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` | `manipulator_cable/manipulator_cable_obj.urdf` | SEA elbow cable, no exo |
| `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` | `manipulator_cable_exo_springs_elbow_follow/manipulator_cable_exo_springs_elbow_follow_obj.urdf` | SEA + exosuit Method B (full implementation) |
| `script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py` | (Isaac Sim mirror) | Same exo design in Isaac Sim |

**Do not assume** these URDFs or exo mechanics match the real `manipulator-hybrid-planar` robot.

---

## Exo PyDrake Script — Full Digital-Twin Implementation

`script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` is the **complete** simulation of the old exo manipulator:

### Control stack
```
Trajectory → Computed Torque (CT) → SEA drive cable → Plant
                              ExoCmd → SEA exo cables → (co-contraction / transparency)
```

### Exo modes
| Mode | Flags | Behaviour |
|------|-------|-----------|
| **Transparency** (default) | `--no-exo-activate` | Exo passive — motors track encoder, zero added stiffness |
| **Timed co-contraction** | `--exo-activate --exo-activate-time T` | Stiffness injection at time T |
| **Disturbance-sync** | `--exo-activate --exo-dist-sync --disturbance` | Co-contraction only during disturbance window |
| **Reactive** | `--exo-reactive --exo-e-on --exo-e-off` | Error-triggered co-contraction |

### Disturbance + co-contraction experiment pattern
Tasks compare **exo OFF** (baseline) vs **exo ON** (co-contraction) under the same disturbance:
- Disturbance at `--disturbance-time` (default 8 s)
- Modes: `sine`, `square`, `pulse`, `vel`, `pos`
- CT gains typical in tasks: `--ct-kp 400 --ct-kd 80`

---

## Running Modes — `.vscode/tasks.json`

All exo PyDrake run configs are in **`.vscode/tasks.json`**. Run via **Cmd+Shift+P** → **Tasks: Run Task**.

### PyDrake exo tasks (digital twin)

| Task label | Trajectory | Exo | Disturbance |
|------------|------------|-----|-------------|
| `🧪 Exo Sine: OFF` | line (tiny) | OFF | sine 1 Hz, 1 cycle |
| `🧪 Exo Sine: ON` | line (tiny) | ON + dist-sync | sine 1 Hz, 1 cycle |
| `⭕ Exo Circle: OFF` | circle | OFF | sine @ 8s |
| `⭕ Exo Circle: ON` | circle | ON | sine @ 8s |
| `🔔 Exo Circle: PULSE OFF/ON` | circle | OFF/ON | pulse @ 8s |
| `🟦 Exo Circle: SQUARE OFF/ON` | circle | OFF/ON | square wave @ 8s |
| `⭕ Exo Circle: REACTIVE` | circle | reactive | sine @ 8s |
| `∞ Exo Figure8: OFF/ON` | figure8 | OFF/ON | sine @ 8s |
| `▭ Exo Rect: OFF/ON` | rect | OFF/ON | sine @ 8s |
| `🔍 Exo Learning: Probe → Identify → Adapt` | param learning script | — | — |

### Isaac Sim exo tasks (also digital twin only)
Labels prefixed `🩹 Exo Isaac Sim:` — same legacy exo design, GPU renderer.

### Launch configs
Matching debug configs in `.vscode/launch.json`:
- `🦿 Exo+SEA: Deactivated (passive)`
- `🦿 Exo+SEA: Activated at 5s`

---

## Decision Guide

| Question | Answer |
|----------|--------|
| Which URDF is the real robot? | `manipulator-hybrid-planar/manipulator_hybrid_planar.urdf` |
| Does the real robot have exo? | **No** (planned later) |
| Does the real robot have series springs? | **No** — cable only (rigid tendon) |
| Should MHP sim use SEA? | **No** — CT → joint torque or rigid cable passthrough; not `SEACableActuator` |
| Which script demonstrates exo co-contraction? | `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` (sim only) |
| Which script is SEA without exo (older design)? | `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` (sim only) |
| How do I run exo experiments? | `.vscode/tasks.json` tasks listed above |
| Where is cable routing for real robot? | `cable/test_mhp_cable_routing_actual_viz.py` + `manipulator_hybrid_planar_fusion/` |

## Related Skills

- `repo-overview` — full repo map
- `exosuit-cable-routing` — Method A/B geometry (applies to legacy exo URDF)
- `sea-tuning` — **legacy digital-twin only** (cup/exo sim); not for real MHP cable-only hardware
- `onshape-to-urdf` — regenerate `manipulator-hybrid-planar` from Onshape
