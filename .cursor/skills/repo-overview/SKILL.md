---
name: repo-overview
description: >-
  Overview of the Isaac_sim_robotics research repo: dual PyDrake/Isaac Sim stack,
  cup manipulator with SEA cable elbow and exosuit co-contraction, control theory
  (CT, LQR, CCM/C3M), URDF pipeline, and project conventions. Use when onboarding,
  answering what this repo does, choosing which script/env to use, or navigating
  the codebase for the first time.
---

# Isaac Sim Robotics — Repo Overview

## What This Repo Is

Research codebase for a **2-DOF planar manipulator** combining PyDrake (control design) and Isaac Sim (RL/GPU path).

### Real hardware (exists today)
- **URDF:** `model_using_onshape_to_robot/manipulator-hybrid-planar/manipulator_hybrid_planar.urdf`
- **No exosuit yet** on the physical robot — exo will be added later
- **Cable-only actuation** — no series springs on the real manipulator (see `manipulator-hardware`)
- See skill **`manipulator-hardware`** for real robot vs simulation distinction

### Actuation: cable only, no springs

The **real manipulator has cable drive only — no series elastic springs** in the
transmission path (for the time being).

**Hardware topology:**
- **Shoulder** — direct-drive MIT motor (no cable)
- **Elbow** — one motor, antagonistic lower (+Y) / upper (−Y) cables (one taut, one slack)

| Real MHP | Legacy cup / exo sim |
|----------|----------------------|
| Shoulder direct + elbow antagonistic cable | `SEACableActuator` with `k_s`, `b_c` spring |
| `τ₁` direct; `τ₂ = r_p·(T_lower − T_upper)` | Software spring model `F = k_s·δ + b_c·δ̇` |
| `script_mhp_manipulator_cable_framework_pydrake.py` | CT → SEA → plant |

When implementing **MHP** control (`script_mhp_manipulator_cable_framework_pydrake.py`):

- **Shoulder**: direct-drive MIT.
- **Elbow**: one motor; `T_lower = max(F_net,0)`, `T_upper = max(−F_net,0)`.
- Use `cable/*_mhp.py` for routing viz only.

### Legacy digital-twin only (no physical hardware)
Older cup-manipulator + exosuit designs simulated for research but **never built**:
- `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` — SEA elbow, no exo
- `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` — full exo + CT + co-contraction / transparency modes

Run modes for the exo script are in **`.vscode/tasks.json`** (🧪/⭕/▭ Exo tasks).

## Dual-Framework Rule (Critical)

| Engine | File patterns | Conda env | Import rule |
|--------|---------------|-----------|-------------|
| **PyDrake** | `script_*.py`, `demo_*.py` | `pydrake` / `pydrake_cursor` | `from pydrake.all import ...` |
| **Isaac Sim** | `test_*.py`, `*isaac_sim*.py` | `env_isaacsim` | `from isaacsim import SimulationApp` **first** |

**Never mix** imports between frameworks in one file.

## Repository Map

```
actuators/          SEA cable, motor dynamics, exo actuators (Drake + NumPy)
cable/              Pulley routing, DrakeCablePlant, exo Methods A/B
controller/         Computed torque, trajectories, IK
robots/             Cup manipulator, tendon, exo, Isaac wrappers
rl/                 PPO residual RL on Isaac Sim
model_using_onshape_to_robot/   URDF + mesh assets from CAD
notes_all/          LaTeX research notes (theory + implementation docs)
.github/            Copilot instructions, path-scoped rules, task skills
```

## Key Scripts

| Script | Hardware? | Purpose |
|--------|-----------|---------|
| `cable/test_mhp_cable_routing_actual_viz.py` | **Real robot** (MHP) | Cable routing viz for `manipulator-hybrid-planar` |
| `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` | Digital twin only | Full exo sim: CT, transparency, co-contraction on disturbance |
| `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` | Digital twin only | Earlier SEA-only cup manipulator (no exo) |
| `script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py` | Digital twin only | Isaac Sim exo mirror |
| `script_cup_manipulator_pendulam_lqr_min_effort_2d.py` | — | LQR / OFC cart-pendulum modes |
| `rl/train_ppo_residual.py` | — | PPO residual RL training |

## Critical Conventions

### Drake joint order
Joints follow **URDF parse order**, not alphabetical. Cup manipulator: `[q2, q1]` not `[q1, q2]`. Always verify:
```python
for idx in plant.GetJointIndices(model_instance):
    print(plant.get_joint(idx).name())
```

### State vectors
- Cart-pendulum 8D: `[x, y, α, β, ẋ, ẏ, α̇, β̇]`
- SEA torque mode 6D: `[q₁, q₂, q̇₁, q̇₂, θ_m, θ̇_m]`
- SEA position mode 5D: `[q₁, q₂, q̇₁, q̇₂, l_m]`

### Control wiring (Drake)
```
Trajectory → ComputedTorque → SEA (drive) ─┐
ExoCmd → SEA (exo) ─────────────────────────┼→ ActuationSum → Plant
```

### Configs
Frozen dataclasses in `configs/robot/robot_types.py`. JSON-serializable via `dataclasses.asdict()`.

### Linearization
Use Drake `Linearize()`, not hand-built Jacobians.

## Environments

| Env | Python | Use |
|-----|--------|-----|
| `pydrake_cursor` | 3.13 | Mac PyDrake dev (Drake 1.54+) |
| `pydrake` | 3.14 | PyDrake scripts |
| `env_isaacsim` | 3.11 | Isaac Sim (Linux; needs full Isaac install) |

On macOS: PyDrake scripts only. Isaac Sim tasks in `.vscode/tasks.json` target Linux paths.

## Related Skills (`.cursor/skills/`)

| Skill | When |
|-------|------|
| `manipulator-hardware` | Real robot URDF vs legacy exo digital-twin scripts |
| `notes-all-index` | Find LaTeX docs, derivations, meeting notes |
| `exosuit-cable-routing` | Exo pulleys, Z-planes, Method A vs B |
| `sea-tuning` | **Legacy digital-twin only** — cup/exo SEA sim; not real MHP cable-only hardware |
| `onshape-to-urdf` | CAD → URDF pipeline |
| `run-isaac-sim` | Launch/debug Isaac Sim scripts |
| `add-trajectory` | New trajectory classes |
| `write-latex-notes` | New notes in `notes_all/` |
| `huggingface-models` | HF/LeRobot model integration |

## Path-Scoped Instructions (`.github/instructions/`)

Auto-applied rules by file glob: `pydrake-scripts`, `isaac-sim-scripts`, `controller-design`, `sea-actuator`, `exosuit-cables`, `urdf-models`, `latex-notes`.

## Common Pitfalls

1. Wrong joint order when setting Drake positions
2. `SimulationApp()` not first import in Isaac Sim scripts
3. Linearizing before setting equilibrium context
4. IK without warm-start from previous solution
5. Installing wrong `pydrake` pip package (use `drake`, not `pydrake`)
6. RL reward missing `w₂·(τ_des − τ_applied)²` term (spring lag ignored)

## Further Reading

- `.github/copilot-instructions.md` — full AI guide
- `notes_all/notes_cup_manipulator_tendon/meeting_notes/2026_05_20_meeting_notes.md` — narrative project summary
- `SYSTEM_ARCHITECTURE_GUIDE.md`, `LINEARIZATION_IMPLEMENTATION_SUMMARY.md` — in-repo guides

