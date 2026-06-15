---
name: notes-all-index
description: >-
  Navigate notes_all/ LaTeX research documentation for Isaac_sim_robotics: cup
  manipulator SEA/exosuit theory, PyDrake and Isaac Sim implementation docs,
  contraction theory, hardware, and meeting notes. Use when deriving equations,
  writing docs, finding prior derivations, or understanding control/architecture
  decisions documented outside the code.
---

# notes_all Documentation Index

`notes_all/` is the LaTeX-first research notebook (~11 top-level folders, no root README). Docs are **script-anchored** — titles reference Python files and CLI modes.

## Start Here by Goal

| Goal | Document |
|------|----------|
| Big-picture system model | `notes_cup_manipulator_tendon/system/cup_manipulator_sea_system_description.tex` |
| Narrative project summary | `notes_cup_manipulator_tendon/meeting_notes/2026_05_20_meeting_notes.md` |
| PyDrake CT + SEA impl | `notes_cup_manipulator_tendon/pydrake/Computed_Torque_PyDrake_Implementation.tex` |
| PyDrake SEA cable | `notes_cup_manipulator_tendon/pydrake/SEA_Cable_PyDrake_Implementation.tex` |
| Exo impedance learning | `notes_cup_manipulator_tendon/pydrake/Exo_Impedance_Learning_PyDrake_Implementation.tex` |
| Isaac Sim CT | `notes_cup_manipulator_tendon/isaac-sim/Computed_Torque_IsaacSim_Implementation.tex` |
| Isaac Sim exo cables | `notes_cup_manipulator_tendon/isaac-sim/Exo_Cable_IsaacSim_Implementation.tex` |
| Exosuit co-contraction theory | `notes_cup_manipulator_tendon/exosuit_stiffness_and_co_contraction.tex` |
| CCM / C3M research plan | `notes_cup_manipulator_tendon/contraction/ccm_exosuit_regional_strategy.tex` |
| CubeMars hardware | `notes_cup_manipulator_tendon/hardware/CubeMars_Hardware_Implementation.tex` |
| Control theory foundations | `notes_control/notes_contraction_theory.tex`, `notes_computed_torque.tex` |
| LQR modes | `notes_control/notes_lqr_cart_manip_following.tex`, `notes_lqr_both_cart_manip.tex` |

## Top-Level Folders

```
notes_all/
├── notes_cup_manipulator_tendon/   ★ Main hub (SEA, exo, CT, hardware, meetings)
│   ├── pydrake/                    Implementation ↔ equations (Drake)
│   ├── isaac-sim/                  Implementation ↔ equations (Isaac)
│   ├── system/                     6D state-space, C3M applicability
│   ├── contraction/                CCM/C3M synthesis strategy
│   ├── hardware/                   AK60-6, AK80-8, Jetson, CAN
│   └── meeting_notes/              Progress decks, transcripts, .md summaries
├── notes_control/                  CT, LQR, contraction (compile_notes.tex)
├── notes_cup/                      Cart-pendulum EOM derivations
├── notes_cart_pendulum_manipulator/  Coupled cart-pendulum + manipulator (Isaac)
├── notes_ball_plate/               Isaac Sim onboarding tutorials
├── notes_ball_plate_dips/          Plate-with-dips OOP demo
├── notes_general/                  Literature (internal impedance control)
├── notes_teaching_demo_video/      Contraction theory beamer lecture
├── notes/                          Legacy PDF archive (superseded copies)
└── proposal/                       Proposal_short.docx
```

## Main Hub: notes_cup_manipulator_tendon

### Theory (root `.tex`)
- `computed_torque_controller_dynamics.tex` — CT derivation for manipulator
- `cup_manipulator_modes_and_pulley_routing.tex` — pulley geometry, routing modes
- `hybrid_jacobian_derivation.tex` — cable Jacobian
- `cup_manipulator_tendon_and_ik_diagram.tex` — IK / tendon kinematics
- `exosuit_stiffness_and_co_contraction.tex` — K_eff = 2·k_exo·r²

### Dual-simulator mirroring
Same control stack documented twice:
- **pydrake/** — analytical, Meshcat, LQR/CCM friendly
- **isaac-sim/** — GPU, RL path, USD/URDF import

### Research layers
| Layer | Folder | Content |
|-------|--------|---------|
| Dynamics | `system/`, root `.tex` | EOM, state vectors, linearization |
| Implementation | `pydrake/`, `isaac-sim/` | Code listings + equation boxes |
| Control synthesis | `contraction/` | CCM vs CT, C3M training strategy |
| Hardware | `hardware/` | Motors, CAN, real robot |
| Collaboration | `meeting_notes/` | Beamer decks, May 2026 MD summary |

## Key Technical Topics → Where Documented

| Topic | Location |
|-------|----------|
| Computed torque | `computed_torque_controller_dynamics`, `notes_control/notes_computed_torque` |
| SEA cable dynamics | `pydrake/SEA_Cable_PyDrake_Implementation`, `system/` |
| Exosuit stiffness | `exosuit_stiffness_and_co_contraction`, meeting notes §3–4 |
| Cable pulley routing | `cup_manipulator_modes_and_pulley_routing`, `cable/` code |
| Parameter learning | `Exo_Impedance_Learning_PyDrake_Implementation`, meeting notes §5 |
| LQR / OFC | `notes_control/notes_lqr_*` |
| Contraction / CCM | `notes_contraction_theory`, `contraction/`, `template_CCM/` |
| Digital twin pipeline | meeting notes §2 (Onshape → URDF → OBJ/USD) |

## Writing New Notes

Use skill `write-latex-notes`. Conventions:
- Place in appropriate `notes_all/<topic>/` subfolder
- Reference related `script_*.py` in title/subtitle
- Use project macros: `\bq`, `\btau`, `\bM`, `\bJ`, etc.
- Pair `codemath` tcolorbox blocks with equation derivations
- Compile with `latexmk -pdf` in the note's directory

## Meeting Notes as Living Log

Best readable entry: `meeting_notes/2026_05_20_meeting_notes.md` covers:
1. Dual simulator rationale (PyDrake now, Isaac for RL later)
2. Hardware: shoulder direct-drive, elbow SEA, exo Method B
3. Control: CT → SEA → optional exo co-contraction
4. Exo OFF vs ON disturbance results
5. Probe → identify → adapt parameter learning
6. CCM vs LQR roadmap

Progress presentation: `meeting_notes/progress_overall.tex` (beamer).

## Related Code Cross-Refs

| Notes topic | Primary script | Hardware? |
|-------------|----------------|-----------|
| Exo+SEA PyDrake | `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` | Digital twin only |
| SEA without exo | `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` | Digital twin only |
| **Real robot URDF** | `model_using_onshape_to_robot/manipulator-hybrid-planar/` | **Physical hardware** (no exo yet) |
| MHP cable routing | `cable/test_mhp_cable_routing_actual_viz.py` | Real robot CAD |
| Exo Isaac Sim | `script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py` | Digital twin only |
| Exo param learning | `script_cup_manipulator_pendulam_tendon_with_exo_param_learning_pydrake.py` | Digital twin only |
| C3M training | `contraction-theory/C3M/main.py` | — |
| LQR cart-pendulum | `script_cup_manipulator_pendulam_lqr_min_effort_2d.py` | — |

Exo run modes: `.vscode/tasks.json` (see skill `manipulator-hardware`).
