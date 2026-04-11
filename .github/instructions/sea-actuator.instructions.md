---
description: "Use when working on Series Elastic Actuator (SEA) code, cable actuation, tendon routing, spring-damper models, motor bandwidth tuning, pulley mechanics, motor dynamics modes."
applyTo: ["actuators/**", "*sea*.py", "*tendon*.py"]
---

# SEA Cable Actuator Conventions

## Motor Dynamics Modes

The motor model is **separate from the SEA** and lives in `actuators/motor_dynamics.py`.
Two modes are available, selected via `MotorMode` enum:

### Torque mode (default) — `MotorMode.TORQUE`
2nd-order rotor dynamics matching CubeMars MIT torque mode:

```
J_m · θ̈_m = τ_m − b_m · θ̇_m − r_p · F / N

where:
  τ_m = τ₂_des / N                            motor-side torque command
  δ   = r_p · (θ_m / N − q₂)                  linear spring extension  [m]
  F   = k_s · δ + b_c · δ̇                     cable force              [N]
  r_p · F / N                                  spring load on rotor     [Nm]
```

State: `[θ_m, θ̇_m]` (motor-side angle and velocity).
Parameters `J_m` and `b_m` come from the motor datasheet config (`MotorModelConfig`).
Motor-side resonance: `ω_n = sqrt(r_p² · k_s / (N² · J_m))`.

### Position mode (legacy) — `MotorMode.POSITION`
1st-order position servo with bandwidth `ω_m`:

```
l_m_des = r_p · q₂ + τ₂_des / (k_s · r_p)    steady-state spring inversion
l̇_m    = ω_m · (l_m_des − l_m)                first-order position servo
δ       = l_m − r_p · q₂                       spring extension
F       = k_s · δ + b_c · (l̇_m − r_p · q̇₂)    spring–damper force
```

State: `[l_m]` (cable displacement).

### Unilateral Cable Model (shared by both modes)
Cables can only PULL (tension ≥ 0), never push:
- `δ > 0` → green taut: `T_green = max(F_raw, 0)`, `T_red = 0`
- `δ < 0` → red taut: `T_green = 0`, `T_red = max(−F_raw, 0)`
- `τ₂_out = r_p · (T_green − T_red)`

Joint 1 is always rigid direct-drive (τ₁ passes through unchanged).

## Key Parameters
- `r_p` — pulley radius (HTD 5M 60T: ≈ 47.75 mm)
- `k_s` — spring stiffness [N/m] (typical: 100–500)
- `b_c` — cable damping [N·s/m] (typical: 1–5)
- `ω_m` — motor servo bandwidth [rad/s] (position mode only, typical: 20–50)
- `J_m` — rotor inertia [kg·m²] (torque mode, from `motor_cfg.rotor_inertia_motor`)
- `b_m` — motor-side viscous damping [Nm·s/rad] (torque mode, = `viscous_damping_joint / N²`)
- `N`   — gear ratio (from `motor_cfg.gear_ratio`)

## Motor Dynamics Classes (`actuators/motor_dynamics.py`)
- `MotorDynamics` — abstract base; subclasses implement `step()` and `compute_spring_force()`
- `PositionServoMotor(MotorDynamics)` — 1st-order position servo
- `TorqueMotor(MotorDynamics)` — 2nd-order rotor dynamics
- `MotorMode` — enum: `TORQUE` (default), `POSITION`
- `create_motor_dynamics(mode, motor_cfg, k_s, b_c, r_p, dt, omega_m)` — factory function

## Dual Implementations
- **`actuators/sea.py`** — PyDrake `LeafSystem` version (for Drake diagrams)
- **`actuators/sea_isaacsim.py`** — Pure NumPy version (`SEACableActuatorNP`)

The NumPy version is engine-agnostic. Its `step()` method returns `(tau_out, SEADiagnostics)`.

## Two-Block Architecture (Preferred)
Wire `ComputedTorqueController` → `SEACableActuator` → `Plant` as separate Drake LeafSystems:

```python
from actuators.sea import SEACableActuator
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor

_motor = get_motor("AK60_6_KV80_Config")

ct  = builder.AddSystem(ComputedTorqueController(plant, manip, Kp=100, Kd=40, tau_max=9.0))

# Torque mode (default — recommended for CubeMars MIT mode):
sea = builder.AddSystem(SEACableActuator(
    plant, manip, k_s=300, b_c=2.0, tau_max=9.0, dt=0.002,
    motor_mode=MotorMode.TORQUE, motor_cfg=_motor,
))

# Position mode (legacy):
# sea = builder.AddSystem(SEACableActuator(
#     plant, manip, k_s=300, b_c=2.0, tau_max=9.0, dt=0.002,
#     motor_mode=MotorMode.POSITION, motor_cfg=_motor, omega_m=11.17,
# ))

builder.Connect(ct.GetOutputPort("actuation"),      sea.GetInputPort("tau_desired"))
builder.Connect(plant.get_state_output_port(),      sea.GetInputPort("plant_state"))
builder.Connect(sea.GetOutputPort("actuation"),     plant.get_actuation_input_port())
```

- **Diagnostics** come from `sea.GetOutputPort("diagnostics")` (8-element vector)
- **IK/q_des** come from `ct.GetOutputPort("joint_positions")`
- **Initialize spring**: `sea.initialize_spring_at_rest(sea_ctx, q2_init)` — NOT manual state access

The legacy `SEACableController` in `controller/controller.py` is a monolithic (CT + SEA in one class) kept for backward compatibility.

## Diagnostics Layout
8-element vector from `sea.GetOutputPort("diagnostics")`:

| Slot | Torque mode          | Position mode           |
|------|----------------------|-------------------------|
| [0]  | θ_m / N (joint pos)  | l_m (cable displ.)      |
| [1]  | θ̇_m / N (joint vel) | l_m_des (target displ.) |
| [2]  | δ (spring extension) | δ (spring extension)    |
| [3]  | F_cable (net force)  | F_cable (net force)     |
| [4]  | τ₁_des               | τ₁_des                  |
| [5]  | τ₂_des               | τ₂_des                  |
| [6]  | T_green              | T_green                 |
| [7]  | T_red                | T_red                   |

## Initialization
Call `sea.initialize_spring_at_rest(sea_ctx, q2_init)` before the sim loop so the spring starts with `δ = 0` (no pre-load).

## CLI
The main script accepts `--sea-mode {torque,position}` (default: `torque`).
