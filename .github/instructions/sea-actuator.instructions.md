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

## Motor Catalog (`actuators/motor.py`)
- `AK60_6_KV80_Config` — CubeMars AK60-6: N=6, τ_peak=9 Nm, J_m=3.32e-5, 24/48V
- `AK80_8_KV60_Config` — CubeMars AK80-8: N=8, τ_peak=25 Nm, J_m=6.09e-5, 48V
- Use `get_motor("AK60_6_KV80_Config")` to get a `MotorModelConfig` dataclass
- `MOTOR_CHOICES` list for CLI argument validation

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
`SEADiagnostics` dataclass from `sea.step()` (10 fields):

| Field        | Description                          | Unit      |
|--------------|--------------------------------------|-----------|
| `motor_pos`  | θ_m/N (torque) or l_m (position)     | rad / m   |
| `motor_aux`  | θ̇_m/N (torque) or l_m_des (position)| rad/s / m |
| `delta`      | spring extension δ                   | m         |
| `F_cable`    | net cable force                      | N         |
| `tau1_des`   | desired τ₁                           | Nm        |
| `tau2_des`   | desired τ₂                           | Nm        |
| `T_green`    | retracting cable tension             | N         |
| `T_red`      | extending cable tension              | N         |
| `tau_sea`    | actual τ₂ applied via spring         | Nm        |
| `tau_motor`  | motor-side electromagnetic torque     | Nm        |

Backward-compatible aliases: `.l_m` → `motor_pos`, `.l_m_des` → `motor_aux`.

## Initialization
Call `sea.initialize_spring_at_rest(sea_ctx, q2_init)` before the sim loop so the spring starts with `δ = 0` (no pre-load).

## CLI
The main script accepts `--sea-mode {torque,position}` (default: `torque`).
