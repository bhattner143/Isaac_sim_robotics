---
description: "Use when writing or modifying controllers: computed torque, LQR, OFC, muscle dynamics, trajectory tracking, IK, Jacobian computations, impedance control."
applyTo: ["controller/**", "*controller*.py", "*lqr*.py", "*ofc*.py"]
---

# Controller Design Conventions

## Computed Torque (CT)
Inverse-dynamics feedback linearization: `τ = M(q)·a_des + h(q, q̇)` where `a_des = q̈_ref + Kd·(q̇_ref − q̇) + Kp·(q_ref − q)`.

Dual implementations:
- `controller/computed_torque_isaacsim.py` — NumPy-only (Isaac Sim + general)
- Drake version uses `MultibodyPlant` dynamics directly

## Trajectories (controller/trajectory.py)
Available trajectory types: `RectTrajectory`, `CircleTrajectory`, `LineTrajectory`.
All provide `.eval_position(t)`, `.eval_velocity(t)`, `.eval_acceleration(t)`.

Use `PreambleTrajectorySource` to wrap a cubic Hermite move-to-start before the main trajectory.

## IK Pattern
```python
q_des, q_dot_ref, q_ddot_ref, ik_ok = ik_to_joint_space_references(
    ee_ref, ee_vel, ee_acc, L1, L2, q_seed, solve_2r_ik)
if ik_ok:
    q_seed = q_des.copy()  # warm-start for next step
```

Always warm-start IK with the previous solution to avoid discontinuities.

## Cable Tension Decomposition
Tendon scripts decompose joint-2 torque into green (retract) / red (extend) cable tensions. The `ComputedTorqueController` with `pulley_radius` does this automatically.

## Composable Wiring with SEA
The `ComputedTorqueController` outputs `[τ₁, τ₂]` on its `actuation` port. For SEA simulations, wire this into `SEACableActuator.tau_desired` instead of directly to the plant. The SEA actuator models motor + spring dynamics and outputs the actual torques. See `sea-actuator.instructions.md` for the wiring pattern.

## Residual RL (rl/envs/manipulator_residual_env.py)
RL adds a small correction `Δτ` on top of CT before the SEA:
```
CT Controller → (+Δτ from RL) → SEA Model → PhysX Plant
```

- **14-D observation**: `[q₁, q₂, q̇₁, q̇₂, ee_err_x, ee_err_y, δ, δ̇, F_cable, τ₂_ct, τ₂_sea, τ_motor, τ₁_track_err, τ₂_track_err]`
- **2-D action**: `Δτ ∈ [-5, +5]² Nm` (small relative to CT output)
- **Reward**: tracking (100) + effort (0.01) + smoothness (0.001) + torque_tracking (1.0)
- The torque tracking reward `w₂·(τ_des − τ_applied)²` is critical — without it the policy ignores the spring lag
- For weak springs (k_s ≤ 100 N/m), the RL policy learns a torque-error I+D compensator
- Motor: AK60-6 in torque mode (2nd-order rotor dynamics, N=6, τ_peak=9 Nm)

## State Vectors
- **Cart-Pendulum 2D** (8D): `[x, y, α, β, ẋ, ẏ, α̇, β̇]`
- **Extended System** (14D): adds `[F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]`
