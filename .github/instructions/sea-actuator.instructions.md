---
description: "Use when working on Series Elastic Actuator (SEA) code, cable actuation, tendon routing, spring-damper models, motor bandwidth tuning, pulley mechanics."
applyTo: ["actuators/**", "*sea*.py", "*tendon*.py"]
---

# SEA Cable Actuator Conventions

## Physics Model
Cable-driven joint 2 with motor → spring → pulley → joint:

```
Motor position:  l_m_des = r_p · q₂ + τ₂_des / (k_s · r_p)
Motor servo:     l̇_m = ω_m · (l_m_des − l_m)
Spring extension: δ = l_m − r_p · q₂
Cable force:     F = k_s · δ + b_c · (l̇_m − r_p · q̇₂)
Joint torque:    τ₂ = r_p · F
```

Joint 1 is always rigid direct-drive (τ₁ passes through unchanged).

## Key Parameters
- `r_p` — pulley radius (HTD 5M 60T: ≈ 47.75 mm)
- `k_s` — spring stiffness [N/m] (typical: 100–500)
- `b_c` — cable damping [N·s/m] (typical: 1–5)
- `ω_m` — motor servo bandwidth [rad/s] (typical: 20–50)

## Dual Implementations
- **`actuators/sea.py`** — PyDrake `LeafSystem` version (for Drake diagrams)
- **`actuators/sea_isaacsim.py`** — Pure NumPy version (`SEACableActuatorNP`)

The NumPy version is engine-agnostic. Its `step()` method returns `(tau_out, SEADiagnostics)`.

## SEADiagnostics Dataclass
Always log: `l_m`, `l_m_des`, `delta`, `F_cable`, `tau_sea`, `tau1_des`, `tau2_des`.

## Initialization
Call `sea.initialize(q2_init)` before the sim loop to set `l_m = r_p * q2_init` (no initial spring extension).
