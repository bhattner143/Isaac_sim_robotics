---
name: sea-tuning
description: "Tune SEA (Series Elastic Actuator) parameters for cable-driven manipulators. Use when: adjusting spring stiffness, cable damping, motor bandwidth, diagnosing tracking lag, oscillations, or cable slack."
argument-hint: "Describe the behavior you're seeing (e.g., 'tracking lags by 200ms' or 'joint 2 oscillates')"
---

# SEA Parameter Tuning

## When to Use
- Joint 2 tracking has excessive lag or overshoot
- Spring oscillations or instability
- Cable goes slack (δ < 0)
- Tuning for a new trajectory speed or payload

## Parameter Reference

| Param | Symbol | Default | Range | Effect |
|-------|--------|---------|-------|--------|
| Spring stiffness | `k_s` | 200 N/m | 30–1000 | Higher = stiffer response, less lag, more force transmission |
| Cable damping | `b_c` | 2.0 N·s/m | 0.5–10 | Higher = less oscillation, more energy loss |
| Motor bandwidth | `ω_m` | 30 rad/s | 10–100 | Higher = faster motor servo, less phase lag |
| Pulley radius | `r_p` | 47.75 mm | Fixed | Set by HTD 5M 60T belt geometry |

## Tuning Procedure

### Step 1: Diagnose from plots
Run with current params and check the 2×2 plot:
```bash
python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \
    --spring-stiffness 200 --cable-damping 2 --motor-bandwidth 30
```

Look at:
- **Bottom-left (τ₂)**: Is SEA-applied torque tracking desired? Large gap = too soft
- **Bottom-right (δ)**: Is δ staying near zero? Large swings = under-damped
- **Top-right (error)**: Steady-state error vs transient

### Step 2: Fix the dominant issue

| Symptom | Primary Fix | Secondary Fix |
|---------|------------|---------------|
| Tracking lag (EE error stays high) | Increase `k_s` (200→500) | Increase `ω_m` (30→60) |
| Oscillations on joint 2 | Increase `b_c` (2→5) | Decrease `ω_m` (30→20) |
| Cable slack (δ < 0) | Increase `k_s` | Add pretension in `initialize()` |
| Torque saturation | Increase `--ct-tau-max` | Reduce trajectory speed |
| Instability / divergence | Decrease `ω_m` | Decrease `k_s`, increase `b_c` |

### Step 3: Iterate
Change ONE parameter at a time. Compare plots side by side.

## Quick Presets

```bash
# Stiff (industrial-like, fast response)
--spring-stiffness 500 --cable-damping 5 --motor-bandwidth 50

# Compliant (bio-inspired, safe interaction)
--spring-stiffness 80 --cable-damping 3 --motor-bandwidth 20

# Default (balanced)
--spring-stiffness 200 --cable-damping 2 --motor-bandwidth 30
```

## Physics Intuition
- `k_s` dominates steady-state behavior: `τ_ss ≈ k_s · r_p · δ_ss`
- `ω_m` dominates transient response: motor settling time ≈ `3/ω_m` seconds
- `b_c` provides viscous damping only during motion — no effect at steady state
- System is stable when `ω_m < k_s · r_p² / b_c` (approximate)
