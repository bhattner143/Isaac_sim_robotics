# Control Modes: Joint-Space PD vs End-Effector Impedance

## Overview

The script now supports two control modes for the manipulator:

1. **Joint-Space PD Control** (default)
2. **End-Effector Impedance Control**

## Configuration

To change control modes, edit the `CONTROL_MODE` parameter in the configuration section:

```python
# --- Control Mode Configuration ---
# Options: "joint_pd" (joint-space PD control) or "impedance" (end-effector impedance control)
CONTROL_MODE = "joint_pd"  # Change to "impedance" to try end-effector impedance control
```

## Mode 1: Joint-Space PD Control

**What it does:**
- Directly controls joint positions using Proportional-Derivative feedback
- Each joint has an error signal computed as: `error = target_position - current_position`
- Control law: `torque = Kp * error + Kd * error_dot`

**Configuration parameters:**
```python
PD_KP = 100000.0  # Proportional gain for joint PD
PD_KD = 100.0/2   # Derivative gain for joint PD
```

**Logged data:**
- `joint1_error`, `joint2_error` - Position errors (rad)
- `joint1_error_dot`, `joint2_error_dot` - Velocity errors (rad/s)
- `joint1_torque`, `joint2_torque` - Command torques (N⋅m)

**Pros:**
- Direct control of joint angles
- Simple and predictable
- Good for precise joint positioning

**Cons:**
- Doesn't directly control end-effector motion
- Joint errors don't directly translate to task-space performance

## Mode 2: End-Effector Impedance Control

**What it does:**
- Controls the end-effector (EE) position in task space (X, Y, Z)
- Computes EE error: `error = desired_ee_position - current_ee_position`
- Applies impedance control in task space: `F = Kp * ee_error + Kd * (-ee_velocity)`
- Converts task-space forces to joint torques using Jacobian transpose method: `τ = J^T * F`

**Configuration parameters:**
```python
IMPEDANCE_KP = 5000.0   # End-effector position gain (N/m)
IMPEDANCE_KD = 500.0    # End-effector velocity gain (N*s/m)
IMPEDANCE_MASS = 1.0    # Virtual mass for impedance control
```

**Logged data:**
- `ee_pos_x`, `ee_pos_y`, `ee_pos_z` - Current EE position (m)
- `ee_error_x`, `ee_error_y`, `ee_error_z` - EE position errors (m)
- `ee_error_dot_x`, `ee_error_dot_y`, `ee_error_dot_z` - EE velocity errors (m/s)

**Pros:**
- Direct control of end-effector trajectory
- More intuitive for manipulation tasks
- Can adapt to joint configuration changes

**Cons:**
- Requires Jacobian computation
- More computationally expensive
- Needs careful tuning of impedance gains

## How to Compare

1. **Run with Joint-Space PD:**
   ```python
   CONTROL_MODE = "joint_pd"
   ```
   - Check plots: Row 4 shows joint errors and torques
   - Observe how joint errors drive the trajectory

2. **Run with Impedance Control:**
   ```python
   CONTROL_MODE = "impedance"
   ```
   - Check plots: Row 4 shows EE errors
   - Observe how task-space errors are handled

## Plotting

The `plot_results()` function automatically adapts based on which control mode was used:

- **Joint-Space PD Mode**: Shows joint position errors and command torques
- **Impedance Mode**: Shows end-effector position errors and velocity errors

## Tuning Guidelines

### Joint-Space PD Control
- Increase `PD_KP` for faster response (but risk oscillation)
- Increase `PD_KD` for damping (reduces overshoot)
- Default values: Kp=100000, Kd=50 work well for trajectory tracking

### Impedance Control
- `IMPEDANCE_KP`: Controls stiffness (N/m) - higher = stiffer tracking
- `IMPEDANCE_KD`: Controls damping - typical ratio Kp/Kd ≈ 10
- Start conservative and increase Kp if tracking is loose
- Increase Kd if oscillations occur

## Advanced: Switching Modes

The infrastructure supports runtime control mode selection. You can extend this to:
- Switch modes based on trajectory phase
- Compare both modes side-by-side in separate simulations
- Use hybrid control (PD for orientation, impedance for position)

## Mathematical Background

### Joint-Space PD
```
τ = Kp * (θ_d - θ) + Kd * (θ̇_d - θ̇)
```

### Task-Space Impedance with Jacobian Transpose
```
F_task = Kp * (x_d - x) + Kd * (ẋ_d - ẋ)
τ = J(θ)^T * F_task
```

Where:
- τ = joint torques
- F_task = task-space forces
- J = Jacobian matrix
- θ = joint angles
- x = end-effector position
