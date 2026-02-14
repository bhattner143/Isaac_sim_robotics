# Controller Configuration Usage Guide

This document demonstrates how to use the controller configuration classes with their `create()` factory methods.

## Overview

Each controller type has a configuration class with:
1. `default()` - Returns default configuration
2. `create()` - Creates custom configuration with optional parameters

## Configuration Classes

### 1. PDControllerConfig

**Default Usage:**
```python
config = PDControllerConfig.default()
```

**Custom Configuration:**
```python
config = PDControllerConfig.create(
    kp=[150.0, 150.0],              # Higher gains for stiffer control
    kd=[15.0, 15.0],
    q_start_deg=[90, -170, 0, 180], # Start position in degrees
    q_goal_deg=[30, -50, 0, 180]    # Goal position in degrees
)
```

### 2. MinJerkControllerConfig

**Default Usage:**
```python
config = MinJerkControllerConfig.default(duration=3.0)
```

**Custom Configuration:**
```python
config = MinJerkControllerConfig.create(
    kp=[120.0, 120.0],
    kd=[12.0, 12.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=5.0  # Longer, smoother trajectory
)
```

### 3. ComputedTorqueControllerConfig

**Default Usage:**
```python
config = ComputedTorqueControllerConfig.default()
```

**Custom Configuration:**
```python
config = ComputedTorqueControllerConfig.create(
    kp=[25.0, 25.0],  # Still smaller than PD since feedforward compensates
    kd=[6.0, 6.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180]
)
```

### 4. TrajectoryOptimizedControllerConfig

**Default Usage:**
```python
config = TrajectoryOptimizedControllerConfig.default(duration=3.0)
```

**Custom Configuration:**
```python
config = TrajectoryOptimizedControllerConfig.create(
    kp=[20.0, 20.0],
    kd=[5.0, 5.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=4.0,
    num_samples=40,          # More knot points for smoother trajectory
    pendulum_weight=200.0,   # Prioritize pendulum stability
    torque_weight=0.05       # Less emphasis on energy efficiency
)
```

### 5. OFCControllerConfig

**Default Usage (Effort-Minimizing):**
```python
config = OFCControllerConfig.default_effort(duration=3.0)
```

**Default Usage (Smoothness-Minimizing):**
```python
config = OFCControllerConfig.default_smoothness(duration=3.0)
```

**Custom Configuration:**
```python
config = OFCControllerConfig.create(
    mode='effort',  # or 'smoothness'
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=2.5,
    
    # LQR cost weights
    Q_position=[150.0, 150.0],      # Tighter position tracking
    Q_pendulum=[1000.0, 1000.0],   # Stronger pendulum stabilization
    Q_velocity=[20.0, 20.0, 100.0, 100.0],  # More velocity damping
    R=[0.05, 0.05],                 # Lower control penalty = more aggressive
    
    # Impedance parameters
    impedance_mass=2.0,    # Higher virtual mass = slower response
    impedance_kp=150.0,    # Stiffer impedance
    impedance_kd=30.0      # More damping
)
```

## Example: Modifying Global Configurations

In `script_cup_manipulator_controller_drake.py`, you can modify the config creation:

```python
# Default approach
OFC_EFFORT_CONFIG = OFCControllerConfig.default_effort(duration=args.traj_duration)

# Custom approach
OFC_EFFORT_CONFIG = OFCControllerConfig.create(
    mode='effort',
    duration=args.traj_duration,
    impedance_mass=1.5,  # Customize specific parameter
    impedance_kp=120.0
)
```

## Parameter Meanings

### PD/MinJerk/ComputedTorque:
- `kp`: Proportional gains (position tracking stiffness)
- `kd`: Derivative gains (velocity damping)
- `q_start_deg`: Initial joint configuration [link1, link2, pitch, roll] in degrees
- `q_goal_deg`: Goal joint configuration [link1, link2, pitch, roll] in degrees
- `duration`: Time to complete motion (MinJerk only)

### TrajectoryOptimized:
- `num_samples`: Number of optimization knot points (higher = smoother but slower)
- `pendulum_weight`: Cost for pendulum swing (higher = prioritize stability)
- `torque_weight`: Cost for control effort (higher = prioritize energy efficiency)

### OFC:
- `Q_position`: Position tracking importance for manipulator joints
- `Q_pendulum`: Pendulum stabilization importance
- `Q_velocity`: Velocity tracking/damping importance
- `R`: Control effort/smoothness penalty (lower = more aggressive control)
- `impedance_mass`: Virtual mass between force and position (kg)
- `impedance_kp`: Impedance spring stiffness (N/m)
- `impedance_kd`: Impedance damping coefficient (N·s/m)

## Tips

1. **Start with defaults** - They work well for most cases
2. **Tune one parameter at a time** - Use `create()` to modify specific values
3. **Higher impedance mass** = slower, more compliant motion
4. **Lower R values** = more aggressive, less smooth control
5. **Higher Q_pendulum** = prioritize keeping pendulum stable over speed
