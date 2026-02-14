# Controller Configuration Create Methods - Quick Reference

All controller configuration classes have `create()` methods to customize parameters. Angles are specified in **degrees** and automatically converted to radians.

## 1. PDControllerConfig

```python
from script_cup_manipulator_controller_drake import PDControllerConfig

# Use default values
config = PDControllerConfig.default()

# Customize specific parameters
config = PDControllerConfig.create(
    kp=[150.0, 150.0],              # Proportional gains [link1, link2]
    kd=[15.0, 15.0],                # Derivative gains [link1, link2]
    q_start_deg=[90, -170, 0, 180], # Initial config [deg]
    q_goal_deg=[30, -50, 0, 180]    # Goal config [deg]
)
```

## 2. MinJerkControllerConfig

```python
from script_cup_manipulator_controller_drake import MinJerkControllerConfig

# Use default values with custom duration
config = MinJerkControllerConfig.default(duration=5.0)

# Customize all parameters
config = MinJerkControllerConfig.create(
    kp=[100.0, 100.0],
    kd=[10.0, 10.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=4.0                    # Trajectory duration [s]
)
```

## 3. ComputedTorqueControllerConfig

```python
from script_cup_manipulator_controller_drake import ComputedTorqueControllerConfig

# Use default values (smaller gains since feedforward handles dynamics)
config = ComputedTorqueControllerConfig.default()

# Customize parameters
config = ComputedTorqueControllerConfig.create(
    kp=[25.0, 25.0],                # Smaller than PD (feedforward compensates)
    kd=[6.0, 6.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180]
)
```

## 4. TrajectoryOptimizedControllerConfig

```python
from script_cup_manipulator_controller_drake import TrajectoryOptimizedControllerConfig

# Use default values
config = TrajectoryOptimizedControllerConfig.default(duration=3.0)

# Customize optimization parameters
config = TrajectoryOptimizedControllerConfig.create(
    kp=[20.0, 20.0],
    kd=[5.0, 5.0],
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=3.0,
    num_samples=40,                 # More knot points = smoother
    pendulum_weight=200.0,          # Higher = less pendulum swing
    torque_weight=0.05              # Higher = less control effort
)
```

## 5. OFCControllerConfig

```python
from script_cup_manipulator_controller_drake import OFCControllerConfig

# Preset configurations
config = OFCControllerConfig.default_effort(duration=3.0)      # Minimize torque
config = OFCControllerConfig.default_smoothness(duration=3.0)  # Minimize jerk

# Full customization
config = OFCControllerConfig.create(
    mode='effort',                  # 'effort' or 'smoothness'
    q_start_deg=[80, -160, 0, 180],
    q_goal_deg=[20, -40, 0, 180],
    duration=3.0,
    
    # LQR cost weights
    Q_position=[100.0, 100.0],      # Position tracking [link1, link2]
    Q_pendulum=[500.0, 500.0],      # Pendulum angle tracking [pitch, roll]
    Q_velocity=[10.0, 10.0, 50.0, 50.0],  # Velocity tracking
    R=[0.1, 0.1],                   # Control penalty
    
    # Impedance parameters
    impedance_mass=2.0,             # Virtual mass [kg]
    impedance_kp=150.0,             # Stiffness [N/m]
    impedance_kd=25.0               # Damping [N·s/m]
)
```

## Parameter Tuning Tips

### PD / MinJerk / ComputedTorque
- **Higher kp**: Faster response, less tracking error, but may oscillate
- **Higher kd**: More damping, reduces oscillations
- **kp/kd ratio**: Typically 10:1 for critically damped behavior

### Trajectory Optimized
- **num_samples**: 20-50 knot points (more = smoother but slower optimization)
- **pendulum_weight**: 100-500 (higher = less pendulum swing)
- **torque_weight**: 0.01-0.5 (higher = less aggressive control)

### OFC (Optimal Feedback Control)
- **Q_position/Q_pendulum**: Increase to improve tracking (100-1000)
- **Q_velocity**: Increase to reduce velocity overshoots (10-100)
- **R**: Increase to reduce control effort (0.01-1.0)
- **Impedance mass**: 0.5-5.0 kg (higher = smoother but slower response)
- **Impedance kp**: 50-200 N/m (higher = stiffer virtual spring)

## Example: Tuning for Different Behaviors

### Fast, Aggressive Motion
```python
config = PDControllerConfig.create(
    kp=[200.0, 200.0],              # High gains
    kd=[20.0, 20.0],                # Strong damping
    q_goal_deg=[10, -30, 0, 180]    # Large motion
)
```

### Smooth, Gentle Motion
```python
config = OFCControllerConfig.create(
    mode='smoothness',
    impedance_mass=3.0,             # High virtual mass
    impedance_kp=80.0,              # Lower stiffness
    impedance_kd=30.0,              # Higher damping
    duration=5.0                    # Longer duration
)
```

### Minimal Pendulum Swing
```python
config = TrajectoryOptimizedControllerConfig.create(
    pendulum_weight=500.0,          # Very high weight on pendulum
    torque_weight=0.01,             # Allow high torques if needed
    num_samples=50                  # Fine-grained trajectory
)
```

## Integration with Main Script

The configs are used in the global configuration section:

```python
# In script_cup_manipulator_controller_drake.py (around line 499-520)
PD_CONFIG = PDControllerConfig.default()
MIN_JERK_CONFIG = MinJerkControllerConfig.default(duration=args.traj_duration)
COMPUTED_TORQUE_CONFIG = ComputedTorqueControllerConfig.default()
TRAJECTORY_OPTIMIZED_CONFIG = TrajectoryOptimizedControllerConfig.default(duration=args.traj_duration)
OFC_EFFORT_CONFIG = OFCControllerConfig.default_effort(duration=args.traj_duration)
OFC_SMOOTHNESS_CONFIG = OFCControllerConfig.default_smoothness(duration=args.traj_duration)
```

To customize, replace `default()` with `create()`:

```python
# Custom PD configuration
PD_CONFIG = PDControllerConfig.create(
    kp=[150.0, 150.0],
    kd=[15.0, 15.0]
)

# Custom OFC configuration
OFC_EFFORT_CONFIG = OFCControllerConfig.create(
    mode='effort',
    impedance_mass=2.0,
    Q_pendulum=[1000.0, 1000.0]     # Very high pendulum stabilization
)
```
