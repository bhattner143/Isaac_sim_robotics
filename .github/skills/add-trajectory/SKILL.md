---
name: add-trajectory
description: "Add a new trajectory type to the controller/trajectory module. Use when: creating custom motion paths (figure-8, spiral, Lissajous, etc.), modifying existing trajectories, or adding trajectory parameters."
argument-hint: "Describe the trajectory shape (e.g., 'figure-8 pattern' or 'spiral outward')"
---

# Add a New Trajectory Type

## When to Use
- Need a new EE motion pattern (figure-8, spiral, Lissajous, etc.)
- Extending the trajectory system for new experiments

## Existing Trajectories

All defined in `controller/trajectory.py`:

| Class | Shape | Parameters |
|-------|-------|-----------|
| `RectTrajectory` | Rectangle with rounded corners | `x_range`, `y_range`, `lap_duration`, `v_corner`, `corner_blend` |
| `CircleTrajectory` | Circle | `cx`, `cy`, `radius`, `lap_duration` |
| `LineTrajectory` | Back-and-forth line | `cx`, `cy`, `radius`, `lap_duration` |

## Required Interface

Every trajectory class MUST implement these three methods:

```python
class MyTrajectory:
    def eval_position(self, t: float) -> np.ndarray:
        """Return [x, y] at time t."""
        ...
    
    def eval_velocity(self, t: float) -> np.ndarray:
        """Return [ẋ, ẏ] at time t."""
        ...
    
    def eval_acceleration(self, t: float) -> np.ndarray:
        """Return [ẍ, ÿ] at time t."""
        ...
```

All three are needed because the CT controller uses position, velocity, AND acceleration references for feedforward.

## Procedure

### 1. Add the class to `controller/trajectory.py`
Follow the pattern of existing classes. Key requirements:
- Periodic: `t` wraps via `t_mod = t % lap_duration`
- C² continuous: position, velocity, acceleration must be smooth at the wrap point
- Stay within robot workspace: typical EE range is x ∈ [0.35, 0.55], y ∈ [-0.15, 0.15] meters

### 2. Add a spec entry in the multi-instance script
In `test_cup_manipulator_tendon_multi_instance_isaac_sim.py`, add to `_TRAJ_SPECS`:
```python
{"type": "my_traj", "lap": 8.0, "cx": 0.42, "cy": 0.00, "radius": 0.08},
```

### 3. Handle it in `_make_trajectory()`
```python
if traj_type == "my_traj":
    return MyTrajectory(cx=spec["cx"], cy=spec["cy"], ...)
```

### 4. Export from `__init__` if needed
Update `controller/__init__.py` if the class should be importable as `from controller.trajectory import MyTrajectory`.

## C² Continuity Check
Verify smoothness by plotting derivatives:
```python
t = np.linspace(0, traj.lap_duration, 1000)
pos = np.array([traj.eval_position(ti) for ti in t])
vel = np.array([traj.eval_velocity(ti) for ti in t])
acc = np.array([traj.eval_acceleration(ti) for ti in t])
# Check: vel ≈ np.gradient(pos, t, axis=0)
# Check: no jumps at t=0 / t=lap_duration
```
