# Complete Methods Summary - PlanarManipulator Class

## Kinematics Methods

### Forward Kinematics
```python
x, z = manipulator.forward_kinematics(theta1, theta2)
```
Computes end-effector position from joint angles.

---

### Inverse Kinematics - Analytical (Closed-Form)
```python
theta1, theta2 = manipulator.inverse_kinematics(
    target_x, target_y, target_z
)
```
**Pros:** Instant, exact solution
**Cons:** Limited to XZ plane, returns None if unreachable

---

### Inverse Kinematics - Jacobian-Based (Iterative)
```python
theta1, theta2 = manipulator.inverse_kinematics_jacobian(
    target_x, target_y, target_z,
    theta1_init=0.0,
    theta2_init=0.0,
    method="damped_ls",      # "transpose" or "damped_ls"
    max_iterations=50,
    tolerance=1e-6,
    alpha=0.1
)
```
**Methods:**
- `"transpose"`: Simple, fast, can diverge at singularities
- `"damped_ls"`: Robust, handles singularities (RECOMMENDED)

**Pros:** Works anywhere, robust to singularities
**Cons:** Iterative (slower), approximate solution

---

### Inverse Kinematics - Hybrid (Smart)
```python
theta1, theta2 = manipulator.inverse_kinematics_hybrid(
    target_x, target_y, target_z,
    try_analytical=True
)
```
**How it works:**
1. Try analytical IK (fast)
2. Fall back to damped LS (robust)
3. Return whichever works

**Pros:** Best performance, fallback safety, always tries to solve
**Cons:** Slightly more complex

---

## Jacobian Methods

### Compute Jacobian - Analytical
```python
J = manipulator.compute_jacobian_analytical(theta1, theta2)
# Returns: 3x2 matrix
```
Analytical derivatives from closed-form FK.
- Fast ⚡⚡⚡
- Exact ✅
- Default method

---

### Compute Jacobian - Numerical
```python
J = manipulator.compute_jacobian_numerical(theta1, theta2, eps=1e-6)
# Returns: 3x2 matrix
```
Finite differences for verification.
- General ✅
- Slower ⚡

---

### Compute Jacobian - Unified Interface
```python
J = manipulator.compute_jacobian(
    theta1, theta2,
    method="analytical"  # or "numerical"
)
# Returns: 3x2 matrix
```
Wrapper supporting both methods.

---

## Position Methods

### Get EE World Position
```python
x, y, z = manipulator.get_ee_world_position(plant, context)
```
Get current end-effector position in world frame.

---

### Get Joint Positions
```python
positions = manipulator.get_joint_positions(plant, context)
# Returns: [theta1, theta2]
```

---

### Set Joint Positions
```python
success = manipulator.set_joint_positions(plant, context, [theta1, theta2])
```

---

## Coordinate Transformation

### Transform Point: World to Base
```python
x_rel, y_rel, z_rel = manipulator.transform_point_world_to_base(
    world_x, world_y, world_z
)
```

---

## URDF/Drake Integration

### Load URDF
```python
manipulator.load_urdf_to_plant(plant, parser)
```

---

## Quick Reference Table

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `forward_kinematics` | θ1, θ2 | x, z | FK analysis |
| `inverse_kinematics` | x, y, z | θ1, θ2 | Fast IK (XZ plane) |
| `inverse_kinematics_jacobian` | x, y, z | θ1, θ2 | General IK |
| `inverse_kinematics_hybrid` | x, y, z | θ1, θ2 | Best overall ⭐ |
| `compute_jacobian_analytical` | θ1, θ2 | 3×2 matrix | Task-space control |
| `compute_jacobian_numerical` | θ1, θ2 | 3×2 matrix | Verification |
| `compute_jacobian` | θ1, θ2 | 3×2 matrix | Generic use |
| `get_ee_world_position` | plant, ctx | x, y, z | Get current pos |
| `get_joint_positions` | plant, ctx | [θ1, θ2] | Get current angles |
| `set_joint_positions` | plant, ctx, [θ1, θ2] | success | Set joint angles |

---

## Recommendation by Use Case

### Trajectory Tracking
```python
# Each point on trajectory
for point in trajectory:
    # Use hybrid for best performance
    angles = manipulator.inverse_kinematics_hybrid(
        point.x, point.y, point.z
    )
```

### Real-Time Control with Impedance
```python
# In control loop
J = manipulator.compute_jacobian(theta1, theta2)
tau = J.T @ F_task
```

### Reaching Single Target
```python
# Quick reach
angles = manipulator.inverse_kinematics_hybrid(target_x, target_y, target_z)
```

### Research/Education
```python
# Compare all methods
analytical = manipulator.inverse_kinematics(x, y, z)
transpose = manipulator.inverse_kinematics_jacobian(x, y, z, method="transpose")
damped = manipulator.inverse_kinematics_jacobian(x, y, z, method="damped_ls")
```

### Singularity Analysis
```python
J = manipulator.compute_jacobian(theta1, theta2)
det = np.linalg.det(J[:2, :2])
if abs(det) < 1e-3:
    print("Near singularity!")
```

---

## Architecture Overview

```
PlanarManipulator
├── Forward Kinematics
│   └── forward_kinematics(θ1, θ2) → (x, z)
│
├── Inverse Kinematics (4 methods)
│   ├── inverse_kinematics (analytical, fast)
│   ├── inverse_kinematics_jacobian (iterative)
│   │   ├── method="transpose" (simple)
│   │   └── method="damped_ls" (robust)
│   └── inverse_kinematics_hybrid (smart)
│
├── Jacobian (3 methods)
│   ├── compute_jacobian_analytical (fast)
│   ├── compute_jacobian_numerical (general)
│   └── compute_jacobian (wrapper)
│
├── Position Queries
│   ├── get_ee_world_position()
│   ├── get_joint_positions()
│   └── set_joint_positions()
│
└── Coordinate Transforms
    └── transform_point_world_to_base()
```

---

## Data Flow Example: Trajectory Tracking with Impedance Control

```python
# Get desired position from trajectory
x_d, y_d, z_d = get_next_trajectory_point()

# Get current joint angles
theta1, theta2 = manipulator.get_joint_positions(plant, context)

# Get current EE position
x_cur, y_cur, z_cur = manipulator.get_ee_world_position(plant, context)

# Compute Jacobian
J = manipulator.compute_jacobian(theta1, theta2)

# Compute impedance control force
error = np.array([x_d - x_cur, y_d - y_cur, z_d - z_cur])
F = Kp * error + Kd * (-v_cur)

# Convert to joint torques
tau = J.T @ F

# Apply torques
plant.get_actuation_input_port().FixValue(context, tau)
```

---

## Performance Characteristics

| Method | Time | Space | Complexity |
|--------|------|-------|-----------|
| Forward Kinematics | 0.1 μs | O(1) | O(1) |
| Analytical IK | 1 μs | O(1) | O(1) |
| Jacobian (analytical) | 1 μs | O(1) | O(1) |
| Jacobian (numerical) | 10 μs | O(1) | O(1) |
| IK (transpose, 10 iter) | 50 μs | O(1) | O(n) |
| IK (damped LS, 10 iter) | 100 μs | O(1) | O(n) |

*Estimates for 2-DOF arm on modern CPU*

---

## Error Handling

All methods return:
- **Success:** Proper tuple/matrix
- **Failure:** `None` (IK methods) or fallback value

Always check:
```python
result = manipulator.some_ik_method(...)
if result is not None:
    theta1, theta2 = result
    # use it
else:
    print("IK failed!")
```
