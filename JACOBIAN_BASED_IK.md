# Jacobian-Based Inverse Kinematics Implementation

## Overview

Added three IK methods to the `PlanarManipulator` class:

1. **Analytical IK** (closed-form) - Original
2. **Jacobian-Based IK** (iterative) - NEW
3. **Hybrid IK** (try both) - NEW

## Method 1: Analytical Inverse Kinematics (Original)

```python
result = manipulator.inverse_kinematics(target_x, target_y, target_z)
```

**How it works:**
- Uses Law of Cosines to solve directly
- Closed-form solution (no iteration)
- Only works for 2-DOF planar arm in XZ plane

**Advantages:**
- ✅ Very fast (no iteration)
- ✅ Exact solution
- ✅ Always converges if reachable

**Disadvantages:**
- ❌ Only works for specific arm configurations
- ❌ Limited to XZ plane
- ❌ Returns None if unreachable

---

## Method 2: Jacobian-Based IK (NEW - Iterative)

```python
result = manipulator.inverse_kinematics_jacobian(
    target_x, target_y, target_z,
    theta1_init=0.0,
    theta2_init=0.0,
    method="damped_ls",
    max_iterations=50,
    tolerance=1e-6,
    alpha=0.1
)
```

### Algorithm Options

#### Option A: Jacobian Transpose Method

```python
method = "transpose"
```

**Control law:**
```
Δθ = α * J^T * Δx
```

Where:
- `J` = Jacobian matrix (3×2)
- `Δx` = position error (3×1)
- `α` = step size / learning rate (0.0 to 1.0)
- `Δθ` = joint velocity update

**Advantages:**
- ✅ Simple to understand
- ✅ Fast computation
- ✅ Intuitive

**Disadvantages:**
- ❌ Can diverge near singularities
- ❌ Slow convergence
- ❌ Not true least-squares solution

**When to use:**
- Quick rough solutions
- Well-behaved configurations
- Prototyping/testing

---

#### Option B: Damped Least-Squares / Levenberg-Marquardt

```python
method = "damped_ls"  # RECOMMENDED
```

**Control law:**
```
Δθ = J^T * (J*J^T + λ²I)^(-1) * Δx
```

Where:
- `λ` = damping factor (controls singularity robustness)
- `I` = identity matrix
- Larger `λ` = more damping (smoother, slower)
- Smaller `λ` = less damping (faster, less stable)

**Mathematical intuition:**
```
Pseudo-inverse with regularization:
J_pinv_damped = J^T * (J*J^T + λ²I)^(-1)
Δθ = J_pinv_damped * Δx
```

**Advantages:**
- ✅ Handles singularities gracefully
- ✅ More stable convergence
- ✅ True least-squares solution
- ✅ Recommended for robustness

**Disadvantages:**
- ⚠️ Slightly more computation
- ⚠️ Damping introduces tracking error
- ⚠️ Need to tune damping factor

**When to use:**
- Production code
- Near singularities
- High reliability required
- Most applications

**Tuning damping:**
```python
# More robust (slower near singularities)
result = manipulator.inverse_kinematics_jacobian(
    target_x, target_y, target_z,
    method="damped_ls"
)

# To adjust damping factor (inside code):
# Change lambda_damp = 0.01 to higher (more robust) or lower (faster)
```

---

## Method 3: Hybrid IK (NEW - Smart Selection)

```python
result = manipulator.inverse_kinematics_hybrid(
    target_x, target_y, target_z,
    try_analytical=True
)
```

**Strategy:**
1. Try **analytical IK** first (fast if available)
2. If it fails, use **Jacobian-based IK** (general fallback)
3. Return whichever works

**Advantages:**
- ✅ Best of both worlds
- ✅ Fast when possible
- ✅ Falls back gracefully
- ✅ Works everywhere

**Disadvantages:**
- ⚠️ Slightly more complex logic
- ⚠️ May return different quality solutions

**When to use:**
- Always! (if you want best performance)
- General-purpose code

---

## Comparison Table

| Feature | Analytical | Jacobian (Transpose) | Jacobian (Damped LS) | Hybrid |
|---------|-----------|----------------------|----------------------|--------|
| Speed | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| Accuracy | ✅ Perfect | ⚠️ Approximate | ✅ Good | ✅ Best available |
| Singularity handling | ❌ None | ⚠️ Can diverge | ✅ Robust | ✅ Robust |
| Iteration needed | ❌ No | ✅ Yes | ✅ Yes | ✅ Only if needed |
| Works everywhere | ❌ Limited | ✅ Yes | ✅ Yes | ✅ Yes |
| Reliability | 🟢 High* | 🟡 Medium | 🟢 High | 🟢 High |
| Code complexity | 📚 Medium | 📚 Medium | 📚 Medium | 📚 High |

*Analytical: High when solution exists, fails otherwise

---

## Usage Examples

### Example 1: Simple Reaching (Use Analytical)

```python
# Try to reach position
target = (-2.5, 0.0, 0.8)
result = manipulator.inverse_kinematics(*target)

if result:
    theta1, theta2 = result
    print(f"Analytical IK: θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°")
else:
    print("Target unreachable")
```

### Example 2: Iterative IK (Guaranteed Solution)

```python
# Jacobian-based IK - always tries (may not be perfect)
result = manipulator.inverse_kinematics_jacobian(
    target_x=-2.5, 
    target_y=0.0, 
    target_z=0.8,
    theta1_init=0.0,
    theta2_init=0.0,
    method="damped_ls",
    max_iterations=50
)

if result:
    print("Iterative IK converged")
else:
    print("Failed to converge")
```

### Example 3: Trajectory Tracking (Use Hybrid)

```python
# Track a trajectory with automatic best method
for t in trajectory_times:
    x_desired = trajectory_x(t)
    y_desired = trajectory_y(t)
    z_desired = trajectory_z(t)
    
    # Hybrid: tries analytical first, falls back to iterative
    result = manipulator.inverse_kinematics_hybrid(
        x_desired, y_desired, z_desired,
        try_analytical=True
    )
    
    if result:
        theta1, theta2 = result
        set_joint_positions([theta1, theta2])
```

### Example 4: Debug/Verify Solutions

```python
target = (-2.5, 0.0, 0.8)

# Get analytical solution
result_analytical = manipulator.inverse_kinematics(*target)

# Get iterative solution  
result_iterative = manipulator.inverse_kinematics_jacobian(
    *target,
    method="damped_ls"
)

# Compare
if result_analytical and result_iterative:
    theta1_a, theta2_a = result_analytical
    theta1_i, theta2_i = result_iterative
    
    print(f"Analytical:  θ1={np.degrees(theta1_a):.2f}°, θ2={np.degrees(theta2_a):.2f}°")
    print(f"Iterative:   θ1={np.degrees(theta1_i):.2f}°, θ2={np.degrees(theta2_i):.2f}°")
    print(f"Difference:  Δθ1={np.degrees(theta1_a-theta1_i):.4f}°")
```

---

## Mathematical Background

### Jacobian Matrix

For our 2-DOF planar arm:
```
Forward kinematics:
x = L1*cos(θ1) + L2*cos(θ1+θ2)
y = L1*sin(θ1) + L2*sin(θ1+θ2)
z = constant

Jacobian (analytical derivatives):
J = [∂x/∂θ1  ∂x/∂θ2]   [-L1*sin(θ1)-L2*sin(θ1+θ2)  -L2*sin(θ1+θ2)    ]
    [∂y/∂θ1  ∂y/∂θ2] = [L1*cos(θ1)+L2*cos(θ1+θ2)   L2*cos(θ1+θ2)     ]
    [∂z/∂θ1  ∂z/∂θ2]   [0                             0                  ]
```

### Iteration Update Rules

**Jacobian Transpose:**
```
Δθ = α * J^T * error
θ_new = θ_old + Δθ
```

**Damped Least-Squares:**
```
Δθ = J^T * (J*J^T + λ²I)^(-1) * error
θ_new = θ_old + Δθ
```

### Singularities

Singularities occur when `det(J*J^T) ≈ 0`:
- Arm fully extended (θ2 ≈ 0)
- Arm folded (θ2 ≈ -π)

Damping helps by adding regularization:
```
det(J*J^T + λ²I) ≥ λ² > 0  (always invertible)
```

---

## Convergence Tips

1. **Choose good initial guess**: Closer to solution = faster convergence
   ```python
   # Bad: arbitrary initial guess
   theta1_init = 0.0
   
   # Better: use previous solution or analytical IK
   theta1_init = last_theta1
   ```

2. **Tune max_iterations**: Balance speed vs convergence
   ```python
   max_iterations=10   # Fast, may not converge
   max_iterations=50   # Good balance (default)
   max_iterations=100  # Thorough, slower
   ```

3. **Tune tolerance**: How accurate do you need?
   ```python
   tolerance=1e-3   # Coarse (1mm)
   tolerance=1e-6   # Fine (1 micrometer)
   ```

4. **Monitor convergence**: Check for divergence
   ```python
   if result is None:
       print("IK failed to converge - check initial guess or tolerance")
   ```

---

## Advanced: Extending to Other Robots

To use Jacobian IK on other robots:

1. **Implement forward kinematics**: Must compute EE position from joints
2. **Implement/get Jacobian**: Analytical or numerical
3. **Use same algorithm**: Jacobian transpose or damped LS

For Drake robots with URDF:
```python
# Use Drake's native methods instead:
from pydrake.multibody.inverse_kinematics import InverseKinematics

ik = InverseKinematics(plant)
# ... configure constraints ...
result = ik.Solve()
```

---

## Summary

✅ **Added:** Jacobian-based IK methods to `PlanarManipulator`
- Jacobian Transpose: Simple, fast
- Damped Least-Squares: Robust, recommended
- Hybrid: Best of both worlds

✅ **Use cases:**
- Trajectory tracking
- Singularity avoidance
- General-purpose reaching
- Educational understanding

✅ **Key insight:** Jacobian IK trades exact solutions for generality and robustness
