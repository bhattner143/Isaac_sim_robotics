# Jacobian Computation in Drake - Implementation Guide

## What Was Added

Added three Jacobian computation methods to the `PlanarManipulator` class:

### 1. **Analytical Jacobian** (Recommended)
```python
compute_jacobian_analytical(theta1, theta2) -> np.ndarray
```

**How it works:**
- Uses closed-form derivatives from forward kinematics
- For 2-DOF planar arm in XY plane:
  - Position: `x = L1*cos(θ1) + L2*cos(θ1+θ2)`
  - Position: `y = L1*sin(θ1) + L2*sin(θ1+θ2)`
  - Z is constant (0.5m from base)

**Derivatives computed:**
```
∂x/∂θ1 = -L1*sin(θ1) - L2*sin(θ1+θ2)
∂x/∂θ2 = -L2*sin(θ1+θ2)
∂y/∂θ1 = L1*cos(θ1) + L2*cos(θ1+θ2)
∂y/∂θ2 = L2*cos(θ1+θ2)
∂z/∂θ1 = 0 (constant Z)
∂z/∂θ2 = 0 (constant Z)
```

**Advantages:**
- ✅ Fast (no numerical perturbations needed)
- ✅ Exact (no approximation errors)
- ✅ Efficient for real-time control
- ✅ Used by default in impedance control

### 2. **Numerical Jacobian** (For Verification)
```python
compute_jacobian_numerical(theta1, theta2, eps=1e-6) -> np.ndarray
```

**How it works:**
- Uses finite differences: `J[i,j] = (f(θ + δθ) - f(θ)) / δθ`
- Perturbs each joint by small amount (default: 1e-6 radians)
- Computes change in EE position for each perturbation

**Advantages:**
- ✅ General (works without deriving closed-form expressions)
- ✅ Good for verification (compare against analytical)
- ✅ Automatically works if kinematics change

**Disadvantages:**
- ❌ Slower (multiple function evaluations)
- ❌ Approximation error from finite differences
- ❌ Sensitive to perturbation size selection

### 3. **Unified Jacobian Method** (User Interface)
```python
compute_jacobian(theta1, theta2, method="analytical") -> np.ndarray
```

**Usage:**
```python
# Get analytical Jacobian (recommended)
J_analytical = manipulator.compute_jacobian(theta1, theta2, method="analytical")

# Get numerical Jacobian (for testing)
J_numerical = manipulator.compute_jacobian(theta1, theta2, method="numerical")

# Default is analytical
J = manipulator.compute_jacobian(theta1, theta2)
```

## Drake's Native Jacobian Support

### What Drake Provides

Drake's `MultibodyPlant` has **native Jacobian computation**, but it's complex to use:

```python
# Drake's Jacobian method (if needed)
plant.CalcJacobianSpatialVelocity(
    context, 
    JacobianWrtVariable.kV,  # Jacobian w.r.t. velocities
    frame_on_body,
    position_vector,
    base_frame,
    expressed_in_frame
) -> Matrix
```

### Why We Implemented Custom Methods Instead

1. **Simplicity**: Our 2-DOF planar arm is simple enough for analytical solution
2. **Transparency**: You can see and understand the math
3. **Efficiency**: Analytical is faster than Drake's generic computation
4. **Control**: Full control over which method to use (analytical vs numerical)

### If You Need Drake's Jacobian for Complex Robots

For a full 6-DOF or URDF-based manipulator, use:

```python
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.math import JacobianWrtVariable

# Get Jacobian from Drake
J_drake = plant.CalcJacobianSpatialVelocity(
    context=context,
    with_respect_to=JacobianWrtVariable.kV,  # w.r.t. generalized velocities
    frame_B=end_effector_frame,
    p_BP=np.zeros(3),  # Position vector in frame B
    frame_A=world_frame,
    frame_E=world_frame
)
```

## Usage in Impedance Control

The Jacobian is used to convert task-space forces to joint torques:

```python
# Task-space impedance control
F_task = Kp * ee_error + Kd * (-ee_velocity)

# Get Jacobian
J = manipulator.compute_jacobian(theta1, theta2)

# Convert to joint torques using transpose method
tau = J.T @ F_task
```

This is known as the **Jacobian Transpose Method** - simple but effective for impedance control.

## Comparing Methods

| Aspect | Analytical | Numerical | Drake Native |
|--------|-----------|-----------|--------------|
| Speed | ⚡⚡⚡ Fast | ⚡ Slow | ⚡⚡ Medium |
| Accuracy | ✅ Exact | ⚠️ Approximate | ✅ Exact |
| Complexity | 📚 Medium | 📚 Simple | 📚📚 High |
| General | ❌ 2-DOF only | ✅ Works anywhere | ✅ Works anywhere |
| Real-time | ✅ Yes | ⚠️ Maybe | ✅ Yes |
| Learning | ✅ Great | ✅ Great | ❌ Complex |

## Testing & Verification

To verify our analytical solution against numerical:

```python
# Get both
J_analytical = manipulator.compute_jacobian(theta1, theta2, method="analytical")
J_numerical = manipulator.compute_jacobian(theta1, theta2, method="numerical")

# Should be nearly identical
error = np.linalg.norm(J_analytical - J_numerical)
print(f"Jacobian difference: {error:.2e}")  # Should be < 1e-5
```

## Advanced: Singularities

When the manipulator reaches a singularity (fully extended or fully folded), the Jacobian determinant becomes zero:

```python
det_J = np.linalg.det(J[:2, :2])  # Use 2x2 sub-matrix for 2-DOF arm
if abs(det_J) < 1e-6:
    print("Near singularity!")
    # Use damped least-squares instead of transpose
    lambda_damp = 0.01
    tau = J.T @ np.linalg.inv(J @ J.T + lambda_damp**2 * np.eye(3)) @ F_task
```

## Summary

✅ **Added:** Three Jacobian methods to `PlanarManipulator`
- Analytical (fast, exact) - **RECOMMENDED**
- Numerical (general, slow)
- Unified interface

✅ **Updated:** Impedance control now uses analytical Jacobian

✅ **Available:** Drake native Jacobian for complex robots

Use `compute_jacobian()` for any future task-space control needs!
