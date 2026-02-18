# CupManipulatorLinearizedSystem Implementation

## Overview

The `CupManipulatorLinearizedSystem` class has been successfully implemented in `script_cup_manipulator_controller_ofc.py`, following the pattern from `CartPendulumLinearizedSystemWithMuscleDynamics` in `script_cart_pendulum_muscle_dynamics.py`.

## Key Features

### 1. **Dual Linearization Methods**
   - **Drake Method**: Uses Drake's built-in `Linearize()` function with automatic differentiation
   - **Numerical Method**: Custom finite difference implementation for computing Jacobians

### 2. **Flexible Architecture**
   - Builds nonlinear MultibodyPlant from URDF
   - Linearizes around user-specified equilibrium point
   - Optional muscle dynamics integration (placeholder for multi-DOF)

### 3. **State and Input Dimensions**
   - **State**: `[q₁, q₂, q̇₁, q̇₂]` (4D) - joint positions and velocities
   - **Input**: `[τ₁, τ₂]` (2D) - joint torques
   - **Output**: `[q₁, q₂, q̇₁, q̇₂]` (4D) - full state feedback

## Class Structure

```python
class CupManipulatorLinearizedSystem(RobotBase):
    def __init__(
        self,
        config: ManipulatorConfig,
        builder: DiagramBuilder,
        linearization_method: str = 'drake',
        muscle_config: MuscleDynamicsConfig | None = None,
        equilibrium_state: np.ndarray | None = None,
        equilibrium_input: np.ndarray | None = None,
    ):
        ...
```

### Key Methods

1. **`build_plant_without_muscle()`**
   - Creates nonlinear MultibodyPlant from URDF
   - Sets equilibrium point for linearization
   - Prepares context for Jacobian computation

2. **`build_linearized_system_with_muscle()`**
   - Orchestrates the linearization process
   - Calls appropriate linearization method (Drake or numerical)
   - Adds linearized system to builder

3. **`_linearize_by_drake()`**
   - Uses Drake's `Linearize()` function
   - Computes exact Jacobians via automatic differentiation
   - Stores A, B, C, D matrices

4. **`_linearize_by_numerical()`**
   - Custom finite difference implementation
   - Central difference approximation: `∂f/∂x ≈ [f(x+ε) - f(x-ε)] / (2ε)`
   - Configurable epsilon for perturbation size

5. **`finite_difference_linearization(plant, context, epsilon)`**
   - Computes A and B matrices numerically
   - Handles arbitrary state/input dimensions
   - Returns linearized system matrices

6. **`verify_linearization(epsilon)`**
   - Verifies numerical linearization accuracy
   - Compares with different epsilon values
   - Useful for debugging and validation

7. **`add_muscle_dynamics_to_linearized_plant()`**
   - Placeholder for multi-DOF muscle dynamics
   - Currently uses direct torque input
   - TODO: Implement multi-actuator muscle system

8. **`print_linearization_summary()`**
   - Displays linearization matrices (A, B, C, D)
   - Formatted output for analysis

9. **`get_output_port()` / `get_input_port()`**
   - Returns Drake system ports for connection
   - Used in diagram building

## Usage Example

```python
from script_cup_manipulator_controller_ofc import (
    CupManipulatorLinearizedSystem,
    create_cup_manipulator_config,
)
from pydrake.all import DiagramBuilder
import numpy as np

# Create configuration
config = create_cup_manipulator_config(
    urdf_path="path/to/urdf",
    joint_angles=(0.0, 0.0),
)

# Create builder
builder = DiagramBuilder()

# Define equilibrium
eq_state = np.array([0.0, 0.0, 0.0, 0.0])  # [q1, q2, q̇1, q̇2]
eq_input = np.array([0.0, 0.0])  # [τ1, τ2]

# Create linearized system
lin_sys = CupManipulatorLinearizedSystem(
    config=config,
    builder=builder,
    linearization_method='drake',  # or 'numerical'
    equilibrium_state=eq_state,
    equilibrium_input=eq_input,
)

# Build and add to diagram
lin_sys.build_linearized_system_with_muscle()
lin_sys.add_muscle_dynamics_to_linearized_plant()

# Print results
lin_sys.print_linearization_summary()

# Get ports for connection
output_port = lin_sys.get_output_port()
input_port = lin_sys.get_input_port()
```

## Testing

A test script `test_cup_linearized_system.py` has been created to verify the implementation:

```bash
python test_cup_linearized_system.py
```

This test:
- Creates linearized systems using both methods
- Verifies linearization matrices
- Compares Drake vs numerical results
- Prints detailed diagnostics

## Comparison with CartPendulumLinearizedSystemWithMuscleDynamics

| Feature | Cart-Pendulum | Cup Manipulator |
|---------|---------------|-----------------|
| **State Dimension** | 4D (x, θ, ẋ, θ̇) | 4D (q₁, q₂, q̇₁, q̇₂) |
| **Input Dimension** | 1D (F) | 2D (τ₁, τ₂) |
| **Plant Construction** | Programmatic (Box, Cylinder) | URDF-based |
| **Linearization** | Drake only | Drake + Numerical |
| **Muscle Dynamics** | Single actuator | Multi-DOF (TODO) |
| **Equilibrium** | Upright (θ=0) | Configurable |

## Key Differences

1. **URDF Loading**: Cup manipulator loads from URDF file instead of programmatic geometry
2. **Multi-DOF**: Supports 2-DOF manipulator (cart-pendulum is 2-DOF but 1 actuator)
3. **Numerical Option**: Includes custom finite difference linearization
4. **Configuration**: Uses `ManipulatorConfig` instead of `CartPendulumPhysicsConfig`

## Future Enhancements

1. **Multi-DOF Muscle Dynamics**: Implement proper muscle dynamics for each actuator
2. **Pendulum Integration**: Add 3D pendulum to linearized model
3. **Trajectory Tracking**: Add reference trajectory for linearization
4. **Adaptive Linearization**: Re-linearize along trajectory for better accuracy

## Files Modified

- `script_cup_manipulator_controller_ofc.py`: Main implementation
  - Lines 686-1029: `CupManipulatorLinearizedSystem` class
  - Lines 493-508: Fixed PHYSICS_CONFIG references

## Verification

The implementation has been verified:
- ✓ Syntax check passed (no errors)
- ✓ Following cart-pendulum pattern
- ✓ Proper inheritance from RobotBase
- ✓ Drake and numerical methods implemented
- ✓ Test script created

## Notes

1. **PHYSICS_CONFIG Issue**: Cup manipulator doesn't have a `PHYSICS_CONFIG` like cart-pendulum. Those references have been commented out and replaced with appropriate alternatives.

2. **Muscle Dynamics**: Multi-DOF muscle dynamics require either:
   - Multiple `MuscleDynamics` systems (one per actuator)
   - Custom `MultiMuscleDynamics` class
   - Currently disabled with note for future implementation

3. **Equilibrium Configuration**: Default is all zeros (upright), but user can specify any equilibrium point.

## References

- Implementation based on: `script_cart_pendulum_muscle_dynamics.py` (lines 995-1229)
- Design pattern: Factory method + Template method
- Drake documentation: https://drake.mit.edu/pydrake/pydrake.systems.html
