# CartPendulum3D Class Documentation

## Overview

`CartPendulum3D` is a reusable class for creating a cart-pendulum 3D system with 2D cart motion and gimbal-mounted pendulum. This class was extracted from test code to provide a clean, modular interface for cart-pendulum systems.

## System Architecture

```
SYSTEM STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cart: 2 DOF (x, y position in horizontal plane)
├── Actuated by forces [F_x, F_y]
└── Mass: configurable

Pendulum: 2 DOF (pitch, roll gimbal angles)
├── Attached to cart at specified offset
└── Passive (no direct actuation)

Total System: 4 DOF → 8D state [x, y, α, β, ẋ, ẏ, α̇, β̇]
```

## Coupling Dynamics

- Cart acceleration affects pendulum motion (inertial coupling)
- Pendulum motion creates reaction forces on cart
- Full nonlinear coupled dynamics via Drake's MultibodyPlant

## Coordinate System

- **x, y**: Cart position in horizontal plane (m)
- **α (pitch)**: Pendulum rotation about Y-axis (rad)
- **β (roll)**: Pendulum rotation about X-axis (rad)
- **Zero angles**: Pendulum hanging down

## Usage Example

```python
from script_cup_manipulator_controller_ofc import CartPendulum3D
from robot_types import PendulumConfig

# Create pendulum configuration
pendulum_config = PendulumConfig(
    mass=0.5,
    length=0.2,
    radius=0.05,
    damping=0.0,
    attachment_point=(0.0, 0.0, 0.0),
    initial_pitch=0.0,
    initial_roll=0.0,
)

# Create cart-pendulum system
cart_pendulum = CartPendulum3D(
    pendulum_config=pendulum_config,
    cart_mass=1.0,
    cart_size=0.1
)

# Attach to Drake plant
plant = MultibodyPlant(time_step=0.0)
model_instance = plant.AddModelInstance("cart_pendulum")
cart_pendulum.attach_to_plant(plant, model_instance)
plant.Finalize()

# Set state
context = plant.CreateDefaultContext()
cart_pendulum.set_cart_state(context, x=0.1, y=0.2, x_dot=0.0, y_dot=0.0)
cart_pendulum.set_pendulum_state(context, pitch=0.1, roll=0.0, pitch_dot=0.0, roll_dot=0.0)

# Get state
full_state = cart_pendulum.get_full_state(context)  # 8D vector

# Linearize system
A, B = cart_pendulum.finite_difference_linearization(plant, context, epsilon=1e-6)
```

## Key Methods

### Constructor

```python
def __init__(self, pendulum_config, cart_mass=1.0, cart_size=0.1)
```

### System Building

```python
def attach_to_plant(self, plant, model_instance)
```
Creates cart body, prismatic joints, pendulum, and actuators.

### State Management

```python
def set_cart_state(self, context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0)
def set_pendulum_state(self, context, pitch=0.0, roll=0.0, pitch_dot=0.0, roll_dot=0.0)
def get_cart_state(self, context) → np.ndarray  # [x, y, x_dot, y_dot]
def get_pendulum_state(self, context) → np.ndarray  # [pitch, roll, pitch_dot, roll_dot]
def get_full_state(self, context) → np.ndarray  # 8D state vector
```

### Linearization

```python
def finite_difference_linearization(self, plant, context, epsilon=1e-6)
```
Returns (A, B) matrices using numerical finite differences.

## Implementation Details

### Joint Chain
```
world --[prismatic x]--> x_slider --[prismatic y]--> cart --[gimbal]--> pendulum
```

### Inertia Properties
- **Cart**: Cube with moment of inertia I = (1/6) * m * s²
- **x_slider**: Negligible mass (0.001 kg) to avoid singularities
- **Pendulum**: Cylinder with configurable mass, length, radius

## Validation

The class has been validated against Drake's automatic differentiation:

```
Test Results (from test_pendulum_linearization.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Hanging Down (θ=0°):
  - A matrix error: 4.56e-11 (machine precision)
  - B matrix error: 8.88e-16 (machine precision)

✓ Inverted (θ=180°):
  - A matrix error: 9.83e-09 (sub-micron accuracy)
  - B matrix error: 8.88e-16 (machine precision)

Conclusion: Numerical linearization matches automatic differentiation
```

## Use Cases

1. **Testing Linearization Methods**: Validate numerical vs analytical linearization
2. **OFC Controller Development**: Build optimal feedback controllers for coupled systems
3. **Coupled Dynamics Analysis**: Study inertial coupling between cart and pendulum
4. **Educational Examples**: Demonstrate multi-body dynamics in Drake

## Files

- **Class Definition**: [script_cup_manipulator_controller_ofc.py](../script_cup_manipulator_controller_ofc.py) (lines ~1720-1930)
- **Test Script**: [test_pendulum_linearization.py](../test_pendulum_linearization.py)
- **Configuration Types**: [robot_types.py](../robot_types.py)

## Related Classes

- `Pendulum3D`: Base class for gimbal-mounted 3D pendulum
- `MuscleDynamics`: Muscle activation dynamics for biological control
- `ImpedanceForce`: Impedance control force computation
- `ZFTReferenceMass`: Zero-force trajectory reference mass

## Future Work

- [ ] Add damping configuration for cart joints
- [ ] Support 3D cart motion (x, y, z)
- [ ] Add visualization helper methods
- [ ] Support multiple pendulums attached to single cart
- [ ] Add LQR controller synthesis method
- [ ] Integrate with full cup manipulator system (14D state)

---
**Last Updated**: January 2025  
**Author**: Extracted from test code during linearization validation
