# Virtual Mass Implementation Notes

## Overview
Modified `test_impedance_interaction_cart_pendulum.py` to add a virtual mass (admittance dynamics) between the end effector and cart.

## New Architecture

```
┌──────────────┐
│  Manipulator │ (follows joint trajectory)
└──────┬───────┘
       │
       ▼ (CalcPosition)
┌──────────────────┐
│  End Effector    │ 
│  Position & Vel  │
└──────┬───────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Coupling Force Computer                │
│  F_coupling = -K_c(x_ee - x_cart) - D_c(ẋ_ee - ẋ_cart) │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Virtual Mass (Admittance Dynamics)     │
│  M_v ẍ_des + D_v ẋ_des + K_v(x_des - x₀) = F_coupling │
│                                                  │
│  Outputs: x_des, ẋ_des (desired cart motion)   │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Cart Impedance Controller              │
│  F_cart = K_p(x_des - x_cart) + K_d(ẋ_des - ẋ_cart)  │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│     Cart     │ (passive - driven by forces)
└──────────────┘
```

## New Classes

### 1. CouplingForceComputer
- Computes the "spring-damper" force between EE and cart
- Input: EE position/velocity, Cart position/velocity  
- Output: Coupling force (2D)
- Parameters:
  - `k_coupling`: Coupling stiffness [N/m] (default: 50.0)
  - `d_coupling`: Coupling damping [N·s/m] (default: 10.0)

### 2. VirtualMassAdmittance  
- Implements admittance dynamics: M ẍ + D ẋ + K(x - x₀) = F
- State: [x_des, y_des, vx_des, vy_des] (4D continuous)
- Input: Coupling force
- Output: Desired cart position and velocity
- Parameters:
  - `M_virtual`: Virtual mass [kg] (default: 2.0)
  - `D_virtual`: Virtual damping [N·s/m] (default: 5.0)
  - `K_virtual`: Virtual stiffness [N/m] (default: 10.0)
  - `x0`: Equilibrium position [m]

### 3. CartImpedanceController
- PD controller to make cart follow virtual mass trajectory
- Input: Desired position/velocity, Actual cart position/velocity
- Output: Control force on cart
- Parameters:
  - `kp`: Position gain [N/m] (default: 100.0)
  - `kd`: Velocity gain [N·s/m] (default: 20.0)

## Updated Function Signature

```python
def simulate_passive_cart(
    duration=10.0,
    k_coupling=50.0,
    d_coupling=10.0,
    M_virtual=2.0,
    D_virtual=5.0,
    K_virtual=10.0,
    kp_cart=100.0,
    kd_cart=20.0,
):
```

## Command Line Usage

```bash
# Run with default parameters
python test_impedance_interaction_cart_pendulum.py

# Customize virtual mass parameters
python test_impedance_interaction_cart_pendulum.py \
  --mass 5.0 \
  --damping 10.0 \
  --stiffness 20.0 \
  --k-coupling 100.0 \
  --d-coupling 20.0 \
  --kp-cart 200.0 \
  --kd-cart 40.0

# Run kinematics test only
python test_impedance_interaction_cart_pendulum.py --mode kinematics
```

## Return Data

The `simulate_passive_cart()` function now returns:

```python
{
    'time': time_data,
    'q1': q1,                          # Manipulator joint 1 [rad]
    'q2': q2,                          # Manipulator joint 2 [rad]
    'cart_x': cart_x,                  # Cart X position [m]
    'cart_y': cart_y,                  # Cart Y position [m]
    'pitch': pitch,                    # Pendulum pitch [rad]
    'roll': roll,                      # Pendulum roll [rad]
    'ee_x': ee_x,                      # End effector X [m]
    'ee_y': ee_y,                      # End effector Y [m]
    'virtual_x': virtual_x,            # Virtual mass X [m] (NEW)
    'virtual_y': virtual_y,            # Virtual mass Y [m] (NEW)
    'cart_force_x': cart_force_x,      # Cart control force X [N] (NEW)
    'cart_force_y': cart_force_y,      # Cart control force Y [N] (NEW)
    'coupling_force_x': coupling_force_x,  # Coupling force X [N] (NEW)
    'coupling_force_y': coupling_force_y,  # Coupling force Y [N] (NEW)
}
```

## Physical Interpretation

### Heavy Virtual Mass (M_v ↑)
- Slower response to coupling force
- More inertia in the system
- Smoother motion, less "twitchy"
- Good for: Filtering high-frequency disturbances

### Light Virtual Mass (M_v ↓)
- Fast response to coupling force  
- Less inertia
- More responsive, tracks EE closely
- Good for: Quick reactions

### High Virtual Damping (D_v ↑)
- Less oscillation
- Smoother convergence
- May feel "sluggish"
- Good for: Preventing overshoot

### Low Virtual Damping (D_v ↓)
- More oscillatory
- Faster but less damped
- May overshoot
- Good for: Fast response (if M_v also chosen well)

### Virtual Stiffness (K_v)
- K_v = 0: No restoring force (integration behavior)
- K_v > 0: Returns to equilibrium x₀ when coupling force removed
- Higher K_v → stronger "spring" back to equilibrium

## Damping Ratio

For second-order system M ẍ + D ẋ + K x = F:

- Natural frequency: ωₙ = √(K_v/M_v)
- Damping ratio: ζ = D_v/(2√(M_v K_v))

Critical damping: ζ = 1 → D_v = 2√(M_v K_v)

Example with defaults (M_v=2, K_v=10):
- ωₙ = √(10/2) = 2.24 rad/s
- D_v_critical = 2√(2×10) = 8.94 N·s/m
- Actual D_v = 5.0 → ζ = 5/(2√20) = 0.56 (underdamped)

## TODO: Plot Function

The `plot_results()` function needs minor updates to use new data fields:
- Replace `log_data['force_x']` with `log_data['cart_force_x']` and `log_data['coupling_force_x']`
- Add plots for `log_data['virtual_x']` and `log_data['virtual_y']`
- Show separation distances: EE←Virtual, Virtual←Cart, EE←Cart

See the implementation in the code (lines ~997-1120).
