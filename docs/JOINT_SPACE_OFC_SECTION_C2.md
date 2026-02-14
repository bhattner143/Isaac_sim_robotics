# Joint-Space Optimal Feedback Control (Section C.2)

## Overview

This document explains the implementation of **Section C.2** from Razavian et al. (2021) "Dynamic Primitives and Optimal Feedback Control for the Manipulation of Complex Objects" in **joint space** of the manipulator.

## Simplifications Applied

As requested, we assume:
- **d = 0**: No external disturbances
- **ε = 0**: No error dynamics  
- **ω = 0**: No oscillator dynamics

This gives us the **core impedance-based optimal feedback control** formulation.

---

## Mathematical Formulation

### 1. Impedance Dynamics (Equations C.13-C.15)

The impedance filter creates a virtual spring-damper system between the actual joint position $y$ and a zero-force trajectory $y_{zf}$:

$$
Ma \cdot \ddot{y} + kp \cdot (y - y_{zf}) + kd \cdot (\dot{y} - \dot{y}_{zf}) = 0
$$

Where:
- $y$: Actual joint position (from robot)
- $y_{zf}$: Zero-force trajectory (optimized variable)
- $Ma$: Virtual/impedance mass [kg]
- $kp$: Spring stiffness [N/m]
- $kd$: Damping coefficient [N·s/m]

### 2. Control Law

Solving the impedance equation for the required torque:

$$
\tau = kp \cdot (y_{zf} - y) + kd \cdot (\dot{y}_{zf} - \dot{y}) + F
$$

Where:
- $\tau$: Joint torque (control output)
- $F$: Driving force (optimized in effort mode, 0 in smoothness mode)

### 3. Robot Dynamics

The manipulator follows standard rigid body dynamics:

$$
M(q) \cdot \ddot{q} + C(q, \dot{q}) \cdot \dot{q} + G(q) = \tau
$$

Combining with the control law, we get:

$$
M(q) \cdot \ddot{q} = kp \cdot (y_{zf} - q) + kd \cdot (\dot{y}_{zf} - \dot{q}) + F - C(q, \dot{q}) \cdot \dot{q} - G(q)
$$

---

## State-Space Formulation

### Effort-Minimizing Mode

**State vector** (per joint):
$$
x = [q, \dot{q}, F, y_{zf}, \dot{y}_{zf}]
$$

**Control input**:
$$
u = \dot{F}
$$
(rate of change of driving force)

**Dynamics**:
$$
\begin{bmatrix}
\dot{q} \\
\ddot{q} \\
\dot{F} \\
\dot{y}_{zf} \\
\ddot{y}_{zf}
\end{bmatrix}
=
\begin{bmatrix}
\dot{q} \\
-\frac{kp}{M} q - \frac{kd}{M} \dot{q} + \frac{kp}{M} y_{zf} + \frac{kd}{M} \dot{y}_{zf} + \frac{1}{M} F \\
u \\
\dot{y}_{zf} \\
0
\end{bmatrix}
$$

### Smoothness-Minimizing Mode

**State vector** (per joint):
$$
x = [q, \dot{q}, \ddot{y}_{zf}, y_{zf}, \dot{y}_{zf}]
$$

**Control input**:
$$
u = \dddot{y}_{zf}
$$
(jerk of zero-force trajectory)

**Dynamics**:
$$
\begin{bmatrix}
\dot{q} \\
\ddot{q} \\
\dddot{y}_{zf} \\
\dot{y}_{zf} \\
\ddot{y}_{zf}
\end{bmatrix}
=
\begin{bmatrix}
\dot{q} \\
-\frac{kp}{M} q - \frac{kd}{M} \dot{q} + \frac{kp}{M} y_{zf} + \frac{kd}{M} \dot{y}_{zf} \\
u \\
\dot{y}_{zf} \\
\ddot{y}_{zf}
\end{bmatrix}
$$

---

## LQR Optimization

### Cost Function

We minimize the infinite-horizon LQR cost:

$$
J = \int_0^\infty \left( x^T Q x + u^T R u \right) dt
$$

Where:
- $Q$: State cost matrix (penalizes deviations from equilibrium)
- $R$: Control cost matrix (penalizes effort or jerk)

### State Cost Matrix Q

For 2-joint manipulator with augmented states:

$$
Q = \text{diag}([Q_{q1}, Q_{q2}, Q_{\dot{q}1}, Q_{\dot{q}2}, Q_{aux1}, Q_{y_{zf}1}, Q_{\dot{y}_{zf}1}, Q_{aux2}, Q_{y_{zf}2}, Q_{\dot{y}_{zf}2}])
$$

Where:
- $Q_{q}$: Position tracking weight (e.g., 100)
- $Q_{\dot{q}}$: Velocity tracking weight (e.g., 10)
- $Q_{aux}$: Auxiliary state weight (small, e.g., 0.1)
- $Q_{y_{zf}}$: Zero-force trajectory position weight (e.g., 100)
- $Q_{\dot{y}_{zf}}$: Zero-force trajectory velocity weight (e.g., 10)

### Control Cost Matrix R

$$
R = \text{diag}([R_1, R_2])
$$

Where:
- **Effort mode**: $R_i$ = 0.1 (small, allows larger forces)
- **Smoothness mode**: $R_i$ = 0.1 (penalizes jerk)

### LQR Solution

Solve the continuous-time algebraic Riccati equation (CARE):

$$
A^T P + P A - P B R^{-1} B^T P + Q = 0
$$

The optimal feedback gain is:

$$
K = R^{-1} B^T P
$$

And the control law becomes:

$$
u_{opt} = -K (x - x_{desired})
$$

---

## Linearization

The dynamics are linearized around the goal equilibrium point:

**Equilibrium state**:
$$
x_{eq} = [q_{goal}, 0, 0, q_{goal}, 0]
$$

**Linearized system**:
$$
\dot{x} = A x + B u
$$

Where $A$ and $B$ are computed by taking partial derivatives of the nonlinear dynamics at $x_{eq}$.

### A Matrix Structure (Effort Mode)

For a single joint (dimension 5×5):

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 \\
-\frac{kp}{M} & -\frac{kd}{M} & \frac{1}{M} & \frac{kp}{M} & \frac{kd}{M} \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

### B Matrix Structure (Effort Mode)

$$
B = \begin{bmatrix}
0 \\
0 \\
1 \\
0 \\
0
\end{bmatrix}
$$

---

## Implementation Details

### Per-Joint Augmented State

For **2 joints**, the full augmented state has **10 dimensions**:

**Effort mode**:
```
x = [q1, q2, q̇1, q̇2, F1, y_zf1, ẏ_zf1, F2, y_zf2, ẏ_zf2]
```

**Smoothness mode**:
```
x = [q1, q2, q̇1, q̇2, ÿ_zf1, y_zf1, ẏ_zf1, ÿ_zf2, y_zf2, ẏ_zf2]
```

### Control Computation Steps

1. **Measure state**: Get $q, \dot{q}$ from robot
2. **Augment state**: Combine with internal states $(F, y_{zf}, \dot{y}_{zf})$ or $(\ddot{y}_{zf}, y_{zf}, \dot{y}_{zf})$
3. **Compute error**: $e = x - x_{desired}$
4. **LQR feedback**: $u_{opt} = -K \cdot e$
5. **Integrate control**: Update $F$ or $\ddot{y}_{zf}$ using $u_{opt}$
6. **Integrate ZFT**: Update $y_{zf}, \dot{y}_{zf}$ forward in time
7. **Apply torque**: $\tau = kp(y_{zf} - q) + kd(\dot{y}_{zf} - \dot{q}) + F$

---

## Parameter Tuning Guide

### Impedance Parameters

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| Virtual mass | $Ma$ | 0.5 - 5.0 kg | Higher = smoother, slower response |
| Stiffness | $kp$ | 50 - 200 N/m | Higher = tighter tracking of $y_{zf}$ |
| Damping | $kd$ | 10 - 50 N·s/m | Higher = more damping, less oscillation |

**Recommended starting point**: $Ma = 1.0$, $kp = 100$, $kd = 20$

### Cost Weights

| Weight | Symbol | Effort Mode | Smoothness Mode |
|--------|--------|-------------|-----------------|
| Position | $Q_q$ | 100 | 100 |
| Velocity | $Q_{\dot{q}}$ | 10 | 10 |
| Control | $R$ | 0.1 (force) | 0.1 (jerk) |

**Trade-offs**:
- Higher $Q$ → Better tracking, more aggressive control
- Higher $R$ → Smoother control, worse tracking
- Typical ratio: $Q/R \approx 1000$

---

## Comparison: Effort vs. Smoothness Modes

| Aspect | Effort Mode | Smoothness Mode |
|--------|-------------|-----------------|
| **Control variable** | $\dot{F}$ (force rate) | $\dddot{y}_{zf}$ (jerk) |
| **Optimizes** | Minimize torque | Minimize jerk |
| **Auxiliary state** | $F$ (driving force) | $\ddot{y}_{zf}$ (acceleration) |
| **Best for** | Energy efficiency | Smooth motion |
| **Typical R** | 0.1 | 0.1 |

---

## Code Usage Example

```python
from joint_space_ofc_implementation import JointSpaceOFC

# Create OFC controller
ofc = JointSpaceOFC(
    plant=plant,
    q_start=np.deg2rad([80, -160]),
    q_goal=np.deg2rad([20, -40]),
    duration=3.0,
    mode='effort',  # or 'smoothness'
    Q_position=np.array([100.0, 100.0]),
    Q_velocity=np.array([10.0, 10.0]),
    R=np.array([0.1, 0.1]),
    Ma=1.0,   # Virtual mass
    kp=100.0,  # Spring stiffness
    kd=20.0    # Damping
)

# Connect to Drake diagram
builder.AddSystem(ofc)
builder.Connect(plant.get_state_output_port(),
                ofc.get_input_port(0))
builder.Connect(ofc.get_output_port(0),
                plant.get_actuation_input_port())
```

---

## Key Differences from Original Implementation

### Original (in `script_cup_manipulator_controller_drake.py`):
- Solves LQR for simplified 4-DOF system (manipulator only)
- Impedance applied as post-processing filter
- Separate trajectory generator

### New Joint-Space OFC:
- Solves LQR for full augmented system (10-DOF with ZFT states)
- Impedance integrated into dynamics model
- Zero-force trajectory is part of optimized state
- **Directly implements Section C.2 equations**

---

## Benefits of This Approach

1. **Theoretically grounded**: Directly implements paper equations
2. **Unified optimization**: $y_{zf}$ trajectory is optimized, not prescribed
3. **Better performance**: LQR sees full system including impedance
4. **Clearer separation**: Robot dynamics vs. impedance dynamics
5. **Easier tuning**: Impedance parameters have direct physical meaning

---

## References

1. Razavian, R. S., et al. (2021). "Dynamic Primitives and Optimal Feedback Control for the Manipulation of Complex Objects." PhD Thesis.
   - **Section C.2**: Optimal Feedback Control
   - **Equations C.13-C.15**: Impedance dynamics
   - **Equations C.16-C.17**: Zero-force trajectory

2. Drake Documentation: LinearQuadraticRegulator
   - https://drake.mit.edu/pydrake/pydrake.all.html

---

## Validation Checklist

- [x] Impedance dynamics match eq. C.13-C.15
- [x] Control law includes spring-damper + driving force
- [x] State-space formulation correct for both modes
- [x] LQR cost function properly structured
- [x] Linearization around equilibrium
- [x] Per-joint implementation (2 joints)
- [x] Integration of ZFT states
- [x] Simplifications applied (d=ε=ω=0)

---

## Next Steps

1. **Test with simulation**: Integrate into `script_cup_manipulator_controller_drake.py`
2. **Compare modes**: Run effort vs. smoothness mode
3. **Tune parameters**: Adjust $Ma, kp, kd, Q, R$ for best performance
4. **Visualize ZFT**: Plot $y_{zf}$ trajectory vs. actual $q$
5. **Benchmark**: Compare against existing OFC implementation
