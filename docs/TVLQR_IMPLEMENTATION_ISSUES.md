# TVLQR Implementation Issues and Corrections

**Date:** February 9, 2026  
**Status:** Educational documentation of implementation issues  
**Recommendation:** Use `ComputedTorqueController` for production (achieves 3.7° RMS error)

---

## Executive Summary

The current TVLQR implementation in [`script_cup_manipulator_controller_drake.py`](../script_cup_manipulator_controller_drake.py) has **four major correctness issues** that prevent it from functioning as proper Time-Varying LQR. This document explains each issue and provides guidance for correct implementation.

### Performance Comparison

| Controller | Position RMS Error (L1, L2) | Status |
|------------|---------------------------|--------|
| **Computed Torque** | 3.7°, 1.5° | ✅ Excellent - **USE THIS** |
| **TVLQR (current)** | 188°, 103° | ❌ Poor - has correctness issues |

---

## Issue #1: Reference Trajectory Dimension Mismatch

### Problem

**Location:** `TVLQRController._precompute_tvlqr_gains()` (lines ~1200-1210)

```python
q_d, qd_d, qdd_d = self.trajectory_generator.compute_trajectory(t)
self.x_ref_samples[i, :] = np.concatenate([q_d, qd_d])
```

**What's wrong:**
- `MIN_JERK_Q_START` and `MIN_JERK_Q_GOAL` are 4D arrays: `[L1, L2, pitch, roll]`
- `MinJerkTrajectoryGenerator` interpolates element-wise, producing 4D trajectories
- **BUT**: For the passive DOFs (pitch, roll), the "reference" is just a straight-line interpolation between start and goal
- **This is NOT the actual motion** the passive joints will exhibit during the trajectory!

For underactuated systems, **you cannot precompute passive DOF trajectories analytically** - they depend nonlinearly on the actuated motion and must be either:
1. Simulated forward with a working controller
2. Optimized via direct collocation

### Impact
The reference state `x_ref(t)` used for linearization is dynamically infeasible, causing the TVLQR gains to be computed around a trajectory the system will never actually follow.

---

## Issue #2: Feedforward Not Dynamically Feasible

### Problem

**Location:** `TVLQRController._precompute_tvlqr_gains()` (lines ~1213-1230)

```python
# Commanded accelerations (full system)
qdd_commanded = np.zeros(self.num_velocities)
qdd_commanded[:self.num_actuated] = qdd_d[:self.num_actuated]  # Only actuated joints

# Inverse dynamics: τ = M·q̈ + C·v + g
tau_full = self.model.CalcInverseDynamics(self.model_context, qdd_commanded, external_forces)
self.u_ref_samples[i, :] = tau_full[:self.num_actuated]
```

**What's wrong:**
- Sets passive joint accelerations to **zero**: `qdd_commanded[2:4] = 0`
- Then computes control inputs via inverse dynamics
- The pair `(x_ref(t), u_ref(t))` is **NOT a solution to the actual dynamics**!

### Why This Matters
For TVLQR to work, the nominal trajectory must satisfy:

```
ẋ₀(t) = f(x₀(t), u₀(t))
```

The current approach violates this by forcing passive accelerations to zero, which the underactuated system cannot actually achieve.

### Correct Approach
Generate nominal trajectory by:
1. Simulating the system with a working controller (e.g., Computed Torque)
2. Recording the actual state and control trajectories
3. Using those as `x₀(t)` and `u₀(t)`

---

## Issue #3: Incorrect Riccati Equation Integration

### Problem

**Location:** `TVLQRController._precompute_tvlqr_gains()` (lines ~1280-1295)

```python
# Compute optimal gain
K = np.linalg.solve(self.R, B.T @ S)

# Update S using "Riccati equation"
A_closed = A - B @ K
CARE_rhs = A_closed.T @ S + S @ A_closed + self.Q
S = S + dt * CARE_rhs  # Forward Euler integration
```

**What's wrong:**

The continuous-time Riccati differential equation is:

```
-Ṡ(t) = A(t)ᵀS(t) + S(t)A(t) - S(t)B(t)R⁻¹B(t)ᵀS(t) + Q
```

The implementation computes:
```
A_closed = A - B·K  (where K = R⁻¹BᵀS)
CARE_rhs = A_closed^T·S + S·A_closed + Q
```

This **drops the crucial term** `-S·B·R⁻¹·Bᵀ·S`. While `A_closed` implicitly contains feedback, this is not equivalent to the correct Riccati equation when integrated numerically.

### Impact
The computed gains `K(t)` are not the true optimal TVLQR gains. The actual optimal gains would come from solving the correct differential equation backward in time.

### Correct Approach
Use Drake's built-in TVLQR:

```python
from pydrake.all import MakeFiniteHorizonLinearQuadraticRegulator

tvlqr = MakeFiniteHorizonLinearQuadraticRegulator(
    system=model,
    context=model_context,
    t0=0.0,
    tf=duration,
    Q=Q_matrix,
    R=R_matrix,
    options=options
)
```

This correctly solves the Riccati equation and returns a time-varying linear system that implements `u = u₀(t) - K(t)·[x - x₀(t)]`.

---

## Issue #4: Incomplete Dynamics Linearization

### Problem

**Location:** `TVLQRController._linearize_at_point()` (lines ~1360-1380)

```python
def compute_xdot(q, v, u):
    M = self.model.CalcMassMatrix(self.model_context)
    Cv = self.model.CalcBiasTerm(self.model_context)  # Coriolis + gravity
    B = self.model.MakeActuationMatrix()
    tau_applied = B @ u
    v_dot = np.linalg.solve(M, tau_applied - Cv)
    xdot = [v, v_dot]
    return xdot
```

**What's wrong:**
- Manually computes dynamics as: `Mv̇ + Cv = Bu`
- Assumes `CalcBiasTerm()` correctly gives `C(q,v)v + g(q)`
- Assumes simple actuation matrix `B`
- **Ignores:**
  - Constraints (holonomic/nonholonomic)
  - Complex joint structures
  - Potential frame transformations

While this works for simple multibody systems, it's less robust than using Drake's built-in time derivative computation.

### Correct Approach
Use Drake's `CalcTimeDerivatives()` or automatic differentiation:

```python
# Option 1: Use Drake's derivatives
derivatives = model.EvalTimeDerivatives(context)
xdot = derivatives.CopyToVector()

# Option 2: Use symbolic/automatic differentiation
# (requires system to be set up with symbolic expressions)
```

---

## Correct TVLQR Implementation Steps

### 1. Generate Feasible Nominal Trajectory

```python
def generate_nominal_trajectory(plant, controller_type, duration):
    """
    Simulate system with a working controller to get feasible trajectory.
    """
    # Build temporary diagram with working controller (e.g., Computed Torque)
    builder = DiagramBuilder()
    temp_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # ... load model ...
    
    controller = ComputedTorqueController(...)
    builder.Connect(temp_plant.get_state_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), temp_plant.get_actuation_input_port())
    
    # Add loggers
    state_logger = LogVectorOutput(temp_plant.get_state_output_port(), builder)
    control_logger = LogVectorOutput(controller.get_output_port(), builder)
    
    # Simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.AdvanceTo(duration)
    
    # Extract trajectories
    times = state_logger.sample_times()
    states = state_logger.data()
    controls = control_logger.data()
    
    # Create PiecewisePolynomial trajectories
    x0_traj = PiecewisePolynomial.FirstOrderHold(times, states)
    u0_traj = PiecewisePolynomial.ZeroOrderHold(times, controls)
    
    return x0_traj, u0_traj
```

### 2. Use Drake's TVLQR Builder

```python
from pydrake.all import (
    MakeFiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions
)

# Create options
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.Qf = Q  # Terminal cost

# Build TVLQR system
tvlqr_system = MakeFiniteHorizonLinearQuadraticRegulator(
    system=model,
    context=model.CreateDefaultContext(),
    t0=0.0,
    tf=duration,
    Q=Q,
    R=R,
    options=options,
    input_port_index=model.get_actuation_input_port().get_index()
)
```

### 3. Wire TVLQR into Diagram

```python
class TVLQRController(LeafSystem):
    def __init__(self, tvlqr_system, x0_traj, u0_traj, duration):
        LeafSystem.__init__(self)
        self.tvlqr = tvlqr_system
        self.x0_traj = x0_traj
        self.u0_traj = u0_traj
        self.duration = duration
        
        # Ports
        self.DeclareVectorInputPort("state", BasicVector(num_states))
        self.DeclareVectorOutputPort("control", BasicVector(num_actuators), 
                                      self.CalcControl)
    
    def CalcControl(self, context, output):
        x = self.get_input_port(0).Eval(context)
        t = min(context.get_time(), self.duration)
        
        # Evaluate TVLQR at current time and state
        tvlqr_context = self.tvlqr.CreateDefaultContext()
        tvlqr_context.SetTime(t)
        self.tvlqr.get_input_port(0).FixValue(tvlqr_context, x)
        
        u = self.tvlqr.get_output_port(0).Eval(tvlqr_context)
        output.SetFromVector(u)
```

---

## Why Computed Torque Works Better

The `ComputedTorqueController` achieves **3.7° and 1.5° RMS tracking error** because:

1. **Dynamically Feasible:** Uses `CalcInverseDynamics()` with properly computed desired accelerations
2. **Simple and Robust:** `u = M(q)·[q̈_d + Kp·e + Kd·ė] + C(q,v) + g(q)`
3. **No Linearization Needed:** Works directly with nonlinear dynamics
4. **Handles Underactuation:** Properly accounts for passive DOFs through full system dynamics

For highly nonlinear underactuated systems like this cup manipulator, computed torque is often superior to TVLQR unless you have:
- Very tight optimality requirements
- Explicit state/control constraints
- Need to minimize a specific cost functional

---

## Recommendations

### For Learning TVLQR

Study Drake examples:
- [`examples/multibody/cart_pole/cart_pole.cc`](https://github.com/RobotLocomotion/drake/blob/master/examples/multibody/cart_pole/cart_pole.cc)
- [`examples/pendulum/`](https://github.com/RobotLocomotion/drake/tree/master/examples/pendulum)
- [Drake TVLQR documentation](https://drake.mit.edu/doxygen_cxx/group__control.html)

### For Production

**Use Computed Torque mode:**

```bash
python script_cup_manipulator_controller_drake.py --mode computed-torque --visualize True
```

**Advantages:**
- ✅ Proven to work: 3.7° tracking error
- ✅ Simpler implementation
- ✅ Robust to model uncertainties
- ✅ No trajectory optimization needed

---

## References

1. **Tedrake, R.** (2024). *Underactuated Robotics*. Chapter 8: LQR and TVLQR. http://underactuated.mit.edu
2. **Drake Documentation:** MakeFiniteHorizonLinearQuadraticRegulator
3. **Anderson, B. D., & Moore, J. B.** (1990). *Optimal Control: Linear Quadratic Methods*. Riccati equation chapter.

---

## File Locations

- Main script: [`script_cup_manipulator_controller_drake.py`](../script_cup_manipulator_controller_drake.py)
- TVLQR class: Lines 1085-1420 (marked with ⚠️ warnings)
- Computed Torque class: Lines 930-1075 (working implementation)
- Mode selection: Line 2047+ (shows performance comparison)

---

**Bottom Line:** The TVLQR implementation demonstrates the *concept* but has correctness issues. For actual trajectory tracking on this system, use `--mode computed-torque` which works excellently.
