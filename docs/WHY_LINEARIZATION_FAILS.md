# Why Linearization Fails (And What Actually Works)

## TL;DR

**The system IS linearizable** - Drake can do it automatically. The problems were:

1. **Manual linearization was incomplete** (missing Coriolis Jacobians)
2. **Single-point linearization doesn't work** for trajectory tracking
3. **Need time-varying linearization** along the trajectory (what OFC does)

---

## The Full Nonlinear Dynamics

```
M(q)q̈ + C(q,v)v + g(q) = τ
```

Where:
- **M(q)** = Mass matrix (4×4) - configuration dependent
- **C(q,v)v** = Coriolis/centrifugal forces - depends on BOTH position AND velocity
- **g(q)** = Gravitational forces - configuration dependent
- **q** = [L1, L2, pitch, roll] (4 DOFs)
- **τ** = [τ_L1, τ_L2, 0, 0] (only 2 actuated)

---

## What Linearization Requires

To linearize, we need the state-space form:

```
ẋ = f(x, u)
```

Where x = [q, v] (8-dimensional state) and u = τ (2-dimensional input).

The Jacobians are:

```
A = ∂f/∂x = [      0           I      ]  (8×8 matrix)
             [ ∂q̈/∂q      ∂q̈/∂v  ]

B = ∂f/∂u = [    0     ]  (8×2 matrix)
             [ ∂q̈/∂τ  ]
```

---

## The Manual Linearization That FAILED

### What Was Attempted

Looking at the failed LQR attempts, they likely computed:

```python
# Simplified (WRONG) approach
M_inv = np.linalg.inv(M(q₀))
A = [[0, I],
     [-M_inv @ ∂g/∂q, 0]]  # ← Missing Coriolis terms!
     
B = [[0],
     [M_inv @ selection_matrix]]
```

### What Was MISSING

The complete linearization requires:

```
∂q̈/∂q = M⁻¹[∂τ/∂q - ∂C/∂q·v - ∂M/∂q·q̈ - ∂g/∂q]
∂q̈/∂v = M⁻¹[∂τ/∂v - ∂C/∂v·v - C]
```

The missing terms:
1. **∂C/∂q** - How Coriolis changes with configuration (3D tensor!)
2. **∂C/∂v** - How Coriolis changes with velocity (matrix)
3. **∂M/∂q** - How inertia changes with configuration (3D tensor!)

These are CRITICAL because:
- Coriolis coupling (C) is what makes L2→pitch work
- Without ∂C/∂q and ∂C/∂v, the A matrix is completely wrong
- The linearization doesn't capture the actual dynamics!

---

## Why Even Correct Linearization Isn't Enough

Even if we computed the full Jacobians correctly, we'd still have problems:

### Problem 1: Configuration-Dependent Coupling

Look at how the mass matrix changes:

```
Configuration          M[2,0]   M[2,1]   M[3,0]   M[3,1]
                     (Pitch-L1)(Pitch-L2)(Roll-L1)(Roll-L2)
─────────────────────────────────────────────────────────
P=0°,   R=0°         -0.061    0.000    -0.198   -0.125
P=0°,   R=90°         0.000    0.000     0.000    0.000  ← SINGULAR!
P=0°,   R=180°        0.061    0.000     0.198    0.125
P=30°,  R=150°        0.043   -0.008     0.174    0.098
```

**The coupling varies by 100× depending on configuration!**

A single linearization at q₀ gives you A₀ and B₀, but:
- At R=0°: System has moderate coupling
- At R=90°: System is SINGULAR (zero coupling)
- At R=180°: System has opposite-sign coupling

### Problem 2: Trajectory Tracking vs Equilibrium Regulation

**LQR was designed for equilibrium regulation:**
```
Stabilize around x* = 0 (or constant x*)
Control: u = -K(x - x*)
Works because: x ≈ x* always (small deviations)
```

**Our problem is trajectory tracking:**
```
Follow x_d(t) that varies significantly over time
Control: u = -K(x - x_d(t)) ← WRONG for nonlinear systems!
Problem: x_d(t) moves through different regions with different A, B
```

The trajectory passes through:
- q = [0°, 0°, 0°, 180°] at t=0 (coupling exists)
- q = [10°, -20°, 5°, 170°] at t=1 (different coupling)  
- q = [20°, -40°, 10°, 160°] at t=2 (different coupling)
- ...

**Each point needs different A, B matrices!**

### Problem 3: Underactuated System

We have:
- 4 DOFs to control [L1, L2, pitch, roll]
- Only 2 actuators [τ_L1, τ_L2]
- Pitch and roll are PASSIVE (controlled only through coupling)

For this to work:
- Coupling must be strong everywhere along trajectory
- System must be "controllable" at every point
- But at R≈90°, coupling→0, system becomes uncontrollable

LQR assumes:
```
Controllability: rank([B, AB, A²B, ...]) = n
```

But our system loses controllability at certain configurations!

---

## What DOES Work

### 1. Trajectory Optimization (What the Paper Uses)

Instead of:
```
Pick q_d(t) arbitrarily → Try to track it with LQR → FAILS
```

Do:
```
Solve for feasible trajectory that respects dynamics:

minimize ∫[cost] dt
subject to: M(q)q̈ + C(q,v)v + g(q) = τ
           |τ| ≤ τ_max
           Avoid singular configurations
           
→ Finds trajectory the system can ACTUALLY follow
```

Drake's trajectory optimization:
```python
prog = MathematicalProgram()
# Add dynamics constraints (full nonlinear M, C, g)
# Add cost function
# Add state/input limits
# Solve using direct collocation or direct transcription
```

Benefits:
- Considers full nonlinear dynamics
- Finds trajectories that exploit available coupling
- Avoids regions with weak/zero coupling
- Returns both trajectory AND feedforward torques

### 2. Time-Varying LQR (What OFC Does)

After getting optimal trajectory [q*(t), v*(t), τ*(t)] from optimization:

```python
for each timestep t:
    # Linearize around the trajectory point (not a fixed point!)
    A(t) = ∂f/∂x | at x*(t)
    B(t) = ∂f/∂u | at x*(t)
    
    # Solve time-varying Riccati equation
    K(t) = LQR(A(t), B(t), Q, R)
    
    # Time-varying feedback control
    u(t) = τ*(t) - K(t)·[x(t) - x*(t)]
          ↑              ↑
    feedforward    feedback gain
    from traj opt   (time-varying!)
```

This works because:
- A(t), B(t) valid locally around x*(t)
- Feedforward τ*(t) does most of the work
- Feedback only corrects small deviations
- Linearization updated at each timestep

### 3. Drake's Automatic Linearization

Drake can compute the FULL Jacobians correctly:

```python
# Let Drake do it - it handles all the complex derivatives
context_linearization = plant.CreateDefaultContext()
plant.SetPositions(context_linearization, q₀)
plant.SetVelocities(context_linearization, v₀)

# This computes the COMPLETE A and B matrices
linearized_system = Linearize(
    plant, 
    context_linearization,
    input_port_index=plant.get_actuation_input_port().get_index(),
    output_port_index=plant.get_state_output_port().get_index()
)

A = linearized_system.A()  # Includes all ∂M/∂q, ∂C/∂q, ∂C/∂v terms!
B = linearized_system.B()
```

Drake's `Linearize()` computes all the nasty derivatives automatically!

---

## Why Cart-Pole Works But This Doesn't

### Cart-Pole (Classic LQR Success Story)

```
States: x = [cart_pos, cart_vel, pole_angle, pole_vel]
Input:  u = [cart_force]
Goal:   Stabilize pole at θ=0 (upright - UNSTABLE equilibrium)
```

Why LQR works:
1. **Equilibrium regulation** - not trajectory tracking
2. **Unstable equilibrium** - system naturally wants to fall, LQR stabilizes it
3. **Strong coupling everywhere** near θ=0
4. **Fully actuated in the space** - cart controls everything through coupling
5. **Small operating region** - linearization valid (θ ≈ 0)

### Cup-Manipulator-Pendulum (Why LQR Fails)

```
States: x = [L1, L2, pitch, roll, v_L1, v_L2, v_pitch, v_roll]
Input:  u = [τ_L1, τ_L2]
Goal:   Track trajectory L1: 0→20°, L2: 0→-40° over 3 seconds
```

Why LQR fails:
1. **Trajectory tracking** - not equilibrium regulation
2. **Stable equilibrium** - pendulum wants to hang down (R=180°)
3. **Configuration-dependent coupling** - varies 100× along trajectory
4. **Singular regions exist** - R=90° has zero coupling
5. **Large motion** - trajectory spans huge state space region
6. **Underactuated** - 2 inputs, 4 states, coupling is the only way

---

## The Correct Approach

### Step 1: Trajectory Optimization
Find a dynamically feasible trajectory:
```python
result = SolveDircolTrajectory(
    plant=plant,
    q_initial=[0, 0, 0, 180°],
    q_final=[20°, -40°, 0, 180°],
    duration=3.0,
    timesteps=50
)

q_traj = result.q_trajectory
v_traj = result.v_trajectory  
τ_traj = result.tau_trajectory  # Feedforward torques
```

### Step 2: Time-Varying LQR (OFC)
Stabilize around the trajectory:
```python
tvlqr = FiniteHorizonLinearQuadraticRegulator(
    plant,
    context,
    t0=0.0,
    tf=3.0,
    Q=np.diag([100, 100, 10, 10, 1, 1, 1, 1]),  # State cost
    R=np.diag([1, 1]),  # Effort cost
    x_trajectory=state_traj,
    u_trajectory=τ_traj
)

# Controller = feedforward + time-varying feedback
u(t) = τ_traj(t) - K(t)·[x(t) - x_traj(t)]
```

This is what works and what the paper uses!

---

## Summary Table

| Approach | Why It Fails / Works |
|----------|---------------------|
| **Manual LQR** | ✗ Incomplete linearization (missing ∂C/∂q, ∂C/∂v)<br>✗ Single-point linearization<br>✗ No feedforward<br>✗ Doesn't handle configuration-dependent coupling |
| **Correct LQR at one point** | ✗ Still single-point linearization<br>✗ Trajectory moves far from linearization point<br>✗ System uncontrollable at some configurations<br>✗ No feedforward torques |
| **Trajectory Optimization** | ✓ Considers full nonlinear dynamics<br>✓ Finds feasible paths<br>✓ Avoids singular regions<br>✓ Provides feedforward torques |
| **Time-Varying LQR (OFC)** | ✓ Linearization valid locally at each point<br>✓ Feedforward does heavy lifting<br>✓ Feedback only corrects small errors<br>✓ Handles changing coupling along trajectory |
| **Drake's Auto-Linearize** | ✓ Computes complete Jacobians correctly<br>✓ Can be used for TVLQR<br>⚠️ Still need trajectory optimization first |

---

## The Bottom Line

**You CAN linearize this system** - Drake does it perfectly with `Linearize()`.

**You CAN'T use simple constant-gain LQR** because:
1. Manual linearization missed critical Coriolis terms
2. Coupling changes dramatically along the trajectory  
3. System loses controllability at certain configurations
4. Trajectory tracking ≠ equilibrium regulation

**What works:**
1. **Trajectory Optimization** to find feasible path
2. **Time-Varying LQR (TVLQR)** to stabilize around that path
3. **Drake's automatic tools** to compute everything correctly

This is exactly what OFC (Optimal Feedback Control) does, and why it works when simple LQR doesn't!
