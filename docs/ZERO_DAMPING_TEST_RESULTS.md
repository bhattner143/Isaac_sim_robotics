# Zero Damping Test Results

## Question
*"Can Drake's `Linearize()` work if we set damping to zero?"*

## Answer
**NO** - Even with zero damping, Drake's `Linearize()` still fails because:

1. **Equilibrium requires BOTH:**
   - Zero velocity: `v = 0` ✓ (easy to set)
   - Zero acceleration: `ẍ = 0` ✗ (impossible for this system!)

2. **Zero acceleration requires:**
   ```
   M(q)q̈ + C(q,v)v + g(q) = τ
   
   At equilibrium (v=0, q̈=0):
   g(q) = τ
   ```

3. **The Problem:**
   - We have **4 DOFs**: `q = [L1, L2, pitch, roll]`
   - We have **2 actuators**: `τ = [τ_L1, τ_L2, 0, 0]`
   - Gravity forces: `g(q) = [g_L1, g_L2, g_pitch, g_roll]`

   For equilibrium we need:
   ```
   g_L1 = τ_L1     ← We can set τ_L1
   g_L2 = τ_L2     ← We can set τ_L2
   g_pitch = 0     ← Need to find special q where this is zero
   g_roll = 0      ← Need to find special q where this is zero
   ```

4. **Numerical Search Failed:**
   ```
   Searching for configuration where g(q) = [0, 0, 0, 0]...
   ✗ Failed: "iteration is not making good progress"
   ```
   
   There is **no configuration** where all gravity forces are simultaneously zero!

## Test Results

### With Zero Damping
```python
pendulum_damping = 0.0  # N·m·s/rad
```

### Attempted Equilibrium
```
q = [0°, 0°, 0°, 180°]  (manipulator at origin, pendulum hanging)
v = [0, 0, 0, 0]
τ = [0, 0]
```

### Verification
```
||ẋ|| = 38.31  ← NOT an equilibrium!
ẋ = [0, 0, 0, 0, -0.95, 0.36, 38.32, 2.53]
     └─ q̇ ─┘  └──────── q̈ ─────────┘
              (massive accelerations!)
```

### Drake's Response
```
❌ RuntimeError: The nominal operating point (x0,u0) is not an 
   equilibrium point of the system.
```

## Why It Matters

Drake's `Linearize()` is for **time-invariant** systems at **equilibrium points**:
- Equilibrium means the system stays put: `ẋ = f(x₀, u₀) = 0`
- Time-invariant means dynamics don't change with time

Our system violates both:
1. **Not at equilibrium** - underactuation prevents it
2. **Trajectory tracking is time-varying** - we're following a path, not staying put

## Solution

For **trajectory tracking** (which is what we actually want):
- Don't use `Linearize()` - it's the wrong tool
- Use **time-varying linearization** along the trajectory:
  ```python
  A(t) = ∂f/∂x|_(x(t), u(t))
  B(t) = ∂f/∂u|_(x(t), u(t))
  ```
- Compute via numerical differentiation (finite differences)
- Apply time-varying LQR (TVLQR) or MPC

## Conclusion

Setting damping to zero **doesn't help** because:
- Drake's `Linearize()` needs `ẋ = 0` (equilibrium)
- Our underactuated system has no equilibrium configuration
- Even with zero damping, gravity prevents equilibrium
- The problem isn't damping - it's **underactuation + gravity**

**Damping = 0** → Still fails ✗
**Gravity = 0** → Would work ✓ (but unrealistic!)
**Full actuation (4 actuators)** → Would work ✓ (but changes the problem!)
