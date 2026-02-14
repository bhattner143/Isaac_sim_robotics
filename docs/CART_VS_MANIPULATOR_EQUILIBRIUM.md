# Cart-Pendulum vs Cup Manipulator: Why One Has Equilibrium and the Other Doesn't

## TL;DR

**Cart-Pendulum:** ✅ HAS equilibrium → Drake's `Linearize()` works!
**Cup Manipulator:** ❌ NO equilibrium → Drake's `Linearize()` fails!

## The Systems

### Cart-Pendulum System
```
DOFs: 3 (cart_x, pitch, roll)
Actuators: 1 (force on cart)
Underactuation: 1 actuator for 3 DOFs

Generalized coordinates: q = [x, θ_pitch, θ_roll]
Control: τ = [F_cart, 0, 0]
```

### Cup Manipulator + 3D Pendulum
```
DOFs: 4 (L1, L2, pitch, roll)
Actuators: 2 (torque on L1, L2)
Underactuation: 2 actuators for 4 DOFs

Generalized coordinates: q = [θ_L1, θ_L2, θ_pitch, θ_roll]
Control: τ = [τ_L1, τ_L2, 0, 0]
```

## The Critical Difference: Gravity Coupling

### Cart-Pendulum: DECOUPLED Gravity

At equilibrium with pendulum hanging down:
```python
q = [x, π, π]  # Any cart position, pendulum hanging straight down

Gravity forces:
g = [0,       # Cart: gravity acts vertically, cart moves horizontally → g_cart = 0
     0,       # Pitch: pendulum centered → no pitch torque
     0]       # Roll: pendulum centered → no roll torque

Required control for equilibrium:
τ = [0, 0, 0]  # Zero force needed!
```

**Why cart has zero gravity torque:**
- Gravity acts **vertically** (downward)
- Cart motion is **horizontal** (perpendicular to gravity)
- No coupling! `g_cart = 0` regardless of cart position

**Result:** Equilibrium exists at `(x=anywhere, v=0, θ=π, ω=0, τ=0)`

### Cup Manipulator: COUPLED Gravity

Attempting equilibrium with pendulum hanging:
```python
q = [0, 0, π, π]  # Manipulator at origin, pendulum hanging

Gravity forces:
g = [1.24e-04,    # L1: Arm weight creates torque → g_L1 ≠ 0
     3.87e-05,    # L2: Arm weight creates torque → g_L2 ≠ 0
     -2.35e-10,   # Pitch: nearly zero (pendulum hanging)
     2.60e-06]    # Roll: nearly zero (pendulum hanging)

Required control for equilibrium:
τ = [1.24e-04,    # Need torque to counteract arm weight on L1
     3.87e-05,    # Need torque to counteract arm weight on L2
     (can't control pitch),
     (can't control roll)]
```

**Why manipulator has non-zero gravity torque:**
- Gravity acts on the **arm links themselves**
- Each link has distributed mass
- Torque = r × F_gravity (cross product)
- Creates moments about joint axes
- **Always coupled** to joint angles!

**Problem:** Even with gravity compensation on L1 and L2:
```
Set τ_L1 = g_L1, τ_L2 = g_L2  (gravity compensation)

But now pendulum isn't perfectly at equilibrium:
- Numerical solver tried to find q where g(q) = [0,0,0,0]
- ✗ FAILED: No such configuration exists!
- Result: ||ẋ|| = 38.31 → massive accelerations
```

## Why This Matters for Linearization

### Cart-Pendulum: Linearization Works ✅

```python
# At equilibrium: pendulum hanging down
q_eq = [0.0, π, π]
v_eq = [0.0, 0.0, 0.0]
τ_eq = [0.0, 0.0, 0.0]

# Verify equilibrium
ẋ = f(x_eq, τ_eq) = [0, 0, 0, 0, 0, 0]  ✓

# Drake's Linearize() works!
A = Linearize(plant, context, ...)  ✓ SUCCESS!
```

You can linearize around:
- **Downward equilibrium** (stable): θ = [π, π] → Use LQR for swing-up
- **Upward equilibrium** (unstable): θ = [0, 0] → Use LQR for balancing

### Cup Manipulator: Linearization Fails ❌

```python
# Attempt equilibrium: pendulum hanging, arms at origin
q_attempt = [0.0, 0.0, π, π]
v_attempt = [0.0, 0.0, 0.0, 0.0]
τ_attempt = [g_L1, g_L2, 0, 0]  # Gravity compensation

# Check equilibrium
ẋ = f(x_attempt, τ_attempt) = [0, 0, 0, 0, -0.95, 0.36, 38.32, 2.53]  ✗

# Drake's Linearize() fails!
RuntimeError: Not an equilibrium point!
```

## Root Cause Analysis

| Aspect | Cart-Pendulum | Cup Manipulator |
|--------|---------------|-----------------|
| **Motion Type** | Translation (horizontal) | Rotation (angular) |
| **Gravity Direction** | Vertical (⊥ to motion) | Radial (creates torques) |
| **Mass Distribution** | Point mass cart | Distributed mass along arms |
| **Gravity on Actuated DOFs** | Zero! | Non-zero! |
| **Equilibrium Exists?** | YES ✅ | NO ❌ |
| **Can use Linearize()?** | YES ✅ | NO ❌ |

## The Geometric Insight

### Cart-Pendulum
```
Gravity: ↓ (vertical)
Cart:    ← → (horizontal)
         
Dot product = 0 → No coupling!
```

The cart's motion direction is **orthogonal** to gravity. This geometric decoupling means:
- Cart experiences zero gravity force in its motion direction
- Can be stationary with zero applied force
- Pendulum can hang at equilibrium

### Cup Manipulator
```
Gravity: ↓ (vertical)
L1 joint: ⟳ (rotation about vertical axis - but arm extends horizontally!)
L2 joint: ⟳ (rotation - creates moment arm)

Cross product ≠ 0 → Always coupled!
```

The manipulator joints rotate masses **at a distance** from the joint axis:
- Creates moment arms: τ = r × F
- Gravity creates torques about joint axes
- Torques depend on configuration: τ(q)
- **Cannot be zero everywhere**

## Numerical Evidence

### Cart-Pendulum (with zero damping)
```python
# Equilibrium search
q_eq = [0.0, 3.14159, 3.14159]  # x=0, pendulum down
g(q_eq) = [0.0, ~0.0, ~0.0]
||ẋ|| = 1e-16  ✓ TRUE EQUILIBRIUM!

Linearize() → ✅ SUCCESS
```

### Cup Manipulator (with zero damping)
```python
# Equilibrium search
fsolve(lambda q: g(q), initial_guess)
Result: ✗ "iteration is not making good progress"

No configuration found where g(q) = [0,0,0,0]
||ẋ|| = 38.31  ✗ NOT AN EQUILIBRIUM

Linearize() → ❌ FAILS
```

## Conclusion

The fundamental difference is **motion type**:

**Translational motion (cart):**
- Can be perpendicular to gravity
- Gravity force doesn't oppose motion
- Equilibrium possible with zero control

**Rotational motion (manipulator):**
- Gravity creates torques on extended masses
- Torques always present (unless massless or at specific angles)
- Equilibrium requires active compensation
- With underactuation, can't compensate all DOFs

This is why:
- Cart-pendulum is a **classic textbook example** for LQR/swing-up control
- Cup manipulator requires **trajectory optimization + tracking control** (TVLQR/MPC)

**Bottom line:** The cart's translational motion breaks the gravity coupling that the manipulator's rotational joints cannot escape!
