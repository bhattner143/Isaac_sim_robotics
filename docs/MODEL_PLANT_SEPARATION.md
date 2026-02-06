# Model-Plant Separation in Computed Torque Controller

## Overview

The computed torque controller now implements **model-plant separation**, a fundamental architecture in robust control systems. This separates the "real" system (plant) from the controller's internal dynamics model (model).

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   PLANT (Real System)   │         │   MODEL (Controller)     │
│   ═══════════════════   │         │   ══════════════════     │
│                         │         │                          │
│  - In Drake Diagram     │         │  - NOT in diagram        │
│  - Executes physics     │         │  - For computation only  │
│  - "True" parameters    │         │  - Nominal parameters    │
│  - Unknown/uncertain    │         │  - Controller's belief   │
│                         │         │                          │
│  Provides: [q, v]       │────────>│  Reads state             │
│  (via output port)      │         │  (via input port)        │
│                         │         │                          │
│                         │         │  Computes:               │
│                         │         │  M_model, C_model, g     │
│                         │         │  CalcInverseDynamics()   │
│                         │         │         │                │
│  Receives: u ───────────│<────────│  Outputs: u              │
│  (via actuation port)   │         │                          │
└─────────────────────────┘         └──────────────────────────┘
      EXECUTES                           COMPUTES
```

## Key Concepts

### What is Plant?
- The **actual robot** (in simulation or hardware)
- Part of Drake's diagram system
- Executes forward dynamics: `M(q)q̈ + C(q,v)v = τ_g(q) + u`
- Has "true" unknown parameters (masses, inertias, friction)
- Cannot be changed in deployed system

### What is Model?
- Controller's **internal representation** of the robot
- NOT in Drake diagram (standalone for calculations)
- Used only for inverse dynamics: `τ* = M_model(q)v̇ + C_model(q,v) + g_model(q)`
- Has nominal/estimated parameters
- Can be updated independently of plant

## Implementation

### Controller Initialization
```python
class ComputedTorqueController(LeafSystem):
    def __init__(self, 
                 plant: MultibodyPlant,  # Real system
                 model: MultibodyPlant,  # Controller's model
                 ...):
        self.plant = plant  # For monitoring only
        self.model = model  # For calculations
        self.model_context = model.CreateDefaultContext()
```

### Control Computation
```python
def CalcControlTorque(self, context, output):
    # 1. Read state from PLANT (via input port)
    state = self.get_input_port(0).Eval(context)
    q, v = state[:nq], state[nq:]
    
    # 2. Update MODEL context with observed state
    self.model.SetPositions(self.model_context, q)
    self.model.SetVelocities(self.model_context, v)
    
    # 3. Compute control using MODEL dynamics
    torque = self.model.CalcInverseDynamics(
        self.model_context, vdot_cmd, forces
    )
    
    # 4. Send command to PLANT (via output port)
    output.SetFromVector(u)
```

## Benefits

### 1. Sim-to-Real Transfer
```python
# In simulation:
plant = SimulatedRobot()  # Drake MultibodyPlant
model = NominalModel()    # Same initially

# On real hardware:
plant = RealRobotInterface()  # Actual robot!
model = NominalModel()        # SAME controller model
# No code changes needed!
```

### 2. Robustness Testing
```python
# Test with parameter mismatch
plant.SetMass("link1", 2.0)    # True mass
model.SetMass("link1", 1.5)    # Wrong estimate!

# Controller still works due to feedback:
# Kp, Kd terms compensate for modeling errors
```

### 3. Adaptive Control
```python
# Update model parameters online
model.SetMass("link1", estimated_mass)
# Plant unchanged, controller adapts
```

### 4. Unknown Payloads
```python
# Ball in cup changes plant mass
plant.AddBallToCup(mass=0.1)  # Physical change
# Model doesn't know about ball
# Feedback compensates automatically!
```

## Current Implementation

### For Now: Perfect Model
```python
# Plant and model have SAME parameters
model_plant = CreateSameRobotAs(plant)
```

**Why?** Start with ideal case to verify implementation.

### Future: Imperfect Model
```python
# Introduce uncertainty
plant.SetMass("link1", true_mass)
model.SetMass("link1", estimated_mass)

# Test controller robustness
assert tracking_error < tolerance
```

## Mathematical Foundation

### Plant Dynamics (Unknown)
```
M_plant(q) v̇ + C_plant(q,v) + g_plant(q) = B u
```

### Controller Computes (Using Model)
```
u = B† [M_model(q) v̇_cmd + C_model(q,v) + g_model(q)]
```

where:
```
v̇_cmd = v̇_d + Kp·e + Kd·ė
```

### Closed-Loop (Model ≠ Plant)
```
M_plant v̇ = M_model v̇_cmd + (C_model - C_plant) + (g_model - g_plant)
```

**Error terms** (model-plant mismatch) are compensated by **feedback** (Kp·e + Kd·ė).

## How to Test Robustness

### 1. Mass Uncertainty
```python
# In DrakeSceneManager.add_controller():
# After creating model_plant, modify parameters:
model_link1 = model_plant.GetBodyByName("link1", ...)
model_link1.SetMass(1.5)  # Controller thinks mass is 1.5 kg

# Plant keeps true mass (2.0 kg from URDF)
# Run simulation and check tracking performance
```

### 2. Friction Mismatch
```python
# Model assumes no friction
for joint in model_plant.GetJointIndices(...):
    model_plant.GetJoint(joint).set_default_damping(0.0)

# Plant has real friction (from URDF)
# Feedback should compensate
```

### 3. Unknown External Forces
```python
# Apply disturbance to plant, not in model
plant.AddExternalForce(...)

# Model doesn't know about force
# Controller should reject disturbance
```

## Comparison: Old vs New

| Aspect | Old (Single Plant) | New (Model-Plant Sep) |
|--------|-------------------|----------------------|
| Controller uses | `self.plant` | `self.model` |
| Inverse dynamics | Same plant as physics | Separate model |
| Sim-to-real | Need to rewrite controller | Just swap plant |
| Robustness test | Hard to introduce mismatch | Easy: modify model params |
| Adaptive control | Not possible | Can update model online |

## Example: Testing with 10% Mass Error

```python
# In DrakeSceneManager.add_controller() around line 1187:

# After model_plant.Finalize():
print("Introducing 10% mass error for robustness test...")

for body_index in range(model_plant.num_bodies()):
    body = model_plant.get_body(body_index)
    if body.name() != "world":
        true_mass = body.default_mass()
        estimated_mass = true_mass * 0.9  # 10% underestimate
        body.SetMass(model_plant.get_mutable_body(body_index), estimated_mass)
        print(f"  {body.name()}: {true_mass:.3f} kg → {estimated_mass:.3f} kg")
```

## References

1. Slotine & Li, "Applied Nonlinear Control", Chapter 6
2. Spong et al., "Robot Modeling and Control", Chapter 8
3. Drake Documentation: MultibodyPlant API

## Next Steps

1. ✅ Implement model-plant separation (DONE)
2. ⬜ Test with perfect model (baseline)
3. ⬜ Introduce parameter mismatch (robustness)
4. ⬜ Compare with PD controller under uncertainty
5. ⬜ Implement adaptive update law for model parameters
