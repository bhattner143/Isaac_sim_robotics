# Refactoring: script_cart_pendulum_diagram.py - Scene Manager Architecture

## Summary

Refactored `script_cart_pendulum_diagram.py` to follow the **clean separation of concerns** pattern established in `script_cart_pendulum_muscle_dynamics.py`.

---

## Architecture: Before vs After

### BEFORE: Mixed Responsibilities

```
CartPendulumSystem
├── Plant building
├── Data logging fields
├── Controller state
├── Mesh cat/visualization fields
├── Diagram/simulator fields
└── Everything mixed together
```

### AFTER: Clean Separation

#### `CartPendulumSystem` - PLANT BUILDER ONLY ✨

**Responsibility:** Build the physics model

```python
class CartPendulumSystem:
    """Cart-Pendulum Plant Builder."""
    
    def __init__(self, builder: DiagramBuilder):
        self.builder = builder
        self.plant = None
        self.scene_graph = None
    
    def build_plant(self):
        # Creates: cart, pendulum, joints, actuator
        # Finalizes plant
    
    def create_model_for_controller(self):
        # Creates identical model for controller
```

**What it DOESN'T do:**
- No diagram building
- No controller management
- No visualization setup
- No simulation execution
- No data logging

---

#### `DrakeSceneManager` - EVERYTHING ELSE ✨

**Responsibility:** Orchestrate the entire simulation

**5-Step Pipeline:**

```python
def __init__(self, controller_mode, plant_type, visualize):
    self.builder = DiagramBuilder()
    self.system = CartPendulumSystem(self.builder)
    # ... initialization

def setup_drake_system(self):
    """[STEP 1/5] Build the plant"""
    self.system.build_plant()

def add_controller(self):
    """[STEP 2/5] Add controller to diagram"""
    # Create and wire controller

def setup_visualization(self):
    """[STEP 3/5] Setup Meshcat"""
    # Initialize visualization

def build_diagram(self):
    """[STEP 4/5] Build diagram"""
    self.diagram = self.builder.Build()

def create_simulator(self):
    """[STEP 5/5] Create simulator"""
    self.simulator = Simulator(self.diagram)
    # Set initial conditions

def run_full_simulation(self):
    # Execute all 5 steps
    self.setup_drake_system()
    self.add_controller()
    self.setup_visualization()
    self.build_diagram()
    self.create_simulator()
    self.run_simulation()
```

---

## Key Improvements

### 1. **Single Responsibility Principle** ✅
- Plant builder = builds the plant
- Scene manager = orchestrates everything

### 2. **Dependency Injection** ✅
- `CartPendulumSystem` takes `builder` as parameter
- No longer creates its own builder
- Promotes reusability

### 3. **Explicit 5-Step Pipeline** ✅
```
[STEP 1/5] Setting up Drake system...
[STEP 2/5] Adding controller...
[STEP 3/5] Setting up visualization...
[STEP 4/5] Building diagram...
[STEP 5/5] Creating simulator...
```

### 4. **Cleaner Plant Class** ✅
- Removed: controller, meshcat, diagram, simulator fields
- Removed: time_log, state_log, force_log, error_log fields
- Kept: Only builder, plant, scene_graph

### 5. **Follows Cup Manipulator Pattern** ✅
- Consistent design across codebase
- Professional robotics software architecture

### 6. **Improved Auto-Browser Opening** ✅
- Meshcat URL automatically opens in browser
- Graceful error handling if browser unavailable

---

## Code Comparison

### Creating CartPendulumSystem

**BEFORE:**
```python
system = CartPendulumSystem()
system.builder = some_builder  # Manual assignment
system.build_plant()
```

**AFTER:**
```python
system = CartPendulumSystem(builder)  # Dependency injection
system.build_plant()
```

### Running Simulation

**BEFORE:**
```python
manager = DrakeSceneManager(...)
manager.setup_drake_system()
manager.add_controller()
manager.setup_visualization()
manager.build_diagram()
manager.create_simulator()
# ... unclear pipeline
```

**AFTER:**
```python
manager = DrakeSceneManager(...)
manager.run_full_simulation()  # All 5 steps explicit
```

---

## Impact on Other Classes

The following classes remain UNCHANGED:
- `CartPendulumSystemByEqns` (LeafSystem for nonlinear equations)
- `CartPendulumSystemLinearizedWithMuscleDynamics` (LeafSystem for linearized dynamics)
- `PDController`, `ComputedTorqueController`, etc. (controller classes)

Only the orchestration changed, not the physics or control logic.

---

## Testing

✅ Refactoring maintains all functionality:
- Plant building works
- Controller wiring works
- Visualization works
- Simulation runs
- All modes supported (pd, computed-torque, scene-viz, etc.)

---

## Design Pattern Reference

This follows the **SceneManager Pattern**:
- `CartPendulumSystem` ≈ Robot class (physics model)
- `DrakeSceneManager` ≈ Manager class (orchestration)

Same pattern used in:
- `script_cup_manipulator_controller_drake.py`
- `script_cart_pendulum_muscle_dynamics.py` (recently refactored)

Professional robotics software architecture! 🎉
