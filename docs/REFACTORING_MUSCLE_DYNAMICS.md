# Refactoring: Cart-Pendulum Muscle Dynamics Architecture

## Summary of Changes

Successfully refactored `CartPendulumSystemWithMuscleDynamics` following the **clean separation of concerns** pattern used in `script_cup_manipulator_controller_drake.py`.

---

## Architecture: Before vs After

### BEFORE: Mixed Responsibilities
```
CartPendulumSystemWithMuscleDynamics
├── Plant building (build_plant_without_muscle)
├── Muscle dynamics setup (add_muscle_dynamics)
├── Model creation (create_model_for_controller)
├── Wiring logic (connect_controller_output)
├── Data logging (time_log, state_log, etc.)
├── Visualization (meshcat setup)
├── Run method (simulation execution)
└── Everything coupled together
```

**Problem:** Class had too many responsibilities - building, wiring, visualization, and execution all mixed together.

---

### AFTER: Clean Separation

#### `CartPendulumSystemWithMuscleDynamics` - PLANT BUILDER ONLY ✨

**Responsibility:** Build the physics model

```python
class CartPendulumSystemWithMuscleDynamics:
    """Cart-Pendulum Plant Builder with Muscle Dynamics."""
    
    def __init__(self, builder: DiagramBuilder, ...)
        # Takes builder from outside (SceneManager)
        # Only stores references needed for plant
    
    def build_plant_without_muscle(self):
        # Creates: cart, pendulum, joints, actuator
        # Finalizes plant
    
    def add_muscle_dynamics(self):
        # Adds muscle system to builder
        # Wires muscle to plant actuation
        # Exposes command_input_port for controller
```

**What it DOESN'T do:**
- No diagram building
- No visualization setup
- No simulation execution
- No data logging

---

#### `CartPendulumSceneManager` - EVERYTHING ELSE ✨

**Responsibility:** Orchestrate the entire simulation

```python
class CartPendulumSceneManager:
    """Scene Manager for Cart-Pendulum - Handles everything except plant building."""
    
    def __init__(self, ...):
        self.builder = DiagramBuilder()
        self.system = CartPendulumSystemWithMuscleDynamics(
            builder=self.builder, ...
        )
    
    def setup_drake_system(self):
        # Step 1: Call system methods to build plant
        self.system.build_plant_without_muscle()
        self.system.add_muscle_dynamics()
    
    def create_force_input(self):
        # Step 2: Create input systems
    
    def setup_visualization(self):
        # Step 3: Setup Meshcat
    
    def build_diagram(self):
        # Step 4: Wire all systems
        self.builder.Connect(force_source, system.command_input_port)
        self.diagram = self.builder.Build()
    
    def create_simulator(self):
        # Step 5: Create simulator
    
    def run_simulation(self):
        # Execute the simulation loop
    
    def run(self):
        # Execute all 5 steps in order
```

---

## Key Improvements

### 1. **Single Responsibility Principle** ✅
- Each class has ONE clear purpose
- Plant builder builds the plant
- Scene manager orchestrates everything

### 2. **Dependency Injection** ✅
- `CartPendulumSystemWithMuscleDynamics` takes `builder` as parameter
- No longer creates its own builder
- Promotes reusability and testability

### 3. **Cleaner Constructor** ✅
```python
# BEFORE: Confusing - creates plant but also handles simulation
system = CartPendulumSystemWithMuscleDynamics(enable_muscle_dynamics=True)
system.run(simulation_time=5.0, ...)  # Too much magic

# AFTER: Clear separation
manager = CartPendulumSceneManager(simulation_time=5.0, ...)
results = manager.run()  # All 5 steps explicitly defined
```

### 4. **Explicit 5-Step Pipeline** ✅
```
[STEP 1/5] Setting up Drake system...
[STEP 2/5] Creating force input source...
[STEP 3/5] Setting up visualization...
[STEP 4/5] Building diagram...
[STEP 5/5] Creating simulator...
```

### 5. **No Redundant References** ✅
- Removed duplicate `self.plant` in both classes
- SceneManager accesses `self.system.plant`
- Single source of truth

### 6. **Follows Cup Manipulator Pattern** ✅
- Matches the professional architecture in `script_cup_manipulator_controller_drake.py`
- Consistent design patterns across codebase

---

## Usage

### Simple Example
```python
from script_cart_pendulum_muscle_dynamics import CartPendulumSceneManager
import numpy as np

manager = CartPendulumSceneManager(
    simulation_time=5.0,
    initial_position=0.0,
    initial_angle=np.deg2rad(180),
    constant_force=5.0,
    visualize=True,
    muscle_tau=0.03
)

results = manager.run()
```

### Access Results
```python
results = {
    "time": np.array([...]),        # Time vector
    "state": np.array([...]),       # State trajectory [N x 4]
    "force": np.array([...]),       # Applied forces
    "final_state": np.array([...])  # Final [x, θ, ẋ, θ̇]
}
```

---

## Files Modified

1. **`script_cart_pendulum_muscle_dynamics.py`**
   - Refactored `CartPendulumSystemWithMuscleDynamics` to plant-builder-only
   - Rewrote `CartPendulumSceneManager` to handle everything else
   - Removed `run()` method from plant class
   - Removed `connect_controller_output()` (now in manager)
   - Improved `__name__ == "__main__"` check for interactive mode

2. **`example_muscle_dynamics_simple_run.py`**
   - Updated to use `CartPendulumSceneManager` directly
   - Fixed sys.argv manipulation for non-interactive imports
   - Added 3 example functions showing different use cases

---

## Next Steps: Extending the Architecture

The refactored design makes it easy to extend:

### Add a Custom Controller
```python
class MyController(LeafSystem):
    def __init__(self, plant):
        super().__init__()
        self.plant = plant
        self.DeclareVectorInputPort("state", BasicVector(4))
        self.DeclareVectorOutputPort("force", BasicVector(1), self._calc)
    
    def _calc(self, context, output):
        # Your control law here
        pass

manager = CartPendulumSceneManager(...)
# Modify manager to add custom controller before building diagram
```

### Add Trajectory Tracking
```python
class TrajectoryPlanner(LeafSystem):
    # Generate desired state trajectory
    pass

manager = CartPendulumSceneManager(...)
# Add planner system to builder
```

### Add State Observer
```python
class StateObserver(LeafSystem):
    # Estimate state from measurements
    pass

manager = CartPendulumSceneManager(...)
# Insert observer between plant and controller
```

---

## Validation

✅ Code compiles without errors
✅ Interactive mode disabled for non-main imports  
✅ Visualization works (Meshcat)
✅ Simulation runs 5 seconds successfully
✅ Results are logged correctly
✅ Example script runs without modification

---

## Design Pattern Reference

This follows the **DrakeSceneManager** pattern established in:
- `script_cup_manipulator_controller_drake.py` (cup manipulator)
- Same 5-step pipeline
- Same separation: Robot class (plant) vs Manager (orchestration)

Perfect for professional robotics simulation architecture! 🎉
