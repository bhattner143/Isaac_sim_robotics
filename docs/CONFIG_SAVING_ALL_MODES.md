## Configuration Saving for ALL Simulation Modes

### Implementation Summary

✅ **Configuration is now saved for EVERY simulation mode** in both Isaac Sim and PyDrake scripts.

### Where Configuration is Saved

**Isaac Sim Script** (`script_cart_pendulum_manipulator_controller.py`):
- `run_test_scene()` - Scene visualization mode (2 calls: init + optional end)
- `run_coupled_motion()` - Coupled motion simulation (2 calls: init + end)

**PyDrake Script** (`script_cart_pendulum_manipulator_controller_pydrake.py`):
- `run_scene_viz()` - Scene visualization mode
- `run_simulation()` - Physics simulation mode
- `run_coupled_motion()` - Coupled motion simulation

### How It Works

All `save_configuration_to_json()` calls are placed at the **start** of each simulation mode, right after the mode header is printed:

```python
def run_coupled_motion(self):
    """Run simulation with manipulator controlling cart via joint coupling."""
    print(f"\n{'='*70}")
    print("COUPLED MOTION MODE: MANIPULATOR MOVES CART-PENDULUM")
    print(f"{'='*70}\n")
    
    # Save configuration for this run  ← RIGHT HERE!
    self.save_configuration_to_json()
    
    self.setup_drake_system()
    # ... rest of simulation
```

### Benefits of This Approach

| Benefit | Details |
|---------|---------|
| **Automatic** | Every simulation automatically saves its config |
| **Centralized** | Method is in SceneManager (accessible to all modes) |
| **No Duplication** | One method handles all serialization |
| **Timestamped** | Each run gets unique filename: `simulation_config_YYYYMMDD_HHMMSS.json` |
| **Complete** | Captures all parameters regardless of which mode runs |
| **Early Capture** | Config saved at START, not end (captures intended config, not modified state) |

### Execution Flow

When user runs any simulation mode:
```
main()
  ↓
Parse arguments
  ↓
Create SceneManager
  ↓
Initialize stage
  ↓
Run selected mode:
  ├─ run_scene_viz()          ← Saves config
  ├─ run_simulation()         ← Saves config
  └─ run_coupled_motion()     ← Saves config
  ↓
Simulation runs...
  ↓
Plot results (optional)
```

### JSON Files Generated

Location: `configs/` directory

**Example filenames:**
```
configs/
├── simulation_config_20260202_211755.json
├── simulation_config_20260203_100240.json
├── simulation_config_20260203_145632.json
└── ...
```

Each file contains:
- Simulation metadata (timestamp, mode, device, framework)
- All robot configurations (URDFs, positions, joint properties)
- Lighting settings
- Coupling joint parameters
- Video recording settings
- Simulation parameters

### Code Changes Made

**Isaac Sim Script:**
- Added `self.save_configuration_to_json()` in `run_test_scene()` (line 2206)
- Added `self.save_configuration_to_json()` in `run_coupled_motion()` (line 2384)
- Removed duplicate call at end of `run_coupled_motion()`

**PyDrake Script:**
- Added `self.save_configuration_to_json()` in `run_scene_viz()` (line 700)
- Added `self.save_configuration_to_json()` in `run_simulation()` (line 739)
- `run_coupled_motion()` already had it (line 804)

### Testing

Run any simulation mode to verify:
```bash
# Scene visualization
python script_cart_pendulum_manipulator_controller.py --mode scene-viz

# Coupled motion simulation
python script_cart_pendulum_manipulator_controller.py --mode coupled-motion

# PyDrake simulation
python script_cart_pendulum_manipulator_controller_pydrake.py --mode simulation
```

Each will:
1. Print "Saving configuration..." 
2. Generate `configs/simulation_config_YYYYMMDD_HHMMSS.json`
3. Continue with simulation

### Accessing Configuration from Any Mode

Since `save_configuration_to_json()` is a public method of `SceneManager`, you can:

```python
# In main function or anywhere with scene object
scene = SceneManager(...)
scene.initialize_stage()

# Manually save at any point
scene.save_configuration_to_json()
scene.save_configuration_to_json("custom_config.json")  # Custom path
```

### Key Insight

By placing the save call at the **start** of each simulation mode (not end), you capture:
- **What was intended** (the configuration user requested)
- **Not what was mutated** (final state after simulation)
- **Clean audit trail** (config saved before any modifications)

This approach is perfect for reproducibility - you can load the saved JSON later to replay the exact same experiment.

