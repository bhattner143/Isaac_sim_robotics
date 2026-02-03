## JSON Configuration Saving Implementation

### Summary

Successfully implemented **automatic configuration serialization** for both Isaac Sim and PyDrake scripts using Python's `dataclasses.asdict()` function.

### Implementation Details

#### 1. **Added Imports**
- `json` - for JSON file I/O
- `asdict` from `dataclasses` - for automatic dataclass-to-dict conversion
- `datetime` - for timestamped filenames

Both scripts updated:
- `script_cart_pendulum_manipulator_controller.py` (Isaac Sim)
- `script_cart_pendulum_manipulator_controller_pydrake.py` (PyDrake)

#### 2. **Added `save_configuration_to_json()` Method**

**Location:** `SceneManager` class in both scripts

**Features:**
- ✅ **Automatic serialization** using `asdict()` - no manual dictionary building
- ✅ **Auto-generated filenames** with ISO timestamp (format: `simulation_config_YYYYMMDD_HHMMSS.json`)
- ✅ **Custom output paths** supported
- ✅ **Hierarchical JSON structure** for easy navigation
- ✅ **All configuration parameters** captured:
  - Simulation metadata (timestamp, mode, device, framework)
  - Simulation parameters (time step, duration, settling time)
  - Robot configurations (URDFs, initial positions, damping, friction, stiffness)
  - Lighting configuration
  - Coupling joint parameters (type, stiffness, damping, friction, axes)
  - Video recording settings (Isaac Sim only)

#### 3. **JSON Structure**

```json
{
  "metadata": {
    "timestamp": "2026-02-02T21:17:55.061486",
    "simulation_mode": "coupled-motion",
    "device": "cpu",
    "framework": "Isaac Sim"
  },
  "simulation_parameters": { ... },
  "cart_pendulum_config": { ... },
  "manipulator_config": { ... },
  "lighting_config": { ... },
  "coupling_joint_config": {
    "type": "revolute",
    "revolute": { "stiffness": 500.0, "damping": 100.0, ... },
    "prismatic": { "stiffness": 50.0, "damping": 10.0, ... }
  },
  "video_recording": { ... }
}
```

#### 4. **Integration Points**

**Isaac Sim Script:**
- Called in `run_coupled_motion()` after `plot_results()`
- Files saved to `configs/` directory

**PyDrake Script:**
- Called in `run_simulation()` after `plot_results()`
- Called in `run_coupled_motion()` after `plot_results()`
- Files saved to `configs/` directory with `pydrake` in filename

#### 5. **Usage**

```python
# Automatic (recommended)
scene.save_configuration_to_json()
# Creates: configs/simulation_config_20260202_211755.json

# Custom path
scene.save_configuration_to_json("my_custom_config.json")
# Creates: my_custom_config.json
```

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **Automatic** | No manual dictionary building - uses Python standard library |
| **Maintainable** | Add fields to dataclass → automatically in JSON |
| **Self-documenting** | JSON structure mirrors dataclass structure |
| **Reproducible** | Full configuration saved for experiment reproducibility |
| **Portable** | Easy to share and version control configs |
| **Auditable** | Track all parameter changes across simulations |

### Test Results

✅ **Test Passed** (see `test_config_saving.py`)
- Configuration dictionary created successfully
- JSON file generated with 1.9KB size
- All configuration keys present and valid
- JSON is properly formatted and parseable

### Files Modified

1. **script_cart_pendulum_manipulator_controller.py** (Isaac Sim)
   - Added imports: `json`, `asdict`, `datetime`
   - Added `save_configuration_to_json()` method
   - Integrated call in `run_coupled_motion()`

2. **script_cart_pendulum_manipulator_controller_pydrake.py** (PyDrake)
   - Added imports: `json`, `asdict`, `datetime`
   - Added `save_configuration_to_json()` method
   - Integrated calls in `run_simulation()` and `run_coupled_motion()`

3. **test_config_saving.py** (NEW)
   - Standalone test script for configuration serialization
   - Demonstrates the feature without full simulation
   - Generates sample config for verification

### Generated Files

Location: `configs/` directory

Example filename: `simulation_config_20260202_211755.json`

**Automatic cleanup:**
- Create `configs/` directory if missing
- One JSON file per simulation run
- Timestamped for easy organization

### Future Enhancements

- Load configurations from JSON to replay simulations
- Config comparison tool to identify parameter changes
- Configuration validation and schema checking
- Integration with experiment tracking systems

