# GitHub Copilot Instructions for Isaac_sim_robotics

## Project Overview

Research codebase for robotics control systems combining **NVIDIA Isaac Sim** and **PyDrake** for simulation, control theory (LQR, OFC), and biomechanics (muscle dynamics). Focus on cart-pendulum systems, manipulators, and impedance control.

## Critical Architecture Patterns

### Dual-Framework Design
This project uses TWO physics engines depending on the file:
- **Isaac Sim** (`test_*.py`, `check_*.py`): GPU-accelerated PhysX, real-time 3D visualization
- **PyDrake** (`script_*.py`): MultibodyPlant with analytical dynamics, Meshcat visualization

**Never mix**: Isaac Sim scripts import `from isaacsim import SimulationApp` FIRST, PyDrake scripts use `from pydrake.all import`.

### Drake's Non-Intuitive Joint Ordering
**CRITICAL**: Drake's `GetJointIndices()` returns joints in **URDF parse order**, NOT alphabetical:
```python
# For cup_manipulator.urdf: joints are [link2_link1, link1_base]
# When setting positions, Drake expects [q2, q1], NOT [q1, q2]
plant.SetPositions(context, model_instance, np.array([q2, q1]))  # Correct
plant.SetPositions(context, model_instance, np.array([q1, q2]))  # WRONG!
```

Always verify with:
```python
for i, idx in enumerate(plant.GetJointIndices(model_instance)):
    joint = plant.get_joint(idx)
    print(f"[{i}] {joint.name()}")
```

### Configuration System (`robot_types.py`)
All robots use **dataclass-based configs** for type safety and reproducibility:
```python
from robot_types import create_cup_manipulator_config, ManipulatorConfig
config = create_cup_manipulator_config(
    urdf_path="path/to/urdf",
    joint_angles=(np.deg2rad(-10), np.deg2rad(20)),  # [q1, q2]
    damping=(0.1, 0.1),
)
```
Configs are **frozen** (`@dataclass(frozen=True)`) and auto-serialize to JSON via `dataclasses.asdict()` for experiment tracking.

### Two-System Architecture (Drake Controllers)
Scripts like `script_cup_manipulator_controller_drake.py` build **separate plants for physics and control**:
```
MultibodyPlant (physics) → [state] → PDController (separate LeafSystem) → [τ] → Plant
```
This is NOT redundant—it's Drake's pattern for custom controllers. The controller plant is just for computing reference dynamics/Jacobians.

## Essential Workflows

### Running Simulations

**PyDrake scripts** support multiple modes via `--mode`:
```bash
# Scene visualization only
python script_cart_pendulum_2d_extended_ofc.py --mode scene-viz

# LQR control with muscle dynamics
python script_cart_pendulum_2d_extended_ofc.py --mode finite-horizon-lqr-for-min-effort_cart_pend_only --duration 10.0

# Manipulator IK tracking cart position
python script_cart_pendulum_2d_extended_ofc.py --mode manip-ik-follows-cart --target-x -0.8 --target-y 1.0
```

**Isaac Sim scripts** require conda environment:
```bash
conda activate env_isaacsim
python test_cart_pendulum_2dof.py  # Isaac Sim + Python API
```

### Model Conversion Pipeline
URDF models from Onshape (CAD):
```bash
# 1. Convert Onshape → URDF (requires API keys in .env)
./step1_convert_from_onshape_cup_manipulator.sh

# 2. Convert STL meshes → OBJ (for Drake/Isaac Sim compatibility)
python step2_convert_stl_to_obj.py cup_manipulator

# 3. Update URDF mesh references
python step3_urdf_stl_to_obj.py cup_manipulator
```

### Environment Setup
```bash
# PyDrake environment (most script_*.py files)
conda activate pydrake
pip install pydrake termcolor matplotlib scipy

# Isaac Sim environment (test_*.py files)
conda activate env_isaacsim  
pip install isaacsim  # Requires full Isaac Sim installation
```

## Project-Specific Conventions

### State Vectors
State ordering is **physics-driven**, not alphabetical:
- **Cart-Pendulum 2D** (8D): `[x, y, α, β, ẋ, ẏ, α̇, β̇]`  
  α = pitch, β = roll (gimbal angles)
- **Extended System** (14D): adds `[F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]`  
  Muscle forces + ZFT reference mass states

### Linearization Method
Use Drake's **automatic Jacobian** (NOT manual formulas):
```python
from pydrake.systems.primitives import Linearize

linearized = Linearize(
    nonlinear_plant,
    context,
    input_port_index=plant.get_actuation_input_port().get_index(),
    output_port_index=plant.get_state_output_port().get_index(),
)
A, B, C, D = linearized.A(), linearized.B(), linearized.C(), linearized.D()
```
See `LINEARIZATION_IMPLEMENTATION_SUMMARY.md` for why this replaced manual Jacobian construction.

### Inverse Kinematics Pattern
When using IK for manipulator positioning (e.g., `test_manipulator_ee_trajectory.py`):
```python
from pydrake.all import InverseKinematics, Solve

def solve_manipulator_ik(plant, manipulator, target_xy, q_seed, pos_tol=0.01):
    ik = InverseKinematics(plant)
    ik_context = ik.context()
    
    # CRITICAL: Seed must match Drake's joint ordering
    plant.SetPositions(ik_context, manipulator.model_instance, 
                      np.array([q_seed[1], q_seed[0]]))  # [q2, q1]
    
    ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
    ik.AddPositionConstraint(
        frameB=ee_frame,
        p_BQ=manipulator.EE_OFFSET,
        frameA=plant.world_frame(),
        p_AQ_lower=target_xy - pos_tol,
        p_AQ_upper=target_xy + pos_tol,
    )
    
    result = Solve(ik.prog())
    return result.GetSolution(ik.q()), result.is_success()
```

### Muscle Dynamics Integration
First-order actuator model (biologically inspired):
```python
# Dynamics: Ḟ = (-F + u) / τ
# τ ≈ 0.03s (human muscle time constant)
class MuscleDynamics2D(LeafSystem):
    def DoCalcTimeDerivatives(self, context, derivatives):
        F = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port().Eval(context)
        F_dot = (-F + u) / self.tau
        derivatives.get_mutable_vector().SetFromVector(F_dot)
```
This adds dynamics between neural commands `u` and applied forces `F`.

## Testing & Debugging

### Verify URDF Loading
```python
# Print all joints to verify ordering
for idx in plant.GetJointIndices(model_instance):
    joint = plant.get_joint(idx)
    print(f"{joint.name()}: {joint.num_velocities()} DOF")
```

### Check IK Feasibility
Before running long simulations, test trajectory points:
```python
test_points = [(x, y) for x, y in zip(x_traj, y_traj)]
for target in test_points[::50]:  # Every 50th point
    q_sol, success = solve_manipulator_ik(plant, manip, target, q_seed)
    if not success:
        print(f"IK failed at {target}")
```

### Visualize Meshcat
All PyDrake scripts print:
```
Meshcat: http://localhost:7000
```
Open in browser. Click ▶ to replay recorded animations.

## Key Files & Their Roles

- **`robot_types.py`**: Central config system—ALL robots/scenes defined here
- **`script_cart_pendulum_2d_extended_ofc.py`**: Main research script for 2D OFC with multiple modes
- **`script_cup_manipulator_controller_drake.py`**: Two-system architecture example (plant + controller)
- **`LQR_IMPLEMENTATION_SUMMARY.md`**: Finite-horizon LQR theory & implementation
- **`SYSTEM_ARCHITECTURE_GUIDE.md`**: Block diagrams of muscle dynamics + linearization
- **`model/manipulators/`**: URDF files and mesh assets
- **`tests/`**: Isaac Sim examples (require GPU + Isaac Sim installed)

## Common Pitfalls

1. **Joint Order Confusion**: Always check Drake's `GetJointIndices()` order before setting positions
2. **Missing Actuators**: URDF joints need explicit actuators: `plant.AddJointActuator("tau_joint", joint)`
3. **Isaac Sim Import Order**: `SimulationApp()` MUST be first import, before any `isaacsim.*` modules
4. **Linearization Point**: Drake linearizes around `context` state—set equilibrium BEFORE calling `Linearize()`
5. **IK Seed Warm-Starting**: Use previous IK solution as seed for next iteration to avoid discontinuities

## Documentation References

- **Drake API**: https://drake.mit.edu/pydrake/
- **Isaac Sim**: https://docs.isaacsim.omniverse.nvidia.com/
- **In-Repo Guides**: `LINEARIZATION_IMPLEMENTATION_SUMMARY.md`, `LQR_IMPLEMENTATION_SUMMARY.md`
- **Config System**: `docs/CONFIGURATION_SYSTEM_SUMMARY.md`

---

**Last Updated**: February 13, 2026  
**Frameworks**: PyDrake + Isaac Sim 5.1.0  
**Python**: 3.11 (conda environments: `pydrake`, `env_isaacsim`)
