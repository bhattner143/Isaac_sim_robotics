# Context Cache: Isaac Sim Robotics Repository

> **Generated**: 2026-04-24
> **Purpose**: Comprehensive reference for the entire codebase. This cache captures all modules, classes, key equations, data flows, and configuration schemas.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Modules](#core-modules)
   - [actuators/](#actuators)
   - [cable/](#cable)
   - [controller/](#controller)
   - [robots/](#robots)
   - [configs/](#configs)
   - [contraction-theory/](#contraction-theory)
   - [pydrake/](#pydrake)
4. [Key Classes Reference](#key-classes-reference)
5. [Mathematical Models](#mathematical-models)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Configuration Schema](#configuration-schema)
8. [Simulation Scripts](#simulation-scripts)
9. [Integration Patterns](#integration-patterns)
10. [Testing & Validation](#testing--validation)

---

## Project Overview

This repository contains robotics simulation projects using **NVIDIA Isaac Sim** and **PyDrake** for control systems, manipulation, and dynamics research.

### Primary Research Areas
- **Cable-driven tendon manipulators** (2-DOF planar arms)
- **Series Elastic Actuators (SEA)** for compliant joint actuation
- **Cart-pendulum systems** (inverted pendulum on cart, 2D/3D variants)
- **Contraction theory** for robust control (CCM, C3M, CVSTEM)
- **Exoskeleton integration** with spring-damper models

### Technology Stack
| Component | Technology |
|-----------|------------|
| Physics Engine | NVIDIA Isaac Sim 5.1.0, PyDrake |
| Control Design | Drake (automatic differentiation, linearization) |
| Optimization | YALMIP, SOSTOOLS, Mosek (MATLAB) |
| RL Training | Robot-RL framework |
| Visualization | MeshCat, Isaac Sim GUI |

---

## Directory Structure

```
isaac_sim_robotics/
├── actuators/              # Actuator dynamics models
│   ├── __init__.py
│   ├── motor.py            # Motor configuration classes
│   ├── motor_dynamics.py   # Motor dynamics (torque/position modes)
│   ├── sea.py              # Series Elastic Actuator (PyDrake)
│   ├── sea_exo.py          # SEA with exoskeleton
│   └── sea_isaacsim.py     # SEA for Isaac Sim
│
├── cable/                  # Cable routing and mechanics
│   ├── __init__.py
│   ├── cable.py            # Basic cable model
│   ├── cable_with_exo_springs.py      # Cable + exo springs
│   ├── cable_with_exo_springs_elbow_follow.py
│   ├── drake_plant.py      # Drake plant integration
│   ├── pulley.py           # Pulley geometry
│   ├── routing.py          # Cable routing logic
│   └── test_cable_routing_viz.py
│
├── controller/             # Control system implementations
│   ├── __init__.py
│   ├── controller.py       # ComputedTorqueController, SEACableController
│   ├── c3m_controller.py   # Contraction-based control
│   ├── computed_torque_isaacsim.py
│   ├── ik_system.py        # Inverse kinematics
│   └── trajectory.py       # Trajectory generation
│
├── robots/                 # Robot model definitions
│   ├── __init__.py
│   ├── cup_manipulator.py              # Base manipulator + CartPendulum2DExtended
│   ├── cup_manipulator_tendon.py       # Tendon-driven variant
│   ├── cup_manipulator_tendon_with_exo.py
│   ├── cup_manipulator_cable.py
│   └── cup_manipulator_tendon_isaac.py
│
├── configs/                # Configuration system
│   ├── __init__.py
│   ├── simulation_config_*.json        # Simulation snapshots
│   ├── robot/              # Robot-specific configs
│   └── controller/         # Controller-specific configs
│
├── contraction-theory/     # Mathematical control theory (MATLAB)
│   ├── C3M/                # Contraction-based control metrics
│   ├── cvstem/             # CVSTEM algorithms
│   ├── ncm/                # Neural Contraction Metrics
│   ├── script_*.m          # MATLAB scripts for CCM analysis
│   └── YALMIP/, SOSTOOLS/, sedumi/, mosek/
│
├── pydrake/                # PyDrake-specific scripts
│   ├── script_pydrake_cart_pole_pydrake.py
│   └── script_pydrake_simple_pendulum_lqr_*.py
│
├── model/                  # Robot models (URDF, USD)
│   ├── model_using_onshape_robotics_toolkit/
│   └── model_using_onshape_to_robot/
│
├── docs/                   # Documentation
│   ├── BALL_GIMBAL_ANALYSIS.md
│   ├── BUILD_CUP_WITH_PENDULUM.md
│   ├── CODE_ORGANIZATION_ANALYSIS.md
│   ├── CONFIGURATION_SYSTEM_SUMMARY.md
│   ├── CONTROL_MODES.md
│   ├── CUP_LINEARIZED_IMPLEMENTATION.md
│   ├── LQR_IMPLEMENTATION_SUMMARY.md
│   └── ... (30+ documentation files)
│
├── data/                   # Simulation output data
│   └── cart_position_velocity.csv
│
├── plots/                  # Generated plots and figures
├── resources/              # Static resources
├── utils/                  # Utility functions
├── project_utils/          # Project-specific utilities
│
├── rl/                     # Reinforcement learning
├── robot-rl/               # Robot RL framework
│
├── examples/               # Example scripts
├── tests/                  # Test suite
├── tests2/                 # Additional tests
├── scripts_old/            # Legacy scripts
├── archive/                # Archived code
├── backup/                 # Backup files
│
├── script_*.py             # Main simulation scripts (root level)
├── sweep_exo_ab*.py        # Parameter sweep scripts
├── test_*.py               # Test scripts
├── demo_*.py               # Demo scripts
│
├── README.md               # Project overview and setup guide
├── SYSTEM_ARCHITECTURE_GUIDE.md
├── requirements_env_isaacsim.txt
├── setup_environment.sh
└── launch_isaac.sh
```

---

## Core Modules

### actuators/

#### `motor.py`
Motor configuration dataclasses for parameterizing motor dynamics.

```python
@dataclass
class MotorModelConfig:
    gear_ratio: float = 6.0              # N: motor-to-output gear ratio
    rotor_inertia_motor: float = 1e-4    # J_m [kg·m²]
    viscous_damping_joint: float = 0.1   # b_v_joint [N·m·s/rad]
    position_servo_bandwidth: float = 30.0  # ω_m [rad/s]
    max_velocity_joint: float = 10.0     # max joint velocity [rad/s]
```

#### `motor_dynamics.py`
Abstract motor dynamics models with two operational modes.

**Key Classes:**
- `MotorMode(Enum)` - `TORQUE` (2nd-order) or `POSITION` (1st-order)
- `MotorDynamics(ABC)` - Abstract base class
- `PositionServoMotor(MotorDynamics)` - First-order position servo
- `TorqueMotor(MotorDynamics)` - Second-order rotor dynamics
- `create_motor_dynamics()` - Factory function

**Equations:**
- Position mode: `l̇_m = ω_m · (l_m_des − l_m)`
- Torque mode: `J_m · θ̈_m = τ_m − b_m · θ̇_m − r_p · F / N`

#### `sea.py`
Series Elastic Actuator for cable-driven joint 2.

**Key Class:** `SEACableActuator(LeafSystem)`

**Physical Topology:**
```
Motor drum → cable → Big Pulley → SPRING → Link 2 anchor
```

**Ports:**
- Input: `tau_desired` [2], `plant_state` [n]
- Output: `actuation` [2], `diagnostics` [9]

**Unilateral Cable Model:**
```
δ > 0  →  green taut:  T_green = max(F_raw, 0),  T_red = 0
δ < 0  →  red taut:    T_green = 0,  T_red = max(−F_raw, 0)
δ = 0  →  both slack:  T_green = T_red = 0
τ₂_out = r_p · (T_green − T_red)
```

**Diagnostics Vector [9]:**
`[motor_pos, motor_aux, δ, F_cable, τ₁_des, τ₂_des, T_green, T_red, τ_motor]`

#### `sea_exo.py`
SEA variant with exoskeleton spring integration.

#### `sea_isaacsim.py`
SEA implementation for Isaac Sim simulation environment.

---

### cable/

#### `cable.py`
Basic cable model with spring-damper dynamics.

**Key Parameters:**
- `k_s` - Cable spring stiffness [N/m]
- `b_c` - Cable dashpot damping [N·s/m]
- `r_p` - Pulley radius [m]

#### `cable_with_exo_springs.py`
Cable model with exoskeleton spring attachment points.

#### `cable_with_exo_springs_elbow_follow.py`
Cable routing that follows elbow joint motion.

#### `pulley.py`
Pulley geometry calculations.

**Key Class:** `PulleyGeometry`
- Computes wrap angles
- Calculates cable length from joint angles
- Handles multi-pulley routing

#### `routing.py`
Cable routing logic for complex paths.

**Key Functions:**
- `compute_cable_length(q1, q2)` - Total cable length for given joint config
- `get_routing_points()` - 3D waypoints for cable path

#### `drake_plant.py`
Drake plant integration for cable systems.

---

### controller/

#### `controller.py`
Main controller implementations.

**Key Classes:**

1. **`ComputedTorqueController(LeafSystem)`**
   - Feedback-linearizes 2-DOF manipulator
   - Uses analytical 2R IK
   - PD law in joint space with feedforward

   **Control Law:**
   ```
   a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_ref − q̇)
   τ = M(q)·a_des + h(q, q̇)
   ```

   **Ports:**
   - Input: `desired_ee_pos` [2], `ee_vel_ref` [2], `ee_acc_ref` [2], `plant_state` [n]
   - Output: `actuation` [2], `joint_positions` [2], `torques_raw` [2], `cable_tensions` [2]

   **Gains:**
   - `Kp = 400.0` [s⁻²] → ωn = 20 rad/s
   - `Kd = 40.0` [s⁻¹] → ζ = 1 (critically damped)

2. **`SEACableController(LeafSystem)`**
   - Monolithic CT + first-order SEA cable model
   - Joint 1: CT direct drive (rigid)
   - Joint 2: Cable spring actuation

   **Motor Target:**
   ```
   l_m_des = r_p·q₂ + τ₂_des / (k_s·r_p)
   dl_m/dt = ω_m·(l_m_des − l_m)
   ```

#### `c3m_controller.py`
Contraction-based control using C3M (Constrained Control Contraction Metrics).

#### `computed_torque_isaacsim.py`
Isaac Sim variant of computed torque controller.

#### `ik_system.py`
Inverse kinematics system.

**Key Features:**
- Analytical 2R IK solver
- Jacobian-based velocity/acceleration mapping
- Warm-started seed for continuity

#### `trajectory.py`
Trajectory generation for reference tracking.

---

### robots/

#### `cup_manipulator.py`
Base manipulator class with Drake integration.

**Key Classes:**

1. **`RobotBase(ABC)`** - Abstract base class
   - Template method pattern
   - URDF loading via Drake Parser
   - Joint property management
   - State initialization

2. **`CupManipulator(RobotBase)`** - 2-DOF planar manipulator
   - Joint names: `JT1_NAME` (base→link1), `JT2_NAME` (link1→link2)
   - End-effector frame at cup center
   - IK solver integration

   **EE Offsets (from link2):**
   ```python
   EE_XYZ_LINK2 = [1.2515, 0.0, 0.15]   # meters
   EE_RPY_LINK2 = [0.0, 0.0, 0.0]       # radians
   ```

   **Key Methods:**
   - `add_end_effector_frame()` - Creates cup_center frame
   - `add_joint_actuators()` - Adds torque actuators
   - `solve_initial_pose_via_ik()` - IK solver
   - `get_positions_user_order()` - Returns [q1, q2]
   - `get_jt()`, `set_jt()` - Joint angle access
   - `get_jt_velocity()`, `set_jt_velocity()` - Joint velocity access

3. **`CartPendulum2DExtended`** - 2D cart-pendulum
   - State: [x, y, α, β, ẋ, ẏ, α̇, β̇] (8D)
   - Input: [F_x, F_y] (2D force)
   - Structure: world → x_slider → y_slider → cart → pendulum (gimbal)

#### `cup_manipulator_tendon.py`
Tendon-driven variant with cable actuation.

**Key Constants:**
```python
PULLEY_RADIUS = 0.04775  # meters
```

#### `cup_manipulator_tendon_with_exo.py`
Tendon manipulator with exoskeleton attachment.

---

### configs/

Configuration system using JSON and Python dataclasses.

**Structure:**
```
configs/
├── __init__.py
├── simulation_config_*.json    # Timestamped snapshots
├── robot/
│   ├── __init__.py
│   └── robot_types.py          # ManipulatorConfig dataclass
│   └── robot_configs.py        # CartPendulumPhysicsConfig
└── controller/
    ├── __init__.py
    └── controller_types.py     # ControllerConfig dataclass
```

---

### contraction-theory/

MATLAB-based contraction theory analysis tools.

**Key Scripts:**
| Script | Purpose |
|--------|---------|
| `script_1*.m` | Basic contraction analysis |
| `script_6*.m` | 2-link planar arm with PD control |
| `script_9*.m` | 2-link computed torque control |
| `script_10*.m` | Cart-pendulum CCM/C3M |
| `script_11*.m` | Polynomial CCM with YALMIP |
| `script_12*.m` | Van der Pol oscillator CCM |
| `script_13*.m` | Cup manipulator SEA CCM |

**Dependencies:**
- YALMIP (optimization)
- SOSTOOLS (sum-of-squares)
- Mosek (solver)
- SeDuMi (solver)

---

### pydrake/

PyDrake-specific demonstration scripts.

- `script_pydrake_cart_pole_pydrake.py` - Cart-pole simulation
- `script_pydrake_simple_pendulum_lqr_demo.py` - LQR demo
- `script_pydrake_simple_pendulum_lqr_from_scratch.py` - LQR from scratch

---

## Key Classes Reference

### Class Hierarchy

```
RobotBase (ABC)
├── CupManipulator
│   ├── CupManipulatorTendon
│   ├── CupManipulatorTendonWithExo
│   ├── CupManipulatorCable
│   └── CupManipulatorTendonIsaac

MotorDynamics (ABC)
├── PositionServoMotor
└── TorqueMotor

LeafSystem (PyDrake)
├── ComputedTorqueController
├── SEACableController
└── SEACableActuator
```

### Important Class Attributes

#### CupManipulator
| Attribute | Type | Description |
|-----------|------|-------------|
| `JT1_NAME` | str | Joint 1 name (base→link1) |
| `JT2_NAME` | str | Joint 2 name (link1→link2) |
| `LINK2_NAME` | str | End-effector link name |
| `EE_XYZ_LINK2` | np.array | EE offset from link2 |
| `EE_FRAME_NAME` | str | "cup_center" |
| `model_instance` | int | Drake model instance ID |
| `config` | ManipulatorConfig | Robot configuration |

#### SEACableActuator
| Attribute | Type | Description |
|-----------|------|-------------|
| `_k_s` | float | Cable spring stiffness [N/m] |
| `_b_c` | float | Cable damping [N·s/m] |
| `_r_p` | float | Pulley radius [m] |
| `_tau_max` | float | Torque saturation [Nm] |
| `_motor` | MotorDynamics | Motor dynamics model |
| `_motor_mode` | MotorMode | Active motor mode |

#### ComputedTorqueController
| Attribute | Type | Description |
|-----------|------|-------------|
| `_Kp` | np.array | Position gain [s⁻²] |
| `_Kd` | np.array | Velocity gain [s⁻¹] |
| `_tau_max` | float | Torque limit [Nm] |
| `_ik` | object | IK solver |
| `_r_p` | float | Pulley radius [m] |

---

## Mathematical Models

### 1. Series Elastic Actuator (SEA)

**Spring Force:**
```
δ = l_m − r_p·q₂                    # spring extension [m]
δ̇ = l̇_m − r_p·q̇₂                   # extension rate [m/s]
F_raw = k_s·δ + b_c·δ̇              # raw force [N]
```

**Unilateral Cable:**
```
if δ > 0:  T_green = max(F_raw, 0), T_red = 0
if δ < 0:  T_green = 0, T_red = max(-F_raw, 0)
if δ = 0:  T_green = T_red = 0
F_cable = T_green − T_red
τ₂ = r_p · F_cable
```

**Motor Dynamics (Torque Mode):**
```
J_m · θ̈_m = τ_m − b_m · θ̇_m − r_p·F/N
τ_m = τ₂_des / N
```

**Motor Dynamics (Position Mode):**
```
l̇_m = ω_m · (l_m_des − l_m)
l_m_des = r_p·q₂ + τ₂_des / (k_s·r_p)
```

### 2. Computed Torque Control

**Feedback Linearization:**
```
a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_ref − q̇)
τ = M(q)·a_des + C(q,q̇)·q̇ + g(q)
```

**Closed-Loop Error Dynamics:**
```
ë + Kd·ė + Kp·e = 0
poles at: −ζωn ± ωn√(ζ²−1)
```

**Gain Interpretation:**
```
Kp [s⁻²] → ωn = √Kp        (e.g., Kp=400 → ωn=20 rad/s)
Kd [s⁻¹] → ζ = Kd/(2√Kp)   (e.g., Kd=40 → ζ=1, critically damped)
```

### 3. 2R Manipulator Jacobian

```
J = [−L₁·s₁ − L₂·s₁₂   −L₂·s₁₂]
    [ L₁·c₁ + L₂·c₁₂    L₂·c₁₂]

q̇ = J⁻¹ · ẋ_ee
q̈ = J⁻¹ · (ẍ_ee − J̇·q̇)
```

### 4. Cart-Pendulum Linearization

**State:** X = [x, θ, ẋ, θ̇]ᵀ

**Linearized System:**
```
Ẋ = AX + BU
Y = CX + DU
```

**A Matrix Structure:**
```
A = [  0     0     1     0   ]  ← Kinematic layer
    [  0     0     0     1   ]
    [  0   A[2,1]  A[2,2] A[2,3]]  ← Dynamic layer
    [  0   A[3,1]  A[3,2] A[3,3]]
```

### 5. Muscle Dynamics

**First-Order Actuator:**
```
Ḟ = (−F + u) / τ
τ = 0.03 s  (time constant)
Rise time ≈ 3τ = 0.09 s
```

**Transfer Function:**
```
F(s)/U(s) = 1/(τs + 1)  (low-pass filter)
```

---

## Data Flow Architecture

### Control Loop (PyDrake)

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LOOP                           │
└─────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  Reference   │  (desired_ee_pos, ee_vel_ref, ee_acc_ref)
  │  Generator   │
  └──────┬───────┘
         │
         ↓
  ┌──────────────────────┐
  │  Controller          │  ComputedTorqueController or SEACableController
  │  (LeafSystem)        │
  │  - IK solve          │
  │  - Feedforward       │
  │  - PD feedback       │
  │  - Inverse dynamics  │
  └──────┬───────────────┘
         │ τ_desired [2]
         ↓
  ┌──────────────────────┐
  │  SEA Actuator        │  SEACableActuator (optional)
  │  (LeafSystem)        │
  │  - Motor dynamics    │
  │  - Spring force      │
  │  - Cable tension     │
  └──────┬───────────────┘
         │ τ_actual [2]
         ↓
  ┌──────────────────────┐
  │  MultibodyPlant      │  Drake plant with manipulator
  │  (LeafSystem)        │
  │  - Forward dynamics  │
  │  - State integration │
  └──────┬───────────────┘
         │ plant_state [n]
         ↓
  ┌──────────────────────┐
  │  State Feedback      │  Back to controller
  └──────────────────────┘
```

### Isaac Sim Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    ISAAC SIM LOOP                            │
└─────────────────────────────────────────────────────────────┘

  Isaac Sim World
       │
       ├── Scene (USD stage)
       │    ├── Robot (URDF/USD)
       │    ├── Environment
       │    └── Sensors
       │
       ├── Physics Engine (PhysX)
       │    └── Step simulation
       │
       └── Python API
            ├── SimulationApp
            ├── World
            ├── Scene
            └── Controllers
```

---

## Configuration Schema

### ManipulatorConfig

```python
@dataclass
class ManipulatorConfig:
    name: str = "cup_manipulator"
    urdf_path: Path = Path("model/.../robot.urdf")
    package_map: Dict[str, str] = {}  # ROS package paths
    
    # Joint configurations
    joint_configs: Dict[str, JointConfig] = field(default_factory=dict)
    
    # Initial conditions
    initial_positions: Dict[str, float] = field(default_factory=dict)
    initial_velocities: Dict[str, float] = field(default_factory=dict)
    
    # Physical parameters
    mass_properties: Dict[str, MassProperty] = field(default_factory=dict)
    
    def get_urdf_path(self) -> Path:
        """Resolve URDF path, checking existence."""
```

### JointConfig

```python
@dataclass
class JointConfig:
    name: str
    damping: float = 0.0        # Viscous damping [N·m·s/rad]
    friction: float = 0.0       # Coulomb friction [N·m]
    position: float = 0.0       # Initial position [rad]
    velocity: float = 0.0       # Initial velocity [rad/s]
    effort_limit: float = inf   # Max torque [N·m]
    velocity_limit: float = inf # Max velocity [rad/s]
```

### CartPendulumPhysicsConfig

```python
@dataclass
class CartPendulumPhysicsConfig:
    mass_cart: float = 1.0          # [kg]
    mass_pendulum: float = 0.5      # [kg]
    length_pendulum: float = 0.5    # [m]
    damping_cart: float = 0.1       # [N·s/m]
    damping_pendulum: float = 0.2   # [N·m·s/rad]
```

### Simulation Config (JSON)

```json
{
    "timestamp": "20260408_231148",
    "robot": {
        "name": "cup_manipulator",
        "urdf_path": "model/.../robot.urdf",
        "joint_configs": {...}
    },
    "controller": {
        "type": "computed_torque",
        "Kp": 400.0,
        "Kd": 40.0,
        "tau_max": 10.0
    },
    "simulation": {
        "dt": 0.002,
        "duration": 10.0,
        "realtime": true
    }
}
```

---

## Simulation Scripts

### Cup Manipulator Scripts

| Script | Description |
|--------|-------------|
| `script_cup_manipulator_pydrake_C3M.py` | C3M control with PyDrake |
| `script_cup_manipulator_c3m_pydrake.py` | C3M controller variant |
| `script_cup_manipulator_lqr.py` | LQR control |
| `script_cup_manipulator_linearized.py` | Linearized model analysis |
| `script_cup_manipulator_pendulam_lqr_min_effort_2d.py` | Minimum effort LQR |
| `script_cup_manipulator_pendulam_computed_torque_isaac_sim.py` | CT control in Isaac Sim |
| `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py` | Tendon + exo with PyDrake |
| `script_cup_manipulator_pendulam_tendon_with_spring_sea_pydrake.py` | SEA with PyDrake |
| `script_cup_manipulator_pendulam_tendon_with_spring_sea_isaac_sim.py` | SEA with Isaac Sim |
| `script_cup_manipulator_pendulam_tendon_with_spring_only_viz_pydrake.py` | Spring-only visualization |
| `script_cup_manipulator_pendulam_tendon_scene_viz_isaac_sim.py` | Scene visualization |
| `script_cup_manipulator_welded_pendulum_various_control.py` | Welded pendulum variants |

### Cart Pendulum Scripts

| Script | Description |
|--------|-------------|
| `script_cart_pendulum_lqr_min_effort_1d.py` | 1D LQR control |
| `script_cart_pendulum_manipulator_basic_run.py` | Basic manipulator run |
| `script_cart_pendulum_manipulator_controller_isaacsim.py` | Isaac Sim controller |
| `script_cart_pendulum_manipulator_controller_pydrake.py` | PyDrake controller |
| `script_cart_pendulum_muscle_dynamics.py` | Muscle dynamics simulation |
| `script_cart_pendulum_various_control_1d.py` | Various 1D controllers |

### Parameter Sweep Scripts

| Script | Description |
|--------|-------------|
| `sweep_exo_ab.py` | Exo parameter sweep (base) |
| `sweep_exo_ab_v2.py` | Version 2 |
| `sweep_exo_ab_v3.py` | Version 3 |
| `sweep_exo_ab_v4.py` | Version 4 |

### Test Scripts

| Script | Description |
|--------|-------------|
| `test_cup_manipulator_pendulam_with_spring_damper_pydrake_v2.py` | Spring-damper test |
| `test_cup_manipulator_tendon_multi_instance_isaac_sim.py` | Multi-instance test |
| `demo_pendulum_jt_properties_comparison.py` | Joint properties demo |
| `demo_simple_pendulum_lqr_from_scratch.py` | LQR from scratch demo |
| `demo_simple_plant_simulation.py` | Simple plant demo |

---

## Integration Patterns

### PyDrake Diagram Construction

```python
from pydrake.all import DiagramBuilder, Simulator, MultibodyPlant

builder = DiagramBuilder()

# 1. Create plant
plant = MultibodyPlant(time_step=dt)
parser = Parser(plant)
robot.load_urdf_to_plant(plant, parser)
robot.add_joint_actuators(plant)
plant.Finalize()

# 2. Create controller
controller = ComputedTorqueController(
    plant=plant,
    manipulator=robot,
    Kp=400.0, Kd=40.0, tau_max=10.0
)

# 3. Wire ports
builder.Connect(
    controller.GetOutputPort("actuation"),
    plant.GetInputPort("actuation")
)
builder.Connect(
    plant.GetStateOutputPort(),
    controller.GetInputPort("plant_state")
)

# 4. Build diagram
diagram = builder.Build()
context = diagram.CreateDefaultContext()

# 5. Simulate
simulator = Simulator(diagram)
simulator.Initialize()
simulator.AdvanceTo(duration)
```

### Isaac Sim Script Pattern

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World

world = World()
world.scene.add_default_ground_plane()

# Add robot
robot = world.scene.add(
    Articulation(
        prim_path="/World/Robot",
        usd_path="model/robot.usd"
    )
)

world.reset()

for i in range(1000):
    # Get state
    positions = robot.get_joint_positions()
    velocities = robot.get_joint_velocities()
    
    # Compute control
    torques = controller.compute(positions, velocities)
    
    # Apply control
    robot.set_joint_efforts(torques)
    
    # Step simulation
    world.step(render=True)

simulation_app.close()
```

### SEA + Controller Composition

```python
# Option 1: Monolithic controller (backward compat)
controller = SEACableController(
    plant=plant,
    manipulator=robot,
    k_s=200.0, b_c=2.0, omega_m=30.0,
    Kp=10000.0, Kd=400.0
)

# Option 2: Modular (recommended)
controller = ComputedTorqueController(
    plant=plant,
    manipulator=robot,
    Kp=400.0, Kd=40.0
)
actuator = SEACableActuator(
    plant=plant,
    manipulator=robot,
    k_s=200.0, b_c=2.0,
    motor_mode=MotorMode.TORQUE,
    motor_cfg=MotorModelConfig(...)
)

# Wire: controller → actuator → plant
builder.Connect(
    controller.GetOutputPort("actuation"),
    actuator.GetInputPort("tau_desired")
)
builder.Connect(
    actuator.GetOutputPort("actuation"),
    plant.GetInputPort("actuation")
)
```

---

## Testing & Validation

### Test Suite Structure

```
tests/
├── test_linearized_muscle_dynamics.py   # Matrix dimensions, linearization
├── test_linearized_control.py           # Stability analysis
└── verify_linearized_matrices.py        # Physical interpretation

tests2/
├── test_cup_manipulator_*.py            # Manipulator tests
└── test_*.py                            # Additional tests
```

### Validation Checks

1. **Matrix Dimensions**: A(4,4), B(4,1), C(4,4), D(4,1)
2. **Physical Structure**: Gravity, damping, coupling present
3. **Controllability**: Full rank B matrix
4. **Observability**: Full rank C matrix
5. **Stability**: Stabilizable with feedback
6. **Torque Limits**: Saturation handling verified

---

## Quick Reference

### Common Constants

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| `PULLEY_RADIUS` | 0.04775 | m | Joint 2 pulley radius |
| `EE_XYZ_LINK2` | [1.2515, 0, 0.15] | m | EE offset from link2 |
| `k_s` | 200-300 | N/m | Cable spring stiffness |
| `b_c` | 2.0 | N·s/m | Cable damping |
| `tau_max` | 10.0 | N·m | Torque saturation |
| `dt` | 0.002 | s | Simulation timestep |
| `Kp` | 400 | s⁻² | Position gain |
| `Kd` | 40 | s⁻¹ | Velocity gain |

### Joint Names (Default)

| Joint | Name | Description |
|-------|------|-------------|
| JT1 | `link1_base` | Base to link1 (q1) |
| JT2 | `link2_link1` | Link1 to link2 (q2) |

### File Locations

| Resource | Path |
|----------|------|
| Robot URDFs | `model/*/` |
| Configs | `configs/` |
| Documentation | `docs/` |
| MATLAB scripts | `contraction-theory/` |
| Test scripts | `tests/`, `tests2/` |

---

## Environment Setup

### Conda Environment

```bash
conda create -n env_isaacsim python=3.11 -y
conda activate env_isaacsim
pip install isaacsim drake pyyaml
```

### Isaac Sim Local Build

```bash
source ~/Documents/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh
# Or use alias: isaacsim-env
```

### Environment Variables

```bash
export ISAAC_SIM_PATH="$HOME/isaacsim"
```

---

*End of Context Cache*