"""
Cup Manipulator - Drake Controller Architecture

═══════════════════════════════════════════════════════════════════════════════
TWO-SYSTEM ARCHITECTURE EXPLANATION
═══════════════════════════════════════════════════════════════════════════════

YES - We ARE using TWO separate systems (Plant + Controller Model):

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DRAKE DIAGRAM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  state[q,v]   ┌──────────────────┐                  │
│  │                  │───────────────>│                  │                  │
│  │  MultibodyPlant  │                │   PDController   │                  │
│  │   (Physics)      │<───────────────│   (Control Law)  │                  │
│  │                  │  torque[τ]     │                  │                  │
│  └──────────────────┘                └──────────────────┘                  │
│         │                                                                   │
│         │ geometry                                                          │
│         v                                                                   │
│  ┌──────────────────┐                                                      │
│  │   SceneGraph     │───────> MeshcatVisualizer                           │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

SYSTEM 1: MultibodyPlant (Physics Model)
─────────────────────────────────────────
• Type: Drake's built-in LeafSystem
• Role: Simulates robot physics and dynamics
• Physics Engine: Solves equations of motion M(q)q̈ + C(q,v)v = τ_g(q) + τ_applied
• Inputs: Control torques τ (from controller)
• Outputs: State [q, v] = positions and velocities
• Contains: Robot URDF, joints, links, collision geometry, inertias
• Computes: Forward dynamics, gravity, Coriolis forces, constraints

SYSTEM 2: PDController (Control Model - Custom LeafSystem)
───────────────────────────────────────────────────────────
• Type: Custom LeafSystem we created
• Role: Computes control torques based on desired trajectory
• Control Law: τ = Kp*(q_desired - q) + Kd*(v_desired - v)
• Inputs: State [q, v] from plant
• Outputs: Control torques τ to apply to plant
• Contains: Gains (Kp, Kd), trajectory generation, control logic
• Does NOT do physics - only computes what torques to apply

KEY DIFFERENCES FROM ORIGINAL SCRIPT:
═════════════════════════════════════════

Original script (script_cup_manipulator_pydrake.py):
────────────────────────────────────────────────────
• Direct control loop approach
• Python function computes torques inline in simulation loop
• Controller is just Python code, NOT a Drake system
• Manual torque application: plant.get_actuation_input_port().FixValue(context, τ)
• Simple but harder to extend to advanced controllers

This script (script_cup_manipulator_controller_drake.py):
─────────────────────────────────────────────────────────
• Diagram-based architecture
• Controller is a Drake LeafSystem with ports
• Automatic data flow through ports
• No manual torque setting - ports handle it
• Modular: Easy to swap PD → Inverse Dynamics → Computed Torque
• Professional robotics control architecture

ADVANTAGES OF TWO-SYSTEM APPROACH:
═══════════════════════════════════════

1. Modularity: Swap controllers without touching plant code
2. Data Flow: Ports make data dependencies explicit
3. Extensibility: Easy to add trajectory planners, observers, filters
4. Reusability: Same plant with different controllers
5. Debugging: Each system can be tested independently
6. Professional: Matches industry robotics software patterns

WHY SEPARATE PLANT AND CONTROLLER?
═══════════════════════════════════════

Physical Analogy:
• Plant = The actual robot hardware (motors, links, physics)
• Controller = The computer/brain sending commands to the robot

Software Analogy:
• Plant = Physics simulator (knows HOW robot moves)
• Controller = Decision maker (knows WHAT robot should do)

This separation is fundamental in robotics:
• Plant can't change (fixed hardware/physics)
• Controller can be upgraded (new algorithms)

FUTURE EXTENSIONS:
══════════════════════════════════════════════════════════════════════════════

Easy to add because of two-system architecture:

1. Inverse Dynamics Controller:
   • Controller accesses plant.CalcInverseDynamics()
   • Compensates for gravity and Coriolis forces
   • τ = M(q)a + C(q,v)v + τ_g(q) + Kp*e + Kd*ė

2. Computed Torque Controller:
   • Full feedback linearization
   • τ = M(q)a_desired + C(q,v)v + τ_g(q) + K_p(q_d - q) + K_d(v_d - v)

3. Trajectory Planning System:
   • Add TrajectorySource as another system
   • Connects to controller's desired state input
   • Plant → Controller → Plant, TrajectorySource → Controller

4. State Observer/Estimator:
   • Add Kalman filter as another system
   • Plant.state → Observer → Controller
   • Handles noisy measurements

═══════════════════════════════════════════════════════════════════════════════
"""

from statistics import mode
import numpy as np
import argparse
import os
import time
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    CoulombFriction,
    RevoluteJoint,
    PrismaticJoint,
    SpatialInertia,
    UnitInertia,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    
    # Geometry
    Cylinder,
    Sphere,
    Rgba,
    
    # Controllers
    InverseDynamicsController,
    
    # Mathematical utilities
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,
    
    # Frames
    FixedOffsetFrame,
)

# Custom robot types
from robot_types import (
    ManipulatorConfig,
    SimulationConfig,
    VisualizationConfig,
    PendulumConfig,
    create_cup_manipulator_config,
    create_pendulum_config,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Drake Diagram-based controller architecture')
parser.add_argument('--mode', type=str, choices=['pd', 'inverse-dynamics', 'computed-torque', 'scene-viz'],
                    default='computed-torque', help='Controller type (scene-viz = static visualization only)')
parser.add_argument('--visualize', type=bool, default=True, help='Enable visualization')
parser.add_argument('--plot_frames', type=bool, default=True, help='Plot coordinate frames')
args, _ = parser.parse_known_args()

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Cup Manipulator Configuration ---
CUP_MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.0, 0.0),
    damping=(0.0, 0.0),
    stiffness=(0.0, 0.0),
    friction=(0.05, 0.05),
)

# --- Pendulum Configuration ---
PENDULUM_ENABLED = True
PENDULUM_CONFIG = create_pendulum_config(
    mass=0.5,
    length=0.2,
    radius=0.05,
    damping=0.1,
    attachment_point=(-1.2545, 0.0, -1.188125),
    initial_pitch=0.0,
    initial_roll=0.0,
    name="pendulum"
)

# --- Visualization Configuration ---
VISUALIZATION_CONFIG = VisualizationConfig(
    enabled=args.visualize,
    plot_frames=args.plot_frames,
    interactive=True,
    realtime_rate=0.5,
    update_every_step=True,
    print_interval=0.25,  # Terminal output frequency (seconds)
    logging_interval=0.02,  # Data logging frequency for plots (seconds) - 50 Hz for smooth curves
    show_frames=False,
    show_contact_forces=True,
    show_hydroelastic=True,
)

# --- Simulation Configuration ---
SIMULATION_CONFIG = SimulationConfig(
    mode=args.mode,
    timestep=0.001,  # 1 kHz simulation
    simulation_time=8.0,
    gravity=(0.0, 0.0, -9.81),
    visualization=VISUALIZATION_CONFIG,
)

# --- Controller Configuration ---
CONTROLLER_MODE = args.mode
MANIPULATOR_MOTION_DURATION = 3.0  # seconds
JOINT_MOTION_AMPLITUDE = [np.pi/3, np.pi/2.5]  # radians
JOINT_MOTION_FREQUENCY = [1.2, 1.0]  # Hz - increased for smoother motion


# ============================================================================
# ROBOT BASE CLASS (ABSTRACT)
# ============================================================================

class RobotBase(ABC):
    """
    Abstract Base Class for Robots using Drake
    
    DESIGN PATTERN: Template Method Pattern
    Provides common interface for all robots
    """
    
    def __init__(self, config: ManipulatorConfig, name: Optional[str] = None):
        """Initialize robot with configuration."""
        self.config = config
        self.name = name or config.name
        self.model_instance: Optional[int] = None
        self.dof_names: List[str] = []
    
    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        """
        Load URDF to plant using Drake's URDF parser.
        
        Args:
            plant: Drake MultibodyPlant
            parser: Drake URDF parser
            
        Returns:
            model_instance: Drake's model instance ID
        """
        urdf_path = str(self.config.get_urdf_path())
        print(f"\nLoading robot from URDF: {urdf_path}")
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Set package map for mesh loading
        for package_name, package_path in self.config.package_map.items():
            parser.package_map().Add(package_name, package_path)
        
        # AddModels returns a list of model instances
        model_instances = parser.AddModels(urdf_path)
        if not model_instances:
            raise RuntimeError(f"Failed to load URDF from {urdf_path}")
        
        print(colored(f"✓ Loaded {len(model_instances)} model instance(s) from URDF", 'green'))
        for idx, instance in enumerate(model_instances):
            print(colored(f"  [{idx}] Model instance: {instance}", 'cyan'))
        
        model_instance = model_instances[0]
        self.model_instance = model_instance
        print(colored(f"✓ Robot '{self.name}' using model instance: {model_instance}", 'green'))
        return model_instance
    
    def initialize_state(self, plant: MultibodyPlant):
        """Initialize robot state after plant is finalized."""
        if not self.model_instance:
            raise RuntimeError("Model not loaded - call load_urdf_to_plant first")
        
        # Get DOF names (only actuated joints)
        self.dof_names = []
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0 and joint.num_positions() > 0:
                self.dof_names.append(joint.name())
        
        num_dof = len(self.dof_names)
        print(colored(f"✓ Robot '{self.name}' initialized with {num_dof} DOFs", 'green', attrs=['bold']))
        print(colored(f"  DOF names: {self.dof_names}", 'cyan'))
    
    def set_joint_properties(self, plant: MultibodyPlant):
        """Set joint properties (damping, friction) BEFORE plant is finalized."""
        print(colored(f"\nSetting joint properties for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint.num_velocities() > 0 and joint_name in self.config.joint_configs:
                config = self.config.joint_configs[joint_name]
                
                if hasattr(joint, 'set_default_damping_vector') and config.damping > 0:
                    joint.set_default_damping_vector([config.damping])
                    print(colored(f"  ✓ {joint_name}: damping={config.damping}", 'cyan'))
                else:
                    print(colored(f"  ✓ {joint_name}: damping=0.0 (default)", 'cyan'))
        print(colored(f"✓ Joint properties configured", 'green'))
    
    def set_initial_positions(self, plant: MultibodyPlant, context):
        """Set initial joint positions from configuration."""
        print(colored(f"\nSetting initial positions for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint_name in self.config.joint_configs:
                position = self.config.joint_configs[joint_name].position
                
                if isinstance(joint, RevoluteJoint):
                    joint.set_angle(context, position)
                    print(colored(f"  ✓ {joint_name}: {np.rad2deg(position):.2f}° ({position:.4f} rad)", 'cyan'))
                elif isinstance(joint, PrismaticJoint):
                    joint.set_translation(context, position)
                    print(colored(f"  ✓ {joint_name}: {position:.4f} m", 'cyan'))
        print(colored(f"✓ Initial positions set", 'green'))


# ============================================================================
# CUP MANIPULATOR CLASS
# ============================================================================

class CupManipulator(RobotBase):
    """
    Cup Manipulator for Drake with controller integration.
    
    Manages:
    - URDF loading and joint configuration
    - State queries (positions, velocities)
    - End-effector kinematics
    """
    
    def __init__(self, config: ManipulatorConfig):
        super().__init__(config)
    
    def get_joint_positions(self, plant: MultibodyPlant, context) -> Dict[str, float]:
        """Get current joint positions as a dictionary."""
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        
        # Also get pendulum joints if they exist
        if PENDULUM_ENABLED:
            try:
                pitch_joint = plant.GetJointByName("pendulum_pitch", self.model_instance)
                positions['pendulum_pitch'] = pitch_joint.get_angle(context)
            except:
                pass
            try:
                roll_joint = plant.GetJointByName("pendulum_roll", self.model_instance)
                positions['pendulum_roll'] = roll_joint.get_angle(context)
            except:
                pass
        
        return positions
    
    def get_joint_velocities(self, plant: MultibodyPlant, context) -> Dict[str, float]:
        """Get current joint velocities as a dictionary."""
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities
    
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get end effector (cup) position in world frame."""
        try:
            cup_body = plant.GetBodyByName("link2", self.model_instance)
            world_frame = plant.world_frame()
            cup_frame = cup_body.body_frame()
            
            X_WC = plant.CalcRelativeTransform(context, world_frame, cup_frame)
            return X_WC.translation()
        except Exception as e:
            print(f"Warning: Could not get end effector position: {e}")
            return np.array([0.0, 0.0, 0.0])


# ============================================================================
# PD CONTROLLER LEAFSYSTEM (SYSTEM 2 - Control Model)
# ============================================================================

class PDController(LeafSystem):
    """
    PD Controller as a Drake LeafSystem - THIS IS SYSTEM 2 (Control Model)
    
    ═══════════════════════════════════════════════════════════════════════
    SYSTEM 2: CONTROLLER (separate from physics plant)
    ═══════════════════════════════════════════════════════════════════════
    
    Role: Compute control torques based on desired trajectory and feedback
    
    This is NOT the plant - it's a separate system that:
    1. Receives state [q, q_dot] from MultibodyPlant (SYSTEM 1)
    2. Computes desired trajectory (sinusoidal motion)
    3. Applies PD control law: τ = Kp*(q_d - q) + Kd*(q_dot_d - q_dot)
    4. Sends torques τ back to MultibodyPlant
    
    The plant and controller communicate via Drake ports, not Python code.
    This is the key difference from the original script.
    
    ═══════════════════════════════════════════════════════════════════════
    
    Inputs (Port 0):
        - state: [q, q_dot] joint positions and velocities (4-dim for 2 actuated + 2 passive joints)
    
    Outputs (Port 0):
        - torque: control torques for actuated joints (2-dim for link1_base, link2_link1)
    
    This design allows easy extension to:
    - Inverse dynamics: add plant reference for gravity/Coriolis compensation
    - Computed torque: add desired acceleration input port
    - Feedforward terms: add trajectory ports
    """
    
    def __init__(self, plant: MultibodyPlant, model_instance, 
                 Kp: np.ndarray, Kd: np.ndarray,
                 motion_duration: float = 3.0):
        """
        Initialize PD controller.
        
        Args:
            plant: MultibodyPlant reference (for future inverse dynamics)
            model_instance: Model instance ID
            Kp: Proportional gains [2] for actuated joints
            Kd: Derivative gains [2] for actuated joints
            motion_duration: Duration of motion phase (seconds)
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.model_instance = model_instance
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Kp_hold = self.Kp * 10.0  # Higher gains for settling phase
        self.Kd_hold = self.Kd * 10.0
        self.motion_duration = motion_duration
        
        print(colored(f"\n--- PDController Configuration ---", 'yellow', attrs=['bold']))
        print(colored(f"  Kp (motion): {self.Kp}", 'cyan'))
        print(colored(f"  Kd (motion): {self.Kd}", 'cyan'))
        print(colored(f"  Kp (hold): {self.Kp_hold}", 'cyan'))
        print(colored(f"  Kd (hold): {self.Kd_hold}", 'cyan'))
        print(colored(f"  Motion duration: {motion_duration} s", 'cyan'))
        
        # Get number of actuated joints (link1_base, link2_link1)
        self.num_actuated = 2
        
        # Get total DOF (actuated + passive pendulum joints)
        self.num_positions = plant.num_positions()
        self.num_velocities = plant.num_velocities()
        
        # Input port: full state [q, v]
        self.DeclareVectorInputPort(
            "estimated_state",
            BasicVector(self.num_positions + self.num_velocities)
        )
        
        # Output port: actuator torques (only for actuated joints)
        self.DeclareVectorOutputPort(
            "control_torque",
            BasicVector(self.num_actuated),
            self.CalcControlTorque
        )
        
        # Store stop position for settling phase
        self.stop_position = np.zeros(self.num_actuated)
        self.motion_stopped = False
        
        print(colored(f"✓ PDController initialized:", 'green'))
        print(colored(f"  - Actuated joints: {self.num_actuated}", 'cyan'))
        print(colored(f"  - Total DOF: {self.num_positions}", 'cyan'))
        print(colored(f"  - Kp: {self.Kp}", 'cyan'))
        print(colored(f"  - Kd: {self.Kd}", 'cyan'))
    
    def CalcControlTorque(self, context, output):
        """
        Compute PD control torques.
        
        This is called automatically by Drake at each timestep.
        """
        # Get current state from input port
        state = self.get_input_port(0).Eval(context)
        q = state[:self.num_positions]  # All joint positions
        q_dot = state[self.num_positions:]  # All joint velocities
        
        # Extract actuated joint states (first 2 joints: link1_base, link2_link1)
        q_actuated = q[:self.num_actuated]
        q_dot_actuated = q_dot[:self.num_actuated]
        
        # Get current time
        t = context.get_time()
        
        # Check if we should stop manipulator motion
        if t >= self.motion_duration and not self.motion_stopped:
            self.stop_position = q_actuated.copy()
            self.motion_stopped = True
            print(f"\n{'='*70}")
            print(f"t={t:.2f}s: CONTROLLER SWITCHED - Ball settling phase begins")
            print(f"  Holding position: {np.rad2deg(self.stop_position)}")
            print(f"  PD gains increased: Kp={self.Kp_hold}, Kd={self.Kd_hold}")
            print(f"{'='*70}\n")
        
        # Calculate desired trajectory
        if t < self.motion_duration:
            # Moving phase: sinusoidal motion
            q_desired = np.array([
                JOINT_MOTION_AMPLITUDE[0] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                JOINT_MOTION_AMPLITUDE[1] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
            ])
            q_dot_desired = np.array([
                JOINT_MOTION_AMPLITUDE[0] * 2 * np.pi * JOINT_MOTION_FREQUENCY[0] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                JOINT_MOTION_AMPLITUDE[1] * 2 * np.pi * JOINT_MOTION_FREQUENCY[1] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
            ])
            Kp_current = self.Kp
            Kd_current = self.Kd
        else:
            # Settling phase: hold fixed position
            q_desired = self.stop_position
            q_dot_desired = np.zeros(self.num_actuated)
            Kp_current = self.Kp_hold
            Kd_current = self.Kd_hold
        
        # PD control law: τ = Kp*(q_d - q) + Kd*(q_dot_d - q_dot)
        torque = Kp_current * (q_desired - q_actuated) + Kd_current * (q_dot_desired - q_dot_actuated)
        
        # Set output
        output.SetFromVector(torque)


# ============================================================================
# COMPUTED TORQUE / INVERSE DYNAMICS CONTROLLER (SYSTEM 2 Alternative)
# ============================================================================

class ComputedTorqueController(LeafSystem):
    """
    Computed Torque Controller with Inverse Dynamics Compensation.
    
    ═══════════════════════════════════════════════════════════════════════
    ADVANCED CONTROLLER - Uses plant dynamics for feedforward compensation
    ═══════════════════════════════════════════════════════════════════════
    
    Control Law:
        τ = M(q) · [q_ddot_d + Kp·e + Kd·ė] + C(q,q_dot) + g(q)
    
    Where:
        - M(q): Mass/inertia matrix
        - q_ddot_d: Desired acceleration from trajectory
        - e = q_d - q: Position error
        - ė = q_dot_d - q_dot: Velocity error
        - C(q,q_dot): Coriolis and centrifugal forces
        - g(q): Gravity forces
        - Kp, Kd: Feedback gains (much smaller than PD controller)
    
    Key Insight:
        The feedback term (Kp·e + Kd·ė) is ADDED to the desired acceleration,
        then the TOTAL commanded acceleration is passed through inverse dynamics.
        This ensures the mass matrix properly scales the feedback torques.
    
    Benefits over PD:
        - Perfect tracking in theory (if model is accurate)
        - No steady-state error from gravity/dynamics
        - Faster response with smaller gains
        - More energy efficient
    
    ═══════════════════════════════════════════════════════════════════════
    
    Inputs (Port 0):
        - state: [q, q_dot] joint positions and velocities
    
    Outputs (Port 0):
        - torque: control torques for actuated joints
    """
    
    def __init__(self, plant: MultibodyPlant, model: MultibodyPlant, model_instance,
                 Kp: np.ndarray, Kd: np.ndarray,
                 motion_duration: float = 3.0,
                 control_mode: str = "full"
                 ):
        """
        Initialize Computed Torque controller with model-plant separation.
        
        Args:
            plant: MultibodyPlant reference (the "real" system - for state only)
            model: MultibodyPlant reference (controller's internal model - for dynamics)
            model_instance: Model instance ID
            Kp: Proportional gains [2] for actuated joints
            Kd: Derivative gains [2] for actuated joints
            motion_duration: Duration of motion phase (seconds)
            control_mode: "truncate" or "full" for underactuation handling
            
        IMPORTANT: Plant vs Model Separation
        ────────────────────────────────────
        - plant: The "real" system (simulation or actual robot)
                 Used ONLY for reading state via input ports
                 Can have different parameters than model
        
        - model: Controller's internal dynamics model
                 Used for CalcInverseDynamics calculations
                 Should represent nominal/estimated parameters
                 
        This separation enables:
        - Sim-to-real transfer (swap plant, keep model)
        - Robustness testing (plant ≠ model parameters)
        - Controller doesn't need to know actual hardware details
        """
        LeafSystem.__init__(self)
        
        self.plant = plant    # Real system (for state monitoring)
        self.model = model    # Controller's internal model (for control calculations)
        self.model_instance = model_instance
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Kp_hold = self.Kp * 10.0
        self.Kd_hold = self.Kd * 10.0
        self.motion_duration = motion_duration
        self.control_mode = control_mode
        
        print(colored(f"\n--- ComputedTorqueController Configuration ---", 'yellow', attrs=['bold']))
        print(colored(f"  Control Law: τ = M(q)·[q_ddot_d + Kp·e + Kd·ė] + C(q,q_dot) + g(q)", 'cyan'))
        print(colored(f"  Kp (motion): {self.Kp}", 'cyan'))
        print(colored(f"  Kd (motion): {self.Kd}", 'cyan'))
        print(colored(f"  Kp (hold): {self.Kp_hold}", 'cyan'))
        print(colored(f"  Kd (hold): {self.Kd_hold}", 'cyan'))
        print(colored(f"  Motion duration: {motion_duration} s", 'cyan'))
        print(colored(f"  Model-Plant Separation: ENABLED", 'yellow', attrs=['bold']))
        print(colored(f"    Plant: Used for state observation only", 'cyan'))
        print(colored(f"    Model: Used for inverse dynamics calculations", 'cyan'))
        
        # Get dimensions
        self.num_actuated = 2  # link1_base, link2_link1
        self.num_positions = plant.num_positions()
        self.num_velocities = plant.num_velocities()
        
        # Create a context for the MODEL (needed for dynamics calculations)
        # This is the key separation: we compute dynamics using the model, not the plant
        self.model_context = model.CreateDefaultContext()
        
        # Input port: full state [q, v]
        self.DeclareVectorInputPort(
            "estimated_state",
            BasicVector(self.num_positions + self.num_velocities)
        )
        
        # Output port: actuator torques
        self.DeclareVectorOutputPort(
            "control_torque",
            BasicVector(self.num_actuated),
            self.CalcControlTorque
        )
        
        # Store stop position for settling phase
        self.stop_position = np.zeros(self.num_actuated)
        self.motion_stopped = False
        
        print(colored(f"✓ ComputedTorqueController initialized:", 'green'))
        print(colored(f"  - Actuated joints: {self.num_actuated}", 'cyan'))
        print(colored(f"  - Total DOF: {self.num_positions}", 'cyan'))
        print(colored(f"  - Using inverse dynamics from CONTROLLER MODEL", 'cyan'))
        print(colored(f"  - Feedforward: Gravity + Coriolis compensation", 'cyan'))
        print(colored(f"  - State observation: from PLANT (via input port)", 'cyan'))
    
    def CalcControlTorque(self, context, output):
        """
        Compute Computed Torque control with inverse dynamics.
        
        This is called automatically by Drake at each timestep.
        """
        # Get current state from input port
        state = self.get_input_port(0).Eval(context)
        q = state[:self.num_positions]
        q_dot = state[self.num_positions:]
        
        # Extract actuated joint states
        q_actuated = q[:self.num_actuated]
        q_dot_actuated = q_dot[:self.num_actuated]
        
        # Get current time
        t = context.get_time()
        
        # Check if we should stop manipulator motion
        if t >= self.motion_duration and not self.motion_stopped:
            self.stop_position = q_actuated.copy()
            self.motion_stopped = True
            print(f"\n{'='*70}")
            print(f"t={t:.2f}s: CONTROLLER SWITCHED - Ball settling phase begins")
            print(f"  Holding position: {np.rad2deg(self.stop_position)}")
            print(f"  PD gains increased: Kp={self.Kp_hold}, Kd={self.Kd_hold}")
            print(f"  Inverse dynamics: Active (gravity + Coriolis compensation)")
            print(f"{'='*70}\n")
        
        # Calculate desired trajectory
        if t < self.motion_duration:
            # Moving phase: sinusoidal motion
            q_desired = np.array([
                JOINT_MOTION_AMPLITUDE[0] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                JOINT_MOTION_AMPLITUDE[1] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
            ])
            q_dot_desired = np.array([
                JOINT_MOTION_AMPLITUDE[0] * 2 * np.pi * JOINT_MOTION_FREQUENCY[0] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                JOINT_MOTION_AMPLITUDE[1] * 2 * np.pi * JOINT_MOTION_FREQUENCY[1] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
            ])
            q_ddot_desired = np.array([
                -JOINT_MOTION_AMPLITUDE[0] * (2 * np.pi * JOINT_MOTION_FREQUENCY[0])**2 * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                -JOINT_MOTION_AMPLITUDE[1] * (2 * np.pi * JOINT_MOTION_FREQUENCY[1])**2 * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
            ])
            Kp_current = self.Kp
            Kd_current = self.Kd
        else:
            # Settling phase: hold fixed position
            q_desired = self.stop_position
            q_dot_desired = np.zeros(self.num_actuated)
            q_ddot_desired = np.zeros(self.num_actuated)
            Kp_current = self.Kp_hold
            Kd_current = self.Kd_hold
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTED TORQUE CONTROL LAW (CORRECT IMPLEMENTATION)
        # ═══════════════════════════════════════════════════════════════════
        # Correct form: τ = M(q) · [q_ddot_d + Kp·e + Kd·ė] + C(q,q_dot) + g(q)
        #
        # KEY: Use MODEL for dynamics, not plant!
        # - State q, q_dot: observed from PLANT (real system)
        # - Dynamics M, C, g: computed from MODEL (controller's belief)
        # ═══════════════════════════════════════════════════════════════════
        
        # Update MODEL context with current state (from plant observation)
        self.model.SetPositions(self.model_context, q)
        # Update MODEL context with current state (from plant observation)
        self.model.SetPositions(self.model_context, q)
        self.model.SetVelocities(self.model_context, q_dot)
        
        # Compute tracking errors
        e = q_desired - q_actuated
        e_dot = q_dot_desired - q_dot_actuated
        
        # Compute COMMANDED acceleration (includes feedback)
        # q_ddot_cmd = q_ddot_d + Kp·e + Kd·ė
        q_ddot_commanded = q_ddot_desired + Kp_current * e + Kd_current * e_dot
        
        # Prepare commanded acceleration for full system (including passive joints)
        q_ddot_commanded_full = np.zeros(self.num_velocities)
        q_ddot_commanded_full[:self.num_actuated] = q_ddot_commanded
        
        # Create external forces object (no external forces applied)
        from pydrake.multibody.tree import MultibodyForces
        external_forces = MultibodyForces(self.model)  # Use MODEL, not plant!

        # Apply inverse dynamics using the CONTROLLER'S MODEL
        # This computes: τ = M_model(q)·q_ddot_cmd + C_model(q,q_dot) + g_model(q)
        # which expands to: τ = M_model(q)·[q_ddot_d + Kp·e + Kd·ė] + C_model(q,q_dot) + g_model(q)
        # 
        # If model ≠ plant, feedback will compensate for the mismatch!
        torque_full = self.model.CalcInverseDynamics(
            self.model_context,
            q_ddot_commanded_full,  # Commanded accelerations (includes feedback!)
            external_forces  # External forces (none)
        )  # torque_full is generalized forces for all DOF (length nv) (manipulator + pendulum)

        # Choose control mode
        mode = getattr(self, "control_mode", "truncate")

        if mode == "truncate":
            # Simple assumption: actuator commands correspond to the first m generalized forces
            # u := τ*[0:m]
            u = np.asarray(torque_full[:self.num_actuated]).reshape((-1,))

        elif mode == "full":
            # General case: map desired generalized forces τ* into actuator inputs u via B
            # Solve least-squares: u = argmin ||B u - τ*||^2
            # Use MODEL's actuation matrix (should match plant's structure)
            B = np.asarray(self.model.MakeActuationMatrix())  # shape (nv, nu)

            # Pseudoinverse solution: u = B† · τ* where B† = (B^T B)^{-1} B^T
            # This projects desired forces onto achievable subspace Range(B)
            u = np.linalg.pinv(B) @ np.asarray(torque_full)  # Equivalent to lstsq
            u = np.asarray(u).reshape((-1,))
            # Mq_ddot + b = Bu* \approx τ* = Mq_ddot_cmd + b
            # Mq_ddot = Mq_ddot_cmd + b - b = Mq_ddot_cmd

            # Optional actuator limits (must be sized to nu)
            if hasattr(self, "u_min") and hasattr(self, "u_max"):
                u = np.clip(u, self.u_min, self.u_max)

        else:
            raise ValueError(f"Unknown control_mode: {mode}. Use 'truncate' or 'full'.")

        # Output actuator command u (IMPORTANT: output port size must equal plant.num_actuators()).
        output.SetFromVector(u)
        # - Add Coriolis compensation
        pass


# ============================================================================
# PENDULUM 3D CLASS
# ============================================================================

class Pendulum3D:
    """3D Pendulum with 2-DOF gimbal joints (pitch and roll)."""
    
    def __init__(self, config: PendulumConfig):
        self.config = config
        self.mass = config.mass
        self.length = config.length
        self.radius = config.radius
        self.damping = config.damping
        self.attachment_point = config.attachment_point
        self.name = config.name
        
        self.pivot_frame = None
        self.pitch_parent_frame = None
        self.gimbal1_body = None
        self.pendulum_body = None
        self.pitch_joint = None
        self.roll_joint = None
    
    def attach_to_body(self, plant: MultibodyPlant, parent_body, model_instance):
        """Attach pendulum to parent body."""
        # Create pivot frame
        roll = np.deg2rad(0)
        pitch = np.deg2rad(0)
        yaw = np.deg2rad(0)
        pivot_rotation = RotationMatrix(RollPitchYaw(roll, pitch, yaw))
        
        X_parent_pivot = RigidTransform(pivot_rotation, self.attachment_point)
        self.pivot_frame = plant.AddFrame(
            FixedOffsetFrame(
                name=f"{self.name}_pivot_frame",
                P=parent_body.body_frame(),
                X_PF=X_parent_pivot,
                model_instance=model_instance,
            )
        )
        
        # Create gimbal1 intermediate body (for pitch rotation)
        gimbal1_inertia = SpatialInertia(
            mass=0.01,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-4, 1e-4, 1e-4),
        )
        self.gimbal1_body = plant.AddRigidBody(
            f"{self.name}_gimbal1", model_instance, gimbal1_inertia
        )
        
        # Add pitch joint (Y-axis rotation)
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.name}_pitch",
                frame_on_parent=self.pivot_frame,
                frame_on_child=self.gimbal1_body.body_frame(),
                axis=[0.0, 1.0, 0.0],
                damping=self.damping,
            )
        )
        
        # Create pendulum body with proper inertia
        m = float(self.mass)
        r = float(self.radius)
        L = float(self.length)
        
        I_ball_com = (2.0 / 5.0) * m * (r ** 2)
        I_pivot_x = I_ball_com + m * (L ** 2)
        I_pivot_y = I_ball_com + m * (L ** 2)
        I_pivot_z = I_ball_com
        
        G_ball = UnitInertia(I_pivot_x / m, I_pivot_y / m, I_pivot_z / m)
        pendulum_inertia = SpatialInertia(
            mass=m,
            p_PScm_E=[0.0, 0.0, -L],
            G_SP_E=G_ball,
        )
        self.pendulum_body = plant.AddRigidBody(
            f"{self.name}_ball", model_instance, pendulum_inertia
        )
        
        # Add roll joint (X-axis rotation)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.name}_roll",
                frame_on_parent=self.gimbal1_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.damping,
            )
        )
        
        # Add visual/collision geometry (only if plant has SceneGraph registered)
        if plant.geometry_source_is_registered():
            self._add_geometry(plant, L, r)
        
        print(colored(f"\n✓ 3D Pendulum Attached to {parent_body.name()}:", 'green', attrs=['bold']))
        print(colored(f"  Mass: {m} kg", 'cyan'))
        print(colored(f"  Length: {L} m", 'cyan'))
        print(colored(f"  Radius: {r} m", 'cyan'))
        print(colored(f"  Damping: {self.damping}", 'cyan'))
        print(colored(f"  Attachment point: {self.attachment_point}", 'cyan'))
        print(colored(f"  Joints: {self.name}_pitch (Y-axis), {self.name}_roll (X-axis)", 'cyan'))
    
    def _add_geometry(self, plant, L, r):
        """Add visual and collision geometry."""
        from pydrake.geometry import Cylinder, Sphere
        
        # Rod visual
        plant.RegisterVisualGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L / 2.0]),
            Cylinder(radius=0.001, length=L),
            f"{self.name}_rod_visual",
            [0.6, 0.4, 0.2, 1.0],
        )
        
        # Ball visual
        plant.RegisterVisualGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L]),
            Sphere(r),
            f"{self.name}_ball_visual",
            [0.8, 0.2, 0.2, 1.0],
        )
        
        # Ball collision
        plant.RegisterCollisionGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L]),
            Sphere(r),
            f"{self.name}_ball_collision",
            CoulombFriction(0.3, 0.2),
        )
    
    def set_initial_swing(self, context, pitch_angle: float = 0.0, roll_angle: float = 0.0):
        """Set initial swing angles."""
        if self.pitch_joint:
            self.pitch_joint.set_angle(context, pitch_angle)
        if self.roll_joint:
            self.roll_joint.set_angle(context, roll_angle)


# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
    Scene Manager for Drake simulation with Diagram-based controller.
    
    RESPONSIBILITIES:
    1. Setup: Create MultibodyPlant, add robot, add controller
    2. Build Diagram: Wire ports between systems
    3. Initialization: Finalize plant, create simulator
    4. Execution: Run simulation
    5. Visualization: Set up Meshcat visualization
    6. Data logging: Record and plot simulation results (future)
    """
    
    def __init__(self, cup_manipulator_config: ManipulatorConfig, simulation_config: SimulationConfig):
        """Initialize scene manager."""
        self.cup_manipulator_config = cup_manipulator_config
        self.simulation_config = simulation_config
        
        # Drake objects
        self.builder = None
        self.plant = None
        self.model = None  #For control
        self.scene_graph = None
        self.meshcat = None
        self.controller = None
        self.diagram = None
        self.simulator = None
        self.context = None
        
        # Robots
        self.cup_manipulator: Optional[CupManipulator] = None
        self.pendulum: Optional[Pendulum3D] = None
        
        # Data logging
        self.time_log = []
        self.joint_positions_log = []  # Actual positions [link1, link2]
        self.joint_velocities_log = []  # Actual velocities [link1, link2]
        self.desired_positions_log = []  # Desired positions [link1, link2]
        self.desired_velocities_log = []  # Desired velocities [link1, link2]
        self.desired_accelerations_log = []  # Desired accelerations [link1_ddot, link2_ddot]
        self.commanded_accelerations_log = []  # Commanded accelerations (desired + feedback) [link1_ddot_cmd, link2_ddot_cmd]
        self.control_torques_log = []  # Control torques [tau1, tau2]
        self.position_errors_log = []  # Position tracking errors
        self.velocity_errors_log = []  # Velocity tracking errors
        self.pendulum_positions_log = []  # Pendulum [pitch, roll]
        self.pendulum_velocities_log = []  # Pendulum velocities [pitch_dot, roll_dot]
        self.pendulum_ball_position_log = []  # Pendulum ball center position in world frame [x, y, z]
        self.pendulum_ball_distance_log = []  # Euclidean distance from pivot to ball center (should be constant = L)
        
        # Frame visualization
        self.frame_list = []  # List of (frame_name, frame, length) tuples for updating
        
        print("\n" + "=" * 70)
        print("Drake Scene Manager Initialized (Controller Architecture)")
        print("=" * 70)
    
    def setup_drake_system(self):
        """
        Setup Drake's MultibodyPlant, load robots, and add controller.
        
        This builds the core Diagram structure:
            Plant → [state] → Controller → [torque] → Plant
        
        CREATES SYSTEM 1 (PLANT): The physics simulation model
        """
        print(colored("\n[1/5] Setting up Drake system (SYSTEM 1: Physics Plant)...", 'blue', attrs=['bold']))
        
        # Create diagram builder
        self.builder = DiagramBuilder()
        
        # ═══════════════════════════════════════════════════════════════════
        # SYSTEM 1: MultibodyPlant - Physics Simulation
        # ═══════════════════════════════════════════════════════════════════
        # This is a Drake LeafSystem that simulates robot dynamics
        # Inputs: Control torques τ
        # Outputs: State [q, v] = [positions, velocities]
        # ═══════════════════════════════════════════════════════════════════
        print(colored("\n--- Adding MultibodyPlant and SceneGraph ---", 'yellow', attrs=['bold']))
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=self.simulation_config.timestep
        )
        print(colored("  ✓ MultibodyPlant (SYSTEM 1) added to diagram", 'cyan'))
        print(colored("    Role: Physics engine - solves M(q)q̈ + C(q,v)v = τ_g(q) + τ_applied", 'cyan'))
        
        # Create parser for URDF loading
        parser = Parser(self.plant)
        
        # Load cup manipulator
        print(colored("\n--- Loading Cup Manipulator ---", 'yellow', attrs=['bold']))
        self.cup_manipulator = CupManipulator(self.cup_manipulator_config)
        self.cup_manipulator.load_urdf_to_plant(self.plant, parser)
        
        # Weld base to world
        print(colored("\n--- Welding Base to World ---", 'yellow', attrs=['bold']))
        base_frame = self.plant.GetBodyByName("base_mount_manipulator", self.cup_manipulator.model_instance).body_frame()
        self.plant.WeldFrames(self.plant.world_frame(), base_frame)
        print(colored(f"✓ Base welded to world", 'green'))
        
        # Add actuators
        print(colored("\n--- Adding Actuators ---", 'yellow', attrs=['bold']))
        for joint_name in ["link1_base", "link2_link1"]:
            joint = self.plant.GetJointByName(joint_name, self.cup_manipulator.model_instance)
            self.plant.AddJointActuator(joint_name, joint)
        print(colored(f"✓ Actuators added: link1_base, link2_link1", 'green'))
        
        # Set joint properties
        print(colored("\n--- Setting Joint Properties ---", 'yellow', attrs=['bold']))
        self.cup_manipulator.set_joint_properties(self.plant)
        
        # Add pendulum if enabled
        if PENDULUM_ENABLED:
            print(colored("\n--- Adding Programmatic Pendulum ---", 'yellow', attrs=['bold']))
            self.pendulum = Pendulum3D(PENDULUM_CONFIG)
            link2_body = self.plant.GetBodyByName("link2", self.cup_manipulator.model_instance)
            self.pendulum.attach_to_body(self.plant, link2_body, self.cup_manipulator.model_instance)
            print(colored(f"✓ Added 3D pendulum to link2", 'green'))
        else:
            self.pendulum = None
        
        # Set gravity
        gravity_field = self.plant.mutable_gravity_field()
        gravity_field.set_gravity_vector(list(self.simulation_config.gravity))
        
        # Finalize plant
        print(colored("\n--- Finalizing Plant ---", 'yellow', attrs=['bold']))
        self.plant.Finalize()
        print(colored(f"✓ SYSTEM 1 (Plant) finalized with {self.plant.num_positions()} positions and {self.plant.num_velocities()} velocities", 'green', attrs=['bold']))
        print(colored(f"  State dimension: {self.plant.num_positions() + self.plant.num_velocities()}", 'cyan'))
        print(colored(f"  Input dimension: {self.plant.num_actuators()} (control torques)", 'cyan'))
        
        # Initialize robot state
        self.cup_manipulator.initialize_state(self.plant)
        
        print(colored("\n✓ Drake MultibodyPlant (SYSTEM 1) setup complete", 'green', attrs=['bold']))
    
    def create_model_for_controller(self):
        """
        Create a separate MultibodyPlant model for the controller's internal dynamics calculations.
        
        This is a key architectural improvement:
        - The controller uses its own MODEL for inverse dynamics, separate from the PLANT.
        - The MODEL can have different parameters (e.g., mass, length) to simulate model-plant mismatch.
        - The controller observes state from the PLANT but computes control using the MODEL.
        
        Benefits:
        1. Sim-to-real transfer: Swap out the PLANT for real hardware without changing controller code.
        2. Robustness testing: Intentionally create model-plant mismatch to test controller performance.
        3. Adaptability: Update model parameters online (e.g., for adaptive control) without touching plant.
        
        Returns:
            model_plant: A MultibodyPlant instance representing the controller's internal model.
        """
        
        print(colored("\n--- Creating Controller's Internal Model (Separate from Plant) ---", 'yellow', attrs=['bold']))
        print(colored("  This model is NOT in the Drake diagram", 'cyan'))
        print(colored("  It is ONLY used by controller for inverse dynamics calculations", 'cyan'))
        
        # Create a separate MultibodyPlant for the controller
        # This plant will NOT be added to the diagram - it's just for computations
        model_plant = MultibodyPlant(time_step=self.simulation_config.timestep)
        model_parser = Parser(model_plant)
        
        # Load same robot structure into model
        model_manipulator = CupManipulator(self.cup_manipulator_config)
        model_manipulator.load_urdf_to_plant(model_plant, model_parser)
        
        # Weld base (same as plant)
        model_base_frame = model_plant.GetBodyByName("base_mount_manipulator", model_manipulator.model_instance).body_frame()
        model_plant.WeldFrames(model_plant.world_frame(), model_base_frame)
        
        # Add actuators (same as plant)
        for joint_name in ["link1_base", "link2_link1"]:
            joint = model_plant.GetJointByName(joint_name, model_manipulator.model_instance)
            model_plant.AddJointActuator(joint_name, joint)
        
        # Set joint properties (same as plant for now, but could differ!)
        model_manipulator.set_joint_properties(model_plant)
        
        # Add pendulum if enabled (same structure as plant)
        if PENDULUM_ENABLED:
            model_pendulum = Pendulum3D(PENDULUM_CONFIG)
            model_link2_body = model_plant.GetBodyByName("link2", model_manipulator.model_instance)
            model_pendulum.attach_to_body(model_plant, model_link2_body, model_manipulator.model_instance)
        
        # Set gravity (same as plant)
        model_gravity_field = model_plant.mutable_gravity_field()
        model_gravity_field.set_gravity_vector(list(self.simulation_config.gravity))
        
        # Finalize the model plant (NOT added to diagram!)
        model_plant.Finalize()
        
        print(colored("  ✓ Controller's internal model created and finalized", 'green'))
        print(colored("  ✓ For now: Model parameters = Plant parameters (perfect model)", 'cyan'))
        print(colored("  → Future: Can modify model params to test robustness!", 'yellow'))

        self.model = model_plant  # Store for potential future use (e.g., adaptive control)
    
    
    def add_controller(self):
        """
        Add controller system to the diagram and wire ports.
        
        CREATES SYSTEM 2 (CONTROLLER): The control law model
        WIRES: Plant.state → Controller.input, Controller.output → Plant.torque
        """
        print(colored("\n[2/5] Adding controller to diagram (SYSTEM 2: Control Model)...", 'blue', attrs=['bold']))
        
        # ═══════════════════════════════════════════════════════════════════
        # SYSTEM 2: Controller - Control Law Computation
        # ═══════════════════════════════════════════════════════════════════
        # This is our custom LeafSystem that computes control torques
        # Inputs: State [q, v] from plant
        # Outputs: Control torques τ to apply to plant
        # ═══════════════════════════════════════════════════════════════════
        
        # Create controller with appropriate gains
        # IMPORTANT: Computed torque uses MUCH SMALLER feedback gains than PD
        # because the feedforward term already compensates for dynamics
        if CONTROLLER_MODE == 'pd':
            Kp = np.array([100.0, 100.0])
            Kd = np.array([10.0, 10.0])
        elif CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics':
            # Reduced gains: feedforward handles dynamics, feedback only corrects errors
            Kp = np.array([20.0, 20.0])  # 5x smaller than PD
            Kd = np.array([5.0, 5.0])    # 2x smaller than PD
        else:
            Kp = np.array([100.0, 100.0])
            Kd = np.array([10.0, 10.0])
        
        print(colored(f"\n--- Creating Controller: {CONTROLLER_MODE.upper()} ---", 'yellow', attrs=['bold']))
        print(colored(f"  Gains: Kp={Kp}, Kd={Kd}", 'cyan'))
        
        if CONTROLLER_MODE == 'pd':
            self.controller = self.builder.AddSystem(
                PDController(self.plant, self.cup_manipulator.model_instance, Kp, Kd, MANIPULATOR_MOTION_DURATION)
            )
            print(colored(f"✓ PDController (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: Computes control torques τ = Kp*(q_d - q) + Kd*(q_dot_d - q_dot)", 'cyan'))
        
        elif CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics':
            # ═══════════════════════════════════════════════════════════════════
            # CREATE SEPARATE MODEL FOR CONTROLLER (Model-Plant Separation)
            # ═══════════════════════════════════════════════════════════════════
            # Key architectural improvement:
            # - PLANT: The "real" system (in diagram, executes physics)
            # - MODEL: Controller's internal model (NOT in diagram, for computation)
            # 
            # Benefits:
            # 1. Sim-to-real: Same controller works with different plants
            # 2. Robustness: Test controller with model-plant mismatch
            # 3. Adaptability: Update model parameters without touching plant
            # ═══════════════════════════════════════════════════════════════════
            
            self.create_model_for_controller()  # Create separate model for controller computations
            model_plant = self.model  # For clarity
            
            # Create controller with BOTH plant and model
            self.controller = self.builder.AddSystem(
                ComputedTorqueController(
                    plant=self.plant,              # Real system (for state reading via ports)
                    model=model_plant,             # Controller's model (for dynamics calculations)
                    model_instance=self.cup_manipulator.model_instance,
                    Kp=Kp,
                    Kd=Kd,
                    motion_duration=MANIPULATOR_MOTION_DURATION
                )
            )
            print(colored(f"✓ ComputedTorqueController (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: τ = M_model(q)·[q_ddot_d + Kp·e + Kd·ė] + C_model(q,q_dot) + g_model(q)", 'cyan'))
            print(colored(f"  Plant: Observes state [q, v] via input port", 'cyan'))
            print(colored(f"  Model: Computes inverse dynamics for control", 'cyan'))
            print(colored(f"  Feedforward: Inverse dynamics with feedback-modified acceleration", 'cyan'))
            print(colored(f"  Note: Feedback compensates for model-plant mismatch!", 'yellow'))
        
        else:
            raise ValueError(f"Unknown controller mode: {CONTROLLER_MODE}")
        
        # ═══════════════════════════════════════════════════════════════════
        # WIRE THE TWO SYSTEMS TOGETHER VIA PORTS
        # ═══════════════════════════════════════════════════════════════════
        # Connection 1: Plant → Controller (state feedback)
        # Connection 2: Controller → Plant (torque commands)
        # This creates a closed-loop control system
        # ═══════════════════════════════════════════════════════════════════
        print(colored("\n--- Wiring SYSTEM 1 ↔ SYSTEM 2 Ports ---", 'yellow', attrs=['bold']))
        
        # Connection 1: Plant state output → Controller input
        self.builder.Connect(
            self.plant.get_state_output_port(self.cup_manipulator.model_instance),
            self.controller.get_input_port(0)
        )
        print(colored("  ✓ Connection 1: Plant.state_output → Controller.input", 'cyan'))
        print(colored("    Data: [q, q_dot] = [positions, velocities]", 'cyan'))
        
        # Connection 2: Controller output → Plant actuation input
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.plant.get_actuation_input_port(self.cup_manipulator.model_instance)
        )
        print(colored("  ✓ Connection 2: Controller.output → Plant.actuation_input", 'cyan'))
        print(colored("    Data: τ = [torque_link1, torque_link2]", 'cyan'))
        
        print(colored("\n✓ TWO-SYSTEM CLOSED-LOOP CONTROL established!", 'green', attrs=['bold']))
        print(colored("  Flow: Plant → state → Controller → torque → Plant (feedback loop)", 'green'))
        
        # Print ASCII diagram of the two-system architecture
        print(colored("\n" + "─"*70, 'cyan'))
        print(colored("TWO-SYSTEM DIAGRAM:", 'cyan', attrs=['bold']))
        print(colored("─"*70, 'cyan'))
        print(colored("                    ┌─────────────────┐", 'white'))
        print(colored("                    │   SYSTEM 2:     │", 'yellow'))
        print(colored("         ┌──────────│  PDController   │◄─────────┐", 'white'))
        print(colored("         │  τ       │  (Control Law)  │  [q, v]  │", 'yellow'))
        print(colored("         │  torques └─────────────────┘  state   │", 'white'))
        print(colored("         ▼                                        │", 'white'))
        print(colored("    ┌─────────────────┐                          │", 'white'))
        print(colored("    │   SYSTEM 1:     │                          │", 'green'))
        print(colored("    │ MultibodyPlant  ├──────────────────────────┘", 'white'))
        print(colored("    │   (Physics)     │", 'green'))
        print(colored("    └─────────────────┘", 'white'))
        print(colored("─"*70 + "\n", 'cyan'))
    
    def setup_visualization(self):
        """Setup Meshcat visualization."""
        if not self.simulation_config.visualization.enabled:
            return
        
        print(colored("\n[3/5] Setting up visualization...", 'blue', attrs=['bold']))
        self.meshcat = StartMeshcat()
        
        visualizer_params = MeshcatVisualizerParams()
        visualizer_params.show_hydroelastic = self.simulation_config.visualization.show_hydroelastic
        visualizer_params.show_contact_forces = self.simulation_config.visualization.show_contact_forces
        
        MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat, visualizer_params
        )
        
        print(colored(f"\n✓ Meshcat Visualization Started", 'green', attrs=['bold']))
        print(colored(f"  URL: {self.meshcat.web_url()}", 'cyan', attrs=['bold']))
        print(colored(f"  Hydroelastic: {visualizer_params.show_hydroelastic}", 'cyan'))
        print(colored(f"  Contact forces: {visualizer_params.show_contact_forces}", 'cyan'))
        print(colored(f"  Interactive controls: {self.simulation_config.visualization.interactive}", 'cyan'))
        print(colored(f"\n  👉 Open the URL above in your browser to view the simulation", 'yellow', attrs=['bold']))
    
    def _add_frame_visualizations(self, context):
        """Add coordinate frame visualizations to Meshcat after plant is finalized."""
        if not SIMULATION_CONFIG.visualization.plot_frames or not SIMULATION_CONFIG.visualization.enabled or not self.meshcat:
            return
            
        print(colored("\n--- Adding Frame Visualizations ---", 'yellow', attrs=['bold']))
        
        # Helper function to create a coordinate frame triad
        def add_frame_triad(meshcat, path, length=0.1):
            """Add XYZ coordinate frame to Meshcat."""
            # X-axis (red)
            meshcat.SetObject(f"{path}/X", Cylinder(radius=length*0.02, length=length),
                            rgba=Rgba(1.0, 0.0, 0.0, 1.0))
            meshcat.SetTransform(f"{path}/X", 
                               RigidTransform(RotationMatrix.MakeYRotation(np.pi/2), 
                                            [length/2, 0, 0]))
            # Y-axis (green)
            meshcat.SetObject(f"{path}/Y", Cylinder(radius=length*0.02, length=length),
                            rgba=Rgba(0.0, 1.0, 0.0, 1.0))
            meshcat.SetTransform(f"{path}/Y", 
                               RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2), 
                                            [0, length/2, 0]))
            # Z-axis (blue)
            meshcat.SetObject(f"{path}/Z", Cylinder(radius=length*0.02, length=length),
                            rgba=Rgba(0.0, 0.0, 1.0, 1.0))
            meshcat.SetTransform(f"{path}/Z", 
                               RigidTransform([0, 0, length/2]))
        
        # Add world frame at origin
        add_frame_triad(self.meshcat, "/Frames/World", length=0.20)
        self.meshcat.SetTransform("/Frames/World", RigidTransform())
        print(colored("  ✓ World frame (origin)", 'cyan'))
        
        # Loop through all frames in the plant and add them
        from pydrake.multibody.tree import FrameIndex
        for i in range(self.plant.num_frames()):
            frame = self.plant.get_frame(FrameIndex(i))
            frame_name = frame.name()
            
            # Skip world frame (already added)
            if frame_name == "world":
                continue
            
            # Determine frame length based on frame type
            if "pivot" in frame_name.lower():
                length = 0.15
            elif "gimbal" in frame_name.lower() or "pendulum" in frame_name.lower():
                length = 0.10
            else:
                length = 0.12
            
            # Add frame triad
            path = f"/Frames/{frame_name}"
            add_frame_triad(self.meshcat, path, length=length)
            
            # Store for updates
            self.frame_list.append((frame_name, frame, length))
            print(colored(f"  ✓ {frame_name}", 'cyan'))
        
        # Update all frame positions
        self._update_frame_positions(context)
        
        print(colored(f"✓ {len(self.frame_list) + 1} frame visualizations added", 'green'))
        print(colored("  Legend: X=Red, Y=Green, Z=Blue", 'yellow'))
    
    def _update_frame_positions(self, context):
        """Update frame positions in Meshcat."""
        if not SIMULATION_CONFIG.visualization.plot_frames or not SIMULATION_CONFIG.visualization.enabled or not self.meshcat:
            return
        
        if not hasattr(self, 'frame_list'):
            return
            
        # Update all frames in the list
        for frame_name, frame, length in self.frame_list:
            X_WF = self.plant.CalcRelativeTransform(context, self.plant.world_frame(), frame)
            self.meshcat.SetTransform(f"/Frames/{frame_name}", X_WF)
    
    def create_simulator(self):
        """Build diagram and create simulator."""
        print(colored("\n[4/5] Building diagram and creating simulator...", 'blue', attrs=['bold']))
        
        # Build the complete diagram
        self.diagram = self.builder.Build()
        print(colored("✓ Diagram built", 'green'))
        
        # Create simulator from diagram
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        print(colored("✓ Simulator created", 'green', attrs=['bold']))
        
        # Add frame visualizations after simulator is created
        if SIMULATION_CONFIG.visualization.plot_frames and SIMULATION_CONFIG.visualization.enabled:
            plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
            self._add_frame_visualizations(plant_context)
        
        # Add frame visualizations after simulator is created
        if SIMULATION_CONFIG.visualization.plot_frames and SIMULATION_CONFIG.visualization.enabled:
            plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
            self._add_frame_visualizations(plant_context)
    
    def set_initial_conditions(self):
        """Set initial joint positions and velocities."""
        print(colored("\n[5/5] Setting initial conditions...", 'blue', attrs=['bold']))
        
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Set manipulator joints to zero
        link1_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
        link2_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
        link1_joint.set_angle(plant_context, 0.0)
        link2_joint.set_angle(plant_context, 0.0)
        print(colored("  ✓ Manipulator joints: link1=0°, link2=0°", 'cyan'))
        
        # Set pendulum initial swing if enabled
        if PENDULUM_ENABLED and self.pendulum:
            pitch_joint = self.plant.GetJointByName("pendulum_pitch", self.cup_manipulator.model_instance)
            roll_joint = self.plant.GetJointByName("pendulum_roll", self.cup_manipulator.model_instance)
            pitch_joint.set_angle(plant_context, np.deg2rad(PENDULUM_CONFIG.initial_pitch))
            roll_joint.set_angle(plant_context, np.deg2rad(PENDULUM_CONFIG.initial_roll))
            print(colored(f"  ✓ Pendulum: pitch={PENDULUM_CONFIG.initial_pitch}°, roll={PENDULUM_CONFIG.initial_roll}°", 'cyan'))
        
        print(colored("\n✓ Initial conditions set", 'green', attrs=['bold']))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored("\n" + "="*70, 'green', attrs=['bold']))
        print(colored("Starting Simulation", 'green', attrs=['bold']))
        print(colored("="*70, 'green', attrs=['bold']))
        
        print(f"\nSimulation Parameters:")
        print(f"  Duration: {self.simulation_config.simulation_time} s")
        print(f"  Timestep: {self.simulation_config.timestep} s")
        print(f"  Realtime Rate: {self.simulation_config.visualization.realtime_rate}x")
        print(f"  Controller: {CONTROLLER_MODE.upper()}")
        print(f"  Motion Duration: {MANIPULATOR_MOTION_DURATION} s (then settling)")
        print()
        
        # Initialize and configure simulator
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(self.simulation_config.visualization.realtime_rate)
        
        # Run simulation with progress updates
        print(colored("Running simulation...\n", 'yellow'))
        try:
            # Separate intervals for data logging (high freq) vs terminal printing (low freq)
            print_interval = self.simulation_config.visualization.print_interval
            logging_interval = self.simulation_config.visualization.logging_interval
            sim_time = self.simulation_config.simulation_time
            current_time = 0.0
            next_print_time = 0.0
            
            print(colored(f"Data sampling: {1/logging_interval:.0f} Hz (every {logging_interval}s) for smooth plots", 'cyan'))
            print(colored(f"Terminal output: {1/print_interval:.1f} Hz (every {print_interval}s)\n", 'cyan'))
            
            while current_time < sim_time:
                # Advance to next logging point
                next_time = min(current_time + logging_interval, sim_time)
                self.simulator.AdvanceTo(next_time)
                
                # Get current state
                plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
                joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
                joint_velocities = self.cup_manipulator.get_joint_velocities(self.plant, plant_context)
                
                # Extract joint states
                t = next_time
                link1_pos = joint_positions.get('link1_base', 0.0)
                link2_pos = joint_positions.get('link2_link1', 0.0)
                link1_vel = joint_velocities.get('link1_base', 0.0)
                link2_vel = joint_velocities.get('link2_link1', 0.0)
                
                # Compute desired trajectory (same as in controller)
                if t < MANIPULATOR_MOTION_DURATION:
                    q_desired = np.array([
                        JOINT_MOTION_AMPLITUDE[0] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                        JOINT_MOTION_AMPLITUDE[1] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
                    ])
                    q_dot_desired = np.array([
                        JOINT_MOTION_AMPLITUDE[0] * 2 * np.pi * JOINT_MOTION_FREQUENCY[0] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                        JOINT_MOTION_AMPLITUDE[1] * 2 * np.pi * JOINT_MOTION_FREQUENCY[1] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
                    ])
                    q_ddot_desired = np.array([
                        -JOINT_MOTION_AMPLITUDE[0] * (2 * np.pi * JOINT_MOTION_FREQUENCY[0])**2 * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * t),
                        -JOINT_MOTION_AMPLITUDE[1] * (2 * np.pi * JOINT_MOTION_FREQUENCY[1])**2 * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * t)
                    ])
                else:
                    # Use the controller's stop position if available
                    if hasattr(self.controller, 'stop_position'):
                        q_desired = self.controller.stop_position
                    else:
                        q_desired = np.array([link1_pos, link2_pos])
                    q_dot_desired = np.zeros(2)
                    q_ddot_desired = np.zeros(2)
                
                # Get control torques from controller output port
                controller_context = self.controller.GetMyContextFromRoot(self.context)
                control_torques = self.controller.get_output_port(0).Eval(controller_context)
                
                # Compute errors
                position_error = q_desired - np.array([link1_pos, link2_pos])
                velocity_error = q_dot_desired - np.array([link1_vel, link2_vel])
                
                # Compute commanded acceleration (includes feedback for computed torque mode)
                # For computed torque: q_ddot_cmd = q_ddot_d + Kp*e + Kd*e_dot
                if CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics':
                    # Get gains from controller
                    if t < MANIPULATOR_MOTION_DURATION:
                        Kp = self.controller.Kp
                        Kd = self.controller.Kd
                    else:
                        Kp = self.controller.Kp_hold
                        Kd = self.controller.Kd_hold
                    q_ddot_commanded = q_ddot_desired + Kp * position_error + Kd * velocity_error
                else:
                    # For PD mode, commanded acceleration is just desired (no feedforward)
                    q_ddot_commanded = q_ddot_desired.copy()
                
                # Log data at high frequency for smooth plots
                self.time_log.append(t)
                self.joint_positions_log.append([link1_pos, link2_pos])
                self.joint_velocities_log.append([link1_vel, link2_vel])
                self.desired_positions_log.append(q_desired.copy())
                self.desired_velocities_log.append(q_dot_desired.copy())
                self.desired_accelerations_log.append(q_ddot_desired.copy())
                self.commanded_accelerations_log.append(q_ddot_commanded.copy())
                self.control_torques_log.append(control_torques.copy())
                self.position_errors_log.append(position_error)
                self.velocity_errors_log.append(velocity_error)
                
                # Log pendulum states if enabled
                if PENDULUM_ENABLED:
                    pitch = joint_positions.get('pendulum_pitch', 0.0)
                    roll = joint_positions.get('pendulum_roll', 0.0)
                    pitch_dot = joint_velocities.get('pendulum_pitch', 0.0)
                    roll_dot = joint_velocities.get('pendulum_roll', 0.0)
                    self.pendulum_positions_log.append([pitch, roll])
                    self.pendulum_velocities_log.append([pitch_dot, roll_dot])
                    
                    # Get pendulum ball center position relative to pivot frame
                    if self.pendulum and self.pendulum.pendulum_body:
                        ball_frame = self.pendulum.pendulum_body.body_frame()
                        # Ball center is at [0, 0, -L] in body frame (straight down when pendulum is upright)
                        ball_offset_in_body = np.array([0.0, 0.0, -PENDULUM_CONFIG.length])
                        # Get transform from body frame to world frame
                        X_WB = self.plant.CalcRelativeTransform(
                            plant_context,
                            self.plant.world_frame(),
                            ball_frame
                        )
                        # Rotate the ball offset from body frame to pivot frame coordinates
                        # This shows how the ball position changes due to pendulum rotation
                        # When pendulum is upright (0,0), this gives [0, 0, -0.2]
                        # When pendulum tilts, the ball swings in X and Y
                        ball_wrt_pivot = X_WB.rotation() @ ball_offset_in_body
                        self.pendulum_ball_position_log.append(ball_wrt_pivot.copy())
                        
                        # Verify rigid body constraint: distance should be constant = L
                        distance = np.linalg.norm(ball_wrt_pivot)
                        self.pendulum_ball_distance_log.append(distance)
                    
                    # Update frame positions
                    self._update_frame_positions(plant_context)
                
                # Print progress at lower frequency (only at print_interval)
                if next_time >= next_print_time:
                    progress_pct = (next_time / sim_time) * 100
                    print(colored(f"[{next_time:5.2f}s / {sim_time:.1f}s ({progress_pct:5.1f}%)] ", 'yellow'), end='')
                    print(f"link1={np.rad2deg(link1_pos):7.2f}°  link2={np.rad2deg(link2_pos):7.2f}°", end='')
                    
                    if PENDULUM_ENABLED:
                        print(f"  pendulum: pitch={np.rad2deg(pitch):7.2f}°  roll={np.rad2deg(roll):7.2f}°", end='')
                    
                    print()  # New line
                    next_print_time += print_interval
                
                current_time = next_time
            
            print(colored("\n✓ Simulation completed successfully!", 'green', attrs=['bold']))
        except Exception as e:
            print(colored(f"\n✗ Simulation error: {e}", 'red', attrs=['bold']))
            import traceback
            traceback.print_exc()
    
    def run_scene_viz(self):
        """Run static scene visualization (no physics simulation)."""
        print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
        print(colored("Starting Scene Visualization (Static)", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan', attrs=['bold']))
        
        print(colored("\nVisualization Mode: Static Scene", 'yellow'))
        print(colored("  - No physics simulation", 'yellow'))
        print(colored("  - Shows initial configuration", 'yellow'))
        print(colored("  - All coordinate frames visible", 'yellow'))
        print(colored("  - Press Ctrl+C to exit\n", 'yellow'))
        
        if not self.meshcat:
            print(colored("\n✗ Visualization not enabled", 'red'))
            return
        
        print(colored(f"\n✓ Meshcat URL: {self.meshcat.web_url()}", 'green', attrs=['bold']))
        print(colored("  👉 Open this URL in your browser to view the scene\n", 'yellow', attrs=['bold']))
        
        # Keep the program running so Meshcat stays active
        try:
            print(colored("Scene visualization active. Press Ctrl+C to exit...", 'cyan'))
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print(colored("\n\n✓ Scene visualization closed by user", 'green'))
            print(colored("="*70 + "\n", 'green'))
    
    def plot_results(self):
        """Plot simulation results with desired vs actual trajectories and errors."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if len(self.time_log) == 0:
            print(colored("\n⚠ No data to plot", 'yellow'))
            return
        
        print(colored("\n" + "="*70, 'blue', attrs=['bold']))
        print(colored("Generating Plots...", 'blue', attrs=['bold']))
        print(colored("="*70, 'blue', attrs=['bold']))
        
        # Convert lists to numpy arrays
        time = np.array(self.time_log)
        q_actual = np.array(self.joint_positions_log)  # [N x 2]
        q_dot_actual = np.array(self.joint_velocities_log)  # [N x 2]
        q_desired = np.array(self.desired_positions_log)  # [N x 2]
        q_dot_desired = np.array(self.desired_velocities_log)  # [N x 2]
        q_ddot_desired = np.array(self.desired_accelerations_log)  # [N x 2]
        q_ddot_commanded = np.array(self.commanded_accelerations_log)  # [N x 2]
        control_torques = np.array(self.control_torques_log)  # [N x 2]
        pos_errors = np.array(self.position_errors_log)  # [N x 2]
        vel_errors = np.array(self.velocity_errors_log)  # [N x 2]
        
        # Create figure with subplots (4x2 layout to accommodate ball position)
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Joint names for plots
        joint_names = ['Link1 (Base)', 'Link2 (Elbow)']
        colors_actual = ['#2E86AB', '#A23B72']
        colors_desired = ['#06D6A0', '#F18F01']
        
        # ===================================================================
        # Row 1: Joint Positions (Both joints in one plot)
        # ===================================================================
        ax = fig.add_subplot(gs[0, 0])
        for i in range(2):
            ax.plot(time, np.rad2deg(q_actual[:, i]), label=f'{joint_names[i]} - Actual', 
                   color=colors_actual[i], linewidth=2)
            ax.plot(time, np.rad2deg(q_desired[:, i]), '--', label=f'{joint_names[i]} - Desired', 
                   color=colors_desired[i], linewidth=2, alpha=0.8)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label='Hold Start')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (deg)', fontsize=11)
        ax.set_title('Manipulator Joint Positions - Tracking', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 1: Joint Velocities (Both joints in one plot)
        # ===================================================================
        ax = fig.add_subplot(gs[0, 1])
        for i in range(2):
            ax.plot(time, np.rad2deg(q_dot_actual[:, i]), label=f'{joint_names[i]} - Actual', 
                   color=colors_actual[i], linewidth=2)
            ax.plot(time, np.rad2deg(q_dot_desired[:, i]), '--', label=f'{joint_names[i]} - Desired', 
                   color=colors_desired[i], linewidth=2, alpha=0.8)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label='Hold Start')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (deg/s)', fontsize=11)
        ax.set_title('Manipulator Joint Velocities - Tracking', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 2: Tracking Errors (Position and Velocity)
        # ===================================================================
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(time, np.rad2deg(pos_errors[:, 0]), label='Link1', 
               color=colors_actual[0], linewidth=1.5)
        ax.plot(time, np.rad2deg(pos_errors[:, 1]), label='Link2', 
               color=colors_actual[1], linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position Error (deg)', fontsize=11)
        ax.set_title('Position Tracking Errors', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(time, np.rad2deg(vel_errors[:, 0]), label='Link1', 
               color=colors_actual[0], linewidth=1.5)
        ax.plot(time, np.rad2deg(vel_errors[:, 1]), label='Link2', 
               color=colors_actual[1], linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity Error (deg/s)', fontsize=11)
        ax.set_title('Velocity Tracking Errors', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 3: Control Torques (Both joints in one plot)
        # ===================================================================
        ax = fig.add_subplot(gs[2, 0])
        for i in range(2):
            ax.plot(time, control_torques[:, i], 
                   color=colors_actual[i], linewidth=1.5, label=f'{joint_names[i]}')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label='Hold Start')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Torque (N⋅m)', fontsize=11)
        ax.set_title('Control Torques (Manipulator Joints)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 3: Pendulum States (if enabled)
        # ===================================================================
        if PENDULUM_ENABLED and len(self.pendulum_positions_log) > 0:
            pendulum_pos = np.array(self.pendulum_positions_log)  # [N x 2]
            pendulum_vel = np.array(self.pendulum_velocities_log)  # [N x 2]
            pendulum_ball_pos = np.array(self.pendulum_ball_position_log)  # [N x 3]
            
            # Left plot: Pendulum joint angles and velocities
            ax = fig.add_subplot(gs[2, 1])
            ax.plot(time, np.rad2deg(pendulum_pos[:, 0]), label='Pitch', 
                   color='#E63946', linewidth=1.5)
            ax.plot(time, np.rad2deg(pendulum_pos[:, 1]), label='Roll', 
                   color='#457B9D', linewidth=1.5)
            ax.plot(time, np.rad2deg(pendulum_vel[:, 0]), '--', label='Pitch Rate', 
                   color='#E63946', linewidth=1.2, alpha=0.7)
            ax.plot(time, np.rad2deg(pendulum_vel[:, 1]), '--', label='Roll Rate', 
                   color='#457B9D', linewidth=1.2, alpha=0.7)
            ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                      linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Angle (deg) / Rate (deg/s)', fontsize=11)
            ax.set_title('Pendulum Motion (Angles & Rates)', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Row 4: Ball center position (X, Y, Z) vs time
            ax = fig.add_subplot(gs[3, :])
            ax.plot(time, pendulum_ball_pos[:, 0], label='X', 
                   color='#2E86AB', linewidth=1.5)
            ax.plot(time, pendulum_ball_pos[:, 1], label='Y', 
                   color='#A23B72', linewidth=1.5)
            ax.plot(time, pendulum_ball_pos[:, 2], label='Z', 
                   color='#06D6A0', linewidth=1.5)
            ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                      linewidth=1.5, alpha=0.5, label='Hold Start')
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Position (m)', fontsize=11)
            ax.set_title('Pendulum Ball Center Position (Relative to Pivot)', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Verify constraint: distance from pivot should be constant = L
            if len(self.pendulum_ball_distance_log) > 0:
                distances = np.array(self.pendulum_ball_distance_log)
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                max_error = np.max(np.abs(distances - PENDULUM_CONFIG.length))
                
                print(colored(f"\n{'='*70}", 'yellow'))
                print(colored(f"Pendulum Rigid Body Constraint Verification:", 'yellow', attrs=['bold']))
                print(colored(f"  Expected length: {PENDULUM_CONFIG.length:.6f} m", 'cyan'))
                print(colored(f"  Mean distance:   {mean_dist:.6f} m", 'cyan'))
                print(colored(f"  Std deviation:   {std_dist:.9f} m", 'cyan'))
                print(colored(f"  Max error:       {max_error:.9f} m", 'cyan'))
                if max_error < 1e-6:
                    print(colored(f"  ✓ PASSED: Distance is constant (error < 1μm)", 'green', attrs=['bold']))
                else:
                    print(colored(f"  ✗ FAILED: Distance varies beyond numerical precision", 'red', attrs=['bold']))
                print(colored(f"{'='*70}\n", 'yellow'))
        
        # Overall title
        fig.suptitle(f'Simulation Results - Controller: {CONTROLLER_MODE.upper()}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save plot
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'plots/simulation_results_{CONTROLLER_MODE}_{timestamp}.png'
        os.makedirs('plots', exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(colored(f"\n✓ Plot saved: {plot_filename}", 'green', attrs=['bold']))
        
        # Display plot
        plt.show()
        print(colored("✓ Plots displayed", 'green'))


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution flow for Drake controller-based simulation.
    
    DEMONSTRATES TWO-SYSTEM ARCHITECTURE:
    ────────────────────────────────────────────────────────────────────────
    This script creates TWO separate Drake systems and connects them:
    
    1. MultibodyPlant (SYSTEM 1 - Physics):
       - Loaded with robot URDF
       - Simulates dynamics: M(q)q̈ + C(q,v)v = τ_g(q) + τ_applied
       - Provides state [q, v] to controller
       - Receives torques τ from controller
    
    2. PDController (SYSTEM 2 - Control):
       - Custom LeafSystem we wrote
       - Computes control law: τ = Kp*(q_d - q) + Kd*(v_d - v)
       - Receives state from plant
       - Sends torques to plant
    
    They communicate via Drake ports (not manual Python code).
    The Diagram automatically handles data flow each timestep.
    ────────────────────────────────────────────────────────────────────────
    """
    print("\n" + "=" * 70)
    print(colored("PYDRAKE: Two-System Controller Architecture", 'cyan', attrs=['bold']))
    print(colored("SYSTEM 1 (Plant) ↔ SYSTEM 2 (Controller)", 'cyan'))
    print("=" * 70)
    print(colored(f"Controller Mode: {CONTROLLER_MODE}", 'yellow', attrs=['bold']))
    print(colored(f"Time Step: {SIMULATION_CONFIG.timestep} s", 'yellow'))
    print(colored(f"Duration: {SIMULATION_CONFIG.simulation_time} s", 'yellow'))
    print(colored(f"Gravity: {SIMULATION_CONFIG.gravity} m/s²", 'yellow'))
    print(colored(f"Visualization: {'Enabled' if SIMULATION_CONFIG.visualization.enabled else 'Disabled'}", 'yellow'))
    print(colored(f"Realtime Rate: {SIMULATION_CONFIG.visualization.realtime_rate}x", 'yellow'))
    if PENDULUM_ENABLED:
        print(colored(f"Pendulum: Enabled (mass={PENDULUM_CONFIG.mass}kg, length={PENDULUM_CONFIG.length}m)", 'yellow'))
    else:
        print(colored(f"Pendulum: Disabled", 'yellow'))
    print("=" * 70 + "\n")
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # BUILD THE TWO-SYSTEM DIAGRAM
        # ═══════════════════════════════════════════════════════════════════
        print(colored("Building two-system architecture...\n", 'magenta', attrs=['bold']))
        
        # Create scene manager
        scene_manager = DrakeSceneManager(
            CUP_MANIPULATOR_CONFIG,
            SIMULATION_CONFIG
        )
        
        # Step 1: Setup SYSTEM 1 (Plant) - the physics model
        scene_manager.setup_drake_system()
        
        # Step 2: Add SYSTEM 2 (Controller) and wire to SYSTEM 1 (skip for scene-viz)
        if CONTROLLER_MODE != 'scene-viz':
            scene_manager.add_controller()
        
        # Setup visualization
        scene_manager.setup_visualization()
        
        # Create simulator
        scene_manager.create_simulator()
        
        # Set initial conditions
        scene_manager.set_initial_conditions()
        
        # Run simulation or scene visualization
        if CONTROLLER_MODE == 'scene-viz':
            scene_manager.run_scene_viz()
        else:
            scene_manager.run_simulation()
            # Generate plots
            scene_manager.plot_results()
        
            # Print final summary
            print(colored("\n" + "="*70, 'green', attrs=['bold']))
            print(colored("Simulation Complete - Summary", 'green', attrs=['bold']))
            print(colored("="*70, 'green', attrs=['bold']))
            print(colored(f"✓ Total simulation time: {SIMULATION_CONFIG.simulation_time} s", 'cyan'))
            print(colored(f"✓ Controller mode: {CONTROLLER_MODE.upper()}", 'cyan'))
            print(colored(f"✓ Manipulator DOFs: 2 (link1_base, link2_link1)", 'cyan'))
            if PENDULUM_ENABLED:
                print(colored(f"✓ Pendulum DOFs: 2 (pitch, roll)", 'cyan'))
                print(colored(f"✓ Total system DOFs: 4", 'cyan'))
            else:
                print(colored(f"✓ Total system DOFs: 2", 'cyan'))
            
            # Display tracking performance metrics
            if len(scene_manager.position_errors_log) > 0:
                pos_errors = np.array(scene_manager.position_errors_log)
                vel_errors = np.array(scene_manager.velocity_errors_log)
                pos_rms = np.sqrt(np.mean(pos_errors**2, axis=0))
                vel_rms = np.sqrt(np.mean(vel_errors**2, axis=0))
                print(colored(f"\nTracking Performance:", 'cyan', attrs=['bold']))
                print(colored(f"  Position RMS Error: Link1={np.rad2deg(pos_rms[0]):.3f}°, Link2={np.rad2deg(pos_rms[1]):.3f}°", 'cyan'))
                print(colored(f"  Velocity RMS Error: Link1={np.rad2deg(vel_rms[0]):.3f}°/s, Link2={np.rad2deg(vel_rms[1]):.3f}°/s", 'cyan'))
            
            print(colored("="*70 + "\n", 'green', attrs=['bold']))
        
    except KeyboardInterrupt:
        print(colored("\n\n⚠ Simulation interrupted by user (Ctrl+C)", 'yellow', attrs=['bold']))
    except Exception as e:
        print(colored(f"\n\n✗ Error: {e}", 'red', attrs=['bold']))
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
