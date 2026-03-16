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
    LinearQuadraticRegulator,
    Linearize,
    
    # Mathematical utilities
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,
    
    # Frames
    FixedOffsetFrame,
)

# Custom robot types
from configs.robot.robot_types import (
    ManipulatorConfig,
    SimulationConfig,
    VisualizationConfig,
    PendulumConfig,
    create_cup_manipulator_config,
    create_pendulum_config,
)

# Task-space OFC controller
from task_space_ofc_implementation import TaskSpaceOFC

# Joint-space OFC controller
from joint_space_ofc_implementation import JointSpaceOFC

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Drake Diagram-based controller architecture')
parser.add_argument('--mode', type=str, choices=['pd', 'inverse-dynamics', 'computed-torque', 'scene-viz', 'min-jerk-joint', 'task-space-ofc', 'ofc-effort', 'ofc-smoothness', 'lqr'],
                    default='lqr', help='Controller type (scene-viz = static visualization only, lqr = LQR with minimum jerk)')
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
    attachment_point=(-1.2545, 0.0, -0.188125),
    initial_pitch=0.0,
    initial_roll=180.0,
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
JOINT_MOTION_FREQUENCY = [1.2/4, 1.0/4]  # Hz - increased for smoother motion

# --- Minimum-Jerk Controller Configuration ---
MIN_JERK_Q_START = np.deg2rad(np.array([80.0, -160.0, 0.0, 180.0]))  # [link1, link2, pitch, roll]
MIN_JERK_Q_GOAL = np.deg2rad(np.array([20.0, -40.0, 0.0, 180.0]))   # [link1, link2, pitch, roll]
MIN_JERK_DURATION = MANIPULATOR_MOTION_DURATION

# --- Task-Space OFC Configuration ---
TASK_SPACE_PIVOT_START = np.array([-0.5, 0.5])  # Initial pivot position [x, z]
TASK_SPACE_PIVOT_GOAL = np.array([0.5, 0.8])    # Goal pivot position [x, z]
TASK_SPACE_DURATION = MANIPULATOR_MOTION_DURATION
TASK_SPACE_MODE = 'effort'  # 'effort' or 'smoothness'

# --- Joint-Space OFC Configuration ---
JOINT_SPACE_Q_START = [80.0, -160.0]  # degrees
JOINT_SPACE_Q_GOAL = [20.0, -40.0]    # degrees
JOINT_SPACE_DURATION = MANIPULATOR_MOTION_DURATION
JOINT_SPACE_MA = 1.0      # Virtual mass [kg]
JOINT_SPACE_KP = 100.0    # Spring stiffness [N/m]
JOINT_SPACE_KD = 20.0     # Damping [N·s/m]
JOINT_SPACE_TAU_FILTER = 0.01  # F-dot filter time constant [s]

# --- LQR Configuration ---
LQR_Q_START = np.deg2rad(np.array([80.0, -160.0, 0.0, 180.0]))  # [link1, link2, pitch, roll]
LQR_Q_GOAL = np.deg2rad(np.array([20.0, -40.0, 0.0, 180.0]))     # [link1, link2, pitch, roll]
LQR_DURATION = MANIPULATOR_MOTION_DURATION
LQR_Q_WEIGHTS = np.array([100.0, 100.0, 500.0, 500.0])  # State cost for positions [manip, manip, pend, pend]
LQR_QDOT_WEIGHTS = np.array([10.0, 10.0, 50.0, 50.0])   # State cost for velocities
LQR_R_WEIGHTS = np.array([0.1, 0.1])                     # Control effort cost

# ============================================================================
# TRAJECTORY GENERATOR CLASS
# ============================================================================

class SinusoidalTrajectoryGenerator:
    """
    Generate sinusoidal trajectories for manipulator joints.
    Centralizes trajectory computation to avoid duplication.
    """
    
    def __init__(self, amplitudes, frequencies, motion_duration):
        """
        Args:
            amplitudes: List of amplitudes for each joint [rad]
            frequencies: List of frequencies for each joint [Hz]
            motion_duration: Duration of motion phase [s]
        """
        self.amplitudes = np.array(amplitudes)
        self.frequencies = np.array(frequencies)
        self.motion_duration = motion_duration
        self.stop_position = None  # Set when motion completes
    
    def compute_trajectory(self, t):
        """
        Compute desired position, velocity, and acceleration at time t.
        
        Args:
            t: Current time [s]
        
        Returns:
            tuple: (q_desired, q_dot_desired, q_ddot_desired)
        """
        if t < self.motion_duration:
            # Active motion phase - sinusoidal trajectory
            omega = 2 * np.pi * self.frequencies
            
            q_desired = self.amplitudes * np.sin(omega * t)
            q_dot_desired = self.amplitudes * omega * np.cos(omega * t)
            q_ddot_desired = -self.amplitudes * (omega ** 2) * np.sin(omega * t)
            
            # Save stop position at end of motion
            if t >= self.motion_duration - 1e-6:
                self.stop_position = q_desired.copy()
        else:
            # Holding phase - maintain final position
            if self.stop_position is None:
                # Fallback: compute final position
                omega = 2 * np.pi * self.frequencies
                self.stop_position = self.amplitudes * np.sin(omega * self.motion_duration)
            
            q_desired = self.stop_position
            q_dot_desired = np.zeros_like(self.amplitudes)
            q_ddot_desired = np.zeros_like(self.amplitudes)
        
        return q_desired, q_dot_desired, q_ddot_desired


# ============================================================================
# MINIMUM-JERK TRAJECTORY GENERATOR
# ============================================================================

class MinJerkTrajectoryGenerator:
    """
    Minimum-jerk joint-space trajectory generator.

    Uses 5th-order polynomial time scaling to minimize jerk.
    """

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float):
        self.q_start = np.array(q_start, dtype=float)
        self.q_goal = np.array(q_goal, dtype=float)
        self.motion_duration = float(duration)

    def _min_jerk_profile(self, t: float):
        if self.motion_duration <= 0:
            return 1.0, 0.0, 0.0
        s = np.clip(t / self.motion_duration, 0.0, 1.0)
        h = 10 * s**3 - 15 * s**4 + 6 * s**5
        hdot = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.motion_duration
        hddot = (60 * s - 180 * s**2 + 120 * s**3) / (self.motion_duration**2)
        return h, hdot, hddot

    def compute_trajectory(self, t: float):
        if t <= self.motion_duration:
            h, hdot, hddot = self._min_jerk_profile(t)
            q_desired = self.q_start + (self.q_goal - self.q_start) * h
            q_dot_desired = (self.q_goal - self.q_start) * hdot
            q_ddot_desired = (self.q_goal - self.q_start) * hddot
        else:
            q_desired = self.q_goal.copy()
            q_dot_desired = np.zeros_like(self.q_goal)
            q_ddot_desired = np.zeros_like(self.q_goal)

        return q_desired, q_dot_desired, q_ddot_desired


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
                 trajectory_generator: SinusoidalTrajectoryGenerator):
        """
        Initialize PD controller.
        
        Args:
            plant: MultibodyPlant reference (for future inverse dynamics)
            model_instance: Model instance ID
            Kp: Proportional gains [2] for actuated joints
            Kd: Derivative gains [2] for actuated joints
            trajectory_generator: Trajectory generator instance
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.model_instance = model_instance
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Kp_hold = self.Kp * 10.0  # Higher gains for settling phase
        self.Kd_hold = self.Kd * 10.0
        self.trajectory_generator = trajectory_generator
        self.motion_duration = trajectory_generator.motion_duration
        
        print(colored(f"\n--- PDController Configuration ---", 'yellow', attrs=['bold']))
        print(colored(f"  Kp (motion): {self.Kp}", 'cyan'))
        print(colored(f"  Kd (motion): {self.Kd}", 'cyan'))
        print(colored(f"  Kp (hold): {self.Kp_hold}", 'cyan'))
        print(colored(f"  Kd (hold): {self.Kd_hold}", 'cyan'))
        print(colored(f"  Motion duration: {self.motion_duration} s", 'cyan'))
        
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
        
        # Calculate desired trajectory using trajectory generator
        if t < self.motion_duration:
            q_desired, q_dot_desired, _ = self.trajectory_generator.compute_trajectory(t)
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
# LQR CONTROLLER WITH MINIMUM JERK TRAJECTORY
# ============================================================================

class LQRController(LeafSystem):
    """
    LQR controller for equilibrium regulation (not trajectory tracking).
    
    Linearizes plant dynamics ONCE at the goal equilibrium and computes
    constant LQR gains for stabilization.
    
    Control Law:
        τ = τ_ff + K·[x_goal - x]
    
    Where:
        - τ_ff: Feedforward torque (gravity compensation at equilibrium)
        - K: LQR gain matrix (CONSTANT, computed at initialization)
        - x_goal: Goal equilibrium state [q_goal, 0]
        - x: Current state [q, v]
    
    This approach works for underactuated systems because we're regulating
    around an equilibrium point (like cart-pole stabilization), not tracking
    arbitrary trajectories.
    """
    
    def __init__(self, plant: MultibodyPlant, model: MultibodyPlant, model_instance,
                 goal_position: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, use_drake_linearization: bool = False):
        """
        Initialize LQR controller for equilibrium regulation.
        
        Args:
            plant: Physics plant (for actuation)
            model: Controller's model (for dynamics computation)
            model_instance: Model instance ID
            goal_position: Target equilibrium configuration [L1, L2, P, R]
            Q: State cost matrix (8x8 for 4 DOF system)
            R: Control cost matrix (2x2 for 2 actuators)
            use_drake_linearization: If True, use Drake's automatic linearization (default)
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.model = model
        self.model_instance = model_instance
        self.goal_position = goal_position
        self.use_drake_linearization = use_drake_linearization
        
        # Dimensions
        self.num_positions = model.num_positions()
        self.num_velocities = model.num_velocities()
        self.num_actuated = 2  # Manipulator joints
        
        # Cost matrices
        self.Q = Q
        self.R = R
        
        # Create model context
        self.model_context = self.model.CreateDefaultContext()
        
        # Compute LQR gain ONCE at equilibrium
        self.K = self._compute_equilibrium_lqr_gain()
        
        # Input port: state from plant
        self.DeclareVectorInputPort("state", BasicVector(self.num_positions + self.num_velocities))
        
        # Output port: control torques
        self.DeclareVectorOutputPort("control", BasicVector(self.num_actuated), self.CalcControl)
        
        linearization_method = "Drake's automatic linearization" if use_drake_linearization else "manual analytical"
        print(colored(f"✓ LQR Equilibrium Regulation Controller Initialized", 'green', attrs=['bold']))
        print(colored(f"  Goal: {np.rad2deg(goal_position)} deg", 'green'))
        print(colored(f"  Linearization: {linearization_method}", 'green'))
        print(colored(f"  State dimension: {self.num_positions + self.num_velocities}", 'green'))
        print(colored(f"  Control dimension: {self.num_actuated}", 'green'))
        print(colored(f"  Q: diag({np.diag(Q)})", 'cyan'))
        print(colored(f"  R: diag({np.diag(R)})", 'cyan'))
    
    def _compute_equilibrium_lqr_gain(self):
        """Compute LQR gain matrix at the goal equilibrium."""
        # Set equilibrium state (zero velocities)
        self.model.SetPositions(self.model_context, self.goal_position)
        self.model.SetVelocities(self.model_context, np.zeros(self.num_velocities))
        
        # Compute dynamics at equilibrium
        M = self.model.CalcMassMatrix(self.model_context)
        C = self.model.CalcBiasTerm(self.model_context)
        
        if self.use_drake_linearization:
            try:
                # Linearize the plant around equilibrium
                linearized_system = Linearize(
                    self.model,
                    self.model_context,
                    input_port_index=self.model.get_actuation_input_port(self.model_instance).get_index(),
                    output_port_index=self.model.get_state_output_port(self.model_instance).get_index(),
                    equilibrium_check_tolerance=1e-3
                )
                
                A = linearized_system.A()
                B = linearized_system.B()
                
            except Exception as e:
                print(colored(f"Warning: Drake linearization failed, using manual approximation", 'yellow'))
                print(colored(f"  Error: {e}", 'yellow'))
                A, B = self._manual_linearization(M, C)
        else:
            A, B = self._manual_linearization(M, C)
        
        # Compute LQR gain
        try:
            K, S = LinearQuadraticRegulator(A, B, self.Q, self.R)
            print(colored(f"✓ LQR gain computed successfully at equilibrium", 'green'))
            print(colored(f"  K shape: {K.shape}", 'cyan'))
            return K
        except Exception as e:
            print(colored(f"✗ LQR failed even at equilibrium!", 'red', attrs=['bold']))
            print(colored(f"  Error: {e}", 'red'))
            print(colored(f"  A shape: {A.shape}, B shape: {B.shape}", 'yellow'))
            print(colored(f"  System may not be stabilizable even at equilibrium", 'yellow'))
            return np.zeros((self.num_actuated, self.num_positions + self.num_velocities))
    
    def CalcControl(self, context, output):
        """Compute LQR control for equilibrium regulation."""
        # Get current state from plant
        state = self.get_input_port(0).Eval(context)
        q = state[0:self.num_positions]
        v = state[self.num_positions:]
        
        # Goal equilibrium state (zero velocities)
        x_goal = np.concatenate([self.goal_position, np.zeros(self.num_velocities)])
        
        # Current state
        x = np.concatenate([q, v])
        
        # State error
        x_error = x_goal - x
        
        # LQR feedback control
        tau_fb = self.K @ x_error
        
        # Feedforward: gravity compensation at goal
        self.model.SetPositions(self.model_context, self.goal_position)
        self.model.SetVelocities(self.model_context, np.zeros(self.num_velocities))
        C_goal = self.model.CalcBiasTerm(self.model_context)
        tau_ff = C_goal[0:self.num_actuated]  # Gravity term for actuated joints
        
        # Total control
        torque = tau_ff + tau_fb
        
        # Set output
        output.SetFromVector(torque)
    
    def _manual_linearization(self, M, C):
        """
        Manual linearization (simplified approximation).
        
        Linearizes: ẋ = [q̇, v̇] = [v, M⁻¹(τ - C)]
        
        Returns approximate A, B matrices assuming:
        - M, C don't vary significantly near equilibrium
        - Ignores ∂M/∂q and ∂C/∂q terms
        """
        M_inv = np.linalg.inv(M)
        
        # State matrix A (simplified)
        A = np.zeros((self.num_positions + self.num_velocities, 
                      self.num_positions + self.num_velocities))
        A[0:self.num_positions, self.num_positions:] = np.eye(self.num_positions)
        # Note: Missing ∂/∂q [M⁻¹(τ - C)] and ∂/∂v [M⁻¹(τ - C)] terms
        
        # Control matrix B
        B = np.zeros((self.num_positions + self.num_velocities, self.num_actuated))
        B[self.num_positions:self.num_positions+self.num_actuated, :] = M_inv[0:self.num_actuated, 0:self.num_actuated]
        
        return A, B


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
                 trajectory_generator: SinusoidalTrajectoryGenerator,
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
            trajectory_generator: Trajectory generator instance
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
        self.trajectory_generator = trajectory_generator
        self.motion_duration = trajectory_generator.motion_duration
        self.control_mode = control_mode
        
        print(colored(f"\n--- ComputedTorqueController Configuration ---", 'yellow', attrs=['bold']))
        print(colored(f"  Control Law: τ = M(q)·[q_ddot_d + Kp·e + Kd·ė] + C(q,q_dot) + g(q)", 'cyan'))
        print(colored(f"  Kp (motion): {self.Kp}", 'cyan'))
        print(colored(f"  Kd (motion): {self.Kd}", 'cyan'))
        print(colored(f"  Kp (hold): {self.Kp_hold}", 'cyan'))
        print(colored(f"  Kd (hold): {self.Kd_hold}", 'cyan'))
        print(colored(f"  Motion duration: {self.motion_duration} s", 'cyan'))
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
        
        # Calculate desired trajectory using trajectory generator
        q_desired, q_dot_desired, q_ddot_desired = self.trajectory_generator.compute_trajectory(t)
        
        # Adjust gains based on phase
        if t < self.motion_duration:
            Kp_current = self.Kp
            Kd_current = self.Kd
        else:
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
    
    def compute_ball_state(self, plant: MultibodyPlant, context):
        """
        Compute ball position and spherical coordinates.
        
        Returns:
            dict with keys:
                - ball_wrt_pivot: [x, y, z] relative to pivot frame (constant for same pendulum angles)
                - ball_wrt_world: [x, y, z] relative to world frame (changes with manipulator)
                - ball_in_ball_frame: [x, y, z] in ball's own frame (always [0, 0, -L])
                - pivot_in_ball_frame: [x, y, z] of pivot as seen from ball frame
                - theta: polar angle from vertical (radians)
                - phi: azimuthal angle in x-y plane (radians)
                - r: radial distance (should equal L)
                - roll_wrt_pivot: roll angle of ball frame w.r.t. pivot frame (radians)
                - pitch_wrt_pivot: pitch angle of ball frame w.r.t. pivot frame (radians)
                - yaw_wrt_pivot: yaw angle of ball frame w.r.t. pivot frame (radians)
                - roll_wrt_world: roll angle of ball frame w.r.t. world frame (radians)
                - pitch_wrt_world: pitch angle of ball frame w.r.t. world frame (radians)
                - yaw_wrt_world: yaw angle of ball frame w.r.t. world frame (radians)
        """
        if not self.pendulum_body:
            return None
        
        ball_frame = self.pendulum_body.body_frame()
        pivot_frame = self.pivot_frame
        ball_offset_in_body = np.array([0.0, 0.0, -self.length])
        
        # Transform from PIVOT frame to ball frame (only depends on pendulum angles!)
        X_PB = plant.CalcRelativeTransform(context, pivot_frame, ball_frame)
        ball_wrt_pivot = X_PB.rotation() @ ball_offset_in_body
        
        # Extract roll-pitch-yaw angles of ball frame relative to pivot frame
        rpy_pivot = RollPitchYaw(X_PB.rotation())
        roll_wrt_pivot = rpy_pivot.roll_angle()
        pitch_wrt_pivot = rpy_pivot.pitch_angle()
        yaw_wrt_pivot = rpy_pivot.yaw_angle()
        
        # Transform from WORLD frame to ball frame (includes manipulator motion)
        X_WB = plant.CalcRelativeTransform(context, plant.world_frame(), ball_frame)
        ball_wrt_world = X_WB.rotation() @ ball_offset_in_body
        
        # Extract roll-pitch-yaw angles of ball frame relative to world frame
        rpy_world = RollPitchYaw(X_WB.rotation())
        roll_wrt_world = rpy_world.roll_angle()
        pitch_wrt_world = rpy_world.pitch_angle()
        yaw_wrt_world = rpy_world.yaw_angle()
        
        # Ball position in ball_frame coordinates (always constant)
        ball_in_ball_frame = ball_offset_in_body  # [0, 0, -L] by definition
        
        # Pivot position as seen from ball frame (inverse perspective)
        X_BP = X_PB.inverse()
        pivot_in_ball_frame = X_BP.translation()
        
        # Compute spherical coordinates (relative to pivot frame)
        x, y, z = ball_wrt_pivot
        r = np.linalg.norm(ball_wrt_pivot)
        theta = np.arccos(z / r) if r > 1e-10 else 0.0  # Polar angle from -z axis
        phi = np.arctan2(y, x)  # Azimuthal angle
        
        return {
            'ball_wrt_pivot': ball_wrt_pivot,
            'ball_wrt_world': ball_wrt_world,
            'ball_in_ball_frame': ball_in_ball_frame,
            'pivot_in_ball_frame': pivot_in_ball_frame,
            'theta': theta,
            'phi': phi,
            'r': r,
            'x': x,
            'y': y,
            'z': z,
            'roll_wrt_pivot': roll_wrt_pivot,
            'pitch_wrt_pivot': pitch_wrt_pivot,
            'yaw_wrt_pivot': yaw_wrt_pivot,
            'roll_wrt_world': roll_wrt_world,
            'pitch_wrt_world': pitch_wrt_world,
            'yaw_wrt_world': yaw_wrt_world,
        }


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
        self.pendulum_spherical_log = []  # Pendulum spherical coords [theta (polar), phi (azimuth)] in radians
        self.pendulum_rpy_pivot_log = []  # Pendulum ball frame RPY angles w.r.t. pivot frame [roll, pitch, yaw] in radians
        
        # Frame visualization
        self.frame_list = []  # List of (frame_name, frame, length) tuples for updating
        
        # Create trajectory generator (shared by controller and simulation logging)
        if CONTROLLER_MODE == 'min-jerk-joint':
            self.trajectory_generator = MinJerkTrajectoryGenerator(
                q_start=MIN_JERK_Q_START[:2],  # Only manipulator joints (first 2)
                q_goal=MIN_JERK_Q_GOAL[:2],    # Only manipulator joints (first 2)
                duration=MIN_JERK_DURATION
            )
        else:
            self.trajectory_generator = SinusoidalTrajectoryGenerator(
                amplitudes=JOINT_MOTION_AMPLITUDE,
                frequencies=JOINT_MOTION_FREQUENCY,
                motion_duration=MANIPULATOR_MOTION_DURATION
            )
        
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
        elif CONTROLLER_MODE == 'min-jerk-joint':
            Kp = np.array([100.0, 100.0])
            Kd = np.array([10.0, 10.0])
        elif CONTROLLER_MODE == 'lqr':
            # LQR uses its own Q and R matrices
            Kp = None
            Kd = None
        elif CONTROLLER_MODE == 'task-space-ofc':
            # Task-space OFC uses internal LQR gains
            Kp = None
            Kd = None
        elif CONTROLLER_MODE in ['ofc-effort', 'ofc-smoothness']:
            # Joint-space OFC uses internal LQR gains
            Kp = None
            Kd = None
        elif CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics':
            # Reduced gains: feedforward handles dynamics, feedback only corrects errors
            Kp = np.array([20.0, 20.0])  # 5x smaller than PD
            Kd = np.array([5.0, 5.0])    # 2x smaller than PD
        else:
            Kp = np.array([100.0, 100.0])
            Kd = np.array([10.0, 10.0])
        
        print(colored(f"\n--- Creating Controller: {CONTROLLER_MODE.upper()} ---", 'yellow', attrs=['bold']))
        if Kp is not None:
            print(colored(f"  Gains: Kp={Kp}, Kd={Kd}", 'cyan'))
        
        if CONTROLLER_MODE == 'pd' or CONTROLLER_MODE == 'min-jerk-joint':
            self.controller = self.builder.AddSystem(
                PDController(self.plant, self.cup_manipulator.model_instance, Kp, Kd, self.trajectory_generator)
            )
            print(colored(f"✓ PDController (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: Computes control torques τ = Kp*(q_d - q) + Kd*(q_dot_d - q_dot)", 'cyan'))
            if CONTROLLER_MODE == 'min-jerk-joint':
                print(colored(f"  Trajectory: Minimum-jerk 5th-order polynomial", 'cyan'))
        
        elif CONTROLLER_MODE == 'lqr':
            # LQR equilibrium regulation controller
            # Create separate model for controller (same as computed torque)
            self.create_model_for_controller()
            model_plant = self.model
            
            # Goal equilibrium configuration
            goal_position = LQR_Q_GOAL
            
            # State cost matrix Q (8x8 for [q(4), v(4)])
            Q = np.diag(np.concatenate([LQR_Q_WEIGHTS, LQR_QDOT_WEIGHTS]))
            
            # Control cost matrix R (2x2)
            R = np.diag(LQR_R_WEIGHTS)
            
            self.controller = self.builder.AddSystem(
                LQRController(
                    plant=self.plant,
                    model=model_plant,
                    model_instance=self.cup_manipulator.model_instance,
                    goal_position=goal_position,
                    Q=Q,
                    R=R
                )
            )
            print(colored(f"✓ LQR Equilibrium Regulation Controller (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Goal: {np.rad2deg(goal_position)}°", 'cyan'))
            print(colored(f"  Controller type: Regulation (not trajectory tracking)", 'cyan'))
        
        elif CONTROLLER_MODE == 'task-space-ofc':
            # Task-space OFC controller
            self.controller = self.builder.AddSystem(
                TaskSpaceOFC(
                    plant=self.plant,
                    model_instance=self.cup_manipulator.model_instance,
                    pivot_start=TASK_SPACE_PIVOT_START,
                    pivot_goal=TASK_SPACE_PIVOT_GOAL,
                    duration=TASK_SPACE_DURATION,
                    mode=TASK_SPACE_MODE
                )
            )
            print(colored(f"✓ TaskSpaceOFC (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: Optimal feedback control in task space (pivot position)", 'cyan'))
            print(colored(f"  Mode: {TASK_SPACE_MODE}", 'cyan'))
            print(colored(f"  Pivot: {TASK_SPACE_PIVOT_START} → {TASK_SPACE_PIVOT_GOAL}", 'cyan'))
        
        elif CONTROLLER_MODE in ['ofc-effort', 'ofc-smoothness']:
            # Joint-space OFC controller
            mode = 'effort' if CONTROLLER_MODE == 'ofc-effort' else 'smoothness'
            q_start_rad = np.deg2rad(np.array(JOINT_SPACE_Q_START + [0.0, 180.0]))  # Add pendulum
            q_goal_rad = np.deg2rad(np.array(JOINT_SPACE_Q_GOAL + [0.0, 180.0]))    # Add pendulum
            
            self.controller = self.builder.AddSystem(
                JointSpaceOFC(
                    plant=self.plant,
                    q_start=q_start_rad,
                    q_goal=q_goal_rad,
                    duration=JOINT_SPACE_DURATION,
                    mode=mode,
                    Ma=JOINT_SPACE_MA,
                    kp=JOINT_SPACE_KP,
                    kd=JOINT_SPACE_KD,
                    tau_filter=JOINT_SPACE_TAU_FILTER,
                    include_pendulum=True  # Full state feedback including pendulum
                )
            )
            print(colored(f"✓ JointSpaceOFC ({mode} mode, SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: Optimal feedback control in joint space", 'cyan'))
            print(colored(f"  Joints: {JOINT_SPACE_Q_START}° → {JOINT_SPACE_Q_GOAL}°", 'cyan'))
        
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
                    trajectory_generator=self.trajectory_generator
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
        def add_frame_triad(meshcat, path, length=0.1, use_custom_colors=False):
            """Add XYZ coordinate frame to Meshcat.
            
            Args:
                meshcat: Meshcat instance
                path: Path for the frame
                length: Length of axes
                use_custom_colors: If True, use cyan/magenta/yellow instead of RGB
            """
            if use_custom_colors:
                # Pivot frame: Cyan, Magenta, Yellow for better visibility
                x_color = Rgba(0.0, 1.0, 1.0, 1.0)  # Cyan
                y_color = Rgba(1.0, 0.0, 1.0, 1.0)  # Magenta
                z_color = Rgba(1.0, 1.0, 0.0, 1.0)  # Yellow
            else:
                # Standard RGB colors
                x_color = Rgba(1.0, 0.0, 0.0, 1.0)  # Red
                y_color = Rgba(0.0, 1.0, 0.0, 1.0)  # Green
                z_color = Rgba(0.0, 0.0, 1.0, 1.0)  # Blue
            
            # X-axis
            meshcat.SetObject(f"{path}/X", Cylinder(radius=length*0.01, length=length),
                            rgba=x_color)
            meshcat.SetTransform(f"{path}/X", 
                               RigidTransform(RotationMatrix.MakeYRotation(np.pi/2), 
                                            [length/2, 0, 0]))
            # Y-axis
            meshcat.SetObject(f"{path}/Y", Cylinder(radius=length*0.01, length=length),
                            rgba=y_color)
            meshcat.SetTransform(f"{path}/Y", 
                               RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2), 
                                            [0, length/2, 0]))
            # Z-axis
            meshcat.SetObject(f"{path}/Z", Cylinder(radius=length*0.01, length=length),
                            rgba=z_color)
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
                use_custom_colors = True  # Use cyan/magenta/yellow for pivot frame
            elif "gimbal" in frame_name.lower() or "pendulum" in frame_name.lower():
                length = 0.10
                use_custom_colors = False
            else:
                length = 0.12
                use_custom_colors = False
            
            # Add frame triad
            path = f"/Frames/{frame_name}"
            add_frame_triad(self.meshcat, path, length=length, use_custom_colors=use_custom_colors)
            
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
        
        # Set manipulator joints
        link1_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
        link2_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
        
        # For min-jerk-joint mode, use start configuration from MIN_JERK_Q_START
        if CONTROLLER_MODE == 'min-jerk-joint':
            link1_joint.set_angle(plant_context, MIN_JERK_Q_START[0])
            link2_joint.set_angle(plant_context, MIN_JERK_Q_START[1])
            print(colored(f"  ✓ Manipulator joints (min-jerk start): link1={np.rad2deg(MIN_JERK_Q_START[0]):.1f}°, link2={np.rad2deg(MIN_JERK_Q_START[1]):.1f}°", 'cyan'))
        elif CONTROLLER_MODE == 'lqr':
            # For LQR mode, use LQR_Q_START configuration
            link1_joint.set_angle(plant_context, LQR_Q_START[0])
            link2_joint.set_angle(plant_context, LQR_Q_START[1])
            print(colored(f"  ✓ Manipulator joints (LQR start): link1={np.rad2deg(LQR_Q_START[0]):.1f}°, link2={np.rad2deg(LQR_Q_START[1]):.1f}°", 'cyan'))
        elif CONTROLLER_MODE == 'task-space-ofc':
            # For task-space OFC, compute initial joint angles from pivot position
            # Use IK to find configuration that reaches TASK_SPACE_PIVOT_START
            # For now, use a reasonable starting configuration
            link1_joint.set_angle(plant_context, np.deg2rad(45.0))
            link2_joint.set_angle(plant_context, np.deg2rad(-90.0))
            print(colored(f"  ✓ Manipulator joints (task-space start): link1=45.0°, link2=-90.0°", 'cyan'))
        elif CONTROLLER_MODE in ['ofc-effort', 'ofc-smoothness']:
            # For joint-space OFC, use start configuration
            link1_joint.set_angle(plant_context, np.deg2rad(JOINT_SPACE_Q_START[0]))
            link2_joint.set_angle(plant_context, np.deg2rad(JOINT_SPACE_Q_START[1]))
            print(colored(f"  ✓ Manipulator joints (joint-space OFC start): link1={JOINT_SPACE_Q_START[0]}°, link2={JOINT_SPACE_Q_START[1]}°", 'cyan'))
        else:
            link1_joint.set_angle(plant_context, 0.0)
            link2_joint.set_angle(plant_context, 0.0)
            print(colored("  ✓ Manipulator joints: link1=0°, link2=0°", 'cyan'))
        
        # Set pendulum initial swing if enabled
        if PENDULUM_ENABLED and self.pendulum:
            pitch_joint = self.plant.GetJointByName("pendulum_pitch", self.cup_manipulator.model_instance)
            roll_joint = self.plant.GetJointByName("pendulum_roll", self.cup_manipulator.model_instance)
            
            # For min-jerk-joint mode, use pendulum angles from config
            if CONTROLLER_MODE == 'min-jerk-joint':
                pitch_joint.set_angle(plant_context, MIN_JERK_Q_START[2])
                roll_joint.set_angle(plant_context, MIN_JERK_Q_START[3])
                print(colored(f"  ✓ Pendulum (min-jerk start): pitch={np.rad2deg(MIN_JERK_Q_START[2]):.1f}°, roll={np.rad2deg(MIN_JERK_Q_START[3]):.1f}°", 'cyan'))
            elif CONTROLLER_MODE == 'lqr':
                # For LQR mode, use LQR_Q_START configuration (0°, 180° hanging)
                pitch_joint.set_angle(plant_context, LQR_Q_START[2])
                roll_joint.set_angle(plant_context, LQR_Q_START[3])
                print(colored(f"  ✓ Pendulum (LQR start): pitch={np.rad2deg(LQR_Q_START[2]):.1f}°, roll={np.rad2deg(LQR_Q_START[3]):.1f}°", 'cyan'))
            elif CONTROLLER_MODE == 'task-space-ofc':
                # For task-space OFC, pendulum starts hanging down
                pitch_joint.set_angle(plant_context, 0.0)
                roll_joint.set_angle(plant_context, np.deg2rad(180.0))
                print(colored(f"  ✓ Pendulum (task-space): pitch=0.0°, roll=180.0° (hanging)", 'cyan'))
            elif CONTROLLER_MODE in ['ofc-effort', 'ofc-smoothness']:
                # For joint-space OFC, pendulum starts hanging down
                pitch_joint.set_angle(plant_context, 0.0)
                roll_joint.set_angle(plant_context, np.deg2rad(180.0))
                print(colored(f"  ✓ Pendulum (joint-space OFC): pitch=0.0°, roll=180.0° (hanging)", 'cyan'))
            else:
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
                
                # Compute desired trajectory using trajectory generator
                q_desired, q_dot_desired, q_ddot_desired = self.trajectory_generator.compute_trajectory(t)
                
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
                        # Use Pendulum3D method to compute all ball state info
                        ball_state = self.pendulum.compute_ball_state(self.plant, plant_context)
                        
                        if ball_state:
                            # Log ball position relative to pivot frame
                            self.pendulum_ball_position_log.append(ball_state['ball_wrt_pivot'].copy())
                            
                            # Verify rigid body constraint: distance should be constant = L
                            self.pendulum_ball_distance_log.append(ball_state['r'])
                            
                            # Log spherical coordinates (theta, phi)
                            self.pendulum_spherical_log.append([ball_state['theta'], ball_state['phi']])
                            
                            # Log RPY angles relative to pivot frame (constant for fixed pendulum angles)
                            self.pendulum_rpy_pivot_log.append([ball_state['roll_wrt_pivot'], 
                                                                ball_state['pitch_wrt_pivot'], 
                                                                ball_state['yaw_wrt_pivot']])
                    
                    # Update frame positions
                    self._update_frame_positions(plant_context)
                
                # Print progress at lower frequency (only at print_interval)
                if next_time >= next_print_time:
                    progress_pct = (next_time / sim_time) * 100
                    print(colored(f"[{next_time:5.2f}s/{sim_time:.0f}s {progress_pct:3.0f}%]", 'yellow'), end=' ')
                    print(f"L1={np.rad2deg(link1_pos):6.1f}° L2={np.rad2deg(link2_pos):6.1f}°", end='')
                    
                    if PENDULUM_ENABLED:
                        # Get latest spherical coordinates and RPY angles
                        if len(self.pendulum_spherical_log) > 0 and len(self.pendulum_rpy_pivot_log) > 0:
                            theta, phi = self.pendulum_spherical_log[-1]
                            rpy_roll, rpy_pitch, rpy_yaw = self.pendulum_rpy_pivot_log[-1]
                            print(f" | P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}° | θ={np.rad2deg(theta):5.1f}° φ={np.rad2deg(phi):6.1f}° | RPY=[{np.rad2deg(rpy_roll):5.1f}°,{np.rad2deg(rpy_pitch):5.1f}°,{np.rad2deg(rpy_yaw):5.1f}°]", end='')
                        elif len(self.pendulum_spherical_log) > 0:
                            theta, phi = self.pendulum_spherical_log[-1]
                            print(f" | P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}° | θ={np.rad2deg(theta):5.1f}° φ={np.rad2deg(phi):6.1f}°", end='')
                        else:
                            print(f" | P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}°", end='')
                    
                    print()  # New line
                    next_print_time += print_interval
                
                current_time = next_time
            
            print(colored("\n✓ Simulation completed successfully!", 'green', attrs=['bold']))
        except Exception as e:
            print(colored(f"\n✗ Simulation error: {e}", 'red', attrs=['bold']))
            import traceback
            traceback.print_exc()
    
    def run_scene_viz(self):
        """Run interactive scene visualization with terminal joint control.
        
        Note: This is a STATIC visualization mode - no physics simulation runs.
        The robot is displayed at the initial configuration and can be manually
        controlled via terminal input.
        """
        print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
        print(colored("Interactive Scene Visualization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan', attrs=['bold']))
        
        print(colored("\nVisualization Mode: Interactive Static Scene", 'yellow'))
        print(colored("  - No physics simulation", 'yellow'))
        print(colored("  - Manual joint control via terminal", 'yellow'))
        print(colored("  - All coordinate frames visible", 'yellow'))
        print(colored("  - Type 'q' to exit\n", 'yellow'))
        
        if not self.meshcat:
            print(colored("\n✗ Visualization not enabled", 'red'))
            return
        
        print(colored(f"\n✓ Meshcat URL: {self.meshcat.web_url()}", 'green', attrs=['bold']))
        print(colored("  👉 Open this URL in your browser to view the scene\n", 'yellow', attrs=['bold']))
        
        # Initialize and force publish
        self.simulator.Initialize()
        diagram = self.simulator.get_system()
        diagram.ForcedPublish(self.context)
        
        # Get plant context
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Print initial state
        joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
        print(colored(f"\nInitial Joint Positions:", 'magenta', attrs=['bold']))
        for name, pos in joint_positions.items():
            print(colored(f"  {name}: {np.rad2deg(pos):+7.2f}° ({pos:+.4f} rad)", 'cyan'))
        
        # Interactive joint control
        print("\n" + "=" * 70)
        print("Interactive Joint Control")
        print("=" * 70)
        
        if PENDULUM_ENABLED:
            print(f"\nEnter joint positions (4 values in degrees, space-separated):")
            print(f"  Format: <link1_base> <link2_link1> <pendulum_pitch> <pendulum_roll>")
            print(f"  Example: 0 45 0 0  (manipulator at 45°, pendulum upright)")
            print(f"  Example: 30 60 20 10 (all joints moved)")
            joint_names = ['link1_base', 'link2_link1', 'pendulum_pitch', 'pendulum_roll']
            expected_count = 4
        else:
            print(f"\nEnter joint positions (2 values in degrees, space-separated):")
            print(f"  Format: <link1_base> <link2_link1>")
            print(f"  Example: 0 45")
            joint_names = ['link1_base', 'link2_link1']
            expected_count = 2
        
        print(f"  Type 'q' or 'quit' to exit")
        print(f"  Type 'frames' to list all coordinate frames (debug)")
        print("=" * 70 + "\n")
        
        # Interactive loop
        try:
            while True:
                # Prompt for input
                user_input = input(f"\nJoint angles (deg) [{', '.join(joint_names)}]: ").strip()
                
                # Check for exit
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\nExiting interactive mode...")
                    break
                
                # Check for frames debug command
                if user_input.lower() == 'frames':
                    print(colored("\n" + "="*70, 'magenta', attrs=['bold']))
                    print(colored("🔬 DEBUG: All Frames in Plant", 'magenta', attrs=['bold']))
                    print(colored("="*70, 'magenta', attrs=['bold']))
                    
                    from pydrake.multibody.tree import FrameIndex
                    world_frame = self.plant.world_frame()
                    
                    print(colored(f"\nTotal frames: {self.plant.num_frames()}", 'yellow'))
                    print(colored(f"{'Frame Name':<35} {'Parent':<20} {'Position [x,y,z]':<30} {'Orientation [RPY]'}", 'cyan', attrs=['bold']))
                    print(colored("-"*120, 'cyan'))
                    
                    for i in range(self.plant.num_frames()):
                        frame = self.plant.get_frame(FrameIndex(i))
                        frame_name = frame.name()
                        
                        try:
                            # Get transform from world to this frame
                            X_WF = self.plant.CalcRelativeTransform(
                                plant_context,
                                world_frame,
                                frame
                            )
                            
                            # Extract position and orientation
                            position = X_WF.translation()
                            rpy = RollPitchYaw(X_WF.rotation())
                            
                            # Get parent body name
                            parent_body = frame.body()
                            parent_name = parent_body.name() if parent_body else "N/A"
                            
                            # Format output
                            pos_str = f"[{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}]"
                            rpy_deg = np.rad2deg([rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()])
                            rpy_str = f"[{rpy_deg[0]:+7.2f}, {rpy_deg[1]:+7.2f}, {rpy_deg[2]:+7.2f}]°"
                            
                            # Color code by frame type
                            if "world" in frame_name.lower():
                                color = 'green'
                            elif "pivot" in frame_name.lower():
                                color = 'yellow'
                            elif "pendulum" in frame_name.lower() or "gimbal" in frame_name.lower():
                                color = 'magenta'
                            else:
                                color = 'white'
                            
                            print(colored(f"{frame_name:<35} {parent_name:<20} {pos_str:<30} {rpy_str}", color))
                            
                        except Exception as e:
                            print(colored(f"{frame_name:<35} ERROR: {e}", 'red'))
                    
                    print(colored("\n" + "="*70, 'magenta', attrs=['bold']))
                    print(colored("Tip: Enter joint angles to continue, or 'q' to quit", 'yellow'))
                    print(colored("="*70 + "\n", 'magenta', attrs=['bold']))
                    continue  # Skip normal processing
                
                # Parse input
                try:
                    values = [float(x.strip()) for x in user_input.split()]
                    
                    if len(values) != expected_count:
                        print(colored(f"❌ Error: Expected {expected_count} values, got {len(values)}. Try again.", 'red'))
                        continue
                    
                    # Convert degrees to radians
                    angles_rad = [np.deg2rad(v) for v in values]
                    
                    # Display what we're about to set
                    print(colored(f"\n→ Setting joints:", 'yellow'))
                    for joint_name, angle_deg, angle_rad in zip(joint_names, values, angles_rad):
                        print(colored(f"    {joint_name}: {angle_deg:+7.2f}° ({angle_rad:+.4f} rad)", 'yellow'))
                    
                    # Update joint positions
                    for joint_name, angle in zip(joint_names, angles_rad):
                        try:
                            joint = self.plant.GetJointByName(joint_name, self.cup_manipulator.model_instance)
                            if isinstance(joint, RevoluteJoint):
                                joint.set_angle(plant_context, angle)
                                print(colored(f"  ✓ Set {joint_name}", 'green'))
                        except Exception as e:
                            print(colored(f"  ⚠ Warning: Could not set joint {joint_name}: {e}", 'red'))
                    
                    # Force publish to update Meshcat visualization
                    diagram.ForcedPublish(self.context)
                    
                    # Update frame positions
                    self._update_frame_positions(plant_context)
                    
                    # Get updated state
                    joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
                    
                    # Calculate spherical coordinates if pendulum enabled
                    ball_state = None
                    if PENDULUM_ENABLED and self.pendulum and self.pendulum.pendulum_body:
                        # Use Pendulum3D method to compute all ball state info
                        ball_state = self.pendulum.compute_ball_state(self.plant, plant_context)
                    
                    # Display updated state (actual values read back from plant)
                    print(colored(f"\n← Actual joint values (read from plant):", 'cyan'))
                    for name, pos in joint_positions.items():
                        print(colored(f"    {name}: {np.rad2deg(pos):+7.2f}° ({pos:+.4f} rad)", 'cyan'))
                    
                    # Display spherical coordinates if pendulum enabled
                    if PENDULUM_ENABLED and self.pendulum and ball_state:
                        print(colored(f"\n📐 Spherical (θ,φ,r):", 'cyan', attrs=['bold']), colored(f"{np.rad2deg(ball_state['theta']):+6.2f}°, {np.rad2deg(ball_state['phi']):+6.2f}°, {ball_state['r']:.4f}m", 'cyan'))
                        print(colored(f"🔄 RPY (pivot):", 'cyan', attrs=['bold']), colored(f"R={np.rad2deg(ball_state['roll_wrt_pivot']):+6.2f}°  P={np.rad2deg(ball_state['pitch_wrt_pivot']):+6.2f}°  Y={np.rad2deg(ball_state['yaw_wrt_pivot']):+6.2f}°", 'cyan'))
                        print(colored(f"📍 Ball (pivot):", 'cyan', attrs=['bold']), colored(f"[{ball_state['x']:+.4f}, {ball_state['y']:+.4f}, {ball_state['z']:+.4f}]m", 'cyan'))
                        print(colored(f"📍 Ball (world):", 'cyan', attrs=['bold']), colored(f"[{ball_state['ball_wrt_world'][0]:+.4f}, {ball_state['ball_wrt_world'][1]:+.4f}, {ball_state['ball_wrt_world'][2]:+.4f}]m", 'cyan'))
                        print(colored(f"\n� Roll-Pitch-Yaw Angles (ball frame w.r.t. PIVOT frame):", 'cyan', attrs=['bold']))
                        print(colored(f"    Roll:                           {np.rad2deg(ball_state['roll_wrt_pivot']):+7.2f}° ({ball_state['roll_wrt_pivot']:+.4f} rad)", 'cyan'))
                        print(colored(f"    Pitch:                          {np.rad2deg(ball_state['pitch_wrt_pivot']):+7.2f}° ({ball_state['pitch_wrt_pivot']:+.4f} rad)", 'cyan'))
                        print(colored(f"    Yaw:                            {np.rad2deg(ball_state['yaw_wrt_pivot']):+7.2f}° ({ball_state['yaw_wrt_pivot']:+.4f} rad)", 'cyan'))
                        print(colored(f"\n�📍 Ball Position (relative to PIVOT frame - constant for same pendulum angles):", 'cyan', attrs=['bold']))
                        print(colored(f"    [x,y,z]:                        [{ball_state['x']:+.4f}, {ball_state['y']:+.4f}, {ball_state['z']:+.4f}] m", 'cyan'))
                        print(colored(f"\n📍 Ball Position (relative to WORLD frame - changes with manipulator):", 'cyan', attrs=['bold']))
                        print(colored(f"    [x,y,z]:                        [{ball_state['ball_wrt_world'][0]:+.4f}, {ball_state['ball_wrt_world'][1]:+.4f}, {ball_state['ball_wrt_world'][2]:+.4f}] m", 'cyan'))
                        print(colored(f"\n📍 Ball Position (in ball_frame coordinates):", 'cyan', attrs=['bold']))
                        print(colored(f"    [x,y,z]:                        [{ball_state['ball_in_ball_frame'][0]:+.4f}, {ball_state['ball_in_ball_frame'][1]:+.4f}, {ball_state['ball_in_ball_frame'][2]:+.4f}] m (constant)", 'cyan'))
                        print(colored(f"\n📍 Pivot Position (as seen from ball frame):", 'cyan', attrs=['bold']))
                        print(colored(f"    [x,y,z]:                        [{ball_state['pivot_in_ball_frame'][0]:+.4f}, {ball_state['pivot_in_ball_frame'][1]:+.4f}, {ball_state['pivot_in_ball_frame'][2]:+.4f}] m", 'cyan'))
                    
                    # Check for discrepancies
                    print(colored(f"\n🔍 Verification (set vs. read):", 'magenta'))
                    for joint_name, set_value in zip(joint_names, values):
                        if joint_name in joint_positions:
                            read_value = np.rad2deg(joint_positions[joint_name])
                            diff = read_value - set_value
                            if abs(diff) > 0.01:  # More than 0.01° difference
                                print(colored(f"  ⚠ {joint_name}: set={set_value:+7.2f}° → read={read_value:+7.2f}° (Δ={diff:+.2f}°)", 'yellow'))
                            else:
                                print(colored(f"  ✓ {joint_name}: {set_value:+7.2f}° (match)", 'green'))
                    
                except ValueError as e:
                    print(colored(f"❌ Error: Invalid input. Please enter {expected_count} numbers separated by spaces.", 'red'))
                    print(f"   Example: {'0 45 20 10' if PENDULUM_ENABLED else '0 45'}")
                except Exception as e:
                    print(colored(f"❌ Error: {e}", 'red'))
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print(colored("\n\n✓ Scene visualization closed by user", 'green'))
        
        print(colored("\n" + "="*70, 'green'))
        print(colored("Scene visualization complete!", 'green', attrs=['bold']))
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
            
            # Row 4: Ball center position (X, Y, Z) vs time - PIVOT FRAME coordinates
            ax = fig.add_subplot(gs[3, :])
            ax.plot(time, pendulum_ball_pos[:, 0], label='X (pivot frame)', 
                   color='#2E86AB', linewidth=1.5)
            ax.plot(time, pendulum_ball_pos[:, 1], label='Y (pivot frame)', 
                   color='#A23B72', linewidth=1.5)
            ax.plot(time, pendulum_ball_pos[:, 2], label='Z (pivot frame)', 
                   color='#06D6A0', linewidth=1.5)
            ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                      linewidth=1.5, alpha=0.5, label='Hold Start')
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Position in Pivot Frame (m)', fontsize=11)
            ax.set_title('Pendulum Ball Position in Pivot Frame (Should be constant for fixed pendulum angles)', fontsize=12, fontweight='bold')
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
