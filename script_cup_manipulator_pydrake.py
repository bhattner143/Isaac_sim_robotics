"""
Cup Manipulator - PyDrake Scene Visualization

This is a PyDrake visualization script for the cup manipulator imported from Onshape.
It demonstrates visualization and basic simulation of a 3-DOF manipulator.

Demonstrates:
1. Custom URDF robot import via Drake's URDF parser
2. 3-DOF manipulator with revolute joints
3. Scene visualization with Meshcat
4. Multi-body dynamics simulation with Drake

System:
- Base mount: Fixed base platform
- Link 1: First link connected to base
- Link 2: Second link 
- Cup end-effector: Attached to link 2
"""

# ============================================================================
# IMPORTS: Standard Python Libraries and Drake
# ============================================================================
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
from datetime import datetime
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    CoulombFriction,
    RevoluteJoint,
    PrismaticJoint,
    SpatialVelocity,
    SpatialInertia,
    UnitInertia,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    
    # Scene graph
    SceneGraph,
    
    # Geometry
    Cylinder,
    Sphere,
    
    # Controllers
    TrajectorySource,
    PassThrough,
    LogVectorOutput,
    
    # Mathematical utilities
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,
    
    # Time
    AbstractValue,
    BasicVector,
    
    # Frames
    Frame,
    FixedOffsetFrame,
)

# Custom robot types
from robot_types import (
    Pose,
    JointConfig,
    ManipulatorConfig,
    SimulationConfig,
    VisualizationConfig,
    SceneConfig,
    PendulumConfig,
    create_cup_manipulator_config,
    create_ball_config,
    create_pendulum_config,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["scene-viz", "simulation", "joint-motion", "run-all-jts", "min-jerk-joint"],
    default="simulation",
    help="Simulation mode: 'scene-viz' (static), 'simulation' (physics), 'joint-motion' (animated joint movement), 'run-all-jts' (interactive terminal control of all joints), 'min-jerk-joint' (minimum-jerk joint-space motion)",
)
parser.add_argument(
    "--visualize",
    type=bool,
    default=True,
    help="Enable Meshcat visualization",
)
parser.add_argument(
    "--interactive",
    type=bool,
    default=True,
    help="Enable interactive play/pause/repeat controls in Meshcat",
)
parser.add_argument(
    "--plot_frames",
    type=bool,
    default=True,
    help="Plot coordinate frames (world, pivot, gimbal) in Meshcat",
)
args, _ = parser.parse_known_args()

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Cup Manipulator Configuration ---
CUP_MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    # Initial joint angles: [link1_base, link2_link1] in radians
    # 2-DOF manipulator - pendulum added programmatically
    joint_angles=(0.0, 0.0),
    # Damping: resist motion, dissipate energy [link1, link2]
    damping=(0.0, 0.0),
    # Stiffness: spring-like resistance to deflection [link1, link2]
    stiffness=(0.0, 0.0),
    # Friction: Coulomb friction [link1, link2]
    friction=(0.05, 0.05),
)

# --- Visualization Configuration ---
VISUALIZATION_CONFIG = VisualizationConfig(
    enabled=args.visualize,
    plot_frames=args.plot_frames,
    interactive=args.interactive,
    realtime_rate=0.5,  # 1.0 = real-time, 0.5 = half speed (slower for better observation)
    update_every_step=True,  # Update Meshcat every simulation step
    print_interval=0.25,  # Print status every N seconds
    show_frames=False,
    show_contact_forces=True,
    show_hydroelastic=True,
)

# --- Simulation Configuration ---
SIMULATION_CONFIG = SimulationConfig(
    mode=args.mode,  # Simulation mode from command-line: 'scene-viz', 'simulation', 'joint-motion', 'run-all-jts', 'min-jerk-joint'
    timestep=0.001,  # 1 kHz simulation
    simulation_time=15.0,  # seconds - longer to see ball swing dynamics
    gravity=(0.0, 0.0, -9.81),
    visualization=VISUALIZATION_CONFIG,
)

# --- Pendulum Configuration (added programmatically) ---
PENDULUM_ENABLED = True  # Set to False to disable pendulum
PENDULUM_CONFIG = create_pendulum_config(
    mass=0.5,  # kg
    length=0.2,  # meters (from pivot to COM)
    radius=0.05,  # meters (ball radius)
    damping=0.1,  # Very low damping for free swinging
    attachment_point=(-1.2545, 0.0, -0.188125),  # Ball center on link2 (from URDF)
    initial_pitch=0.0,  # degrees - initial swing angle from vertical (equilibrium at 180°)
    initial_roll=0.0,  # degrees - initial roll angle (equilibrium at -180°)
    name="pendulum"
)

# --- Joint Motion Parameters ---
# For manipulator joints only (link1_base, link2_link1)
# Simulation modes:
# - 'scene-viz': Static visualization, no physics
# - 'simulation': Full physics simulation with PD control on manipulator joints
# - 'joint-motion': Smooth joint motion demonstration
# - 'run-all-jts': Interactive terminal control of both joints
# - 'min-jerk-joint': Minimum-jerk joint-space motion
# 
# Note: This is now a 2-DOF manipulator (link1_base, link2_link1)
# The ball is rigidly attached to link2
# Increased amplitude and frequency to excite ball pendulum motion
JOINT_MOTION_AMPLITUDE = [np.pi/3, np.pi/2.5]  # Amplitude for manipulator joints (radians) - 60°, 72°
JOINT_MOTION_FREQUENCY = [0.8, 0.6]  # Frequency for manipulator joints (Hz) - faster motion

# --- Manipulator Motion Duration ---
MANIPULATOR_MOTION_DURATION = 3.0  # seconds - manipulator moves for this duration, then stops to let ball settle

# --- Minimum-Jerk Joint-Space Parameters ---
MIN_JERK_DURATION = 4.0  # seconds
MIN_JERK_TARGET_ANGLES = (np.deg2rad(45.0), np.deg2rad(-30.0))  # [link1_base, link2_link1]

# =========================================================================
# TRAJECTORY UTILITIES
# =========================================================================

def min_jerk_profile(t: float, T: float) -> Tuple[float, float, float]:
    """
    Minimum-jerk time-scaling profile.

    Returns:
        h(s), hdot(s), hddot(s) where s = t/T (clamped to [0, 1]).
    """
    if T <= 0:
        return 1.0, 0.0, 0.0
    s = np.clip(t / T, 0.0, 1.0)
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    hdot = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    hddot = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)
    return h, hdot, hddot

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
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint.num_velocities() > 0 and joint_name in self.config.joint_configs:
                config = self.config.joint_configs[joint_name]
                
                if hasattr(joint, 'set_default_damping_vector') and config.damping > 0:
                    joint.set_default_damping_vector([config.damping])
    
    def set_initial_positions(self, plant: MultibodyPlant, context):
        """Set initial joint positions from configuration."""
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint_name in self.config.joint_configs:
                position = self.config.joint_configs[joint_name].position
                
                if isinstance(joint, RevoluteJoint):
                    joint.set_angle(context, position)
                elif isinstance(joint, PrismaticJoint):
                    joint.set_translation(context, position)



# ============================================================================
# CUP MANIPULATOR CLASS
# ============================================================================

class CupManipulator(RobotBase):
    """
    3-DOF Cup Manipulator for Drake.
    
    Manages:
    - Three revolute joints for spatial manipulation
    - End-effector (cup) frame computation
    - Joint control and monitoring
    """
    
    def __init__(self, config: ManipulatorConfig):
        super().__init__(config)
    
    def get_joint_positions(self, plant: MultibodyPlant, context) -> Dict[str, float]:
        """Get current joint positions as a dictionary."""
        positions = {}
        # Get joints from the cup manipulator model instance
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        
        # Also get gimbal joints if they exist (they're in the world instance)
        if PENDULUM_ENABLED:
            try:
                gimbal_pitch = plant.GetJointByName("pendulum_pitch", self.model_instance)
                positions['pendulum_pitch'] = gimbal_pitch.get_angle(context)
            except:
                pass
            try:
                gimbal_roll = plant.GetJointByName("pendulum_roll", self.model_instance)
                positions['pendulum_roll'] = gimbal_roll.get_angle(context)
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
            # Get the cup link (link2 - the cup is attached to link2)
            cup_body = plant.GetBodyByName("link2", self.model_instance)
            world_frame = plant.world_frame()
            cup_frame = cup_body.body_frame()
            
            # Get transform from cup frame to world frame
            X_WC = plant.CalcRelativeTransform(context, world_frame, cup_frame)
            position = X_WC.translation()
            
            return position
        except Exception as e:
            print(f"Warning: Could not get end effector position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def apply_torques(self, plant: MultibodyPlant, context, link1_torque: float, link2_torque: float):
        """Apply torques to the two manipulator joints.
        
        Args:
            plant: The MultibodyPlant
            context: The plant context
            link1_torque: Torque for link1_base joint (N·m)
            link2_torque: Torque for link2_link1 joint (N·m)
        """
        # Actuators are ordered by the order they were added: [link1_base, link2_link1]
        actuator_forces = np.array([link1_torque, link2_torque])
        plant.get_actuation_input_port().FixValue(context, actuator_forces)

# ============================================================================
# PENDULUM 3D CLASS
# ============================================================================

class Pendulum3D:
    """
    3D Pendulum with 2-DOF gimbal joints (pitch and roll).
    
    Attaches to a parent body at a specified attachment point and creates
    a pendulum that can swing in 3D space.
    """
    
    def __init__(self, config: PendulumConfig):
        """
        Initialize pendulum from configuration.
        
        Args:
            config: PendulumConfig dataclass with all pendulum parameters
        """
        self.config = config
        self.mass = config.mass
        self.length = config.length
        self.radius = config.radius
        self.damping = config.damping
        self.attachment_point = config.attachment_point
        self.name = config.name
        
        # Bodies and frames created during attachment
        self.pivot_frame = None
        self.pitch_parent_frame = None  # Shifted frame for pitch joint (if shifting enabled)
        self.gimbal1_body = None
        self.pendulum_body = None
        self.pitch_joint = None
        self.roll_joint = None
    


    def attach_to_body(self,  
                      plant: MultibodyPlant,
                      parent_body,
                      model_instance,
                      pivot_rotation: Optional[RotationMatrix] = None,
                      shifting: bool = True,
                      shift_pitch_by_pi: bool = True,
                      shift_roll_by_pi: bool = False,
                    ):
        """
        Attach pendulum to a parent body.
        
        COORDINATE FRAME HIERARCHY:
        ===========================
        World Frame (W)
            └─> Link2 Frame (L2) - parent_body
                └─> Pivot Frame (P) = L2 rotated by pivot_rotation (180° about X) + translated by attachment_point
                    └─> PitchParent Frame = P rotated by 180° about X (if shifting enabled)
                        └─> Gimbal1 Body Frame (G1) - pitch joint rotates about +Y axis
                            └─> Ball Body Frame (B) - roll joint rotates about +X axis
                                └─> Ball COM at [0, 0, -L] in B frame
        
        JOINT ANGLES ARE MEASURED IN THEIR PARENT FRAMES:
        - pendulum_pitch: Rotation about +Y axis in PitchParent frame (or Pivot frame if not shifting)
        - pendulum_roll: Rotation about +X axis in Gimbal1 frame
        
        The "angle from vertical" shown in terminal is computed in WORLD FRAME:
        - It measures the angle between the pendulum vector (pivot to COM) and world -Z direction
        - This is a geometric angle independent of joint coordinates
        
        Args:
            plant: Drake MultibodyPlant (before finalization)
            parent_body: Parent body to attach to
            model_instance: Model instance to add pendulum bodies to
            pivot_rotation: Optional rotation of pivot frame relative to parent
        """

        # Default: no rotation
        # if pivot_rotation is None:
        roll  = np.deg2rad(0)   # rotation about X
        pitch = np.deg2rad(0)   # rotation about Y
        yaw   = np.deg2rad(0)   # rotation about Z

        pivot_rotation = RotationMatrix(RollPitchYaw(roll, pitch, yaw))
        
        # Create pivot frame on parent body at attachment point
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
            f"{self.name}_gimbal1", 
            model_instance, 
            gimbal1_inertia
        )

        pitch_parent_frame = None
        if shifting:
            print(colored(f"✓ Applying zero-shift of π to pitch joint for better initial pendulum motion", 'yellow'))
            print(colored(f"  (If the pendulum swings in the opposite direction than expected, try setting shift_pitch_negative to False)", 'yellow'))  
            #Pitch joint (about +Y in pivot frame), with optional zero shift by pi
            pitch_parent_frame = self.pivot_frame
            if shift_pitch_by_pi:
                sign = 1.0
                pitch_parent_frame = plant.AddFrame(
                    FixedOffsetFrame(
                        name=f"{self.name}_pitch_parent_frame",
                        P=self.pivot_frame,
                        X_PF=RigidTransform(
                            RotationMatrix.MakeYRotation(0),
                            [0.0, 0.0, 0.0],
                        ),
                        model_instance=model_instance,
                    )
                )
                # Store shifted frame for visualization
                self.pitch_parent_frame = pitch_parent_frame
        # Add pitch joint (Y-axis rotation in pivot frame)
        self.pitch_joint = plant.AddJoint(
        RevoluteJoint(
            name=f"{self.name}_pitch",
            frame_on_parent= self.pitch_parent_frame if shifting else self.pivot_frame,
            frame_on_child=self.gimbal1_body.body_frame(),
            axis=[0.0, 1.0, 0.0],
            damping=self.damping,
        )
        )         

        # Create pendulum body with proper inertia
        m = float(self.mass)
        r = float(self.radius)
        L = float(self.length)
        
        # Ball inertia about its COM (solid sphere): I = (2/5)*m*r²
        I_ball_com = (2.0 / 5.0) * m * (r ** 2)
        
        # Apply parallel axis theorem: I_pivot = I_com + m*d²
        I_pivot_x = I_ball_com + m * (L ** 2)  # About X axis
        I_pivot_y = I_ball_com + m * (L ** 2)  # About Y axis
        I_pivot_z = I_ball_com                  # About Z axis (along rod)
        
        # Unit inertia about body origin (pivot)
        G_ball = UnitInertia(I_pivot_x / m, I_pivot_y / m, I_pivot_z / m)
        
        pendulum_inertia = SpatialInertia(
            mass=m,
            p_PScm_E=[0.0, 0.0, -L],   # COM is -Z in pendulum body frame
            G_SP_E=G_ball,
        )
        self.pendulum_body = plant.AddRigidBody(
            f"{self.name}_ball", 
            model_instance, 
            pendulum_inertia
        )
        
        # Add roll joint (X-axis rotation in gimbal1 frame)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.name}_roll",
                frame_on_parent=self.gimbal1_body.body_frame() if shifting else self.gimbal1_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.damping,
            )
        )

        # Print all rotation matrices of parent frames
        print(colored(f"\n{'='*70}", 'cyan'))
        print(colored(f"ROTATION MATRICES OF PARENT FRAMES:", 'cyan', attrs=['bold']))
        print(colored(f"{'='*70}", 'cyan'))
        
        # Pivot frame transform (X_parent_pivot)
        print(colored(f"\n1. PIVOT FRAME (relative to parent body frame):", 'yellow', attrs=['bold']))
        print(colored(f"   Rotation Matrix (pivot_rotation):", 'yellow'))
        print(colored(f"{pivot_rotation.matrix()}", 'green'))
        print(colored(f"   Translation (attachment_point): {self.attachment_point}", 'yellow'))
        
        # Pitch parent frame (if it exists)
        if pitch_parent_frame is not None:
            print(colored(f"\n2. PITCH PARENT FRAME (relative to pivot frame):", 'yellow', attrs=['bold']))
            print(colored(f"   Frame Name: {pitch_parent_frame.name()}", 'yellow'))
            print(colored(f"   Rotation Matrix: MakeYRotation(0) =", 'yellow'))
            print(colored(f"{RotationMatrix.MakeYRotation(0).matrix()}", 'green'))
            print(colored(f"   Translation: [0.0, 0.0, 1.0]", 'yellow'))
        else:
            print(colored(f"\n2. PITCH PARENT FRAME: Using pivot frame directly (no offset)", 'yellow'))
        
        print(colored(f"\n{'='*70}\n", 'cyan'))
        
        # Add visual geometry
        self._add_visual_geometry(plant, L, r)
        
        # Add collision geometry
        self._add_collision_geometry(plant, L, r)
    
    def _add_visual_geometry(self, plant: MultibodyPlant, L: float, r: float):
        """Add visual geometry for pendulum rod and ball."""
        # Rod: centered halfway down the -Z direction
        plant.RegisterVisualGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L / 2.0]),
            Cylinder(radius=0.001, length=L),
            f"{self.name}_rod_visual",
            [0.6, 0.4, 0.2, 1.0],  # Brown color
        )
        
        # Ball: at the COM location
        plant.RegisterVisualGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L]),
            Sphere(r),
            f"{self.name}_ball_visual",
            [0.8, 0.2, 0.2, 1.0],  # Red color
        )
    
    def _add_collision_geometry(self, plant: MultibodyPlant, L: float, r: float):
        """Add collision geometry for pendulum ball."""
        plant.RegisterCollisionGeometry(
            self.pendulum_body,
            RigidTransform([0.0, 0.0, -L]),
            Sphere(r),
            f"{self.name}_ball_collision",
            CoulombFriction(0.3, 0.2),
        )
    
    def set_initial_swing(self, context, pitch_angle: float = 0.0, roll_angle: float = 0.0):
        """
        Set initial swing angles for the pendulum.
        
        Args:
            context: Drake context
            pitch_angle: Initial pitch angle (radians)
            roll_angle: Initial roll angle (radians)
        """
        if self.pitch_joint:
            self.pitch_joint.set_angle(context, pitch_angle)
        if self.roll_joint:
            self.roll_joint.set_angle(context, roll_angle)



# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
    Scene Manager for Drake simulation.
    
    RESPONSIBILITIES:
    1. Setup: Create MultibodyPlant, add robot
    2. Initialization: Finalize plant, create simulator
    3. Execution: Run different simulation modes
    4. Visualization: Set up Meshcat visualization
    5. Data logging: Record and plot simulation results
    """
    
    def __init__(self, cup_manipulator_config: ManipulatorConfig, simulation_config: SimulationConfig, ball_config: Optional[ManipulatorConfig] = None):
        """Initialize scene manager."""
        self.cup_manipulator_config = cup_manipulator_config
        self.simulation_config = simulation_config
        self.ball_config = ball_config
        
        # Drake objects
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.meshcat = None
        self.simulator = None
        self.context = None
        
        # Robots
        self.cup_manipulator: Optional[CupManipulator] = None
        
        # Data logging
        self.time_log = []
        self.joint_positions_log = []
        self.joint_velocities_log = []
        self.ee_position_log = []
        self.ball_position_log = []
        
        print("\n" + "=" * 70)
        print("Drake Scene Manager Initialized")
        print("=" * 70)


    def setup_drake_system(self):
        """Setup Drake's MultibodyPlant and load robots."""
        print(colored("\n[1/4] Setting up Drake system...", 'blue', attrs=['bold']))
        
        # Create diagram builder
        self.builder = DiagramBuilder()
        
        # Add MultibodyPlant and SceneGraph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=self.simulation_config.timestep
        )
        
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
        
        # Ball gimbal info
        print(colored("\n--- Ball Gimbal Joints (2-DOF) ---", 'yellow', attrs=['bold']))
        print(colored(f"✓ Joints loaded from URDF: link1_base, link2_link1", 'green'))
        print(colored(f"✓ Ball is rigidly attached to link2 (no gimbal joints)", 'green'))
        
        # Add actuators
        print(colored("\n--- Adding Actuators ---", 'yellow', attrs=['bold']))
        for joint_name in ["link1_base", "link2_link1"]:
            joint = self.plant.GetJointByName(joint_name, self.cup_manipulator.model_instance)
            self.plant.AddJointActuator(joint_name, joint)
        print(colored(f"✓ Added actuators to: link1_base, link2_link1", 'green'))
        print(colored(f"✓ Actuators added: link1_base, link2_link1", 'green'))
        
        # Set joint properties
        print(colored("\n--- Setting Joint Properties ---", 'yellow', attrs=['bold']))
        self.cup_manipulator.set_joint_properties(self.plant)
        
        # Add pendulum if enabled
        if PENDULUM_ENABLED:
            print(colored("\n--- Adding Programmatic Pendulum ---", 'yellow', attrs=['bold']))
            self.pendulum = Pendulum3D(PENDULUM_CONFIG)
            
            link2_body = self.plant.GetBodyByName("link2", self.cup_manipulator.model_instance)
            pivot_rotation = RotationMatrix.MakeXRotation(np.pi)  # 180° rotation about X
            
            self.pendulum.attach_to_body(
                plant=self.plant,
                parent_body=link2_body,
                model_instance=self.cup_manipulator.model_instance,
                pivot_rotation=pivot_rotation
            )
            
            print(colored(f"✓ Added 3D pendulum to link2", 'green'))
            print(colored(f"  - Attachment point (link2 frame): {PENDULUM_CONFIG.attachment_point}", 'yellow'))
            print(colored(f"  - Mass: {PENDULUM_CONFIG.mass} kg, length: {PENDULUM_CONFIG.length} m, radius: {PENDULUM_CONFIG.radius} m", 'yellow'))
            print(colored(f"  - Joints: {PENDULUM_CONFIG.name}_pitch (Y-axis), {PENDULUM_CONFIG.name}_roll (X-axis)", 'yellow'))
        else:
            self.pendulum = None
        
        # Set joint properties BEFORE finalization
        print(colored("\n--- Setting Joint Properties ---", 'yellow', attrs=['bold']))
        self.cup_manipulator.set_joint_properties(self.plant)
        
        # Load ball if configured
        # if self.ball_config:
        #     print("\n--- Loading Ball ---")
        #     self.ball = Ball(self.ball_config)
        #     self.ball.load_urdf_to_plant(self.plant, parser)
        
        # Set gravity
        gravity_field = self.plant.mutable_gravity_field()
        gravity_field.set_gravity_vector(list(self.simulation_config.gravity))
        
        # Finalize plant
        print(colored("\n--- Finalizing Plant ---", 'yellow', attrs=['bold']))
        self.plant.Finalize()
        print(colored(f"✓ Plant finalized with {self.plant.num_positions()} positions and {self.plant.num_velocities()} velocities", 'green', attrs=['bold']))
        
        # Initialize robot state
        self.cup_manipulator.initialize_state(self.plant)
        
        # Setup visualization
        if SIMULATION_CONFIG.visualization.enabled:
            print(colored("\n--- Setting up Meshcat Visualization ---", 'yellow', attrs=['bold']))
            self.meshcat = StartMeshcat()
            visualizer_params = MeshcatVisualizerParams()
            # Enable contact visualization to debug collision issues
            visualizer_params.show_hydroelastic = SIMULATION_CONFIG.visualization.show_hydroelastic
            visualizer_params.show_contact_forces = SIMULATION_CONFIG.visualization.show_contact_forces
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder, self.scene_graph, self.meshcat, visualizer_params
            )
            print(colored(f"✓ Meshcat visualization ready at: {self.meshcat.web_url()}", 'cyan'))
            print(colored(f"✓ Contact visualization enabled (hydroelastic + contact forces)", 'cyan'))
        
        print(colored("\n✓ Drake system setup complete", 'green', attrs=['bold']))
    
    def _add_frame_visualizations(self, context):
        """Add coordinate frame visualizations to Meshcat after plant is finalized."""
        if not SIMULATION_CONFIG.visualization.plot_frames or not SIMULATION_CONFIG.visualization.enabled or not self.meshcat:
            return
            
        print(colored("\n--- Adding Frame Visualizations ---", 'yellow', attrs=['bold']))
        
        # Helper function to create a coordinate frame triad
        def add_frame_triad(meshcat, path, length=0.1):
            """Add RGB coordinate frame axes at the given path."""
            from pydrake.geometry import Cylinder as GeoCylinder, Rgba
            radius = length * 0.02
            
            # X-axis (Red)
            meshcat.SetObject(f"{path}/x_axis", GeoCylinder(radius, length), Rgba(1, 0, 0, 0.8))
            meshcat.SetTransform(f"{path}/x_axis", 
                RigidTransform(RotationMatrix.MakeYRotation(np.pi/2), [length/2, 0, 0]))
            
            # Y-axis (Green)
            meshcat.SetObject(f"{path}/y_axis", GeoCylinder(radius, length), Rgba(0, 1, 0, 0.8))
            meshcat.SetTransform(f"{path}/y_axis",
                RigidTransform(RotationMatrix.MakeXRotation(np.pi/2), [0, length/2, 0]))
            
            # Z-axis (Blue)
            meshcat.SetObject(f"{path}/z_axis", GeoCylinder(radius, length), Rgba(0, 0, 1, 0.8))
            meshcat.SetTransform(f"{path}/z_axis",
                RigidTransform([0, 0, length/2]))
        
        # Add world frame at origin
        add_frame_triad(self.meshcat, "/Frames/World", length=0.20)
        self.meshcat.SetTransform("/Frames/World", RigidTransform())
        print(colored("  ✓ World frame (origin)", 'cyan'))
        
        # Store frame info for updates
        self.frame_list = []
        
        # Loop through all frames in the plant and add them
        from pydrake.multibody.tree import FrameIndex
        for i in range(self.plant.num_frames()):
            frame = self.plant.get_frame(FrameIndex(i))
            frame_name = frame.name()
            
            # Skip world frame (already added)
            if frame_name == "world":
                continue
            
            # Determine frame length based on name
            if "pivot" in frame_name.lower():
                length = 0.12
            elif "pitch_parent" in frame_name.lower():
                length = 0.11
            elif "gimbal" in frame_name.lower():
                length = 0.10
            elif "ball" in frame_name.lower():
                length = 0.14
            else:
                length = 0.15  # Default for body frames
            
            # Add frame visualization
            frame_path = f"/Frames/{frame_name}"
            add_frame_triad(self.meshcat, frame_path, length=length)
            self.frame_list.append((frame_name, frame, length))
            print(colored(f"  ✓ {frame_name} frame", 'cyan'))
        
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
            try:
                # Get frame pose in world
                X_WF = frame.CalcPoseInWorld(context)
                self.meshcat.SetTransform(f"/Frames/{frame_name}", X_WF)
            except Exception as e:
                # Skip frames that can't be evaluated
                continue
    
    
    def create_simulator(self):
        """Build diagram and create simulator."""
        print(colored("\n[2/4] Creating simulator...", 'blue', attrs=['bold']))
        diagram = self.builder.Build()
        self.simulator = Simulator(diagram)
        self.context = self.simulator.get_mutable_context()
        print(colored("✓ Simulator created", 'green'))
        
        # Add frame visualizations after simulator is created
        if SIMULATION_CONFIG.visualization.plot_frames and SIMULATION_CONFIG.visualization.enabled:
            plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
            self._add_frame_visualizations(plant_context)
    
    def set_initial_conditions(self):
        """Set initial joint positions, velocities, and robot pose."""
        print(colored("\n[3/4] Setting initial conditions...", 'blue', attrs=['bold']))
        
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Set initial positions using the config
        self.cup_manipulator.set_initial_positions(self.plant, plant_context)
        
        # Give pendulum an initial swing angle if enabled
        if PENDULUM_ENABLED and self.pendulum:
            # Set initial pendulum swing angle
            # Both modes use the same equilibrium values (hanging down)
            # Equilibrium (hanging straight down): pitch=+180°, roll=-180°
            if SIMULATION_CONFIG.mode == "simulation":
                # Simulation: start at configured initial angles
                initial_pitch = np.deg2rad(PENDULUM_CONFIG.initial_pitch)  
                initial_roll = np.deg2rad(PENDULUM_CONFIG.initial_roll)
            else:
                # Scene-viz: start at equilibrium (hanging down)
                initial_pitch = np.deg2rad(180)
                initial_roll = np.deg2rad(-180)
            
            self.pendulum.set_initial_swing(plant_context, pitch_angle=initial_pitch, roll_angle=initial_roll)
            print(colored(f"  ✓ pendulum_pitch: {np.rad2deg(initial_pitch):.3f}° (equilibrium at +180°)", 'cyan'))
            print(colored(f"  ✓ pendulum_roll: {np.rad2deg(initial_roll):.3f}° (equilibrium at -180°)", 'cyan'))
        
        # Print actual joint positions (from URDF, not config)
        joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
        for joint_name, position in joint_positions.items():
            print(colored(f"  ✓ {joint_name}: {np.rad2deg(position):.3f}°", 'cyan'))
        
        
        print(colored(f"✓ Initial joint positions set", 'green'))
    
    def log_data(self, time_s: float):
        """Log simulation data."""
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Log time
        self.time_log.append(time_s)
        
        # Log joint positions (convert dict to list of values)
        joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
        self.joint_positions_log.append(list(joint_positions.values()))
        
        # Log joint velocities (convert dict to list of values)
        joint_velocities = self.cup_manipulator.get_joint_velocities(self.plant, plant_context)
        self.joint_velocities_log.append(list(joint_velocities.values()))
        
        # Log end effector position
        ee_pos = self.cup_manipulator.get_end_effector_position(self.plant, plant_context)
        self.ee_position_log.append(ee_pos)
        
        # Log ball position if present
        # if self.ball:
        #     ball_pos = self.ball.get_position(self.plant, plant_context)
        #     self.ball_position_log.append(ball_pos)
        
        # Update frame positions
        self._update_frame_positions(plant_context)
    
    def plot_results(self):
        """Plot simulation results."""
        if not self.time_log:
            print("No data to plot")
            return
        
        print("\n[Plotting Results]")
        
        # Convert lists to numpy arrays
        time = np.array(self.time_log)
        joint_pos = np.array(self.joint_positions_log)
        joint_vel = np.array(self.joint_velocities_log)
        ee_pos = np.array(self.ee_position_log)
        
        # Get joint names for better labels
        joint_names = self.dof_names if hasattr(self, 'dof_names') and self.dof_names else [f'Joint {i+1}' for i in range(joint_pos.shape[1])]
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Joint Positions (in degrees)
        ax = axes[0]
        joint_pos_deg = np.rad2deg(joint_pos)
        for i in range(joint_pos.shape[1]):
            label = joint_names[i] if i < len(joint_names) else f'Joint {i+1}'
            ax.plot(time, joint_pos_deg[:, i], label=label, linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Joint Angle (deg)', fontsize=11)
        ax.set_title('Joint Angles', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Joint Velocities (in deg/s)
        ax = axes[1]
        joint_vel_deg = np.rad2deg(joint_vel)
        for i in range(joint_vel.shape[1]):
            label = joint_names[i] if i < len(joint_names) else f'Joint {i+1}'
            ax.plot(time, joint_vel_deg[:, i], label=label, linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Joint Velocity (deg/s)', fontsize=11)
        ax.set_title('Joint Velocities', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: End Effector Position
        ax = axes[2]
        ax.plot(time, ee_pos[:, 0], label='X', linewidth=2)
        ax.plot(time, ee_pos[:, 1], label='Y', linewidth=2)
        ax.plot(time, ee_pos[:, 2], label='Z', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title('End Effector Position', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(f"plots/cup_manipulator_{SIMULATION_CONFIG.mode}_{timestamp}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(colored(f"✓ Plot saved to: {plot_path}", 'green'))
        
        plt.show()
    
    def run_scene_viz(self):
        """Run interactive scene visualization with terminal joint control.
        
        Note: This is a STATIC visualization mode - no physics simulation runs.
        The robot is displayed at the initial configuration and can be manually
        controlled via terminal input. To see physics simulation (pendulum swinging,
        gravity, damping), use --mode simulation instead.
        """
        print("\n[4/4] Running interactive scene visualization...")
        print("=" * 70)
        
        # Initialize simulation
        self.simulator.Initialize()
        
        # Force publish to Meshcat to show the initial configuration
        diagram = self.simulator.get_system()
        diagram.ForcedPublish(self.context)
        
        # Log initial state
        self.log_data(0.0)
        
        # Get joint positions
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
        ee_pos = self.cup_manipulator.get_end_effector_position(self.plant, plant_context)
        
        print(colored(f"\nInitial Cup Manipulator State:", 'magenta', attrs=['bold']))
        print(colored(f"  Joint Positions: {joint_positions}", 'cyan'))
        print(colored(f"  End Effector Position: {ee_pos}", 'cyan'))
        
        # Print ball center location at [0,0,0,0] configuration
        if PENDULUM_ENABLED:
            print(f"\n" + "=" * 70)
            print(colored("Ball Center Location at Joint Configuration [0,0,0,0]:", 'magenta', attrs=['bold']))
            print("=" * 70)
            
            # Save current state
            current_positions = self.plant.GetPositions(plant_context).copy()
            
            # Set all joints to zero
            self.plant.SetPositions(plant_context, np.zeros(len(current_positions)))
            
            # Get ball center in world frame
            try:
                link2_body = self.plant.GetBodyByName("link2", self.cup_manipulator.model_instance)
                X_WL2 = self.plant.EvalBodyPoseInWorld(plant_context, link2_body)
                ball_center_world = X_WL2 @ PENDULUM_CONFIG.attachment_point
                
                print(colored(f"  Link2 attachment point (link2 frame): {PENDULUM_CONFIG.attachment_point}", 'yellow'))
                print(colored(f"  Ball center (world frame): [{ball_center_world[0]:+.6f}, {ball_center_world[1]:+.6f}, {ball_center_world[2]:+.6f}]", 'cyan'))
                
                # Also print the pendulum ball body origin (pivot point)
                if hasattr(self.plant, 'GetBodyByName'):
                    try:
                        pendulum_ball_body = self.plant.GetBodyByName("pendulum_ball", self.cup_manipulator.model_instance)
                        X_WB = self.plant.EvalBodyPoseInWorld(plant_context, pendulum_ball_body)
                        pivot_pos = X_WB.translation()
                        print(colored(f"  Pendulum pivot (world frame): [{pivot_pos[0]:+.6f}, {pivot_pos[1]:+.6f}, {pivot_pos[2]:+.6f}]", 'cyan'))
                        
                        # Ball COM is at [0,0,+L] in pendulum body frame
                        ball_com_world = X_WB @ [0, 0, PENDULUM_CONFIG.length]
                        print(colored(f"  Pendulum ball COM (world frame): [{ball_com_world[0]:+.6f}, {ball_com_world[1]:+.6f}, {ball_com_world[2]:+.6f}]", 'cyan'))
                        
                        # Calculate distances
                        dist_ball_to_pivot = np.linalg.norm(ball_center_world - pivot_pos)
                        dist_pivot_to_com = np.linalg.norm(ball_com_world - pivot_pos)
                        
                        print(colored(f"\n  Distances:", 'yellow', attrs=['bold']))
                        print(colored(f"    Ball center to pivot: {dist_ball_to_pivot:.6f} m (should be ~0)", 'green'))
                        print(colored(f"    Pivot to pendulum COM: {dist_pivot_to_com:.6f} m (should be {PENDULUM_CONFIG.length} m)", 'green'))
                    except:
                        pass
                
                # Print end effector location in both frames
                print(colored(f"\n  End Effector Location:", 'yellow', attrs=['bold']))
                # Link2 origin is the end effector, so in link2 frame it's at origin
                ee_link2_frame = np.array([0.0, 0.0, 0.0])
                ee_world_frame = X_WL2.translation()
                print(colored(f"    wrt link2 frame: [{ee_link2_frame[0]:+.6f}, {ee_link2_frame[1]:+.6f}, {ee_link2_frame[2]:+.6f}]", 'cyan'))
                print(colored(f"    wrt world frame: [{ee_world_frame[0]:+.6f}, {ee_world_frame[1]:+.6f}, {ee_world_frame[2]:+.6f}]", 'cyan'))
                
            except Exception as e:
                print(f"  Error calculating ball position: {e}")
            
            # Restore original state
            self.plant.SetPositions(plant_context, current_positions)
            diagram = self.simulator.get_system()
            diagram.ForcedPublish(self.context)
            
            print("=" * 70)
        
        print("\n" + "=" * 70)
        print("Interactive Joint Control")
        print("=" * 70)
        print(f"View visualization at: {self.meshcat.web_url()}")
        
        print(colored("\n⚠ COORDINATE FRAME NOTE:", 'yellow', attrs=['bold']))
        print(colored("  Due to frame transformations in the pendulum setup:", 'yellow'))
        print(colored("  • Pendulum hanging DOWN (equilibrium) = pitch:+180°, roll:-180°", 'yellow'))
        print(colored("  • This matches the simulation mode behavior", 'yellow'))
        print(colored("  • Use these offsets when setting angles\n", 'yellow'))
        
        if PENDULUM_ENABLED:
            print(f"\nEnter joint positions (4 values in degrees, space-separated):")
            print(f"  Format: <link1_base> <link2_link1> <pendulum_pitch> <pendulum_roll>")
            print(f"  Example: 0 45 180 -180  (pendulum hanging down)")
            print(f"  Example: 0 45 160 -180 (pendulum swung 20° from vertical)")
            joint_names = ['link1_base', 'link2_link1', 'pendulum_pitch', 'pendulum_roll']
            expected_count = 4
        else:
            print(f"\nEnter joint positions (2 values in degrees, space-separated):")
            print(f"  Format: <link1_base> <link2_link1>")
            print(f"  Example: 0 45")
            joint_names = ['link1_base', 'link2_link1']
            expected_count = 2
            
        print(f"  Type 'q' or 'quit' to exit")
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
                    ee_pos = self.cup_manipulator.get_end_effector_position(self.plant, plant_context)
                    
                    # Display updated state (actual values read back from plant)
                    print(colored(f"\n← Actual joint values (read from plant):", 'cyan'))
                    for name, pos in joint_positions.items():
                        print(colored(f"    {name}: {np.rad2deg(pos):+7.2f}° ({pos:+.4f} rad)", 'cyan'))
                    
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
                    
                    print(colored(f"  End Effector: [{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]", 'cyan'))
                    
                except ValueError as e:
                    print(colored(f"❌ Error: Invalid input. Please enter {expected_count} numbers separated by spaces.", 'red'))
                    print(f"   Example: {'0 45 20 10' if PENDULUM_ENABLED else '0 45'}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
        
        print("\n" + "=" * 70)
        print("Scene visualization complete!")
        print("=" * 70)
    
    def run_simulation(self):
        """
        Run physics simulation with actuated 2-DOF manipulator.
        
        System:
        - Manipulator joints (link1_base, link2_link1) are actuated with PD control
        - Ball is rigidly attached to link2 (no gimbal joints)
        - Simplified 2-DOF demonstration
        
        Note: We DON'T use set_angle() because that's kinematic control which
        overrides physics. Instead, we apply torques via PD control.
        """
        print("\n[4/4] Running cart-pendulum style simulation...")
        print("=" * 70)
        
        # Initialize simulation
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(SIMULATION_CONFIG.visualization.realtime_rate)
        
        print(f"\nSimulation Duration: {self.simulation_config.simulation_time} s")
        print(f"Time Step: {self.simulation_config.timestep} s")
        print(f"Realtime Rate: {SIMULATION_CONFIG.visualization.realtime_rate}x")
        print(f"\nActuated Joints (PD Control): link1_base, link2_link1")
        print(f"Ball: Rigidly attached to link2")
        print(f"\nManipulator Motion: Sinusoidal (for {MANIPULATOR_MOTION_DURATION:.1f}s)")
        print(f"  Link1 Base: Amplitude={np.rad2deg(JOINT_MOTION_AMPLITUDE[0]):.1f}°, Freq={JOINT_MOTION_FREQUENCY[0]:.2f} Hz")
        print(f"  Link2-Link1: Amplitude={np.rad2deg(JOINT_MOTION_AMPLITUDE[1]):.1f}°, Freq={JOINT_MOTION_FREQUENCY[1]:.2f} Hz")
        print(f"\nPD Control Gains: Kp=100, Kd=10")
        print(f"\n⚠️  Manipulator will STOP at t={MANIPULATOR_MOTION_DURATION:.1f}s to let ball settle")
        print("\nStarting simulation...\n")
        
        

        # Get joint references (only 2 joints now)
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
        link1_base_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
        link2_link1_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
        
        # PD control gains
        Kp = 100.0  # Proportional gain (for motion phase)
        Kd = 10.0   # Derivative gain (for motion phase)
        Kp_hold = 1000.0  # Much higher gain to rigidly hold position during settling
        Kd_hold = 100.0   # Higher damping to resist disturbances
        
        # Simulation loop
        last_print_time = 0.0
        motion_stopped = False
        stop_position_link1 = 0.0
        stop_position_link2 = 0.0
        
        while self.context.get_time() < self.simulation_config.simulation_time:
            current_time = self.context.get_time()
            
            # Check if we should stop manipulator motion
            if current_time >= MANIPULATOR_MOTION_DURATION and not motion_stopped:
                stop_position_link1 = link1_base_joint.get_angle(plant_context)
                stop_position_link2 = link2_link1_joint.get_angle(plant_context)
                motion_stopped = True
                print(f"\n{'='*70}")
                print(f"t={current_time:.2f}s: MANIPULATOR STOPPED - Ball settling phase begins")
                print(f"  Holding position: Link1={np.rad2deg(stop_position_link1):+.1f}°, Link2={np.rad2deg(stop_position_link2):+.1f}°")
                print(f"  PD gains increased to: Kp={Kp_hold}, Kd={Kd_hold} (rigid hold)")
                print(f"{'='*70}\n")
            
            # Calculate desired manipulator joint positions
            if current_time < MANIPULATOR_MOTION_DURATION:
                # Moving phase: sinusoidal motion
                link1_desired = JOINT_MOTION_AMPLITUDE[0] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * current_time)
                link2_desired = JOINT_MOTION_AMPLITUDE[1] * np.sin(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * current_time)
                
                # Calculate desired velocities (derivatives of desired positions)
                link1_desired_vel = JOINT_MOTION_AMPLITUDE[0] * 2 * np.pi * JOINT_MOTION_FREQUENCY[0] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[0] * current_time)
                link2_desired_vel = JOINT_MOTION_AMPLITUDE[1] * 2 * np.pi * JOINT_MOTION_FREQUENCY[1] * np.cos(2 * np.pi * JOINT_MOTION_FREQUENCY[1] * current_time)
                
                # Use normal gains for motion tracking
                current_Kp = Kp
                current_Kd = Kd
            else:
                # Settling phase: hold fixed position with zero velocity
                link1_desired = stop_position_link1
                link2_desired = stop_position_link2
                link1_desired_vel = 0.0
                link2_desired_vel = 0.0
                
                # Use much higher gains to rigidly hold position
                current_Kp = Kp_hold
                current_Kd = Kd_hold
            
            # Get current joint states
            link1_actual = link1_base_joint.get_angle(plant_context)
            link2_actual = link2_link1_joint.get_angle(plant_context)
            link1_actual_vel = link1_base_joint.get_angular_rate(plant_context)
            link2_actual_vel = link2_link1_joint.get_angular_rate(plant_context)
            
            # PD control: torque = Kp * (desired - actual) + Kd * (desired_vel - actual_vel)
            link1_torque = current_Kp * (link1_desired - link1_actual) + current_Kd * (link1_desired_vel - link1_actual_vel)
            link2_torque = current_Kp * (link2_desired - link2_actual) + current_Kd * (link2_desired_vel - link2_actual_vel)
            
            # Apply torques to manipulator joints
            self.cup_manipulator.apply_torques(self.plant, plant_context, link1_torque, link2_torque)
            
            # Advance simulation (dynamics will handle passive joints)
            self.simulator.AdvanceTo(current_time + self.simulation_config.timestep)
            
            # Log data
            self.log_data(current_time)
            
            # Print status
            if current_time - last_print_time >= SIMULATION_CONFIG.visualization.print_interval:
                joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
                
                # Get joint positions (2 manipulator joints + optional 2 gimbal joints)
                link1_pos = joint_positions.get('link1_base', 0)
                link2_pos = joint_positions.get('link2_link1', 0)
                
                # Add phase indicator
                phase = "[MOVING]" if current_time < MANIPULATOR_MOTION_DURATION else "[SETTLING]"
                
                if PENDULUM_ENABLED:
                    pitch_pos = joint_positions.get('pendulum_pitch', 0)
                    roll_pos = joint_positions.get('pendulum_roll', 0)
                    
                    # Get pendulum ball world position to check if it's hanging down
                    try:
                        pendulum_ball_body = self.plant.GetBodyByName("pendulum_ball", self.cup_manipulator.model_instance)
                        # Get the body pose (includes rotation and translation)
                        X_WB = self.plant.EvalBodyPoseInWorld(plant_context, pendulum_ball_body)
                        # The COM is at [0,0,-PENDULUM_LENGTH] in body frame, transform to world
                        ball_com_world = X_WB @ [0, 0, -PENDULUM_CONFIG.length]
                        
                        # Get attachment point in world frame
                        link2_body = self.plant.GetBodyByName("link2", self.cup_manipulator.model_instance)
                        X_WL2 = self.plant.EvalBodyPoseInWorld(plant_context, link2_body)
                        attach_world_pos = X_WL2 @ PENDULUM_CONFIG.attachment_point
                        
                        # Vector from attachment to COM
                        pendulum_vec = ball_com_world - attach_world_pos
                        # Angle from vertical (down is -Z direction in WORLD FRAME)
                        # This is a geometric measurement independent of joint coordinates
                        vertical_vec = np.array([0, 0, -1])
                        cos_angle = np.dot(pendulum_vec, vertical_vec) / (np.linalg.norm(pendulum_vec) + 1e-10)
                        cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
                        angle_from_vertical = np.rad2deg(np.arccos(cos_angle))
                        
                        # Extract ball position
                        ball_pos = X_WB.translation()
                    except Exception as e:
                        angle_from_vertical = -999
                        ball_pos = None
                    
                    # Print joint angles and pendulum info
                    print(colored(f"t={current_time:.2f}s {phase}", 'blue') + 
                          colored(f" | link1_base: {np.rad2deg(link1_pos):+6.1f}°", 'cyan') +
                          colored(f" | link2_link1: {np.rad2deg(link2_pos):+6.1f}°", 'cyan') +
                          colored(f" | pendulum_pitch: {np.rad2deg(pitch_pos):+6.1f}°", 'yellow') +
                          colored(f" | pendulum_roll: {np.rad2deg(roll_pos):+6.1f}°", 'yellow') +
                          colored(f" | ∠vertical: {angle_from_vertical:+5.1f}°", 'magenta'))
                    
                    if ball_pos is not None:
                        print(colored(f"  Ball position (world): [{ball_pos[0]:+7.4f}, {ball_pos[1]:+7.4f}, {ball_pos[2]:+7.4f}]", 'green'))
                else:
                    # Print joint angles only (no pendulum)
                    print(colored(f"t={current_time:.2f}s {phase}", 'blue') + 
                          colored(f" | link1_base: {np.rad2deg(link1_pos):+6.1f}°", 'cyan') +
                          colored(f" | link2_link1: {np.rad2deg(link2_pos):+6.1f}°", 'cyan'))
                last_print_time = current_time
        
        print("\n" + "=" * 70)
        print("Simulation complete!")
        print(f"Manipulator moved for {MANIPULATOR_MOTION_DURATION:.1f}s, then stopped.")
        print("=" * 70)
        
        # Plot results
        self.plot_results()

    def run_min_jerk_joint(self):
        """Run minimum-jerk joint-space trajectory with PD control."""
        print("\n[4/4] Running minimum-jerk joint-space simulation...")
        print("=" * 70)

        # Initialize simulation
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(SIMULATION_CONFIG.visualization.realtime_rate)

        print(f"\nSimulation Duration: {self.simulation_config.simulation_time} s")
        print(f"Time Step: {self.simulation_config.timestep} s")
        print(f"Realtime Rate: {SIMULATION_CONFIG.visualization.realtime_rate}x")
        print(f"\nActuated Joints (PD Control): link1_base, link2_link1")
        print(f"Ball: Rigidly attached to link2")
        print(f"\nMinimum-Jerk Duration: {MIN_JERK_DURATION:.1f}s")
        print(f"Target Angles: link1={np.rad2deg(MIN_JERK_TARGET_ANGLES[0]):+.1f}°, link2={np.rad2deg(MIN_JERK_TARGET_ANGLES[1]):+.1f}°")
        print(f"\nPD Control Gains: Kp=100, Kd=10")
        print(f"\n⚠️  Manipulator will HOLD at t={MIN_JERK_DURATION:.1f}s")
        print("\nStarting simulation...\n")

        # Get joint references (only 2 joints now)
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        link1_base_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
        link2_link1_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)

        # Initial joint positions
        q0_link1 = link1_base_joint.get_angle(plant_context)
        q0_link2 = link2_link1_joint.get_angle(plant_context)
        qf_link1, qf_link2 = MIN_JERK_TARGET_ANGLES

        # PD control gains
        Kp = 100.0
        Kd = 10.0
        Kp_hold = 1000.0
        Kd_hold = 100.0

        # Simulation loop
        last_print_time = 0.0
        holding_phase = False

        while self.context.get_time() < self.simulation_config.simulation_time:
            current_time = self.context.get_time()

            # Minimum-jerk profile
            if current_time <= MIN_JERK_DURATION:
                h, hdot, _ = min_jerk_profile(current_time, MIN_JERK_DURATION)
                link1_desired = q0_link1 + (qf_link1 - q0_link1) * h
                link2_desired = q0_link2 + (qf_link2 - q0_link2) * h
                link1_desired_vel = (qf_link1 - q0_link1) * hdot
                link2_desired_vel = (qf_link2 - q0_link2) * hdot
                current_Kp = Kp
                current_Kd = Kd
                phase = "[MOVING]"
            else:
                if not holding_phase:
                    holding_phase = True
                    print(f"\n{'='*70}")
                    print(f"t={current_time:.2f}s: HOLDING FINAL POSE")
                    print(f"  Holding position: Link1={np.rad2deg(qf_link1):+.1f}°, Link2={np.rad2deg(qf_link2):+.1f}°")
                    print(f"  PD gains increased to: Kp={Kp_hold}, Kd={Kd_hold} (rigid hold)")
                    print(f"{'='*70}\n")

                link1_desired = qf_link1
                link2_desired = qf_link2
                link1_desired_vel = 0.0
                link2_desired_vel = 0.0
                current_Kp = Kp_hold
                current_Kd = Kd_hold
                phase = "[HOLD]"

            # Get current joint states
            link1_actual = link1_base_joint.get_angle(plant_context)
            link2_actual = link2_link1_joint.get_angle(plant_context)
            link1_actual_vel = link1_base_joint.get_angular_rate(plant_context)
            link2_actual_vel = link2_link1_joint.get_angular_rate(plant_context)

            # PD control torques
            link1_torque = current_Kp * (link1_desired - link1_actual) + current_Kd * (link1_desired_vel - link1_actual_vel)
            link2_torque = current_Kp * (link2_desired - link2_actual) + current_Kd * (link2_desired_vel - link2_actual_vel)

            # Apply torques
            self.cup_manipulator.apply_torques(self.plant, plant_context, link1_torque, link2_torque)

            # Advance simulation
            self.simulator.AdvanceTo(current_time + self.simulation_config.timestep)

            # Log data
            self.log_data(current_time)

            # Print status
            if current_time - last_print_time >= SIMULATION_CONFIG.visualization.print_interval:
                joint_positions = self.cup_manipulator.get_joint_positions(self.plant, plant_context)
                link1_pos = joint_positions.get('link1_base', 0)
                link2_pos = joint_positions.get('link2_link1', 0)

                if PENDULUM_ENABLED:
                    pitch_pos = joint_positions.get('pendulum_pitch', 0)
                    roll_pos = joint_positions.get('pendulum_roll', 0)
                    print(colored(f"t={current_time:.2f}s {phase}", 'blue') +
                          colored(f" | link1_base: {np.rad2deg(link1_pos):+6.1f}°", 'cyan') +
                          colored(f" | link2_link1: {np.rad2deg(link2_pos):+6.1f}°", 'cyan') +
                          colored(f" | pendulum_pitch: {np.rad2deg(pitch_pos):+6.1f}°", 'yellow') +
                          colored(f" | pendulum_roll: {np.rad2deg(roll_pos):+6.1f}°", 'yellow'))
                else:
                    print(colored(f"t={current_time:.2f}s {phase}", 'blue') +
                          colored(f" | link1_base: {np.rad2deg(link1_pos):+6.1f}°", 'cyan') +
                          colored(f" | link2_link1: {np.rad2deg(link2_pos):+6.1f}°", 'cyan'))
                last_print_time = current_time

        print("\n" + "=" * 70)
        print("Minimum-jerk simulation complete!")
        print(f"Manipulator moved for {MIN_JERK_DURATION:.1f}s, then held.")
        print("=" * 70)

        # Plot results
        self.plot_results()


    def run_all_jts(self):
        """Interactive mode: control all joints from terminal input."""
        print("\n[4/4] Running interactive joint control...")
        print("=" * 70)
        
        # Initialize simulation
        self.simulator.Initialize()
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Get all joints
        link1_base_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
        link2_link1_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
        ball_pitch_joint = self.plant.GetJointByName("ball_pitch", self.cup_manipulator.model_instance)
        ball_yaw_joint = self.plant.GetJointByName("ball_yaw", self.cup_manipulator.model_instance)
        
        print("\nInteractive Joint Control Mode")
        print("=" * 70)
        print("Joint order: link1_base, link2_link1, ball_pitch, ball_yaw")
        print("Input format: Enter 4 comma-separated angles in DEGREES")
        print("Example: 90, 0, 45, 10")
        print("Type 'q' or 'exit' to quit")
        print("=" * 70)
        
        if SIMULATION_CONFIG.visualization.enabled:
            print(f"\nMeshcat visualization: http://localhost:7007")
        
        while True:
            try:
                # Get current positions
                current = [
                    np.degrees(link1_base_joint.get_angle(plant_context)),
                    np.degrees(link2_link1_joint.get_angle(plant_context)),
                    np.degrees(ball_pitch_joint.get_angle(plant_context)),
                    np.degrees(ball_yaw_joint.get_angle(plant_context))
                ]
                
                print(f"\nCurrent positions (deg): [{current[0]:.1f}, {current[1]:.1f}, {current[2]:.1f}, {current[3]:.1f}]")
                user_input = input("Enter joint angles (deg): ").strip()
                
                # Check for exit
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\nExiting interactive mode...")
                    break
                
                # Parse input
                try:
                    values = [float(x.strip()) for x in user_input.split(',')]
                    if len(values) != 4:
                        print(f"Error: Expected 4 values, got {len(values)}. Try again.")
                        continue
                    
                    # Convert degrees to radians
                    angles_rad = [np.radians(v) for v in values]
                    
                    # Set joint positions
                    link1_base_joint.set_angle(plant_context, angles_rad[0])
                    link2_link1_joint.set_angle(plant_context, angles_rad[1])
                    ball_pitch_joint.set_angle(plant_context, angles_rad[2])
                    ball_yaw_joint.set_angle(plant_context, angles_rad[3])
                    
                    # Update visualization
                    self.simulator.AdvanceTo(self.context.get_time() + 0.01)
                    
                    # Display feedback
                    print(f"✓ Moved to: [{values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}, {values[3]:.1f}] deg")
                    print(f"           [{angles_rad[0]:.3f}, {angles_rad[1]:.3f}, {angles_rad[2]:.3f}, {angles_rad[3]:.3f}] rad")
                    
                    # Get end effector and ball positions
                    ee_pos = self.cup_manipulator.get_end_effector_position(self.plant, plant_context)
                    ball_body = self.plant.GetBodyByName("part_1", self.cup_manipulator.model_instance)
                    ball_pose = self.plant.EvalBodyPoseInWorld(plant_context, ball_body)
                    ball_pos = ball_pose.translation()
                    
                    print(f"  End effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                    print(f"  Ball position: [{ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}]")
                    
                except ValueError as e:
                    print(f"Error parsing input: {e}. Use format: 90, 0, 45, 10")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution flow for PyDrake simulation."""
    print("\n" + "=" * 70)
    print(colored("PYDRAKE: Cup Manipulator Visualization", 'cyan', attrs=['bold']))
    print(colored("Educational Robotics Simulation", 'cyan'))
    print("=" * 70)
    print(colored(f"Mode: {SIMULATION_CONFIG.mode}", 'yellow'))
    print(colored(f"Time Step: {SIMULATION_CONFIG.timestep} s", 'yellow'))
    print(colored(f"Duration: {SIMULATION_CONFIG.simulation_time} s", 'yellow'))
    print("=" * 70 + "\n")
    
    try:
        # Create scene manager with config
        scene_manager = DrakeSceneManager(CUP_MANIPULATOR_CONFIG, SIMULATION_CONFIG)  # BALL_CONFIG commented out
        
        # Setup Drake system
        scene_manager.setup_drake_system()
        
        # Create simulator
        scene_manager.create_simulator()
        
        # Set initial conditions
        scene_manager.set_initial_conditions()
        
        # Run simulation based on mode
        if SIMULATION_CONFIG.mode == "scene-viz":
            scene_manager.run_scene_viz()
        elif SIMULATION_CONFIG.mode == "simulation":
            scene_manager.run_simulation()
        elif SIMULATION_CONFIG.mode == "joint-motion":
            scene_manager.run_joint_motion()
        elif SIMULATION_CONFIG.mode == "run-all-jts":
            scene_manager.run_all_jts()
        elif SIMULATION_CONFIG.mode == "min-jerk-joint":
            scene_manager.run_min_jerk_joint()
        else:
            raise ValueError(f"Unknown simulation mode: {SIMULATION_CONFIG.mode}")
    
    except Exception as e:
        print(f"\n{colored('ERROR:', 'red', attrs=['bold'])} {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
