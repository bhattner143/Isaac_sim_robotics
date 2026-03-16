import numpy as np
import os
from typing import Optional, List
from abc import ABC, abstractmethod
from termcolor import colored

from pydrake.all import (
    MultibodyPlant,
    Parser,
    SpatialInertia,
    UnitInertia,
    RigidTransform,
    RevoluteJoint,
    PrismaticJoint,
    Sphere,
    Cylinder,
    JacobianWrtVariable,
)
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.math import RigidTransform, RollPitchYaw

from configs.robot.robot_types import ManipulatorConfig
from configs.robot.robot_configs import CartPendulumPhysicsConfig

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
        
        # Auto-detect joint names from URDF (excluding weld joints)
        # CRITICAL: Identify joints by their connectivity, NOT parse order
        # Drake's GetJointIndices() returns in URDF parse order, which varies between files
        revolute_joints_info = []
        for joint_idx in plant.GetJointIndices(model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0 and "weld" not in joint.name().lower():
                revolute_joints_info.append({
                    'name': joint.name(),
                    'parent': joint.parent_body().name(),
                    'child': joint.child_body().name()
                })
        
        if len(revolute_joints_info) == 2:
            # Identify q1 (base→link1) and q2 (link1→link2) by checking parent/child bodies
            jt1_name, jt2_name = None, None
            for jt_info in revolute_joints_info:
                # q1 connects base to link1
                if 'base' in jt_info['parent'].lower() and 'link1' in jt_info['child'].lower():
                    jt1_name = jt_info['name']
                # q2 connects link1 to link2
                elif 'link1' in jt_info['parent'].lower() and 'link2' in jt_info['child'].lower():
                    jt2_name = jt_info['name']
            
            if jt1_name and jt2_name:
                self.JT1_NAME = jt1_name  # Joint from base to link1 (q1)
                self.JT2_NAME = jt2_name  # Joint from link1 to link2 (q2)
                self.ACT1_NAME = f"tau_{self.JT1_NAME}"
                self.ACT2_NAME = f"tau_{self.JT2_NAME}"
                self.joint_names = [self.JT1_NAME, self.JT2_NAME]
                print(colored(f"✓ Auto-detected joint names by connectivity:", 'green'))
                print(colored(f"  JT1 (q1, base→link1): {self.JT1_NAME}", 'cyan'))
                print(colored(f"  JT2 (q2, link1→link2): {self.JT2_NAME}", 'cyan'))
            else:
                print(colored(f"⚠️  Could not identify joints by connectivity. Using default names.", 'yellow'))
        else:
            print(colored(f"⚠️  Expected 2 revolute joints, found {len(revolute_joints_info)}", 'yellow'))
        
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

    CRITICAL: Drake joint ordering is [link2_link1, link1_base] = [q2, q1]
    This class handles all conversions internally, so external code can use [q1, q2].

    NOTE: For cup_manipulator_obj_right_frame.urdf, the coordinate frame is already 
    aligned correctly (no -90° Y rotation needed), so EE offsets use positive X/Z.

    Manages:
    - URDF loading and joint configuration
    - State queries (positions, velocities)
    - End-effector kinematics
    - Automatic conversion between user [q1, q2] and Drake [q2, q1] ordering
    """

    # --- Cup-center EE pose relative to link2, from URDF simple_ball (cup middle) origin ---
    # For cup_manipulator_obj_natural_order.urdf: actual joint names from URDF
    # URDF joint names: link1_base, link2_link1
    # Note: Joint and link names are auto-detected from URDF in load_urdf_to_plant()
    JT1_NAME = "link1_base"  # q1: base to link1 (will be overridden by auto-detection)
    JT2_NAME = "link2_link1"  # q2: link1 to link2 (will be overridden by auto-detection)
    ACT1_NAME = f"tau_{JT1_NAME}"
    ACT2_NAME = f"tau_{JT2_NAME}"
    LINK2_NAME = "link2"  # End-effector attachment link (consistent across URDFs)

    EE_XYZ_LINK2 = np.array([1.2515, 0.0, 0.15])   # meters, when q1 = q2 = 0 
    EE_XYZ_LINK1 = np.array([2.2045,0,0.071875])             # meters, when q1 = 0
    EE_XYZ_BASE = np.array([2.2045,0,1.248125])            # meters (no offset)
    EE_RPY_LINK2 = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = q2 = 0 
    EE_RPY_LINK1 = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = 0 
    EE_RPY_BASE = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = q2 = 0 
    EE_FRAME_NAME = "cup_center"  # the canonical EE frame name inside Drake 
    
    # Alias for end-effector offset (offset from link2 frame to cup center)
    # Used by IK and controllers - points to the same location as the cup_center frame
    EE_OFFSET = EE_XYZ_LINK2  # [1.2515, 0.0, 0.15] meters from link2 origin

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization = enable_visualization

    # ------------------------------------------------------------------
    # ADD END EFFECTOR FRAME (call this BEFORE plant.Finalize())
    # ------------------------------------------------------------------
    def add_end_effector_frame(self, plant: MultibodyPlant):
        """
        Creates a Drake frame at the simple_ball's location (cup center).
        
        The URDF's simple_ball visual element is at offset [1.2515, 0, 0.15] from link2,
        but Drake doesn't auto-create frames for visual elements. This method explicitly
        creates a named frame at that exact ball location for IK and kinematics queries.

        Must be called AFTER the model is added (self.model_instance is valid)
        and BEFORE plant.Finalize().

        Returns:
            The created Frame (FixedOffsetFrame) at the ball's location
        """
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")

        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)

        X_L2_EE = RigidTransform(
            RollPitchYaw(self.EE_RPY_LINK2),
            self.EE_XYZ_LINK2
        )

        # Avoid double-adding if called multiple times
        try:
            existing = plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
            return existing
        except Exception:
            pass

        ee_frame = plant.AddFrame(
            FixedOffsetFrame(
                self.EE_FRAME_NAME,
                link2_body.body_frame(),
                X_L2_EE,
                self.model_instance
            )
        )
        return ee_frame

    # Convenience accessor
    def get_end_effector_frame(self, plant: MultibodyPlant):
        """Get the Drake frame object for the end effector (cup center)."""
        return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
    
    # ------------------------------------------------------------------
    # ADD JOINT ACTUATORS (call this BEFORE plant.Finalize())
    # ------------------------------------------------------------------
    def add_joint_actuators(self, plant: MultibodyPlant):
        """
        Add actuators to manipulator joints using explicit joint names.
        
        Must be called AFTER the model is added (self.model_instance is valid)
        and BEFORE plant.Finalize().
        
        This allows torques to be applied to the manipulator joints.
        """
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        
        # Add actuators using explicit joint names
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        plant.AddJointActuator(self.ACT1_NAME, jt1)
        plant.AddJointActuator(self.ACT2_NAME, jt2)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ Added actuators: {self.ACT1_NAME}, {self.ACT2_NAME}", 'green'))
    
    # ------------------------------------------------------------------
    # EE QUERIES (fixed to use cup_center frame)
    # ------------------------------------------------------------------
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """World position of the cup middle (end effector)."""
        ee_frame = self.get_end_effector_frame(plant)
        X_WE = plant.CalcRelativeTransform(context, plant.world_frame(), ee_frame)
        return X_WE.translation()

    def CalcPosition(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Same as get_end_effector_position(), kept for compatibility."""
        return self.get_end_effector_position(plant, context)

    # ------------------------------------------------------------------
    # STATE HELPERS (unchanged)
    # ------------------------------------------------------------------
    def get_state_from_plant(self, plant: MultibodyPlant, context) -> np.ndarray:
        return plant.GetPositionsAndVelocities(context, self.model_instance)

    def set_state_in_plant(self, plant: MultibodyPlant, context, user_state: np.ndarray):
        """Set full state in user order [q1, q2, q1_dot, q2_dot]."""
        q1, q2, q1_dot, q2_dot = user_state
        # Use new unified methods with Drake ordering [JT1=q2, JT2=q1]
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q2, q1])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q2_dot, q1_dot])

    def get_positions_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get positions in user order [q1, q2]."""
        # Get in Drake order [q2, q1], then reverse to user order
        drake_positions = self.get_jt([self.JT1_NAME, self.JT2_NAME], plant, context)
        return np.array([drake_positions[1], drake_positions[0]])  # [q1, q2]

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions):
        """Set positions by joint name using a dictionary.
        
        Args:
            user_positions: Dict[str, float] mapping joint names to angles, e.g.
                           {'link1_base': 0.0, 'link2_link1': 0.349}
                           OR np.ndarray [q1, q2] for backward compatibility
        """
        if isinstance(user_positions, dict):
            # Use dict directly - explicit and unambiguous
            for joint_name, angle in user_positions.items():
                self.set_jt([joint_name], plant, context, [angle])
        else:
            # Backward compatibility: array [q1, q2]
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get velocities in user order [q1_dot, q2_dot]."""
        # JT1_NAME="link1_base", JT2_NAME="link2_link1" - returns [q1_dot, q2_dot]
        return self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities):
        """Set velocities by joint name using a dictionary.
        
        Args:
            user_velocities: Dict[str, float] mapping joint names to velocities, e.g.
                            {'link1_base': 0.0, 'link2_link1': 0.1}
                            OR np.ndarray [q1_dot, q2_dot] for backward compatibility
        """
        if isinstance(user_velocities, dict):
            # Use dict directly - explicit and unambiguous
            for joint_name, velocity in user_velocities.items():
                self.set_jt_velocity([joint_name], plant, context, [velocity])
        else:
            # Backward compatibility: array [q1_dot, q2_dot]
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_joint_positions(self, plant: MultibodyPlant, context):
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions

    def get_joint_velocities(self, plant: MultibodyPlant, context):
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities

    # ------------------------------------------------------------------
    # JOINT-SPECIFIC HELPERS (using JT1_NAME and JT2_NAME)
    # ------------------------------------------------------------------
    def get_joint_by_name(self, plant: MultibodyPlant, joint_name: str):
        """Get joint object by name."""
        return plant.GetJointByName(joint_name, self.model_instance)
    
    def get_jt(self, joint_name: str | List[str], plant: MultibodyPlant, context):
        """
        Get joint angle(s) [rad] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
                       (e.g., self.JT1_NAME or [self.JT1_NAME, self.JT2_NAME])
            plant: Drake MultibodyPlant
            context: Drake context
            
        Returns:
            float if single joint name provided
            np.ndarray if list of joint names provided
        """
        if isinstance(joint_name, list):
            return np.array([self.get_joint_by_name(plant, name).get_angle(context) 
                           for name in joint_name])
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            return joint.get_angle(context)
    
    def set_jt(self, joint_name: str | List[str], plant: MultibodyPlant, context, 
               angle: float | np.ndarray | List[float]):
        """
        Set joint angle(s) [rad] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            angle: Single angle or array/list of angles (must match joint_name length)
        """
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            if len(angles) != len(joint_name):
                raise ValueError(f"Number of angles ({len(angles)}) must match "
                               f"number of joints ({len(joint_name)})")
            for name, ang in zip(joint_name, angles):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            joint.set_angle(context, float(angle))
    
    def get_jt_velocity(self, joint_name: str | List[str], plant: MultibodyPlant, context):
        """
        Get joint angular velocity(ies) [rad/s] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            
        Returns:
            float if single joint name provided
            np.ndarray if list of joint names provided
        """
        if isinstance(joint_name, list):
            return np.array([self.get_joint_by_name(plant, name).get_angular_rate(context) 
                           for name in joint_name])
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            return joint.get_angular_rate(context)
    
    def set_jt_velocity(self, joint_name: str | List[str], plant: MultibodyPlant, context, 
                       velocity: float | np.ndarray | List[float]):
        """
        Set joint angular velocity(ies) [rad/s] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            velocity: Single velocity or array/list of velocities (must match joint_name length)
        """
        if isinstance(joint_name, list):
            velocities = np.atleast_1d(velocity)
            if len(velocities) != len(joint_name):
                raise ValueError(f"Number of velocities ({len(velocities)}) must match "
                               f"number of joints ({len(joint_name)})")
            for name, vel in zip(joint_name, velocities):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            joint.set_angular_rate(context, float(velocity))
    
    # ------------------------------------------------------------------
    # INVERSE KINEMATICS
    # ------------------------------------------------------------------
    def solve_initial_pose_via_ik(
        self,
        plant,
        target_xy,
        q_seed,
        pos_tol=1e-3,
        verbose=False,
        ee_frame_name=None,
        target_z=None,
    ):
        """
        Solve for joint angles [q1, q2] (USER order) that place the EE at target (x, y, z).

        Args:
            plant: MultibodyPlant containing this manipulator
            target_xy: Target [x, y] position for end-effector
            q_seed: Initial guess for joint angles [q1, q2] in user order
            pos_tol: Position tolerance [m] for IK constraint
            verbose: Print detailed solver information
            ee_frame_name: Name of EE frame (defaults to self.EE_FRAME_NAME)
            target_z: Target Z coordinate (if None, uses seed configuration's Z)

        Returns:
            q_sol_user: Solution joint angles [q1, q2] in user order
            success: Boolean indicating if IK succeeded
        """
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve
        
        target_xy = np.asarray(target_xy).reshape(2,)
        q_seed = np.asarray(q_seed).reshape(2,)

        # Create IK and use its internal context for FK evaluations
        ik = InverseKinematics(plant)
        ik_context = ik.context()

        # Put IK context at seed configuration (so we can read seed z)
        self.set_positions_user_order(plant, ik_context, q_seed)

        world = plant.world_frame()

        # Prefer a named EE frame you added (e.g., "cup_center")
        if ee_frame_name is None:
            ee_frame_name = self.EE_FRAME_NAME

        if ee_frame_name is not None:
            try:
                ee_frame = plant.GetFrameByName(ee_frame_name, self.model_instance)
                # constrain the ORIGIN of the EE frame
                p_BQ = np.zeros(3)
            except:
                # Fallback: constrain a point on link2 body frame using EE_XYZ_LINK2 offset
                link2_body = plant.GetBodyByName("link2", self.model_instance)
                ee_frame = link2_body.body_frame()
                p_BQ = np.asarray(self.EE_XYZ_LINK2).reshape(3,)
        else:
            # Fallback: constrain a point on link2 body frame using EE_XYZ_LINK2 offset
            link2_body = plant.GetBodyByName("link2", self.model_instance)
            ee_frame = link2_body.body_frame()
            p_BQ = np.asarray(self.EE_XYZ_LINK2).reshape(3,)

        # Compute seed EE position (to pick z if not specified)
        ee_pos_seed = plant.CalcPointsPositions(
            ik_context,
            ee_frame,
            p_BQ.reshape(3, 1),
            world,
        ).ravel()
        z_target = target_z if target_z is not None else ee_pos_seed[2]

        if verbose:
            print(f"  Seed EE position: ({ee_pos_seed[0]:.3f}, {ee_pos_seed[1]:.3f}, {ee_pos_seed[2]:.3f})")
            print(f"  Target: ({target_xy[0]:.3f}, {target_xy[1]:.3f}, {z_target:.3f})")
            print(f"  Tolerance: ±{pos_tol:.6f} m")

        # Position constraint in world
        lower = np.array([target_xy[0], target_xy[1], z_target]) - pos_tol
        upper = np.array([target_xy[0], target_xy[1], z_target]) + pos_tol
        ik.AddPositionConstraint(
            frameB=ee_frame,
            p_BQ=p_BQ,
            frameA=world,
            p_AQ_lower=lower,
            p_AQ_upper=upper,
        )

        prog = ik.prog()
        q_vars = ik.q()

        # Add cost: stay near seed configuration with STRONG weight
        q0_all = plant.GetPositions(ik_context)  # All positions in the plant
        
        # Use much higher weight (1000x) to strongly prefer staying near current config
        # This prevents jumping between different IK solutions (elbow-up vs elbow-down)
        weight_matrix = 1000.0 * np.eye(len(q0_all))
        prog.AddQuadraticErrorCost(weight_matrix, q0_all, q_vars)
        prog.SetInitialGuess(q_vars, q0_all)

        result = Solve(prog)

        if verbose:
            print(f"  IK solver status: {result.get_solver_id().name()}")
            print(f"  Success: {result.is_success()}")
            if not result.is_success():
                print(f"  Solver details: {result.get_solution_result()}")

        if not result.is_success():
            return q_seed, False

        # Extract the solution
        q_sol_all = result.GetSolution(q_vars)
        
        # Create a temporary context to extract the manipulator-specific positions
        temp_context = plant.CreateDefaultContext()
        plant.SetPositions(temp_context, q_sol_all)
        
        # Get manipulator positions in Drake order [q2, q1]
        q_sol_drake = plant.GetPositions(temp_context, self.model_instance)
        
        # Convert from Drake order [q2, q1] to user order [q1, q2]
        q_sol_user = np.array([q_sol_drake[1], q_sol_drake[0]])
        
        if verbose:
            print(f"  Solution (user): q1={np.rad2deg(q_sol_user[0]):.2f}°, q2={np.rad2deg(q_sol_user[1]):.2f}°")

        return q_sol_user, True

    # ------------------------------------------------------------------
    # WELD BASE (unchanged; but note you want the BODY frame, not a frame named as the link)
    # ------------------------------------------------------------------
    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0])
    ):
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")

        world_frame = plant.world_frame()

        # Better: weld the base BODY frame (robust) rather than GetFrameByName on a link name
        base_body = plant.GetBodyByName("base_mount_manipulator", self.model_instance)
        base_frame = base_body.body_frame()

        X_WB = RigidTransform(RollPitchYaw(orientation), position)
        plant.WeldFrames(world_frame, base_frame, X_WB)



# ============================================================================
# EXTENDED 2D CART-PENDULUM CLASS
# ============================================================================
class CartPendulum2DExtended:
    """
    Extended CartPendulum class for 2D motion.
    
    Extends the architecture from script_cart_pendulum_muscle_dynamics_ofc.py
    to support 2D cart motion (x, y) with 3D pendulum (pitch, roll).
    
    STATE: [x, y, α, β, ẋ, ẏ, α̇, β̇] (8D)
    INPUT: [F_x, F_y] (2D force)
    
    Structure:
    - world → x_slider (prismatic X) → y_slider (prismatic Y) → cart → pendulum (gimbal)
    """
    
    def __init__(self, config: CartPendulumPhysicsConfig, z_offset: float = 0.0):
        """
        Initialize 2D cart-pendulum system.
        
        Args:
            config: Physical parameters
            z_offset: Vertical offset for cart base [m]
        """
        self.config = config
        self.z_offset = z_offset
        
        # Will be populated during build
        self.cart_body = None
        self.x_slider_body = None
        self.y_slider_body = None
        self.x_joint = None
        self.y_joint = None
        self.pitch_joint = None
        self.roll_joint = None
        self.pendulum_body = None
        self.pitch_body = None
    
    def build_plant(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True) -> None:
        """
        Build 2D cart-pendulum in the given plant.
        
        Similar to CartPendulumSystemDynamics.build_plant() but extended to 2D.
        
        Args:
            plant: MultibodyPlant to add bodies to
            model_instance: Model instance index
            register_visuals: Whether to register visual geometry (requires SceneGraph)
        """
        # ====================================================================
        # CREATE CART BODY
        # ====================================================================
        cart_size = 0.1
        m_c = self.config.mass_cart
        
        # Cart inertia (box approximation)
        I_c = (1/12) * m_c * cart_size**2
        cart_inertia = SpatialInertia(
            mass=m_c,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_c, I_c, I_c)
        )
        
        self.cart_body = plant.AddRigidBody(
            "cart",
            model_instance,
            cart_inertia
        )
        
        # Cart visual geometry (only if SceneGraph is registered)
        if register_visuals:
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                Sphere(cart_size / 2),
                "cart_visual",
                np.array([0.3, 0.3, 0.8, 1.0])
            )
        
        # ====================================================================
        # CREATE SLIDER BODIES (for 2D motion)
        # ====================================================================
        slider_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        
        self.x_slider_body = plant.AddRigidBody(
            "x_slider",
            model_instance,
            slider_inertia
        )
        
        self.y_slider_body = plant.AddRigidBody(
            "y_slider",
            model_instance,
            slider_inertia
        )
        
        # ====================================================================
        # CREATE PRISMATIC JOINTS FOR 2D MOTION
        # ====================================================================
        # Create offset base if z_offset is specified
        if abs(self.z_offset) > 1e-6:
            offset_body = plant.AddRigidBody(
                "base_offset",
                model_instance,
                slider_inertia
            )
            plant.WeldFrames(
                plant.world_frame(),
                offset_body.body_frame(),
                RigidTransform([0.0, 0.0, self.z_offset])
            )
            parent_frame = offset_body.body_frame()
        else:
            parent_frame = plant.world_frame()
        
        # X-axis joint
        self.x_joint = plant.AddJoint(
            PrismaticJoint(
                name="cart_x",
                frame_on_parent=parent_frame,
                frame_on_child=self.x_slider_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.config.damping_cart
            )
        )
        
        # Y-axis joint
        self.y_joint = plant.AddJoint(
            PrismaticJoint(
                name="cart_y",
                frame_on_parent=self.x_slider_body.body_frame(),
                frame_on_child=self.y_slider_body.body_frame(),
                axis=[0.0, 1.0, 0.0],
                damping=self.config.damping_cart
            )
        )
        
        # Connect y_slider to cart (fixed)
        plant.WeldFrames(
            self.y_slider_body.body_frame(),
            self.cart_body.body_frame(),
            RigidTransform()
        )
        
        # Add actuators
        plant.AddJointActuator("force_x", self.x_joint)
        plant.AddJointActuator("force_y", self.y_joint)
        
        # ====================================================================
        # CREATE PENDULUM (GIMBAL MOUNT)
        # ====================================================================
        m_p = self.config.mass_pendulum
        L = self.config.length_pendulum
        r = 0.05
        
        # Pitch body (intermediate gimbal)
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            "pitch_gimbal",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body (ball at end)
        I_p = (2/5) * m_p * r**2 + m_p * L**2
        pendulum_inertia = SpatialInertia(
            mass=m_p,
            p_PScm_E=[0, 0, -L],
            G_SP_E=UnitInertia(I_p / m_p, I_p / m_p, (2/5) * r**2)
        )
        self.pendulum_body = plant.AddRigidBody(
            "pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Pitch joint (rotation about Y-axis)
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0, 1, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Roll joint (rotation about X-axis)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1, 0, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Pendulum visual geometry (only if SceneGraph is registered)
        if register_visuals:
            # Rod
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L/2]),
                Cylinder(radius=0.01, length=L),
                "pendulum_rod",
                np.array([0.6, 0.4, 0.2, 1.0])
            )
            
            # Ball
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L]),
                Sphere(r),
                "pendulum_ball",
                np.array([0.8, 0.2, 0.2, 1.0])
            )
    
    def build_plant_welded(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True):
        """
        Build cart-pendulum for welded mode (no world connection).
        
        Creates cart body as a free body (to be welded by caller) with 2-DOF gimbal pendulum attached.
        This avoids kinematic loops when cart is welded to manipulator EE.
        
        Args:
            plant: MultibodyPlant to add bodies to
            model_instance: Model instance index
            register_visuals: Whether to register visual geometry
            
        Returns:
            cart_body: The cart RigidBody (for welding by caller)
        """
        # ====================================================================
        # CREATE CART BODY (FREE BODY - NO WORLD CONNECTION)
        # ====================================================================
        cart_size = 0.1
        m_c = self.config.mass_cart
        
        # Cart inertia (box approximation)
        I_c = (1/12) * m_c * cart_size**2
        cart_inertia = SpatialInertia(
            mass=m_c,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_c, I_c, I_c)
        )
        
        self.cart_body = plant.AddRigidBody(
            "cart",
            model_instance,
            cart_inertia
        )
        
        # Cart visual geometry
        if register_visuals:
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                Sphere(cart_size / 2),
                "cart_visual",
                np.array([0.3, 0.3, 0.8, 1.0])
            )
        
        # ====================================================================
        # CREATE PENDULUM (2-DOF GIMBAL MOUNT)
        # ====================================================================
        m_p = self.config.mass_pendulum
        L = self.config.length_pendulum
        r = 0.05
        
        # Pitch body (intermediate gimbal)
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            "pitch_gimbal",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body (ball at end)
        I_p = (2/5) * m_p * r**2 + m_p * L**2
        pendulum_inertia = SpatialInertia(
            mass=m_p,
            p_PScm_E=[0, 0, -L],
            G_SP_E=UnitInertia(I_p / m_p, I_p / m_p, (2/5) * r**2)
        )
        self.pendulum_body = plant.AddRigidBody(
            "pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Pitch joint (rotation about Y-axis)
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0, 1, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Roll joint (rotation about X-axis)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1, 0, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Pendulum visual geometry
        if register_visuals:
            # Rod
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L/2]),
                Cylinder(radius=0.01, length=L),
                "pendulum_rod",
                np.array([0.6, 0.4, 0.2, 1.0])
            )
            
            # Ball
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L]),
                Sphere(r),
                "pendulum_ball",
                np.array([0.8, 0.2, 0.2, 1.0])
            )
        
        return self.cart_body