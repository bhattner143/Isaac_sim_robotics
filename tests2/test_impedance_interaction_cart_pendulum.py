#!/usr/bin/env python3
"""
Test Script: Passive Cart Driven by Manipulator End Effector via Virtual Mass

This script demonstrates:
1. Cup manipulator following a reference joint trajectory
2. Virtual mass (admittance dynamics) between EE and cart
3. Impedance control makes cart track virtual mass position
4. Meshcat 3D visualization

System Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: COUPLING FORCE (EE ← Cart)
    F_coupling = -K_coupling (x_ee - x_cart) - D_coupling (ẋ_ee - ẋ_cart)
    
    This force represents the "push/pull" between manipulator and cart.

Step 2: VIRTUAL MASS (Admittance Dynamics)
    M_v ẍ_des + D_v ẋ_des + K_v (x_des - x₀) = F_coupling
    
    The virtual mass "feels" the coupling force and produces desired cart motion.
    Acts as compliant buffer - heavy mass = slow response, light mass = fast.

Step 3: IMPEDANCE CONTROL (Cart Tracking)
    F_cart = K_p (x_des - x_cart) + K_d (ẋ_des - ẋ_cart)
    
    Applied to cart to make it follow the virtual mass trajectory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Physical Interpretation:
• Manipulator moves according to joint trajectory (independent)
• EE position generates coupling force when different from cart
• Virtual mass integrates this force to produce smooth desired motion
• Cart impedance control tracks this desired motion
• Result: Compliant, smooth interaction with tunable dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    Simulator,
    VectorLogSink,
    PiecewisePolynomial,
    LeafSystem,
    BasicVector,
    AbstractValue,
    ExternallyAppliedSpatialForce,
    SpatialForce,
    MeshcatVisualizer,
    StartMeshcat,
    SceneGraph,
    AddMultibodyPlantSceneGraph,
)
from pydrake.multibody.tree import JacobianWrtVariable
from termcolor import colored

# Import from main script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.robot.robot_types import create_cup_manipulator_config, create_cart_pendulum_config
from script_cup_manipulator_controller_ofc import CupManipulator, CartPendulum3D


class CouplingForceComputer(LeafSystem):
    """
    Computes coupling force between end effector and cart.
    
    F_coupling = -K_coupling (x_ee - x_cart) - D_coupling (ẋ_ee - ẋ_cart)
    
    This represents the "virtual spring-damper" connecting EE and cart.
    The force is fed into the virtual mass (admittance dynamics).
    
    Inputs:
    - End effector position (2D): [x_ee, y_ee]
    - End effector velocity (2D): [vx_ee, vy_ee]
    - Cart position (2D): [x_cart, y_cart]
    - Cart velocity (2D): [vx_cart, vy_cart]
    
    Outputs:
    - Coupling force (2D): [Fx, Fy]
    """
    
    def __init__(self, k_coupling=50.0, d_coupling=10.0):
        LeafSystem.__init__(self)
        
        self.k_coupling = k_coupling
        self.d_coupling = d_coupling
        
        # Input ports
        self.ee_pos_input = self.DeclareVectorInputPort("ee_position", BasicVector(2))
        self.ee_vel_input = self.DeclareVectorInputPort("ee_velocity", BasicVector(2))
        self.cart_pos_input = self.DeclareVectorInputPort("cart_position", BasicVector(2))
        self.cart_vel_input = self.DeclareVectorInputPort("cart_velocity", BasicVector(2))
        
        # Output port
        self.DeclareVectorOutputPort(
            "coupling_force",
            BasicVector(2),
            self.CalcCouplingForce
        )
    
    def CalcCouplingForce(self, context, output):
        """Compute coupling force between EE and cart."""
        ee_pos = self.ee_pos_input.Eval(context)
        ee_vel = self.ee_vel_input.Eval(context)
        cart_pos = self.cart_pos_input.Eval(context)
        cart_vel = self.cart_vel_input.Eval(context)
        
        # Compute coupling force: F = -K*(x_ee - x_cart) - D*(v_ee - v_cart)
        position_error = ee_pos - cart_pos
        velocity_error = ee_vel - cart_vel
        
        force = -self.k_coupling * position_error - self.d_coupling * velocity_error
        
        # Saturate force
        max_force = 200.0
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > max_force:
            force = force * (max_force / force_magnitude)
        
        output.SetFromVector(force)


class VirtualMassAdmittance(LeafSystem):
    """
    Virtual mass with admittance dynamics between EE and cart.
    
    ═══════════════════════════════════════════════════════════════════════
    DYNAMICS
    ═══════════════════════════════════════════════════════════════════════
    
    M_v ẍ_des + D_v ẋ_des + K_v (x_des - x₀) = F_coupling
    
    Where:
        x_des : Desired cart position (output of virtual mass) [m]
        ẋ_des : Desired cart velocity [m/s]
        F_coupling : Coupling force from EE-cart interaction [N]
        M_v : Virtual mass [kg]
        D_v : Virtual damping [N·s/m]
        K_v : Virtual stiffness [N/m]
        x₀ : Equilibrium position [m]
    
    ═══════════════════════════════════════════════════════════════════════
    STATE-SPACE FORM
    ═══════════════════════════════════════════════════════════════════════
    
    State: s = [x_des, y_des, vx_des, vy_des]ᵀ ∈ ℝ⁴
    
    Dynamics:
        ẋ_des = v_des
        v̇_des = M_v⁻¹ (F_coupling - D_v v_des - K_v (x_des - x₀))
    
    ═══════════════════════════════════════════════════════════════════════
    PHYSICAL INTERPRETATION
    ═══════════════════════════════════════════════════════════════════════
    
    The virtual mass acts as a compliant buffer:
    • High M_v → slow, smooth response (feels heavy)
    • Low M_v → fast, responsive (feels light)
    • D_v provides damping to prevent oscillations
    • K_v creates restoring force toward equilibrium
    
    This filters the interaction force, creating smooth desired motion
    for the cart to follow.
    """
    
    def __init__(self, M_virtual=2.0, D_virtual=5.0, K_virtual=10.0, x0=None):
        LeafSystem.__init__(self)
        
        self.M_v = M_virtual
        self.D_v = D_virtual
        self.K_v = K_virtual
        self.x0 = x0 if x0 is not None else np.zeros(2)
        
        # Input port: coupling force
        self.force_input = self.DeclareVectorInputPort("coupling_force", BasicVector(2))
        
        # Continuous state: [x_des, y_des, vx_des, vy_des]
        self.DeclareContinuousState(4)
        
        # Output ports
        self.DeclareVectorOutputPort(
            "desired_cart_position",
            BasicVector(2),
            self.OutputDesiredPosition
        )
        self.DeclareVectorOutputPort(
            "desired_cart_velocity",
            BasicVector(2),
            self.OutputDesiredVelocity
        )
    
    def SetDefaultState(self, context, state):
        """Initialize state at equilibrium."""
        # [x_des, y_des, vx_des, vy_des] = [x0, 0, 0]
        state.SetFromVector(np.array([self.x0[0], self.x0[1], 0.0, 0.0]))
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """Compute state derivatives: admittance dynamics."""
        # Get current state
        state = context.get_continuous_state_vector().CopyToVector()
        x_des = state[0:2]   # [x, y]
        v_des = state[2:4]   # [vx, vy]
        
        # Get coupling force
        F_coupling = self.force_input.Eval(context)
        
        # Admittance dynamics: M ẍ + D ẋ + K (x - x₀) = F
        # Solve for acceleration:
        a_des = (F_coupling - self.D_v * v_des - self.K_v * (x_des - self.x0)) / self.M_v
        
        # State derivatives: [ẋ, ẏ, v̇x, v̇y]
        derivatives.SetFromVector(np.concatenate([v_des, a_des]))
    
    def OutputDesiredPosition(self, context, output):
        """Output desired cart position."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[0:2])
    
    def OutputDesiredVelocity(self, context, output):
        """Output desired cart velocity."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[2:4])


class CartImpedanceController(LeafSystem):
    """
    Impedance controller to make cart track desired position from virtual mass.
    
    F_cart = K_p (x_des - x_cart) + K_d (ẋ_des - ẋ_cart)
    
    Inputs:
    - Desired cart position (2D): [x_des, y_des]
    - Desired cart velocity (2D): [vx_des, vy_des]
    - Actual cart position (2D): [x_cart, y_cart]
    - Actual cart velocity (2D): [vx_cart, vy_cart]
    
    Outputs:
    - Control force (2D): [Fx, Fy]
    """
    
    def __init__(self, kp=100.0, kd=20.0):
        LeafSystem.__init__(self)
        
        self.kp = kp
        self.kd = kd
        
        # Input ports
        self.des_pos_input = self.DeclareVectorInputPort("desired_position", BasicVector(2))
        self.des_vel_input = self.DeclareVectorInputPort("desired_velocity", BasicVector(2))
        self.cart_pos_input = self.DeclareVectorInputPort("cart_position", BasicVector(2))
        self.cart_vel_input = self.DeclareVectorInputPort("cart_velocity", BasicVector(2))
        
        # Output port
        self.DeclareVectorOutputPort(
            "control_force",
            BasicVector(2),
            self.CalcControlForce
        )
    
    def CalcControlForce(self, context, output):
        """Compute PD control force for cart tracking."""
        x_des = self.des_pos_input.Eval(context)
        v_des = self.des_vel_input.Eval(context)
        x_cart = self.cart_pos_input.Eval(context)
        v_cart = self.cart_vel_input.Eval(context)
        
        # PD control
        position_error = x_des - x_cart
        velocity_error = v_des - v_cart
        
        force = self.kp * position_error + self.kd * velocity_error
        
        # Saturate force
        max_force = 200.0
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > max_force:
            force = force * (max_force / force_magnitude)
        
        output.SetFromVector(force)


class EndEffectorKinematics(LeafSystem):
    """
    Compute end effector position and velocity using Drake's built-in methods.
    
    Uses manipulator's CalcPosition method which wraps Drake's forward kinematics.
    
    Inputs:
    - Manipulator state (4D): [q1, q2, q1_dot, q2_dot]
    
    Outputs:
    - End effector position (2D): [x_ee, y_ee]
    - End effector velocity (2D): [vx_ee, vy_ee]
    """
    
    def __init__(self, plant: MultibodyPlant, manipulator: CupManipulator):
        """
        Initialize end effector kinematics using manipulator's custom methods.
        
        Args:
            plant: MultibodyPlant instance
            manipulator: CupManipulator instance with CalcPosition method
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        # Create context for plant queries
        self.plant_context = plant.CreateDefaultContext()
        
        # Input port: [q1, q2, q1_dot, q2_dot]
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output ports
        self.DeclareVectorOutputPort(
            "ee_position",
            BasicVector(2),
            self.CalcPosition
        )
        self.DeclareVectorOutputPort(
            "ee_velocity",
            BasicVector(2),
            self.CalcVelocity
        )
    
    def CalcPosition(self, context, output):
        """Compute end effector position using manipulator's CalcPosition method."""
        state = self.state_input.Eval(context)
        q = state[:2]  # [q1, q2]
        
        # Set plant state
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        
        # Use manipulator's CalcPosition which includes EE_OFFSET automatically
        ee_pos_world = self.manipulator.CalcPosition(self.plant, self.plant_context)
        
        # Extract X-Y position (planar projection)
        output.SetFromVector([ee_pos_world[0], ee_pos_world[1]])
    
    def CalcVelocity(self, context, output):
        """Compute end effector velocity using Drake's Jacobian."""
        state = self.state_input.Eval(context)
        q = state[:2]   # [q1, q2]
        v = state[2:]   # [q1_dot, q2_dot]
        
        # Set plant state
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        self.plant.SetVelocities(self.plant_context, self.model_instance, v)
        
        # Compute Jacobian for end effector origin point
        ee_origin_in_body = np.zeros(3)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            ee_origin_in_body,
            self.world_frame,
            self.world_frame
        )
        
        # Extract translational part (last 3 rows) and project to X-Y
        # Spatial velocity Jacobian: [angular; linear]
        J_translational = J_spatial[3:6, :]  # Linear velocity part
        
        # Get only the manipulator DOFs (first 2 columns for 2-DOF manipulator)
        J_manip = J_translational[:, :2]
        
        # Compute end effector velocity: v_ee = J * q_dot
        v_ee_world = J_manip @ v
        
        # Extract X-Y velocity
        output.SetFromVector([v_ee_world[0], v_ee_world[1]])


class JointTrajectoryController(LeafSystem):
    """
    Simple PD controller for joint trajectory tracking.
    
    Inputs:
    - Manipulator state (4D): [q1, q2, q1_dot, q2_dot]
    - Desired state (4D): [q1_des, q2_des, q1_dot_des, q2_dot_des]
    
    Outputs:
    - Joint torques (2D): [tau1, tau2]
    """
    
    def __init__(self, kp=100.0, kd=20.0):
        """
        Initialize PD controller.
        
        Args:
            kp: Position gain
            kd: Velocity gain
        """
        LeafSystem.__init__(self)
        
        self.kp = kp
        self.kd = kd
        
        # Input ports
        self.state_input = self.DeclareVectorInputPort("current_state", BasicVector(4))
        self.desired_input = self.DeclareVectorInputPort("desired_state", BasicVector(4))
        
        # Output port
        self.DeclareVectorOutputPort(
            "torque_output",
            BasicVector(2),
            self.CalcTorque
        )
    
    def CalcTorque(self, context, output):
        """Compute PD control torque."""
        current = self.state_input.Eval(context)
        desired = self.desired_input.Eval(context)
        
        # PD control: tau = Kp*(q_des - q) + Kd*(q_dot_des - q_dot)
        q_error = desired[:2] - current[:2]
        q_dot_error = desired[2:] - current[2:]
        
        torque = self.kp * q_error + self.kd * q_dot_error
        
        output.SetFromVector(torque)


def create_reference_trajectory(duration=10.0):
    """
    Create a reference joint trajectory for the manipulator.
    
    Simple linear motion in joint space:
    - q1: -10° to -30° (-0.175 to -0.524 rad)
    - q2: +20° to +80° (+0.349 to +1.396 rad)
    
    Args:
        duration: Trajectory duration [s]
        
    Returns:
        trajectory: PiecewisePolynomial trajectory
    """
    # Time points
    num_points = 11
    times = np.linspace(0, duration, num_points)
    
    # Linear motion in joint space
    q1_start = np.deg2rad(-10.0)  # -10 degrees
    q1_end = np.deg2rad(-30.0)    # -30 degrees
    q2_start = np.deg2rad(20.0)   # +20 degrees
    q2_end = np.deg2rad(60.0)     # +80 degrees
    
    q1_trajectory = np.linspace(q1_start, q1_end, num_points)
    q2_trajectory = np.linspace(q2_start, q2_end, num_points)
    
    # Stack into position matrix [2 x num_points]
    positions = np.vstack([q1_trajectory, q2_trajectory])
    
    # Create piecewise polynomial (cubic spline)
    trajectory = PiecewisePolynomial.CubicShapePreserving(
        times, positions, zero_end_point_derivatives=True
    )
    
    return trajectory


def test_manipulator_kinematics(q_initial=(0.0, 0.0), q_final=(np.pi/4, np.pi/3), num_points=20):
    """
    Test manipulator forward kinematics: joint trajectory → end effector positions.
    
    No dynamics simulation - just pure kinematics visualization.
    
    Args:
        q_initial: Initial joint angles [q1, q2] in radians
        q_final: Final joint angles [q1, q2] in radians
        num_points: Number of points along trajectory
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR KINEMATICS TEST: JOINT TRAJECTORY → EE POSITIONS", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Create plant for kinematics queries
    plant = MultibodyPlant(time_step=0.0)  # Continuous plant for kinematics
    parser = Parser(plant)
    
    # Load manipulator
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(0.0, 0.0),
        damping=(0.0, 0.0),
        friction=(0.0, 0.0),
    )
    
    manipulator = CupManipulator(manipulator_config, enable_visualization=False)
    manipulator.load_urdf_to_plant(plant, parser)
    manipulator.weld_base_to_world(plant)
    
    plant.Finalize()
    context = plant.CreateDefaultContext()
    
    print(colored(f"\n✓ Plant created for kinematics", "green"))
    print(colored(f"  DOF: {plant.num_positions()}", "cyan"))
    
    # Get end effector body
    ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
    world_frame = plant.world_frame()
    
    # Create joint trajectory
    q1_traj = np.linspace(q_initial[0], q_final[0], num_points)
    q2_traj = np.linspace(q_initial[1], q_final[1], num_points)
    
    # Compute end effector positions along trajectory
    ee_positions = []
    for i in range(num_points):
        q = np.array([q1_traj[i], q2_traj[i]])
        plant.SetPositions(context, manipulator.model_instance, q)
        
        # Forward kinematics: joint angles → EE position
        ee_origin = np.zeros(3)
        ee_pos = plant.CalcPointsPositions(
            context,
            ee_body.body_frame(),
            ee_origin,
            world_frame
        ).flatten()
        
        ee_positions.append(ee_pos)
    
    ee_positions = np.array(ee_positions)
    
    # Print results
    print(colored(f"\n📐 Kinematic Analysis:", "yellow", attrs=["bold"]))
    print(colored(f"  Initial joints: q1={np.rad2deg(q_initial[0]):.1f}°, q2={np.rad2deg(q_initial[1]):.1f}°", "cyan"))
    print(colored(f"  Final joints:   q1={np.rad2deg(q_final[0]):.1f}°, q2={np.rad2deg(q_final[1]):.1f}°", "cyan"))
    print(colored(f"\n  Initial EE pos: X={ee_positions[0, 0]:.4f} m, Y={ee_positions[0, 1]:.4f} m, Z={ee_positions[0, 2]:.4f} m", "green"))
    print(colored(f"  Final EE pos:   X={ee_positions[-1, 0]:.4f} m, Y={ee_positions[-1, 1]:.4f} m, Z={ee_positions[-1, 2]:.4f} m", "green"))
    
    ee_displacement = np.linalg.norm(ee_positions[-1] - ee_positions[0])
    print(colored(f"\n  EE displacement: {ee_displacement:.4f} m", "yellow"))
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Joint angles vs step
    ax1 = axes[0, 0]
    ax1.plot(np.rad2deg(q1_traj), 'b-o', linewidth=2, markersize=4, label='q₁')
    ax1.plot(np.rad2deg(q2_traj), 'r-s', linewidth=2, markersize=4, label='q₂')
    ax1.set_xlabel('Step', fontweight='bold')
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
    ax1.set_title('Joint Trajectory', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: EE position components vs step
    ax2 = axes[0, 1]
    ax2.plot(ee_positions[:, 0], 'b-o', linewidth=2, markersize=4, label='X')
    ax2.plot(ee_positions[:, 1], 'r-s', linewidth=2, markersize=4, label='Y')
    ax2.plot(ee_positions[:, 2], 'g-^', linewidth=2, markersize=4, label='Z')
    ax2.set_xlabel('Step', fontweight='bold')
    ax2.set_ylabel('Position [m]', fontweight='bold')
    ax2.set_title('End Effector Position Components', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: EE trajectory in X-Y plane
    ax3 = axes[1, 0]
    ax3.plot(ee_positions[:, 0], ee_positions[:, 1], 'purple', linewidth=2)
    ax3.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=12, label='Start')
    ax3.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=12, label='End')
    ax3.set_xlabel('X Position [m]', fontweight='bold')
    ax3.set_ylabel('Y Position [m]', fontweight='bold')
    ax3.set_title('End Effector Trajectory (X-Y Plane)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.legend()
    
    # Plot 4: EE trajectory in 3D projection (X-Z plane)
    ax4 = axes[1, 1]
    ax4.plot(ee_positions[:, 0], ee_positions[:, 2], 'purple', linewidth=2)
    ax4.plot(ee_positions[0, 0], ee_positions[0, 2], 'go', markersize=12, label='Start')
    ax4.plot(ee_positions[-1, 0], ee_positions[-1, 2], 'ro', markersize=12, label='End')
    ax4.set_xlabel('X Position [m]', fontweight='bold')
    ax4.set_ylabel('Z Position [m]', fontweight='bold')
    ax4.set_title('End Effector Trajectory (X-Z Plane)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.legend()
    
    plt.suptitle('Forward Kinematics: Joint Space → Task Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print(colored("\n✓ Plots generated", "green"))
    
    return {
        'q1': q1_traj,
        'q2': q2_traj,
        'ee_positions': ee_positions,
    }


def simulate_passive_cart(
    duration=10.0,
    k_coupling=50.0,
    d_coupling=10.0,
    M_virtual=2.0,
    D_virtual=5.0,
    K_virtual=10.0,
    kp_cart=100.0,
    kd_cart=20.0,
):
    """
    Simulate passive cart with virtual mass between EE and cart.
    
    Args:
        duration: Simulation duration [s]
        k_coupling: Coupling stiffness between EE and cart [N/m]
        d_coupling: Coupling damping between EE and cart [N·s/m]
        M_virtual: Virtual mass [kg]
        D_virtual: Virtual damping [N·s/m]
        K_virtual: Virtual stiffness [N/m]
        kp_cart: Cart impedance position gain [N/m]
        kd_cart: Cart impedance velocity gain [N·s/m]
        
    Returns:
        log_data: Dictionary with logged data
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("VIRTUAL MASS INTERACTION: MANIPULATOR ← VIRTUAL MASS ← CART", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Start Meshcat server
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat server started at: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Create configurations
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(0.0, 0.0),
        damping=(0.5, 0.5),
        friction=(0.05, 0.05),
    )
    
    cart_pendulum_config = create_cart_pendulum_config(
        cart_mass=5.0,  # Increased from 1.0 to 5.0 kg for more inertia
        cart_size=0.1,
        cart_damping=2.0,  # Increased damping for stability
        pendulum_mass=0.5,
        pendulum_length=0.2,
        pendulum_radius=0.05,
        pendulum_damping=0.05,
        attachment_offset=(0.0, 0.0, 0.0),
        initial_cart_x=0.0,  # Will be set dynamically from EE position
        initial_cart_y=0.0,  # Will be set dynamically from EE position
        initial_pitch=0.0,
        initial_roll=0.0,
        name="cart_pendulum"
    )
    
    # Build system
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Add cup manipulator
    manipulator = CupManipulator(manipulator_config, enable_visualization=False)
    parser = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser)
    manipulator.weld_base_to_world(plant)
    
    # Add actuators to manipulator joints
    manip_joint1 = plant.GetJointByName("link1_base", manipulator.model_instance)
    manip_joint2 = plant.GetJointByName("link2_link1", manipulator.model_instance)
    plant.AddJointActuator("joint1_actuator", manip_joint1)
    plant.AddJointActuator("joint2_actuator", manip_joint2)
    
    # Add cart-pendulum (PASSIVE - no actuators, only driven by impedance forces)
    cart_pendulum = CartPendulum3D(
        cart_pendulum_config, 
        visualize_cart=True,
        add_cart_actuators=False  # Cart is passive - only impedance forces
    )
    model_instance_cart = plant.AddModelInstance("cart_pendulum_model")
    cart_pendulum.attach_to_plant(plant, model_instance_cart, register_visuals=False)
    
    # Finalize plant
    plant.Finalize()
    
    print(colored(f"\n✓ System created", "green"))
    print(colored(f"  Manipulator DOF: 2 (actuated)", "cyan"))
    print(colored(f"  Cart-Pendulum DOF: 4 (cart: 2 passive, pendulum: 2 passive)", "cyan"))
    print(colored(f"  Total DOF: {plant.num_positions()}", "cyan"))
    print(colored(f"  Total Actuators: {plant.num_actuators()} (manipulator only)", "cyan"))
    print(colored(f"\nCoupling Parameters (EE ← Cart):", "yellow", attrs=["bold"]))
    print(colored(f"  K_coupling: {k_coupling:.1f} N/m", "cyan"))
    print(colored(f"  D_coupling: {d_coupling:.1f} N·s/m", "cyan"))
    print(colored(f"\nVirtual Mass Parameters:", "yellow", attrs=["bold"]))
    print(colored(f"  M_virtual: {M_virtual:.2f} kg", "cyan"))
    print(colored(f"  D_virtual: {D_virtual:.2f} N·s/m", "cyan"))
    print(colored(f"  K_virtual: {K_virtual:.2f} N/m", "cyan"))
    print(colored(f"\nCart Impedance Gains:", "yellow", attrs=["bold"]))
    print(colored(f"  Kp: {kp_cart:.1f} N/m", "cyan"))
    print(colored(f"  Kd: {kd_cart:.1f} N·s/m", "cyan"))
    
    # Create reference trajectory
    trajectory = create_reference_trajectory(duration)
    
    # Create trajectory source (outputs desired state)
    from pydrake.systems.primitives import TrajectorySource
    trajectory_source = builder.AddSystem(TrajectorySource(trajectory, output_derivative_order=1))
    
    # Create end effector kinematics using manipulator's CalcPosition method
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(
        plant=plant,
        manipulator=manipulator  # Uses manipulator.CalcPosition with EE_OFFSET
    ))
    
    # Create joint trajectory controller
    joint_controller = builder.AddSystem(JointTrajectoryController(kp=100.0, kd=20.0))
    
    # Get initial cart position for virtual mass equilibrium
    # Use config values (will be 0.0, then updated based on actual EE position)
    initial_cart_x = cart_pendulum_config.initial_cart_x
    initial_cart_y = cart_pendulum_config.initial_cart_y
    temp_context = plant.CreateDefaultContext()
    temp_plant_context = temp_context
    plant.SetPositions(temp_plant_context, model_instance_cart, 
                      np.array([initial_cart_x, initial_cart_y, 0.0, 0.0]))
    x0_virtual = np.array([initial_cart_x, initial_cart_y])
    
    # Create coupling force computer (EE ← Cart)
    coupling_force = builder.AddSystem(
        CouplingForceComputer(k_coupling=k_coupling, d_coupling=d_coupling)
    )
    
    # Create virtual mass (admittance dynamics)
    virtual_mass = builder.AddSystem(
        VirtualMassAdmittance(
            M_virtual=M_virtual,
            D_virtual=D_virtual,
            K_virtual=K_virtual,
            x0=x0_virtual
        )
    )
    
    # Create cart impedance controller
    cart_controller = builder.AddSystem(
        CartImpedanceController(kp=kp_cart, kd=kd_cart)
    )
    
    # Create demux for plant state (manipulator + cart-pendulum)
    from pydrake.systems.primitives import Demultiplexer
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))  # [manip_state, cart_pend_state]
    
    # Create demux for cart-pendulum state
    cart_pend_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [x,y] [pitch,roll] [xdot,ydot] [pdot,rdot]
    
    # Connect trajectory source to joint controller
    builder.Connect(
        trajectory_source.get_output_port(),
        joint_controller.GetInputPort("desired_state")
    )
    
    # Connect plant state to demux
    builder.Connect(
        plant.get_state_output_port(), # Full state: [manipulator (4), cart-pendulum (8)]
        state_demux.get_input_port()
    )
    
    # Connect manipulator state (first 4 states) to joint controller and EE kinematics
    builder.Connect(
        state_demux.get_output_port(0),  # Manipulator state [q1, q2, q1_dot, q2_dot]
        joint_controller.GetInputPort("current_state")
    )
    # Also connect manipulator state to EE kinematics
    builder.Connect(
        state_demux.get_output_port(0),
        ee_kinematics.GetInputPort("manipulator_state")
    )
    
    # Connect cart-pendulum state to demux
    builder.Connect(
        state_demux.get_output_port(1),  # Cart-pendulum state
        cart_pend_demux.get_input_port()
    )
    
    # Connect EE and cart to coupling force computer
    builder.Connect(
        ee_kinematics.GetOutputPort("ee_position"),
        coupling_force.GetInputPort("ee_position")
    )
    builder.Connect(
        ee_kinematics.GetOutputPort("ee_velocity"),
        coupling_force.GetInputPort("ee_velocity")
    )
    builder.Connect(
        cart_pend_demux.get_output_port(0),  # Cart position [x, y]
        coupling_force.GetInputPort("cart_position")
    )
    builder.Connect(
        cart_pend_demux.get_output_port(2),  # Cart velocity [x_dot, y_dot]
        coupling_force.GetInputPort("cart_velocity")
    )
    
    # Connect coupling force to virtual mass
    builder.Connect(
        coupling_force.get_output_port(),
        virtual_mass.GetInputPort("coupling_force")
    )
    
    # Connect virtual mass and cart to impedance controller
    builder.Connect(
        virtual_mass.GetOutputPort("desired_cart_position"),
        cart_controller.GetInputPort("desired_position")
    )
    builder.Connect(
        virtual_mass.GetOutputPort("desired_cart_velocity"),
        cart_controller.GetInputPort("desired_velocity")
    )
    builder.Connect(
        cart_pend_demux.get_output_port(0),  # Cart position [x, y]
        cart_controller.GetInputPort("cart_position")
    )
    builder.Connect(
        cart_pend_demux.get_output_port(2),  # Cart velocity [x_dot, y_dot]
        cart_controller.GetInputPort("cart_velocity")
    )
    
    # Apply impedance force to cart only (manipulator gets torques via ID)
    cart_body = cart_pendulum.cart_body
    
    # Create system to apply force only to cart
    class CartForceApplicator(LeafSystem):
        """Applies impedance force to cart body only."""
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("impedance_force", 2)  # [Fx, Fy]
            self.DeclareAbstractOutputPort(
                "spatial_forces",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self.CalcSpatialForces
            )
        
        def CalcSpatialForces(self, context, output):
            force_2d = self.get_input_port(0).Eval(context)
            
            # Force on cart
            cart_force = ExternallyAppliedSpatialForce()
            cart_force.body_index = cart_body.index()
            cart_force.F_Bq_W = SpatialForce(
                tau=np.zeros(3),
                f=np.array([force_2d[0], force_2d[1], 0.0])
            )
            cart_force.p_BoBq_B = np.zeros(3)
            
            output.set_value([cart_force])
    
    force_applicator = builder.AddSystem(CartForceApplicator())
    
    # Connect cart controller to force applicator (for cart)
    builder.Connect(
        cart_controller.get_output_port(),
        force_applicator.get_input_port(0)
    )
    
    # Connect force applicator to plant's applied spatial forces input
    builder.Connect(
        force_applicator.get_output_port(0),
        plant.get_applied_spatial_force_input_port()
    )
    
    # Connect joint controller to plant actuation (manipulator only)
    builder.Connect(
        joint_controller.get_output_port(),
        plant.get_actuation_input_port()
    )
    
    # Add Meshcat visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Add loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    cart_force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(cart_controller.get_output_port(), cart_force_logger.get_input_port())
    
    coupling_force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(coupling_force.get_output_port(), coupling_force_logger.get_input_port())
    
    virtual_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(virtual_mass.GetOutputPort("desired_cart_position"), virtual_pos_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    # Manipulator initial angles: match trajectory starting point
    q1_init = np.deg2rad(-10.0)  # Start at -10 degrees
    q2_init = np.deg2rad(20.0)   # Start at +20 degrees
    
    # Compute initial EE position using manipulator's CalcPosition method
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, manipulator.model_instance, np.array([q1_init, q2_init]))
    ee_pos_init = manipulator.CalcPosition(plant, plant_context)
    x_ee_init = ee_pos_init[0]
    y_ee_init = ee_pos_init[1]
    
    # Set full plant state: cart positioned at end effector location
    plant.SetPositions(plant_context, np.array([
        q1_init, q2_init,     # Manipulator joints
        x_ee_init, y_ee_init, # Cart position (exactly at EE initial position)
        0.0, 0.0              # Pendulum angles
    ]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    print(colored(f"\nInitial State:", "cyan"))
    print(colored(f"  Manipulator: q1={np.rad2deg(q1_init):.1f}° ({q1_init:.3f} rad), q2={np.rad2deg(q2_init):.1f}° ({q2_init:.3f} rad)", "cyan"))
    print(colored(f"  End-effector: x={x_ee_init:.3f} m, y={y_ee_init:.3f} m (using CalcPosition with EE_OFFSET)", "cyan"))
    print(colored(f"  Cart: x={x_ee_init:.3f} m, y={y_ee_init:.3f} m (aligned with EE)", "cyan"))
    print(colored(f"  Pendulum: pitch={0.0:.3f} rad, roll={0.0:.3f} rad", "cyan"))
    
    # Start recording
    visualizer.StartRecording()
    
    print(colored(f"\nSimulating for {duration} s...", "yellow"))
    simulator.AdvanceTo(duration)
    print(colored("✓ Simulation complete", "green"))
    
    # Publish recording
    visualizer.PublishRecording()
    print(colored(f"\n🎬 Animation published to Meshcat: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    cart_force_log = cart_force_logger.FindLog(context)
    coupling_force_log = coupling_force_logger.FindLog(context)
    virtual_pos_log = virtual_pos_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    
    time_data = state_log.sample_times()
    state_data = state_log.data()
    cart_force_data = cart_force_log.data()
    coupling_force_data = coupling_force_log.data()
    virtual_pos_data = virtual_pos_log.data()
    ee_pos_data = ee_pos_log.data()
    
    # Parse state data
    # State: [q1, q2, cart_x, cart_y, pitch, roll, q1_dot, q2_dot, cart_xdot, cart_ydot, pitch_dot, roll_dot]
    q1 = state_data[0, :]
    q2 = state_data[1, :]
    cart_x = state_data[2, :]
    cart_y = state_data[3, :]
    pitch = state_data[4, :]
    roll = state_data[5, :]
    
    ee_x = ee_pos_data[0, :]
    ee_y = ee_pos_data[1, :]
    
    cart_force_x = cart_force_data[0, :]
    cart_force_y = cart_force_data[1, :]
    
    coupling_force_x = coupling_force_data[0, :]
    coupling_force_y = coupling_force_data[1, :]
    
    virtual_x = virtual_pos_data[0, :]
    virtual_y = virtual_pos_data[1, :]
    
    return {
        'time': time_data,
        'q1': q1,
        'q2': q2,
        'cart_x': cart_x,
        'cart_y': cart_y,
        'pitch': pitch,
        'roll': roll,
        'ee_x': ee_x,
        'ee_y': ee_y,
        'virtual_x': virtual_x,
        'virtual_y': virtual_y,
        'cart_force_x': cart_force_x,
        'cart_force_y': cart_force_y,
        'coupling_force_x': coupling_force_x,
        'coupling_force_y': coupling_force_y,
    }


def plot_results(log_data):
    """Plot simulation results with virtual mass visualization."""
    print(colored("\nGenerating plots...", "cyan"))
    
    t = log_data['time']
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Joint angles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, np.rad2deg(log_data['q1']), 'b-', linewidth=2, label='q₁')
    ax1.plot(t, np.rad2deg(log_data['q2']), 'r-', linewidth=2, label='q₂')
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_title('Manipulator Joint Angles', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Coupling force
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, log_data['coupling_force_x'], 'b-', linewidth=2, label='Fₓ coupling')
    ax2.plot(t, log_data['coupling_force_y'], 'r-', linewidth=2, label='Fᵧ coupling')
    ax2.set_ylabel('Force [N]', fontweight='bold')
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_title('Coupling Force (EE ← Cart)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: X positions (EE, Virtual, Cart)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, log_data['ee_x'], 'b-', linewidth=2.5, label='EE X')
    ax3.plot(t, log_data['virtual_x'], 'g--', linewidth=2, label='Virtual X')
    ax3.plot(t, log_data['cart_x'], 'c:', linewidth=2.5, label='Cart X')
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('X Position [m]', fontweight='bold')
    ax3.set_title('EE → Virtual Mass → Cart (X)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Y positions (EE, Virtual, Cart)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, log_data['ee_y'], 'b-', linewidth=2.5, label='EE Y')
    ax4.plot(t, log_data['virtual_y'], 'g--', linewidth=2, label='Virtual Y')
    ax4.plot(t, log_data['cart_y'], 'c:', linewidth=2.5, label='Cart Y')
    ax4.set_xlabel('Time [s]', fontweight='bold')
    ax4.set_ylabel('Y Position [m]', fontweight='bold')
    ax4.set_title('EE → Virtual Mass → Cart (Y)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Cart control forces
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, log_data['cart_force_x'], 'b-', linewidth=2, label='Fₓ cart')
    ax5.plot(t, log_data['cart_force_y'], 'r-', linewidth=2, label='Fᵧ cart')
    ax5.set_ylabel('Force [N]', fontweight='bold')
    ax5.set_xlabel('Time [s]', fontweight='bold')
    ax5.set_title('Cart Impedance Control Forces', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Paths in X-Y plane
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(log_data['ee_x'], log_data['ee_y'], 'b-', linewidth=2, alpha=0.6, label='EE')
    ax6.plot(log_data['virtual_x'], log_data['virtual_y'], 'g--', linewidth=2.5, alpha=0.7, label='Virtual')
    ax6.plot(log_data['cart_x'], log_data['cart_y'], 'c-', linewidth=3, alpha=0.8, label='Cart')
    ax6.plot(log_data['cart_x'][0], log_data['cart_y'][0], 'go', markersize=12, label='Start')
    ax6.plot(log_data['cart_x'][-1], log_data['cart_y'][-1], 'ro', markersize=12, label='End')
    ax6.set_xlabel('X Position [m]', fontweight='bold')
    ax6.set_ylabel('Y Position [m]', fontweight='bold')
    ax6.set_title('Paths: EE → Virtual → Cart', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.legend()
    
    # Plot 7: Pendulum angles
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, np.rad2deg(log_data['pitch']), 'g-', linewidth=2, label='Pitch')
    ax7.plot(t, np.rad2deg(log_data['roll']), 'm-', linewidth=2, label='Roll')
    ax7.set_ylabel('Angle [deg]', fontweight='bold')
    ax7.set_xlabel('Time [s]', fontweight='bold')
    ax7.set_title('Pendulum Angles', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.legend()
    
    # Plot 8: Separation distances
    ax8 = fig.add_subplot(gs[2, 1])
    ee_virtual_dist = np.sqrt((log_data['ee_x'] - log_data['virtual_x'])**2 + 
                              (log_data['ee_y'] - log_data['virtual_y'])**2)
    virtual_cart_dist = np.sqrt((log_data['virtual_x'] - log_data['cart_x'])**2 + 
                                (log_data['virtual_y'] - log_data['cart_y'])**2)
    ee_cart_dist = np.sqrt((log_data['ee_x'] - log_data['cart_x'])**2 + 
                           (log_data['ee_y'] - log_data['cart_y'])**2)
    ax8.plot(t, ee_virtual_dist * 1000, 'b-', linewidth=2, label='EE ← Virtual')
    ax8.plot(t, virtual_cart_dist * 1000, 'g-', linewidth=2, label='Virtual ← Cart')
    ax8.plot(t, ee_cart_dist * 1000, 'r--', linewidth=1.5, alpha=0.6, label='EE ← Cart')
    ax8.set_ylabel('Distance [mm]', fontweight='bold')
    ax8.set_xlabel('Time [s]', fontweight='bold')
    ax8.set_title('Separation Distances', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Plot 9: Force magnitudes
    ax9 = fig.add_subplot(gs[2, 2])
    coupling_mag = np.sqrt(log_data['coupling_force_x']**2 + log_data['coupling_force_y']**2)
    cart_force_mag = np.sqrt(log_data['cart_force_x']**2 + log_data['cart_force_y']**2)
    ax9.plot(t, coupling_mag, 'b-', linewidth=2, label='Coupling Force')
    ax9.plot(t, cart_force_mag, 'r-', linewidth=2, label='Cart Control Force')
    ax9.set_ylabel('Force Magnitude [N]', fontweight='bold')
    ax9.set_xlabel('Time [s]', fontweight='bold')
    ax9.set_title('Force Magnitudes', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # Plot 10: Virtual mass phase plot (X)
    ax10 = fig.add_subplot(gs[3, 0])
    virtual_vx = np.gradient(log_data['virtual_x'], t)
    ax10.plot(log_data['virtual_x'], virtual_vx, 'g-', linewidth=2, alpha=0.7)
    ax10.plot(log_data['virtual_x'][0], virtual_vx[0], 'go', markersize=10, label='Start')
    ax10.plot(log_data['virtual_x'][-1], virtual_vx[-1], 'ro', markersize=10, label='End')
    ax10.set_xlabel('Virtual X Position [m]', fontweight='bold')
    ax10.set_ylabel('Virtual X Velocity [m/s]', fontweight='bold')
    ax10.set_title('Phase Plot: Virtual Mass X', fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    
    # Plot 11: Cart phase plot (X)
    ax11 = fig.add_subplot(gs[3, 1])
    cart_vx = np.gradient(log_data['cart_x'], t)
    ax11.plot(log_data['cart_x'], cart_vx, 'c-', linewidth=2, alpha=0.7)
    ax11.plot(log_data['cart_x'][0], cart_vx[0], 'go', markersize=10, label='Start')
    ax11.plot(log_data['cart_x'][-1], cart_vx[-1], 'ro', markersize=10, label='End')
    ax11.set_xlabel('Cart X Position [m]', fontweight='bold')
    ax11.set_ylabel('Cart X Velocity [m/s]', fontweight='bold')
    ax11.set_title('Phase Plot: Cart X', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    # Plot 12: EE trajectory
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(log_data['ee_x'], log_data['ee_y'], 'purple', linewidth=2)
    ax12.plot(log_data['ee_x'][0], log_data['ee_y'][0], 'go', markersize=10, label='Start')
    ax12.plot(log_data['ee_x'][-1], log_data['ee_y'][-1], 'ro', markersize=10, label='End')
    ax12.set_xlabel('X Position [m]', fontweight='bold')
    ax12.set_ylabel('Y Position [m]', fontweight='bold')
    ax12.set_title('End Effector Trajectory', fontweight='bold')
    ax12.grid(True, alpha=0.3)
    ax12.axis('equal')
    ax12.legend()
    
    plt.suptitle('Virtual Mass Interaction: EE ← Virtual Mass ← Cart', 
                 fontsize=16, fontweight='bold')
    
    print(colored("✓ Plots generated", "green"))
    return fig


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual mass interaction between manipulator and cart')
    parser.add_argument('--mode', type=str, choices=['kinematics', 'simulation'], 
                        default='simulation', help='Test mode')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration [s]')
    parser.add_argument('--k-coupling', type=float, default=50.0, help='Coupling stiffness [N/m]')
    parser.add_argument('--d-coupling', type=float, default=10.0, help='Coupling damping [N·s/m]')
    parser.add_argument('--mass', type=float, default=2.0, help='Virtual mass [kg]')
    parser.add_argument('--damping', type=float, default=5.0, help='Virtual damping [N·s/m]')
    parser.add_argument('--stiffness', type=float, default=10.0, help='Virtual stiffness [N/m]')
    parser.add_argument('--kp-cart', type=float, default=100.0, help='Cart impedance Kp [N/m]')
    parser.add_argument('--kd-cart', type=float, default=20.0, help='Cart impedance Kd [N·s/m]')
    parser.add_argument('--q1_init', type=float, default=0.0, help='Initial q1 [deg]')
    parser.add_argument('--q2_init', type=float, default=0.0, help='Initial q2 [deg]')
    parser.add_argument('--q1_final', type=float, default=45.0, help='Final q1 [deg]')
    parser.add_argument('--q2_final', type=float, default=60.0, help='Final q2 [deg]')
    args = parser.parse_args()
    
    if args.mode == 'kinematics':
        # Test forward kinematics only
        result = test_manipulator_kinematics(
            q_initial=(np.deg2rad(args.q1_init), np.deg2rad(args.q2_init)),
            q_final=(np.deg2rad(args.q1_final), np.deg2rad(args.q2_final)),
            num_points=30
        )
        plt.show()
        print(colored("\n" + "="*80, "green"))
        print(colored("KINEMATICS TEST COMPLETE", "green", attrs=["bold"]))
        print(colored("="*80, "green"))
        return
    
    # Run full dynamics simulation
    log_data = simulate_passive_cart(
        duration=args.duration,
        k_coupling=args.k_coupling,
        d_coupling=args.d_coupling,
        M_virtual=args.mass,
        D_virtual=args.damping,
        K_virtual=args.stiffness,
        kp_cart=args.kp_cart,
        kd_cart=args.kd_cart,
    )
    
    # Plot results
    fig = plot_results(log_data)
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    
    # Keep Meshcat window open
    input(colored("\nPress Enter to close Meshcat and exit...", "yellow"))


if __name__ == "__main__":
    main()
