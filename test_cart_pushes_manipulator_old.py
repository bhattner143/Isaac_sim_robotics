#!/usr/bin/env python3
"""
Test Script: Manipulator Pushes Cart via Virtual Mass (Human Arm Analogy)

This script demonstrates:
1. Manipulator follows impedance control (moves EE 1m in X direction)
2. Virtual mass coupled between manipulator EE and cart-pendulum
3. Cart-pendulum is passive (no active control, only joint damping)
4. Manipulator receives reactive forces from virtual mass
5. Equivalent to human arm pushing a cart from initial to final position

Based on notes: notes_ss_cart_pendulam_manipulator.tex
- Muscle force F_muscle = 0 (removed)
- Impedance controller on manipulator (human arm analog)
- Cart-pendulum passive dynamics

System Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: MANIPULATOR IMPEDANCE CONTROL (Human Arm)
    F_imp = K(x_ee - x_ref) + D(ẋ_ee - ẋ_ref)
    M_ref ẍ_ref = -F_imp
    
    x_ref tracks desired trajectory (1m in X direction)
    Impedance force applied as joint torques via Jacobian transpose

Step 2: VIRTUAL MASS (Admittance Dynamics)
    M_v ẍ_v + D_v ẋ_v + K_v (x_v - x₀) = F_ee_coupling + F_cart_coupling
    
    Compliant buffer between manipulator and cart

Step 3: COUPLING FORCES
    F_ee_coupling = -K_ee (x_virtual - x_ee) - D_ee (ẋ_virtual - ẋ_ee)
    F_cart_coupling = -K_c (x_virtual - x_cart) - D_c (ẋ_virtual - ẋ_cart)

Step 4: FORCE APPLICATION
    - F_ee_coupling applied to manipulator EE (reactive force)
    - F_cart_coupling applied to cart (pushing force)

Step 5: CART-PENDULUM PASSIVE DYNAMICS
    Cart-pendulum moves under applied forces
    No active control, only joint damping

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
from robot_types import create_cup_manipulator_config, create_cart_pendulum_config
from script_cup_manipulator_controller_ofc import CupManipulator, CartPendulum3D


class CartTrajectoryController(LeafSystem):
    """
    PD controller to make cart follow a desired trajectory.
    
    Cart moves 1 meter in X direction over specified duration.
    
    Inputs:
    - Cart position (2D): [x, y]
    - Cart velocity (2D): [vx, vy]
    
    Outputs:
    - Control force (2D): [Fx, Fy]
    """
    
    def __init__(self, x_start=0.0, x_end=1.0, duration=10.0, kp=200.0, kd=40.0):
        LeafSystem.__init__(self)
        
        self.x_start = x_start
        self.x_end = x_end
        self.duration = duration
        self.kp = kp
        self.kd = kd
        
        # Input ports
        self.cart_pos_input = self.DeclareVectorInputPort("cart_position", BasicVector(2))
        self.cart_vel_input = self.DeclareVectorInputPort("cart_velocity", BasicVector(2))
        
        # Output port
        self.DeclareVectorOutputPort(
            "control_force",
            BasicVector(2),
            self.CalcControlForce
        )
    
    def CalcControlForce(self, context, output):
        """Compute PD control force to track trajectory."""
        t = context.get_time()
        
        # Desired trajectory: linear motion in X, constant in Y
        if t >= self.duration:
            x_des = self.x_end
            vx_des = 0.0
        else:
            # Linear interpolation
            alpha = t / self.duration
            x_des = self.x_start + alpha * (self.x_end - self.x_start)
            vx_des = (self.x_end - self.x_start) / self.duration
        
        y_des = 0.0  # Keep Y constant
        vy_des = 0.0
        
        # Get cart state
        cart_pos = self.cart_pos_input.Eval(context)
        cart_vel = self.cart_vel_input.Eval(context)
        
        # PD control
        pos_error = np.array([x_des - cart_pos[0], y_des - cart_pos[1]])
        vel_error = np.array([vx_des - cart_vel[0], vy_des - cart_vel[1]])
        
        force = self.kp * pos_error + self.kd * vel_error
        
        # Saturate force
        max_force = 500.0
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force * (max_force / force_mag)
        
        output.SetFromVector(force)


class VirtualMassSystem(LeafSystem):
    """
    Virtual mass between cart and end effector.
    
    Dynamics:
        M_v ẍ_v + D_v ẋ_v + K_v (x_v - x₀) = F_cart_coupling + F_ee_coupling
    
    Where:
        F_cart_coupling: Force from cart (spring-damper)
        F_ee_coupling: Force from end effector (spring-damper)
    
    State: [x_v, y_v, vx_v, vy_v] (4D)
    """
    
    def __init__(self, M_virtual=2.0, D_virtual=5.0, K_virtual=10.0, x0=None):
        LeafSystem.__init__(self)
        
        self.M_v = M_virtual
        self.D_v = D_virtual
        self.K_v = K_virtual
        self.x0 = x0 if x0 is not None else np.zeros(2)
        
        # Input ports
        self.cart_pos_input = self.DeclareVectorInputPort("cart_position", BasicVector(2))
        self.cart_vel_input = self.DeclareVectorInputPort("cart_velocity", BasicVector(2))
        self.ee_pos_input = self.DeclareVectorInputPort("ee_position", BasicVector(2))
        self.ee_vel_input = self.DeclareVectorInputPort("ee_velocity", BasicVector(2))
        
        # Coupling parameters
        self.k_cart = 50.0   # Cart-virtual coupling stiffness
        self.d_cart = 10.0   # Cart-virtual coupling damping
        self.k_ee = 50.0     # EE-virtual coupling stiffness
        self.d_ee = 10.0     # EE-virtual coupling damping
        
        # Continuous state
        self.DeclareContinuousState(4)
        
        # Output ports
        self.DeclareVectorOutputPort(
            "virtual_position",
            BasicVector(2),
            self.OutputPosition
        )
        self.DeclareVectorOutputPort(
            "virtual_velocity",
            BasicVector(2),
            self.OutputVelocity
        )
        self.DeclareVectorOutputPort(
            "ee_force",  # Force to apply to EE
            BasicVector(2),
            self.OutputEEForce
        )
    
    def SetDefaultState(self, context, state):
        """Initialize at equilibrium."""
        state.SetFromVector(np.array([self.x0[0], self.x0[1], 0.0, 0.0]))
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """Virtual mass dynamics."""
        # Get state
        state = context.get_continuous_state_vector().CopyToVector()
        x_v = state[0:2]
        v_v = state[2:4]
        
        # Get cart and EE states
        x_cart = self.cart_pos_input.Eval(context)
        v_cart = self.cart_vel_input.Eval(context)
        x_ee = self.ee_pos_input.Eval(context)
        v_ee = self.ee_vel_input.Eval(context)
        
        # Coupling forces
        F_cart = -self.k_cart * (x_v - x_cart) - self.d_cart * (v_v - v_cart)
        F_ee = -self.k_ee * (x_v - x_ee) - self.d_ee * (v_v - v_ee)
        
        # Virtual mass dynamics
        a_v = (F_cart + F_ee - self.D_v * v_v - self.K_v * (x_v - self.x0)) / self.M_v
        
        derivatives.SetFromVector(np.concatenate([v_v, a_v]))
    
    def OutputPosition(self, context, output):
        """Output virtual mass position."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[0:2])
    
    def OutputVelocity(self, context, output):
        """Output virtual mass velocity."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[2:4])
    
    def OutputEEForce(self, context, output):
        """Compute force to apply to end effector."""
        # Get state
        state = context.get_continuous_state_vector().CopyToVector()
        x_v = state[0:2]
        v_v = state[2:4]
        
        # Get EE state
        x_ee = self.ee_pos_input.Eval(context)
        v_ee = self.ee_vel_input.Eval(context)
        
        # Force from virtual mass to EE (Newton's 3rd law)
        F_ee = self.k_ee * (x_v - x_ee) + self.d_ee * (v_v - v_ee)
        
        output.SetFromVector(F_ee)


class EndEffectorKinematics(LeafSystem):
    """
    Compute end effector position and velocity.
    
    Inputs:
    - Manipulator state (4D): [q1, q2, q1_dot, q2_dot]
    
    Outputs:
    - End effector position (2D): [x_ee, y_ee]
    - End effector velocity (2D): [vx_ee, vy_ee]
    """
    
    def __init__(self, plant: MultibodyPlant, manipulator: CupManipulator):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        self.plant_context = plant.CreateDefaultContext()
        
        # Input port
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
        """Compute end effector position."""
        state = self.state_input.Eval(context)
        q = state[:2]
        
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        ee_pos_world = self.manipulator.CalcPosition(self.plant, self.plant_context)
        
        output.SetFromVector([ee_pos_world[0], ee_pos_world[1]])
    
    def CalcVelocity(self, context, output):
        """Compute end effector velocity."""
        state = self.state_input.Eval(context)
        q = state[:2]
        v = state[2:]
        
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        self.plant.SetVelocities(self.plant_context, self.model_instance, v)
        
        ee_origin = np.zeros(3)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            ee_origin,
            self.world_frame,
            self.world_frame
        )
        
        J_translational = J_spatial[3:6, :]
        J_manip = J_translational[:, :2]
        v_ee_world = J_manip @ v
        
        output.SetFromVector([v_ee_world[0], v_ee_world[1]])


class ManipulatorJointDamping(LeafSystem):
    """
    Simple joint-space damping for manipulator.
    
    τ = -K_q q̇
    
    This allows the manipulator to move compliantly in response to EE forces.
    """
    
    def __init__(self, kd=5.0):
        LeafSystem.__init__(self)
        
        self.kd = kd
        
        # Input port
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output port
        self.DeclareVectorOutputPort(
            "torque_output",
            BasicVector(2),
            self.CalcTorque
        )
    
    def CalcTorque(self, context, output):
        """Compute damping torque."""
        state = self.state_input.Eval(context)
        q_dot = state[2:]
        
        torque = -self.kd * q_dot
        
        output.SetFromVector(torque)


def simulate_cart_pushes_manipulator(
    duration=10.0,
    cart_distance=1.0,
    M_virtual=2.0,
    D_virtual=5.0,
    K_virtual=10.0,
    kd_joint=5.0,
):
    """
    Simulate cart pushing manipulator via virtual mass.
    
    Args:
        duration: Simulation duration [s]
        cart_distance: Distance cart moves in X [m]
        M_virtual: Virtual mass [kg]
        D_virtual: Virtual damping [N·s/m]
        K_virtual: Virtual stiffness [N/m]
        kd_joint: Joint damping [N·m·s/rad]
        
    Returns:
        log_data: Dictionary with logged data
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("CART PUSHES MANIPULATOR VIA VIRTUAL MASS", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat server started at: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Create configurations
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),
        damping=(0.5, 0.5),
        friction=(0.05, 0.05),
    )
    
    # Build system
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Add manipulator
    manipulator = CupManipulator(manipulator_config, enable_visualization=False)
    parser = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser)
    manipulator.weld_base_to_world(plant)
    
    # Add actuators
    joint1 = plant.GetJointByName("link1_base", manipulator.model_instance)
    joint2 = plant.GetJointByName("link2_link1", manipulator.model_instance)
    plant.AddJointActuator("joint1_actuator", joint1)
    plant.AddJointActuator("joint2_actuator", joint2)
    
    # Add cart-pendulum
    cart_pendulum_config = create_cart_pendulum_config(
        cart_mass=5.0,
        cart_size=0.1,
        cart_damping=2.0,
        pendulum_mass=0.5,
        pendulum_length=0.2,
        pendulum_radius=0.05,
        pendulum_damping=0.05,
        attachment_offset=(0.0, 0.0, 0.0),
        initial_cart_x=0.0,
        initial_cart_y=0.0,
        initial_pitch=0.0,
        initial_roll=0.0,
        name="cart_pendulum"
    )
    
    cart_pendulum = CartPendulum3D(
        cart_pendulum_config,
        visualize_cart=True,
        add_cart_actuators=False
    )
    model_instance_cart = plant.AddModelInstance("cart_pendulum_model")
    cart_pendulum.attach_to_plant(plant, model_instance_cart, register_visuals=False)
    
    plant.Finalize()
    
    # Get initial EE position
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, manipulator.model_instance, 
                      np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    ee_pos_init = manipulator.CalcPosition(plant, temp_context)[:2]
    
    print(colored(f"\n✓ System created", "green"))
    print(colored(f"  Manipulator DOF: 2", "cyan"))
    print(colored(f"  Cart DOF: 4 (2 cart + 2 pendulum)", "cyan"))
    print(colored(f"  Initial EE position: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m", "cyan"))
    print(colored(f"\nCart Motion:", "yellow", attrs=["bold"]))
    print(colored(f"  Start X: {ee_pos_init[0]:.3f} m", "cyan"))
    print(colored(f"  End X: {ee_pos_init[0] + cart_distance:.3f} m", "cyan"))
    print(colored(f"  Distance: {cart_distance:.3f} m", "cyan"))
    print(colored(f"  Duration: {duration:.1f} s", "cyan"))
    print(colored(f"\nVirtual Mass:", "yellow", attrs=["bold"]))
    print(colored(f"  M_v: {M_virtual:.2f} kg", "cyan"))
    print(colored(f"  D_v: {D_virtual:.2f} N·s/m", "cyan"))
    print(colored(f"  K_v: {K_virtual:.2f} N/m", "cyan"))
    
    # Create systems
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    
    cart_controller = builder.AddSystem(CartTrajectoryController(
        x_start=ee_pos_init[0],
        x_end=ee_pos_init[0] + cart_distance,
        duration=duration,
        kp=200.0,
        kd=40.0
    ))
    
    virtual_mass = builder.AddSystem(VirtualMassSystem(
        M_virtual=M_virtual,
        D_virtual=D_virtual,
        K_virtual=K_virtual,
        x0=ee_pos_init
    ))
    
    joint_damping = builder.AddSystem(ManipulatorJointDamping(kd=kd_joint))
    
    # Demultiplexers
    from pydrake.systems.primitives import Demultiplexer
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))
    cart_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))
    
    # Connect plant state
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    
    # Connect manipulator state
    builder.Connect(state_demux.get_output_port(0), ee_kinematics.GetInputPort("manipulator_state"))
    builder.Connect(state_demux.get_output_port(0), joint_damping.GetInputPort("manipulator_state"))
    
    # Connect cart state
    builder.Connect(state_demux.get_output_port(1), cart_demux.get_input_port())
    
    # Connect cart controller
    builder.Connect(cart_demux.get_output_port(0), cart_controller.GetInputPort("cart_position"))
    builder.Connect(cart_demux.get_output_port(2), cart_controller.GetInputPort("cart_velocity"))
    
    # Connect virtual mass inputs
    builder.Connect(cart_demux.get_output_port(0), virtual_mass.GetInputPort("cart_position"))
    builder.Connect(cart_demux.get_output_port(2), virtual_mass.GetInputPort("cart_velocity"))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), virtual_mass.GetInputPort("ee_position"))
    builder.Connect(ee_kinematics.GetOutputPort("ee_velocity"), virtual_mass.GetInputPort("ee_velocity"))
    
    # Apply forces
    cart_body = cart_pendulum.cart_body
    manip_ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
    
    class ForceApplicator(LeafSystem):
        """Applies forces to both cart and manipulator EE."""
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("cart_force", 2)
            self.DeclareVectorInputPort("ee_force", 2)
            self.DeclareAbstractOutputPort(
                "spatial_forces",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self.CalcSpatialForces
            )
        
        def CalcSpatialForces(self, context, output):
            cart_force_2d = self.get_input_port(0).Eval(context)
            ee_force_2d = self.get_input_port(1).Eval(context)
            
            forces = []
            
            # Force on cart
            cart_force = ExternallyAppliedSpatialForce()
            cart_force.body_index = cart_body.index()
            cart_force.F_Bq_W = SpatialForce(
                tau=np.zeros(3),
                f=np.array([cart_force_2d[0], cart_force_2d[1], 0.0])
            )
            cart_force.p_BoBq_B = np.zeros(3)
            forces.append(cart_force)
            
            # Force on EE
            ee_force = ExternallyAppliedSpatialForce()
            ee_force.body_index = manip_ee_body.index()
            ee_force.F_Bq_W = SpatialForce(
                tau=np.zeros(3),
                f=np.array([ee_force_2d[0], ee_force_2d[1], 0.0])
            )
            ee_force.p_BoBq_B = manipulator.EE_OFFSET
            forces.append(ee_force)
            
            output.set_value(forces)
    
    force_applicator = builder.AddSystem(ForceApplicator())
    builder.Connect(cart_controller.get_output_port(), force_applicator.get_input_port(0))
    builder.Connect(virtual_mass.GetOutputPort("ee_force"), force_applicator.get_input_port(1))
    builder.Connect(force_applicator.get_output_port(0), plant.get_applied_spatial_force_input_port())
    
    # Connect joint damping to actuation
    builder.Connect(joint_damping.get_output_port(), plant.get_actuation_input_port())
    
    # Visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    virtual_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(virtual_mass.GetOutputPort("virtual_position"), virtual_pos_logger.get_input_port())
    
    ee_force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(virtual_mass.GetOutputPort("ee_force"), ee_force_logger.get_input_port())
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, np.array([
        np.deg2rad(-10.0), np.deg2rad(20.0),  # Manipulator
        ee_pos_init[0], ee_pos_init[1],        # Cart at EE
        0.0, 0.0                               # Pendulum
    ]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    visualizer.StartRecording()
    
    print(colored(f"\nSimulating for {duration} s...", "yellow"))
    simulator.AdvanceTo(duration)
    print(colored("✓ Simulation complete", "green"))
    
    visualizer.PublishRecording()
    print(colored(f"\n🎬 Animation published to Meshcat: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    virtual_pos_log = virtual_pos_logger.FindLog(context)
    ee_force_log = ee_force_logger.FindLog(context)
    
    time_data = state_log.sample_times()
    state_data = state_log.data()
    ee_pos_data = ee_pos_log.data()
    virtual_pos_data = virtual_pos_log.data()
    ee_force_data = ee_force_log.data()
    
    # Parse state
    q1 = state_data[0, :]
    q2 = state_data[1, :]
    cart_x = state_data[2, :]
    cart_y = state_data[3, :]
    
    return {
        'time': time_data,
        'q1': q1,
        'q2': q2,
        'cart_x': cart_x,
        'cart_y': cart_y,
        'ee_x': ee_pos_data[0, :],
        'ee_y': ee_pos_data[1, :],
        'virtual_x': virtual_pos_data[0, :],
        'virtual_y': virtual_pos_data[1, :],
        'ee_force_x': ee_force_data[0, :],
        'ee_force_y': ee_force_data[1, :],
    }


def plot_results(log_data):
    """Plot simulation results."""
    print(colored("\n📈 Generating plots...", "yellow"))
    
    t = log_data['time']
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Joint angles vs time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, np.rad2deg(log_data['q1']), 'b-', linewidth=2, label='q₁')
    ax1.plot(t, np.rad2deg(log_data['q2']), 'r-', linewidth=2, label='q₂')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
    ax1.set_title('Manipulator Joint Configuration', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: X positions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, log_data['cart_x'], 'c-', linewidth=2.5, label='Cart X')
    ax2.plot(t, log_data['virtual_x'], 'g--', linewidth=2, label='Virtual X')
    ax2.plot(t, log_data['ee_x'], 'b:', linewidth=2, label='EE X')
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('X Position [m]', fontweight='bold')
    ax2.set_title('X Position: Cart → Virtual → EE', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Y positions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, log_data['cart_y'], 'c-', linewidth=2.5, label='Cart Y')
    ax3.plot(t, log_data['virtual_y'], 'g--', linewidth=2, label='Virtual Y')
    ax3.plot(t, log_data['ee_y'], 'b:', linewidth=2, label='EE Y')
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Y Position [m]', fontweight='bold')
    ax3.set_title('Y Position: Cart → Virtual → EE', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Force on EE
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, log_data['ee_force_x'], 'b-', linewidth=2, label='Fₓ')
    ax4.plot(t, log_data['ee_force_y'], 'r-', linewidth=2, label='Fᵧ')
    ax4.set_xlabel('Time [s]', fontweight='bold')
    ax4.set_ylabel('Force [N]', fontweight='bold')
    ax4.set_title('Reactive Force on End Effector', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Paths in X-Y plane
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(log_data['cart_x'], log_data['cart_y'], 'c-', linewidth=3, alpha=0.8, label='Cart')
    ax5.plot(log_data['virtual_x'], log_data['virtual_y'], 'g--', linewidth=2.5, alpha=0.7, label='Virtual')
    ax5.plot(log_data['ee_x'], log_data['ee_y'], 'b:', linewidth=2, alpha=0.6, label='EE')
    ax5.plot(log_data['cart_x'][0], log_data['cart_y'][0], 'go', markersize=12, label='Start')
    ax5.plot(log_data['cart_x'][-1], log_data['cart_y'][-1], 'ro', markersize=12, label='End')
    ax5.set_xlabel('X Position [m]', fontweight='bold')
    ax5.set_ylabel('Y Position [m]', fontweight='bold')
    ax5.set_title('Trajectories: Cart → Virtual → EE', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.legend()
    
    # Plot 6: Separation distances
    ax6 = fig.add_subplot(gs[1, 2])
    cart_virtual_dist = np.sqrt((log_data['cart_x'] - log_data['virtual_x'])**2 + 
                                (log_data['cart_y'] - log_data['virtual_y'])**2)
    virtual_ee_dist = np.sqrt((log_data['virtual_x'] - log_data['ee_x'])**2 + 
                              (log_data['virtual_y'] - log_data['ee_y'])**2)
    ax6.plot(t, cart_virtual_dist * 1000, 'g-', linewidth=2, label='Cart ← Virtual')
    ax6.plot(t, virtual_ee_dist * 1000, 'b-', linewidth=2, label='Virtual ← EE')
    ax6.set_xlabel('Time [s]', fontweight='bold')
    ax6.set_ylabel('Distance [mm]', fontweight='bold')
    ax6.set_title('Separation Distances', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Joint configuration evolution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(np.rad2deg(log_data['q1']), np.rad2deg(log_data['q2']), 'purple', linewidth=2)
    ax7.plot(np.rad2deg(log_data['q1'][0]), np.rad2deg(log_data['q2'][0]), 'go', markersize=12, label='Start')
    ax7.plot(np.rad2deg(log_data['q1'][-1]), np.rad2deg(log_data['q2'][-1]), 'ro', markersize=12, label='End')
    ax7.set_xlabel('q₁ [deg]', fontweight='bold')
    ax7.set_ylabel('q₂ [deg]', fontweight='bold')
    ax7.set_title('Joint Configuration Evolution', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Force magnitude
    ax8 = fig.add_subplot(gs[2, 1])
    force_mag = np.sqrt(log_data['ee_force_x']**2 + log_data['ee_force_y']**2)
    ax8.plot(t, force_mag, 'purple', linewidth=2)
    ax8.set_xlabel('Time [s]', fontweight='bold')
    ax8.set_ylabel('Force Magnitude [N]', fontweight='bold')
    ax8.set_title('Total EE Force Magnitude', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Final configuration summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = f"""
FINAL JOINT CONFIGURATION

Initial:
  q₁ = {np.rad2deg(log_data['q1'][0]):.2f}°
  q₂ = {np.rad2deg(log_data['q2'][0]):.2f}°

Final (after {t[-1]:.1f}s):
  q₁ = {np.rad2deg(log_data['q1'][-1]):.2f}°
  q₂ = {np.rad2deg(log_data['q2'][-1]):.2f}°

Change:
  Δq₁ = {np.rad2deg(log_data['q1'][-1] - log_data['q1'][0]):.2f}°
  Δq₂ = {np.rad2deg(log_data['q2'][-1] - log_data['q2'][0]):.2f}°

Cart Displacement:
  ΔX = {log_data['cart_x'][-1] - log_data['cart_x'][0]:.3f} m
  
EE Displacement:
  ΔX = {log_data['ee_x'][-1] - log_data['ee_x'][0]:.3f} m
    """
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cart Pushes Manipulator via Virtual Mass', 
                 fontsize=16, fontweight='bold')
    
    print(colored("✓ Plots generated", "green"))
    return fig


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cart pushes manipulator via virtual mass'
    )
    parser.add_argument('--duration', type=float, default=10.0, help='Duration [s]')
    parser.add_argument('--distance', type=float, default=1.0, help='Cart travel distance [m]')
    parser.add_argument('--mass', type=float, default=2.0, help='Virtual mass [kg]')
    parser.add_argument('--damping', type=float, default=5.0, help='Virtual damping [N·s/m]')
    parser.add_argument('--stiffness', type=float, default=10.0, help='Virtual stiffness [N/m]')
    parser.add_argument('--kd-joint', type=float, default=5.0, help='Joint damping [N·m·s/rad]')
    args = parser.parse_args()
    
    # Run simulation
    log_data = simulate_cart_pushes_manipulator(
        duration=args.duration,
        cart_distance=args.distance,
        M_virtual=args.mass,
        D_virtual=args.damping,
        K_virtual=args.stiffness,
        kd_joint=args.kd_joint,
    )
    
    # Plot results
    fig = plot_results(log_data)
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("SIMULATION COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    
    # Keep Meshcat open
    input(colored("\nPress Enter to close Meshcat and exit...", "yellow"))


if __name__ == "__main__":
    main()
