#!/usr/bin/env python3
"""
3D Cart-Pendulum System in PyDrake

A cart with a spherical (3D) pendulum attached via 2 revolute joints.
System DOFs:
  1. Cart position along track (prismatic)
  2. Pendulum pitch (revolute - first gimbal)
  3. Pendulum roll (revolute - second gimbal)

This creates a 3-DOF underactuated system with only the cart actuated.
"""

import numpy as np
from pathlib import Path
import argparse
from typing import Optional

from pydrake.all import (
    # Core
    DiagramBuilder,
    Simulator,
    
    # Multibody
    MultibodyPlant,
    Parser,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
    UnitInertia,
    CoulombFriction,
    
    # Shapes
    Box,
    Sphere,
    Cylinder,
    
    # Visualization
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    
    # Math
    RollPitchYaw,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical parameters
CART_MASS = 10.0  # kg
CART_SIZE = [0.3, 0.2, 0.1]  # [length, width, height] in meters

PENDULUM_MASS = 1.0  # kg
PENDULUM_LENGTH = 0.5  # meters (from pivot to COM)
PENDULUM_RADIUS = 0.05  # meters (ball radius)

# Track
TRACK_LENGTH = 2.0  # meters

# Simulation
TIME_STEP = 0.001  # seconds
SIMULATION_TIME = 10.0  # seconds
REALTIME_RATE = 1.0

# Initial conditions
INITIAL_CART_POSITION = 0.0  # meters
INITIAL_PITCH = 0.2  # radians (~11°)
INITIAL_ROLL = 0.1  # radians (~6°)

# Control (very weak PD + slow sinusoidal motion)
CART_KP = 0.5  # Very weak - allows pendulum to drive cart motion
CART_KD = 0.2  # Very weak damping
CART_MOTION_AMPLITUDE = 0.5  # meters - how far cart moves (increased)
CART_MOTION_FREQUENCY = 0.3  # Hz - how fast cart oscillates (slower)
CART_TARGET = 0.0  # meters (center of track)


# ============================================================================
# SYSTEM BUILDER
# ============================================================================

def build_cart_3d_pendulum(plant: MultibodyPlant):
    """
    Build cart with 3D pendulum using Drake's programmatic API.
    
    System structure:
    world → cart (prismatic joint along X)
      └→ gimbal1 (revolute joint - pitch/Y-axis)
         └→ gimbal2 (revolute joint - roll/X-axis)
            └→ pendulum_ball
    """
    
    # Get world body
    world_body = plant.world_body()
    world_frame = plant.world_frame()
    
    # ========================================================================
    # 1. CART
    # ========================================================================
    
    # Cart spatial inertia (simple box approximation)
    # For a box: Ixx = (1/12)*m*(h²+d²), Iyy = (1/12)*m*(w²+d²), Izz = (1/12)*m*(w²+h²)
    Ixx = (1/12) * CART_MASS * (CART_SIZE[1]**2 + CART_SIZE[2]**2)
    Iyy = (1/12) * CART_MASS * (CART_SIZE[0]**2 + CART_SIZE[2]**2)
    Izz = (1/12) * CART_MASS * (CART_SIZE[0]**2 + CART_SIZE[1]**2)
    
    cart_inertia = SpatialInertia(
        mass=CART_MASS,
        p_PScm_E=np.zeros(3),  # COM at geometric center
        G_SP_E=UnitInertia(Ixx, Iyy, Izz)
    )
    
    # Create cart body
    cart_body = plant.AddRigidBody("cart", cart_inertia)
    
    # Cart visual
    plant.RegisterVisualGeometry(
        cart_body,
        RigidTransform(),
        Box(CART_SIZE[0], CART_SIZE[1], CART_SIZE[2]),
        "cart_visual",
        [0.3, 0.3, 0.8, 1.0]  # Blue
    )
    
    # Cart collision
    plant.RegisterCollisionGeometry(
        cart_body,
        RigidTransform(),
        Box(CART_SIZE[0], CART_SIZE[1], CART_SIZE[2]),
        "cart_collision",
        CoulombFriction(0.5, 0.3)
    )
    
    # Cart prismatic joint (slides along X-axis)
    from pydrake.multibody.tree import PrismaticJoint
    cart_joint_obj = PrismaticJoint(
        name="cart_slider",
        frame_on_parent=plant.world_frame(),
        frame_on_child=cart_body.body_frame(),
        axis=[1, 0, 0],  # Slides along X
        damping=0.1
    )
    cart_joint = plant.AddJoint(cart_joint_obj)
    cart_joint.set_position_limits([-TRACK_LENGTH/2], [TRACK_LENGTH/2])
    
    # Add actuator to cart
    cart_actuator = plant.AddJointActuator("cart_force", cart_joint)
    
    # ========================================================================
    # 2. GIMBAL 1 (First revolute - Pitch/Y-axis)
    # ========================================================================
    
    # Small mass for gimbal link (negligible)
    gimbal1_inertia = SpatialInertia(
        mass=0.01,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
    )
    
    gimbal1_body = plant.AddRigidBody("gimbal1", gimbal1_inertia)
    
    # Gimbal1 visual (small cylinder for Y-axis)
    plant.RegisterVisualGeometry(
        gimbal1_body,
        RigidTransform(RotationMatrix(RollPitchYaw([0, np.pi/2, 0])), [0, 0, 0]),
        Cylinder(radius=0.01, length=0.05),
        "gimbal1_visual",
        [0.5, 0.5, 0.5, 1.0]  # Gray
    )
    
    # Gimbal1 revolute joint (rotates around Y-axis for pitch)
    from pydrake.multibody.tree import RevoluteJoint
    gimbal1_joint_obj = RevoluteJoint(
        name="pendulum_pitch",
        frame_on_parent=cart_body.body_frame(),
        frame_on_child=gimbal1_body.body_frame(),
        axis=[0, 1, 0],  # Y-axis rotation (pitch)
        damping=0.01
    )
    gimbal1_joint = plant.AddJoint(gimbal1_joint_obj)
    gimbal1_joint.set_position_limits([-np.pi], [np.pi])
    
    # ========================================================================
    # 3. GIMBAL 2 (Second revolute - Roll/X-axis)
    # ========================================================================
    
    gimbal2_inertia = SpatialInertia(
        mass=0.01,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
    )
    
    gimbal2_body = plant.AddRigidBody("gimbal2", gimbal2_inertia)
    
    # Gimbal2 visual (small cylinder for X-axis)
    plant.RegisterVisualGeometry(
        gimbal2_body,
        RigidTransform(RotationMatrix(RollPitchYaw([np.pi/2, 0, 0])), [0, 0, 0]),
        Cylinder(radius=0.01, length=0.05),
        "gimbal2_visual",
        [0.5, 0.5, 0.5, 1.0]  # Gray
    )
    
    # Gimbal2 revolute joint (rotates around X-axis for roll)
    gimbal2_joint_obj = RevoluteJoint(
        name="pendulum_roll",
        frame_on_parent=gimbal1_body.body_frame(),
        frame_on_child=gimbal2_body.body_frame(),
        axis=[1, 0, 0],  # X-axis rotation (roll)
        damping=0.01
    )
    gimbal2_joint = plant.AddJoint(gimbal2_joint_obj)
    gimbal2_joint.set_position_limits([-np.pi], [np.pi])
    
    # ========================================================================
    # 4. PENDULUM BALL
    # ========================================================================
    
    # Ball at end of massless rod
    # COM is at distance PENDULUM_LENGTH below gimbal2
    # For a point mass at distance r: I = m*r² (simplified pendulum)
    # Using parallel axis theorem: I_pivot = I_com + m*d²
    # For a sphere about its center: I = (2/5)*m*R²
    # About pivot point d away: I = (2/5)*m*R² + m*d²
    
    ball_I_com = (2/5) * PENDULUM_MASS * PENDULUM_RADIUS**2
    ball_I_pivot = ball_I_com + PENDULUM_MASS * PENDULUM_LENGTH**2
    
    ball_inertia = SpatialInertia(
        mass=PENDULUM_MASS,
        p_PScm_E=[0, 0, -PENDULUM_LENGTH],  # COM below pivot
        G_SP_E=UnitInertia(ball_I_pivot / PENDULUM_MASS, 
                           ball_I_pivot / PENDULUM_MASS, 
                           ball_I_com / PENDULUM_MASS)  # Iz about vertical axis
    )
    
    pendulum_body = plant.AddRigidBody("pendulum", ball_inertia)
    
    # Rod visual (connecting to ball)
    plant.RegisterVisualGeometry(
        pendulum_body,
        RigidTransform([0, 0, -PENDULUM_LENGTH/2]),
        Cylinder(radius=0.01, length=PENDULUM_LENGTH),
        "rod_visual",
        [0.6, 0.4, 0.2, 1.0]  # Brown
    )
    
    # Ball visual
    plant.RegisterVisualGeometry(
        pendulum_body,
        RigidTransform([0, 0, -PENDULUM_LENGTH]),
        Sphere(PENDULUM_RADIUS),
        "ball_visual",
        [0.8, 0.2, 0.2, 1.0]  # Red
    )
    
    # Ball collision
    plant.RegisterCollisionGeometry(
        pendulum_body,
        RigidTransform([0, 0, -PENDULUM_LENGTH]),
        Sphere(PENDULUM_RADIUS),
        "ball_collision",
        CoulombFriction(0.3, 0.2)
    )
    
    # Weld pendulum to gimbal2
    plant.WeldFrames(
        gimbal2_body.body_frame(),
        pendulum_body.body_frame(),
        RigidTransform()
    )
    
    # ========================================================================
    # 5. GROUND PLANE
    # ========================================================================
    
    ground_inertia = SpatialInertia(
        mass=1.0,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
    )
    
    ground_body = plant.AddRigidBody("ground", ground_inertia)
    
    # Large flat box for ground
    plant.RegisterVisualGeometry(
        ground_body,
        RigidTransform([0, 0, -0.05]),
        Box(TRACK_LENGTH*1.5, 1.0, 0.1),
        "ground_visual",
        [0.4, 0.4, 0.4, 1.0]  # Gray
    )
    
    plant.RegisterCollisionGeometry(
        ground_body,
        RigidTransform([0, 0, -0.05]),
        Box(TRACK_LENGTH*1.5, 1.0, 0.1),
        "ground_collision",
        CoulombFriction(0.8, 0.5)
    )
    
    # Weld ground to world
    plant.WeldFrames(world_frame, ground_body.body_frame())
    
    return {
        'cart_joint': cart_joint,
        'pitch_joint': gimbal1_joint,
        'roll_joint': gimbal2_joint,
        'cart_actuator': cart_actuator,
    }


# ============================================================================
# CONTROLLER
# ============================================================================

class CartPDController:
    """PD controller with slow sinusoidal motion to make cart responsive to pendulum."""
    
    def __init__(self, plant, cart_joint, kp=5.0, kd=2.0, amplitude=0.3, frequency=0.5):
        self.plant = plant
        self.cart_joint = cart_joint
        self.kp = kp
        self.kd = kd
        self.amplitude = amplitude
        self.frequency = frequency
    
    def calc_control(self, context):
        """Calculate control force for cart with slow sinusoidal target."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        # Get all positions and velocities (cart is first DOF)
        q = self.plant.GetPositions(plant_context)
        v = self.plant.GetVelocities(plant_context)
        
        cart_pos = q[0]  # First DOF is cart position
        cart_vel = v[0]  # First velocity is cart velocity
        
        # Time-varying target: slow sinusoidal motion
        t = context.get_time()
        target = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
        # PD control with weak gains - allows cart to respond to pendulum
        error = target - cart_pos
        force = self.kp * error - self.kd * cart_vel
        
        return force


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation():
    """Run the 3D cart-pendulum simulation."""
    
    print("="*70)
    print("3D CART-PENDULUM SIMULATION")
    print("="*70)
    print(f"\nSystem Configuration:")
    print(f"  Cart mass: {CART_MASS} kg")
    print(f"  Pendulum mass: {PENDULUM_MASS} kg")
    print(f"  Pendulum length: {PENDULUM_LENGTH} m")
    print(f"  Track length: {TRACK_LENGTH} m")
    print(f"\nInitial Conditions:")
    print(f"  Cart position: {INITIAL_CART_POSITION} m")
    print(f"  Pitch angle: {np.rad2deg(INITIAL_PITCH):.1f}°")
    print(f"  Roll angle: {np.rad2deg(INITIAL_ROLL):.1f}°")
    print(f"\nControl:")
    print(f"  PD gains: Kp={CART_KP}, Kd={CART_KD}")
    print(f"  Motion: {CART_MOTION_AMPLITUDE}m amplitude @ {CART_MOTION_FREQUENCY}Hz")
    print("="*70 + "\n")
    
    # Build system
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    
    # Build cart-pendulum
    components = build_cart_3d_pendulum(plant)
    
    # Finalize plant
    plant.Finalize()
    
    print(f"System finalized:")
    print(f"  DOFs: {plant.num_positions()}")
    print(f"  Velocities: {plant.num_velocities()}")
    print(f"  Actuators: {plant.num_actuators()}\n")
    
    # Setup visualization
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat
    )
    
    print(f"Visualization: {meshcat.web_url()}\n")
    
    # Build diagram
    diagram = builder.Build()
    
    # Create simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(REALTIME_RATE)
    context = simulator.get_mutable_context()
    
    # Set initial conditions
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    # Set joint positions [cart, pitch, roll]
    positions = [INITIAL_CART_POSITION, INITIAL_PITCH, INITIAL_ROLL]
    plant.SetPositions(plant_context, positions)
    
    print(f"Initial state set:")
    print(f"  Positions: {positions}")
    print(f"  Starting simulation...\n")
    
    # Create controller with slow sinusoidal motion
    controller = CartPDController(
        plant, 
        components['cart_joint'],
        kp=CART_KP,
        kd=CART_KD,
        amplitude=CART_MOTION_AMPLITUDE,
        frequency=CART_MOTION_FREQUENCY
    )
    
    # Simulation loop
    print("Running simulation...")
    last_print_time = 0.0
    print_interval = 0.5  # seconds
    
    while context.get_time() < SIMULATION_TIME:
        current_time = context.get_time()
        
        # Calculate control
        plant_context = plant.GetMyMutableContextFromRoot(context)
        control_force = controller.calc_control(context)
        
        # Apply control
        plant.get_actuation_input_port().FixValue(
            plant_context, [control_force]
        )
        
        # Step simulation
        simulator.AdvanceTo(current_time + TIME_STEP)
        
        # Print status
        if current_time - last_print_time >= print_interval:
            cart_pos = components['cart_joint'].get_translation(plant_context)
            pitch = components['pitch_joint'].get_angle(plant_context)
            roll = components['roll_joint'].get_angle(plant_context)
            
            print(f"t={current_time:.1f}s | Cart: {cart_pos:+.3f}m | "
                  f"Pitch: {np.rad2deg(pitch):+6.1f}° | "
                  f"Roll: {np.rad2deg(roll):+6.1f}°")
            
            last_print_time = current_time
    
    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    global SIMULATION_TIME
    
    parser = argparse.ArgumentParser(
        description="3D Cart-Pendulum Simulation with PyDrake"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SIMULATION_TIME,
        help="Simulation duration in seconds"
    )
    args = parser.parse_args()
    
    SIMULATION_TIME = args.duration
    
    run_simulation()


if __name__ == "__main__":
    main()
