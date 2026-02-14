#!/usr/bin/env python3
"""
Cup Manipulator LQR Controller

Move the 2-DOF manipulator from initial position to goal position using
Linear Quadratic Regulator (LQR) control based on linearized dynamics.

SYSTEM:
-------
- 2-DOF cup manipulator: link1_base, link2_link1
- State: [θ₁, θ₂, ω₁, ω₂] (4D)
- Input: [τ₁, τ₂] (2D torques)
- Initial position: θ₁=-10°, θ₂=+20°
- Goal position: θ₁=-40°, θ₂=+80°

CONTROL LAW:
------------
LQR minimizes: J = ∫ (x'Qx + u'Ru) dt
Optimal control: u = -K(x - x_goal)
where K is computed from continuous-time algebraic Riccati equation.

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg
from termcolor import colored
from datetime import datetime

# Drake imports
from pydrake.all import (
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    MeshcatVisualizer,
    StartMeshcat,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class CupManipulatorLQRConfig:
    """LQR controller configuration."""
    
    def __init__(self):
        # Initial position
        self.q_init = np.array([-10.0, 20.0])  # degrees
        
        # Goal position
        self.q_goal = np.array([-40.0, 80.0])  # degrees
        
        # LQR cost weights - tuned for smooth, bell-shaped velocity profiles
        # Q penalizes state error: position > velocity (gentler acceleration)
        # R penalizes control effort: higher R = smoother motion
        self.Q = np.diag([10, 10, 0.1, 0.1])    # Gentle position tracking
        self.R = np.diag([100.0, 100.0])             # High torque penalty for smooth motion
        
        # Simulation
        self.simulation_time = 10.0  # seconds
        self.timestep = 0.001        # seconds (1 ms for smooth visualization)
        self.print_interval = 0.2    # seconds (print every 200 ms to reduce output)
        self.viz_update_interval = 0.01  # seconds (publish to viz every 10 ms for smooth animation)
        
        # URDF path
        self.urdf_path = str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute())
        
        # Linearized matrices file
        self.matrices_file = "cup_manipulator_linearized_matrices.npz"

# ============================================================================
# LQR CONTROLLER CLASS
# ============================================================================

class LQRController(LeafSystem):
    """
    LQR-based state feedback controller for cup manipulator.
    
    Control law: u = -K(x - x_goal)
    
    where K is computed from the continuous-time algebraic Riccati equation:
        K = R^{-1} B' P
        A'P + PA - PBR^{-1}B'P + Q = 0
    """
    
    def __init__(self, A, B, Q, R, x_goal, max_torque=10.0):
        """
        Initialize LQR controller.
        
        Args:
            A: State transition matrix (4x4)
            B: Input matrix (4x2)
            Q: State cost matrix (4x4)
            R: Input cost matrix (2x2)
            x_goal: Goal state (4,)
            max_torque: Maximum torque magnitude (for saturation)
        """
        super().__init__()
        
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_goal = x_goal
        self.max_torque = max_torque
        
        # Solve continuous-time algebraic Riccati equation
        self.P = linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute LQR gain: K = R^{-1} B' P
        self.K = np.linalg.solve(R, B.T @ self.P)
        
        print(colored("\nLQR Controller Created:", "cyan", attrs=["bold"]))
        print(colored(f"  LQR Gain Matrix K (2x4):", "cyan"))
        print(colored(f"    {self.K[0]}", "cyan"))
        print(colored(f"    {self.K[1]}", "cyan"))
        print(colored(f"  Eigenvalues of (A - BK):", "cyan"))
        A_cl = A - B @ self.K
        eigs = np.linalg.eigvals(A_cl)
        for i, ev in enumerate(eigs):
            print(colored(f"    λ_{i}: {ev.real:10.6f} ± {abs(ev.imag):10.6f}i", "cyan"))
        
        # Declare input port (state)
        self.DeclareVectorInputPort("state", BasicVector(4))
        
        # Declare output port (control torque)
        self.DeclareVectorOutputPort("torque", BasicVector(2), self.CalcTorque)
    
    def CalcTorque(self, context, output):
        """Compute LQR control torque."""
        # Get state from input port
        x = self.GetInputPort("state").Eval(context)
        
        # Compute error state
        x_error = x - self.x_goal
        
        # LQR control law: u = -K * x_error
        u = -self.K @ x_error
        
        # Saturate torques
        u_saturated = np.clip(u, -self.max_torque, self.max_torque)
        
        # Set output
        output.SetFromVector(u_saturated)

# ============================================================================
# CUP MANIPULATOR SYSTEM CLASS
# ============================================================================

class CupManipulatorSystem:
    """Build and manage the cup manipulator plant with SceneGraph."""
    
    def __init__(self, config, builder=None):
        self.config = config
        self.builder = builder
        self.plant = None
        self.scene_graph = None
        self.model_instance = None
        self.plant_sys = None
    
    def build_plant(self):
        """Build MultibodyPlant from URDF with SceneGraph."""
        print(colored("\n[3/5] Building plant with SceneGraph...", "cyan", attrs=["bold"]))
        
        # Create plant with scene graph using builder
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        
        # Load URDF
        parser = Parser(self.plant)
        urdf_path = self.config.urdf_path
        
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        # Set up package map for meshes
        urdf_dir = Path(urdf_path).parent
        parser.package_map().Add("assets", str(urdf_dir / "assets"))
        
        # Load model
        model_instances = parser.AddModels(urdf_path)
        if not model_instances:
            raise RuntimeError("Failed to load URDF")
        
        self.model_instance = model_instances[0]
        
        # Weld base to world
        base_link = self.plant.GetBodyByName("base_mount_manipulator", self.model_instance)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            base_link.body_frame(),
            RigidTransform()
        )
        
        # Add actuators to the 2 manipulator joints
        for joint_name in ['link1_base', 'link2_link1']:
            joint = self.plant.GetJointByName(joint_name, self.model_instance)
            self.plant.AddJointActuator(joint_name, joint)
        
        # Finalize
        self.plant.Finalize()
        
        print(colored(f"  ✓ Plant created: {self.plant.num_positions()} DOF, {self.plant.num_actuators()} actuators", "green"))
        
        return self.plant

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main LQR simulation."""
    print("\n" + "=" * 80)
    print(colored("CUP MANIPULATOR LQR CONTROL", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Load configuration
    config = CupManipulatorLQRConfig()
    
    # Load linearized matrices from NPZ file
    print(colored("\n[1/5] Loading linearized matrices...", "cyan", attrs=["bold"]))
    
    if not Path(config.matrices_file).exists():
        print(colored(f"✗ Error: {config.matrices_file} not found.", "red"))
        print(colored("  Run script_cup_manipulator_linearized.py first to generate it.", "red"))
        return 1
    
    data = np.load(config.matrices_file)
    A = data['A']
    B = data['B']
    eq_state = data['eq_state']
    eq_input = data['eq_input']
    
    print(colored(f"  ✓ Loaded A(4x4), B(4x2) matrices", "green"))
    print(colored(f"  ✓ Linearization point: q=[{np.rad2deg(eq_state[0]):+.1f}°, {np.rad2deg(eq_state[1]):+.1f}°]", "cyan"))
    
    # Convert goal from degrees to radians
    q_goal_rad = np.deg2rad(config.q_goal)
    x_goal = np.concatenate([q_goal_rad, [0.0, 0.0]])  # Goal state with zero velocities
    
    print(colored(f"  Goal state: q=[{config.q_goal[0]:+.1f}°, {config.q_goal[1]:+.1f}°], ω=[0°/s, 0°/s]", "cyan"))
    
    # Create LQR controller
    print(colored("\n[2/5] Creating LQR controller...", "cyan", attrs=["bold"]))
    lqr_controller = LQRController(A, B, config.Q, config.R, x_goal, max_torque=10.0)
    
    # Start Meshcat first
    print(colored("\n[3/5] Starting Meshcat visualization...", "cyan", attrs=["bold"]))
    meshcat = StartMeshcat()
    print(colored(f"  ✓ Meshcat running at {meshcat.web_url()}", "green"))
    
    # Create diagram builder
    print(colored("\n[4/5] Building Drake diagram with plant and controller...", "cyan", attrs=["bold"]))
    builder = DiagramBuilder()
    
    # Build plant with scene graph in the builder
    system = CupManipulatorSystem(config, builder)
    plant = system.build_plant()
    
    # Add controller
    controller_sys = builder.AddSystem(lqr_controller)
    
    # Add Meshcat visualizer - connect to scene graph geometry port
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        system.scene_graph,
        meshcat
    )
    
    # Connect plant output (state) to controller input
    builder.Connect(plant.get_state_output_port(), controller_sys.GetInputPort("state"))
    
    # Connect controller output (torques) to plant input
    builder.Connect(controller_sys.GetOutputPort("torque"), plant.get_actuation_input_port())
    
    # Build diagram once
    diagram = builder.Build()
    
    # Create simulator
    print(colored("\n[5/5] Running simulation with visualization...", "cyan", attrs=["bold"]))
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)  # Run at real-time speed for visualization
    context = simulator.get_mutable_context()
    
    # Set initial conditions
    plant_context = system.plant.GetMyMutableContextFromRoot(context)
    q_init_rad = np.deg2rad(config.q_init)
    
    system.plant.SetPositions(plant_context, q_init_rad)
    system.plant.SetVelocities(plant_context, [0.0, 0.0])
    
    print(colored(f"\n  Initial position: θ₁={config.q_init[0]:+.1f}°, θ₂={config.q_init[1]:+.1f}°", "cyan"))
    print(colored(f"  Goal position:    θ₁={config.q_goal[0]:+.1f}°, θ₂={config.q_goal[1]:+.1f}°", "cyan"))
    
    # Simulation logging
    time_log = []
    q_log = []         # Joint angles
    qdot_log = []      # Joint velocities
    tau_log = []       # Applied torques
    
    # Run simulation
    last_print_time = 0.0
    last_viz_time = 0.0
    
    while context.get_time() < config.simulation_time:
        current_time = context.get_time()
        
        # Log data
        plant_context = system.plant.GetMyMutableContextFromRoot(context)
        q = system.plant.GetPositions(plant_context)
        qdot = system.plant.GetVelocities(plant_context)
        
        # Get applied torques from controller
        state_input = system.plant.get_state_output_port().Eval(plant_context)
        tau = -lqr_controller.K @ (state_input - x_goal)
        tau = np.clip(tau, -10.0, 10.0)
        
        time_log.append(current_time)
        q_log.append(np.rad2deg(q))
        qdot_log.append(np.rad2deg(qdot))
        tau_log.append(tau)
        
        # Publish to visualizer (update Meshcat) at specified interval
        if current_time - last_viz_time >= config.viz_update_interval:
            diagram.ForcedPublish(context)
            last_viz_time = current_time
        
        # Print status
        if current_time - last_print_time >= config.print_interval:
            error_q = config.q_goal - np.rad2deg(q)
            error_norm = np.linalg.norm(error_q)
            
            status = colored(f"t={current_time:.2f}s", "blue")
            status += colored(f" | θ=[{np.rad2deg(q[0]):+6.1f}°, {np.rad2deg(q[1]):+6.1f}°]", "cyan")
            status += colored(f" | ω=[{np.rad2deg(qdot[0]):+7.1f}°/s, {np.rad2deg(qdot[1]):+7.1f}°/s]", "yellow")
            status += colored(f" | τ=[{tau[0]:+6.2f}, {tau[1]:+6.2f}] N⋅m", "magenta")
            status += colored(f" | ||e||={error_norm:.2f}°", "green")
            
            print(status)
            last_print_time = current_time
        
        # Advance simulation
        simulator.AdvanceTo(current_time + config.timestep)
    
    print(colored("\n✓ Simulation complete\n", "green", attrs=["bold"]))
    
    # Convert logs to arrays
    time_log = np.array(time_log)
    q_log = np.array(q_log)
    qdot_log = np.array(qdot_log)
    tau_log = np.array(tau_log)
    
    # Create plots
    print(colored("Generating plots...", "cyan"))
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    
    # ===== Row 1: Joint Angles =====
    ax = axes[0, 0]
    ax.plot(time_log, q_log[:, 0], 'b-', linewidth=2, label='θ₁ (link1_base)')
    ax.axhline(config.q_goal[0], color='b', linestyle='--', alpha=0.5, label='Goal')
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Joint 1 Angle (link1_base)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(time_log, q_log[:, 1], 'r-', linewidth=2, label='θ₂ (link2_link1)')
    ax.axhline(config.q_goal[1], color='r', linestyle='--', alpha=0.5, label='Goal')
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Joint 2 Angle (link2_link1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Row 2: Joint Velocities =====
    ax = axes[1, 0]
    ax.plot(time_log, qdot_log[:, 0], 'b-', linewidth=2, label='ω₁')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax.set_title('Joint 1 Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 1]
    ax.plot(time_log, qdot_log[:, 1], 'r-', linewidth=2, label='ω₂')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax.set_title('Joint 2 Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Row 3: Applied Torques =====
    ax = axes[2, 0]
    ax.plot(time_log, tau_log[:, 0], 'g-', linewidth=2, label='τ₁')
    ax.set_ylabel('Torque (N⋅m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Joint 1 Torque', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2, 1]
    ax.plot(time_log, tau_log[:, 1], 'm-', linewidth=2, label='τ₂')
    ax.set_ylabel('Torque (N⋅m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Joint 2 Torque', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('Cup Manipulator LQR Control: Trajectory Tracking', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"plots/cup_manipulator_lqr_{timestamp}.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved: {plot_file}", "green"))
    
    # Print summary
    print(colored("\n" + "=" * 80, "cyan"))
    print(colored("SIMULATION SUMMARY", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan"))
    
    final_q = q_log[-1]
    final_qdot = qdot_log[-1]
    final_error = config.q_goal - final_q
    
    print(colored(f"Initial position: θ₁={config.q_init[0]:+.1f}°, θ₂={config.q_init[1]:+.1f}°", "cyan"))
    print(colored(f"Final position:   θ₁={final_q[0]:+.1f}°, θ₂={final_q[1]:+.1f}°", "cyan"))
    print(colored(f"Goal position:    θ₁={config.q_goal[0]:+.1f}°, θ₂={config.q_goal[1]:+.1f}°", "cyan"))
    print(colored(f"Final error:      Δθ₁={final_error[0]:+.2f}°, Δθ₂={final_error[1]:+.2f}°", "yellow" if np.linalg.norm(final_error) < 5 else "red"))
    print(colored(f"Final velocity:   ω₁={final_qdot[0]:+.2f}°/s, ω₂={final_qdot[1]:+.2f}°/s", "cyan"))
    print(colored(f"Max torque (J1):  {np.max(np.abs(tau_log[:, 0])):.3f} N⋅m", "cyan"))
    print(colored(f"Max torque (J2):  {np.max(np.abs(tau_log[:, 1])):.3f} N⋅m", "cyan"))
    print(colored("=" * 80 + "\n", "cyan"))
    
    plt.show()
    return 0

if __name__ == "__main__":
    exit(main())
