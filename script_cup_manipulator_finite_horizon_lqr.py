#!/usr/bin/env python3
"""
Cup Manipulator Finite Horizon LQR Controller

Move the 2-DOF manipulator from initial position to goal position using
Finite-Horizon Linear Quadratic Regulator with time-varying gains.

SYSTEM:
-------
- 2-DOF cup manipulator: link1_base, link2_link1
- State: [θ₁, θ₂, ω₁, ω₂] (4D)
- Input: [τ₁, τ₂] (2D torques)
- Initial position: θ₁=-10°, θ₂=+20°
- Goal position: θ₁=-40°, θ₂=+80°

CONTROL LAW (FINITE HORIZON):
-----------------------------
Minimizes: J = ∫₀ᵀ (x'Qx + u'Ru) dt + x(T)'QN·x(T)
           where T = horizon time, QN = terminal cost

Optimal control: u(t) = -K(t)(x - x_goal)
where K(t) is time-varying gain computed from discrete Riccati recursion.

ADVANTAGES OVER STANDARD LQR:
- Explicit finite time horizon
- Terminal cost emphasizes reaching goal at final time
- Time-varying gains better for transient performance
- Matches paper formulation

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

class CupManipulatorFiniteHorizonLQRConfig:
    """Finite Horizon LQR controller configuration."""
    
    def __init__(self):
        # Initial position
        self.q_init = np.array([-10.0, 20.0])  # degrees
        
        # Goal position
        self.q_goal = np.array([-40.0, 80.0])  # degrees
        
        # Finite Horizon LQR cost weights
        # Q: running state cost
        # QN: terminal state cost (higher weight at end time)
        # R: input cost
        self.Q = np.diag([10.0, 10.0, 0.1, 0.1])    # Running cost: penalize position error
        self.QN = np.diag([100.0, 100.0, 10.0, 10.0])  # Terminal cost: 10x weight at end time
        self.R = np.diag([100.0, 100.0])            # Input cost: penalize control effort
        
        # Time horizon
        self.horizon_time = 8.0  # seconds (total trajectory time)
        self.dt = 0.01           # seconds (discretization for Riccati)
        
        # Simulation
        self.simulation_time = 10.0  # seconds (run simulation longer than horizon)
        self.timestep = 0.001        # seconds (1 ms for smooth visualization)
        self.print_interval = 0.2    # seconds
        self.viz_update_interval = 0.01  # seconds
        
        # URDF path
        self.urdf_path = str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute())
        
        # Linearized matrices file
        self.matrices_file = "cup_manipulator_linearized_matrices.npz"

# ============================================================================
# FINITE HORIZON LQR CONTROLLER CLASS
# ============================================================================

class FiniteHorizonLQRController(LeafSystem):
    """
    Finite Horizon LQR controller with time-varying gains.
    
    Solves the finite-horizon continuous-time LQR problem:
        min J = ∫₀ᵀ (x'Qx + u'Ru) dt + x(T)'QN·x(T)
    
    Implementation:
    1. Discretize continuous system (A, B) with timestep dt
    2. Solve discrete Riccati equation backward from T to 0
    3. At runtime, compute time-varying gain K(t) from P(t)
    """
    
    def __init__(self, A, B, Q, R, QN, horizon_time, dt, x_goal, max_torque=10.0):
        """
        Initialize Finite Horizon LQR controller.
        
        Args:
            A: State transition matrix (4x4)
            B: Input matrix (4x2)
            Q: Running cost matrix (4x4)
            R: Input cost matrix (2x2)
            QN: Terminal cost matrix (4x4)
            horizon_time: Total horizon duration (seconds)
            dt: Discretization timestep for Riccati (seconds)
            x_goal: Goal state (4,)
            max_torque: Maximum torque magnitude (for saturation)
        """
        super().__init__()
        
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.QN = QN
        self.horizon_time = horizon_time
        self.dt = dt
        self.x_goal = x_goal
        self.max_torque = max_torque
        
        # Discretize continuous system: Ad = I + A*dt, Bd = B*dt
        self.Ad = np.eye(4) + A * dt
        self.Bd = B * dt
        
        # Solve finite-horizon discrete Riccati backward
        self._solve_finite_horizon_riccati()
        
        # Print information
        self._print_controller_info()
        
        # Declare input port (state)
        self.DeclareVectorInputPort("state", BasicVector(4))
        
        # Declare output port (control torque)
        self.DeclareVectorOutputPort("torque", BasicVector(2), self.CalcTorque)
    
    def _solve_finite_horizon_riccati(self):
        """
        Solve discrete-time finite-horizon Riccati equation backward.
        
        P(k) = Q + Ad'·P(k+1)·Ad - Ad'·P(k+1)·Bd·(R + Bd'·P(k+1)·Bd)^{-1}·Bd'·P(k+1)·Ad
        K(k) = (R + Bd'·P(k+1)·Bd)^{-1}·Bd'·P(k+1)·Ad
        
        Boundary condition: P(N) = QN at final time T
        """
        # Number of timesteps
        N = int(np.ceil(self.horizon_time / self.dt))
        
        # Initialize P at final time
        P = self.QN.copy()
        
        # Store gains and P matrices for all timesteps
        # Index 0 corresponds to t=0, index N corresponds to t=T
        self.P_history = []
        self.K_history = []
        
        # Backward recursion: from k = N down to k = 0
        for k_idx in range(N + 1):
            self.P_history.append(P.copy())
            
            # Compute gain: K(k) = (R + Bd'·P(k)·Bd)^{-1}·Bd'·P(k)·Ad
            BdT_P = self.Bd.T @ P
            S = self.R + BdT_P @ self.Bd
            S_inv = np.linalg.inv(S)
            K = S_inv @ BdT_P @ self.Ad
            self.K_history.append(K.copy())
            
            # Update P for previous timestep (going backward)
            # P(k) = Q + Ad'·P(k+1)·(Ad - Bd·K(k))
            if k_idx < N:
                P_new = self.Q + self.Ad.T @ P @ (self.Ad - self.Bd @ K)
                P = P_new
        
        # Reverse so index 0 = t=0, index N = t=T
        self.P_history.reverse()
        self.K_history.reverse()
        
        print(colored(f"\n  ✓ Finite-Horizon Riccati solved:", "green"))
        print(colored(f"    Horizon: {self.horizon_time:.2f} s, Discretization: {self.dt:.3f} s", "cyan"))
        print(colored(f"    Timesteps: {N}", "cyan"))
        print(colored(f"    K(0) initial gain:", "cyan"))
        print(colored(f"      {self.K_history[0][0]}", "cyan"))
        print(colored(f"      {self.K_history[0][1]}", "cyan"))
    
    def _print_controller_info(self):
        """Print controller information."""
        print(colored("\nFinite Horizon LQR Controller Created:", "cyan", attrs=["bold"]))
        print(colored(f"  Horizon time: {self.horizon_time:.2f} s", "cyan"))
        print(colored(f"  Terminal cost matrix QN:", "cyan"))
        print(colored(f"    diag = [{self.QN[0,0]:.1f}, {self.QN[1,1]:.1f}, {self.QN[2,2]:.1f}, {self.QN[3,3]:.1f}]", "cyan"))
        
        # Compute closed-loop eigenvalues at initial time
        A_cl_0 = self.A - self.B @ self.K_history[0]
        eigs_0 = np.linalg.eigvals(A_cl_0)
        print(colored(f"  Closed-loop eigenvalues at t=0:", "cyan"))
        for i, ev in enumerate(eigs_0):
            print(colored(f"    λ_{i}: {ev.real:10.6f} ± {abs(ev.imag):10.6f}i", "cyan"))
    
    def CalcTorque(self, context, output):
        """Compute finite horizon LQR control torque."""
        # Get state from input port
        x = self.GetInputPort("state").Eval(context)
        
        # Get current time
        t = context.get_time()
        
        # Compute which timestep we're in
        k = int(np.round(t / self.dt))
        k = np.clip(k, 0, len(self.K_history) - 1)
        
        # Get time-varying gain for this timestep (use final gain if past horizon)
        K = self.K_history[k]
        if K is None:
            # Past horizon, use last valid gain
            for i in range(k, -1, -1):
                if self.K_history[i] is not None:
                    K = self.K_history[i]
                    break
        
        # Compute error state
        x_error = x - self.x_goal
        
        # Finite-horizon LQR control law: u = -K(t) * x_error
        u = -K @ x_error
        
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
    """Main finite-horizon LQR simulation."""
    print("\n" + "=" * 80)
    print(colored("CUP MANIPULATOR FINITE HORIZON LQR CONTROL", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Load configuration
    config = CupManipulatorFiniteHorizonLQRConfig()
    
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
    
    # Create Finite Horizon LQR controller
    print(colored("\n[2/5] Creating Finite Horizon LQR controller...", "cyan", attrs=["bold"]))
    fh_lqr_controller = FiniteHorizonLQRController(
        A, B, config.Q, config.R, config.QN, 
        config.horizon_time, config.dt, x_goal, 
        max_torque=10.0
    )
    
    # Start Meshcat
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
    controller_sys = builder.AddSystem(fh_lqr_controller)
    
    # Add Meshcat visualizer
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
    print(colored(f"  Horizon time:     {config.horizon_time:.1f} s", "cyan"))
    
    # Simulation logging
    time_log = []
    q_log = []         # Joint angles
    qdot_log = []      # Joint velocities
    tau_log = []       # Applied torques
    K_norm_log = []    # Norm of feedback gain at each time
    
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
        
        # Get current gain from controller
        k = int(np.round(current_time / config.dt))
        k = np.clip(k, 0, len(fh_lqr_controller.K_history) - 1)
        K_t = fh_lqr_controller.K_history[k]
        x_error = state_input - x_goal
        tau = -K_t @ x_error
        tau = np.clip(tau, -10.0, 10.0)
        
        time_log.append(current_time)
        q_log.append(np.rad2deg(q))
        qdot_log.append(np.rad2deg(qdot))
        tau_log.append(tau)
        K_norm_log.append(np.linalg.norm(K_t))
        
        # Publish to visualizer at specified interval
        if current_time - last_viz_time >= config.viz_update_interval:
            diagram.ForcedPublish(context)
            last_viz_time = current_time
        
        # Print status
        if current_time - last_print_time >= config.print_interval:
            error_q = config.q_goal - np.rad2deg(q)
            error_norm = np.linalg.norm(error_q)
            
            # Indicator: are we still in horizon?
            in_horizon = "IN" if current_time < config.horizon_time else "POST"
            
            status = colored(f"t={current_time:.2f}s", "blue")
            status += colored(f" [{in_horizon}]", "magenta")
            status += colored(f" | θ=[{np.rad2deg(q[0]):+6.1f}°, {np.rad2deg(q[1]):+6.1f}°]", "cyan")
            status += colored(f" | ω=[{np.rad2deg(qdot[0]):+7.1f}°/s, {np.rad2deg(qdot[1]):+7.1f}°/s]", "yellow")
            status += colored(f" | ||K(t)||={np.linalg.norm(K_t):.3f}", "green")
            status += colored(f" | ||e||={error_norm:.2f}°", "white")
            
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
    K_norm_log = np.array(K_norm_log)
    
    # Create plots
    print(colored("Generating plots...", "cyan"))
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    
    # ===== Row 1: Joint Angles =====
    ax = axes[0, 0]
    ax.plot(time_log, q_log[:, 0], 'b-', linewidth=2, label='θ₁ (link1_base)')
    ax.axhline(config.q_goal[0], color='b', linestyle='--', alpha=0.5, label='Goal')
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5, label='Horizon end')
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Joint 1 Angle (Finite Horizon LQR)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(time_log, q_log[:, 1], 'r-', linewidth=2, label='θ₂ (link2_link1)')
    ax.axhline(config.q_goal[1], color='r', linestyle='--', alpha=0.5, label='Goal')
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5, label='Horizon end')
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Joint 2 Angle (Finite Horizon LQR)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Row 2: Joint Velocities =====
    ax = axes[1, 0]
    ax.plot(time_log, qdot_log[:, 0], 'b-', linewidth=2, label='ω₁')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax.set_title('Joint 1 Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 1]
    ax.plot(time_log, qdot_log[:, 1], 'r-', linewidth=2, label='ω₂')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax.set_title('Joint 2 Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Row 3: Applied Torques & Gain Norm =====
    ax = axes[2, 0]
    ax.plot(time_log, tau_log[:, 0], 'g-', linewidth=2, label='τ₁')
    ax.plot(time_log, tau_log[:, 1], 'm-', linewidth=2, label='τ₂')
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5, label='Horizon end')
    ax.set_ylabel('Torque (N⋅m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Applied Torques', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2, 1]
    ax.plot(time_log, K_norm_log, 'orange', linewidth=2, label='||K(t)||')
    ax.axvline(config.horizon_time, color='gray', linestyle=':', alpha=0.5, label='Horizon end')
    ax.set_ylabel('Gain Norm', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Feedback Gain Norm (Time-Varying)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('Cup Manipulator Finite Horizon LQR Control', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"plots/cup_manipulator_finite_horizon_lqr_{timestamp}.png"
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
    
    # Check performance at horizon end
    horizon_idx = int(config.horizon_time / config.timestep)
    if horizon_idx < len(q_log):
        q_at_horizon = q_log[horizon_idx]
        error_at_horizon = config.q_goal - q_at_horizon
        print(colored(f"\nAt horizon end (t={config.horizon_time:.1f}s):", "cyan"))
        print(colored(f"  Position: θ₁={q_at_horizon[0]:+.1f}°, θ₂={q_at_horizon[1]:+.1f}°", "cyan"))
        print(colored(f"  Error:    Δθ₁={error_at_horizon[0]:+.2f}°, Δθ₂={error_at_horizon[1]:+.2f}°", "yellow"))
    
    print(colored(f"\nInitial position: θ₁={config.q_init[0]:+.1f}°, θ₂={config.q_init[1]:+.1f}°", "cyan"))
    print(colored(f"Final position:   θ₁={final_q[0]:+.1f}°, θ₂={final_q[1]:+.1f}°", "cyan"))
    print(colored(f"Goal position:    θ₁={config.q_goal[0]:+.1f}°, θ₂={config.q_goal[1]:+.1f}°", "cyan"))
    print(colored(f"Final error:      Δθ₁={final_error[0]:+.2f}°, Δθ₂={final_error[1]:+.2f}°", "yellow" if np.linalg.norm(final_error) < 5 else "red"))
    print(colored(f"Final velocity:   ω₁={final_qdot[0]:+.2f}°/s, ω₂={final_qdot[1]:+.2f}°/s", "cyan"))
    print(colored(f"Max torque (J1):  {np.max(np.abs(tau_log[:, 0])):.3f} N⋅m", "cyan"))
    print(colored(f"Max torque (J2):  {np.max(np.abs(tau_log[:, 1])):.3f} N⋅m", "cyan"))
    print(colored(f"Gain norm range:  [{np.min(K_norm_log):.3f}, {np.max(K_norm_log):.3f}]", "cyan"))
    print(colored("=" * 80 + "\n", "cyan"))
    
    plt.show()
    return 0

if __name__ == "__main__":
    exit(main())
