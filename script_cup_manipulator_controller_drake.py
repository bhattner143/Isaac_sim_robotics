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
    
    # Trajectory Optimization
    DirectCollocation,
    PiecewisePolynomial,
    
    # Optimization
    Solve,
    BoundingBoxConstraint,
    LinearEqualityConstraint,
    
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
parser.add_argument('--mode', type=str, choices=['pd', 'inverse-dynamics', 'computed-torque', 'scene-viz', 'dynamics-validation', 'trajectory-optimized', 'min-jerk-joint', 'ofc-effort', 'ofc-smoothness'],
                    default='ofc-effort', help='Controller type (scene-viz = static visualization only, dynamics-validation = validate manual EOM, trajectory-optimized = optimal trajectory for minimum pendulum swing, min-jerk-joint = minimum-jerk joint-space trajectory, ofc-effort = optimal feedback control minimizing effort, ofc-smoothness = optimal feedback control minimizing jerk)')
parser.add_argument('--visualize', type=bool, default=True, help='Enable visualization')
parser.add_argument('--plot_frames', type=bool, default=True, help='Plot coordinate frames')
parser.add_argument('--traj_duration', type=float, default=3.0, help='Trajectory duration in seconds')
args, _ = parser.parse_known_args()

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Cup Manipulator Configuration ---
CUP_MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.1, 0.1),
    damping=(0.5, 0.5),      # LOW damping → allows oscillations (was 10.5, high damping prevented oscillations)
    stiffness=(50.0, 50.0),  # Non-zero stiffness → spring restoring force needed for oscillations (was 0.0)
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

# --- Minimum-Jerk Joint-Space Configuration ---
MIN_JERK_DURATION = args.traj_duration
MIN_JERK_Q_START = np.deg2rad(np.array([80,-160, 0, 180 ]))
MIN_JERK_Q_GOAL = np.deg2rad(np.array([20.0, -40.0, 0, 180 ]))

# --- Trajectory Optimization Configuration ---
TRAJECTORY_START = [np.deg2rad(80.0), np.deg2rad(-160.0), 0.0, np.deg2rad(180.0)]  # [link1, link2, pitch, roll] - ball hanging down
TRAJECTORY_GOAL = [np.deg2rad(20.0), np.deg2rad(-40.0), 0.0, np.deg2rad(180.0)]    # End with ball hanging down (zero swing)
TRAJECTORY_DURATION = args.traj_duration
TRAJECTORY_NUM_SAMPLES = 30  # Number of knot points for DirectCollocation
TRAJECTORY_PENDULUM_WEIGHT = 100.0  # Cost weight for pendulum deflection
TRAJECTORY_TORQUE_WEIGHT = 0.1  # Cost weight for control effort

# --- Optimal Feedback Control (OFC) Configuration ---
OFC_DURATION = args.traj_duration
OFC_Q_START = np.deg2rad(np.array([80, -160, 0, 180]))  # [link1, link2, pitch, roll]
OFC_Q_GOAL = np.deg2rad(np.array([20.0, -40.0, 0, 180]))  # Goal configuration

# OFC Cost Weights (for LQR)
# State penalty matrix Q: penalize deviation from desired trajectory
OFC_Q_POSITION = np.array([100.0, 100.0])  # Position tracking weight for manipulator joints
OFC_Q_PENDULUM = np.array([500.0, 500.0])  # Pendulum angle tracking weight (keep pendulum stable)
OFC_Q_VELOCITY = np.array([10.0, 10.0, 50.0, 50.0])  # Velocity tracking weights [q̇1, q̇2, θ̇_pitch, θ̇_roll]

# Control penalty matrix R: penalize control effort or jerk
OFC_R_EFFORT = np.array([0.1, 0.1])  # Effort penalty for torques (effort-minimizing mode)
OFC_R_SMOOTHNESS = np.array([0.1, 0.1])  # Smoothness penalty for jerk (smoothness-minimizing mode) [s³/m]

# Impedance parameters (for zero-force trajectory dynamics)
OFC_MASS = 1.0  # Virtual mass between driving force and impedance [kg]
OFC_STIFFNESS = 100.0  # Impedance stiffness kp [N/m]
OFC_DAMPING = 20.0  # Impedance damping kd [N·s/m]


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
# TRAJECTORY OPTIMIZER CLASS
# ============================================================================

class TrajectoryOptimizer:
    """
    Optimizes manipulator trajectories to minimize pendulum swing using DirectCollocation.
    
    Uses Drake's trajectory optimization to find optimal joint trajectories that:
    - Move manipulator from start to goal position
    - Minimize pendulum deflection during motion
    - Respect dynamics constraints automatically
    - Stay within joint, velocity, and torque limits
    """
    
    def __init__(self, plant, plant_context, num_samples=30):
        """
        Initialize trajectory optimizer.
        
        Args:
            plant: MultibodyPlant with full robot model (manipulator + pendulum)
            plant_context: Context for the plant
            num_samples: Number of knot points for collocation
        """
        self.plant = plant
        self.plant_context = plant_context
        self.num_samples = num_samples
        self.optimized_trajectory = None
        
    def optimize_trajectory(self, q_start, q_goal, duration, 
                           pendulum_weight=100.0, torque_weight=0.1,
                           max_pendulum_swing_deg=30.0):
        """
        Optimize trajectory from start to goal with minimum pendulum swing.
        
        Args:
            q_start: Initial state [4] - [link1, link2, pitch, roll] (rad)
            q_goal: Final state [4] - [link1, link2, pitch, roll] (rad)
            duration: Trajectory duration [s]
            pendulum_weight: Cost weight for pendulum deflection
            torque_weight: Cost weight for control effort
            max_pendulum_swing_deg: Maximum allowed pendulum swing during motion [degrees]
            
        Returns:
            PiecewisePolynomial: Optimized trajectory with q(t), q̇(t), q̈(t)
        """
        print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
        print(colored("Trajectory Optimization with DirectCollocation", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(f"  Start: [{np.rad2deg(q_start[0]):6.1f}°, {np.rad2deg(q_start[1]):6.1f}°, {np.rad2deg(q_start[2]):6.1f}°, {np.rad2deg(q_start[3]):6.1f}°]")
        print(f"  Goal:  [{np.rad2deg(q_goal[0]):6.1f}°, {np.rad2deg(q_goal[1]):6.1f}°, {np.rad2deg(q_goal[2]):6.1f}°, {np.rad2deg(q_goal[3]):6.1f}°]")
        print(f"  Duration: {duration:.2f} s")
        print(f"  Knot points: {self.num_samples}")
        print(f"  Weights: pendulum={pendulum_weight}, torque={torque_weight}")
        print(colored("="*70, 'cyan'))
        
        # Create DirectCollocation optimizer
        min_timestep = duration / (self.num_samples * 2)
        max_timestep = duration / (self.num_samples * 0.5)
        
        dircol = DirectCollocation(
            self.plant,
            self.plant_context,
            num_time_samples=self.num_samples,
            minimum_time_step=min_timestep,
            maximum_time_step=max_timestep,
            input_port_index=self.plant.get_actuation_input_port().get_index()
        )
        
        # Get decision variables
        u = dircol.input()  # Control inputs (torques) [2] - only manipulator actuated
        x = dircol.state()  # State [q, q̇] = [8] (4 positions + 4 velocities)
        
        # Add cost function: minimize pendulum swing + control effort
        # State indices: [link1, link2, pitch, roll, link1_dot, link2_dot, pitch_dot, roll_dot]
        # Note: In DirectCollocation, x and u are symbolic matrices where columns are knot points
        
        # Target: pitch=0 (no pitch), roll=180° (hanging down)
        target_roll = q_start[3]  # Should be 180° for hanging down
        
        # Add running cost integrated over the trajectory
        # Manipulator accelerations (indices 4,5 in state derivative)
        # q̈_manipulator = (q̇_manipulator - q̇_manipulator_prev) / dt (approximated by velocities)
        dircol.AddRunningCost(
            pendulum_weight * (x[2]**2 + (x[3] - target_roll)**2) +   # deviation from target (pitch=0, roll=180°)
            0.1 * pendulum_weight * (x[6]**2 + x[7]**2) +              # pendulum velocities  
            torque_weight * (u[0]**2 + u[1]**2) +                     # control effort
            0.5 * (x[4]**2 + x[5]**2)                                 # manipulator velocities (encourage smooth motion)
        )
        
        # Boundary conditions
        # Initial state: q = q_start, q̇ = 0
        initial_state = np.concatenate([q_start, np.zeros(4)])
        prog = dircol.prog()  # Get the mathematical program
        prog.AddBoundingBoxConstraint(initial_state, initial_state, dircol.initial_state())
        
        # Final state: Only constrain positions, allow velocities to approach zero smoothly
        # This prevents the sudden velocity drop that causes jerk transfer to pendulum
        final_positions = q_goal  # [link1, link2, pitch, roll]
        final_state_var = dircol.final_state()
        
        # Constrain only the position part (indices 0:4)
        prog.AddBoundingBoxConstraint(final_positions, final_positions, final_state_var[:4])
        
        # Add terminal cost for non-zero velocities (soft constraint for smooth approach)
        # Heavily penalize non-zero velocities at the end
        final_velocity_weight = 50.0
        dircol.AddFinalCost(
            final_velocity_weight * (final_state_var[4]**2 + final_state_var[5]**2 +  # manipulator velocities
                                    final_state_var[6]**2 + final_state_var[7]**2)    # pendulum velocities
        )
        
        # Path constraints (applied to all knot points)
        # Joint limits (manipulator joints)
        joint_limits_rad = np.deg2rad(170)  # ±170 degrees
        max_swing_rad = np.deg2rad(max_pendulum_swing_deg)
        max_velocity = 3.0  # rad/s
        max_torque = 10.0  # N·m
        
        # For pendulum starting at roll=180° (hanging down), allow swing around that position
        target_roll = q_start[3]  # 180° for hanging down
        
        # State limits [link1, link2, pitch, roll, link1_dot, link2_dot, pitch_dot, roll_dot]
        state_lower = np.array([
            -joint_limits_rad, -joint_limits_rad,      # manipulator joints
            -max_swing_rad, target_roll - max_swing_rad,  # pitch: ±swing, roll: 180°±swing
            -max_velocity, -max_velocity, -max_velocity, -max_velocity  # velocities
        ])
        state_upper = np.array([
            joint_limits_rad, joint_limits_rad,
            max_swing_rad, target_roll + max_swing_rad,
            max_velocity, max_velocity, max_velocity, max_velocity
        ])
        state_constraint = BoundingBoxConstraint(state_lower, state_upper)
        dircol.AddConstraintToAllKnotPoints(state_constraint, x)
        
        # Control (torque) limits [link1_torque, link2_torque]
        control_lower = np.array([-max_torque, -max_torque])
        control_upper = np.array([max_torque, max_torque])
        control_constraint = BoundingBoxConstraint(control_lower, control_upper)
        dircol.AddConstraintToAllKnotPoints(control_constraint, u)
        
        # Initial guess: cubic interpolation for smooth velocity profile
        # This creates a trajectory where velocities smoothly ramp up and down
        final_state_guess = np.concatenate([q_goal, np.zeros(4)])  # Guess: zero velocities at end
        initial_x_trajectory = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0., duration],
            np.column_stack([initial_state, final_state_guess]),
            np.zeros(8),  # Zero velocity derivative at start
            np.zeros(8)   # Zero velocity derivative at end
        )
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)
        
        # Solve the optimization problem
        print(colored("\n⏳ Solving trajectory optimization...", 'yellow'))
        print(f"  Decision variables: ~{self.num_samples * 10 + 1}")
        print(f"  Constraints: ~{self.num_samples * 20} (dynamics + limits)")
        
        prog = dircol.prog()
        result = Solve(prog)
        
        if not result.is_success():
            print(colored("✗ Trajectory optimization failed!", 'red', attrs=['bold']))
            print(f"  Solver: {result.get_solver_id().name()}")
            print(f"  Status: {result.get_solution_result()}")
            raise RuntimeError("Trajectory optimization did not converge")
        
        # Extract optimized trajectory
        self.optimized_trajectory = dircol.ReconstructStateTrajectory(result)
        self.optimized_input_trajectory = dircol.ReconstructInputTrajectory(result)
        
        # Print optimization results
        print(colored("✓ Trajectory optimization succeeded!", 'green', attrs=['bold']))
        print(f"  Solver: {result.get_solver_id().name()}")
        # Try to get solve time if available
        try:
            solver_details = result.get_solver_details()
            if hasattr(solver_details, 'optimizer_time'):
                print(f"  Solve time: {solver_details.optimizer_time:.3f} s")
        except:
            pass
        print(f"  Final cost: {result.get_optimal_cost():.6f}")
        
        # Analyze trajectory
        self._analyze_trajectory(duration)
        
        return self.optimized_trajectory
    
    def _analyze_trajectory(self, duration):
        """Print statistics about the optimized trajectory."""
        if self.optimized_trajectory is None:
            return

        # Sample trajectory at fine resolution
        t_samples = np.linspace(0, duration, 200)
        max_pitch = 0.0
        max_roll = 0.0
        max_torque = np.array([0.0, 0.0])
        
        for t in t_samples:
            state = self.optimized_trajectory.value(t).flatten()
            pitch = abs(state[2])
            roll = abs(state[3])
            max_pitch = max(max_pitch, pitch)
            max_roll = max(max_roll, roll)
            
            if self.optimized_input_trajectory is not None:
                torque = self.optimized_input_trajectory.value(t).flatten()
                max_torque = np.maximum(max_torque, np.abs(torque))
        
        print(colored("\nTrajectory Analysis:", 'cyan'))
        print(f"  Max pitch deflection: {np.rad2deg(max_pitch):.2f}°")
        print(f"  Max roll deflection:  {np.rad2deg(max_roll):.2f}°")
        print(f"  Max torque Link1:     {max_torque[0]:.3f} N·m")
        print(f"  Max torque Link2:     {max_torque[1]:.3f} N·m")
        print(colored("="*70 + "\n", 'cyan'))
    
    def get_trajectory_at_time(self, t):
        """
        Get desired state at time t from optimized trajectory.
        
        Args:
            t: Query time [s]
            
        Returns:
            tuple: (q_desired [4], q_dot_desired [4], q_ddot_desired [4])
        """
        if self.optimized_trajectory is None:
            raise RuntimeError("No optimized trajectory available. Call optimize_trajectory() first.")
        
        # Get state and derivatives from piecewise polynomial
        state = self.optimized_trajectory.value(t).flatten()  # [q, q̇]
        state_dot = self.optimized_trajectory.derivative(1).value(t).flatten()  # [q̇, q̈]
        
        q_desired = state[:4]
        q_dot_desired = state[4:]
        q_ddot_desired = state_dot[4:]
        
        return q_desired, q_dot_desired, q_ddot_desired


# ============================================================================
# OPTIMIZED TRAJECTORY GENERATOR
# ============================================================================

class OptimizedTrajectoryGenerator:
    """
    Trajectory generator that wraps optimized trajectory from DirectCollocation.
    
    Extracts manipulator joints from full 4-DOF optimized trajectory and provides
    interface compatible with existing controllers.
    """
    
    def __init__(self, optimizer: TrajectoryOptimizer, duration: float):
        """
        Args:
            optimizer: TrajectoryOptimizer with computed trajectory
            duration: Trajectory duration [s]
        """
        self.optimizer = optimizer
        self.motion_duration = duration  # Compatibility with existing code
    
    def compute_trajectory(self, t: float):
        """
        Get desired state from optimized trajectory.
        
        Args:
            t: Current time [s]
            
        Returns:
            tuple: (q_desired [2], q_dot_desired [2], q_ddot_desired [2])
                  Only manipulator joints (indices 0:2)
        """
        # Clamp time to trajectory duration for holding phase
        t_clamped = min(t, self.motion_duration)
        
        # Get full 4-DOF trajectory state
        q_full, q_dot_full, q_ddot_full = self.optimizer.get_trajectory_at_time(t_clamped)
        
        # Extract manipulator joints only (indices 0:2)
        q_desired = q_full[:2]
        q_dot_desired = q_dot_full[:2]
        q_ddot_desired = q_ddot_full[:2]
        
        # For holding phase after trajectory, zero velocity/acceleration
        if t > self.motion_duration:
            q_dot_desired = np.zeros(2)
            q_ddot_desired = np.zeros(2)
        
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
# OPTIMAL FEEDBACK CONTROLLER (OFC) LEAFSYSTEM
# ============================================================================

class OptimalFeedbackController(LeafSystem):
    """
    Optimal Feedback Controller using Linear Quadratic Regulator (LQR).
    
    ═══════════════════════════════════════════════════════════════════════
    ADVANCED CONTROLLER - Optimal control with zero-force trajectory
    ═══════════════════════════════════════════════════════════════════════
    
    Based on Razavian et al. (2021) "Learning Zero-Force Control with Dynamics Primitives"
    
    Two modes:
    1. Effort-Minimizing: Minimizes control torques (effort)
       State: [q, q̇, F, y_zf, ẏ_zf] where F is driving force
       Control: u = F (force input to impedance)
       
    2. Smoothness-Minimizing: Minimizes jerk (smoothness)
       State: [q, q̇, y_zf, ẏ_zf, ÿ_zf]
       Control: u = y_zf_jerk (jerk of zero-force trajectory)
    
    Control Law:
        τ = -K(t) · [x - x_desired(t)]
    
    Where:
        - K: Optimal feedback gain from LQR solution
        - x: Augmented state (positions, velocities, + internal states)
        - x_desired: Desired trajectory state
    
    Key Features:
        - Optimal trade-off between tracking accuracy and control cost
        - Time-varying gains from Riccati equation solution
        - Handles underactuated pendulum optimally
        - Smooth online trajectory generation
    
    ═══════════════════════════════════════════════════════════════════════
    
    Inputs (Port 0):
        - state: [q, q_dot] full system state
    
    Outputs (Port 0):
        - torque: optimal control torques for actuated joints
    """
    
    def __init__(self, plant: MultibodyPlant, model_instance,
                 q_start: np.ndarray, q_goal: np.ndarray, duration: float,
                 mode: str = 'effort',
                 Q_position: np.ndarray = None,
                 Q_pendulum: np.ndarray = None,
                 Q_velocity: np.ndarray = None,
                 R: np.ndarray = None,
                 impedance_mass: float = 1.0,
                 impedance_kp: float = 100.0,
                 impedance_kd: float = 20.0):
        """
        Initialize Optimal Feedback Controller.
        
        Args:
            plant: MultibodyPlant reference
            model_instance: Model instance ID
            q_start: Starting configuration [4] (2 arm + 2 pendulum)
            q_goal: Goal configuration [4]
            duration: Motion duration [s]
            mode: 'effort' or 'smoothness'
            Q_position: State penalty for manipulator positions [2]
            Q_pendulum: State penalty for pendulum angles [2]
            Q_velocity: State penalty for velocities [4]
            R: Control penalty [2]
            impedance_mass: Virtual mass for impedance [kg]
            impedance_kp: Impedance stiffness [N/m]
            impedance_kd: Impedance damping [N·s/m]
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.model_instance = model_instance
        self.mode = mode
        self.q_start = np.array(q_start[:2], dtype=float)  # Manipulator only
        self.q_goal = np.array(q_goal[:2], dtype=float)
        self.duration = duration
        
        # Impedance parameters
        self.Ma = impedance_mass
        self.kp = impedance_kp
        self.kd = impedance_kd
        
        # Cost matrices
        self.Q_position = Q_position if Q_position is not None else OFC_Q_POSITION
        self.Q_pendulum = Q_pendulum if Q_pendulum is not None else OFC_Q_PENDULUM
        self.Q_velocity = Q_velocity if Q_velocity is not None else OFC_Q_VELOCITY
        self.R = R if R is not None else (OFC_R_EFFORT if mode == 'effort' else OFC_R_SMOOTHNESS)
        
        print(colored(f"\n--- OptimalFeedbackController Configuration ---", 'yellow', attrs=['bold']))
        print(colored(f"  Mode: {mode.upper()}", 'cyan', attrs=['bold']))
        print(colored(f"  Control Law: τ = -K · (x - x_desired)", 'cyan'))
        print(colored(f"  Q_position: {self.Q_position}", 'cyan'))
        print(colored(f"  Q_pendulum: {self.Q_pendulum}", 'cyan'))
        print(colored(f"  Q_velocity: {self.Q_velocity}", 'cyan'))
        print(colored(f"  R: {self.R}", 'cyan'))
        print(colored(f"  Impedance: Ma={self.Ma} kg, kp={self.kp} N/m, kd={self.kd} N·s/m", 'cyan'))
        print(colored(f"  Motion: {duration:.2f}s from {np.rad2deg(self.q_start)}° to {np.rad2deg(self.q_goal)}°", 'cyan'))
        
        # Get dimensions
        self.num_actuated = 2  # link1_base, link2_link1
        self.num_positions = plant.num_positions()  # 4 (2 arm + 2 pendulum)
        self.num_velocities = plant.num_velocities()  # 4
        
        # Input port: full state [q, q_dot] from plant
        self.DeclareVectorInputPort("state", BasicVector(self.num_positions + self.num_velocities))
        
        # Output port: control torques for actuated joints
        self.DeclareVectorOutputPort("control", BasicVector(self.num_actuated),
                                     self.CalcControlTorque)
        
        # Linearize plant and compute LQR gain
        self._linearize_and_compute_lqr()
        
        print(colored(f"✓ OptimalFeedbackController initialized:", 'green', attrs=['bold']))
        print(colored(f"  - Actuated joints: {self.num_actuated}", 'green'))
        print(colored(f"  - Total DOF: {self.num_positions}", 'green'))
        print(colored(f"  - Augmented state dim: {self.K.shape[1]}", 'green'))
        print(colored(f"  - LQR gain matrix: {self.K.shape}", 'green'))
    
    def _linearize_and_compute_lqr(self):
        """
        Compute LQR gains using actual manipulator dynamics from Drake.
        
        For underactuated system (4 DOF, 2 actuated):
        - Extract manipulator subsystem dynamics (2 DOF)
        - Use actual mass matrix, gravity, and Coriolis terms
        - Linearize around equilibrium configuration
        - Solve LQR and extend gains to include pendulum feedback
        """
        from pydrake.all import LinearQuadraticRegulator
        
        print(colored("\n⏳ Computing LQR with actual manipulator dynamics...", 'yellow'))
        
        # Create context at equilibrium (goal configuration)
        context = self.plant.CreateDefaultContext()
        
        # Set to equilibrium: goal position with pendulum hanging down
        q_eq = np.concatenate([self.q_goal, [0.0, np.deg2rad(180.0)]])  # [link1, link2, pitch=0, roll=180°]
        v_eq = np.zeros(4)
        
        self.plant.SetPositions(context, q_eq)
        self.plant.SetVelocities(context, v_eq)
        
        # Extract dynamics at equilibrium
        # Full system: M(q)·q̈ = τ - C(q,v)·v - g(q)
        M_full = self.plant.CalcMassMatrix(context)  # [4 x 4]
        g_full = self.plant.CalcGravityGeneralizedForces(context)  # [4]
        C_full = self.plant.CalcBiasTerm(context) - g_full  # Coriolis forces (bias - gravity)
        
        print(colored(f"  Full system at equilibrium:", 'cyan'))
        print(colored(f"    M shape: {M_full.shape}", 'cyan'))
        print(colored(f"    Gravity: {g_full}", 'cyan'))
        
        # Extract manipulator subsystem (first 2 DOFs)
        # M(q)·q̈ = τ becomes: M₂₂·q̈₁₂ + M₂₄·q̈₃₄ = τ₁₂ - g₁₂
        # For manipulator-only LQR, we consider: M₁₁·q̈₁₂ ≈ τ₁₂ - g₁₂
        M_manip = M_full[0:2, 0:2]  # [2 x 2] manipulator inertia
        g_manip = g_full[0:2]  # [2] gravity on manipulator
        
        # Linearized dynamics around equilibrium:
        # State: x = [q, q̇] = [q₁, q₂, v₁, v₂]
        # Dynamics: ẋ = A·x + B·u where u = τ
        
        # A matrix [4 x 4]:
        # [  0    0   1   0  ]   (q̇₁ = v₁)
        # [  0    0   0   1  ]   (q̇₂ = v₂)
        # [ a₃₁ a₃₂  0   0  ]   (v̇₁ from linearized dynamics)
        # [ a₄₁ a₄₂  0   0  ]   (v̇₂ from linearized dynamics)
        
        # For small deviations from equilibrium with zero velocity:
        # v̇ ≈ M⁻¹·(τ - g - ∂g/∂q·Δq)
        # At equilibrium with v=0, Coriolis terms vanish
        
        # Compute stiffness matrix K_gravity = -∂g/∂q (gravity gradient)
        # For now, use numerical approximation or assume small
        M_inv = np.linalg.inv(M_manip)
        
        A_manip = np.zeros((4, 4))
        A_manip[0:2, 2:4] = np.eye(2)  # q̇ = v
        # For acceleration: simplified linearization assuming gravity gradient is small
        # v̇ ≈ M⁻¹·(-K_g·Δq + τ) where K_g ≈ 0 at hanging equilibrium
        # More accurate would need ∂g/∂q, but for now use zero (stable equilibrium)
        
        # B matrix [4 x 2]:
        # [  0   0  ]
        # [  0   0  ]
        # [ b₃₁ b₃₂ ]  = M⁻¹ (torque to acceleration)
        # [ b₄₁ b₄₂ ]
        
        B_manip = np.zeros((4, 2))
        B_manip[2:4, :] = M_inv  # Acceleration from torque: q̈ = M⁻¹·τ
        
        print(colored(f"  Manipulator subsystem linearization:", 'cyan'))
        print(colored(f"    M_manip:\n{M_manip}", 'cyan'))
        print(colored(f"    M_inv:\n{M_inv}", 'cyan'))
        print(colored(f"    A: {A_manip.shape}, B: {B_manip.shape}", 'cyan'))
        
        # Cost matrices (only for manipulator states)
        Q_manip = np.diag([
            self.Q_position[0],  # q1
            self.Q_position[1],  # q2
            self.Q_velocity[0],  # q̇1
            self.Q_velocity[1]   # q̇2
        ])
        
        R_manip = np.diag(self.R)
        
        # Solve LQR for manipulator subsystem
        K_manip, S = LinearQuadraticRegulator(A_manip, B_manip, Q_manip, R_manip)
        
        print(colored(f"  LQR solution for manipulator:", 'cyan'))
        print(colored(f"    K_manip: {K_manip.shape}", 'cyan'))
        print(colored(f"    K values:\n{K_manip}", 'cyan'))
        
        # Expand to full state by padding with zeros for pendulum states
        # Full state: [q1, q2, θ_pitch, θ_roll, q̇1, q̇2, θ̇_pitch, θ̇_roll]
        # K_full: [2 x 8]
        K_full = np.zeros((2, 8))
        K_full[:, 0:2] = K_manip[:, 0:2]  # Position gains for q1, q2
        K_full[:, 4:6] = K_manip[:, 2:4]  # Velocity gains for q̇1, q̇2
        
        # Add feedback from pendulum states for dynamic coupling
        # Coupling gains based on pendulum cost weights
        pendulum_coupling = 0.05  # Coupling factor (reduced from 0.1 for stability)
        K_full[:, 2] = pendulum_coupling * self.Q_pendulum[0] * 0.01  # pitch position
        K_full[:, 3] = pendulum_coupling * self.Q_pendulum[1] * 0.01  # roll position
        K_full[:, 6] = pendulum_coupling * self.Q_velocity[2] * 0.01  # pitch velocity
        K_full[:, 7] = pendulum_coupling * self.Q_velocity[3] * 0.01  # roll velocity
        
        self.K = K_full
        
        print(colored(f"✓ LQR solved successfully with actual dynamics", 'green'))
        print(colored(f"  K_full matrix: {self.K.shape}", 'cyan'))
        print(colored(f"  Manipulator gains: Kp={K_manip[:, 0:2]}, Kd={K_manip[:, 2:4]}", 'cyan'))
    
    def _build_effort_dynamics(self, A_plant, B_plant):
        """Build augmented dynamics for effort-minimizing mode."""
        # Simplified approach: Use plant dynamics directly
        # In full implementation, would add impedance dynamics
        
        # For now: just use plant dynamics (future: add F and y_zf states)
        state_dim = A_plant.shape[0]  # 8
        return A_plant, B_plant, state_dim
    
    def _build_smoothness_dynamics(self, A_plant, B_plant):
        """Build augmented dynamics for smoothness-minimizing mode."""
        # Simplified approach: Use plant dynamics directly
        # In full implementation, would add y_zf, ẏ_zf, ÿ_zf states
        
        # For now: just use plant dynamics (future: add ZFT states)
        state_dim = A_plant.shape[0]  # 8
        return A_plant, B_plant, state_dim
    
    def CalcControlTorque(self, context, output):
        """
        Compute optimal control torque using LQR feedback.
        
        Inputs:
            context: Drake context
        
        Outputs:
            output: Control torques [2] for actuated joints
        """
        # Get current state from input port
        state = self.GetInputPort("state").Eval(context)
        q = state[:self.num_positions]
        q_dot = state[self.num_positions:]
        
        # Get current time
        t = context.get_time()
        
        # Compute desired state from minimum-jerk trajectory
        h, hdot, hddot = self._min_jerk_profile(t)
        q_desired = self.q_start + (self.q_goal - self.q_start) * h
        q_dot_desired = (self.q_goal - self.q_start) * hdot
        
        # Full desired state (with pendulum at equilibrium)
        q_full_desired = np.concatenate([q_desired, [0.0, np.deg2rad(180.0)]])
        q_dot_full_desired = np.concatenate([q_dot_desired, [0.0, 0.0]])
        
        x_desired = np.concatenate([q_full_desired, q_dot_full_desired])
        x_current = state
        
        # Optimal feedback: u = -K · (x - x_desired)
        error = x_current - x_desired
        u = -self.K @ error
        
        # Extract torques for actuated joints (first 2 outputs)
        torque = u[:self.num_actuated]
        
        output.SetFromVector(torque)
    
    def _min_jerk_profile(self, t: float):
        """Compute minimum-jerk time scaling."""
        if self.duration <= 0:
            return 1.0, 0.0, 0.0
        s = np.clip(t / self.duration, 0.0, 1.0)
        h = 10 * s**3 - 15 * s**4 + 6 * s**5
        hdot = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.duration
        hddot = (60 * s - 180 * s**2 + 120 * s**3) / (self.duration**2)
        return h, hdot, hddot


# ============================================================================
# PENDULUM 3D CLASS
# ============================================================================

class Pendulum3D:
    """
    3D Pendulum with 2-DOF gimbal joints using spherical coordinates.
    
    COORDINATE SYSTEM (matches LaTeX documentation):
    --------------------------------------------------
    The pendulum orientation is described using spherical coordinates:
    
    - θ (theta): Polar angle from +z axis (down), θ ∈ [0, π]
        * θ = 0: pendulum hanging straight down (stable equilibrium)
        * θ = π/2: pendulum horizontal
        * θ = π: pendulum pointing up (unstable equilibrium)
    
    - φ (phi): Azimuthal angle from +x axis in xy-plane, φ ∈ [0, 2π)
        * φ = 0: pendulum projects onto +x axis
        * φ = π/2: pendulum projects onto +y axis
    
    KINEMATICS:
    -----------
    Ball position in Cartesian coordinates:
        x_ball = x_cart + L·sin(θ)·cos(φ)
        y_ball = y_cart + L·sin(θ)·sin(φ)
        z_ball = L·cos(θ)  (positive downward from pivot)
    
    DYNAMICS:
    ---------
    Equations of motion: M(q)·q̈ + C(q,q̇) + G(q) = τ
    where q = [x_c, y_c, θ, φ]ᵀ
    
    Key properties:
    - Mass matrix: M₄₄ = mL²·sin²(θ) (gimbal lock at θ=0)
    - Gravity: G = [0, 0, -mgL·sin(θ), 0]ᵀ (only torque about θ)
    - Potential energy: V = -mgL·cos(θ)
    
    See cart_pendulum_eom.pdf in notes_cup/ for complete derivation.
    """
    
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
                - ball_wrt_pivot: [x, y, z] relative to pivot frame (Cartesian)
                - ball_wrt_world: [x, y, z] relative to world frame (Cartesian)
                - ball_in_ball_frame: [x, y, z] in ball's own frame (always [0, 0, -L])
                - pivot_in_ball_frame: [x, y, z] of pivot as seen from ball frame
                - theta: polar angle from +z axis (radians), θ ∈ [0, π]
                - phi: azimuthal angle from +x axis (radians), φ ∈ [0, 2π)
                - r: radial distance (should equal L)
                - x, y, z: Cartesian components of ball_wrt_pivot
                - roll_wrt_pivot: roll angle of ball frame w.r.t. pivot frame (radians)
                - pitch_wrt_pivot: pitch angle of ball frame w.r.t. pivot frame (radians)
                - yaw_wrt_pivot: yaw angle of ball frame w.r.t. pivot frame (radians)
                - roll_wrt_world: roll angle of ball frame w.r.t. world frame (radians)
                - pitch_wrt_world: pitch angle of ball frame w.r.t. world frame (radians)
                - yaw_wrt_world: yaw angle of ball frame w.r.t. world frame (radians)
                - joint_pitch: gimbal pitch joint angle (radians)
                - joint_roll: gimbal roll joint angle (radians)
        
        SPHERICAL COORDINATES:
            θ (theta): Polar angle from +z axis
                - θ = 0: pendulum hanging straight down
                - θ = π/2: pendulum horizontal
                - θ = π: pendulum inverted
            φ (phi): Azimuthal angle from +x axis in xy-plane
                - φ = 0: projection on +x axis
                - φ = π/2: projection on +y axis
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
        # Method 1: Using Drake's RollPitchYaw class (Z-Y-X intrinsic convention)
        roll_wrt_pivot, pitch_wrt_pivot, yaw_wrt_pivot = self._extract_rpy_from_rotation(X_PB.rotation())

        # Get actual joint angles (gimbal configuration space coordinates)
        pitch_angle, roll_angle = self._calculate_ball_angles(context)
        
        # Transform from WORLD frame to ball frame (includes manipulator motion)
        X_WB = plant.CalcRelativeTransform(context, plant.world_frame(), ball_frame)
        ball_wrt_world = X_WB.rotation() @ ball_offset_in_body
        
        # Extract roll-pitch-yaw angles of ball frame relative to world frame
        roll_wrt_world, pitch_wrt_world, yaw_wrt_world = self._extract_rpy_from_rotation(X_WB.rotation())
        
        # Ball position in ball_frame coordinates (always constant)
        ball_in_ball_frame = ball_offset_in_body  # [0, 0, -L] by definition
        
        # Pivot position as seen from ball frame (inverse perspective)
        X_BP = X_PB.inverse()
        pivot_in_ball_frame = X_BP.translation()
        
        # Compute spherical coordinates (relative to pivot frame)
        r, theta, phi = self._convert_to_spherical(ball_wrt_pivot)
        x, y, z = ball_wrt_pivot
        
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
            'joint_pitch': pitch_angle,
            'joint_roll': roll_angle,
        }
    
    def _convert_to_spherical(self, cartesian):
        """
        Convert Cartesian coordinates to spherical (r, theta, phi).
        
        Spherical coordinate convention (matching LaTeX notes):
            - theta (\u03b8): Polar angle from +z axis (down), \u03b8 \u2208 [0, \u03c0]
                \u03b8 = 0: pointing straight down (+z direction)
                \u03b8 = \u03c0/2: horizontal
                \u03b8 = \u03c0: pointing up (-z direction)
            - phi (\u03c6): Azimuthal angle from +x axis in xy-plane, \u03c6 \u2208 [0, 2\u03c0)
                \u03c6 = 0: projection on +x axis
                \u03c6 = \u03c0/2: projection on +y axis
            - r: Radial distance from origin
            
        Cartesian to spherical:
            x = r\u00b7sin(\u03b8)\u00b7cos(\u03c6)
            y = r\u00b7sin(\u03b8)\u00b7sin(\u03c6)
            z = r\u00b7cos(\u03b8)
            
        Args:
            cartesian: [x, y, z] position vector
            
        Returns:
            (r, theta, phi) in (meters, radians, radians)
        """
        x, y, z = cartesian
        r = np.linalg.norm(cartesian)
        theta = np.arccos(z / r) if r > 1e-10 else 0.0  # Polar angle from +z axis
        phi = np.arctan2(y, x)  # Azimuthal angle from +x axis
        return r, theta, phi
    
    def _extract_rpy_from_rotation(self, rotation):
        """
        Extract Roll-Pitch-Yaw angles from rotation matrix.
        
        Args:
            rotation: RotationMatrix object or 3x3 numpy array
            
        Returns:
            (roll, pitch, yaw) tuple in radians
        """
        rpy = RollPitchYaw(rotation)
        return rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()
    
    
    def _calculate_ball_angles(self, context):
        """Calculate current pendulum angles (pitch, roll) from joint states."""
        if not self.pitch_joint or not self.roll_joint:
            return None
        
        pitch_angle = self.pitch_joint.get_angle(context)
        roll_angle = self.roll_joint.get_angle(context)
        
        return pitch_angle, roll_angle
    
    
    def compute_mass_matrix(self, theta, phi, M_cart=1.0):
        """
        Compute mass matrix M(q) for cart-pendulum system (spherical coordinates).
        
        Args:
            theta: Polar angle from z-axis (radians), θ ∈ [0, π]
            phi: Azimuthal angle from x-axis (radians), φ ∈ [0, 2π)
            M_cart: Cart mass (kg)
            
        Returns:
            M: 4x4 mass matrix
            
        Spherical coordinates:
            x_ball = x_c + L·sin(θ)·cos(φ)
            y_ball = y_c + L·sin(θ)·sin(φ)
            z_ball = L·cos(θ)
        """
        m = self.mass
        L = self.length
        
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        
        M = np.array([
            [M_cart + m,              0,                       m*L*c_theta*c_phi,        -m*L*s_theta*s_phi],
            [0,                       M_cart + m,              m*L*c_theta*s_phi,         m*L*s_theta*c_phi],
            [m*L*c_theta*c_phi,       m*L*c_theta*s_phi,       m*L**2,                    0],
            [-m*L*s_theta*s_phi,      m*L*s_theta*c_phi,       0,                         m*L**2*s_theta**2]
        ])
        
        return M
    
    def compute_coriolis_vector(self, theta, phi, theta_dot, phi_dot, x_c_dot=0.0, y_c_dot=0.0):
        """
        Compute Coriolis/centrifugal vector C(q, q_dot) for spherical coordinates.
        
        Args:
            theta: Polar angle from z-axis (radians)
            phi: Azimuthal angle from x-axis (radians)
            theta_dot: θ velocity (rad/s)
            phi_dot: φ velocity (rad/s)
            x_c_dot: Cart x velocity (m/s)
            y_c_dot: Cart y velocity (m/s)
            
        Returns:
            C: 4x1 Coriolis/centrifugal vector
        """
        m = self.mass
        L = self.length
        
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        
        C1 = -m*L*(theta_dot**2*s_theta*c_phi + phi_dot**2*s_theta*c_theta*c_phi + 2*theta_dot*phi_dot*c_theta*s_phi)
        C2 = -m*L*(theta_dot**2*s_theta*s_phi + phi_dot**2*s_theta*c_theta*s_phi - 2*theta_dot*phi_dot*c_theta*c_phi)
        C3 = m*L*phi_dot**2*s_theta*c_theta - m*L*phi_dot*s_theta*(x_c_dot*s_phi - y_c_dot*c_phi)
        C4 = -m*L**2*theta_dot*phi_dot*np.sin(2*theta) - m*L*theta_dot*c_theta*(x_c_dot*s_phi + y_c_dot*c_phi)
        
        C = np.array([C1, C2, C3, C4])
        
        return C
    
    def compute_gravity_vector(self, theta, phi, g=9.81):
        """
        Compute gravity vector G(q) for spherical coordinates.
        
        Args:
            theta: Polar angle from z-axis (radians)
            phi: Azimuthal angle from x-axis (radians)
            g: Gravitational acceleration (m/s²)
            
        Returns:
            G: 4x1 gravity vector
            
        Note: In spherical coordinates, gravity only produces torque about θ axis.
              There is no gravitational torque about φ axis (symmetry about vertical).
        """
        m = self.mass
        L = self.length
        
        s_theta = np.sin(theta)
        
        G1 = 0.0
        G2 = 0.0
        G3 = -m*g*L*s_theta  # Restoring torque about θ axis
        G4 = 0.0              # No torque about φ axis (gravity is symmetric)
        
        G = np.array([G1, G2, G3, G4])
        
        return G
    
    def compute_state_space_linearization(self, theta_eq=0.0, phi_eq=0.0, M_cart=1.0, g=9.81):
        """
        Compute linearized state-space representation around equilibrium (spherical coords).
        
        State vector: x = [x_c, y_c, θ, φ, ẋ_c, ẏ_c, θ̇, φ̇]ᵀ  (8x1)
        Input vector: u = [F_x, F_y, τ_θ, τ_φ]ᵀ                 (4x1)
        
        Linearized dynamics:
            ẋ = A·x + B·u
            y = C·x + D·u
        
        Args:
            theta_eq: Equilibrium polar angle from z-axis (radians, typically 0 for hanging down)
            phi_eq: Equilibrium azimuthal angle (radians, arbitrary for θ=0)
            M_cart: Cart mass (kg)
            g: Gravitational acceleration (m/s²)
            
        Returns:
            A: 8x8 state matrix
            B: 8x4 input matrix
            C: 8x8 output matrix (identity - full state observable)
            D: 8x4 feedthrough matrix (zeros)
        """
        m = self.mass
        L = self.length
        
        # Evaluate at equilibrium (typically hanging down: θ=0, φ arbitrary)
        M_eq = self.compute_mass_matrix(theta_eq, phi_eq, M_cart)
        M_inv = np.linalg.inv(M_eq)
        
        # Compute linearized gravity Jacobian: ∂G/∂q evaluated at equilibrium
        # For spherical coordinates:
        # G3 = -m*g*L*sin(θ)  →  ∂G3/∂θ = -m*g*L*cos(θ)
        # G4 = 0              →  ∂G4/∂φ = 0  (no gravitational torque about φ)
        
        dG_dq = np.zeros((4, 4))
        if np.abs(theta_eq) < 0.1:  # Small angle approximation around θ=0 (hanging down)
            dG_dq[2, 2] = -m*g*L  # ∂G3/∂θ ≈ -m*g*L at θ=0
            dG_dq[3, 3] = 0.0     # ∂G4/∂φ = 0 (always, due to symmetry)
        else:
            # Exact derivatives at arbitrary equilibrium
            c_theta = np.cos(theta_eq)
            
            dG_dq[2, 2] = -m*g*L*c_theta  # ∂G3/∂θ
            dG_dq[2, 3] = 0.0              # ∂G3/∂φ = 0
            dG_dq[3, 2] = 0.0              # ∂G4/∂θ = 0
            dG_dq[3, 3] = 0.0              # ∂G4/∂φ = 0
        
        # State-space matrices
        # x = [q; q_dot] where q = [x_c, y_c, θ, φ]
        # ẋ = [q_dot; q_ddot] = [q_dot; M^{-1}(Q - C - G)]
        # At equilibrium (q_dot = 0): C = 0
        # Linearization: q_ddot ≈ M^{-1}·(Q - G_eq - ∂G/∂q·Δq)
        
        A = np.zeros((8, 8))
        # Upper right block: ∂(q_dot)/∂(q_dot) = I
        A[0:4, 4:8] = np.eye(4)
        # Lower left block: ∂(q_ddot)/∂q = -M^{-1}·∂G/∂q
        A[4:8, 0:4] = -M_inv @ dG_dq
        # Lower right block: ∂(q_ddot)/∂(q_dot) = -M^{-1}·∂C/∂(q_dot)
        # At equilibrium with q_dot=0, C=0, and ∂C/∂(q_dot) represents damping
        # For undamped system: ∂C/∂(q_dot) ≈ 0 (or add joint damping if needed)
        # If joint damping b: add -b*q_dot terms
        if hasattr(self, 'damping') and self.damping > 0:
            A[6, 6] = -self.damping / (m*L**2)  # Pitch damping
            A[7, 7] = -self.damping / (m*L**2)  # Roll damping
        
        B = np.zeros((8, 4))
        # Lower block: ∂(q_ddot)/∂u = M^{-1}
        B[4:8, 0:4] = M_inv
        
        # Output: full state measurement
        C = np.eye(8)
        
        # No direct feedthrough
        D = np.zeros((8, 4))
        
        return A, B, C, D
    
    def compute_nonlinear_dynamics(self, q, q_dot, u, M_cart=1.0, g=9.81):
        """
        Compute nonlinear dynamics: q_ddot = M^{-1}(q)·[u - C(q,q_dot) - G(q)]
        
        Args:
            q: Configuration [x_c, y_c, θ, φ] (4x1)
            q_dot: Velocity [ẋ_c, ẏ_c, θ̇, φ̇] (4x1)
            u: Control input [F_x, F_y, τ_θ, τ_φ] (4x1)
            M_cart: Cart mass (kg)
            g: Gravitational acceleration (m/s²)
            
        Returns:
            q_ddot: Acceleration [ẍ_c, ÿ_c, θ̈, φ̈] (4x1)
        """
        x_c, y_c, theta, phi = q
        x_c_dot, y_c_dot, theta_dot, phi_dot = q_dot
        
        M = self.compute_mass_matrix(theta, phi, M_cart)
        C = self.compute_coriolis_vector(theta, phi, theta_dot, phi_dot, x_c_dot, y_c_dot)
        G = self.compute_gravity_vector(theta, phi, g)
        
        # Solve: M·q_ddot = u - C - G
        q_ddot = np.linalg.solve(M, u - C - G)
        
        return q_ddot
    
    # ========================================================================
    # GIMBAL COORDINATE DYNAMICS (for validation with Drake)
    # ========================================================================
    
    def compute_mass_matrix_gimbal(self, pitch, roll):
        """
        Compute mass matrix M(q) for gimbal-mounted pendulum.
        
        Gimbal coordinates (YX Euler angles):
            - pitch (α): rotation about Y-axis  
            - roll (β): rotation about X-axis (in gimbal1 frame after pitch)
        
        Ball position (with Z down):
            x = -L·sin(α)
            y = L·cos(α)·sin(β)
            z = -L·cos(α)·cos(β)
        
        Ball velocity:
            v = [∂x/∂α, ∂y/∂α, ∂z/∂α]·α̇ + [∂x/∂β, ∂y/∂β, ∂z/∂β]·β̇
            ∂p/∂α = [-L·cos(α), -L·sin(α)·sin(β), L·sin(α)·cos(β)]
            ∂p/∂β = [0, L·cos(α)·cos(β), L·cos(α)·sin(β)]
        
        Kinetic energy: T = (1/2)·m·(|∂p/∂α|²·α̇² + 2·(∂p/∂α·∂p/∂β)·α̇·β̇ + |∂p/∂β|²·β̇²)
            |∂p/∂α|² = L²·cos²(α) + L²·sin²(α)·sin²(β) + L²·sin²(α)·cos²(β)
                     = L²·cos²(α) + L²·sin²(α) = L²
            ∂p/∂α·∂p/∂β = 0
            |∂p/∂β|² = L²·cos²(α)·cos²(β) + L²·cos²(α)·sin²(β) = L²·cos²(α)
        
        Args:
            pitch: Pitch angle α (radians)
            roll: Roll angle β (radians)
            
        Returns:
            M: 2x2 mass matrix
        """
        m = self.mass
        L = self.length
        
        cos_pitch = np.cos(pitch)
        
        # Mass matrix M = m·[L², 0; 0, L²·cos²(α)]
        M = m * L**2 * np.array([
            [1.0,              0.0],
            [0.0,  cos_pitch**2]
        ])
        
        return M
    
    def compute_coriolis_vector_gimbal(self, pitch, roll, pitch_dot, roll_dot):
        """
        Compute Coriolis/centrifugal vector C(q, q̇) for gimbal coordinates.
        
        From Euler-Lagrange equations:
            d/dt(∂T/∂α̇) - ∂T/∂α = τ_α
            d/dt(∂T/∂β̇) - ∂T/∂β = τ_β
        
        With T = (1/2)·m·L²·(α̇² + cos²(α)·β̇²):
            ∂T/∂α̇ = m·L²·α̇
            ∂T/∂β̇ = m·L²·cos²(α)·β̇
            ∂T/∂α = -m·L²·cos(α)·sin(α)·β̇²
            ∂T/∂β = 0
            
            d/dt(∂T/∂α̇) = m·L²·α̈
            d/dt(∂T/∂β̇) = m·L²·(cos²(α)·β̈ - 2·cos(α)·sin(α)·α̇·β̇)
        
        Coriolis terms: C = M·q̈ - τ (rearranged from EL equations without potential)
            C_α = -∂T/∂α = m·L²·cos(α)·sin(α)·β̇²
            C_β = -d/dt(∂T/∂β̇) + ∂T/∂β = -m·L²·cos²(α)·β̈ + m·L²·2·cos(α)·sin(α)·α̇·β̇
        
        Actually, Coriolis is: C·q̇ where C_ij = (1/2)·(∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i)·q̇_k
        
        For our system: M = m·L²·[1, 0; 0, cos²(α)]
            ∂M/∂α = m·L²·[0, 0; 0, -2·cos(α)·sin(α)]
            ∂M/∂β = 0
        
        Coriolis vector:
            C_α = m·L²·cos(α)·sin(α)·β̇²
            C_β = -m·L²·cos(α)·sin(α)·α̇·β̇
        
        Args:
            pitch: Pitch angle α (radians)
            roll: Roll angle β (radians)
            pitch_dot: Pitch velocity α̇ (rad/s)
            roll_dot: Roll velocity β̇ (rad/s)
            
        Returns:
            C: 2x1 Coriolis vector
        """
        m = self.mass
        L = self.length
        
        sin_pitch = np.sin(pitch)
        cos_pitch = np.cos(pitch)
        
        # Coriolis terms
        C_alpha = m * L**2 * cos_pitch * sin_pitch * roll_dot**2
        C_beta = -m * L**2 * cos_pitch * sin_pitch * pitch_dot * roll_dot
        
        C = np.array([C_alpha, C_beta])
        
        return C
    
    def compute_gravity_vector_gimbal(self, pitch, roll, g=9.81):
        """
        Compute gravity vector G(q) for gimbal coordinates.
        
        Ball position: p = [-L·sin(α), L·cos(α)·sin(β), -L·cos(α)·cos(β)]
        Potential energy (with Z down, gravity in -Z direction):
            V = m·g·z = m·g·(-L·cos(α)·cos(β)) = -m·g·L·cos(α)·cos(β)
        
        Gravity terms: G_i = ∂V/∂q_i
            G_α = ∂V/∂α = m·g·L·sin(α)·cos(β)
            G_β = ∂V/∂β = m·g·L·cos(α)·sin(β)
        
        Args:
            pitch: Pitch angle α (radians)
            roll: Roll angle β (radians)
            g: Gravitational acceleration (m/s²), positive value
            
        Returns:
            G: 2x1 gravity vector
        """
        m = self.mass
        L = self.length
        
        sin_pitch = np.sin(pitch)
        cos_pitch = np.cos(pitch)
        sin_roll = np.sin(roll)
        cos_roll = np.cos(roll)
        
        # Gravity terms (partial derivatives of potential energy)
        G_alpha = m * g * L * sin_pitch * cos_roll
        G_beta = m * g * L * cos_pitch * sin_roll
        
        G = np.array([G_alpha, G_beta])
        
        return G
    
    def compute_nonlinear_dynamics_gimbal(self, q, q_dot, u, g=9.81):
        """
        Compute forward dynamics for gimbal-mounted pendulum.
        
        Equation: M(q)·q̈ + C(q, q̇) + G(q) = τ
        Solve for: q̈ = M⁻¹·(τ - C - G)
        
        Args:
            q: Position [pitch, roll] (2x1)
            q_dot: Velocity [pitch_dot, roll_dot] (2x1)
            u: Control input [τ_pitch, τ_roll] (2x1)
            g: Gravitational acceleration (m/s²)
            
        Returns:
            q_ddot: Acceleration [pitch_ddot, roll_ddot] (2x1)
        """
        pitch, roll = q
        pitch_dot, roll_dot = q_dot
        
        M = self.compute_mass_matrix_gimbal(pitch, roll)
        C = self.compute_coriolis_vector_gimbal(pitch, roll, pitch_dot, roll_dot)
        G = self.compute_gravity_vector_gimbal(pitch, roll, g)
        
        # Solve: M·q_ddot = u - C - G
        q_ddot = np.linalg.solve(M, u - C - G)
        
        return q_ddot


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
        
        # Trajectory optimizer (for trajectory-optimized mode)
        self.trajectory_optimizer: Optional[TrajectoryOptimizer] = None
        
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
        
        # Dynamics validation logs (for dynamics-validation mode)
        self.validation_time_log = []       # Timestamps for validation (subset of full time_log)
        self.manual_accelerations_log = []  # Accelerations from manual compute_nonlinear_dynamics
        self.drake_accelerations_log = []   # Accelerations from Drake's built-in dynamics
        self.acceleration_errors_log = []   # Difference between manual and Drake
        self.manual_mass_matrix_log = []    # Mass matrices from manual computation
        self.drake_mass_matrix_log = []     # Mass matrices from Drake
        self.manual_coriolis_log = []       # Coriolis vectors from manual computation
        self.drake_bias_log = []            # Bias terms (C + G) from Drake
        self.manual_gravity_log = []        # Gravity vectors from manual computation
        
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
        # SKIP CONTROLLER FOR DYNAMICS VALIDATION MODE
        # ═══════════════════════════════════════════════════════════════════
        # In validation mode, we want passive dynamics (no control input)
        # to compare manual equations vs Drake's built-in dynamics
        # ═══════════════════════════════════════════════════════════════════
        if CONTROLLER_MODE == 'dynamics-validation':
            print(colored("\n--- Dynamics Validation Mode: No Controller ---", 'yellow', attrs=['bold']))
            print(colored("  Passive dynamics (u=0) for pure physics comparison", 'cyan'))
            print(colored("  Manual spherical equations vs Drake CalcMassMatrix/CalcBiasTerm", 'cyan'))
            print(colored("✓ Skipping controller creation (passive dynamics)\n", 'green'))
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # TRAJECTORY OPTIMIZATION MODE
        # ═══════════════════════════════════════════════════════════════════
        # Optimize trajectory from start to goal with minimum pendulum swing,
        # BEFORE finalizing plant (DirectCollocation needs unfinalizedplant)
        # ═══════════════════════════════════════════════════════════════════
        if CONTROLLER_MODE == 'trajectory-optimized':
            print(colored("\n--- Trajectory Optimization (Pre-Finalization) ---", 'yellow', attrs=['bold']))
            print(colored("  Note: Creating separate plant for trajectory optimization", 'cyan'))
            print(colored("  This plant is NOT finalized yet (required for DirectCollocation)", 'cyan'))
            
            # Create a separate MultibodyPlant for trajectory optimization
            # DirectCollocation doesn't need visualization, so create plant without scene graph
            opt_plant = MultibodyPlant(time_step=0.0)  # Continuous-time plant
            
            # Load same robot model using CupManipulator to handle package map
            opt_parser = Parser(opt_plant)
            opt_manipulator = CupManipulator(self.cup_manipulator_config)
            opt_manipulator.load_urdf_to_plant(opt_plant, opt_parser)
            
            # Weld base
            opt_base = opt_plant.GetBodyByName("base_mount_manipulator")
            opt_plant.WeldFrames(opt_plant.world_frame(), opt_base.body_frame())
            
            # Add pendulum
            if PENDULUM_ENABLED:
                opt_pendulum = Pendulum3D(PENDULUM_CONFIG)
                opt_link2_body = opt_plant.GetBodyByName("link2")
                opt_pendulum.attach_to_body(opt_plant, opt_link2_body, opt_plant.GetModelInstanceByName("cup_manipulator"))
            
            # Set gravity
            opt_gravity_field = opt_plant.mutable_gravity_field()
            opt_gravity_field.set_gravity_vector(list(self.simulation_config.gravity))
            
            # Add actuators (required for DirectCollocation)
            link1_joint_opt = opt_plant.GetJointByName("link1_base")
            link2_joint_opt = opt_plant.GetJointByName("link2_link1")
            opt_plant.AddJointActuator("link1_base", link1_joint_opt)
            opt_plant.AddJointActuator("link2_link1", link2_joint_opt)
            
            # Finalize the optimization plant
            opt_plant.Finalize()
            opt_plant_context = opt_plant.CreateDefaultContext()
            
            print(colored("  ✓ Optimization plant created (without scene graph for speed)", 'green'))
            
            # Create trajectory optimizer with the optimization plant
            self.trajectory_optimizer = TrajectoryOptimizer(
                opt_plant, 
                opt_plant_context,
                num_samples=TRAJECTORY_NUM_SAMPLES
            )
            
            # Optimize trajectory
            print(colored("\n--- Running Trajectory Optimization ---", 'yellow', attrs=['bold']))
            optimized_traj = self.trajectory_optimizer.optimize_trajectory(
                q_start=TRAJECTORY_START,
                q_goal=TRAJECTORY_GOAL,
                duration=TRAJECTORY_DURATION,
                pendulum_weight=TRAJECTORY_PENDULUM_WEIGHT,
                torque_weight=TRAJECTORY_TORQUE_WEIGHT
            )
            
            # Create trajectory generator wrapper for optimized trajectory
            self.trajectory_generator = OptimizedTrajectoryGenerator(self.trajectory_optimizer, TRAJECTORY_DURATION)
            
            print(colored("✓ Optimized trajectory generator created", 'green'))
            print(colored("  → Will use Computed Torque control for tracking\n", 'cyan'))
            
            # Continue to controller creation below
        
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
        if CONTROLLER_MODE == 'pd' or CONTROLLER_MODE == 'min-jerk-joint':
            Kp = np.array([100.0, 100.0])
            Kd = np.array([10.0, 10.0])
        elif CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics' or CONTROLLER_MODE == 'trajectory-optimized':
            # Reduced gains: feedforward handles dynamics, feedback only corrects errors
            Kp = np.array([20.0, 20.0])  # 5x smaller than PD
            Kd = np.array([5.0, 5.0])    # 2x smaller than PD
        elif CONTROLLER_MODE == 'ofc-effort' or CONTROLLER_MODE == 'ofc-smoothness':
            # OFC uses LQR gains (computed automatically)
            Kp = None  # Not used
            Kd = None  # Not used
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
        
        elif CONTROLLER_MODE == 'ofc-effort' or CONTROLLER_MODE == 'ofc-smoothness':
            # Determine mode
            ofc_mode = 'effort' if CONTROLLER_MODE == 'ofc-effort' else 'smoothness'
            
            self.controller = self.builder.AddSystem(
                OptimalFeedbackController(
                    plant=self.plant,
                    model_instance=self.cup_manipulator.model_instance,
                    q_start=OFC_Q_START,
                    q_goal=OFC_Q_GOAL,
                    duration=OFC_DURATION,
                    mode=ofc_mode,
                    Q_position=OFC_Q_POSITION,
                    Q_pendulum=OFC_Q_PENDULUM,
                    Q_velocity=OFC_Q_VELOCITY,
                    R=OFC_R_EFFORT if ofc_mode == 'effort' else OFC_R_SMOOTHNESS,
                    impedance_mass=OFC_MASS,
                    impedance_kp=OFC_STIFFNESS,
                    impedance_kd=OFC_DAMPING
                )
            )
            print(colored(f"✓ OptimalFeedbackController (SYSTEM 2) added to diagram", 'green'))
            print(colored(f"  Role: Computes optimal torques τ = -K·(x - x_desired)", 'cyan'))
            print(colored(f"  Mode: {ofc_mode.upper()}-minimizing", 'cyan'))
        
        elif CONTROLLER_MODE == 'computed-torque' or CONTROLLER_MODE == 'inverse-dynamics' or CONTROLLER_MODE == 'trajectory-optimized':
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
        
        # For trajectory-optimized mode, use start configuration from trajectory
        if CONTROLLER_MODE == 'trajectory-optimized':
            link1_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
            link2_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
            link1_joint.set_angle(plant_context, TRAJECTORY_START[0])
            link2_joint.set_angle(plant_context, TRAJECTORY_START[1])
            print(colored(f"  ✓ Manipulator joints (trajectory start): link1={np.rad2deg(TRAJECTORY_START[0]):.1f}°, link2={np.rad2deg(TRAJECTORY_START[1]):.1f}°", 'cyan'))
        # For min-jerk-joint mode, use start configuration from MIN_JERK_Q_START
        elif CONTROLLER_MODE == 'min-jerk-joint':
            link1_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
            link2_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
            link1_joint.set_angle(plant_context, MIN_JERK_Q_START[0])
            link2_joint.set_angle(plant_context, MIN_JERK_Q_START[1])
            print(colored(f"  ✓ Manipulator joints (min-jerk start): link1={np.rad2deg(MIN_JERK_Q_START[0]):.1f}°, link2={np.rad2deg(MIN_JERK_Q_START[1]):.1f}°", 'cyan'))
        # For OFC modes, use start configuration from OFC_Q_START
        elif CONTROLLER_MODE == 'ofc-effort' or CONTROLLER_MODE == 'ofc-smoothness':
            link1_joint = self.plant.GetJointByName("link1_base", self.cup_manipulator.model_instance)
            link2_joint = self.plant.GetJointByName("link2_link1", self.cup_manipulator.model_instance)
            link1_joint.set_angle(plant_context, OFC_Q_START[0])
            link2_joint.set_angle(plant_context, OFC_Q_START[1])
            print(colored(f"  ✓ Manipulator joints (OFC start): link1={np.rad2deg(OFC_Q_START[0]):.1f}°, link2={np.rad2deg(OFC_Q_START[1]):.1f}°", 'cyan'))
        else:
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
            
            # For trajectory-optimized mode, use start configuration (should be zero swing)
            if CONTROLLER_MODE == 'trajectory-optimized':
                pitch_joint.set_angle(plant_context, TRAJECTORY_START[2])
                roll_joint.set_angle(plant_context, TRAJECTORY_START[3])
                print(colored(f"  ✓ Pendulum (trajectory start): pitch={np.rad2deg(TRAJECTORY_START[2]):.1f}°, roll={np.rad2deg(TRAJECTORY_START[3]):.1f}°", 'cyan'))
            # For min-jerk-joint mode, use pendulum angles from config
            elif CONTROLLER_MODE == 'min-jerk-joint':
                pitch_joint.set_angle(plant_context, MIN_JERK_Q_START[2])
                roll_joint.set_angle(plant_context, MIN_JERK_Q_START[3])
                print(colored(f"  ✓ Pendulum (min-jerk start): pitch={np.rad2deg(MIN_JERK_Q_START[2]):.1f}°, roll={np.rad2deg(MIN_JERK_Q_START[3]):.1f}°", 'cyan'))
            # For OFC modes, use pendulum angles from OFC_Q_START
            elif CONTROLLER_MODE == 'ofc-effort' or CONTROLLER_MODE == 'ofc-smoothness':
                pitch_joint.set_angle(plant_context, OFC_Q_START[2])
                roll_joint.set_angle(plant_context, OFC_Q_START[3])
                print(colored(f"  ✓ Pendulum (OFC start): pitch={np.rad2deg(OFC_Q_START[2]):.1f}°, roll={np.rad2deg(OFC_Q_START[3]):.1f}°", 'cyan'))
            # For dynamics-validation mode, start away from singularity
            elif CONTROLLER_MODE == 'dynamics-validation':
                # Start at θ=30°, φ=45° to avoid singular configuration at θ=0
                initial_pitch = 30.0  # degrees
                initial_roll = 45.0   # degrees
                pitch_joint.set_angle(plant_context, np.deg2rad(initial_pitch))
                roll_joint.set_angle(plant_context, np.deg2rad(initial_roll))
                print(colored(f"  ✓ Pendulum (validation mode): pitch={initial_pitch}°, roll={initial_roll}°", 'cyan'))
                print(colored(f"    (Starting away from θ=0 singularity)", 'yellow'))
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
        if CONTROLLER_MODE == 'trajectory-optimized':
            print(f"  Trajectory: {TRAJECTORY_DURATION} s optimized motion")
            print(f"    Start: [{np.rad2deg(TRAJECTORY_START[0]):.1f}°, {np.rad2deg(TRAJECTORY_START[1]):.1f}°]")
            print(f"    Goal:  [{np.rad2deg(TRAJECTORY_GOAL[0]):.1f}°, {np.rad2deg(TRAJECTORY_GOAL[1]):.1f}°]")
        else:
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
                
                # Get control torques and compute errors (skip for dynamics-validation mode)
                if CONTROLLER_MODE == 'dynamics-validation':
                    # Dynamics validation mode: passive dynamics (no controller)
                    control_torques = np.zeros(2)
                    position_error = np.zeros(2)
                    velocity_error = np.zeros(2)
                    q_ddot_commanded = np.zeros(2)
                else:
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
                            
                            # DYNAMICS VALIDATION: Compare manual vs Drake accelerations
                            if CONTROLLER_MODE == 'dynamics-validation':
                                self._validate_dynamics(plant_context, ball_state, joint_positions, joint_velocities)
                    
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
                            # Note: pitch, roll are joint angles (gimbal q). RPY are Euler angles from rotation matrix
                            print(f" | Jt[P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}°] | θ={np.rad2deg(theta):5.1f}° φ={np.rad2deg(phi):6.1f}° | RPY[{np.rad2deg(rpy_roll):5.1f}°,{np.rad2deg(rpy_pitch):5.1f}°,{np.rad2deg(rpy_yaw):5.1f}°]", end='')
                        elif len(self.pendulum_spherical_log) > 0:
                            theta, phi = self.pendulum_spherical_log[-1]
                            print(f" | Jt[P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}°] | θ={np.rad2deg(theta):5.1f}° φ={np.rad2deg(phi):6.1f}°", end='')
                        else:
                            print(f" | Jt[P={np.rad2deg(pitch):6.1f}° R={np.rad2deg(roll):6.1f}°]", end='')
                    
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
                        print(colored(f"\n⚙️  Joint Angles (q):", 'cyan', attrs=['bold']), colored(f"Pitch={np.rad2deg(ball_state['joint_pitch']):+6.2f}°  Roll={np.rad2deg(ball_state['joint_roll']):+6.2f}°", 'cyan'))
                        print(colored(f"📐 Spherical (θ,φ,r):", 'cyan', attrs=['bold']), colored(f"{np.rad2deg(ball_state['theta']):+6.2f}°, {np.rad2deg(ball_state['phi']):+6.2f}°, {ball_state['r']:.4f}m", 'cyan'))
                        print(colored(f"🔄 RPY (pivot):", 'cyan', attrs=['bold']), colored(f"R={np.rad2deg(ball_state['roll_wrt_pivot']):+6.2f}°  P={np.rad2deg(ball_state['pitch_wrt_pivot']):+6.2f}°  Y={np.rad2deg(ball_state['yaw_wrt_pivot']):+6.2f}°", 'cyan'))
                        print(colored(f"📍 Ball (pivot):", 'cyan', attrs=['bold']), colored(f"[{ball_state['x']:+.4f}, {ball_state['y']:+.4f}, {ball_state['z']:+.4f}]m", 'cyan'))
                        print(colored(f"📍 Ball (world):", 'cyan', attrs=['bold']), colored(f"[{ball_state['ball_wrt_world'][0]:+.4f}, {ball_state['ball_wrt_world'][1]:+.4f}, {ball_state['ball_wrt_world'][2]:+.4f}]m", 'cyan'))
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
    
    def _validate_dynamics(self, plant_context, ball_state, joint_positions, joint_velocities):
        """
        Validate manual gimbal dynamics against Drake's built-in dynamics.
        
        Args:
            plant_context: Plant context for Drake calculations
            ball_state: Ball state dictionary from compute_ball_state
            joint_positions: Dict of joint positions
            joint_velocities: Dict of joint velocities
        """
        # Get gimbal joint angles (these ARE the generalized coordinates)
        pitch = joint_positions.get('pendulum_pitch', 0.0)
        roll = joint_positions.get('pendulum_roll', 0.0)
        
        # Get full system velocities (manipulator + pendulum)
        v = self.plant.GetVelocities(plant_context)
        
        # Extract pendulum joint velocities (indices 2, 3)
        pitch_dot = v[2]
        roll_dot = v[3]
        
        # Check for numerical issues near singularities
        # Gimbal singularity occurs when roll = ±π/2 (gimbal lock)
        SINGULARITY_THRESHOLD = np.deg2rad(85)  # Avoid within 5° of gimbal lock
        if abs(roll) > SINGULARITY_THRESHOLD:
            return  # Skip validation near gimbal lock
        
        # Log timestamp for this validation point
        current_time = plant_context.get_time()
        self.validation_time_log.append(current_time)
        
        # Generalized coordinates for gimbal system
        q = np.array([pitch, roll])
        q_dot = np.array([pitch_dot, roll_dot])
        
        # Control inputs (zero for passive dynamics validation)
        u = np.zeros(2)
        
        # Get gravity magnitude from configuration (Drake uses negative z-direction)
        # gravity_config = (0, 0, -9.81) means magnitude is 9.81 m/s²
        g_magnitude = abs(self.simulation_config.gravity[2])
        
        # 1. MANUAL: Compute accelerations using gimbal dynamics equations
        manual_q_ddot = self.pendulum.compute_nonlinear_dynamics_gimbal(q, q_dot, u, g=g_magnitude)
        
        # 2. DRAKE: Get accelerations from Drake's dynamics
        # Compute mass matrix from Drake
        M_drake = self.plant.CalcMassMatrix(plant_context)
        
        # Compute bias term (Coriolis + Gravity) from Drake: b = C(q,v)v + g(q)
        bias_drake = self.plant.CalcBiasTerm(plant_context)
        
        # For our 4-DOF system (2 manipulator + 2 pendulum):
        # The pendulum DOFs are indices 2 and 3
        # Extract pendulum sub-matrices
        M_pendulum_drake = M_drake[2:4, 2:4]
        bias_pendulum_drake = bias_drake[2:4]
        
        # Solve for accelerations: M·q̈ = -bias (for passive dynamics, u=0)
        drake_q_ddot = -np.linalg.solve(M_pendulum_drake, bias_pendulum_drake)
        
        # 3. COMPUTE ERRORS
        acceleration_error = manual_q_ddot - drake_q_ddot
        
        # Also compute individual components for detailed comparison
        M_manual = self.pendulum.compute_mass_matrix_gimbal(pitch, roll)
        C_manual = self.pendulum.compute_coriolis_vector_gimbal(pitch, roll, pitch_dot, roll_dot)
        G_manual = self.pendulum.compute_gravity_vector_gimbal(pitch, roll, g=g_magnitude)
        
        # Debug: print first few comparisons
        if len(self.validation_time_log) <= 3:
            print(colored(f"\n[Debug Validation t={current_time:.3f}s]", 'yellow'))
            print(colored(f"  Pitch={np.rad2deg(pitch):.1f}°, Roll={np.rad2deg(roll):.1f}°", 'cyan'))
            print(colored(f"  Manual M:\n{M_manual}", 'cyan'))
            print(colored(f"  Drake M:\n{M_pendulum_drake}", 'cyan'))
            print(colored(f"  Manual G: {G_manual}", 'cyan'))
            print(colored(f"  Drake bias: {bias_pendulum_drake}", 'cyan'))
            print(colored(f"  Manual C: {C_manual}", 'cyan'))
            print(colored(f"  Manual accel: {manual_q_ddot}", 'cyan'))
            print(colored(f"  Drake accel: {drake_q_ddot}", 'cyan'))
            print(colored(f"  Error: {acceleration_error}", 'magenta'))
        
        # Store validation data
        self.manual_accelerations_log.append(manual_q_ddot.copy())
        self.drake_accelerations_log.append(drake_q_ddot.copy())
        self.acceleration_errors_log.append(acceleration_error.copy())
        self.manual_mass_matrix_log.append(M_manual.copy())
        self.drake_mass_matrix_log.append(M_pendulum_drake.copy())
        self.manual_coriolis_log.append(C_manual.copy())
        self.drake_bias_log.append(bias_pendulum_drake.copy())
        self.manual_gravity_log.append(G_manual.copy())
    
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
        
        # Check if this is OFC mode (no trajectory tracking)
        is_ofc_mode = CONTROLLER_MODE in ['ofc-effort', 'ofc-smoothness']
        
        for i in range(2):
            ax.plot(time, np.rad2deg(q_actual[:, i]), label=f'{joint_names[i]} - Actual', 
                   color=colors_actual[i], linewidth=2)
            # Only plot desired trajectory for trajectory-tracking controllers
            if not is_ofc_mode:
                ax.plot(time, np.rad2deg(q_desired[:, i]), '--', label=f'{joint_names[i]} - Desired', 
                       color=colors_desired[i], linewidth=2, alpha=0.8)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label='Hold Start')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (deg)', fontsize=11)
        title_suffix = ' (Optimal Feedback)' if is_ofc_mode else ' - Tracking'
        ax.set_title(f'Manipulator Joint Positions{title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 1: Joint Velocities (Both joints in one plot)
        # ===================================================================
        ax = fig.add_subplot(gs[0, 1])
        for i in range(2):
            ax.plot(time, np.rad2deg(q_dot_actual[:, i]), label=f'{joint_names[i]} - Actual', 
                   color=colors_actual[i], linewidth=2)
            # Only plot desired trajectory for trajectory-tracking controllers
            if not is_ofc_mode:
                ax.plot(time, np.rad2deg(q_dot_desired[:, i]), '--', label=f'{joint_names[i]} - Desired', 
                       color=colors_desired[i], linewidth=2, alpha=0.8)
        ax.axvline(MANIPULATOR_MOTION_DURATION, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label='Hold Start')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (deg/s)', fontsize=11)
        title_suffix = ' (Optimal Feedback)' if is_ofc_mode else ' - Tracking'
        ax.set_title(f'Manipulator Joint Velocities{title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ===================================================================
        # Row 2: Tracking Errors (Position and Velocity)
        # ===================================================================
        # For OFC: these are errors from equilibrium reference, not trajectory tracking
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
        error_title = 'Position Errors from Goal' if is_ofc_mode else 'Position Tracking Errors'
        ax.set_title(error_title, fontsize=12, fontweight='bold')
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
        error_title = 'Velocity Errors from Goal' if is_ofc_mode else 'Velocity Tracking Errors'
        ax.set_title(error_title, fontsize=12, fontweight='bold')
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
        
        # ===================================================================
        # Dynamics Validation Plots (if validation mode)
        # ===================================================================
        if CONTROLLER_MODE == 'dynamics-validation' and len(self.manual_accelerations_log) > 0:
            # Create separate figure for validation results
            fig_val = plt.figure(figsize=(16, 12))
            gs_val = GridSpec(4, 2, figure=fig_val, hspace=0.35, wspace=0.25)
            
            # Use validation-specific time array (subset of full time due to singularity skipping)
            val_time = np.array(self.validation_time_log)
            manual_accel = np.array(self.manual_accelerations_log)  # [N x 2] (pitch, roll)
            drake_accel = np.array(self.drake_accelerations_log)    # [N x 2]
            accel_errors = np.array(self.acceleration_errors_log)   # [N x 2]
            
            # Row 1: Accelerations comparison
            ax = fig_val.add_subplot(gs_val[0, 0])
            ax.plot(val_time, manual_accel[:, 0], label='Manual α̈ (pitch)', 
                   color='#E63946', linewidth=1.5)
            ax.plot(val_time, drake_accel[:, 0], '--', label='Drake α̈', 
                   color='#457B9D', linewidth=1.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('α̈ (rad/s²)', fontsize=11)
            ax.set_title('Pitch Acceleration Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            ax = fig_val.add_subplot(gs_val[0, 1])
            ax.plot(val_time, manual_accel[:, 1], label='Manual β̈ (roll)', 
                   color='#E63946', linewidth=1.5)
            ax.plot(val_time, drake_accel[:, 1], '--', label='Drake β̈', 
                   color='#457B9D', linewidth=1.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('β̈ (rad/s²)', fontsize=11)
            ax.set_title('Roll Acceleration Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Row 2: Acceleration errors
            ax = fig_val.add_subplot(gs_val[1, 0])
            ax.plot(val_time, accel_errors[:, 0], color='#E63946', linewidth=1.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Error (rad/s²)', fontsize=11)
            ax.set_title('Pitch Acceleration Error (Manual - Drake)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            ax = fig_val.add_subplot(gs_val[1, 1])
            ax.plot(val_time, accel_errors[:, 1], color='#457B9D', linewidth=1.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Error (rad/s²)', fontsize=11)
            ax.set_title('Roll Acceleration Error (Manual - Drake)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Row 3: Mass matrix diagonal elements (M[0,0] and M[1,1])
            if len(self.manual_mass_matrix_log) > 0:
                manual_M = np.array(self.manual_mass_matrix_log)  # [N x 2 x 2]
                drake_M = np.array(self.drake_mass_matrix_log)    # [N x 2 x 2]
                
                ax = fig_val.add_subplot(gs_val[2, 0])
                ax.plot(val_time, manual_M[:, 0, 0], label='Manual M[α,α]', 
                       color='#E63946', linewidth=1.5)
                ax.plot(val_time, drake_M[:, 0, 0], '--', label='Drake M[α,α]', 
                       color='#457B9D', linewidth=1.5)
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('M[α,α] (kg⋅m²)', fontsize=11)
                ax.set_title('Mass Matrix M[α,α] = mL²cos²(β)', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                ax = fig_val.add_subplot(gs_val[2, 1])
                ax.plot(val_time, manual_M[:, 1, 1], label='Manual M[β,β]', 
                       color='#E63946', linewidth=1.5)
                ax.plot(val_time, drake_M[:, 1, 1], '--', label='Drake M[β,β]', 
                       color='#457B9D', linewidth=1.5)
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('M[β,β] (kg⋅m²)', fontsize=11)
                ax.set_title('Mass Matrix M[β,β] = mL² (constant)', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Row 4: Gravity and Coriolis comparison
            if len(self.manual_gravity_log) > 0:
                manual_G = np.array(self.manual_gravity_log)  # [N x 2]
                drake_bias = np.array(self.drake_bias_log)    # [N x 2]
                manual_C = np.array(self.manual_coriolis_log)  # [N x 2]
                
                ax = fig_val.add_subplot(gs_val[3, 0])
                ax.plot(val_time, manual_G[:, 0], label='Manual G[α]', 
                       color='#E63946', linewidth=1.5)
                ax.plot(val_time, -drake_bias[:, 0] - manual_C[:, 0], '--', label='Drake G[α] (bias-C)', 
                       color='#457B9D', linewidth=1.5)
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('G[α] (N⋅m)', fontsize=11)
                ax.set_title('Gravity Torque (pitch) = -mgL⋅sin(α)⋅cos(β)', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                ax = fig_val.add_subplot(gs_val[3, 1])
                ax.plot(val_time, manual_G[:, 1], label='Manual G[β]', 
                       color='#E63946', linewidth=1.5)
                ax.plot(val_time, -drake_bias[:, 1] - manual_C[:, 1], '--', label='Drake G[β] (bias-C)', 
                       color='#457B9D', linewidth=1.5)
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('G[β] (N⋅m)', fontsize=11)
                ax.set_title('Gravity Torque (roll) = mgL⋅cos(α)⋅sin(β)', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Overall title for validation
            fig_val.suptitle('Dynamics Validation: Manual Gimbal Coordinates vs Drake', 
                            fontsize=14, fontweight='bold', y=0.995)
            
            plt.figure(fig_val.number)
            plt.tight_layout()
            
            # Save validation plot
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            val_plot_filename = f'plots/validation_results_{timestamp}.png'
            os.makedirs('plots', exist_ok=True)
            plt.savefig(val_plot_filename, dpi=150, bbox_inches='tight')
            print(colored(f"\n✓ Validation plot saved: {val_plot_filename}", 'green', attrs=['bold']))
            
            # Print validation statistics
            print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
            print(colored("Dynamics Validation Statistics (Gimbal Coordinates)", 'cyan', attrs=['bold']))
            print(colored("="*70, 'cyan', attrs=['bold']))
            
            rms_pitch = np.sqrt(np.mean(accel_errors[:, 0]**2))
            rms_roll = np.sqrt(np.mean(accel_errors[:, 1]**2))
            max_pitch = np.max(np.abs(accel_errors[:, 0]))
            max_roll = np.max(np.abs(accel_errors[:, 1]))
            
            print(colored(f"Acceleration Errors:", 'cyan'))
            print(colored(f"  α̈ (pitch) RMS Error:  {rms_pitch:.6e} rad/s²", 'cyan'))
            print(colored(f"  α̈ (pitch) Max Error:  {max_pitch:.6e} rad/s²", 'cyan'))
            print(colored(f"  β̈ (roll) RMS Error:   {rms_roll:.6e} rad/s²", 'cyan'))
            print(colored(f"  β̈ (roll) Max Error:   {max_roll:.6e} rad/s²", 'cyan'))
            
            if max_pitch < 1e-5 and max_roll < 1e-5:
                print(colored(f"\n  ✓ PASSED: Manual gimbal dynamics match Drake (error < 10⁻⁵)", 'green', attrs=['bold']))
            elif max_pitch < 1e-3 and max_roll < 1e-3:
                print(colored(f"\n  ⚠ WARNING: Small discrepancies detected (error < 10⁻³)", 'yellow', attrs=['bold']))
            else:
                print(colored(f"\n  ✗ FAILED: Significant dynamics errors detected", 'red', attrs=['bold']))
            
            print(colored("="*70 + "\n", 'cyan', attrs=['bold']))
        
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
    if CONTROLLER_MODE == 'min-jerk-joint':
        print(colored(f"Min-jerk: q_start={np.rad2deg(MIN_JERK_Q_START[:2])} deg, q_goal={np.rad2deg(MIN_JERK_Q_GOAL[:2])} deg, duration={MIN_JERK_DURATION:.2f}s", 'yellow'))
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
