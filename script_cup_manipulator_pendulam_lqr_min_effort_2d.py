#!/usr/bin/env python3
"""
2D Cart-Pendulum with Muscle Dynamics & Optimal Feedback Control
Extended from 1D CartPendulum architecture

This script extends the CartPendulum class from script_cart_pendulum_muscle_dynamics_ofc.py
to 2D motion (x, y) instead of just 1D (x).

STATE VECTOR (14D):
==================
1. x       - Cart X position [m]
2. y       - Cart Y position [m]
3. α       - Pendulum pitch angle [rad]
4. β       - Pendulum roll angle [rad]
5. ẋ       - Cart X velocity [m/s]
6. ẏ       - Cart Y velocity [m/s]
7. α̇       - Pendulum pitch velocity [rad/s]
8. β̇       - Pendulum roll velocity [rad/s]
9. F_x     - Muscle force state X [N]
10. F_y    - Muscle force state Y [N]
11. x_ref  - ZFT reference X position [m]
12. y_ref  - ZFT reference Y position [m]
13. ẋ_ref  - ZFT reference X velocity [m/s]
14. ẏ_ref  - ZFT reference Y velocity [m/s]

CONTROL: u = [u_x, u_y] (2D neural command)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from termcolor import colored
from scipy.linalg import solve_discrete_are
from typing import Literal

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    LogVectorOutput,
    LeafSystem,
    BasicVector,
    MeshcatVisualizer,
    StartMeshcat,
    Multiplexer,
    Demultiplexer,
    Saturation,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    RigidTransform,
    RevoluteJoint,
    PrismaticJoint,
    Sphere,
    Cylinder,
    Parser,
    ZeroOrderHold,
    JacobianWrtVariable,
    InverseKinematics,
    Solve,
    ConstantVectorSource,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint, PrismaticJoint
from pydrake.math import RigidTransform, RollPitchYaw

from pydrake.multibody.tree import MultibodyForces

# Import from existing script
import sys
sys.path.append(str(Path(__file__).parent))
from configs.robot.robot_types import (
    create_cart_pendulum_config, 
    create_cup_manipulator_config,
    ManipulatorConfig
)


from scipy.linalg import solve_discrete_are

from utils.utils import (
    build_linearized_system_2d,
    build_linearized_for_complete_system_2d,
    check_trajectory_feasibility,
    test_and_visualize_ik_feasibility,
)

# ============================================================================
# VISUALIZATION (moved to viz.py)
# ============================================================================
from utils.viz import (
    visualize_plant_meshcat,
    add_frames_to_meshcat,
    plot_frames_top_view,
    plot_lqr_manip_ee_traj_track_results,
    set_meshcat_camera_view,
)

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

# Add Meshcat camera arguments
parser = argparse.ArgumentParser(description='2D Cart-Pendulum with Muscle Dynamics & OFC')
parser.add_argument('--mode', type=str, 
                    default='lqr-applied-to-cart-manip-following-cart',
                    help='Simulation mode (scene-viz | lqr-applied-to-cart-manip-following-cart | lqr-applied-to-both-cart-manip)')
parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration [s]')
parser.add_argument('--target-x', type=float, default=-1, help='Target X position [m]')
parser.add_argument('--target-y', type=float, default=0.5, help='Target Y position [m]')
parser.add_argument('--cart-x-init', type=float, default=2, help='Initial cart X position [m] (default: use manipulator EE position)')
parser.add_argument('--cart-y-init', type=float, default=0.0, help='Initial cart Y position [m] (default: use manipulator EE position)')
parser.add_argument('--horizon', type=float, default=10.0, help='LQR horizon [s]')
parser.add_argument('--speed-scale', type=float, default=0.5, help='Trajectory speed scaling (0-1, lower=slower)')
parser.add_argument('--meshcat-azimuth', type=float, default=0.0, help='Meshcat camera azimuth angle in degrees (0 = +X, 90 = +Y)')
parser.add_argument('--meshcat-elevation', type=float, default=75.0, help='Meshcat camera elevation angle in degrees (90 = top view, 0 = side view)')
parser.add_argument('--meshcat-distance', type=float, default=3.0, help='Meshcat camera distance from target')
# Parse and save our args FIRST
_parsed_args, _ = parser.parse_known_args()

# Temporarily clear sys.argv to prevent CupManipulator module from parsing our args
# (it has its own argparse that would fail on our mode choices)
import sys
_saved_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Keep only script name

# Now import CupManipulator safely
# from script_cup_manipulator_controller_ofc import CupManipulator

# Restore sys.argv and our parsed args
sys.argv = _saved_argv
args = _parsed_args

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================
# CONFIG CLASSES — imported from dedicated modules
# ============================================================================
from configs.robot.robot_configs import (
    CartPendulumPhysicsConfig,
    EndEffectorKinematics2DConfig,
    create_physics_config,
)
from configs.controller.controller_configs import (
    MuscleDynamicsConfig,
    ImpedanceForceConfig,
    ZFTReferenceMassConfig,
    FiniteHorizonLQRConfig,
    ZFTJointReferenceIKConfig,
    create_muscle_config,
    create_impedance_config,
    create_zft_config,
    create_lqr_config,
)

# ROBOT CLASSES — imported from robots/cup_manipulator.py
from robots.cup_manipulator import RobotBase, CupManipulator, CartPendulum2DExtended

@dataclass
class SimulationConfig:
    """
    Configuration for simulation execution.
    
    Consolidates all parameters needed to set up and run a simulation,
    including physics, control, and visualization settings.
    """
    # System configs
    physics_config: CartPendulumPhysicsConfig
    muscle_config: MuscleDynamicsConfig | None
    impedance_config: ImpedanceForceConfig | None
    zft_config: ZFTReferenceMassConfig | None
    
    # Manipulator setup
    manipulator_urdf_path: str
    manipulator_joint_angles: Dict[str, float]
    manipulator_damping: tuple = (0.1, 0.1)
    
    # Simulation parameters
    target_x: float = 0.0
    target_y: float = 0.5
    duration: float = 10.0
    horizon: float = 10.0
    
    # Visualization
    meshcat: Optional[object] = None
    
    @classmethod
    def from_args(cls, args, physics_config, muscle_config, impedance_config, zft_config, meshcat):
        """
        Create SimulationConfig from command-line arguments and existing configs.
        
        Args:
            args: argparse.Namespace with target_x, target_y, duration, horizon
            physics_config: CartPendulumPhysicsConfig instance
            muscle_config: MuscleDynamicsConfig instance
            impedance_config: ImpedanceForceConfig instance
            zft_config: ZFTReferenceMassConfig instance
            meshcat: Meshcat instance
            
        Returns:
            SimulationConfig instance
        """
        # Extract joint angles and damping from ManipulatorConfig
        joint_angles_dict = MANIPULATOR_CONFIG.get_joint_positions_dict()
        damping_tuple = tuple(
            MANIPULATOR_CONFIG.joint_configs[jt].damping 
            for jt in ['link1_base', 'link2_link1']
        )
        
        return cls(
            physics_config=physics_config,
            muscle_config=muscle_config,
            impedance_config=impedance_config,
            zft_config=zft_config,
            manipulator_urdf_path=MANIPULATOR_CONFIG.urdf_path,
            manipulator_joint_angles=joint_angles_dict,
            manipulator_damping=damping_tuple,
            target_x=args.target_x,
            target_y=args.target_y,
            duration=args.duration,
            horizon=args.horizon,
            meshcat=meshcat,
        )

# Global configurations
PHYSICS_CONFIG = create_physics_config()
MUSCLE_CONFIG = create_muscle_config()
IMPEDANCE_CONFIG = create_impedance_config()
ZFT_CONFIG = create_zft_config()
LQR_CONFIG = create_lqr_config(
    x_goal=np.array([args.target_x, args.target_y, 0, 0, 0, 0, 0, 0, 0, 0, args.target_x, args.target_y, 0, 0]),
    horizon=args.horizon
)
MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path="model_using_onshape_to_robot/cup_manipulator2/cup_manipulator_obj_right_frame.urdf",
    joint_angles={
        'link1_base': np.deg2rad(0.0),   # q1: Base to link1
        'link2_link1': np.deg2rad(20.0), # q2: Link1 to link2
    },
    damping=(0.1, 0.1),
)
SIMULATION_CONFIG = SimulationConfig.from_args(args, PHYSICS_CONFIG, MUSCLE_CONFIG, IMPEDANCE_CONFIG, ZFT_CONFIG, None)

# Set welding mode based on selected mode
WELD_CART_TO_MANIP_EE = True if args.mode == 'lqr-applied-to-both-cart-manip' else False      




# ============================================================================
# FRAME UPDATER SYSTEM
# ============================================================================

class MeshcatFrameUpdater(LeafSystem):
    """
    Updates coordinate frame visualizations in Meshcat during simulation.
    This system reads the plant state and updates all frame transforms.
    """
    
    def __init__(self, meshcat, plant, frame_list, update_period=0.033):
        """
        Args:
            meshcat: Meshcat instance
            plant: MultibodyPlant
            frame_list: List of (frame_name, frame, length) tuples
            update_period: Update frequency in seconds (default 30 Hz)
        """
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.plant = plant
        self.frame_list = frame_list
        
        # Input port for plant state
        self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        
        # Periodic update
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=update_period,
            offset_sec=0.0,
            update=self._update_frames
        )
    
    def _update_frames(self, context, state):
        """Update frame positions in Meshcat."""
        # Get plant context
        plant_context = self.plant.CreateDefaultContext()
        
        # Get state from input port
        x = self.get_input_port(0).Eval(context)
        self.plant.SetPositionsAndVelocities(plant_context, x)
        
        # Update all frames
        for frame_name, frame, length in self.frame_list:
            X_WF = self.plant.CalcRelativeTransform(
                plant_context, 
                self.plant.world_frame(), 
                frame
            )
            self.meshcat.SetTransform(f"/Frames/{frame_name}", X_WF)


# ============================================================================
# MUSCLE DYNAMICS (2D)
# ============================================================================

class MuscleDynamics2D(LeafSystem):
    """
    2D muscle dynamics: Ḟ = (-F + u) / τ
    
    Input: u (2) = [u_x, u_y] neural command
    Output: F (2) = [F_x, F_y] muscle force
    State: F (2)
    """
    
    def __init__(self, config: MuscleDynamicsConfig):
        LeafSystem.__init__(self)
        self.muscle_tau = config.muscle_tau
        self.initial_force = config.initial_force
        
        # State: [F_x, F_y]
        self.DeclareContinuousState(2)
        
        # Input: command u
        self.DeclareVectorInputPort("u", 2)
        
        # Output: force F
        self.DeclareVectorOutputPort("F", 2, self.calc_output)
    
    def SetDefaultState(self, context, state):
        state.SetFromVector(self.initial_force)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        F = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port().Eval(context)
        F_dot = (-F + u) / self.muscle_tau
        derivatives.get_mutable_vector().SetFromVector(F_dot)
    
    def calc_output(self, context, output):
        F = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(F)


# ============================================================================
# ZFT REFERENCE MASS (2D)
# ============================================================================

class ZFTReferenceMass2D(LeafSystem):
    """
    2D ZFT reference mass dynamics:
    
    ẍ_ref = (K*(x - x_ref) + D*(ẋ - ẋ_ref) + F) / M
    ÿ_ref = (K*(y - y_ref) + D*(ẏ - ẏ_ref) + F) / M
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]
      1: F (2) = [F_x, F_y]
    Outputs:
      0: ref_state  (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]  — backward-compat
      1: p_zft      (2) = [x_ref, y_ref]
      2: pdot_zft   (2) = [ẋ_ref, ẏ_ref]
      3: pddot_zft  (2) = [ẍ_ref, ÿ_ref]
    State: [x_ref, y_ref, ẋ_ref, ẏ_ref]
    """
    
    def __init__(self, config: ZFTReferenceMassConfig):
        LeafSystem.__init__(self)
        self.M_ref = config.M_ref
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        self.initial_ref = config.initial_ref
        
        # State: [x_ref, y_ref, ẋ_ref, ẏ_ref]
        self.DeclareContinuousState(4)
        
        # Inputs
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("F", 2)
        
        # Port 0: full ref state (backward compatible)
        self.DeclareVectorOutputPort("ref_state", 4, self.calc_output,
                                     {self.xc_ticket()})
        # Port 1: position only
        self.DeclareVectorOutputPort("p_zft", 2, self._calc_p_zft,
                                     {self.xc_ticket()})
        # Port 2: velocity only
        self.DeclareVectorOutputPort("pdot_zft", 2, self._calc_pdot_zft,
                                     {self.xc_ticket()})
        # Port 3: acceleration (depends on inputs + state)
        self.DeclareVectorOutputPort("pddot_zft", 2, self._calc_pddot_zft,
                                     {self.all_input_ports_ticket(), self.xc_ticket()})
    
    def SetDefaultState(self, context, state):
        state.SetFromVector(self.initial_ref)

    # ------------------------------------------------------------------
    # Shared helper: compute p̈_zft from current context
    # ------------------------------------------------------------------
    def _get_pddot(self, context) -> np.ndarray:
        ref_state  = context.get_continuous_state_vector().CopyToVector()
        cart_state = self.get_input_port(0).Eval(context)
        F          = self.get_input_port(1).Eval(context)
        x,  y,  x_dot,  y_dot  = cart_state
        x_ref, y_ref, x_ref_dot, y_ref_dot = ref_state
        p      = np.array([x,     y    ])
        p_dot  = np.array([x_dot, y_dot])
        p_zft  = np.array([x_ref, y_ref])
        p_zft_dot = np.array([x_ref_dot, y_ref_dot])
        return (self.K_imp * (p - p_zft) + self.D_imp * (p_dot - p_zft_dot) + F) / self.M_ref

    def DoCalcTimeDerivatives(self, context, derivatives):
        ref_state = context.get_continuous_state_vector().CopyToVector()
        x_ref_dot, y_ref_dot = ref_state[2], ref_state[3]
        x_ref_ddot, y_ref_ddot = self._get_pddot(context)
        derivatives.get_mutable_vector().SetFromVector(
            np.array([x_ref_dot, y_ref_dot, x_ref_ddot, y_ref_ddot])
        )
    
    def calc_output(self, context, output):
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())

    def _calc_p_zft(self, context, output):
        s = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(s[:2])   # p_zft = [x_ref, y_ref]

    def _calc_pdot_zft(self, context, output):
        s = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(s[2:])   # p_zft_dot = [ẋ_ref, ẏ_ref]

    def _calc_pddot_zft(self, context, output):
        output.SetFromVector(self._get_pddot(context))   # p_zft_ddot = [ẍ_ref, ÿ_ref]


# ============================================================================
# IMPEDANCE FORCE (2D)
# ============================================================================

class ImpedanceForce2D(LeafSystem):
    """
    2D impedance force: F_imp = K*(r_ref - r) + D*(ṙ_ref - ṙ)
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]
      1: ref_state (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]
    Output:
      0: F_imp (2) = [F_x, F_y]
    """
    
    def __init__(self, config: ImpedanceForceConfig):
        LeafSystem.__init__(self)
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        
        # Inputs
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("ref_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("F_imp", 2, self.calc_output)
    
    def calc_output(self, context, output):
        cart = self.get_input_port(0).Eval(context)
        ref = self.get_input_port(1).Eval(context)
        
        x, y, x_dot, y_dot = cart
        x_ref, y_ref, x_ref_dot, y_ref_dot = ref
        
        F_x = self.K_imp * (x_ref - x) + self.D_imp * (x_ref_dot - x_dot)
        F_y = self.K_imp * (y_ref - y) + self.D_imp * (y_ref_dot - y_dot)
        
        output.SetFromVector(np.array([F_x, F_y]))


# ============================================================================
# FINITE-HORIZON LQR CONTROLLER (2D)
# ============================================================================

class FiniteHorizonLQRController2D(LeafSystem):
    """
    Finite-horizon LQR for 14D system with 2D control.
    
    State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    Control: [u_x, u_y]
    
    Solves backward Riccati recursion and applies time-varying gain.
    """
    
    def __init__(self, A, B, config: FiniteHorizonLQRConfig):
        LeafSystem.__init__(self)
        self.Q = config.Q
        self.QN = config.QN if config.QN is not None else config.Q.copy()
        self.R = config.R
        self.x_goal = config.x_goal
        self.T = float(config.horizon)
        self.dt = float(config.timestep)
        self.u_limits = config.u_limits
        
        # Discretize system
        n = A.shape[0]
        m = B.shape[1]
        I = np.eye(n)
        self.Ad = I + A * self.dt
        self.Bd = B * self.dt
        
        # Solve Riccati recursion backward
        N = int(self.T / self.dt)
        self.K_gains = []
        P = self.QN.copy()
        
        for i in range(N):
            K = np.linalg.solve(self.R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)
            self.K_gains.insert(0, K)
            P = self.Q + self.Ad.T @ P @ (self.Ad - self.Bd @ K)
        
        # Input: state (14D)
        self.DeclareVectorInputPort("state", 14)
        
        # Output: control (2D)
        self.DeclareVectorOutputPort("u", 2, self.calc_control)
    
    def calc_control(self, context, output):
        x = self.get_input_port().Eval(context)
        t = context.get_time()
        
        # Get time-varying gain
        idx = int(t / self.dt)
        idx = min(idx, len(self.K_gains) - 1)
        K = self.K_gains[idx]
        
        # Compute control
        u = -K @ (x - self.x_goal)
        
        # Apply limits if specified
        if self.u_limits is not None:
            u = np.clip(u, self.u_limits[0], self.u_limits[1])
        
        output.SetFromVector(u)




# ============================================================================
# FINITE-HORIZON LQR CONTROLLER FOR COMPLETE WELDED SYSTEM
# ============================================================================

class FiniteHorizonLQRForCompleteSystem(LeafSystem):
    """
    Finite-horizon LQR for the welded arm+pendulum system (dimension-agnostic).
    Uses the same backward Riccati recursion as FiniteHorizonLQRController2D
    but works for any (A, B) pair — n=8 for the welded system.

    State (8D welded): [q_arm(2), q_pend(2), v_arm(2), v_pend(2)]
    Control (2D): u -> muscle neural command
    """

    def __init__(self, A, B, Q, R, horizon, timestep=0.01,
                 x_goal=None, u_max=50.0, QN=None):
        LeafSystem.__init__(self)
        n  = A.shape[0]
        nu = B.shape[1]
        assert Q.shape == (n, n), f"Q shape {Q.shape} != ({n},{n})"
        assert R.shape == (nu, nu), f"R shape {R.shape} != ({nu},{nu})"
        QN = QN if QN is not None else 2.0 * Q

        # Forward-Euler discretisation
        Ad = np.eye(n) + A * timestep
        Bd = B * timestep

        # Backward Riccati recursion
        N = max(1, int(round(horizon / timestep)))
        self.K_gains = []
        P = QN.copy()
        for _ in range(N):
            K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
            self.K_gains.insert(0, K)
            P = Q + Ad.T @ P @ (Ad - Bd @ K)

        self.x_goal  = x_goal if x_goal is not None else np.zeros(n)
        self.u_max   = u_max
        self.dt      = timestep
        self.n_gains = len(self.K_gains)

        print(colored(
            f"\u2713 FiniteHorizonLQRForCompleteSystem: n={n}, nu={nu}, "
            f"N={N} steps, dt={timestep}s, T={horizon}s, u_max=\u00b1{u_max}",
            "green",
        ))

        self.DeclareVectorInputPort("state", n)
        self.DeclareVectorOutputPort(
            "u", nu, self._calc_u, {self.all_input_ports_ticket()}
        )

    def _calc_u(self, context, output):
        x   = self.get_input_port().Eval(context)
        t   = context.get_time()
        idx = min(int(t / self.dt), self.n_gains - 1)
        K   = self.K_gains[idx]
        u   = -K @ (x - self.x_goal)
        u   = np.clip(u, -self.u_max, self.u_max)
        output.SetFromVector(u)


class ComputedTorqueEEController(LeafSystem):
    """
    Computed torque controller for end-effector trajectory tracking.
    
    Inputs:
      0: desired_trajectory (4) = [x_d, y_d, ẋ_d, ẏ_d]
      1: manipulator_state (4) = [q1, q2, q̇1, q̇2] (from plant with natural URDF)
    Output:
      0: joint_torques (2) = [τ1, τ2] (natural order matches actuator order)
    """
    
    def __init__(self, manipulator, plant, Kp=200.0, Kd=30.0, tau_max=100.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max
        self.call_count = 0
        
        # Inputs
        self.DeclareVectorInputPort("desired_trajectory", 4)
        self.DeclareVectorInputPort("manipulator_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("joint_torques", 2, self.calc_torques)
    
    def calc_torques(self, context, output):
        # Get inputs
        traj = self.get_input_port(0).Eval(context)
        manip_state = self.get_input_port(1).Eval(context)
        
        x_d, y_d, x_dot_d, y_dot_d = traj
        
        x_ddot_d, y_ddot_d = 0.0, 0.0  # Zero acceleration
        # manip_state is in natural [q1, q2, q̇1, q̇2] order (from manipulator.get_state_from_plant)
        q1, q2, q1_dot, q2_dot = manip_state
        q_manip = np.array([q1, q2])
        q_dot_manip = np.array([q1_dot, q2_dot])
        
        # Create plant context
        plant_context = self.plant.CreateDefaultContext()
        
        # Set manipulator state using helper method that handles Drake ordering
        temp_state = np.array([q1, q2, q1_dot, q2_dot])
        self.manipulator.set_state_in_plant(self.plant, plant_context, temp_state)
        
        # Get EE frame and compute current position
        ee_frame = self.plant.GetFrameByName(self.manipulator.LINK2_NAME, self.manipulator.model_instance)
        world_frame = self.plant.world_frame()
        EE_OFFSET = self.manipulator.EE_OFFSET
        
        ee_pos = self.plant.CalcPointsPositions(
            plant_context, ee_frame, EE_OFFSET.reshape(3, 1), world_frame
        ).flatten()
        x_current, y_current = ee_pos[0], ee_pos[1]
        
        # Compute Jacobian
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            EE_OFFSET,
            world_frame,
            world_frame
        )
        
        # Extract manipulator velocity indices using joint names (Drake order: [JT1=q2, JT2=q1])
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        manip_velocity_indices = [jt1.velocity_start(), jt2.velocity_start()]
        
        J_xy = J_full[0:2, manip_velocity_indices]  # 2x2
        
        # Current EE velocity
        ee_vel = J_xy @ q_dot_manip
        
        # Task-space errors
        e_pos = np.array([x_d - x_current, y_d - y_current])
        e_vel = np.array([x_dot_d - ee_vel[0], y_dot_d - ee_vel[1]])
        
        # Desired task-space acceleration with PD feedback
        x_ddot_des_vec = np.array([x_ddot_d, y_ddot_d])
        x_ddot_control = x_ddot_des_vec + self.Kp * e_pos + self.Kd * e_vel
        
        # Map to joint space
        J_pinv = np.linalg.pinv(J_xy)
        q_ddot_desired = J_pinv @ x_ddot_control
        
        # Compute inverse dynamics
        vd = np.zeros(self.plant.num_velocities())
        for i, vel_idx in enumerate(manip_velocity_indices):
            vd[vel_idx] = q_ddot_desired[i]
        
        external_forces = MultibodyForces(self.plant)
        tau_all = self.plant.CalcInverseDynamics(
            plant_context,
            vd,
            external_forces
        ).flatten()
        
        tau = np.array([tau_all[idx] for idx in manip_velocity_indices])
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Debug output
        self.call_count += 1
        if self.call_count % 100 == 1:
            error_mm = np.linalg.norm(e_pos) * 1000
            sat = " SAT" if np.any(np.abs(tau) >= self.tau_max - 0.1) else ""
            print(f"[t={context.get_time():.2f}s] EE=[{x_current:+5.2f},{y_current:+5.2f}] "
                  f"Desired=[{x_d:+5.2f},{y_d:+5.2f}] Err={error_mm:4.0f}mm "
                  f"τ=[{tau[0]:+5.1f},{tau[1]:+5.1f}]{sat}")
        
        output.SetFromVector(tau)


class ComputedTorqueJointSpaceController(LeafSystem):
    """
    Joint-space computed torque controller.

    Implements: q̈_a* = q̈_a,ref + Kq*eq + Dq*eqdot
    then τ = InverseDynamics(q, v, [q̈_a*; 0])
    
    Inputs:
      0: desired_joint_state (6) = [q1_d, q2_d, q̇1_d, q̇2_d, q̈1_d, q̈2_d]
              q̈ sourced from ZFTJointReferenceIK port 2, or zeros for open-loop modes
      1: manipulator_state (4) = [q1, q2, q̇1, q̇2] (from plant with natural URDF)
    Output:
      0: joint_torques (2) = [τ1, τ2] (natural order matches actuator order)
    """
    
    def __init__(self, manipulator, plant, Kp=200.0, Kd=60.0, tau_max=100.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max
        self.call_count = 0
        
        # Inputs
        self.DeclareVectorInputPort("desired_joint_state", 6)  # [q1_d, q2_d, q̇1_d, q̇2_d, q̈1_d, q̈2_d]
        self.DeclareVectorInputPort("manipulator_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("joint_torques", 2, self.calc_torques)
    
    def calc_torques(self, context, output):
        # Get inputs
        desired = self.get_input_port(0).Eval(context)       # [q1_d, q2_d, q̇1_d, q̇2_d, q̈1_d, q̈2_d]
        manip_state = self.get_input_port(1).Eval(context)   # [q1, q2, q̇1, q̇2] natural order
        
        q1_d, q2_d, q1_dot_d, q2_dot_d = desired[0], desired[1], desired[2], desired[3]
        qddot_ref = desired[4:6]   # [q̈1_ref, q̈2_ref] feedforward
        q1, q2, q1_dot, q2_dot = manip_state
        
        # Joint space errors (in [q1, q2] order)
        e_q = np.array([q1_d - q1, q2_d - q2])
        e_q_dot = np.array([q1_dot_d - q1_dot, q2_dot_d - q2_dot])
        
        # Desired joint accelerations: q̈_a* = q̈_a,ref + Kq*eq + Dq*eqdot  (image eq. 5)
        # q̈_a,ref comes bundled in desired[4:6]; zero when sourced from ManipulatorIKDesiredAngles
        q_ddot_desired = qddot_ref + self.Kp * e_q + self.Kd * e_q_dot
        
        # Create plant context with current state
        plant_context = self.plant.CreateDefaultContext()
        
        # Set manipulator state using helper method that handles Drake ordering
        temp_state = np.array([q1, q2, q1_dot, q2_dot])
        self.manipulator.set_state_in_plant(self.plant, plant_context, temp_state)
        
        # Compute inverse dynamics with natural [q̈1, q̈2] order
        # Get velocity indices using joint names (Drake order: [JT1=q2, JT2=q1])
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        
        # Compute inverse dynamics
        vd = np.zeros(self.plant.num_velocities())
        # Map user-order accelerations [q̈1, q̈2] to Drake order [q̈2, q̈1]
        vd[jt1.velocity_start()] = q_ddot_desired[1]  # JT1 = q2
        vd[jt2.velocity_start()] = q_ddot_desired[0]  # JT2 = q1
        
        external_forces = MultibodyForces(self.plant)
        tau_all = self.plant.CalcInverseDynamics(
            plant_context,
            vd,
            external_forces
        ).flatten()
        
        # Extract torques in Drake order [τ2, τ1], then convert to user order [τ1, τ2]
        tau_drake = np.array(
            [tau_all[jt1.velocity_start()], 
             tau_all[jt2.velocity_start()]]
             )
        tau = np.array([tau_drake[1], tau_drake[0]])  # Convert to user order [τ1, τ2]
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Debug output
        self.call_count += 1
        if self.call_count % 100 == 1:
            error_deg = np.linalg.norm(e_q) * 180 / np.pi
            sat = " SAT" if np.any(np.abs(tau) >= self.tau_max - 0.1) else ""
            print(f"[JS-CT t={context.get_time():.2f}s] q_err={error_deg:4.1f}° "
                  f"τ=[{tau[0]:+5.1f},{tau[1]:+5.1f}]{sat}")
        
        output.SetFromVector(tau)


class ActuatorLimit2D(LeafSystem):
    """
    Saturates 2-joint torques to ±tau_max (element-wise).

    Input
    -----
      0: joint_torques (2) – [τ1, τ2] raw torques

    Output
    ------
      0: joint_torques_limited (2) – clipped to [-tau_max, tau_max]
    """

    def __init__(self, tau_max: float = 100.0, n_joints: int = 2):
        """
        Args:
            tau_max  : Symmetric torque limit [N·m] applied to every joint.
            n_joints : Number of joints (default 2).
        """
        LeafSystem.__init__(self)
        self.tau_max = float(tau_max)
        self.n_joints = n_joints

        self.DeclareVectorInputPort("joint_torques", n_joints)
        self.DeclareVectorOutputPort(
            "joint_torques_limited", n_joints, self._calc_output
        )

    def _calc_output(self, context, output):
        tau = self.get_input_port(0).Eval(context)
        output.SetFromVector(np.clip(tau, -self.tau_max, self.tau_max))


class ManipulatorIKDesiredAngles(LeafSystem):
    """
    Velocity-based manipulator controller with position feedback.
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]  - desired cart trajectory
      1: plant_state (n) = full plant state vector
    Output: desired_joint_state (6) = [q1_d, q2_d, q̇1_d, q̇2_d, q̈1_d, q̈2_d]
              q̈ = 0 (no feedforward; zero acceleration)
    """
    
    def __init__(self, manipulator, plant, dt=0.001, Kp=10.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.dt = dt
        self.Kp = Kp  # Position feedback gain
        
        # Extract link lengths from URDF
        self.L1, self.L2 = self._extract_link_lengths()
        
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        self.DeclareVectorOutputPort("desired_joint_state", 6, self.calc_desired_angles)
    
    def _extract_link_lengths(self):
        """
        Extract link lengths L1 and L2 from URDF geometry.
        
        L1: Distance from base (joint1) to joint2
        L2: Distance from joint2 to end-effector (EE_OFFSET magnitude)
        
        Returns:
            L1, L2: Link lengths in meters
        """
        # Create a temporary context to query geometry
        temp_context = self.plant.CreateDefaultContext()
        
        # Get joint frames
        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        
        # Get the frames for both joints
        j1_frame = j1.frame_on_child()  # Link1 frame
        j2_frame = j2.frame_on_parent()  # Link1 frame (parent of joint2)
        j2_child_frame = j2.frame_on_child()  # Link2 frame
        
        # Set joints to zero configuration (positions + velocities)
        self.manipulator.set_state_in_plant(self.plant, temp_context, np.array([0.0, 0.0, 0.0, 0.0]))
        
        # Get transform from joint1 frame to joint2 frame at zero configuration
        X_j1_j2 = self.plant.CalcRelativeTransform(temp_context, j1_frame, j2_child_frame)
        
        # L1 is the distance between the two joints (XY plane distance)
        L1 = np.linalg.norm(X_j1_j2.translation()[:2])
        
        # L2 is the EE offset from link2 (already computed in manipulator)
        L2 = np.linalg.norm(self.manipulator.EE_OFFSET[:2])
        
        return L1, L2
    
    def compute_jacobian_manual(self, q1, q2):
        """
        Manually compute 2×2 Jacobian for 2-link planar manipulator.
        
        Forward kinematics (2D):
            x = L1*cos(q1) + L2*cos(q1+q2)
            y = L1*sin(q1) + L2*sin(q1+q2)
        
        Jacobian J = [∂x/∂q1  ∂x/∂q2]
                     [∂y/∂q1  ∂y/∂q2]
        
        Args:
            q1: Joint 1 angle (radians)
            q2: Joint 2 angle (radians)
            
        Returns:
            J_xy: 2×2 Jacobian matrix mapping [q̇1, q̇2] → [ẋ, ẏ]
        """
        # Use link lengths extracted from URDF
        L1 = self.L1
        L2 = self.L2
        
        # Compute Jacobian elements
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)
        
        # J = [[-L1*sin(q1) - L2*sin(q1+q2),  -L2*sin(q1+q2)],
        #      [ L1*cos(q1) + L2*cos(q1+q2),   L2*cos(q1+q2)]]
        
        J_xy = np.array([
            [-L1*s1 - L2*s12,  -L2*s12],
            [ L1*c1 + L2*c12,   L2*c12]
        ])
        
        return J_xy
    
    def calc_desired_angles(self, context, output, 
                            jac_cal: Literal["drake", "manual"] = "drake"):
        
        # Get inputs
        cart_state = self.get_input_port(0).Eval(context)
        cart_pos_xy = cart_state[0:2]  # Desired EE position
        cart_vel_xy = cart_state[2:4]  # Desired EE velocity
        plant_state = self.get_input_port(1).Eval(context)
        
        # Setup plant context with current state
        plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(plant_context, plant_state)
        q_current = self.plant.GetPositions(plant_context)
        
        # Get manipulator joint indices
        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        vel_idx_j1 = j1.velocity_start()
        vel_idx_j2 = j2.velocity_start()
        pos_idxs = [j1.position_start(), j2.position_start()]
        q_current_manip = np.array([q_current[pos_idxs[0]], q_current[pos_idxs[1]]])
        
        # Get EE frame (cup_center already includes EE_OFFSET)
        ee_frame = self.manipulator.get_end_effector_frame(self.plant)
        p_BQ = np.zeros(3)  # Zero offset since cup_center frame already includes EE_OFFSET
        
        # Compute current EE position
        X_WB = self.plant.CalcRelativeTransform(plant_context, self.plant.world_frame(), ee_frame)
        ee_current_3d = X_WB.translation()
        ee_current_xy = ee_current_3d[0:2]
        
        # Position error: how far is EE from desired cart position?
        pos_error_xy = cart_pos_xy - ee_current_xy
        
        # Compute Jacobian: maps joint velocities to EE velocity
        Jv_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV, ee_frame, p_BQ,
            self.plant.world_frame(), self.plant.world_frame()
        )
        J_xy_drake = Jv_full[0:2, [vel_idx_j1, vel_idx_j2]]  # Extract 2×2 manipulator Jacobian
        
        if jac_cal == "drake":
            # Use Drake's Jacobian
            J_xy = J_xy_drake  # or J_xy_manual
        elif jac_cal == "manual":
            # Compute Jacobian manually for comparison
            J_xy = self.compute_jacobian_manual(q_current_manip[0], q_current_manip[1])
        else:
            raise ValueError(f"Invalid jac_cal method: {jac_cal}")
        
        # Desired EE velocity: feedforward + position feedback
        ee_vel_desired = cart_vel_xy + self.Kp * pos_error_xy
        
        # Map to joint velocities
        qdot_des = np.linalg.pinv(J_xy) @ ee_vel_desired
        
        # Integrate to get desired positions: q_des = q_current + qdot_des * dt
        q_des = q_current_manip + qdot_des * self.dt
        
        # Zero acceleration feedforward for this mode (q̈_ref = 0)
        output.SetFromVector(np.concatenate([q_des, qdot_des, np.zeros(2)]))


# ============================================================================
# ZFT → JOINT REFERENCE IK BLOCK
# ============================================================================

class ZFTJointReferenceIK(LeafSystem):
    """
    IK block: converts task-space ZFT reference (p_zft, ṗ_zft, p̈_zft) to
    joint-space reference (q_ref, q̇_ref, q̈_ref) for the manipulator arm.

    Live plant state (port 3) is ALWAYS used for:
      • IK warm-start seed  → avoids discontinuities
      • Bias acceleration   → J̇(q_curr, q̇_curr) q̇_curr physically accurate

    Two modes for computing q_ref, selected at construction via `ik_method`:

    "ik"  (default) — position-level IK:
      q_ref  = solve  h_a(q_ref) ≈ p_zft        (Drake IK, warm from q_current)
      q̇_ref  = J_a(q_ref)†  ṗ_zft
      q̈_ref  = J_a(q_ref)†  (p̈_zft − J̇(q_curr, q̇_curr) q̇_curr)

    "differential" — velocity integration (like ManipulatorIKDesiredAngles):
      q̇_ref  = J_a(q_curr)†  (ṗ_zft + Kp*(p_zft − p_curr))   [FF + FB]
      q_ref  = q_curr + q̇_ref * dt                             [integrate]
      q̈_ref  = J_a(q_curr)†  (p̈_zft − J̇(q_curr, q̇_curr) q̇_curr)

    Inputs
    ------
      0: p_zft        (2) – [x_ref, y_ref]
      1: pdot_zft     (2) – [ẋ_ref, ẏ_ref]
      2: pddot_zft    (2) – [ẍ_ref, ÿ_ref]
      3: plant_state  (n) – full plant [q; v]  (always required)

    Outputs
    -------
      0: q_ref     (2) – [q1_ref, q2_ref]   (user order)
      1: qdot_ref  (2) – [q̇1_ref, q̇2_ref]  (user order)
      2: qddot_ref (2) – [q̈1_ref, q̈2_ref]  (user order)
    """

    IK_METHODS = ("ik", "differential")

    def __init__(self, config: "ZFTJointReferenceIKConfig"):
        LeafSystem.__init__(self)

        ik_method = config.ik_method
        if ik_method not in self.IK_METHODS:
            raise ValueError(f"ik_method='{ik_method}' invalid. Choose: {self.IK_METHODS}")

        self.manipulator = config.manipulator
        self.plant       = config.plant
        self.ik_method   = ik_method
        self.pos_tol     = config.pos_tol
        self.dt          = config.dt      # used only in "differential" mode
        self.Kp          = config.Kp     # position feedback gain for "differential" mode

        # Inputs
        self.DeclareVectorInputPort("p_zft",      2)
        self.DeclareVectorInputPort("pdot_zft",   2)
        self.DeclareVectorInputPort("pddot_zft",  2)
        self.DeclareVectorInputPort("plant_state", config.plant.num_multibody_states())

        # Outputs
        self.DeclareVectorOutputPort("q_ref",     2, self._calc_q_ref,
                                     {self.all_input_ports_ticket()})
        self.DeclareVectorOutputPort("qdot_ref",  2, self._calc_qdot_ref,
                                     {self.all_input_ports_ticket()})
        self.DeclareVectorOutputPort("qddot_ref", 2, self._calc_qddot_ref,
                                     {self.all_input_ports_ticket()})

    # ------------------------------------------------------------------
    # Shared: extract live arm state from plant_state input (port 3)
    # ------------------------------------------------------------------

    def _get_current_arm_state(self, context):
        """
        Returns
        -------
        q_current   : (2,) [q1, q2] user order
        qdot_current: (2,) [q̇1, q̇2] user order
        ctx_live    : plant context at live state (reuse for kinematics)
        p_current   : (2,) current EE position [x, y] in world frame
        """
        plant_state = self.get_input_port(3).Eval(context)
        ctx_live = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(ctx_live, plant_state)

        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)

        q_full = self.plant.GetPositions(ctx_live)
        v_full = self.plant.GetVelocities(ctx_live)

        q_current    = np.array([q_full[j1.position_start()],
                                  q_full[j2.position_start()]])
        qdot_current = np.array([v_full[j1.velocity_start()],
                                  v_full[j2.velocity_start()]])

        # Current EE position (for differential mode position feedback)
        ee_frame  = self.manipulator.get_end_effector_frame(self.plant)
        X_WE      = self.plant.CalcRelativeTransform(
                        ctx_live, self.plant.world_frame(), ee_frame)
        p_current = X_WE.translation()[0:2]

        return q_current, qdot_current, ctx_live, p_current

    # ------------------------------------------------------------------
    # Shared kinematics helpers
    # ------------------------------------------------------------------

    def _make_arm_context(self, q_user: np.ndarray,
                          qdot_user: np.ndarray = None):
        """Plant context at given arm state (user order)."""
        ctx   = self.plant.CreateDefaultContext()
        state = np.concatenate([q_user,
                                 qdot_user if qdot_user is not None
                                 else np.zeros(2)])
        self.manipulator.set_state_in_plant(self.plant, ctx, state)
        return ctx

    def _get_jacobian(self, plant_context) -> np.ndarray:
        """2×2 arm Jacobian J_a([ẋ,ẏ] / [q̇1,q̇2])."""
        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        vel_indices = [j1.velocity_start(), j2.velocity_start()]
        ee_frame = self.manipulator.get_end_effector_frame(self.plant)
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV,
            ee_frame, np.zeros(3),
            self.plant.world_frame(), self.plant.world_frame(),
        )
        return J_full[0:2, vel_indices]   # 2×2

    def _get_bias_accel(self, plant_context) -> np.ndarray:
        """Bias acceleration J̇_a(q,q̇) q̇ from Drake (evaluated at given context).
        Always returns a 1-D array of shape (2,)."""
        ee_frame  = self.manipulator.get_end_effector_frame(self.plant)
        a_bias_3d = self.plant.CalcBiasTranslationalAcceleration(
            plant_context, JacobianWrtVariable.kV,
            ee_frame, np.zeros(3),
            self.plant.world_frame(), self.plant.world_frame(),
        )
        return np.asarray(a_bias_3d).flatten()[0:2]

    # ------------------------------------------------------------------
    # Mode-specific computation
    # ------------------------------------------------------------------

    def _compute_ik_mode(self, p_zft, pdot_zft, pddot_zft,
                         q_current, qdot_current, ctx_live):
        """
        Position-level IK mode:
          q_ref  = IK(p_zft),  warm-started from q_current
          q̇_ref  = J(q_ref)†  ṗ_zft
          q̈_ref  = J(q_ref)†  (p̈_zft − J̇(q_curr,q̇_curr) q̇_curr)
        """
        # Step 1 — IK warm-started from live q
        q_ref, _ = self.manipulator.compute_ik_analytical(
            self.plant,
            target_xy=p_zft,
            q_seed=q_current,
            pos_tol=self.pos_tol,
        )

        # Step 2 — q̇_ref at the IK solution
        ctx_ref  = self._make_arm_context(q_ref)
        J_arm_ref    = self._get_jacobian(ctx_ref)
        qdot_ref = np.linalg.pinv(J_arm_ref) @ pdot_zft

        # Step 3 — q̈_ref with bias from LIVE (q_curr, q̇_curr)
        a_bias    = self._get_bias_accel(ctx_live)
        qddot_ref = (np.linalg.pinv(J_arm_ref) @ (pddot_zft - a_bias)).flatten()  # J_arm_refˆ\psudo(p̈_zft - J̇(q_curr,q̇_curr) q̇_curr)

        return np.asarray(q_ref).flatten(), np.asarray(qdot_ref).flatten(), qddot_ref

    def _compute_differential_mode(self, p_zft, pdot_zft, pddot_zft,
                                   q_current, qdot_current, ctx_live, p_current):
        """
        Differential (velocity-integration) mode:
          J at current q (like ManipulatorIKDesiredAngles)
          q̇_ref  = J(q_curr)†  (ṗ_zft + Kp*(p_zft − p_curr))
          q_ref  = q_curr + q̇_ref * dt
          q̈_ref  = J(q_curr)†  (p̈_zft − J̇(q_curr,q̇_curr) q̇_curr)
        """
        J_curr = self._get_jacobian(ctx_live)
        J_pinv = np.linalg.pinv(J_curr)

        # Step 1 — q̇_ref: feedforward ṗ_zft + position feedback
        pos_error = p_zft - p_current
        qdot_ref  = J_pinv @ (pdot_zft + self.Kp * pos_error)

        # Step 2 — q_ref: integrate from current q
        q_ref = q_current + qdot_ref * self.dt

        # Step 3 — q̈_ref with bias from LIVE state
        a_bias    = self._get_bias_accel(ctx_live)
        qddot_ref = (J_pinv @ (pddot_zft - a_bias)).flatten()

        return np.asarray(q_ref).flatten(), np.asarray(qdot_ref).flatten(), qddot_ref

    # ------------------------------------------------------------------
    # Core dispatcher — shared by all output callbacks
    # ------------------------------------------------------------------

    def _compute(self, context):
        p_zft     = self.get_input_port(0).Eval(context)
        pdot_zft  = self.get_input_port(1).Eval(context)
        pddot_zft = self.get_input_port(2).Eval(context)

        # Live arm state (always used)
        q_curr, qdot_curr, ctx_live, p_curr = self._get_current_arm_state(context)

        if self.ik_method == "ik":
            return self._compute_ik_mode(
                p_zft, pdot_zft, pddot_zft, q_curr, qdot_curr, ctx_live)
        elif self.ik_method == "differential":
            return self._compute_differential_mode(
                p_zft, pdot_zft, pddot_zft, q_curr, qdot_curr, ctx_live, p_curr)
        else:
            raise ValueError(f"Invalid ik_method: {self.ik_method}")

    # ------------------------------------------------------------------
    # Output port callbacks
    # ------------------------------------------------------------------

    def _calc_q_ref(self, context, output):
        q_ref, _, _ = self._compute(context)
        output.SetFromVector(q_ref)

    def _calc_qdot_ref(self, context, output):
        _, qdot_ref, _ = self._compute(context)
        output.SetFromVector(qdot_ref)

    def _calc_qddot_ref(self, context, output):
        _, _, qddot_ref = self._compute(context)
        output.SetFromVector(qddot_ref)



# Add system to compute end-effector position and velocity
class ManipulatorEEStateComputer(LeafSystem):
    """Computes manipulator end-effector position and velocity from joint state."""
    def __init__(self, plant, manipulator):
        LeafSystem.__init__(self)
        self.plant = plant
        self.manipulator = manipulator
        
        # Input: manipulator state in Drake order from plant.get_state_output_port(model_instance)
        # For cup_manipulator with joints [link2_link1, link1_base], Drake order is [q2, q1, q̇2, q̇1]
        self.DeclareVectorInputPort("manip_state", 4)
        
        # Output: EE state [x, y, ẋ, ẏ]
        self.DeclareVectorOutputPort(
            "ee_state",
            4,
            self.CalcEEState
        )
    
    def CalcEEState(self, context, output):
        """Calculate EE position and velocity from joint state."""
        # Get manipulator state in DRAKE order [q2, q1, q̇2, q̇1] from plant output
        manip_state_drake = self.get_input_port(0).Eval(context)
        
        # Convert from Drake order to user order [q1, q2, q̇1, q̇2] for set_state_in_plant
        # Drake order for cup_manipulator: [q2, q1, q̇2, q̇1] (link2_link1, link1_base)
        # User order: [q1, q2, q̇1, q̇2] (link1_base, link2_link1)
        manip_state_user = np.array([manip_state_drake[1], manip_state_drake[0], 
                                      manip_state_drake[3], manip_state_drake[2]])
        
        # Create fresh context for this computation
        temp_context = self.plant.CreateDefaultContext()
        
        # Set state in temp context (expects user order)
        self.manipulator.set_state_in_plant(self.plant, temp_context, manip_state_user)
        
        # Calculate EE position using custom EE frame with offset
        ee_pos = self.manipulator.get_end_effector_position(self.plant, temp_context)
        
        # Calculate EE velocity using Jacobian
        ee_frame = self.plant.GetFrameByName(self.manipulator.LINK2_NAME, self.manipulator.model_instance)
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            temp_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            self.manipulator.EE_OFFSET,
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        
        # Extract manipulator velocity indices
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        manip_velocity_indices = [jt1.velocity_start(), jt2.velocity_start()]
        
        # Compute EE velocity: v_ee = J * q̇
        J_xy = J_full[0:2, manip_velocity_indices]
        ee_vel = J_xy @ manip_state_user[2:4]
        
        # Output [x, y, ẋ, ẏ]
        output.SetFromVector(np.array([ee_pos[0], ee_pos[1], ee_vel[0], ee_vel[1]]))


class EndEffectorKinematics2D(LeafSystem):
    """
    Computes end-effector (x, y) position and velocity from full plant state.

    Used in the welded cart-to-manipulator-EE architecture where the plant
    contains both manipulator arm DOFs and pendulum DOFs.  The system takes
    the full plant *position* and *velocity* vectors (already split by
    plant_state_demux) and extracts the arm joints to evaluate forward
    kinematics and the translational Jacobian.

    Inputs
    ------
      0: q     (nq_total) – full plant joint positions
      1: v     (nv_total) – full plant joint velocities

    Outputs
    -------
      0: p     (2) – EE position  [x_ee, y_ee]  in world frame
      1: pdot  (2) – EE velocity  [vx_ee, vy_ee] in world frame
                     computed as  J(q_arm) @ v_arm
    """

    def __init__(self, config: "EndEffectorKinematics2DConfig"):
        LeafSystem.__init__(self)
        self._config = config
        self._plant = config.plant
        self._manipulator = config.manipulator

        # Internal plant context – reused across callbacks (single-threaded).
        self._ctx = self._plant.CreateDefaultContext()

        # Pre-compute velocity-vector (and position-vector) indices for the
        # two arm joints using Drake's velocity_start() / position_start().
        jt1 = self._manipulator.get_joint_by_name(
            self._plant, self._manipulator.JT1_NAME
        )
        jt2 = self._manipulator.get_joint_by_name(
            self._plant, self._manipulator.JT2_NAME
        )
        # Columns in the full-plant Jacobian that correspond to the arm DOFs.
        self._arm_v_indices = [jt1.velocity_start(), jt2.velocity_start()]

        # Declare ports
        self.DeclareVectorInputPort("q", config.nq_total)
        self.DeclareVectorInputPort("v", config.nv_total)

        self.DeclareVectorOutputPort(
            "p",
            BasicVector(2),
            self._calc_position,
            {self.all_input_ports_ticket()},
        )
        self.DeclareVectorOutputPort(
            "pdot",
            BasicVector(2),
            self._calc_velocity,
            {self.all_input_ports_ticket()},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_arm_state(self, q_full, v_full):
        """Write the full plant state into the cached internal context."""
        self._plant.SetPositions(self._ctx, q_full)
        self._plant.SetVelocities(self._ctx, v_full)

    # ------------------------------------------------------------------
    # Output callbacks
    # ------------------------------------------------------------------

    def _calc_position(self, context, output):
        """p = FK(q_arm)  →  [x_ee, y_ee] in world frame."""
        q_full = self.get_input_port(0).Eval(context)
        v_full = self.get_input_port(1).Eval(context)
        self._update_arm_state(q_full, v_full)

        # Use manipulator's helper which accounts for EE_OFFSET on link2.
        ee_pos = self._manipulator.get_end_effector_position(
            self._plant, self._ctx
        )
        output.SetFromVector([ee_pos[0], ee_pos[1]])

    def _calc_velocity(self, context, output):
        """pdot = J(q_arm) @ v_arm  →  [vx_ee, vy_ee] in world frame."""
        q_full = self.get_input_port(0).Eval(context)
        v_full = self.get_input_port(1).Eval(context)
        self._update_arm_state(q_full, v_full)

        ee_frame = self._plant.GetFrameByName(
            self._manipulator.LINK2_NAME, self._manipulator.model_instance
        )
        # Full translational Jacobian: shape (3, nv_total)
        J_full = self._plant.CalcJacobianTranslationalVelocity(
            self._ctx,
            JacobianWrtVariable.kQDot,
            ee_frame,
            self._manipulator.EE_OFFSET,
            self._plant.world_frame(),
            self._plant.world_frame(),
        )
        # Select X-Y rows and arm velocity columns → (2, 2) sub-Jacobian.
        J_arm = J_full[0:2, self._arm_v_indices]       # (2 × 2)
        v_arm = v_full[self._arm_v_indices]             # (2,)
        ee_vel = J_arm @ v_arm
        output.SetFromVector([ee_vel[0], ee_vel[1]])




# SYSTEM BUILDER CLASS
# ============================================================================

class SystemBuilder:
    """
    Builds a Drake MultibodyPlant with two model instances:
    1. Manipulator (2-DOF cup manipulator)
    2. Cart-Pendulum (4-DOF system)
    
    This class encapsulates the plant setup logic to avoid code duplication
    across different simulation modes.
    
    Attributes:
        builder: DiagramBuilder instance (created during build())
        plant: MultibodyPlant instance (created during build())
        scene_graph: SceneGraph instance (created during build())
    """
    
    def __init__(self, physics_config, manipulator_urdf_path, 
                 manipulator_joint_angles=None, manipulator_damping=(0.1, 0.1)):
        """
        Initialize the system builder with configurations.
        
        Args:
            physics_config: PhysicsConfig for cart-pendulum
            manipulator_urdf_path: Path to manipulator URDF file
            manipulator_joint_angles: Dict of joint names to angles (radians)
            manipulator_damping: Tuple of (q1_damping, q2_damping)
        """
        self.physics_config = physics_config
        self.manipulator_urdf_path = manipulator_urdf_path
        self.manipulator_joint_angles = manipulator_joint_angles or {
            'link1_base': np.deg2rad(0.0),
            'link2_link1': np.deg2rad(20.0),
        }
        self.manipulator_damping = manipulator_damping
        
        # These will be set during build()
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.weld_cart_to_ee = WELD_CART_TO_MANIP_EE  # Set to True to weld cart to manipulator EE, False for independent actuation
        

    def add_manipulator(self, plant):
        """
        Add manipulator to the plant as a model instance.
        
        Args:
            plant: MultibodyPlant to add manipulator to
            
        Returns:
            CupManipulator instance
        """
        manipulator_config = create_cup_manipulator_config(
            urdf_path=self.manipulator_urdf_path,
            joint_angles=self.manipulator_joint_angles,
            damping=self.manipulator_damping,
        )
        
        # Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)
        
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        manipulator.weld_base_to_world(plant, orientation=np.array([0.0, 0.0, 0.0]))
        
        # Add actuators and end-effector frame BEFORE finalization
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        
        print(colored(f"✓ End-effector frame '{manipulator.EE_FRAME_NAME}' added to manipulator", "green"))
        print(colored(f"  - EE_OFFSET (relative to link2): {manipulator.EE_OFFSET}", "cyan"))
        print(colored(f"✓ Manipulator loaded (ModelInstance: {manipulator.model_instance})", "green"))
        print(colored(f"  - State dimension: 4 (2 positions + 2 velocities)", "cyan"))
        print(colored(f"  - Joints: link1_base, link2_link1", "cyan"))
        
        return manipulator
    
    def add_cart_pendulum_components(self, plant, manipulator=None):
        """
        Add cart-pendulum to the plant as a model instance.
        
        Args:
            plant: MultibodyPlant to add cart-pendulum to
            manipulator: CupManipulator instance (optional, required if weld_cart_to_ee=True)
        Returns:
            Tuple of (cart_pendulum, cart_model)
        """
        # Determine z-offset for cart-pendulum based on URDF joint origin
        z_offset_from_urdf = CupManipulator.EE_XYZ_BASE[2]
        print(colored(f"📍 Using z-offset from URDF: {z_offset_from_urdf:.5f} m", "cyan"))
        
        cart_model = plant.AddModelInstance("cart_pendulum")
        
        # ====================================================================
        # DECISION: WELDED TO EE vs INDEPENDENT ACTUATION
        # ====================================================================
        if self.weld_cart_to_ee:
            if manipulator is None:
                raise ValueError("manipulator must be provided when weld_cart_to_ee=True")
            
            print(colored(f"\n🔗 Mode: Cart WELDED to manipulator EE (dependent control)", "yellow"))
            print(colored(f"  - Kinematic chain: World → Manipulator → EE → Cart → Pendulum", "cyan"))
            print(colored(f"  - Cart follows EE motion (NO independent cart actuators)", "cyan"))
            
            # Create cart-pendulum using CartPendulum2DExtended.build_plant_welded()
            cart_pendulum = CartPendulum2DExtended(self.physics_config, z_offset=z_offset_from_urdf)
            cart_body = cart_pendulum.build_plant_welded(plant, cart_model, register_visuals=True)
            
            # Get the manipulator's EE frame
            ee_frame = manipulator.get_end_effector_frame(plant)
            
            # Weld cart body to EE frame (zero offset = cart center aligns with EE)
            plant.WeldFrames(
                frame_on_parent_F=ee_frame,
                frame_on_child_M=cart_body.body_frame(),
                X_FM=RigidTransform()
            )
            
            print(colored(f"✓ Cart welded to EE frame '{manipulator.EE_FRAME_NAME}'", "green"))
            print(colored(f"  - Cart DOF: 0 (welded, follows EE)", "cyan"))
            print(colored(f"  - Pendulum DOF: 2 (pitch + roll gimbal)", "cyan"))
            print(colored(f"  - NO cart actuators (controlled via manipulator)", "yellow"))
            
        else:
            print(colored(f"\n⚙️ Mode: Cart INDEPENDENT actuation (uncoupled control)", "yellow"))
            print(colored(f"  - Kinematic chain: World → Cart (via sliders) → Pendulum", "cyan"))
            print(colored(f"  - Manipulator: World → Manip → EE (separate tree)", "cyan"))
            print(colored(f"  - Cart has independent actuators (LQR controls both systems)", "cyan"))
            
            # Build cart-pendulum with normal world connection via prismatic joints
            cart_pendulum = CartPendulum2DExtended(self.physics_config, z_offset=z_offset_from_urdf)
            cart_pendulum.build_plant(plant, cart_model)
            
            print(colored(f"✓ Cart-pendulum has independent actuation", "green"))
        
        print(colored(f"✓ Cart-Pendulum created (ModelInstance: {cart_model})", "green"))
        print(colored(f"  - Z-plane height: {z_offset_from_urdf:.5f} m", "cyan"))
        
        return cart_pendulum, cart_model
    
    def finalize_and_print_info(self, plant, manipulator):
        """
        Finalize the plant and print configuration information.
        
        Args:
            plant: MultibodyPlant to finalize
            manipulator: CupManipulator instance for info printing
        """
        plant.Finalize()
        
        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                     f"{plant.num_velocities()} total velocities", "green"))
        
        # Extract and display manipulator configuration
        config_q1 = self.manipulator_joint_angles['link1_base']
        config_q2 = self.manipulator_joint_angles['link2_link1']
        
        # Calculate EE position at config angles for display
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, {
            "link1_base": config_q1,
            "link2_link1": config_q2,
        })
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
        print(colored(f"  - EE position in world frame (at config q1={np.rad2deg(config_q1):.1f}°, "
                     f"q2={np.rad2deg(config_q2):.1f}°): {ee_world_pos}", "cyan"))
    
    def build(self, time_step=0.001, meshcat=None):
        """
        Build the complete system with manipulator and cart-pendulum.
        Creates the DiagramBuilder and stores it as a class attribute.
        Adds plant and scene_graph to the builder and connects them.
        
        Args:
            time_step: Simulation time step in seconds
            meshcat: Optional Meshcat instance for printing URL
            
        Returns:
            Tuple of (builder, plant, scene_graph, manipulator, cart_pendulum, cart_model)
        """
        # Create builder, plant, and scene graph (stored as attributes)
        self.builder = DiagramBuilder()
        self.plant = MultibodyPlant(time_step=time_step)
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
    
        
        # Add manipulator
        manipulator = self.add_manipulator(self.plant)
        
        # Add cart-pendulum
        if self.weld_cart_to_ee:
            print(colored(f"\n⚠️ WARNING: Cart-pendulum will be WELDED to manipulator EE. "
                         f"Cart actuation inputs will be IGNORED.", "red"))
            cart_pendulum, cart_model = self.add_cart_pendulum_components(self.plant, manipulator=manipulator)
        else:            
            print(colored(f"\n⚠️ WARNING: Cart-pendulum will have INDEPENDENT ACTUATION. "
                         f"Ensure control system accounts for this.", "red"))
            cart_pendulum, cart_model = self.add_cart_pendulum_components(self.plant)
        
        # Finalize and print info
        self.finalize_and_print_info(self.plant, manipulator)
        
        # Add plant to builder and connect to scene_graph
        self.builder.AddSystem(self.plant)
        self.builder.Connect(
            self.plant.get_geometry_pose_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id())
        )
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port()
        )
        
        # Print meshcat URL if provided
        if meshcat is not None:
            print(colored(f"  - Meshcat will be available at: {meshcat.web_url()}", "cyan"))
        
        return self.builder, self.plant, self.scene_graph, manipulator, cart_pendulum, cart_model


# ============================================================================
# CONTROL SYSTEM BUILDER (STRATEGY PATTERN)
# ============================================================================

class ControlSystemBuilder(ABC):
    """
    Abstract base class for building different control strategies.
    
    Strategy Pattern: Encapsulates control system construction algorithms.
    Each concrete builder creates a specific control architecture:
    - LQR + Muscle Dynamics + ZFT (OFC)
    - PD Control
    - Model Predictive Control
    - etc.
    
    Concrete classes have full freedom to structure control systems.
    Only logging methods must be implemented.
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator):
        """
        Initialize builder with Drake components.
        
        Args:
            builder: DiagramBuilder to add systems to
            plant: MultibodyPlant with cart-pendulum and manipulator
            cart_model: Cart-pendulum ModelInstance
            manipulator: CupManipulator instance
        """
        self.builder = builder
        self.plant = plant
        self.cart_model = cart_model
        self.manipulator = manipulator
        
        # Will store created systems for connection and logging
        self.systems = {}
        self.loggers = {}
    
    @abstractmethod
    def add_loggers(self) -> Dict[str, VectorLogSink]:
        """
        Build data loggers for all control signals.
        
        Must be implemented by concrete classes.
        
        Returns:
            Dictionary of logger names to VectorLogSink instances
        """
        pass
    
    @abstractmethod
    def connect_loggers(self):
        """
        Connect loggers to their signal sources.
        
        Must be implemented by concrete classes.
        """
        pass
    
    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Concrete classes should override this to define their own
        control system construction and connection logic.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build and connect loggers (required by abstract methods)
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers


# ============================================================================
# LQR + OFC CONTROL BUILDER (CONCRETE STRATEGY)
# ============================================================================

class LQRWithOFCOnlyCartPendulumBuilder(ControlSystemBuilder):
    """
    Builds LQR control with Optimal Feedback Control (OFC) architecture:
    
    Cart-Pendulum Control:
    - Muscle Dynamics (low-pass filter on neural commands)
    - ZFT Reference Mass (virtual mass for smooth trajectories)
    - Impedance Force (spring-damper connection to reference)
    - Finite-Horizon LQR (optimal time-varying feedback)
    
    Manipulator Control:
    - IK Solver (converts cart trajectory to joint angles)
    - Computed Torque Controller (joint-space tracking)
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator, config):
        super().__init__(builder, plant, cart_model, manipulator)
        self.config = config
        
        # Pre-compute linearization
        self.A, self.B = build_linearized_system_2d(
            config.physics_config,
            config.impedance_config,
            config.zft_config,
            config.muscle_config
        )
        print(colored(f"✓ Linearized system: A ({self.A.shape}), B ({self.B.shape})", "green"))
    
    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build control systems
        cp_systems = self.add_cart_system_blocks()
        manip_systems = self.add_manipulator_system_blocks()
        
        self.systems.update(cp_systems)
        self.systems.update(manip_systems)
        
        # Connect control loops
        self.connect_cart_pendulum_control()
        self.connect_manipulator_control()
        
        # Build and connect loggers
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers
    
    def add_cart_system_blocks(self) -> Dict[str, LeafSystem]:
        """Build LQR + OFC blocks for cart-pendulum."""
        systems = {}
        
        # Muscle dynamics (2D low-pass filter)
        systems['muscle'] = self.builder.AddSystem(
            MuscleDynamics2D(self.config.muscle_config)
        )
        systems['muscle'].set_name("muscle_dynamics")
        
        # ZFT reference mass (virtual mass dynamics)
        systems['zft'] = self.builder.AddSystem(
            ZFTReferenceMass2D(self.config.zft_config)
        )
        systems['zft'].set_name("zft_reference")
        
        # Impedance force (spring-damper connection)
        systems['impedance'] = self.builder.AddSystem(
            ImpedanceForce2D(self.config.impedance_config)
        )
        systems['impedance'].set_name("impedance_force")
        
        # LQR controller (finite-horizon optimal control)
        systems['lqr'] = self.builder.AddSystem(
            FiniteHorizonLQRController2D(self.A, self.B, LQR_CONFIG)
        )
        systems['lqr'].set_name("lqr_controller")
        
        # ZeroOrderHold (breaks algebraic loop)
        systems['state_hold'] = self.builder.AddSystem(ZeroOrderHold(0.01, 14))
        systems['state_hold'].set_name("state_hold")
        
        # State extraction mux/demux
        systems['cart_state_demux'] = self.builder.AddSystem(Demultiplexer([2, 2, 2, 2]))
        systems['full_state_mux'] = self.builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 4]))
        systems['cart_state_mux'] = self.builder.AddSystem(Multiplexer([2, 2]))
        
        return systems
    
    def add_manipulator_system_blocks (self) -> Dict[str, LeafSystem]:
        """Build IK + computed torque control for manipulator."""
        systems = {}
        
        # IK solver (cart trajectory → desired joint angles)
        systems['manip_ik'] = self.builder.AddSystem(
            ManipulatorIKDesiredAngles(self.manipulator, self.plant)
        )
        systems['manip_ik'].set_name("manipulator_ik_solver")
        
        # Joint-space computed torque controller
        systems['manip_controller'] = self.builder.AddSystem(
            ComputedTorqueJointSpaceController(
                self.manipulator, self.plant, Kp=200.0, Kd=60.0, tau_max=100.0
            )
        )
        systems['manip_controller'].set_name("manipulator_js_controller")
        
        return systems

    def add_loggers(self) -> Dict[str, VectorLogSink]:
        """Build loggers for LQR + OFC signals."""
        loggers = {}
        
        # Base loggers
        loggers['state'] = self.builder.AddSystem(VectorLogSink(8))
        loggers['state'].set_name("state_logger")
        
        loggers['manip_state'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['manip_state'].set_name("manip_state_logger")
        
        ee_computer = self.builder.AddSystem(
            ManipulatorEEStateComputer(self.plant, self.manipulator)
        )
        ee_computer.set_name("ee_state_computer")
        loggers['ee_state'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['ee_state'].set_name("ee_state_logger")
        loggers['ee_computer'] = ee_computer
        
        # OFC-specific loggers
        loggers['ref'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['ref'].set_name("ref_logger")
        
        loggers['force'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['force'].set_name("force_logger")
        
        loggers['impedance'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['impedance'].set_name("impedance_logger")
        
        loggers['cart_traj'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['cart_traj'].set_name("cart_traj_logger")
        
        loggers['manip_desired'] = self.builder.AddSystem(VectorLogSink(6))
        loggers['manip_desired'].set_name("manip_desired_state_logger")
        
        loggers['manip_torque'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['manip_torque'].set_name("manip_js_torque_logger")
        
        return loggers
    
    def connect_cart_pendulum_control(self):
        """Connect LQR + OFC control loop for cart-pendulum."""
        # Extract systems for convenience
        muscle = self.systems['muscle']
        zft = self.systems['zft']
        impedance = self.systems['impedance']
        lqr = self.systems['lqr']
        state_hold = self.systems['state_hold']
        cart_state_demux = self.systems['cart_state_demux']
        full_state_mux = self.systems['full_state_mux']
        cart_state_mux = self.systems['cart_state_mux']
        
        # ====================================================================
        # STATE EXTRACTION: Plant → Demux → Mux
        # ====================================================================
        # Plant state: q = [x, y, α, β]ᵀ ∈ ℝ⁴, q̇ = [ẋ, ẏ, α̇, β̇]ᵀ ∈ ℝ⁴
        # Full state: s_plant = [q; q̇] ∈ ℝ⁸
        self.builder.Connect(
            self.plant.get_state_output_port(self.cart_model),  # s_plant ∈ ℝ⁸
            cart_state_demux.get_input_port()  # Split into 4 blocks of size 2
        )
        # Demux outputs: [0]→[x,y], [1]→[α,β], [2]→[ẋ,ẏ], [3]→[α̇,β̇]
        
        # Build cart state: s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
        self.builder.Connect(
            cart_state_demux.get_output_port(0),  # [x, y]ᵀ ∈ ℝ²
            cart_state_mux.get_input_port(0)
        )
        self.builder.Connect(
            cart_state_demux.get_output_port(2),  # [ẋ, ẏ]ᵀ ∈ ℝ²
            cart_state_mux.get_input_port(1)
        )
        # Output: s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
        
        # ====================================================================
        # ZFT REFERENCE MASS: Dynamics for smooth reference trajectory
        # ====================================================================
        # ZFT dynamics: ṡ_ref = f_zft(s_cart, F_muscle)
        # State: s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
        self.builder.Connect(
            cart_state_mux.get_output_port(),  # s_cart ∈ ℝ⁴
            zft.get_input_port(0)
        )
        self.builder.Connect(
            muscle.get_output_port(),  # F_muscle = [F_x, F_y]ᵀ ∈ ℝ²
            zft.get_input_port(1)
        )
        # Output: s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
        
        # ====================================================================
        # IMPEDANCE FORCE: Spring-damper connection to reference
        # ====================================================================
        # Impedance law: F_imp = K(x_ref - x) + B(ẋ_ref - ẋ)
        # where K ∈ ℝ²ˣ² (stiffness), B ∈ ℝ²ˣ² (damping)
        self.builder.Connect(
            cart_state_mux.get_output_port(),  # s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
            impedance.get_input_port(0)
        )
        self.builder.Connect(
            zft.get_output_port(0),  # s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
            impedance.get_input_port(1)
        )
        # Output: F_imp = [F_imp,x, F_imp,y]ᵀ ∈ ℝ²
        
        # ====================================================================
        # ACTUATION: Impedance force → cart-pendulum
        # ====================================================================
        # Cart-pendulum equations: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_ext
        # where τ_ext = [F_imp,x, F_imp,y, 0, 0]ᵀ (force on cart, no direct torque on gimbal)
        self.builder.Connect(
            impedance.get_output_port(),  # F_imp ∈ ℝ²
            self.plant.get_actuation_input_port(self.cart_model)
        )
        
        # ====================================================================
        # FULL STATE ASSEMBLY: Build 14D state for LQR
        # ====================================================================
        # [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
        for i, port_idx in enumerate([0, 1, 2, 3]):
            self.builder.Connect(
                cart_state_demux.get_output_port(port_idx),
                full_state_mux.get_input_port(i)
            )
        self.builder.Connect(muscle.get_output_port(), full_state_mux.get_input_port(4))
        self.builder.Connect(zft.get_output_port(0), full_state_mux.get_input_port(5))
        
        # ====================================================================
        # LQR FEEDBACK LOOP: State → Hold → LQR → Muscle
        # ====================================================================
        self.builder.Connect(full_state_mux.get_output_port(), state_hold.get_input_port())
        self.builder.Connect(state_hold.get_output_port(), lqr.get_input_port())
        self.builder.Connect(lqr.get_output_port(), muscle.get_input_port())
    
    def connect_manipulator_control(self):
        """Connect IK + computed torque control for manipulator."""
        manip_ik = self.systems['manip_ik']
        manip_controller = self.systems['manip_controller']
        cart_state_mux = self.systems['cart_state_mux']
        
        # ====================================================================
        # IK SOLVER: Cart trajectory → desired joint angles
        # ====================================================================
        self.builder.Connect(
            cart_state_mux.get_output_port(),
            manip_ik.get_input_port(0)  # desired cart trajectory
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            manip_ik.get_input_port(1)  # full plant state
        )
        
        # ====================================================================
        # COMPUTED TORQUE CONTROLLER: IK → Controller → Actuators
        # ====================================================================
        self.builder.Connect(
            manip_ik.get_output_port(),
            manip_controller.get_input_port(0)  # desired joint state
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            manip_controller.get_input_port(1)  # current joint state
        )
        # q̈_ref is bundled in desired_joint_state[4:6] (zeros from ManipulatorIKDesiredAngles)
        self.builder.Connect(
            manip_controller.get_output_port(),
            self.plant.get_actuation_input_port(self.manipulator.model_instance)
        )
    
    
    
    def connect_loggers(self):
        """Connect loggers to OFC-specific signals."""
        # Connect base loggers
        self.builder.Connect(
            self.plant.get_state_output_port(self.cart_model),
            self.loggers['state'].get_input_port()
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers['manip_state'].get_input_port()
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers['ee_computer'].get_input_port(0)
        )
        self.builder.Connect(
            self.loggers['ee_computer'].get_output_port(),
            self.loggers['ee_state'].get_input_port()
        )
        
        # Connect OFC-specific loggers
        self.builder.Connect(
            self.systems['zft'].get_output_port(0),
            self.loggers['ref'].get_input_port()
        )
        self.builder.Connect(
            self.systems['muscle'].get_output_port(),
            self.loggers['force'].get_input_port()
        )
        self.builder.Connect(
            self.systems['impedance'].get_output_port(),
            self.loggers['impedance'].get_input_port()
        )
        self.builder.Connect(
            self.systems['cart_state_mux'].get_output_port(),
            self.loggers['cart_traj'].get_input_port()
        )
        self.builder.Connect(
            self.systems['manip_ik'].get_output_port(),
            self.loggers['manip_desired'].get_input_port()
        )
        self.builder.Connect(
            self.systems['manip_controller'].get_output_port(),
            self.loggers['manip_torque'].get_input_port()
        )

class LQRWithOFCForCompleteSystem(ControlSystemBuilder):
    """
    Builds LQR control with Optimal Feedback Control (OFC) architecture:
    
    Cart-Pendulum Control:
    - Muscle Dynamics (low-pass filter on neural commands)
    - ZFT Reference Mass (virtual mass for smooth trajectories)
    - Impedance Force (spring-damper connection to reference)
    - Finite-Horizon LQR (optimal time-varying feedback)
    
    Manipulator Control:
    - IK Solver (converts cart trajectory to joint angles)
    - Computed Torque Controller (joint-space tracking)
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator, config):
        super().__init__(builder, plant, cart_model, manipulator)
        self.config = config
        
        # Pre-compute augmented 14D linearization for welded system:
        # x = [q_arm(2), q_pend(2), v_arm(2), v_pend(2), F(2), p_ref(2), pdot_ref(2)]
        # u = neural command → muscle  (NOT joint torque directly)
        print(colored("\n🔧 Linearizing welded system (14D augmented: plant+muscle+ZFT)...", "yellow"))
        self.A, self.B = build_linearized_for_complete_system_2d(
            plant=plant,
            manipulator=manipulator,
            cart_model=cart_model,
            physics_config=config.physics_config,
            muscle_config=config.muscle_config,
            zft_config=config.zft_config,
            Kp_ct=200.0,   # must match ComputedTorqueJointSpaceController Kp
            Kd_ct=60.0,    # must match ComputedTorqueJointSpaceController Kd
        )
        print(colored(f"✓ Augmented system: A ({self.A.shape}), B ({self.B.shape})", "green"))

        # ----------------------------
        # Derive dimensions directly from the plant (no config needed)
        # ----------------------------
        self.nq_total = self.plant.num_positions()
        self.nv_total = self.plant.num_velocities()
        self.nq_arm   = self.plant.num_positions(self.manipulator.model_instance)
        self.nv_arm   = self.plant.num_velocities(self.manipulator.model_instance)
        self.nq_pend  = self.nq_total - self.nq_arm
        self.nv_pend  = self.nv_total - self.nv_arm
        self.n_state  = 14  # augmented: 8 plant + 2 muscle + 4 ZFT
        self.n_end_effector = 2  # EE position in XY plane
        self.controller_dt = getattr(self.config, "controller_dt", 0.001)

        self.config.ee_kin_config = EndEffectorKinematics2DConfig(
            plant=self.plant,
            manipulator=self.manipulator,
            nq_total=self.nq_total,
            nv_total=self.nv_total,
        )
        print(colored(f"✓ EndEffectorKinematics2DConfig created: "
                     f"nq={self.nq_total}, nv={self.nv_total}", "green"))

        # ----------------------------
        # ZFTJointReferenceIKConfig — proper dataclass, not a raw dict.
        # Created HERE (plant is finalized, self.manipulator is available).
        # Consumed by add_cart_pen_manip_lqr_computed_torque_blocks() via
        # ZFTJointReferenceIK(self.config.zft_ik_config).
        # ----------------------------
        self.config.zft_ik_config = ZFTJointReferenceIKConfig(
            plant=self.plant,
            manipulator=self.manipulator,
            ik_method="differential",   # velocity-integration (no full IK solve per step)
            pos_tol=0.01,
            dt=self.controller_dt,
            Kp=10.0,
        )
        print(colored(f"✓ ZFTJointReferenceIKConfig: "
                     f"method={self.config.zft_ik_config.ik_method}, "
                     f"dt={self.controller_dt}s, Kp={self.config.zft_ik_config.Kp}", "green"))

        # ----------------------------
        # Pre-compute LQR goal state.
        # CRITICAL: x_goal must be in Drake's plant state ordering:
        #   [q_arm (Drake joint order), q_pend, v_arm, v_pend]
        # For this URDF Drake's parse order is [link2_link1 (q2), link1_base (q1)],
        # so arm equilibrium = [q2=0.349 rad, q1=0.0 rad], NOT [0, 0].
        # ----------------------------

        # Step 1: collect equilibrium arm angles IN Drake's joint order
        # (same order as the plant state vector) — used as IK seed
        drake_joint_names = []  # joint names in Drake's parse order
        q_arm_eq = []
        for ji in self.plant.GetJointIndices(self.manipulator.model_instance):
            jt = self.plant.get_joint(ji)
            if jt.num_positions() > 0 and jt.num_velocities() > 0:
                jt_name = jt.name()
                drake_joint_names.append(jt_name)
                jt_cfg = self.manipulator.config.joint_configs.get(jt_name, None)
                q_arm_eq.append(jt_cfg.position if jt_cfg is not None else 0.0)

        # Step 2: Analytical 2-link IK for q_arm_goal
        # Uses same link-length extraction as ManipulatorIKDesiredAngles._extract_link_lengths()
        # and the same FK as compute_jacobian_manual():
        #   x = L1*cos(q1) + L2*cos(q1+q2)
        #   y = L1*sin(q1) + L2*sin(q1+q2)
        # Analytical inverse:
        #   c2 = (x² + y² - L1² - L2²) / (2·L1·L2)
        #   q2 = atan2(√(1-c2²), c2)          ← elbow-down (s2 ≥ 0)
        #   q1 = atan2(y, x) - atan2(L2·s2, L1 + L2·c2)

        jt1 = self.manipulator.JT1_NAME   # "link1_base"
        jt2 = self.manipulator.JT2_NAME   # "link2_link1"

        target_x = getattr(self.config, "target_x", 0.0)
        target_y = getattr(self.config, "target_y", 0.0)

        # --- Extract L1 and L2 from the plant (identical to _extract_link_lengths) ---
        _tmp_ctx = self.plant.CreateDefaultContext()
        _j1 = self.manipulator.get_joint_by_name(self.plant, jt1)
        _j2 = self.manipulator.get_joint_by_name(self.plant, jt2)
        # Set all joints to zero to measure geometry
        self.manipulator.set_state_in_plant(
            self.plant, _tmp_ctx, np.array([0.0, 0.0, 0.0, 0.0])
        )
        # L1: distance between joint1 child frame (link1) and joint2 child frame (link2) at q=0
        X_j1_j2 = self.plant.CalcRelativeTransform(
            _tmp_ctx, _j1.frame_on_child(), _j2.frame_on_child()
        )
        L1 = np.linalg.norm(X_j1_j2.translation()[:2])
        # L2: XY-plane norm of EE offset from link2 (same as ManipulatorIKDesiredAngles)
        L2 = np.linalg.norm(self.manipulator.EE_OFFSET[:2])

        # --- Origin of joint1 (the fixed pivot) in world frame ---
        X_W_j1 = self.plant.CalcRelativeTransform(
            _tmp_ctx, self.plant.world_frame(), _j1.frame_on_child()
        )
        p_j1 = X_W_j1.translation()[:2]   # [x0, y0] of the first pivot in world XY

        # --- Target relative to joint1 origin ---
        px = target_x - p_j1[0]
        py = target_y - p_j1[1]
        reach = np.hypot(px, py)

        print(colored(
            f"\n🎯 Analytical 2-link IK: target=({target_x:.3f}, {target_y:.3f}) m, "
            f"base=({p_j1[0]:.3f}, {p_j1[1]:.3f}), L1={L1:.4f}, L2={L2:.4f}, "
            f"reach={reach:.4f} (max={L1+L2:.4f})", "yellow"
        ))

        ik_ok = reach <= (L1 + L2) - 1e-4 and reach >= abs(L1 - L2) + 1e-4
        if ik_ok:
            c2 = (px**2 + py**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
            c2 = np.clip(c2, -1.0, 1.0)
            s2 = np.sqrt(max(0.0, 1.0 - c2**2))   # elbow-down (s2 ≥ 0)
            q2_goal = np.arctan2(s2, c2)
            q1_goal = np.arctan2(py, px) - np.arctan2(L2 * s2, L1 + L2 * c2)
            # Map to Drake parse order using joint names
            goal_by_name = {jt1: q1_goal, jt2: q2_goal}
            q_arm_goal = np.array([goal_by_name.get(n, 0.0) for n in drake_joint_names])
            print(colored(
                f"✓ IK success → q1={np.rad2deg(q1_goal):.1f}°, q2={np.rad2deg(q2_goal):.1f}° "
                f"(Drake order: {[f'{np.rad2deg(q):.1f}°' for q in q_arm_goal]})",
                "green",
            ))
        else:
            # Fall back to equilibrium if target is outside workspace
            q_arm_goal = np.array(q_arm_eq)
            print(colored(
                f"⚠️  Target ({target_x:.3f}, {target_y:.3f}) outside workspace "
                f"(reach={reach:.3f}, max={L1+L2:.3f}). Using equilibrium.",
                "yellow",
            ))

        self.x_goal = np.concatenate([
            q_arm_goal,                  # arm at goal config (Drake joint order)
            np.zeros(self.nq_pend),      # pendulum upright (α=0, β=0)
            np.zeros(self.nv_total),     # all velocities = 0
            np.zeros(2),                 # muscle force F = 0 at steady state
            np.array([target_x, target_y, 0.0, 0.0]),  # ZFT ref at target, zero vel
        ])  # 14D total
        print(colored(
            f"✓ LQR x_goal (14D): arm={np.round(np.rad2deg(q_arm_goal), 1)}° "
            f"pendulum upright, F=0, ZFT target=({target_x:.3f}, {target_y:.3f})",
            "green",
        ))

    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build control blocks
        self.add_cart_pen_manip_lqr_computed_torque_blocks()
        
        # Connect control loops
        self.connect_system_control()
        
        # Build and connect loggers
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers

    def add_cart_pen_manip_lqr_computed_torque_blocks(self) -> Dict[str, object]:
        """Create and name all blocks used by connect_system_control()."""
        systems: Dict[str, object] = {}

        # ----------------------------
        # State handling: x -> (q, v)
        # ----------------------------
        systems["plant_state_demux"] = self.builder.AddSystem(
            Demultiplexer([self.nq_total, self.nv_total])
        )
        systems["plant_state_demux"].set_name("plant_state_demux_q_v")

        # Augmented state mux: [plant_state(8), muscle_F(2), zft_ref_state(4)] → 14D
        systems["aug_state_mux"] = self.builder.AddSystem(
            Multiplexer([self.nq_total + self.nv_total, 2, 4])
        )
        systems["aug_state_mux"].set_name("aug_state_mux_14d")

        # 14D ZeroOrderHold — holds augmented state for LQR (breaks algebraic loop)
        systems["state_hold"] = self.builder.AddSystem(
            ZeroOrderHold(self.controller_dt, self.n_state)  # 14D
        )
        systems["state_hold"].set_name("state_hold_x_14d")

        systems["plant_state_demux_hold"] = self.builder.AddSystem(
            Demultiplexer([self.nq_total, self.nv_total])
        )
        systems["plant_state_demux_hold"].set_name("plant_state_demux_hold_q_v")

        # Optional split into arm vs pendulum (kept if you need it later)
        systems["q_demux"] = self.builder.AddSystem(
            Demultiplexer([self.nq_arm, self.nq_pend])
        )
        systems["q_demux"].set_name("q_demux_arm_pend")

        systems["v_demux"] = self.builder.AddSystem(
            Demultiplexer([self.nv_arm, self.nv_pend])
        )
        systems["v_demux"].set_name("v_demux_arm_pend")

        # ----------------------------
        # Kinematics: (q,v) -> (p, pdot)
        # ----------------------------
        systems["ee_kin"] = self.builder.AddSystem(
            EndEffectorKinematics2D(self.config.ee_kin_config)
        )
        systems["ee_kin"].set_name("end_effector_kinematics_2d")

        # ----------------------------
        # Muscle: u -> F
        # ----------------------------
        systems["muscle"] = self.builder.AddSystem(
            MuscleDynamics2D(self.config.muscle_config)
        )
        systems["muscle"].set_name("muscle_dynamics")

        # ----------------------------
        # Infinite-horizon LQR: u_opt = -K(x - x_eq) → muscle neural command
        # State x is the full plant state (8D for welded system).
        # A (n×n) and B (n×nu) come from linearizing around the equilibrium.
        # Q penalises state deviation; R penalises neural command effort.
        # ----------------------------
        # Q weights: [q_arm(2), q_pend(2), v_arm(2), v_pend(2)]
        # Goal: STABILISE pendulum with MINIMAL arm movement.
        # Key insight: penalise arm velocity heavily → arm barely moves.
        # Large R → small muscle command → ZFT reference mass barely accelerates.
        
        q_w = np.array([
            100.0, 100.0,    # q_arm  [q1, q2]        — arm position at goal
            500.0, 500.0,    # q_pend [α, β]          — heavy: stabilise pendulum!
            50.0,  50.0,     # v_arm  [q̇1, q̇2]      — penalise arm motion
            100.0, 100.0,    # v_pend [α̇, β̇]        — damp pendulum swing
            0.1,   0.1,      # F      [F_x, F_y]      — muscle force (small: let it act)
            1.0,  1.0,     # p_ref  [x_ref, y_ref]  — ZFT pos toward target
            1.0,   1.0,      # ṗ_ref  [ẋ_ref, ẏ_ref] — ZFT velocity
        ])  # 14D — matches augmented state
        r_w = np.array([10.0, 10.0])
        n_st  = self.n_state           # 14 for augmented system
        nu    = self.B.shape[1]        # 2
        assert len(q_w) == n_st, (
            f"q_w length {len(q_w)} != n_state {n_st}. Adjust q_w in "
            "add_cart_pen_manip_lqr_computed_torque_blocks()"
        )
        Q_lqr  = np.diag(q_w)
        QN_lqr = 2.0 * Q_lqr          # terminal cost = 5× running cost
        R_lqr  = np.diag(r_w)
        # Use the pre-computed equilibrium goal (arm at linearization config, pendulum upright).
        # NOT np.zeros() — the arm is at q2=20° at equilibrium, so zeros causes constant error.
        x_goal = self.x_goal
        horizon = getattr(self.config, "horizon", 10.0)

        systems["lqr_cmd"] = self.builder.AddSystem(
            FiniteHorizonLQRForCompleteSystem(
                A=self.A, B=self.B,
                Q=Q_lqr, R=R_lqr, QN=QN_lqr,
                horizon=horizon,
                timestep=self.controller_dt,
                x_goal=x_goal,
                u_max=getattr(self.config, "u_max", 5.0),  # tiny max: ~5 N cap
            )
        )
        systems["lqr_cmd"].set_name("finite_horizon_lqr_complete_system")

        # ----------------------------
        # ZFT: (p, pdot, F) -> (pzft, pzft_dot, pzft_ddot)
        # ----------------------------
        systems["zft"] = self.builder.AddSystem(
            ZFTReferenceMass2D(self.config.zft_config)
        )
        systems["zft"].set_name("zft_reference_mass")

        # # Keep impedance block only if you really use it in Option A.
        # # If unused, don't create it (or you’ll forget to connect it).
        # if getattr(self.config, "use_impedance", False):
        #     systems["impedance"] = self.builder.AddSystem(
        #         ImpedanceForce2D(self.config.impedance_config)
        #     )
        #     systems["impedance"].set_name("impedance_force")

        # ----------------------------
        # IK: task -> joint refs
        # ----------------------------
        systems["manip_ik"] = self.builder.AddSystem(
            ZFTJointReferenceIK(self.config.zft_ik_config)
        )
        systems["manip_ik"].set_name("ik_task_to_joint_reference")

        # ----------------------------
        # Computed torque
        # ----------------------------
        systems["computed_torque"] = self.builder.AddSystem(
            ComputedTorqueJointSpaceController(
                self.manipulator, self.plant, Kp=200.0, Kd=60.0, tau_max=100.0
            )
        )
        systems["computed_torque"].set_name("computed_torque_inverse_dynamics")

        # Optional torque limits
        if getattr(self.config, "use_torque_limits", True):
            systems["torque_limit"] = self.builder.AddSystem(
                ActuatorLimit2D(tau_max=100.0)
            )
            systems["torque_limit"].set_name("actuator_torque_limits")

        # ----------------------------
        # Convenience muxes
        # ----------------------------
        systems["ref_mux"] = self.builder.AddSystem(Multiplexer([2, 2, 2]))
        systems["ref_mux"].set_name("joint_reference_mux")

        systems["meas_mux"] = self.builder.AddSystem(
            Multiplexer([self.nq_total, self.nv_total])
        )
        systems["meas_mux"].set_name("measured_state_mux")

        # EE state mux: [p(2), pdot(2)] -> ee_state(4) for ZFT cart_state input
        systems["ee_state_mux"] = self.builder.AddSystem(Multiplexer([2, 2]))
        systems["ee_state_mux"].set_name("ee_state_mux_p_pdot")

        systems["log_mux"] = self.builder.AddSystem(
            Multiplexer([
                self.n_end_effector, self.n_end_effector,      # p, pdot
                self.n_end_effector, self.n_end_effector, self.n_end_effector,   # pzft, pzft_dot, pzft_ddot
                self.n_end_effector,         # F
                self.nq_total,
                self.nv_total,
            ])
        )
        systems["log_mux"].set_name("logging_signal_mux")

        # persist
        self.systems.update(systems)
        return systems

    # ------------------------------------------------------------------
    # Loggers
    # ------------------------------------------------------------------
    def add_loggers(self) -> Dict[str, object]:
        """Build loggers for LQR + OFC signals."""
        loggers = {}

        # Use plant-derived dims (set in __init__, never from config which lacks them)
        n = self.nq_total + self.nv_total
        nu = self.B.shape[1]  # number of actuators from linearized system

        loggers["state"] = self.builder.AddSystem(VectorLogSink(n))
        loggers["state"].set_name("state_logger")

        loggers["torques"] = self.builder.AddSystem(VectorLogSink(nu))
        loggers["torques"].set_name("torques_logger")

        # Full “complete-system” muxed log
        loggers["complete_system_log"] = LogVectorOutput(
            self.systems["log_mux"].get_output_port(), self.builder
        )

        self.loggers.update(loggers)
        return loggers

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    def connect_system_control(self):
        """Wire the blocks together (NO plant creation here)."""
        # Required systems
        plant_state_demux = self.systems["plant_state_demux"]
        state_hold = self.systems["state_hold"]
        plant_state_demux_hold = self.systems["plant_state_demux_hold"]

        q_demux = self.systems["q_demux"]
        v_demux = self.systems["v_demux"]

        ee_kin = self.systems["ee_kin"]
        muscle = self.systems["muscle"]
        zft = self.systems["zft"]
        ik = self.systems["manip_ik"]
        computed_torque = self.systems["computed_torque"]

        torque_limit: Optional[object] = self.systems.get("torque_limit", None)

        ref_mux = self.systems["ref_mux"]
        meas_mux = self.systems["meas_mux"]

        # --------------------------------------------------------------
        # 1) Plant state [q_arm_1, q_arm_2, q_pend_1, q_pend_2, v_arm_1, v_arm_2, v_pend_1, v_pend_2] -> demux -> port0 q, port1 v
        # --------------------------------------------------------------
        self.builder.Connect(self.plant.get_state_output_port(), plant_state_demux.get_input_port())
        # Raw q,v (useful for kinematics/logging)
        q_port = plant_state_demux.get_output_port(0)
        v_port = plant_state_demux.get_output_port(1)

        # --------------------------------------------------------------
        # 1a) Plant state -> plant_state_demux_hold (direct, 8D)
        #     Augmented state [plant(8), F(2), zft(4)] -> aug_state_mux -> state_hold (14D)
        # --------------------------------------------------------------
        aug_state_mux = self.systems["aug_state_mux"]
        self.builder.Connect(self.plant.get_state_output_port(), plant_state_demux_hold.get_input_port())
        # Assemble 14D augmented state: plant(8) | muscle_F(2) | zft_ref_state(4)
        # (ZOH breaks algebraic loop: muscle and zft outputs are continuous states sampled at dt)
        self.builder.Connect(self.plant.get_state_output_port(), aug_state_mux.get_input_port(0))
        self.builder.Connect(muscle.get_output_port(),           aug_state_mux.get_input_port(1))
        self.builder.Connect(zft.get_output_port(0),             aug_state_mux.get_input_port(2))  # ref_state(4)
        self.builder.Connect(aug_state_mux.get_output_port(),    state_hold.get_input_port())
        # Held q,v (useful for controller stability / sampled-data control)
        qh_port = plant_state_demux_hold.get_output_port(0)
        vh_port = plant_state_demux_hold.get_output_port(1)

        # Optional arm/pend split (currently not used downstream, but correct to wire)
        self.builder.Connect(q_port, q_demux.get_input_port())
        self.builder.Connect(v_port, v_demux.get_input_port())

        # --------------------------------------------------------------
        # 2) EE kinematics (use raw q,v)
        # --------------------------------------------------------------
        self.builder.Connect(q_port, ee_kin.get_input_port(0))
        self.builder.Connect(v_port, ee_kin.get_input_port(1))

        # --------------------------------------------------------------
        # 2b) LQR optimal command → muscle
        # u = -K(x - x_eq), solved from DARE at construction time.
        # state_hold output (held plant state) → lqr_cmd → muscle.u
        # --------------------------------------------------------------
        self.builder.Connect(
            state_hold.get_output_port(),
            self.systems["lqr_cmd"].get_input_port()
        )
        self.builder.Connect(
            self.systems["lqr_cmd"].get_output_port(),
            muscle.get_input_port()
        )

        # --------------------------------------------------------------
        # 3) EE kinematics p(2)+pdot(2) -> ee_state_mux -> 4D cart_state for ZFT
        # ZFTReferenceMass2D: port 0 = cart_state(4), port 1 = F(2)
        # --------------------------------------------------------------
        ee_state_mux = self.systems["ee_state_mux"]
        self.builder.Connect(ee_kin.get_output_port(0), ee_state_mux.get_input_port(0))  # p(2)
        self.builder.Connect(ee_kin.get_output_port(1), ee_state_mux.get_input_port(1))  # pdot(2)
        self.builder.Connect(ee_state_mux.get_output_port(), zft.get_input_port(0))       # -> cart_state(4)
        self.builder.Connect(muscle.get_output_port(),        zft.get_input_port(1))       # F(2)

        # --------------------------------------------------------------
        # 4) ZFT outputs (ports 1,2,3 — port 0 is ref_state(4) backward-compat)
        #    port 1: p_zft(2), port 2: pdot_zft(2), port 3: pddot_zft(2)
        # --------------------------------------------------------------
        # (ZFT output ports used in steps 5 and connect_loggers)

        # --------------------------------------------------------------
        # 5) IK: (pzft, pzft_dot, pzft_ddot, plant_state) -> joint refs
        # ZFTJointReferenceIK: port 0=p_zft, 1=pdot_zft, 2=pddot_zft, 3=plant_state
        # ZFTReferenceMass2D: output port 1=p_zft, 2=pdot_zft, 3=pddot_zft
        # --------------------------------------------------------------
        self.builder.Connect(zft.get_output_port(1), ik.get_input_port(0))   # p_zft
        self.builder.Connect(zft.get_output_port(2), ik.get_input_port(1))   # pdot_zft
        self.builder.Connect(zft.get_output_port(3), ik.get_input_port(2))   # pddot_zft
        self.builder.Connect(
            self.plant.get_state_output_port(), ik.get_input_port(3)          # full plant state for IK warm-start
        )

        # --------------------------------------------------------------
        # 6) Reference mux: [q_ref(2), qdot_ref(2), qddot_ref(2)] -> 6D desired_joint_state [q1_ref, q2_ref, q̇1_ref, q̇2_ref, q̈1_ref, q̈2_ref]
        # --------------------------------------------------------------
        # Bundle [q_ref(2), qdot_ref(2), qddot_ref(2)] -> 6D desired_joint_state
        self.builder.Connect(ik.get_output_port(0), ref_mux.get_input_port(0))  # q_ref
        self.builder.Connect(ik.get_output_port(1), ref_mux.get_input_port(1))  # qdot_ref
        self.builder.Connect(ik.get_output_port(2), ref_mux.get_input_port(2))  # qddot_ref

        # --------------------------------------------------------------
        # 7) Computed torque: desired_joint_state(6) + manipulator_state(4) -> tau
        # Port 0: ref_mux output = [q_ref, qdot_ref, qddot_ref] (6D) --> computed torque controller
        # Port 1: plant manipulator state (4D = [q_arm_1, q_arm_2, q̇_arm_1, q̇_arm_2]) --> computed torque controller
        # --------------------------------------------------------------
        self.builder.Connect(ref_mux.get_output_port(), computed_torque.get_input_port(0))
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            computed_torque.get_input_port(1)
        )

        # --------------------------------------------------------------
        # 8) Torque \tau_arm_1 and \tau_arm_2 -> optional torque limits -> plant actuation
        # --------------------------------------------------------------
        if torque_limit is not None:
            self.builder.Connect(computed_torque.get_output_port(), torque_limit.get_input_port())
            self.builder.Connect(torque_limit.get_output_port(), self.plant.get_actuation_input_port())
        else:
            self.builder.Connect(computed_torque.get_output_port(), self.plant.get_actuation_input_port())

        # --------------------------------------------------------------
        # 9) Expose measured state mux (if other controllers use it)
        # --------------------------------------------------------------
        self.builder.Connect(q_port, meas_mux.get_input_port(0))
        self.builder.Connect(v_port, meas_mux.get_input_port(1))

    def connect_loggers(self):
        """Connect all loggers to their signal sources."""
        # Plant state logger
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.loggers["state"].get_input_port()
        )

        # Torques logger – prefer post-limit output if available
        if "torque_limit" in self.systems:
            self.builder.Connect(
                self.systems["torque_limit"].get_output_port(),
                self.loggers["torques"].get_input_port()
            )
        elif "computed_torque" in self.systems:
            self.builder.Connect(
                self.systems["computed_torque"].get_output_port(),
                self.loggers["torques"].get_input_port()
            )

        # Log-mux signals: [p(2), pdot(2), pzft(2), pzft_dot(2), pzft_ddot(2), F(2), q(nq), v(nv)]
        if "log_mux" in self.systems:
            plant_state_demux = self.systems["plant_state_demux"]
            ee_kin = self.systems["ee_kin"]
            muscle = self.systems["muscle"]
            zft = self.systems["zft"]
            log_mux = self.systems["log_mux"]

            self.builder.Connect(ee_kin.get_output_port(0), log_mux.get_input_port(0))           # p
            self.builder.Connect(ee_kin.get_output_port(1), log_mux.get_input_port(1))           # pdot
            self.builder.Connect(zft.get_output_port(1),    log_mux.get_input_port(2))           # pzft      (port 1, not 0)
            self.builder.Connect(zft.get_output_port(2),    log_mux.get_input_port(3))           # pzft_dot  (port 2)
            self.builder.Connect(zft.get_output_port(3),    log_mux.get_input_port(4))           # pzft_ddot (port 3)
            self.builder.Connect(muscle.get_output_port(),  log_mux.get_input_port(5))           # F
            self.builder.Connect(plant_state_demux.get_output_port(0), log_mux.get_input_port(6))  # q
            self.builder.Connect(plant_state_demux.get_output_port(1), log_mux.get_input_port(7))  # v
            # Note: complete_system_log was already created in add_loggers() via LogVectorOutput


# ============================================================================
# SIMULATION CLASS
# ============================================================================

class Simulation:
    """
    Manages simulation execution for cart-pendulum-manipulator system.
    
    Uses composition with:
    - SystemBuilder: Builds the multibody plant
    - ControlSystemBuilder: Builds and connects control systems (strategy pattern)
    
    Responsibilities:
    - Configure initial states (EE position, cart position)
    - Store simulation components (plant, manipulator, etc.)
    - Run simulation loop with pluggable control strategies
    
    Attributes:
        config: SimulationConfig with all simulation parameters
        system_builder: SystemBuilder instance for creating the multibody system
        control_builder: ControlSystemBuilder instance (optional, set later)
        builder: DiagramBuilder (set during setup)
        plant: MultibodyPlant (set during setup)
        scene_graph: SceneGraph (set during setup)
        manipulator: CupManipulator (set during setup)
        cart_model: ModelInstanceIndex for cart-pendulum (set during setup)
        ee_world_pos: End-effector position in world frame
        cart_init_pos: Initial cart position [x, y, α, β]
        control_systems: Dictionary of control system blocks
        loggers: Dictionary of data loggers
    """
    
    def __init__(self, config: SimulationConfig, system_builder: SystemBuilder,
                 control_builder: ControlSystemBuilder = None):
        """
        Initialize simulation with configuration and builders.
        
        Args:
            config: SimulationConfig with all simulation parameters
            system_builder: SystemBuilder for creating the multibody system
            control_builder: ControlSystemBuilder for control architecture (optional)
        """
        self.config = config
        self.system_builder = system_builder
        self.control_builder = control_builder
        
        # Will be set during setup_system()
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.manipulator = None
        self.cart_model = None
        self.cart_pendulum = None
        
        # Will be set during configure_initial_state()
        self.ee_world_pos = None
        self.cart_init_pos = None
        self.manipulator_initial_q = None
        
        # Will be set after control builder runs
        self.control_systems = {}
        self.loggers = {}
        
    def setup_system(self):
        """
        Build the multibody system using the SystemBuilder.
        
        Creates the DiagramBuilder, MultibodyPlant, SceneGraph, and adds
        manipulator and cart-pendulum to the plant.
        """
        (self.builder, self.plant, self.scene_graph, 
         self.manipulator, self.cart_pendulum, self.cart_model) = self.system_builder.build(meshcat=self.config.meshcat)
        
    def configure_initial_state(self, context=None, cart_x_override=None, cart_y_override=None):
        """
        Calculate and optionally apply initial state to a Drake context.
        
        When context=None: Calculates values using temporary context, stores them.
        When context provided: Applies stored values to the given context.
        
        This unified method handles both initial calculation and later application,
        eliminating duplication.
        
        Args:
            context: Optional Drake context to apply state to. If None, only calculates.
            cart_x_override: Optional override for cart X position
            cart_y_override: Optional override for cart Y position
        """
        # Calculate manipulator initial angles (only once)
        if self.manipulator_initial_q is None:
            self.manipulator_initial_q = np.array([
                self.system_builder.manipulator_joint_angles['link1_base'],     # q1
                self.system_builder.manipulator_joint_angles['link2_link1'],    # q2
            ])
        
        # Determine which context to use
        needs_temp_context = context is None
        work_context = self.plant.CreateDefaultContext() if needs_temp_context else context
        plant_context = self.plant.GetMyMutableContextFromRoot(work_context)
        
        # Set manipulator positions (needed for both calculation and application)
        self.manipulator.set_positions_user_order(self.plant, plant_context, {
            "link1_base": self.manipulator_initial_q[0],
            "link2_link1": self.manipulator_initial_q[1],
        })
        
        # Calculate EE position and cart position (only once)
        if self.ee_world_pos is None:
            self.ee_world_pos = self.manipulator.get_end_effector_position(self.plant, plant_context)
        
        # if cart_init_pos is None, calculate it based on config or default to EE position
        if self.cart_init_pos is None:
            # Get cart position from config or default to EE position
            if self.config.physics_config.cart_initial_position is not None:
                cart_x = self.config.physics_config.cart_initial_position[0]
                cart_y = self.config.physics_config.cart_initial_position[1]
            else:
                cart_x = self.ee_world_pos[0]
                cart_y = self.ee_world_pos[1]
            
            # Get pendulum angles from config or default to hanging
            if self.config.physics_config.pendulum_initial_angles is not None:
                alpha = self.config.physics_config.pendulum_initial_angles[0]
                beta = self.config.physics_config.pendulum_initial_angles[1]
            else:
                alpha = 0.0  # α (pitch) = 0 (hanging)
                beta = 0.0   # β (roll) = 0 (hanging)
            
            self.cart_init_pos = np.array([cart_x, cart_y, alpha, beta])
        
        # If context was provided, apply full state (positions + velocities)
        if context is not None:
            # Set manipulator velocities to zero
            self.manipulator.set_velocities_user_order(self.plant, plant_context, {
                "link1_base": 0.0,
                "link2_link1": 0.0,
            })
            
            # ------------------------------------------------------------------
            # Mode-aware cart position setting:
            #   Welded mode      → cart_model nq=2 [α,β] only (no cart sliders)
            #   Independent mode → cart_model nq=4 [x,y,α,β]
            # ------------------------------------------------------------------
            cart_nq = self.plant.num_positions(self.cart_model)
            cart_nv = self.plant.num_velocities(self.cart_model)

            if cart_nq == 4:
                # Independent mode — set [x, y, α, β]
                cart_x = cart_x_override if cart_x_override is not None else self.cart_init_pos[0]
                cart_y = cart_y_override if cart_y_override is not None else self.cart_init_pos[1]
                cart_positions = np.array([cart_x, cart_y,
                                           self.cart_init_pos[2], self.cart_init_pos[3]])
            elif cart_nq == 2:
                # Welded mode — cart body has 0 DOF, only pendulum [α, β]
                cart_positions = np.array([self.cart_init_pos[2], self.cart_init_pos[3]])
            else:
                raise ValueError(
                    f"Unexpected cart_model nq={cart_nq}. Expected 2 (welded) or 4 (independent)."
                )

            self.plant.SetPositions(plant_context, self.cart_model, cart_positions)
            self.plant.SetVelocities(plant_context, self.cart_model, np.zeros(cart_nv))
        
        # Print summary only on first call (with temp context)
        if needs_temp_context:
            self._print_initial_config_summary(work_context)
    
    def _print_initial_config_summary(self, context=None):
        """
        Print summary of initial configuration.
        
        Args:
            context: Optional context for verifying cart world position
        """
        print(colored(f"\n📄 Initial Configuration:", "cyan"))
        print(colored(f"  - Manipulator: q1={np.rad2deg(self.manipulator_initial_q[0]):.1f}°, "
                     f"q2={np.rad2deg(self.manipulator_initial_q[1]):.1f}°", "cyan"))
        print(colored(f"  - Cart positioned at EE: ({self.cart_init_pos[0]:.3f}, "
                     f"{self.cart_init_pos[1]:.3f}) m", "cyan"))
        print(colored(f"  - Pendulum: α={np.rad2deg(self.cart_init_pos[2]):.1f}°, "
                     f"β={np.rad2deg(self.cart_init_pos[3]):.1f}°", "cyan"))
        print(colored(f"\n🌍 World Frame Positions:", "yellow", attrs=["bold"]))
        print(colored(f"  - EE in world frame: ({self.ee_world_pos[0]:.3f}, "
                     f"{self.ee_world_pos[1]:.3f}, {self.ee_world_pos[2]:.3f}) m", "yellow"))
        
        if context is not None:
            plant_context = self.plant.GetMyMutableContextFromRoot(context)

            # Mode-aware: only set cart positions if cart has its own DOF.
            # Welded mode:      cart_model nq=2 [α,β] — pass only pendulum angles
            # Independent mode: cart_model nq=4 [x,y,α,β] — pass first cart_nq elements
            cart_nq = self.plant.num_positions(self.cart_model)
            if cart_nq > 0:
                self.plant.SetPositions(
                    plant_context, self.cart_model,
                    self.cart_init_pos[:cart_nq]   # [α,β] welded, [x,y,α,β] independent
                )

            try:
                cart_body = self.plant.GetBodyByName("cart", self.cart_model)
                cart_world_pos = self.plant.CalcPointsPositions(
                    plant_context, cart_body.body_frame(),
                    np.zeros((3, 1)), self.plant.world_frame()
                ).flatten()
                print(colored(f"  - Cart in world frame: ({cart_world_pos[0]:.3f}, "
                             f"{cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m", "yellow"))
                if cart_nq == 2:
                    print(colored(f"  ℹ️  Cart is WELDED to EE — position follows manipulator FK",
                                 "cyan"))
            except Exception as e:
                print(colored(f"  ⚠ Could not query cart world position: {e}", "yellow"))
    
    def setup_control_builder(self, control_builder: ControlSystemBuilder):
        """
        Set or change the control builder (strategy).
        
        Args:
            control_builder: ControlSystemBuilder instance
        """
        self.control_builder = control_builder
        self.control_builder.builder = self.builder
        self.control_builder.plant = self.plant
        self.control_builder.cart_model = self.cart_model
        self.control_builder.manipulator = self.manipulator
    
    def build_control_system(self):
        """
        Build control system using the provided ControlSystemBuilder.
        
        This is where the strategy pattern is applied:
        Different control builders create different control architectures.
        """
        if self.control_builder is None:
            raise ValueError("No control builder provided - use set_control_builder() first")
        
        # Set the control builder's drake components (if not already set during __init__)
        if not hasattr(self.control_builder, 'builder') or self.control_builder.builder is None:
            self.control_builder.builder = self.builder
            self.control_builder.plant = self.plant
            self.control_builder.cart_model = self.cart_model
            self.control_builder.manipulator = self.manipulator
        
        # Build and connect control systems
        self.control_systems, self.loggers = self.control_builder.build_and_connect()
        
        print(colored(f"✓ Control system configured: {type(self.control_builder).__name__}", "green"))

    def _setup_simulator(self, diagram):
        """
        Setup simulator with initial conditions.
        
        Args:
            cart_x_init: Optional override for cart X position
            cart_y_init: Optional override for cart Y position
        """
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        # Apply pre-calculated initial state to simulation context
        self.configure_initial_state(context)
        
        # Publish initial state and add visualization
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        diagram.ForcedPublish(context)
        add_frames_to_meshcat(self.config.meshcat, self.plant, plant_context,
                            self.manipulator, self.cart_model)
        
        print(colored(f"✓ Initial state configured at: {self.config.meshcat.web_url()}", "green"))
        
        return simulator
    
    def _run_simulation_loop(self, simulator, visualizer):
        """Run the simulation loop."""
        visualizer.StartRecording()
        simulator.set_target_realtime_rate(1.0)
        
        dt_sim = 0.01
        current_time = 0.0
        debug_interval = 1.0
        next_debug_time = debug_interval
        
        print(colored("\n🚀 Starting simulation...", "cyan"))
        
        while current_time < self.config.duration:
            if current_time >= next_debug_time:
                self._print_debug_info(simulator, current_time)
                next_debug_time += debug_interval
            
            simulator.AdvanceTo(current_time + dt_sim)
            current_time += dt_sim
        
        print(colored(f"\n✓ Simulation complete at t={current_time:.2f}s", "green"))
        visualizer.PublishRecording()
    
    def run(self, cart_x_init=None, cart_y_init=None):
        """
        Run simulation with the configured control system.
        
        This method replaces the mode-specific run methods (run_lqr_with_manipulator_tracking, etc.)
        with a single generic run() that works with any control builder.
        
        Args:
            cart_x_init: Initial cart X position (defaults to self.cart_init_pos[0])
            cart_y_init: Initial cart Y position (defaults to self.cart_init_pos[1])
        """
        # Build control system if not already built
        if not self.control_systems:
            self.build_control_system()
        
        # Add visualization
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.config.meshcat
        )
        
        # Add frame updater
        frame_list = self._build_frame_list()
        frame_updater = self.builder.AddSystem(
            MeshcatFrameUpdater(self.config.meshcat, self.plant, frame_list, update_period=0.033)
        )
        frame_updater.set_name("frame_updater")
        self.builder.Connect(self.plant.get_state_output_port(), frame_updater.get_input_port(0))
        
        # Build diagram and create simulator
        diagram = self.builder.Build()
        simulator = self._setup_simulator(diagram)
        
        # Run simulation loop
        self._run_simulation_loop(simulator, visualizer)
        
        # Extract and plot results
        self._extract_and_plot_results(simulator.get_context())

    def run_lqr_applied_to_both_cart_and_manipulator(self):
        """
        Run simulation with the configured control system.
        
        This method replaces the mode-specific run methods (run_lqr_with_manipulator_tracking, etc.)
        with a single generic run() that works with any control builder.
        """
        # Build control system if not already built
        if not self.control_systems:
            self.build_control_system()
        
        # Add visualization
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.config.meshcat
        )
        
        # Add frame updater
        frame_list = self._build_frame_list()
        frame_updater = self.builder.AddSystem(
            MeshcatFrameUpdater(self.config.meshcat, self.plant, frame_list, update_period=0.033)
        )
        frame_updater.set_name("frame_updater")
        self.builder.Connect(self.plant.get_state_output_port(), frame_updater.get_input_port(0))
        
        # Build diagram and create simulator
        diagram = self.builder.Build()
        simulator = self._setup_simulator(diagram)
        
        # Run simulation loop
        self._run_simulation_loop(simulator, visualizer)
        
        # Extract and plot results
        self._extract_and_plot_results(simulator.get_context())
    
    def _build_frame_list(self):
        """Build frame list for visualization."""
        from pydrake.multibody.tree import FrameIndex
        
        temp_context = self.plant.CreateDefaultContext()
        frame_list = []
        
        for i in range(self.plant.num_frames()):
            frame = self.plant.get_frame(FrameIndex(i))
            frame_name = frame.name()
            if frame_name == "world":
                continue
            
            if "link" in frame_name.lower() or "cup_center" in frame_name.lower():
                length = 0.15
            elif "cart" in frame_name.lower():
                length = 0.12
            elif "pendulum" in frame_name.lower() or "gimbal" in frame_name.lower():
                length = 0.10
            else:
                length = 0.08
            
            frame_list.append((frame_name, frame, length))
        
        return frame_list
    
    def _print_debug_info(self, simulator, current_time):
        """Print debug information during simulation."""
        context = simulator.get_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        cart_state = self.plant.GetPositionsAndVelocities(plant_context, self.cart_model)
        manip_state = self.plant.GetPositionsAndVelocities(plant_context, self.manipulator.model_instance)
        ee_pos = self.manipulator.get_end_effector_position(self.plant, plant_context)
        
        print(colored(f"\n[t={current_time:.2f}s]", "cyan"))
        print(f"  Cart: ({cart_state[0]:.3f}, {cart_state[1]:.3f})m")
        print(f"  Manip: q1={np.rad2deg(manip_state[0]):.1f}°, q2={np.rad2deg(manip_state[1]):.1f}°")
        print(f"  EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})m")
        print(f"  Error: {np.sqrt((ee_pos[0]-cart_state[0])**2 + (ee_pos[1]-cart_state[1])**2)*1000:.1f} mm")
    
    def _extract_and_plot_results(self, context):
        """Extract logged data and generate plots. Dispatches by available logger keys."""

        # ------------------------------------------------------------------
        # Mode A: LQRWithOFCOnlyCartPendulumBuilder
        # Loggers: state, manip_state, ee_state, ref, force, impedance,
        #          cart_traj, manip_desired, manip_torque
        # ------------------------------------------------------------------
        if 'ref' in self.loggers:
            t                  = self.loggers['state'].FindLog(context).sample_times()
            state_data         = self.loggers['state'].FindLog(context).data()
            manip_state_data   = self.loggers['manip_state'].FindLog(context).data()
            ee_state_data      = self.loggers['ee_state'].FindLog(context).data()
            ref_data           = self.loggers['ref'].FindLog(context).data()
            force_data         = self.loggers['force'].FindLog(context).data()
            impedance_data     = self.loggers['impedance'].FindLog(context).data()
            cart_traj_data     = self.loggers['cart_traj'].FindLog(context).data()
            manip_desired_data = self.loggers['manip_desired'].FindLog(context).data()
            manip_torque_data  = self.loggers['manip_torque'].FindLog(context).data()

            plot_lqr_manip_ee_traj_track_results(
                t, state_data, ref_data, cart_traj_data,
                ee_state_data[0:2, :], ee_state_data[2:4, :],
                force_data, impedance_data, manip_state_data,
                manip_desired_data, manip_torque_data, self.config
            )

        # ------------------------------------------------------------------
        # Mode B: LQRWithOFCForCompleteSystem
        # Loggers: state, torques, complete_system_log (muxed)
        # ------------------------------------------------------------------
        elif 'complete_system_log' in self.loggers:
            t            = self.loggers['state'].FindLog(context).sample_times()
            state_data   = self.loggers['state'].FindLog(context).data()
            torque_data  = self.loggers['torques'].FindLog(context).data()
            mux_data     = self.loggers['complete_system_log'].FindLog(context).data()

            # Unpack mux: [p(2), pdot(2), pzft(2), pzft_dot(2), pzft_ddot(2), F(2), q(nq), v(nv)]
            nq             = self.plant.num_positions()
            p_data         = mux_data[0:2,   :]
            pdot_data      = mux_data[2:4,   :]
            pzft_data      = mux_data[4:6,   :]
            pzft_dot_data  = mux_data[6:8,   :]
            pzft_ddot_data = mux_data[8:10,  :]
            F_data         = mux_data[10:12, :]
            q_data         = mux_data[12:12+nq, :]
            v_data         = mux_data[12+nq:,   :]

            print(colored(f"\n📊 Results summary:", "cyan"))
            print(colored(f"  - Samples  : {len(t)}", "cyan"))
            print(colored(f"  - EE X     : [{p_data[0].min():.3f}, {p_data[0].max():.3f}] m", "cyan"))
            print(colored(f"  - EE Y     : [{p_data[1].min():.3f}, {p_data[1].max():.3f}] m", "cyan"))
            print(colored(f"  - ZFT X    : [{pzft_data[0].min():.3f}, {pzft_data[0].max():.3f}] m", "cyan"))
            print(colored(f"  - τ1 range : [{torque_data[0].min():.1f}, {torque_data[0].max():.1f}] N·m", "cyan"))
            print(colored(f"  - τ2 range : [{torque_data[1].min():.1f}, {torque_data[1].max():.1f}] N·m", "cyan"))

            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle("LQRWithOFCForCompleteSystem Results", fontsize=13)

            axes[0,0].plot(t, p_data[0],    label='p_x (EE)')
            axes[0,0].plot(t, pzft_data[0], '--', label='pzft_x (ref)')
            axes[0,0].set_title("X: EE vs ZFT"); axes[0,0].legend(); axes[0,0].set_ylabel("m")

            axes[0,1].plot(t, p_data[1],    label='p_y (EE)')
            axes[0,1].plot(t, pzft_data[1], '--', label='pzft_y (ref)')
            axes[0,1].set_title("Y: EE vs ZFT"); axes[0,1].legend(); axes[0,1].set_ylabel("m")

            axes[1,0].plot(t, pdot_data[0],     label='ṗ_x')
            axes[1,0].plot(t, pzft_dot_data[0], '--', label='ṗzft_x')
            axes[1,0].set_title("X velocity"); axes[1,0].legend(); axes[1,0].set_ylabel("m/s")

            axes[1,1].plot(t, pdot_data[1],     label='ṗ_y')
            axes[1,1].plot(t, pzft_dot_data[1], '--', label='ṗzft_y')
            axes[1,1].set_title("Y velocity"); axes[1,1].legend(); axes[1,1].set_ylabel("m/s")

            axes[2,0].plot(t, torque_data[0], label='τ1')
            axes[2,0].plot(t, torque_data[1], label='τ2')
            axes[2,0].set_title("Joint Torques"); axes[2,0].legend(); axes[2,0].set_ylabel("N·m")

            axes[2,1].plot(t, F_data[0], label='F_x')
            axes[2,1].plot(t, F_data[1], label='F_y')
            axes[2,1].set_title("Muscle Force"); axes[2,1].legend(); axes[2,1].set_ylabel("N")

            for ax in axes.flat:
                ax.set_xlabel("t [s]")
                ax.grid(True)
            plt.tight_layout()

        else:
            print(colored("ℹ️  No known loggers found, skipping plots.", "yellow"))

        plt.show(block=True)
        print(colored("\n✓ Simulation Complete!", "green", attrs=["bold"]))
    
    



# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    
    # ========================================================================
    # BUILD CONTROL DIAGRAM
    # ========================================================================
    # Build the diagram for the selected control mode
    # Plant and scene_graph are added fresh to each mode's builder
    # For control modes, we pass them to the mode functions which handle the setup
    
    if args.mode == 'scene-viz':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - MUSCLE DYNAMICS & OFC", "cyan", attrs=["bold"]))
        print("="*80)
        print(colored(f"Mode: {args.mode}", "yellow"))
        print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
        print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
        print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
        print("="*80 + "\n")
        
        # Get global configurations
        physics_config = PHYSICS_CONFIG
        muscle_config = MUSCLE_CONFIG
        impedance_config = IMPEDANCE_CONFIG
        zft_config = ZFT_CONFIG
        
        # Start Meshcat
        meshcat = StartMeshcat()
        print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        # Set Meshcat camera using user arguments
        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3)
        )
        
        # ========================================================================
        # BUILD MULTIBODY PLANT WITH TWO SEPARATE MODEL INSTANCES
        # ========================================================================
        # The plant will contain TWO robots:
        # 1. Manipulator (2-DOF cup manipulator) - for visualization
        # 2. Cart-Pendulum (4-DOF system) - actively controlled
        #
        # Each robot gets its own ModelInstance, which allows us to:
        # - Query states separately: plant.get_state_output_port(model_instance)
        # - Apply forces separately: plant.get_actuation_input_port(model_instance)
        # - Set initial conditions separately
        #
        # This is how we separate "cart-pendulum only" states from "full system" states
        builder = DiagramBuilder()
        plant = MultibodyPlant(time_step=0.001)  # 1ms time step for simulation
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        
        # ========================================================================
        # STEP 1: ADD MANIPULATOR TO MAIN PLANT (MODEL INSTANCE 1)
        # ========================================================================
        # Create manipulator configuration and add it to the main plant.
        # This creates the first ModelInstance.
        # The manipulator will remain FIXED at its initial configuration in most modes.
        
        # Use global MANIPULATOR_CONFIG
        manipulator_config = MANIPULATOR_CONFIG
        
        #Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)  # Loads URDF, creates model instance
        
        # Calculate where to position the base so EE is at desired location
        # First, get EE position with base at origin
        initial_q = np.array([np.deg2rad(-0.0), np.deg2rad(4.0)])  # [q1, q2] - initial joint angles
        temp_plant = MultibodyPlant(0.0)
        temp_parser = Parser(temp_plant)
        temp_manip = CupManipulator(manipulator_config, enable_visualization=False)
        temp_manip.load_urdf_to_plant(temp_plant, temp_parser)
        temp_manip.weld_base_to_world(temp_plant, position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0, 0.0]))
        temp_manip.add_end_effector_frame(temp_plant)
        temp_plant.Finalize()
        temp_context = temp_plant.CreateDefaultContext()
        temp_manip.set_positions_user_order(temp_plant, temp_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        ee_at_origin = temp_manip.get_end_effector_position(temp_plant, temp_context)
        
        # Calculate base offset: to center EE at [0, 0], base needs to be at [-ee_x, -ee_y, 0]
        base_offset = -ee_at_origin  # Negate to bring EE to origin
        base_offset[2] = 0.0  # Keep Z at zero (or desired height)
        
        print(colored(f"\n📍 Manipulator Base Positioning:", "yellow"))
        print(colored(f"  - EE position with base at origin: ({ee_at_origin[0]:.3f}, {ee_at_origin[1]:.3f}, {ee_at_origin[2]:.3f}) m", "yellow"))
        print(colored(f"  - Offsetting base to: ({base_offset[0]:.3f}, {base_offset[1]:.3f}, {base_offset[2]:.3f}) m", "yellow"))
        print(colored(f"  - This will center EE at approximately [0, 0]", "green"))
        
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        # AND position it so the EE is at the origin
        manipulator.weld_base_to_world(plant, position=base_offset, orientation=np.array([0.0, 0, 0.0]))
        # Add actuators and end-effector frame BEFORE finalization
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        print(colored(f"✓ End-effector frame '{manipulator.EE_FRAME_NAME}' added to manipulator", "green"))
        
        print(colored(f"✓ Manipulator loaded (ModelInstance: {manipulator.model_instance})", "green"))
        print(colored(f"  - State dimension: 4 (2 positions + 2 velocities)", "cyan"))
        print(colored(f"  - Joints: link1_base, link2_link1", "cyan"))
        
        
        # Determine z-offset for cart-pendulum based on URDF joint origin
        z_offset_from_urdf = 1.17625  # meters, from link1_base joint origin
        print(colored(f"📍 Using z-offset from URDF: {z_offset_from_urdf:.5f} m", "cyan"))
        

        # Initialize cart-pendulum with this z-offset to ensure it is visually aligned with the manipulator's end-effector
        cart_pendulum = CartPendulum2DExtended(physics_config, z_offset=z_offset_from_urdf)
        cart_model = plant.AddModelInstance("cart_pendulum")  # Creates new model instance
        cart_pendulum.build_plant(plant, cart_model)  # Builds cart-pendulum in this instance
        
        print(colored(f"✓ Cart-Pendulum (2D Extended) created (ModelInstance: {cart_model})", "green"))
        print(colored(f"  - State dimension: 8 (4 positions + 4 velocities)", "cyan"))
        print(colored(f"  - DOFs: x, y (cart), α, β (pendulum gimbal angles)", "cyan"))
        print(colored(f"  - Z-plane height: {z_offset_from_urdf:.5f} m", "cyan"))
        
        
        # Set high damping on manipulator joints to lock them in scene-viz mode
        # Must be done BEFORE finalization
        jt1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        jt2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        jt1.set_default_damping_vector([1000.0])  # High damping to lock joint
        jt2.set_default_damping_vector([1000.0])  # High damping to lock joint

        
        plant.Finalize()  # Must be called before adding to diagram



        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                    f"{plant.num_velocities()} total velocities", "green"))
        

        # ========================================================================
        # CONFIGURE INITIAL STATE
        # ========================================================================
        # Initial configuration in natural [q1, q2] order
        initial_q = np.array([np.deg2rad(-0.0), np.deg2rad(4.0)])  # [q1, q2]
        
        # Calculate EE position at configured joint angles
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        
        # Get EE world frame position using the cup_center frame
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
        # Use world frame position for cart initialization
        # Both manipulator and cart work in X-Y plane after rotation: direct mapping [X, Y]
        ee_pos_3d = ee_world_pos  # Use actual world frame coordinates
        
        # Define cart initial position at manipulator EE world position
        cart_init_pos = np.array([ee_world_pos[0], ee_world_pos[1], 0.0, 0.0])  # [x from EE_X, y from EE_Y, α, β]
        
        # Set cart position in temp context for frame visualization
        plant.SetPositions(temp_context, cart_model, cart_init_pos)
        
        cart_body = plant.GetBodyByName("cart", cart_model)
        cart_world_pos = plant.CalcPointsPositions(
            temp_context, cart_body.body_frame(), [0, 0, 0], plant.world_frame()
        ).flatten()
        
        # Print configuration summary
        print(colored(f"\n📄 Initial Configuration:", "cyan"))
        print(colored(f"  - Manipulator: q1={np.rad2deg(initial_q[0]):.1f}°, q2={np.rad2deg(initial_q[1]):.1f}°", "cyan"))
        print(colored(f"  - Manipulator EE: ({ee_pos_3d[0]:.3f}, {ee_pos_3d[1]:.3f}, {ee_pos_3d[2]:.3f}) m", "cyan"))
        print(colored(f"  - Cart positioned at EE: ({cart_init_pos[0]:.3f}, {cart_init_pos[1]:.3f}) m", "cyan"))
        print(colored(f"  - Pendulum hanging: α=0°, β=0°", "cyan"))
        print(colored(f"\n🌍 World Frame Positions:", "yellow", attrs=["bold"]))
        print(colored(f"  - EE in world frame: ({ee_world_pos[0]:.3f}, {ee_world_pos[1]:.3f}, {ee_world_pos[2]:.3f}) m", "yellow"))
        print(colored(f"  - Cart in world frame: ({cart_world_pos[0]:.3f}, {cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m", "yellow"))
        
        # Calculate and display alignment error
        offset_x = abs(ee_world_pos[0] - cart_world_pos[0])
        offset_y = abs(ee_world_pos[1] - cart_world_pos[1])
        print(colored(f"\n✓ EE-Cart Alignment Check:", "green", attrs=["bold"]))
        print(colored(f"  - X offset: {offset_x*1000:.2f} mm", "green" if offset_x < 0.01 else "red"))
        print(colored(f"  - Y offset: {offset_y*1000:.2f} mm", "green" if offset_y < 0.01 else "red"))
        if offset_x < 0.01 and offset_y < 0.01:
            print(colored(f"  ✓ EE and Cart are aligned (< 1cm error)", "green"))
        else:
            print(colored(f"  ⚠ EE and Cart have significant offset", "yellow"))
        
        # Plot coordinate frames to verify orientation
        plot_frames_top_view(plant, temp_context, manipulator, cart_model, 
                           title="Initial Frame Orientations - Scene Viz Mode")
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to display plot
        
        # Add plant to builder (scene_graph was already added when created)
        builder.AddSystem(plant)
        
        # Connect plant to scene graph
        builder.Connect(
            plant.get_geometry_pose_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id())
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port()
        )
        
        # Add Meshcat visualizer
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        plant_context = plant.GetMyMutableContextFromRoot(context)
        
        # Set manipulator to desired configuration (not zeros!)
        manipulator.set_positions_user_order(plant, plant_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        
        # Set cart to initial position
        plant.SetPositions(plant_context, cart_model, cart_init_pos)
        
        # Set all velocities to zero
        plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
        
        # Initial publish to show the scene
        diagram.ForcedPublish(context)
        
        # Add coordinate frames to meshcat
        add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
        
        print(colored("\n🎬 Scene visualization mode", "cyan"))
        print(colored(f"   View at: {meshcat.web_url()}", "cyan"))
        print(colored("\n   Interactive Mode Commands:", "yellow"))
        print(colored("   - 'c x y'  : Move cart to position (x, y)", "yellow"))
        print(colored("   - 'e x y'  : Move manipulator EE to position (x, y) via IK", "yellow"))
        print(colored("   - Ctrl+C   : Exit", "yellow"))
        print(colored(f"\n   Current cart: ({cart_init_pos[0]:.3f}, {cart_init_pos[1]:.3f})", "yellow"))
        print(colored(f"   Current EE:   ({ee_world_pos[0]:.3f}, {ee_world_pos[1]:.3f})\n", "yellow"))
        
        # Get current cart positions and manipulator joint angles
        current_cart_pos = cart_init_pos.copy()
        current_manip_q = initial_q.copy()
        
        # Get cart body for position queries
        cart_body = plant.GetBodyByName("cart", cart_model)
        
        try:
            while True:
                # Wait for user input (blocking - no continuous simulation)
                user_input = input(colored("Enter command (c x y | e x y) or Ctrl+C to exit: ", "cyan")).strip()
                    
                if user_input:
                    try:
                        parts = user_input.split()
                        if len(parts) == 3 and parts[0] in ['c', 'e']:
                            command = parts[0]
                            new_x = float(parts[1])
                            new_y = float(parts[2])
                            
                            if command == 'c':
                                # Update cart position
                                current_cart_pos[0] = new_x
                                current_cart_pos[1] = new_y
                                
                                # Set new positions in plant context
                                plant.SetPositions(plant_context, cart_model, current_cart_pos)
                                
                                # Force visualization update
                                diagram.ForcedPublish(context)
                                
                                # Update coordinate frames
                                add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
                                
                                # Calculate world frame position
                                cart_world_pos = plant.CalcPointsPositions(
                                    plant_context, cart_body.body_frame(), [0, 0, 0], plant.world_frame()
                                ).flatten()
                                
                                print(colored(f"\n✓ Cart updated to: ({new_x:.3f}, {new_y:.3f})", "green"))
                                print(colored(f"  World frame: ({cart_world_pos[0]:.3f}, {cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m\n", "yellow"))
                            
                            elif command == 'e':
                                # Solve IK for manipulator EE position
                                target_xy = np.array([new_x, new_y])
                                print(colored(f"  Solving IK for target ({new_x:.3f}, {new_y:.3f})...", "cyan"))
                                q_solution, success = manipulator.compute_ik_analytical(
                                    plant, target_xy, current_manip_q, pos_tol=0.001, verbose=True
                                )
                                
                                if success:
                                    current_manip_q = q_solution
                                    
                                    # Update manipulator joint positions
                                    manipulator.set_positions_user_order(plant, plant_context, {
                                        "link1_base": current_manip_q[0],
                                        "link2_link1": current_manip_q[1],
                                    })
                                    
                                    # Force visualization update
                                    diagram.ForcedPublish(context)
                                    
                                    # Update coordinate frames
                                    add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
                                    
                                    # Get actual EE position using cup_center frame
                                    ee_actual = manipulator.get_end_effector_position(plant, plant_context)
                                    
                                    print(colored(f"\n✓ Manipulator EE updated to: ({ee_actual[0]:.3f}, {ee_actual[1]:.3f})", "green"))
                                    print(colored(f"  Joint angles: q1={np.rad2deg(current_manip_q[0]):.1f}°, q2={np.rad2deg(current_manip_q[1]):.1f}°\n", "yellow"))
                                else:
                                    print(colored(f"\n✗ IK failed for target ({new_x:.3f}, {new_y:.3f}) - may be out of reach", "red"))
                                    print(colored(f"  Manipulator workspace is limited by link lengths\n", "yellow"))
                        else:
                            print(colored("Invalid input. Format: 'c x y' or 'e x y'\n", "red"))
                    except ValueError:
                        print(colored("Invalid numbers. Format: 'c x y' or 'e x y'\n", "red"))
                
        except KeyboardInterrupt:
            print(colored("\n✓ Visualization stopped", "green"))
        
        return
    
    
    elif args.mode == 'lqr-applied-to-cart-manip-following-cart':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - MUSCLE DYNAMICS & OFC", "cyan", attrs=["bold"]))
        print("="*80)
        print(colored(f"Mode: {args.mode}", "yellow"))
        print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
        print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
        print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
        print("="*80 + "\n")
        
        # Start Meshcat
        meshcat = StartMeshcat()
        print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        # Set Meshcat camera using user arguments
        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3)
        )
        
        # Create simulation configuration from args and global configs
        sim_config = SimulationConfig.from_args(
            args=args,
            physics_config=PHYSICS_CONFIG,
            muscle_config=MUSCLE_CONFIG,
            impedance_config=IMPEDANCE_CONFIG,
            zft_config=ZFT_CONFIG,
            meshcat=meshcat,
        )
        
        # ====================================================================
        # STEP 1: Build system (multibody plant, manipulator, cart)
        # ====================================================================
        system_builder = SystemBuilder(
            physics_config=sim_config.physics_config,
            manipulator_urdf_path=sim_config.manipulator_urdf_path,
            manipulator_joint_angles=sim_config.manipulator_joint_angles,
            manipulator_damping=sim_config.manipulator_damping,
        )
        
        # Build the system and get Drake components
        (builder, plant, scene_graph, 
         manipulator, cart_pendulum, cart_model) = system_builder.build(meshcat=meshcat)
        
        print(colored("\n🚀 Running LQR with manipulator EE trajectory tracking (computed torque)...", "cyan"))
        
        # ====================================================================
        # STEP 2: Create control builder using system components
        # ====================================================================
        control_builder = LQRWithOFCOnlyCartPendulumBuilder(
            builder=builder,
            plant=plant,
            cart_model=cart_model,
            manipulator=manipulator,
            config=sim_config
        )
        
        # ====================================================================
        # STEP 3: Create simulation with system builder and control builder
        # ====================================================================
        simulation = Simulation(
            config=sim_config,
            system_builder=system_builder,
            control_builder=control_builder
        )
        
        # Set the Drake components (already built by system_builder)
        simulation.builder = builder
        simulation.plant = plant
        simulation.scene_graph = scene_graph
        simulation.manipulator = manipulator
        simulation.cart_pendulum = cart_pendulum
        simulation.cart_model = cart_model
        
        # Configure initial state (EE position, cart position), only for vizualization. The actual initial states
        # are set before the simulation
        simulation.configure_initial_state()
        
        # ====================================================================
        # STEP 4: Run simulation with the configured control strategy
        # ====================================================================
        simulation.run(
            cart_x_init=simulation.cart_init_pos[0],
            cart_y_init=simulation.cart_init_pos[1]
        )

    elif args.mode == 'lqr-applied-to-both-cart-manip':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - LQR on BOTH CART & MANIPULATOR", "cyan", attrs=["bold"]))
        print("="*80)
        print(colored(f"Mode: {args.mode}", "yellow"))
        print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
        print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
        print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
        print("="*80 + "\n")
        
        # Start Meshcat
        meshcat = StartMeshcat()
        print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        # Set Meshcat camera using user arguments
        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3)
        )
        
        # Create simulation configuration from args and global configs
        sim_config = SimulationConfig.from_args(
            args=args,
            physics_config=PHYSICS_CONFIG,
            muscle_config=MUSCLE_CONFIG,
            impedance_config=IMPEDANCE_CONFIG,
            zft_config=ZFT_CONFIG,
            meshcat=meshcat,
        )
        
        # ====================================================================
        # STEP 1: Build system (multibody plant, manipulator, cart)
        # ====================================================================
        system_builder = SystemBuilder(
            physics_config=sim_config.physics_config,
            manipulator_urdf_path=sim_config.manipulator_urdf_path,
            manipulator_joint_angles=sim_config.manipulator_joint_angles,
            manipulator_damping=sim_config.manipulator_damping,
        )
        
        # Build the system and get Drake components
        (builder, plant, scene_graph, 
         manipulator, cart_pendulum, cart_model) = system_builder.build(meshcat=meshcat)
        
        print(colored("\n🚀 Running LQR with manipulator and cart...", "cyan"))

        initial_viz = False  # Set to True to show Meshcat initial config before simulation

        if initial_viz:
            # ------------------------------------------------------------------
            # For Meshcat visualization we need a fully built Diagram (with
            # SceneGraph).  But builder.Build() permanently consumes the builder,
            # making STEP 2 impossible if we reuse it.
            #
            # Solution: call system_builder.build() a SECOND time to get a
            # fresh, throwaway builder/plant/scene_graph used only for the viz
            # publish.  The original builder/plant/scene_graph remain untouched
            # for STEP 2 below.  SystemBuilder.build() is stateless and creates
            # new Drake objects on every call.
            # ------------------------------------------------------------------
            print(colored("\n🔭 Building temporary viz diagram (initial config preview)...", "yellow"))
            (viz_builder, viz_plant, viz_scene_graph,
             viz_manip, _, viz_cart_model) = system_builder.build(meshcat=meshcat)

            initial_q = np.array([
                sim_config.manipulator_joint_angles['link1_base'],
                sim_config.manipulator_joint_angles['link2_link1'],
            ])

            # Add Meshcat visualizer to the throwaway builder — safe to Build()
            MeshcatVisualizer.AddToBuilder(viz_builder, viz_scene_graph, meshcat)
            viz_diagram = viz_builder.Build()
            viz_sim     = Simulator(viz_diagram)
            viz_ctx     = viz_sim.get_mutable_context()
            viz_plant_ctx = viz_plant.GetMyMutableContextFromRoot(viz_ctx)

            # Set manipulator initial config
            viz_manip.set_positions_user_order(viz_plant, viz_plant_ctx, {
                "link1_base":  initial_q[0],
                "link2_link1": initial_q[1],
            })

            # Welded mode: cart_model nq=2 [α,β] only
            pendulum_num_q = viz_plant.num_positions(viz_cart_model)
            pendulum_num_v = viz_plant.num_velocities(viz_cart_model)
            viz_plant.SetPositions(viz_plant_ctx, viz_cart_model, np.zeros(pendulum_num_q))
            viz_plant.SetVelocities(viz_plant_ctx, np.zeros(viz_plant.num_velocities()))

            # Publish to Meshcat
            viz_diagram.ForcedPublish(viz_ctx)

            # FK queries for info print
            ee_world_pos = viz_manip.get_end_effector_position(viz_plant, viz_plant_ctx)
            try:
                cart_body_viz = viz_plant.GetBodyByName("cart", viz_cart_model)
                cart_world_pos = viz_plant.CalcPointsPositions(
                    viz_plant_ctx, cart_body_viz.body_frame(),
                    np.zeros((3, 1)), viz_plant.world_frame()
                ).flatten()
            except Exception as e:
                cart_world_pos = ee_world_pos
                print(colored(f"  ⚠ cart FK fallback: {e}", "yellow"))

            print(colored(f"\n📄 Pre-Simulation System Configuration:", "cyan"))
            print(colored(f"  - Total DOF      : nq={viz_plant.num_positions()}, "
                         f"nv={viz_plant.num_velocities()}", "cyan"))
            print(colored(f"  - cart_model DOF : nq={pendulum_num_q}, nv={pendulum_num_v} "
                         f"(pendulum only — cart body is welded)", "cyan"))
            print(colored(f"  - Manipulator    : q1={np.rad2deg(initial_q[0]):.1f}°, "
                         f"q2={np.rad2deg(initial_q[1]):.1f}°", "cyan"))
            print(colored(f"  - EE position    : ({ee_world_pos[0]:.3f}, "
                         f"{ee_world_pos[1]:.3f}, {ee_world_pos[2]:.3f}) m", "cyan"))
            print(colored(f"  - Cart world pos : ({cart_world_pos[0]:.3f}, "
                         f"{cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m "
                         f"(welded → follows EE)", "green"))
            print(colored(f"  - Pendulum       : α=0°, β=0° (hanging vertically)", "cyan"))
            print(colored(f"\n🌐 View at: {meshcat.web_url()}", "green"))
            # print(colored("   Press Enter to proceed to simulation...", "yellow"))
            # try:
            #     input()
            # except KeyboardInterrupt:
            #     pass
            # viz_diagram / viz_plant go out of scope — GC'd, no effect on main builder

        # ====================================================================
        # STEP 2: Create control builder using the ORIGINAL (untouched) builder
        # ====================================================================
        control_builder = LQRWithOFCForCompleteSystem(
            builder=builder,
            plant=plant,
            cart_model=cart_model,
            manipulator=manipulator,
            config=sim_config
        )

        # ====================================================================
        # STEP 3: Create simulation with system builder and control builder
        # ====================================================================
        simulation = Simulation(
            config=sim_config,
            system_builder=system_builder,
            control_builder=control_builder
        )

        simulation.builder     = builder
        simulation.plant       = plant
        simulation.scene_graph = scene_graph
        simulation.manipulator = manipulator
        simulation.cart_pendulum = cart_pendulum
        simulation.cart_model  = cart_model

        simulation.configure_initial_state()

        # ====================================================================
        # STEP 4: Run simulation (Meshcat live view starts here)
        # ====================================================================
        simulation.run_lqr_applied_to_both_cart_and_manipulator()
if __name__ == "__main__":
    main()

