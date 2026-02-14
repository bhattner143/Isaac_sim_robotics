#!/usr/bin/env python3
"""
Test Script: Manipulator Pushes Cart in 2D (Full X-Y Motion)

This script demonstrates the 2D generalization of the manipulator-pushes-cart system,
where the manipulator can push the cart in arbitrary 2D directions (both X and Y).

SYSTEM ARCHITECTURE:
════════════════════
Manipulator → 2D Impedance Control → Cart-Pendulum System

Key Features:
- Full 2D ZFT reference mass (operates in both X and Y)
- 2D impedance force (F_imp = [F_x, F_y])
- 2D trajectory tracking
- Demonstrates generalization from 1D to 2D

CONTROL EQUATIONS:
══════════════════
ZFT Reference Mass (2D):
    M_ref · ẍ_ref = K·(x_ee - x_ref) + D·(ẋ_ee - ẋ_ref) + F_muscle

Impedance Force (2D):
    F_imp = K·(x_ee - x_ref) + D·(ẋ_ee - ẋ_ref)
    where F_imp = [F_x, F_y]^T

Jacobian Transpose Control:
    τ = -J^T(q) · F_imp

USAGE:
══════
    # Push in X direction only (impedance mode)
    python test_manipulator_pushes_cart_2d.py --mode impedance --dx 0.5 --dy 0.0
    
    # Push with LQR control
    python test_manipulator_pushes_cart_2d.py --mode lqr --dx 0.3 --dy 0.2 --horizon 10.0
    
    # Push diagonally (both X and Y)
    python test_manipulator_pushes_cart_2d.py --dx 0.3 --dy 0.2 --duration 5
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from termcolor import colored
from pathlib import Path
import sys
from scipy.linalg import solve_discrete_are

# IMPORTANT: Parse our arguments BEFORE importing from script_cup_manipulator_controller_ofc
# That script has global argparse code that will interfere otherwise
_parser = argparse.ArgumentParser(
    description="Test manipulator pushes cart in 2D (full X-Y motion)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_parser.add_argument('--dx', type=float, default=0.3, 
                   help='Desired X displacement [m] (default: 0.3)')
_parser.add_argument('--dy', type=float, default=0.2, 
                   help='Desired Y displacement [m] (default: 0.2)')
_parser.add_argument('--duration', type=float, default=5.0, 
                   help='Simulation duration [s] (default: 5.0)')
_parser.add_argument('--K', type=float, default=100.0, 
                   help='Impedance stiffness [N/m] (default: 100.0)')
_parser.add_argument('--D', type=float, default=20.0, 
                   help='Impedance damping [N·s/m] (default: 20.0)')
_parser.add_argument('--M', type=float, default=2.0, 
                   help='Reference mass [kg] (default: 2.0)')
_parser.add_argument('--mode', type=str, default='lqr',
                   choices=['impedance', 'lqr'],
                   help='Control mode: impedance (direct) or lqr (optimal)')
_parser.add_argument('--horizon', type=float, default=10.0,
                   help='LQR planning horizon [s] (default: 10.0, only for --mode lqr)')

# Parse early to avoid conflicts with imported module's argparse
_ARGS = _parser.parse_args()

# Now clear sys.argv to prevent imported module from seeing our arguments
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Keep only script name

sys.path.append(str(Path(__file__).parent))

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    Simulator,
    VectorLogSink,
    LeafSystem,
    BasicVector,
    AbstractValue,
    ExternallyAppliedSpatialForce,
    SpatialForce,
    MeshcatVisualizer,
    StartMeshcat,
    AddMultibodyPlantSceneGraph,
    Multiplexer,
    Demultiplexer,
    ConstantVectorSource,
)
from pydrake.multibody.tree import JacobianWrtVariable

from robot_types import create_cup_manipulator_config, create_cart_pendulum_config
from script_cup_manipulator_controller_ofc import (
    CupManipulator,
    CartPendulum3D,
    ZFTReferenceMass,
    ImpedanceForce,
    EndEffectorKinematics,
    create_zft_reference_mass_config,
    create_impedance_force_config,
)


class MuscleDynamics2D(LeafSystem):
    """
    First-order muscle/actuator dynamics for 2D force:
        F_dot = (-F + u) / tau

    Input:  u (2)  = neural command / desired force [F_x, F_y] (N)
    Output: F (2)  = muscle force applied to system (N)
    State:  F (2)
    """
    def __init__(self, muscle_tau=0.03, initial_force=None):
        LeafSystem.__init__(self)
        
        self.muscle_tau = muscle_tau
        self.initial_force = initial_force if initial_force is not None else np.zeros(2)
        
        # Declare continuous state (2D force)
        self.DeclareContinuousState(2)
        
        # Input: neural command u (2D)
        self.DeclareVectorInputPort("u", BasicVector(2))
        
        # Output: muscle force F (2D) - NOT direct feedthrough (depends only on state)
        self.DeclareVectorOutputPort(
            "F",
            BasicVector(2),
            self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}  # Only depends on state, breaks algebraic loop
        )
    
    def SetDefaultState(self, context, state):
        state.get_mutable_continuous_state_vector().SetFromVector(self.initial_force)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        F = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port(0).Eval(context)
        F_dot = (-F + u) / self.muscle_tau
        derivatives.get_mutable_vector().SetFromVector(F_dot)
    
    def _calc_output(self, context, output):
        F = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(F)


class FiniteHorizonLQRController2D(LeafSystem):
    """
    Finite-horizon, continuous-time LQR for 2D manipulator-cart system.
    
    Control law: u(t) = -K(t) (x(t) - x_goal)
    
    Cost function:
        J = ∫_0^T [ x'Qx + u'Ru ] dt + x(T)'QN·x(T)
    
    State vector: [x_cart, y_cart, θ_pend, ẋ_cart, ẏ_cart, θ̇_pend, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
                  (12D: cart XY + pendulum + cart velocities + muscle forces + ZFT refs)
    Input: command u (2D: [u_x, u_y])
    """
    
    def __init__(self, A, B, Q, QN, R, x_goal, horizon=10.0, timestep=0.01):
        LeafSystem.__init__(self)
        
        self.A = A  # (n, n)
        self.B = B  # (n, m)
        self.Q = Q
        self.QN = QN
        self.R = R
        self.x_goal = x_goal
        self.horizon = horizon
        self.timestep = timestep
        
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        
        # Discretize system
        self.Ad, self.Bd = self._discretize_system(A, B, timestep)
        
        # Solve finite-horizon Riccati recursion
        self.K_history, self.time_points = self._solve_finite_horizon_lqr(
            self.Ad, self.Bd, Q, QN, R, horizon, timestep
        )
        
        # Declare input: full state
        self.DeclareVectorInputPort("state", BasicVector(n_states))
        
        # Declare output: control command u
        self.DeclareVectorOutputPort(
            "u",
            BasicVector(n_inputs),
            self.CalcU,
        )
        
        print(colored(f"✓ LQR Controller: {n_states}D state → {n_inputs}D control", "green"))
        print(colored(f"  Horizon: {horizon:.1f} s, Timestep: {timestep:.3f} s", "cyan"))
    
    def CalcU(self, context, output):
        """Compute control: u = -K(t) (x - x_goal)"""
        # Get current state
        x = self.get_input_port(0).Eval(context)
        
        # Get current time
        t = context.get_time()
        
        # Find closest time index
        idx = int(np.clip(t / self.timestep, 0, len(self.time_points) - 1))
        K_t = self.K_history[idx]
        
        # Control law
        u = -K_t @ (x - self.x_goal)
        output.SetFromVector(u)
    
    @staticmethod
    def _discretize_system(A, B, dt):
        """Discretize continuous system using zero-order hold."""
        n = A.shape[0]
        m = B.shape[1]
        
        # Matrix exponential approximation (first-order)
        Ad = np.eye(n) + A * dt
        Bd = B * dt
        
        return Ad, Bd
    
    @staticmethod
    def _solve_finite_horizon_lqr(Ad, Bd, Q, QN, R, horizon, dt):
        """
        Solve finite-horizon discrete-time LQR via Riccati recursion.
        
        Returns:
            K_history: List of gain matrices K(t) from t=0 to t=T
            time_points: Corresponding time points
        """
        N = int(horizon / dt)
        n = Ad.shape[0]
        m = Bd.shape[1]
        
        # Initialize storage
        P = [None] * (N + 1)  # Cost-to-go matrices
        K_history = [None] * (N + 1)  # Gain matrices
        time_points = np.arange(N + 1) * dt
        
        # Terminal condition
        P[N] = QN
        K_history[N] = np.zeros((m, n))
        
        # Backward recursion
        for k in range(N - 1, -1, -1):
            # Discrete-time Riccati equation
            P_next = P[k + 1]
            
            # K = (R + B^T P B)^{-1} B^T P A
            BtPB = Bd.T @ P_next @ Bd
            K_k = np.linalg.solve(R + BtPB, Bd.T @ P_next @ Ad)
            
            # P = Q + A^T P A - A^T P B K
            P_k = Q + Ad.T @ P_next @ Ad - Ad.T @ P_next @ Bd @ K_k
            
            K_history[k] = K_k
            P[k] = P_k
        
        return K_history, time_points


class ManipulatorJacobianTransposeController2D(LeafSystem):
    """
    2D Jacobian transpose controller for impedance control.
    
    Maps 2D task-space impedance force to joint torques:
        τ = -J^T(q) · F_imp
    
    where F_imp = [F_x, F_y]^T
    """
    
    def __init__(self, plant, manipulator):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        self.plant_context = plant.CreateDefaultContext()
        
        # Input ports
        self.F_imp_input = self.DeclareVectorInputPort("F_imp", BasicVector(2))  # 2D force
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output port
        self.DeclareVectorOutputPort(
            "joint_torques",
            BasicVector(2),
            self.CalcTorques
        )
    
    def CalcTorques(self, context, output):
        """Compute τ = -J^T · F_imp for 2D force."""
        state = self.state_input.Eval(context)
        q = state[:2]
        
        # Get 2D impedance force
        F_imp_2d = self.F_imp_input.Eval(context)  # [F_x, F_y]
        
        # Convert to 3D spatial force
        F_imp_3d = np.array([F_imp_2d[0], F_imp_2d[1], 0.0])
        
        # Update plant context with current joint positions
        full_q = self.plant.GetPositions(self.plant_context)
        full_q[:2] = q
        self.plant.SetPositions(self.plant_context, full_q)
        
        # Compute Jacobian
        ee_origin = np.zeros(3)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            ee_origin,
            self.world_frame,
            self.world_frame
        )
        
        J_translational = J_spatial[3:6, :]  # Linear velocity part (3×n)
        J_manip = J_translational[:, :2]      # Only manipulator DOFs (3×2)
        
        # Torques: τ = -J^T · F
        tau = -J_manip.T @ F_imp_3d  # (2,)
        
        output.SetFromVector(tau)


class CartForceApplicator2D(LeafSystem):
    """Applies 2D force to cart body."""
    
    def __init__(self, cart_body_index):
        LeafSystem.__init__(self)
        self.cart_body_index = cart_body_index
        
        self.DeclareVectorInputPort("cart_force", BasicVector(2))
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_output
        )
    
    def _calc_output(self, context, output):
        """Convert 2D force to spatial force."""
        cart_force_2d = self.GetInputPort("cart_force").Eval(context)
        
        spatial_force = ExternallyAppliedSpatialForce()
        spatial_force.body_index = self.cart_body_index
        spatial_force.F_Bq_W = SpatialForce(
            tau=np.zeros(3),
            f=np.array([cart_force_2d[0], cart_force_2d[1], 0.0])
        )
        spatial_force.p_BoBq_B = np.zeros(3)
        
        output.set_value([spatial_force])


def build_linearized_system_2d(
    plant,
    manipulator,
    cart_pendulum,
    K_imp,
    D_imp,
    M_ref,
    muscle_tau=0.03,
):
    """
    Build linearized 14D system using Drake's Linearize() for cart-pendulum.
    
    APPROACH:
    =========
    1. Use Drake's Linearize() on cart-pendulum subsystem → A_cp, B_cp
    2. Muscle dynamics are already linear: Ḟ = (-F + u) / τ
    3. ZFT dynamics are already linear: ẍ_ref = (K*(x-x_ref) + D*(ẋ-ẋ_ref) + F) / M_ref
    4. Assemble full 14×14 A and 14×2 B by block composition
    
    STATE VECTOR (14D):
    ===================
    [0-7]:   Cart-pendulum [x, y, α, β, ẋ, ẏ, α̇, β̇]  (8D - from Drake linearization)
    [8-9]:   Muscle forces [F_x, F_y]                   (2D - linear first-order)
    [10-13]: ZFT reference [x_ref, y_ref, ẋ_ref, ẏ_ref] (4D - linear second-order)
    
    INPUT (2D):
    ===========
    [u_x, u_y]: Neural commands to muscles
    
    Returns:
        A (14x14), B (14x2): Full system linearized matrices
    """
    
    # ========================================================================
    # STEP 1: Linearize cart-pendulum using Drake's Linearize()
    # ========================================================================
    
    # Create a temporary plant with ONLY cart-pendulum for linearization
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)  # Continuous time
    
    # Recreate cart-pendulum in temp plant
    cart_config = create_cart_pendulum_config(
        cart_mass=3.0,           # from script_cart
        cart_damping=0.0,        # from script_cart (no damping)
        pendulum_mass=0.3,       # from script_cart
        pendulum_length=0.5,     # from script_cart
    )
    temp_cart_pendulum = CartPendulum3D(cart_config, visualize_cart=False, add_cart_actuators=True)
    temp_model = temp_plant.AddModelInstance("cart_pendulum_temp")
    temp_cart_pendulum.attach_to_plant(temp_plant, temp_model, register_visuals=False)
    
    temp_plant.Finalize()
    
    # Linearize around equilibrium: cart at origin, pendulum upright
    temp_context = temp_plant.CreateDefaultContext()
    
    # Set equilibrium state
    # Cart-pendulum has 6 positions: [x, y, z, qw, qx, qy, qz, α, β]
    # We want: x=0, y=0, pendulum upright (α=0, β=0)
    temp_plant.SetPositions(temp_context, np.zeros(temp_plant.num_positions()))
    temp_plant.SetVelocities(temp_context, np.zeros(temp_plant.num_velocities()))
    
    # Set actuation input to zero for linearization
    temp_plant.get_actuation_input_port().FixValue(temp_context, np.zeros(2))
    
    # Linearize using Drake's Linearize function
    # Specify actuation input port (not spatial force port)
    from pydrake.systems.primitives import Linearize
    actuation_port = temp_plant.get_actuation_input_port()
    linear_sys = Linearize(
        temp_plant, 
        temp_context,
        input_port_index=actuation_port.get_index(),
        output_port_index=temp_plant.get_state_output_port().get_index()
    )
    
    A_full = linear_sys.A()
    B_full = linear_sys.B()
    
    # Cart-pendulum state is 8D: [x, y, α, β, ẋ, ẏ, α̇, β̇]
    # Positions (4): x, y, α, β
    # Velocities (4): ẋ, ẏ, α̇, β̇
    # We keep ALL 8 states (both pendulum angles)
    
    # Use full 8×8 A matrix and 8×2 B matrix
    A_cp = A_full  # 8×8
    B_cp = B_full  # 8×2
    
    # ========================================================================
    # STEP 2: Build muscle dynamics (already linear)
    # ========================================================================
    # Ḟ = (-F + u) / τ
    # State: [F_x, F_y]
    # Input: [u_x, u_y]
    
    A_muscle = np.array([
        [-1.0/muscle_tau, 0.0],
        [0.0, -1.0/muscle_tau]
    ])  # 2×2
    
    B_muscle = np.array([
        [1.0/muscle_tau, 0.0],
        [0.0, 1.0/muscle_tau]
    ])  # 2×2
    
    # ========================================================================
    # STEP 3: Build ZFT reference dynamics (already linear)
    # ========================================================================
    # ẋ_ref = ẋ_ref (trivial)
    # ẍ_ref = (K*(x-x_ref) + D*(ẋ-ẋ_ref) + F) / M_ref
    #
    # State: [x_ref, y_ref, ẋ_ref, ẏ_ref]
    # Couples to cart state: [x, y, ẋ, ẏ]
    # Couples to muscle force: [F_x, F_y]
    
    A_zft = np.zeros((4, 4))
    A_zft[0, 2] = 1.0  # ẋ_ref
    A_zft[1, 3] = 1.0  # ẏ_ref
    A_zft[2, 0] = -K_imp / M_ref  # ẍ_ref depends on -x_ref
    A_zft[2, 2] = -D_imp / M_ref  # ẍ_ref depends on -ẋ_ref
    A_zft[3, 1] = -K_imp / M_ref  # ÿ_ref depends on -y_ref
    A_zft[3, 3] = -D_imp / M_ref  # ÿ_ref depends on -ẏ_ref
    
    # ========================================================================
    # STEP 4: Assemble full 14×14 A matrix with coupling
    # ========================================================================
    A = np.zeros((14, 14))
    
    # Block 1: Cart-pendulum dynamics (8×8)
    A[0:8, 0:8] = A_cp
    
    # Block 2: Muscle dynamics (2×2)
    A[8:10, 8:10] = A_muscle
    
    # Block 3: ZFT dynamics (4×4)
    A[10:14, 10:14] = A_zft
    
    # Coupling: Cart acceleration affected by impedance force
    # F_imp = K*(x_ref - x) + D*(ẋ_ref - ẋ)
    # This force affects cart through B_cp (input mapping)
    # But impedance is NOT an external input - it's internal coupling
    # So we add coupling terms directly to A matrix
    
    # State indices: [x=0, y=1, α=2, β=3, ẋ=4, ẏ=5, α̇=6, β̇=7, F_x=8, F_y=9, x_ref=10, y_ref=11, ẋ_ref=12, ẏ_ref=13]
    
    # Cart acceleration depends on reference position/velocity
    # ẍ += (K/M) * x_ref + (D/M) * ẋ_ref  (positive coupling from reference)
    M_cart = 3.0  # Cart mass
    A[4, 10] = K_imp / M_cart   # ẍ depends on x_ref
    A[4, 12] = D_imp / M_cart   # ẍ depends on ẋ_ref
    A[5, 11] = K_imp / M_cart   # ÿ depends on y_ref
    A[5, 13] = D_imp / M_cart   # ÿ depends on ẏ_ref
    
    # Cart acceleration depends on its own position/velocity (negative coupling)
    A[4, 0] += -K_imp / M_cart  # ẍ depends on -x
    A[4, 4] += -D_imp / M_cart  # ẍ depends on -ẋ
    A[5, 1] += -K_imp / M_cart  # ÿ depends on -y
    A[5, 5] += -D_imp / M_cart  # ÿ depends on -ẏ
    
    # ZFT acceleration depends on cart position/velocity
    A[12, 0] = K_imp / M_ref   # ẍ_ref depends on x
    A[12, 4] = D_imp / M_ref   # ẍ_ref depends on ẋ
    A[13, 1] = K_imp / M_ref   # ÿ_ref depends on y
    A[13, 5] = D_imp / M_ref   # ÿ_ref depends on ẏ
    
    # ZFT acceleration depends on muscle force
    A[12, 8] = 1.0 / M_ref     # ẍ_ref depends on F_x
    A[13, 9] = 1.0 / M_ref     # ÿ_ref depends on F_y
    
    # ========================================================================
    # STEP 5: Assemble full 14×2 B matrix
    # ========================================================================
    B = np.zeros((14, 2))
    
    # Only muscle dynamics are directly affected by input
    B[8:10, 0:2] = B_muscle
    
    # Cart-pendulum is NOT directly actuated by neural commands
    # (it's actuated by impedance force, which is internal coupling)
    
    # ZFT is NOT directly actuated by neural commands
    # (it's actuated by muscle force, which is internal state)
    
    return A, B


def simulate_2d_push(
    dx=0.3,
    dy=0.2,
    duration=5.0,
    K_imp=50.0,  # from script_cart
    D_imp=10.0,  # from script_cart
    M_ref=1.0,   # from script_cart (Mh)
):
    """
    Simulate manipulator pushing cart in 2D.
    
    Args:
        dx: Desired X displacement [m]
        dy: Desired Y displacement [m]
        duration: Simulation duration [s]
        K_imp: Impedance stiffness [N/m]
        D_imp: Impedance damping [N·s/m]
        M_ref: Reference mass [kg]
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR PUSHES CART - 2D MOTION TEST", "cyan", attrs=["bold"]))
    print(colored("Full X-Y Impedance Control (Generalized from 1D)", "cyan"))
    print(colored("="*80, "cyan"))
    print(colored(f"Target displacement: ΔX = {dx:.3f} m, ΔY = {dy:.3f} m", "yellow"))
    print(colored(f"Duration: {duration:.1f} s", "yellow"))
    print(colored(f"Impedance: K = {K_imp:.1f} N/m, D = {D_imp:.1f} N·s/m", "yellow"))
    print(colored(f"Reference Mass: M = {M_ref:.1f} kg", "yellow"))
    print(colored("="*80 + "\n", "cyan"))
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Build system
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Add manipulator
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),
        damping=(0.1, 0.1),
    )
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
    cart_config = create_cart_pendulum_config(
        cart_mass=5.0,
        cart_damping=2.0,
        pendulum_mass=0.5,
        pendulum_length=0.2,
    )
    cart_pendulum = CartPendulum3D(cart_config, visualize_cart=True, add_cart_actuators=False)
    cart_model = plant.AddModelInstance("cart_pendulum")
    cart_pendulum.attach_to_plant(plant, cart_model, register_visuals=False)
    
    plant.Finalize()
    
    # Get initial EE position
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, manipulator.model_instance, 
                      np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    ee_pos_init = manipulator.CalcPosition(plant, temp_context)[:2]
    
    print(colored(f"✓ System created", "green"))
    print(colored(f"  Initial EE: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m\n", "cyan"))
    
    # Create control systems
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    
    # 2D ZFT Reference Mass (full 2D, no locking)
    zft_config = create_zft_reference_mass_config(
        Mh=np.diag([M_ref, M_ref]),
        kp=np.diag([K_imp, K_imp]),
        kd=np.diag([D_imp, D_imp]),
        yref0=np.array([ee_pos_init[0], ee_pos_init[1]]),
        vref0=np.array([0.0, 0.0])
    )
    zft_ref_mass = builder.AddSystem(ZFTReferenceMass(zft_config))
    
    # 2D Impedance Force (full 2D)
    imp_config = create_impedance_force_config(
        kp=np.diag([K_imp, K_imp]),
        kd=np.diag([D_imp, D_imp]),
        force_limit=None
    )
    impedance_force = builder.AddSystem(ImpedanceForce(imp_config))
    
    # 2D Jacobian controller
    jacobian_controller = builder.AddSystem(
        ManipulatorJacobianTransposeController2D(plant, manipulator)
    )
    
    # Zero muscle force (2D)
    zero_muscle_force = builder.AddSystem(ConstantVectorSource(np.zeros(2)))
    
    # Cart force applicator
    cart_force_applicator = builder.AddSystem(
        CartForceApplicator2D(cart_pendulum.cart_body.index())
    )
    
    # Demux/Mux for state routing
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))  # [manip(4), cart(8)]
    ee_state_mux = builder.AddSystem(Multiplexer([2, 2]))   # [pos(2), vel(2)] -> state(4)
    
    # Connect systems
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    
    # Manipulator state to kinematics and controller
    builder.Connect(state_demux.get_output_port(0), ee_kinematics.GetInputPort("manipulator_state"))
    builder.Connect(state_demux.get_output_port(0), jacobian_controller.GetInputPort("manipulator_state"))
    
    # EE state (4D: [x, y, vx, vy])
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_state_mux.get_input_port(0))
    builder.Connect(ee_kinematics.GetOutputPort("ee_velocity"), ee_state_mux.get_input_port(1))
    
    # ZFT connections
    builder.Connect(ee_state_mux.get_output_port(0), zft_ref_mass.GetInputPort("y_v"))
    builder.Connect(zero_muscle_force.get_output_port(0), zft_ref_mass.GetInputPort("F"))
    
    # Impedance connections
    builder.Connect(ee_state_mux.get_output_port(0), impedance_force.GetInputPort("y_v"))
    builder.Connect(zft_ref_mass.GetOutputPort("yref_vref"), impedance_force.GetInputPort("yref_vref"))
    
    # Force routing
    builder.Connect(impedance_force.GetOutputPort("F_imp"), jacobian_controller.GetInputPort("F_imp"))
    builder.Connect(impedance_force.GetOutputPort("F_imp"), cart_force_applicator.GetInputPort("cart_force"))
    
    # Actuators
    builder.Connect(cart_force_applicator.GetOutputPort("spatial_forces"), 
                   plant.get_applied_spatial_force_input_port())
    builder.Connect(jacobian_controller.get_output_port(), plant.get_actuation_input_port())
    
    # Visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    ref_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(zft_ref_mass.GetOutputPort("yref_vref"), ref_logger.get_input_port())
    
    F_imp_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(impedance_force.GetOutputPort("F_imp"), F_imp_logger.get_input_port())
    
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
    
    print(colored(f"Simulating for {duration} s...", "yellow"))
    simulator.AdvanceTo(duration)
    print(colored("✓ Simulation complete\n", "green"))
    
    visualizer.PublishRecording()
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    ref_log = ref_logger.FindLog(context)
    F_imp_log = F_imp_logger.FindLog(context)
    
    t = state_log.sample_times()
    state_data = state_log.data()
    ee_pos = ee_pos_log.data()
    ref_data = ref_log.data()
    F_imp = F_imp_log.data()
    
    return {
        'time': t,
        'q1': state_data[0, :],
        'q2': state_data[1, :],
        'cart_x': state_data[2, :],
        'cart_y': state_data[3, :],
        'pend_alpha': state_data[4, :],
        'pend_beta': state_data[5, :],
        'cart_vx': state_data[8, :],
        'cart_vy': state_data[9, :],
        'ee_x': ee_pos[0, :],
        'ee_y': ee_pos[1, :],
        'ref_x': ref_data[0, :],
        'ref_y': ref_data[1, :],
        'ref_vx': ref_data[2, :],
        'ref_vy': ref_data[3, :],
        'F_x': F_imp[0, :],
        'F_y': F_imp[1, :],
    }


def simulate_2d_push_with_lqr(
    dx=0.3,
    dy=0.2,
    duration=5.0,
    K_imp=50.0,      # from script_cart
    D_imp=10.0,      # from script_cart
    M_ref=1.0,       # from script_cart (Mh)
    muscle_tau=0.03, # from script_cart
    horizon=10.0,
):
    """
    Simulate manipulator pushing cart in 2D using Finite-Horizon LQR control.
    
    ARCHITECTURE:
    ═════════════
    LQR → Muscle Dynamics (2D) → ZFT Reference → Impedance → Cart
    
    State: [x, y, θ, ẋ, ẏ, θ̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref] (12D)
    Control: [u_x, u_y] (2D neural commands)
    
    Args:
        dx, dy: Desired XY displacement [m]
        duration: Simulation duration [s]
        K_imp, D_imp: Impedance gains
        M_ref: Reference mass [kg]
        muscle_tau: Muscle time constant [s]
        horizon: LQR planning horizon [s]
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR PUSHES CART - 2D LQR CONTROL", "cyan", attrs=["bold"]))
    print(colored("Finite-Horizon LQR with Muscle Dynamics", "cyan"))
    print(colored("="*80, "cyan"))
    print(colored(f"Target displacement: ΔX = {dx:.3f} m, ΔY = {dy:.3f} m", "yellow"))
    print(colored(f"Duration: {duration:.1f} s", "yellow"))
    print(colored(f"LQR Horizon: {horizon:.1f} s", "yellow"))
    print(colored(f"Muscle τ: {muscle_tau:.3f} s", "yellow"))
    print(colored("="*80 + "\n", "cyan"))
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Build system
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Add manipulator
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),
        damping=(0.1, 0.1),
    )
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
    cart_config = create_cart_pendulum_config(
        cart_mass=5.0,
        cart_damping=2.0,
        pendulum_mass=0.5,
        pendulum_length=0.2,
    )
    cart_pendulum = CartPendulum3D(cart_config, visualize_cart=True, add_cart_actuators=False)
    cart_model = plant.AddModelInstance("cart_pendulum")
    cart_pendulum.attach_to_plant(plant, cart_model, register_visuals=False)
    
    plant.Finalize()
    
    # Get initial EE position
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, manipulator.model_instance, 
                      np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    ee_pos_init = manipulator.CalcPosition(plant, temp_context)[:2]
    
    print(colored(f"✓ System created", "green"))
    print(colored(f"  Initial EE: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m\n", "cyan"))
    
    # Build linearized system
    A, B = build_linearized_system_2d(
        plant, manipulator, cart_pendulum,
        K_imp, D_imp, M_ref, muscle_tau
    )
    
    print(colored(f"✓ Linearized system: A({A.shape[0]}x{A.shape[1]}), B({B.shape[0]}x{B.shape[1]})", "green"))
    
    # Define LQR cost matrices (from script_cart)
    # State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    # Note: Adapted from 7D cart-pendulum to 14D manipulator-cart
    Q = np.diag([
        100.0, 100.0,   # Cart position (x, y) - high cost for position tracking
        1000.0, 1000.0, # Pendulum angles (α, β) - very high cost to keep upright
        10.0, 10.0,     # Cart velocities (ẋ, ẏ)
        100.0, 100.0,   # Pendulum velocities (α̇, β̇) - high cost
        0.1, 0.1,       # Muscle forces (F_x, F_y) - low priority
        1.0, 1.0,       # Reference position (x_ref, y_ref)
        1.0, 1.0,       # Reference velocity (ẋ_ref, ẏ_ref)
    ])
    
    QN = Q.copy()  # Terminal cost same as running cost
    
    R = np.diag([1.0, 1.0])  # Control effort (from script_cart)
    
    # Goal state: cart at target position, rest at zero
    x_goal = np.array([
        ee_pos_init[0] + dx, ee_pos_init[1] + dy,  # Cart target (x, y)
        0.0, 0.0,                                   # Pendulums upright (α=0, β=0)
        0.0, 0.0, 0.0, 0.0,                        # Zero velocities (ẋ, ẏ, α̇, β̇)
        0.0, 0.0,                                   # Zero forces (F_x, F_y)
        ee_pos_init[0] + dx, ee_pos_init[1] + dy,  # Reference target (x_ref, y_ref)
        0.0, 0.0,                                   # Zero ref velocities (ẋ_ref, ẏ_ref)
    ])
    
    print(colored(f"✓ LQR goal: cart=({x_goal[0]:.3f}, {x_goal[1]:.3f})", "green"))
    
    # Create LQR controller
    lqr_controller = builder.AddSystem(
        FiniteHorizonLQRController2D(A, B, Q, QN, R, x_goal, horizon=horizon, timestep=0.01)
    )
    
    # Create muscle dynamics
    muscle_dynamics = builder.AddSystem(MuscleDynamics2D(muscle_tau=muscle_tau))
    
    # Create ZFT reference mass
    zft_config = create_zft_reference_mass_config(
        Mh=np.diag([M_ref, M_ref]),
        kp=np.diag([K_imp, K_imp]),
        kd=np.diag([D_imp, D_imp]),
        yref0=np.array([ee_pos_init[0], ee_pos_init[1]]),
        vref0=np.array([0.0, 0.0])
    )
    zft_ref_mass = builder.AddSystem(ZFTReferenceMass(zft_config))
    
    # Create impedance force
    imp_config = create_impedance_force_config(
        kp=np.diag([K_imp, K_imp]),
        kd=np.diag([D_imp, D_imp]),
        force_limit=None
    )
    impedance_force = builder.AddSystem(ImpedanceForce(imp_config))
    
    # Create controllers
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    jacobian_controller = builder.AddSystem(
        ManipulatorJacobianTransposeController2D(plant, manipulator)
    )
    cart_force_applicator = builder.AddSystem(
        CartForceApplicator2D(cart_pendulum.cart_body.index())
    )
    
    # Demux/Mux for state routing
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))  # [manip(4), cart(8)]
    ee_state_mux = builder.AddSystem(Multiplexer([2, 2]))   # [pos(2), vel(2)] -> state(4)
    
    # Extract cart state: [x, y, α, β, ẋ, ẏ, α̇, β̇] (8D)
    # We need: cart_pos(2), pend_angles(2), cart_vel(2), pend_vel(2)
    cart_state_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, angles, vel, ang_vel]
    
    # Extract ZFT state: [x_ref, y_ref, ẋ_ref, ẏ_ref] (4D)
    zft_state_demux = builder.AddSystem(Demultiplexer([2, 2]))  # [pos_ref, vel_ref]
    
    # Create state assembler for 14D LQR state
    # State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    # Keep BOTH pendulum angles (α and β)
    
    state_assembler = builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 2, 2]))  # 14D total
    
    # Wire basic connections
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    
    # Extract cart state components
    builder.Connect(state_demux.get_output_port(1), cart_state_demux.get_input_port())
    
    # Extract ZFT state components
    builder.Connect(zft_ref_mass.get_output_port(), zft_state_demux.get_input_port())
    
    # Assemble 14D state for LQR
    # Input port order: [cart_pos(2), pend_angles(2), cart_vel(2), pend_vel(2), muscle_F(2), ref_pos(2), ref_vel(2)]
    builder.Connect(cart_state_demux.get_output_port(0), state_assembler.get_input_port(0))  # cart pos [x, y]
    builder.Connect(cart_state_demux.get_output_port(1), state_assembler.get_input_port(1))  # pend angles [α, β]
    builder.Connect(cart_state_demux.get_output_port(2), state_assembler.get_input_port(2))  # cart vel [ẋ, ẏ]
    builder.Connect(cart_state_demux.get_output_port(3), state_assembler.get_input_port(3))  # pend vel [α̇, β̇]
    builder.Connect(muscle_dynamics.get_output_port(), state_assembler.get_input_port(4))     # muscle F [F_x, F_y]
    builder.Connect(zft_state_demux.get_output_port(0), state_assembler.get_input_port(5))    # ref pos [x_ref, y_ref]
    builder.Connect(zft_state_demux.get_output_port(1), state_assembler.get_input_port(6))    # ref vel [ẋ_ref, ẏ_ref]
    
    # Wire LQR control loop: state → LQR → u → muscle
    builder.Connect(state_assembler.get_output_port(), lqr_controller.get_input_port(0))
    builder.Connect(lqr_controller.get_output_port(), muscle_dynamics.get_input_port(0))
    
    # Muscle → ZFT
    builder.Connect(muscle_dynamics.get_output_port(), zft_ref_mass.get_input_port(1))
    
    # EE kinematics → ZFT
    builder.Connect(state_demux.get_output_port(0), ee_kinematics.GetInputPort("manipulator_state"))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_state_mux.get_input_port(0))
    builder.Connect(ee_kinematics.GetOutputPort("ee_velocity"), ee_state_mux.get_input_port(1))
    builder.Connect(ee_state_mux.get_output_port(0), zft_ref_mass.get_input_port(0))
    
    # ZFT + EE → Impedance
    builder.Connect(ee_state_mux.get_output_port(0), impedance_force.get_input_port(0))
    builder.Connect(zft_ref_mass.get_output_port(), impedance_force.get_input_port(1))
    
    # Impedance → Jacobian controller → Manipulator
    builder.Connect(impedance_force.get_output_port(), jacobian_controller.get_input_port(0))
    builder.Connect(state_demux.get_output_port(0), jacobian_controller.get_input_port(1))
    builder.Connect(jacobian_controller.get_output_port(), plant.get_actuation_input_port())
    
    # Impedance → Cart force
    builder.Connect(impedance_force.get_output_port(), cart_force_applicator.get_input_port(0))
    builder.Connect(cart_force_applicator.get_output_port(),
                   plant.get_applied_spatial_force_input_port())
    
    print(colored("✓ LQR Controller: 14D state → 2D control", "green"))
    print(colored("  State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]", "cyan"))
    
    # Visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    ref_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(zft_ref_mass.GetOutputPort("yref_vref"), ref_logger.get_input_port())
    
    F_imp_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(impedance_force.GetOutputPort("F_imp"), F_imp_logger.get_input_port())
    
    muscle_F_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(muscle_dynamics.get_output_port(), muscle_F_logger.get_input_port())
    
    lqr_u_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(lqr_controller.get_output_port(), lqr_u_logger.get_input_port())
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, np.array([
        np.deg2rad(-10.0), np.deg2rad(20.0),  # Manipulator
        ee_pos_init[0], ee_pos_init[1],        # Cart at EE
        0.0, 0.0                               # Pendulum (α=0, β=0)
    ]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    visualizer.StartRecording()
    
    print(colored(f"Simulating for {duration} s...", "yellow"))
    simulator.AdvanceTo(duration)
    print(colored("✓ Simulation complete\n", "green"))
    
    visualizer.PublishRecording()
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    ref_log = ref_logger.FindLog(context)
    F_imp_log = F_imp_logger.FindLog(context)
    muscle_F_log = muscle_F_logger.FindLog(context)
    lqr_u_log = lqr_u_logger.FindLog(context)
    
    t = state_log.sample_times()
    state_data = state_log.data()
    ee_pos = ee_pos_log.data()
    ref_data = ref_log.data()
    F_imp = F_imp_log.data()
    muscle_F = muscle_F_log.data()
    lqr_u = lqr_u_log.data()
    
    print(colored("✓ Data extraction complete", "green"))
    print(colored("="*80 + "\n", "cyan"))
    
    return {
        'time': t,
        'q1': state_data[0, :],
        'q2': state_data[1, :],
        'cart_x': state_data[2, :],
        'cart_y': state_data[3, :],
        'pend_alpha': state_data[4, :],
        'pend_beta': state_data[5, :],
        'cart_vx': state_data[8, :],
        'cart_vy': state_data[9, :],
        'ee_x': ee_pos[0, :],
        'ee_y': ee_pos[1, :],
        'ref_x': ref_data[0, :],
        'ref_y': ref_data[1, :],
        'ref_vx': ref_data[2, :],
        'ref_vy': ref_data[3, :],
        'F_x': F_imp[0, :],
        'F_y': F_imp[1, :],
        'muscle_Fx': muscle_F[0, :],
        'muscle_Fy': muscle_F[1, :],
        'lqr_ux': lqr_u[0, :],
        'lqr_uy': lqr_u[1, :],
    }


def plot_results_2d(data, dx_target, dy_target):
    """Plot 2D simulation results."""
    print(colored("📈 Generating plots...", "yellow"))
    
    t = data['time']
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Joint angles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, np.rad2deg(data['q1']), 'b-', linewidth=2, label='q₁')
    ax1.plot(t, np.rad2deg(data['q2']), 'r-', linewidth=2, label='q₂')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
    ax1.set_title('Manipulator Joint Configuration', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: X positions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, data['ref_x'], 'r-.', linewidth=2, alpha=0.7, label='x_ref (ZFT)')
    ax2.plot(t, data['ee_x'], 'b-', linewidth=2.5, label='x_ee')
    ax2.plot(t, data['cart_x'], 'g:', linewidth=2.5, label='x_cart')
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('X Position [m]', fontweight='bold')
    ax2.set_title('X Motion (2D Generalized)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Y positions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, data['ref_y'], 'r-.', linewidth=2, alpha=0.7, label='y_ref (ZFT)')
    ax3.plot(t, data['ee_y'], 'b-', linewidth=2.5, label='y_ee')
    ax3.plot(t, data['cart_y'], 'g:', linewidth=2.5, label='y_cart')
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Y Position [m]', fontweight='bold')
    ax3.set_title('Y Motion (2D Generalized)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Forces (Impedance or Muscle depending on mode)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'muscle_Fx' in data:
        # LQR mode: show muscle forces
        F_mag = np.sqrt(data['muscle_Fx']**2 + data['muscle_Fy']**2)
        ax4.plot(t, data['muscle_Fx'], 'b-', linewidth=2, label='F_muscle,x')
        ax4.plot(t, data['muscle_Fy'], 'r-', linewidth=2, label='F_muscle,y')
        ax4.plot(t, F_mag, 'k--', linewidth=1.5, label='||F_muscle||')
        ax4.set_title('2D Muscle Forces (LQR)', fontweight='bold')
    else:
        # Impedance mode: show impedance forces
        F_mag = np.sqrt(data['F_x']**2 + data['F_y']**2)
        ax4.plot(t, data['F_x'], 'b-', linewidth=2, label='F_imp,x')
        ax4.plot(t, data['F_y'], 'r-', linewidth=2, label='F_imp,y')
        ax4.plot(t, F_mag, 'k--', linewidth=1.5, label='||F_imp||')
        ax4.set_title('2D Impedance Force Components', fontweight='bold')
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time [s]', fontweight='bold')
    ax4.set_ylabel('Force [N]', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: 2D Trajectory
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(data['ref_x'], data['ref_y'], 'r-.', linewidth=2.5, alpha=0.9, label='Reference (ZFT)')
    ax5.plot(data['ee_x'], data['ee_y'], 'b-', linewidth=2.5, alpha=0.8, label='End Effector')
    ax5.plot(data['cart_x'], data['cart_y'], 'g:', linewidth=2.5, alpha=0.6, label='Cart')
    ax5.plot(data['ee_x'][0], data['ee_y'][0], 'go', markersize=12, label='Start')
    ax5.plot(data['ee_x'][-1], data['ee_y'][-1], 'ro', markersize=12, label='End')
    ax5.set_xlabel('X Position [m]', fontweight='bold')
    ax5.set_ylabel('Y Position [m]', fontweight='bold')
    ax5.set_title('2D Trajectory (X-Y Plane)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.legend()
    
    # Plot 6: Cart velocities
    ax6 = fig.add_subplot(gs[1, 2])
    v_mag = np.sqrt(data['cart_vx']**2 + data['cart_vy']**2)
    ax6.plot(t, data['cart_vx'], 'b-', linewidth=2, label='vₓ')
    ax6.plot(t, data['cart_vy'], 'r-', linewidth=2, label='vᵧ')
    ax6.plot(t, v_mag, 'k--', linewidth=1.5, label='||v||')
    ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Time [s]', fontweight='bold')
    ax6.set_ylabel('Velocity [m/s]', fontweight='bold')
    ax6.set_title('Cart Velocities (2D)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Joint space trajectory
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(np.rad2deg(data['q1']), np.rad2deg(data['q2']), 'purple', linewidth=2)
    ax7.plot(np.rad2deg(data['q1'][0]), np.rad2deg(data['q2'][0]), 
             'go', markersize=12, label='Initial')
    ax7.plot(np.rad2deg(data['q1'][-1]), np.rad2deg(data['q2'][-1]), 
             'ro', markersize=12, label='Final')
    ax7.set_xlabel('q₁ [deg]', fontweight='bold')
    ax7.set_ylabel('q₂ [deg]', fontweight='bold')
    ax7.set_title('Joint Space Trajectory', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Reference velocity
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, data['ref_vx'], 'b-', linewidth=2, label='v_ref,x')
    ax8.plot(t, data['ref_vy'], 'r-', linewidth=2, label='v_ref,y')
    ax8.set_xlabel('Time [s]', fontweight='bold')
    ax8.set_ylabel('Velocity [m/s]', fontweight='bold')
    ax8.set_title('ZFT Reference Velocity', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Plot 9: Force vector field or LQR neural commands
    ax9 = fig.add_subplot(gs[2, 2])
    if 'lqr_ux' in data:
        # LQR mode: show neural commands
        u_mag = np.sqrt(data['lqr_ux']**2 + data['lqr_uy']**2)
        ax9.plot(t, data['lqr_ux'], 'b-', linewidth=2, label='u_x')
        ax9.plot(t, data['lqr_uy'], 'r-', linewidth=2, label='u_y')
        ax9.plot(t, u_mag, 'k--', linewidth=1.5, label='||u||')
        ax9.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax9.set_xlabel('Time [s]', fontweight='bold')
        ax9.set_ylabel('Command [N]', fontweight='bold')
        ax9.set_title('LQR Neural Commands u(t)', fontweight='bold')
        ax9.legend()
    else:
        # Impedance mode: show force vector field
        skip = max(1, len(t) // 20)  # Sample every Nth point
        ax9.quiver(data['ee_x'][::skip], data['ee_y'][::skip], 
                  data['F_x'][::skip], data['F_y'][::skip],
                  color='blue', alpha=0.6, scale=500)
        ax9.plot(data['ee_x'], data['ee_y'], 'r-', linewidth=1, alpha=0.3)
        ax9.set_xlabel('X Position [m]', fontweight='bold')
        ax9.set_ylabel('Y Position [m]', fontweight='bold')
        ax9.set_title('Force Vector Field', fontweight='bold')
        ax9.axis('equal')
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Pendulum angles
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, np.rad2deg(data['pend_alpha']), 'b-', linewidth=2, label='α (Pitch)')
    ax10.plot(t, np.rad2deg(data['pend_beta']), 'r-', linewidth=2, label='β (Roll)')
    ax10.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax10.set_xlabel('Time [s]', fontweight='bold')
    ax10.set_ylabel('Angle [deg]', fontweight='bold')
    ax10.set_title('Pendulum Angles', fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    
    # Plot 11: Cart displacement
    ax11 = fig.add_subplot(gs[3, 1])
    dx_actual = data['cart_x'] - data['cart_x'][0]
    dy_actual = data['cart_y'] - data['cart_y'][0]
    ax11.plot(t, dx_actual, 'b-', linewidth=2, label='ΔX (actual)')
    ax11.plot(t, dy_actual, 'r-', linewidth=2, label='ΔY (actual)')
    ax11.axhline(dx_target, color='b', linestyle='--', alpha=0.5, label=f'ΔX target={dx_target:.2f}')
    ax11.axhline(dy_target, color='r', linestyle='--', alpha=0.5, label=f'ΔY target={dy_target:.2f}')
    ax11.set_xlabel('Time [s]', fontweight='bold')
    ax11.set_ylabel('Displacement [m]', fontweight='bold')
    ax11.set_title('Cart Displacement vs Target', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    # Plot 12: Summary
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    
    mode_str = "LQR Control" if 'lqr_ux' in data else "Impedance Control"
    max_force = np.max(np.sqrt(data['muscle_Fx']**2 + data['muscle_Fy']**2)) if 'muscle_Fx' in data else np.max(np.sqrt(data['F_x']**2 + data['F_y']**2))
    
    summary_text = f"""
2D MOTION SUMMARY
Mode: {mode_str}

Target Displacement:
  ΔX_target = {dx_target:6.3f} m
  ΔY_target = {dy_target:6.3f} m

Actual Displacement:
  ΔX_actual = {dx_actual[-1]:6.3f} m
  ΔY_actual = {dy_actual[-1]:6.3f} m

Error:
  ΔX_error  = {dx_actual[-1] - dx_target:6.3f} m
  ΔY_error  = {dy_actual[-1] - dy_target:6.3f} m

Final Configuration:
  q₁ = {np.rad2deg(data['q1'][-1]):6.2f}°
  q₂ = {np.rad2deg(data['q2'][-1]):6.2f}°

Final Velocities:
  vₓ = {data['cart_vx'][-1]:5.3f} m/s
  vᵧ = {data['cart_vy'][-1]:5.3f} m/s

Max Force:
  F_max = {max_force:5.1f} N

Duration: {t[-1]:.1f} s
    """
    ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    title_suffix = "(LQR Optimal Control)" if 'lqr_ux' in data else "(Generalized from 1D to Full X-Y Motion)"
    plt.suptitle(f'Manipulator Pushes Cart - 2D Control\n{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(colored("✓ Plots generated\n", "green"))


def main():
    # Use pre-parsed arguments
    args = _ARGS
    
    # Run simulation based on mode
    if args.mode == 'lqr':
        print(colored(f"\n🎯 Running with Finite-Horizon LQR Control", "magenta", attrs=["bold"]))
        data = simulate_2d_push_with_lqr(
            dx=args.dx,
            dy=args.dy,
            duration=args.duration,
            K_imp=args.K,
            D_imp=args.D,
            M_ref=args.M,
            horizon=args.horizon,
        )
    else:
        print(colored(f"\n🎯 Running with Direct Impedance Control", "magenta", attrs=["bold"]))
        data = simulate_2d_push(
            dx=args.dx,
            dy=args.dy,
            duration=args.duration,
            K_imp=args.K,
            D_imp=args.D,
            M_ref=args.M,
        )
    
    # Wait for user before plotting
    if data is not None:
        input(colored("\nPress Enter to generate plots...", "yellow"))
        
        # Plot results
        plot_results_2d(data, args.dx, args.dy)
    else:
        print(colored("\n⚠ LQR simulation returned no data (implementation incomplete)", "yellow"))
    
    print(colored("\n" + "="*80, "cyan"))
    print(colored("Execution Complete!", "green", attrs=["bold"]))
    print(colored("="*80 + "\n", "cyan"))


if __name__ == "__main__":
    main()
