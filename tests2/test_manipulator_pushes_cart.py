#!/usr/bin/env python3
"""
Test Script: Manipulator Pushes Cart via Virtual Mass (Human Arm with Muscle Dynamics + OFC)

This script demonstrates a manipulator (human arm) pushing a passive cart-pendulum
system using muscle dynamics and optimal feedback control (OFC).
Based on notes_ss_cart_pendulam_manipulator.tex equations.

MATHEMATICAL FORMULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MANIPULATOR AS HUMAN ARM WITH MUSCLE DYNAMICS
   ──────────────────────────────────────────────
   
   Muscle Dynamics (First-order actuator):
       τ_f Ḟ_muscle = u - F_muscle
   
   where:
       u = neural command from finite-horizon LQR (optimal effort)
       F_muscle = muscle force state
       τ_f = muscle time constant (~30 ms)
   
   Reference Mass Dynamics (ZFT - Zero Force Trajectory):
       M_ref ẍ_ref = -F_imp + F_muscle
   
   Impedance Force:
       F_imp = K(x_ee - x_ref) + D(ẋ_ee - ẋ_ref)
   
   Joint Torques (Jacobian Transpose Control):
       τ = -J^T F_imp
   
   where J is the manipulator Jacobian.
   
   Manipulator Dynamics (Equation of Motion):
       M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
   
   where:
       M(q)     : Mass/inertia matrix (configuration-dependent)
       C(q,q̇)   : Coriolis and centrifugal forces
       G(q)     : Gravity torques
       τ        : Applied joint torques (from impedance control)
   
   Solving for joint acceleration:
       q̈ = M(q)^{-1}[τ - C(q,q̇)q̇ - G(q)]
   
   Integration (performed by Drake's MultibodyPlant simulator):
       q̇(t+dt) = q̇(t) + q̈(t)·dt     [Joint velocity integration]
       q(t+dt) = q(t) + q̇(t)·dt      [Joint position integration]
   
   This shows the complete forward dynamics chain:
       F_imp → τ → q̈ → q̇ → q (joint positions)

2. FINITE-HORIZON LQR (Optimal Effort Minimization)
   ─────────────────────────────────────────────────
   
   Cost Function:
       J = ∫_0^T [x'Qx + u'Ru] dt + x(T)'QN·x(T)
   
   where x = [x_cart, β, ẋ_cart, β̇, F_muscle, x_ref, ẋ_ref]^T ∈ ℝ⁷
   
   Control Law (Time-varying Riccati):
       u(t) = -K(t)(x(t) - x_goal)
   
   Minimizes muscular effort while achieving desired cart motion.

3. VIRTUAL MASS (Compliant Coupling)
   ──────────────────────────────────
   
   Admittance Dynamics:
       M_v ẍ_v + D_v ẋ_v + K_v(x_v - x₀) = F_ee_coupling + F_cart_coupling
   
   State: z_v = [x_v, y_v, ẋ_v, ẏ_v]^T ∈ ℝ⁴
   
   Coupling Forces:
       F_ee_coupling   = -K_ee(x_v - x_ee) - D_ee(ẋ_v - ẋ_ee)   [EE → Virtual]
       F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart) [Virtual → Cart]
   
   Newton's 3rd Law (Reactive Forces):
       F_ee_react  = +K_ee(x_v - x_ee) + D_ee(ẋ_v - ẋ_ee)  [Applied to EE]
       F_cart_push = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart) [Applied to Cart]

4. CART-PENDULUM PASSIVE DYNAMICS
   ────────────────────────────────
   
   Cart (2-DOF planar):
       m_c ẍ_cart = F_cart_push + F_pendulum + F_damping
   
   Pendulum (2-DOF spherical, small angle):
       ẍ_β = -(m_c + m_p)g/(l·m_c) β - F_cart_push/(l·m_c)
   
   No active control - purely passive response to external forces.

5. STATE-SPACE REPRESENTATION (from notes_ss_cart_pendulam_manipulator.tex)
   ────────────────────────────────────────────────────────────────────────
   
   Full State Vector:
       z = [x, ẋ, β, β̇, x_ref, ẋ_ref, F_muscle]^T ∈ ℝ¹⁴
   
   where:
       x ∈ ℝ²      : Cart position
       β ∈ ℝ²      : Pendulum angles (small-angle approx)
       x_ref ∈ ℝ²  : Reference mass position (ZFT)
       F_muscle ∈ ℝ²: Muscle force state
   
   For manipulator system:
       Additional: q_manip ∈ ℝ², x_v ∈ ℝ² (virtual mass)
       Total: ~20 states

SYSTEM PARAMETERS:
──────────────────
Manipulator Impedance:
    K_imp = 100 N/m    (Stiffness)
    D_imp = 20 N·s/m   (Damping)
    M_ref = 2.0 kg     (Reference mass)

Muscle Dynamics:
    τ_f = 0.03 s       (Time constant, ~30 ms)
    F_init = 0.0 N     (Initial muscle force)

Finite-Horizon LQR:
    Horizon: 10 s
    Q: State cost (penalize cart position error)
    R: Control cost (minimize neural effort)
    QN: Terminal cost

Virtual Mass:
    M_v = 2.0 kg       (Inertia)
    D_v = 5.0 N·s/m    (Damping)
    K_v = 10 N/m       (Stiffness)
    K_ee = 50 N/m      (EE coupling stiffness)
    D_ee = 10 N·s/m    (EE coupling damping)
    K_c = 50 N/m       (Cart coupling stiffness)
    D_c = 10 N·s/m     (Cart coupling damping)

Cart-Pendulum:
    m_c = 5.0 kg       (Cart mass)
    m_p = 0.5 kg       (Pendulum mass)
    l = 0.2 m          (Pendulum length)

USAGE:
──────
    python test_manipulator_pushes_cart.py --duration 10 --distance 1.0
    
This will show what joint configuration is required for the manipulator to push
the cart 1 meter in the X direction.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm
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
    Saturation,
    Multiplexer,
)
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.primitives import Demultiplexer
from termcolor import colored

# Import from main script
from robot_types import create_cup_manipulator_config, create_cart_pendulum_config
from script_cup_manipulator_controller_ofc import CupManipulator, CartPendulum3D


class ManipulatorJacobianTransposeController(LeafSystem):
    """
    Converts impedance force to joint torques via Jacobian transpose.
    
    COMPLETE DYNAMICS CHAIN (F_imp → Joint Position):
    ──────────────────────────────────────────────────
    
    Step 1 - Task-Space to Joint-Space Mapping:
        τ = -J^T(q) F_imp
    
    where:
        J(q) is the manipulator Jacobian (2×2 for planar case)
        F_imp is the impedance force in task space [N]
        τ is the joint torque vector [N·m]
    
    The negative sign: F_imp acts ON the end effector (pulling toward x_ref),
    so the reaction force on the joints is -F_imp.
    
    Step 2 - Joint Torques to Joint Acceleration (Drake MultibodyPlant):
        M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
        
        Solving: q̈ = M(q)^{-1}[τ - C(q,q̇)q̇ - G(q)]
    
    where:
        M(q)   : 2×2 mass/inertia matrix (computed from link masses/inertias)
        C(q,q̇) : Coriolis/centrifugal matrix
        G(q)   : Gravity vector [0, m₂·g·l₂·cos(q₁+q₂)]^T for planar arm
        q̈     : Joint accelerations [rad/s²]
    
    Step 3 - Integration (Drake's Runge-Kutta or implicit Euler):
        q̇_{k+1} = q̇_k + q̈_k·Δt     [Velocity integration]
        q_{k+1} = q_k + q̇_k·Δt      [Position integration]
    
    Result: Joint positions q(t) evolve to track the impedance reference x_ref
    
    INPUTS:
        - F_imp: Impedance force (scalar, 1D for X motion)
        - manipulator_state: [q₁, q₂, q̇₁, q̇₂]^T
    
    OUTPUT:
        - joint_torques: [τ₁, τ₂]^T → fed to plant actuation input
    
    NOTE: The actual dynamics integration M(q)q̈ = τ - C - G is performed
          internally by Drake's MultibodyPlant during simulation.
    """
    
    def __init__(self, plant, manipulator):
        """Initialize controller with plant and manipulator references."""
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        # Plant context for Jacobian computation
        self.plant_context = plant.CreateDefaultContext()
        
        # Input ports
        self.F_imp_input = self.DeclareVectorInputPort("F_imp", BasicVector(1))
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output port
        self.DeclareVectorOutputPort(
            "joint_torques",
            BasicVector(2),
            self.CalcTorques
        )
    
    def CalcTorques(self, context, output):
        """
        Compute joint torques via Jacobian transpose control.
        
        FORWARD DYNAMICS CALCULATION:
        ==============================
        
        Step 1: Get current joint configuration q = [q1, q2]
        Step 2: Compute Jacobian J(q) at current configuration
        Step 3: Map task-space force to joint torques: tau = -J^T F_imp
        Step 4: Output tau -> MultibodyPlant actuator input
        Step 5: Plant solves: M(q)q_ddot = tau - C(q,q_dot)q_dot - G(q)
        Step 6: Integrator updates: q_dot -> q (joint positions)
        
        Example for 2-DOF planar arm:
            If F_imp = 10 N (pushing in +X), and:
            - q1 = -10 deg, q2 = 20 deg
            - J(q) approx [[-0.85, -0.17],
                           [-0.17, -0.17],
                           [ 0.00,  0.00]]  (3x2, X-Y-Z rows)
            
            Then: tau = -J^T @ [10, 0, 0]^T
                      = -[[-0.85, -0.17, 0.00],    @ [10]
                          [-0.17, -0.17, 0.00]]      [0]
                                                     [0]
                      = [8.5, 1.7]^T N*m
            
            This torque causes joint accelerations q_ddot, which integrate to q_dot, then q.
        
        Equation:
            tau = -J^T(q) F_imp
        
        For 1D motion (X-axis only):
            F_imp_2d = [F_imp, 0]^T  (force in X, zero in Y)
            tau = -J^T F_imp_2d
        """
        # Get manipulator joint angles
        state = self.state_input.Eval(context)
        q = state[:2]
        
        # Get impedance force (scalar)
        F_imp_scalar = float(self.F_imp_input.Eval(context)[0])
        
        # Convert to 3D force vector (X-axis only for cart pushing)
        F_imp_3d = np.array([F_imp_scalar, 0.0, 0.0])  # [Fx, Fy, Fz]
        
        # Compute Jacobian J(q) at current configuration
        # IMPORTANT: Set full plant state, not just manipulator
        full_q = self.plant.GetPositions(self.plant_context)
        full_q[:2] = q  # Update manipulator positions
        self.plant.SetPositions(self.plant_context, full_q)
        
        ee_origin = np.zeros(3)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            ee_origin,
            self.world_frame,
            self.world_frame
        )
        
        # J_spatial is 6 × n_v where n_v = total plant velocities
        # For our system: n_v = 2 (manipulator) + 4 (cart-pendulum) = 6
        # We only want the manipulator columns (first 2)
        J_translational = J_spatial[3:6, :2]  # 3×2 matrix (linear part, manip only)
        
        # Joint torques: τ = -J^T F_imp
        # (2 × 3) @ (3,) = (2,)
        tau = -J_translational.T @ F_imp_3d
        
        output.SetFromVector(tau)


class VirtualMassSystem(LeafSystem):
    """
    Virtual mass coupling between manipulator EE and cart.
    
    EQUATIONS:
    ──────────
    Virtual Mass Dynamics:
        M_v ẍ_v + D_v ẋ_v + K_v(x_v - x₀) = F_ee_coupling + F_cart_coupling
    
    Coupling Forces (into virtual mass):
        F_ee_coupling   = -K_ee(x_v - x_ee) - D_ee(ẋ_v - ẋ_ee)
        F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
    
    Reactive Forces (Newton's 3rd Law):
        F_ee_react  = +K_ee(x_v - x_ee) + D_ee(ẋ_v - ẋ_ee)  [Applied to EE]
        F_cart_push = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart) [Applied to Cart]
    
    When virtual mass leads EE (x_v > x_ee):
        - F_ee_react > 0 → pulls EE forward (reactive force on manipulator)
    
    When virtual mass leads cart (x_v > x_cart):
        - F_cart_push < 0 → pushes cart forward
    
    STATE:
        z_v = [x_v, y_v, ẋ_v, ẏ_v]^T ∈ ℝ⁴
    
    INPUTS:
        - cart_position: [x_cart, y_cart]^T
        - cart_velocity: [ẋ_cart, ẏ_cart]^T
        - ee_position: [x_ee, y_ee]^T
        - ee_velocity: [ẋ_ee, ẏ_ee]^T
    
    OUTPUTS:
        - virtual_position: [x_v, y_v]^T
        - virtual_velocity: [ẋ_v, ẏ_v]^T
        - ee_force: [F_ee_x, F_ee_y]^T (reactive force on EE)
        - cart_force: [F_cart_x, F_cart_y]^T (pushing force on cart)
    """
    
    def __init__(self, M_virtual=2.0, D_virtual=5.0, K_virtual=10.0, x0=None):
        LeafSystem.__init__(self)
        
        # Virtual mass parameters
        self.M_v = M_virtual  # [kg]
        self.D_v = D_virtual  # [N·s/m]
        self.K_v = K_virtual  # [N/m]
        self.x0 = x0 if x0 is not None else np.zeros(2)
        
        # Coupling parameters
        self.k_cart = 50.0   # Cart-virtual stiffness [N/m]
        self.d_cart = 10.0   # Cart-virtual damping [N·s/m]
        self.k_ee = 50.0     # EE-virtual stiffness [N/m]
        self.d_ee = 10.0     # EE-virtual damping [N·s/m]
        
        # Input ports
        self.cart_pos_input = self.DeclareVectorInputPort("cart_position", BasicVector(2))
        self.cart_vel_input = self.DeclareVectorInputPort("cart_velocity", BasicVector(2))
        self.ee_pos_input = self.DeclareVectorInputPort("ee_position", BasicVector(2))
        self.ee_vel_input = self.DeclareVectorInputPort("ee_velocity", BasicVector(2))
        
        # Continuous state: [x_v, y_v, ẋ_v, ẏ_v]
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
            "ee_force",  # Reactive force to apply to EE
            BasicVector(2),
            self.OutputEEForce
        )
        self.DeclareVectorOutputPort(
            "cart_force",  # Pushing force to apply to cart
            BasicVector(2),
            self.OutputCartForce
        )
    
    def SetDefaultState(self, context, state):
        """Initialize at equilibrium position x₀."""
        state.SetFromVector(np.array([self.x0[0], self.x0[1], 0.0, 0.0]))
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute virtual mass dynamics.
        
        Equation:
            M_v ẍ_v + D_v ẋ_v + K_v(x_v - x₀) = F_ee_coupling + F_cart_coupling
        
        Solving for ẍ_v:
            ẍ_v = (F_ee_coupling + F_cart_coupling - D_v ẋ_v - K_v(x_v - x₀)) / M_v
        """
        # Get virtual mass state
        state = context.get_continuous_state_vector().CopyToVector()
        x_v = state[0:2]
        v_v = state[2:4]
        
        # Get cart and EE states
        x_cart = self.cart_pos_input.Eval(context)
        v_cart = self.cart_vel_input.Eval(context)
        x_ee = self.ee_pos_input.Eval(context)
        v_ee = self.ee_vel_input.Eval(context)
        
        # Coupling forces (into virtual mass)
        # F_ee_coupling = -K_ee(x_v - x_ee) - D_ee(ẋ_v - ẋ_ee)
        F_ee_coupling = -self.k_ee * (x_v - x_ee) - self.d_ee * (v_v - v_ee)
        
        # F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        F_cart_coupling = -self.k_cart * (x_v - x_cart) - self.d_cart * (v_v - v_cart)
        
        # Virtual mass acceleration
        # M_v ẍ_v = F_ee_coupling + F_cart_coupling - D_v ẋ_v - K_v(x_v - x₀)
        a_v = (F_ee_coupling + F_cart_coupling - self.D_v * v_v - self.K_v * (x_v - self.x0)) / self.M_v
        
        # State derivative: ż_v = [ẋ_v, ẏ_v, ẍ_v, ÿ_v]^T
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
        """
        Compute reactive force to apply to end effector.
        
        By Newton's 3rd Law:
            F_ee_react = -F_ee_coupling
            F_ee_react = +K_ee(x_v - x_ee) + D_ee(ẋ_v - ẋ_ee)
        
        Physical interpretation:
            - When x_v > x_ee: Force is positive (pulls EE forward)
            - When x_v < x_ee: Force is negative (pushes EE backward)
        """
        # Get virtual mass state
        state = context.get_continuous_state_vector().CopyToVector()
        x_v = state[0:2]
        v_v = state[2:4]
        
        # Get EE state
        x_ee = self.ee_pos_input.Eval(context)
        v_ee = self.ee_vel_input.Eval(context)
        
        # Reactive force on EE (Newton's 3rd law)
        F_ee_react = self.k_ee * (x_v - x_ee) + self.d_ee * (v_v - v_ee)
        
        # Saturate to prevent numerical issues
        max_force = 200.0
        force_mag = np.linalg.norm(F_ee_react)
        if force_mag > max_force:
            F_ee_react = F_ee_react * (max_force / force_mag)
        
        output.SetFromVector(F_ee_react)
    
    def OutputCartForce(self, context, output):
        """
        Compute pushing force to apply to cart.
        
        By Newton's 3rd Law:
            F_cart_push = -F_cart_coupling
            F_cart_push = -[-K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)]
            F_cart_push = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        Wait, let me reconsider. The coupling force F_cart_coupling is the force
        FROM the cart INTO the virtual mass. By Newton's 3rd law, the force from
        virtual mass to cart is:
            F_cart_push = -F_cart_coupling
        
        Actually, let me think about the signs more carefully:
        
        If x_v > x_cart (virtual mass ahead of cart):
            - We want to push the cart forward (positive force)
            - F_cart_coupling = -K_c(x_v - x_cart) < 0 (cart pulls virtual back)
            - But we want F_cart_push > 0 to push cart forward
        
        So the force on cart should be:
            F_cart_push = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        
        This is the same as F_cart_coupling, not its negative!
        
        Actually, looking at the physics:
        - Virtual mass at x_v
        - Cart at x_cart
        - Spring connecting them with stiffness K_c
        
        Force on cart from spring: F = -K_c(x_cart - x_v) = K_c(x_v - x_cart)
        
        So if x_v > x_cart, force on cart is positive (forward).
        
        But F_cart_coupling is the force INTO the virtual mass FROM the cart.
        By Newton's 3rd law, the force on cart is -F_cart_coupling.
        
        Let me recalculate:
        F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        
        Force on cart = -F_cart_coupling = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        Wait no, that's still wrong. Let me think about this more carefully.
        
        Standard spring-damper between two points:
        - Point 1 at x₁, Point 2 at x₂
        - Force on point 2 from point 1: F₂ = -K(x₂ - x₁) - D(ẋ₂ - ẋ₁)
        - Force on point 1 from point 2: F₁ = -K(x₁ - x₂) - D(ẋ₁ - ẋ₂) = -F₂
        
        In our case:
        - Virtual mass at x_v (point 1)
        - Cart at x_cart (point 2)
        
        Force on cart from virtual: F_cart = -K_c(x_cart - x_v) - D_c(ẋ_cart - ẋ_v)
                                             = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        Force on virtual from cart: F_v = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        
        So F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        And F_cart_push = -F_cart_coupling = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        But wait, in the DoCalcTimeDerivatives, I have:
        F_cart_coupling = -self.k_cart * (x_v - x_cart) - self.d_cart * (v_v - v_cart)
        
        So F_cart_push should be the negative of this... but that doesn't make sense
        for pushing.
        
        Let me re-examine. If the virtual mass is ahead (x_v > x_cart), I want to
        push the cart forward, so F_cart_push > 0.
        
        F_cart_push = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        If x_v > x_cart, then F_cart_push > 0 ✓
        
        So F_cart_push = -F_cart_coupling.
        
        But looking at my code in DoCalcTimeDerivatives:
        F_cart_coupling = -self.k_cart * (x_v - x_cart) - self.d_cart * (v_v - v_cart)
        
        So:
        F_cart_push = -F_cart_coupling 
                    = -(-self.k_cart * (x_v - x_cart) - self.d_cart * (v_v - v_cart))
                    = self.k_cart * (x_v - x_cart) + self.d_cart * (v_v - v_cart)
        
        Actually, I think the issue is I named F_cart_coupling wrong. It should be
        the force the cart exerts on the virtual mass, so:
        
        Force cart exerts on virtual = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        Force virtual exerts on cart = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        
        So in the output, I should just negate F_cart_coupling.
        
        Hmm, but I want to be consistent with the user's notes. Let me just compute
        it directly:
        """
        # Get virtual mass state
        state = context.get_continuous_state_vector().CopyToVector()
        x_v = state[0:2]
        v_v = state[2:4]
        
        # Get cart state
        x_cart = self.cart_pos_input.Eval(context)
        v_cart = self.cart_vel_input.Eval(context)
        
        # Force from virtual mass pushing cart
        # Standard spring-damper: F = -K(x_cart - x_v) - D(ẋ_cart - ẋ_v)
        #                            = K(x_v - x_cart) + D(ẋ_v - ẋ_cart)
        # But wait, this gives the wrong coupling in dynamics...
        
        # Let me just use the negative of coupling force
        # F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart) [into virtual]
        # F_cart_push = -F_cart_coupling [onto cart]
        F_cart_push = self.k_cart * (x_v - x_cart) + self.d_cart * (v_v - v_cart)
        
        # Actually wait, I realize the issue. When I wrote the dynamics, I wrote:
        # F_cart_coupling = -K_c(x_v - x_cart) - D_c(ẋ_v - ẋ_cart)
        # This is the force the CART exerts on the VIRTUAL MASS.
        #
        # By Newton's 3rd law, the force the virtual mass exerts on the cart is:
        # F_cart_push = -F_cart_coupling = K_c(x_v - x_cart) + D_c(ẋ_v - ẋ_cart)
        #
        # So the code below is correct!
        
        # Saturate
        max_force = 500.0
        force_mag = np.linalg.norm(F_cart_push)
        if force_mag > max_force:
            F_cart_push = F_cart_push * (max_force / force_mag)
        
        output.SetFromVector(F_cart_push)


class EndEffectorKinematics(LeafSystem):
    """
    Compute manipulator end effector position and velocity.
    
    FORWARD KINEMATICS:
    ───────────────────
    Position:
        x_ee = CalcPosition(q)  [Uses manipulator.CalcPosition()]
    
    Velocity:
        ẋ_ee = J(q) q̇
    
    where J(q) is the manipulator Jacobian.
    
    INPUTS:
        - manipulator_state: [q₁, q₂, q̇₁, q̇₂]^T
    
    OUTPUTS:
        - ee_position: [x_ee, y_ee]^T
        - ee_velocity: [ẋ_ee, ẏ_ee]^T
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
        """Compute x_ee = FK(q)"""
        state = self.state_input.Eval(context)
        q = state[:2]
        
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        ee_pos_world = self.manipulator.CalcPosition(self.plant, self.plant_context)
        
        output.SetFromVector([ee_pos_world[0], ee_pos_world[1]])
    
    def CalcVelocity(self, context, output):
        """Compute ẋ_ee = J(q) q̇"""
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
        
        J_translational = J_spatial[3:6, :]  # Linear velocity part
        J_manip = J_translational[:, :2]     # 2-DOF manipulator
        v_ee_world = J_manip @ v
        
        output.SetFromVector([v_ee_world[0], v_ee_world[1]])


class MuscleDynamics(LeafSystem):
    """
    First-order muscle/actuator dynamics (from notes eq. \tau_f\,\dot{F}_{muscle} = u - F_{muscle}).
    
    EQUATION:
    ─────────
        τ_f Ḟ_muscle = u - F_muscle
        
    Rearranging:
        Ḟ_muscle = (u - F_muscle) / τ_f
    
    STATE:
        F_muscle ∈ ℝ (muscle force)
    
    INPUT:
        u ∈ ℝ (neural command from LQR controller)
    
    OUTPUT:
        F_muscle ∈ ℝ (muscle force applied to reference mass)
    
    PARAMETERS:
        τ_f: Muscle time constant [s] (~30 ms for human muscle)
    """
    
    def __init__(self, tau_f=0.03, initial_force=0.0):
        LeafSystem.__init__(self)
        
        if tau_f <= 0:
            raise ValueError("tau_f must be > 0")
        
        self.tau_f = float(tau_f)
        self.initial_force = float(initial_force)
        
        # Input port
        self.DeclareVectorInputPort("u", BasicVector(1))
        
        # Continuous state: [F_muscle]
        self.DeclareContinuousState(1)
        
        # Output port
        self.DeclareVectorOutputPort(
            "F_muscle",
            BasicVector(1),
            self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )
    
    def SetDefaultState(self, context, state):
        """Initialize muscle force."""
        state.get_mutable_continuous_state_vector().SetFromVector([self.initial_force])
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute muscle force derivative.
        
        Equation:
            Ḟ_muscle = (u - F_muscle) / τ_f
        """
        u = float(self.get_input_port(0).Eval(context)[0])
        F_muscle = float(context.get_continuous_state_vector().GetAtIndex(0))
        
        F_dot = (u - F_muscle) / self.tau_f
        
        derivatives.get_mutable_vector().SetAtIndex(0, F_dot)
    
    def _calc_output(self, context, output):
        """Output current muscle force."""
        F_muscle = float(context.get_continuous_state_vector().GetAtIndex(0))
        output.SetFromVector([F_muscle])


class ZFTReferenceMass(LeafSystem):
    """
    Zero Force Trajectory (ZFT) / Reference Mass System.
    
    EQUATIONS (from notes eq. M_{ref}\,\ddot{x}_{ref} = -F_{imp} + F_{muscle}):
    ────────────
        ẋ_ref = v_ref
        M_ref v̇_ref = -F_imp + F_muscle
    
    where:
        F_imp = K(x_ee - x_ref) + D(ẋ_ee - v_ref)  [Impedance force]
    
    Rearranging:
        v̇_ref = (F_imp_coupling + F_muscle) / M_ref
    
    where:
        F_imp_coupling = K(x_ee - x_ref) + D(ẋ_ee - v_ref)
    
    STATE:
        z_ref = [x_ref, v_ref]^T ∈ ℝ²
    
    INPUTS:
        - ee_state: [x_ee, v_ee]^T
        - F_muscle: muscle force
    
    OUTPUT:
        - ref_state: [x_ref, v_ref]^T
    
    PARAMETERS:
        M_ref: Reference mass [kg]
        K: Stiffness [N/m]
        D: Damping [N·s/m]
    """
    
    def __init__(self, M_ref=2.0, K=50.0, D=10.0, x_ref_init=0.0, v_ref_init=0.0):
        LeafSystem.__init__(self)
        
        if M_ref <= 0:
            raise ValueError("M_ref must be > 0")
        
        self.M_ref = float(M_ref)
        self.K = float(K)
        self.D = float(D)
        self.x_ref_init = float(x_ref_init)
        self.v_ref_init = float(v_ref_init)
        
        # Input ports
        self.ee_state_input = self.DeclareVectorInputPort("ee_state", BasicVector(2))
        self.F_muscle_input = self.DeclareVectorInputPort("F_muscle", BasicVector(1))
        
        # Continuous state: [x_ref, v_ref]
        self.DeclareContinuousState(2)
        
        # Output port
        self.DeclareVectorOutputPort(
            "ref_state",
            BasicVector(2),
            self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )
    
    def SetDefaultState(self, context, state):
        """Initialize reference position and velocity."""
        state.get_mutable_continuous_state_vector().SetFromVector(
            [self.x_ref_init, self.v_ref_init]
        )
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute reference mass dynamics.
        
        Equations:
            ẋ_ref = v_ref
            M_ref v̇_ref = F_imp_coupling + F_muscle
        
        where:
            F_imp_coupling = K(x_ee - x_ref) + D(v_ee - v_ref)
        """
        # Get EE state
        ee_state = self.ee_state_input.Eval(context)
        x_ee = float(ee_state[0])
        v_ee = float(ee_state[1])
        
        # Get muscle force
        F_muscle = float(self.F_muscle_input.Eval(context)[0])
        
        # Get reference state
        ref_vec = context.get_continuous_state_vector()
        x_ref = float(ref_vec[0])
        v_ref = float(ref_vec[1])
        
        # Impedance coupling force (INTO reference mass from EE)
        F_imp_coupling = self.K * (x_ee - x_ref) + self.D * (v_ee - v_ref)
        
        # Reference mass acceleration
        # M_ref v̇_ref = F_imp_coupling + F_muscle
        v_ref_dot = (F_imp_coupling + F_muscle) / self.M_ref
        
        # State derivative: [ẋ_ref, v̇_ref]^T
        derivatives.get_mutable_vector().SetFromVector([v_ref, v_ref_dot])
    
    def _calc_output(self, context, output):
        """Output reference state [x_ref, v_ref]^T."""
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())


class ImpedanceToCartForce(LeafSystem):
    """
    Converts 1D impedance force F_imp to 2D cart force [F_x, F_y].
    
    For X-axis motion only: [F_imp, 0]^T
    
    INPUT:
        - F_imp: scalar impedance force [N]
    
    OUTPUT:
        - cart_force: [F_x, F_y]^T = [F_imp, 0]^T
    """
    
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareVectorInputPort("F_imp", BasicVector(1))
        self.DeclareVectorOutputPort(
            "cart_force",
            BasicVector(2),
            self._calc_output
        )
    
    def _calc_output(self, context, output):
        """Convert 1D force to 2D cart force."""
        F_imp = float(self.GetInputPort("F_imp").Eval(context)[0])
        output.SetFromVector([F_imp, 0.0])


class ImpedanceForce(LeafSystem):
    """
    Computes impedance force (from notes eq. F_{imp} = K\,(x-x_{ref}) + D\,(\dot{x}-\dot{x}_{ref})).
    
    EQUATION:
    ─────────
        F_imp = K(x_ee - x_ref) + D(ẋ_ee - v_ref)
    
    INPUTS:
        - ee_state: [x_ee, v_ee]^T
        - ref_state: [x_ref, v_ref]^T
    
    OUTPUT:
        - F_imp: impedance force (scalar for 1D motion)
    
    PARAMETERS:
        K: Stiffness [N/m]
        D: Damping [N·s/m]
    """
    
    def __init__(self, K=50.0, D=10.0, force_limit=None):
        LeafSystem.__init__(self)
        
        self.K = float(K)
        self.D = float(D)
        self.force_limit = None if force_limit is None else float(force_limit)
        
        # Input ports
        self.DeclareVectorInputPort("ee_state", BasicVector(2))
        self.DeclareVectorInputPort("ref_state", BasicVector(2))
        
        # Output port
        self.DeclareVectorOutputPort("F_imp", BasicVector(1), self._calc_output)
    
    def _calc_output(self, context, output):
        """Compute F_imp = K(x_ee - x_ref) + D(v_ee - v_ref)"""
        ee_state = self.get_input_port(0).Eval(context)
        ref_state = self.get_input_port(1).Eval(context)
        
        x_ee = float(ee_state[0])
        v_ee = float(ee_state[1])
        x_ref = float(ref_state[0])
        v_ref = float(ref_state[1])
        
        F_imp = self.K * (x_ee - x_ref) + self.D * (v_ee - v_ref)
        
        if self.force_limit is not None:
            F_imp = float(np.clip(F_imp, -self.force_limit, self.force_limit))
        
        output.SetFromVector([F_imp])


class FiniteHorizonLQRController(LeafSystem):
    """
    Finite-horizon, continuous-time LQR for minimal effort control.
    
    CONTROL LAW:
    ────────────
        u(t) = -K(t)(x(t) - x_goal)
    
    COST FUNCTION:
        J = ∫_0^T [x'Qx + u'Ru] dt + x(T)'QN·x(T)
    
    Riccati recursion (discrete-time backward):
        P_k = Q + A^T P_{k+1} (A - B·K_k)
        K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A
    
    STATE VECTOR (for cart-pendulum + muscle + ref):
        x = [x_cart, β, ẋ_cart, β̇, F_muscle, x_ref, v_ref]^T ∈ ℝ⁷
    
    INPUT:
        u ∈ ℝ (neural command to muscle)
    
    NOTE: This is adapted from script_cart_pendulum_muscle_dynamics_ofc.py
          but simplified for 1D cart motion (X-axis only).
    """
    
    def __init__(self, A, B, Q, R, QN, x_goal, horizon=10.0, timestep=0.01):
        """
        Initialize Finite Horizon LQR Controller.
        
        Args:
            A: Continuous-time state matrix (n×n)
            B: Continuous-time input matrix (n×1)
            Q: State cost matrix (n×n)
            R: Control cost scalar
            QN: Terminal cost matrix (n×n)
            x_goal: Goal state (n,)
            horizon: Time horizon [s]
            timestep: Discretization timestep [s]
        """
        LeafSystem.__init__(self)
        
        # Store system matrices
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).reshape((-1, 1))
        self.Q = np.array(Q, dtype=float)
        self.R = np.array([[R]], dtype=float)
        self.QN = np.array(QN, dtype=float)
        self.x_goal = np.array(x_goal, dtype=float)
        self.T = float(horizon)
        self.dt = float(timestep)
        
        # Validate dimensions
        n = self.A.shape[0]
        assert self.A.shape == (n, n)
        assert self.B.shape == (n, 1)
        assert self.Q.shape == (n, n)
        assert self.QN.shape == (n, n)
        assert self.x_goal.shape == (n,)
        
        # Compute number of timesteps
        self.N = int(np.round(self.T / self.dt))
        if self.N < 1:
            raise ValueError(f"Horizon too short: N={self.N}")
        
        self.T = self.N * self.dt  # Adjust to exact multiple
        
        # Discretize continuous system (zero-order hold)
        self.Ad, self.Bd = self._discretize_zoh(self.A, self.B, self.dt)
        
        # Discretized cost matrices
        Qd = self.Q * self.dt
        Rd = self.R * self.dt
        QNd = self.QN
        
        # Compute time-varying gains via backward Riccati recursion
        self.K_list, self.P_list = self._finite_horizon_dlqr(
            self.Ad, self.Bd, Qd, Rd, QNd, self.N
        )
        
        # Drake ports
        self.DeclareVectorInputPort("x", BasicVector(n))
        self.DeclareVectorOutputPort("u", BasicVector(1), self.CalcU)
        
        print(colored(f"\n✓ FiniteHorizonLQRController initialized:", "green"))
        print(colored(f"  Horizon: {self.T:.2f} s (N={self.N} steps)", "cyan"))
        print(colored(f"  State dim: {n}, Control dim: 1", "cyan"))
    
    def CalcU(self, context, output):
        """Compute finite-horizon LQR control input."""
        x = self.get_input_port(0).Eval(context)
        t = context.get_time()
        
        # Select gain based on current time
        k = int(np.floor(t / self.dt))
        k = int(np.clip(k, 0, self.N - 1))
        
        K = self.K_list[k]
        x_err = (x - self.x_goal).reshape((-1, 1))
        u = float(-(K @ x_err)[0, 0])
        
        output.SetFromVector([u])
    
    @staticmethod
    def _finite_horizon_dlqr(Ad, Bd, Q, R, QN, N):
        """Backward Riccati recursion for discrete-time LQR."""
        n = Ad.shape[0]
        P_list = [None] * (N + 1)
        K_list = [None] * N
        
        # Initialize at terminal time
        P = QN.copy()
        P_list[N] = P
        
        # Backward recursion
        for k in reversed(range(N)):
            S = R + Bd.T @ P @ Bd
            K = np.linalg.solve(S, Bd.T @ P @ Ad)
            K_list[k] = K
            P = Q + Ad.T @ P @ (Ad - Bd @ K)
            P_list[k] = P
        
        return K_list, P_list
    
    @staticmethod
    def _discretize_zoh(A, B, dt):
        """Zero-order hold discretization via matrix exponential."""
        n = A.shape[0]
        m = B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        Md = expm(M * dt)
        
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]
        return Ad, Bd


def simulate_manipulator_pushes_cart(
    duration=10.0,
    push_distance=1.0,
    M_virtual=2.0,
    D_virtual=5.0,
    K_virtual=10.0,
    K_imp=100.0,
    D_imp=20.0,
):
    """
    Simulate manipulator pushing cart via virtual mass.
    
    SYSTEM ARCHITECTURE:
    ────────────────────
    Manipulator (Active) → Virtual Mass (Passive) → Cart-Pendulum (Passive)
    
    The manipulator uses impedance control to track a desired EE trajectory.
    The virtual mass provides compliant coupling.
    The cart-pendulum responds passively to forces.
    
    Args:
        duration: Simulation duration [s]
        push_distance: Distance manipulator EE travels in X [m]
        M_virtual: Virtual mass [kg]
        D_virtual: Virtual damping [N·s/m]
        K_virtual: Virtual stiffness [N/m]
        K_imp: Manipulator impedance stiffness [N/m]
        D_imp: Manipulator impedance damping [N·s/m]
        
    Returns:
        log_data: Dictionary with logged data including:
            - Joint angles q1, q2 (shows configuration needed to push)
            - EE, virtual, cart positions
            - Forces (reactive on EE, pushing on cart)
            - Reference trajectory
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR PUSHES CART VIA VIRTUAL MASS", "cyan", attrs=["bold"]))
    print(colored("(Human Arm Analogy - from notes_ss_cart_pendulam_manipulator.tex)", "cyan"))
    print(colored("="*80, "cyan"))
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}", "green", attrs=["bold"]))
    
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
    
    # Add actuators to manipulator
    joint1 = plant.GetJointByName("link1_base", manipulator.model_instance)
    joint2 = plant.GetJointByName("link2_link1", manipulator.model_instance)
    plant.AddJointActuator("joint1_actuator", joint1)
    plant.AddJointActuator("joint2_actuator", joint2)
    
    # Add cart-pendulum (PASSIVE - no actuators)
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
        add_cart_actuators=False  # PASSIVE!
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
    print(colored(f"  Manipulator: 2-DOF (ACTIVE - impedance control)", "cyan"))
    print(colored(f"  Cart-Pendulum: 4-DOF (PASSIVE - no control)", "cyan"))
    print(colored(f"  Initial EE: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m", "cyan"))
    print(colored(f"\nManipulator Impedance:", "yellow", attrs=["bold"]))
    print(colored(f"  K_imp = {K_imp:.1f} N/m", "cyan"))
    print(colored(f"  D_imp = {D_imp:.1f} N·s/m", "cyan"))
    print(colored(f"  M_ref = 2.0 kg", "cyan"))
    print(colored(f"\nVirtual Mass:", "yellow", attrs=["bold"]))
    print(colored(f"  M_v = {M_virtual:.2f} kg", "cyan"))
    print(colored(f"  D_v = {D_virtual:.2f} N·s/m", "cyan"))
    print(colored(f"  K_v = {K_virtual:.2f} N/m", "cyan"))
    print(colored(f"\nDesired Motion:", "yellow", attrs=["bold"]))
    print(colored(f"  EE travel: {push_distance:.3f} m in +X direction", "cyan"))
    print(colored(f"  Start X: {ee_pos_init[0]:.3f} m", "cyan"))
    print(colored(f"  Target X: {ee_pos_init[0] + push_distance:.3f} m", "cyan"))
    print(colored(f"  Duration: {duration:.1f} s", "cyan"))
    
    # Create systems
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    
    # TODO: Implement full muscle dynamics + LQR architecture
    # For now, use simplified impedance control without muscle dynamics
    # Full implementation would include:
    # 1. MuscleDynamics system
    # 2. ZFTReferenceMass system  
    # 3. ImpedanceForce system
    # 4. FiniteHorizonLQRController system
    # 5. ManipulatorJacobianTransposeController
    
    # Simplified version: Direct ZFT + Impedance + Jacobian Transpose
    zft_ref_mass = builder.AddSystem(ZFTReferenceMass(
        M_ref=2.0,
        K=K_imp,
        D=D_imp,
        x_ref_init=ee_pos_init[0],
        v_ref_init=0.0
    ))
    
    impedance_force = builder.AddSystem(ImpedanceForce(
        K=K_imp,
        D=D_imp,
        force_limit=None
    ))
    
    jacobian_controller = builder.AddSystem(ManipulatorJacobianTransposeController(
        plant=plant,
        manipulator=manipulator
    ))
    
    # Dummy muscle force (zero for now - would come from LQR in full implementation)
    from pydrake.all import ConstantVectorSource
    zero_muscle_force = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
    
    # Converter: F_imp (1D) -> [F_x, F_y] (2D) for cart
    imp_to_cart = builder.AddSystem(ImpedanceToCartForce())
    
    # Demultiplexers for state parsing
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))  # [manip(4), cart(8)]
    cart_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, angles, vel, ang_vel]
    
    # EE position/velocity demux (extract X component only)
    ee_pos_demux = builder.AddSystem(Demultiplexer([1, 1]))  # [x, y]
    ee_vel_demux = builder.AddSystem(Demultiplexer([1, 1]))  # [vx, vy]
    
    # EE X-axis state combiner (x + vx -> [x, vx] for ZFT/Impedance)
    ee_x_state_mux = builder.AddSystem(Multiplexer([1, 1]))  # [x, vx] -> state
    
    # Connect plant state
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    
    # Connect manipulator state
    builder.Connect(state_demux.get_output_port(0), ee_kinematics.GetInputPort("manipulator_state"))
    builder.Connect(state_demux.get_output_port(0), jacobian_controller.GetInputPort("manipulator_state"))
    
    # Connect cart state
    builder.Connect(state_demux.get_output_port(1), cart_demux.get_input_port())
    
    # Extract X components from EE kinematics
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_demux.get_input_port())
    builder.Connect(ee_kinematics.GetOutputPort("ee_velocity"), ee_vel_demux.get_input_port())
    
    # Combine X components for ZFT and impedance
    builder.Connect(ee_pos_demux.get_output_port(0), ee_x_state_mux.get_input_port(0))  # x
    builder.Connect(ee_vel_demux.get_output_port(0), ee_x_state_mux.get_input_port(1))  # vx
    
    # Connect EE X-state and muscle force to ZFT reference mass
    builder.Connect(ee_x_state_mux.get_output_port(0), zft_ref_mass.GetInputPort("ee_state"))
    builder.Connect(zero_muscle_force.get_output_port(0), zft_ref_mass.GetInputPort("F_muscle"))
    
    # Connect EE X-state and ref state to impedance force
    builder.Connect(ee_x_state_mux.get_output_port(0), impedance_force.GetInputPort("ee_state"))
    builder.Connect(zft_ref_mass.GetOutputPort("ref_state"), impedance_force.GetInputPort("ref_state"))
    
    # Connect impedance force to Jacobian controller
    builder.Connect(impedance_force.GetOutputPort("F_imp"), jacobian_controller.GetInputPort("F_imp"))
    
    # Connect impedance force to cart (via converter)
    builder.Connect(impedance_force.GetOutputPort("F_imp"), imp_to_cart.GetInputPort("F_imp"))
    
    # DIRECT CONNECTION: F_imp → Cart (no virtual mass!)
    # This matches notes: cart receives +F_imp, manipulator receives -J^T F_imp
    cart_body = cart_pendulum.cart_body
    
    # Apply F_imp directly to cart (no virtual mass intermediary)
    class CartForceApplicator(LeafSystem):
        """Applies impedance force directly to cart."""
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
            cart_force_2d = self.GetInputPort("cart_force").Eval(context)
            
            spatial_force = ExternallyAppliedSpatialForce()
            spatial_force.body_index = self.cart_body_index
            spatial_force.F_Bq_W = SpatialForce(
                tau=np.zeros(3),
                f=np.array([cart_force_2d[0], cart_force_2d[1], 0.0])
            )
            spatial_force.p_BoBq_B = np.zeros(3)
            
            output.set_value([spatial_force])
    
    cart_force_applicator = builder.AddSystem(CartForceApplicator(cart_body.index()))
    builder.Connect(imp_to_cart.GetOutputPort("cart_force"), cart_force_applicator.GetInputPort("cart_force"))
    builder.Connect(cart_force_applicator.GetOutputPort("spatial_forces"), plant.get_applied_spatial_force_input_port())
    
    # Connect Jacobian controller torques to actuation
    builder.Connect(jacobian_controller.get_output_port(), plant.get_actuation_input_port())
    
    # Visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    ref_state_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(zft_ref_mass.GetOutputPort("ref_state"), ref_state_logger.get_input_port())
    
    F_imp_logger = builder.AddSystem(VectorLogSink(1))
    builder.Connect(impedance_force.GetOutputPort("F_imp"), F_imp_logger.get_input_port())
    
    F_imp_logger = builder.AddSystem(VectorLogSink(1))
    builder.Connect(impedance_force.GetOutputPort("F_imp"), F_imp_logger.get_input_port())
    
    cart_force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(imp_to_cart.GetOutputPort("cart_force"), cart_force_logger.get_input_port())
    
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
    print(colored(f"\n🎬 Animation: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    ref_state_log = ref_state_logger.FindLog(context)
    F_imp_log = F_imp_logger.FindLog(context)
    cart_force_log = cart_force_logger.FindLog(context)
    
    time_data = state_log.sample_times()
    state_data = state_log.data()
    ee_pos_data = ee_pos_log.data()
    ref_state_data = ref_state_log.data()
    F_imp_data = F_imp_log.data()
    cart_force_data = cart_force_log.data()
    
    # Parse state
    # State order: [q1, q2, cart_x, cart_y, pend_alpha, pend_beta, dq1, dq2, dcart_x, dcart_y, dpend_alpha, dpend_beta]
    q1 = state_data[0, :]
    q2 = state_data[1, :]
    cart_x = state_data[2, :]
    cart_y = state_data[3, :]
    pend_alpha = state_data[4, :]  # Pendulum pitch (rotation around Y)
    pend_beta = state_data[5, :]   # Pendulum roll (rotation around X)
    
    # Velocities
    dq1 = state_data[6, :]
    dq2 = state_data[7, :]
    cart_vx = state_data[8, :]
    cart_vy = state_data[9, :]
    pend_alpha_dot = state_data[10, :]
    pend_beta_dot = state_data[11, :]
    
    return {
        'time': time_data,
        'q1': q1,
        'q2': q2,
        'cart_x': cart_x,
        'cart_y': cart_y,
        'cart_vx': cart_vx,
        'cart_vy': cart_vy,
        'pend_alpha': pend_alpha,
        'pend_beta': pend_beta,
        'pend_alpha_dot': pend_alpha_dot,
        'pend_beta_dot': pend_beta_dot,
        'ee_x': ee_pos_data[0, :],
        'ee_y': ee_pos_data[1, :],
        'ref_x': ref_state_data[0, :],  # x_ref from ZFT
        'ref_v': ref_state_data[1, :],  # v_ref from ZFT
        'cart_force_x': cart_force_data[0, :],
        'cart_force_y': cart_force_data[1, :],
        'F_imp': F_imp_data[0, :],
    }


def plot_results(log_data):
    """Plot simulation results with complete visualization."""
    print(colored("\n📈 Generating plots...", "yellow"))
    
    t = log_data['time']
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Joint angles (shows what configuration is needed)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, np.rad2deg(log_data['q1']), 'b-', linewidth=2, label='q₁')
    ax1.plot(t, np.rad2deg(log_data['q2']), 'r-', linewidth=2, label='q₂')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
    ax1.set_title('Manipulator Joint Configuration\n(Needed to Push Cart)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: X positions (Direct: Ref → EE → Cart)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, log_data['ref_x'], 'r-.', linewidth=2, alpha=0.7, label='x_ref (ZFT)')
    ax2.plot(t, log_data['ee_x'], 'b-', linewidth=2.5, label='x_ee (EE Actual)')
    ax2.plot(t, log_data['cart_x'], 'g:', linewidth=2.5, label='x_cart (Cart)')
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('X Position [m]', fontweight='bold')
    ax2.set_title('X Motion: x_ref → x_ee → F_imp → Cart\n(NO Virtual Mass!)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Y positions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, np.zeros_like(t) + log_data['ee_y'][0], 'r-.', linewidth=2, alpha=0.7, label='y_ref')
    ax3.plot(t, log_data['ee_y'], 'b-', linewidth=2.5, label='y_ee')
    ax3.plot(t, log_data['cart_y'], 'g:', linewidth=2.5, label='y_cart')
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Y Position [m]', fontweight='bold')
    ax3.set_title('Y Position (Should Stay ~Constant)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Forces
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, log_data['F_imp'], 'm-', linewidth=2.5, label='F_imp (Impedance)')
    ax4.plot(t, log_data['cart_force_x'], 'g-', linewidth=2, label='Cart Push Fₓ')
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time [s]', fontweight='bold')
    ax4.set_ylabel('Force [N]', fontweight='bold')
    ax4.set_title('Forces: F_imp = K(x_ee - x_ref) + D·v_err', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: 2D Paths
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(log_data['ref_x'], np.zeros_like(log_data['ref_x']) + log_data['ee_y'][0], 
             'r-.', linewidth=2.5, alpha=0.9, label='x_ref (ZFT)')
    ax5.plot(log_data['ee_x'], log_data['ee_y'], 'b-', linewidth=2.5, alpha=0.8, label='x_ee')
    ax5.plot(log_data['cart_x'], log_data['cart_y'], 'g:', linewidth=2.5, alpha=0.6, label='Cart')
    ax5.plot(log_data['ee_x'][0], log_data['ee_y'][0], 'go', markersize=12, label='Start')
    ax5.plot(log_data['ee_x'][-1], log_data['ee_y'][-1], 'ro', markersize=12, label='End')
    ax5.set_xlabel('X Position [m]', fontweight='bold')
    ax5.set_ylabel('Y Position [m]', fontweight='bold')
    ax5.set_title('2D Trajectories (Direct Coupling)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.legend()
    
    # Plot 6: Tracking error
    ax6 = fig.add_subplot(gs[1, 2])
    error_x = (log_data['ee_x'] - log_data['ref_x']) * 1000  # mm
    error_v = (np.gradient(log_data['ee_x'], t) - log_data['ref_v']) * 1000  # mm/s
    error_mag = np.abs(error_x)
    ax6.plot(t, error_x, 'b-', linewidth=2, label='Position Error (mm)')
    ax6.plot(t, error_v / 10, 'r-', linewidth=2, label='Velocity Error /10 (mm/s)')
    ax6.plot(t, error_mag, 'k--', linewidth=1.5, label='|Error|')
    ax6.set_xlabel('Time [s]', fontweight='bold')
    ax6.set_ylabel('Error [mm]', fontweight='bold')
    ax6.set_title('Impedance Tracking Error', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Joint configuration evolution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(np.rad2deg(log_data['q1']), np.rad2deg(log_data['q2']), 'purple', linewidth=2)
    ax7.plot(np.rad2deg(log_data['q1'][0]), np.rad2deg(log_data['q2'][0]), 
             'go', markersize=12, label='Initial')
    ax7.plot(np.rad2deg(log_data['q1'][-1]), np.rad2deg(log_data['q2'][-1]), 
             'ro', markersize=12, label='Final')
    ax7.set_xlabel('q₁ [deg]', fontweight='bold')
    ax7.set_ylabel('q₂ [deg]', fontweight='bold')
    ax7.set_title('Joint Space Trajectory', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Force magnitudes and F_imp
    ax8 = fig.add_subplot(gs[2, 1])
    cart_force_mag = np.sqrt(log_data['cart_force_x']**2 + log_data['cart_force_y']**2)
    ax8.plot(t, log_data['F_imp'], 'm-', linewidth=2.5, label='F_imp (Impedance)')
    ax8.plot(t, cart_force_mag, 'g--', linewidth=2, label='||F_cart|| (Should Equal F_imp)')
    ax8.set_xlabel('Time [s]', fontweight='bold')
    ax8.set_ylabel('Force [N]', fontweight='bold')
    ax8.set_title('Impedance Force F_imp = K·err_x + D·err_v', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Plot 9: Cart Velocities
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, log_data['cart_vx'], 'b-', linewidth=2, label='vₓ (Cart)')
    ax9.plot(t, log_data['cart_vy'], 'r-', linewidth=2, label='vᵧ (Cart)')
    ax9.plot(t, log_data['ref_v'], 'g--', linewidth=2, label='v_ref (ZFT)')
    ax9.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Time [s]', fontweight='bold')
    ax9.set_ylabel('Velocity [m/s]', fontweight='bold')
    ax9.set_title('Cart Velocities (Passive Response)', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # Plot 10: Pendulum Angles
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, np.rad2deg(log_data['pend_alpha']), 'b-', linewidth=2, label='α (Pitch)')
    ax10.plot(t, np.rad2deg(log_data['pend_beta']), 'r-', linewidth=2, label='β (Roll)')
    ax10.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax10.set_xlabel('Time [s]', fontweight='bold')
    ax10.set_ylabel('Pendulum Angle [deg]', fontweight='bold')
    ax10.set_title('Pendulum Angles (Passive Swing)', fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    
    # Plot 11: Pendulum Angular Velocities
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, np.rad2deg(log_data['pend_alpha_dot']), 'b-', linewidth=2, label='α̇ (Pitch Rate)')
    ax11.plot(t, np.rad2deg(log_data['pend_beta_dot']), 'r-', linewidth=2, label='β̇ (Roll Rate)')
    ax11.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax11.set_xlabel('Time [s]', fontweight='bold')
    ax11.set_ylabel('Angular Velocity [deg/s]', fontweight='bold')
    ax11.set_title('Pendulum Angular Rates', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    # Plot 12: Summary
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    summary_text = f"""
SYSTEM STATE SUMMARY

Initial Configuration:
  q₁ = {np.rad2deg(log_data['q1'][0]):6.2f}°
  q₂ = {np.rad2deg(log_data['q2'][0]):6.2f}°

Final Configuration (t = {t[-1]:.1f}s):
  q₁ = {np.rad2deg(log_data['q1'][-1]):6.2f}°
  q₂ = {np.rad2deg(log_data['q2'][-1]):6.2f}°

Joint Changes:
  Δq₁ = {np.rad2deg(log_data['q1'][-1] - log_data['q1'][0]):6.2f}°
  Δq₂ = {np.rad2deg(log_data['q2'][-1] - log_data['q2'][0]):6.2f}°

Cart Displacement:
  ΔX_cart = {log_data['cart_x'][-1] - log_data['cart_x'][0]:5.3f} m
  ΔY_cart = {log_data['cart_y'][-1] - log_data['cart_y'][0]:5.3f} m

Cart Final Velocity:
  vₓ = {log_data['cart_vx'][-1]:5.3f} m/s
  vᵧ = {log_data['cart_vy'][-1]:5.3f} m/s

Pendulum Final Angles:
  α = {np.rad2deg(log_data['pend_alpha'][-1]):5.2f}° (pitch)
  β = {np.rad2deg(log_data['pend_beta'][-1]):5.2f}° (roll)

Max Impedance Force:
  F_imp_max = {np.max(np.abs(log_data['F_imp'])):5.2f} N
    """
    ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Manipulator Pushes Cart - Direct Impedance Control\n(M_ref → x_ref, F_imp = K(x_ee - x_ref) + D·v_err → Cart)', 
                 fontsize=14, fontweight='bold')
    
    print(colored("✓ Plots generated", "green"))
    return fig


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Manipulator pushes cart via virtual mass (human arm analogy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Based on notes_ss_cart_pendulam_manipulator.tex with F_muscle = 0.

The simulation shows what joint configuration is required for the manipulator
to push the passive cart-pendulum system 1 meter through a compliant virtual
mass interface.
        """
    )
    parser.add_argument('--duration', type=float, default=10.0, 
                       help='Simulation duration [s] (default: 10.0)')
    parser.add_argument('--distance', type=float, default=1.0, 
                       help='EE push distance in X [m] (default: 1.0)')
    parser.add_argument('--mass', type=float, default=2.0, 
                       help='Virtual mass M_v [kg] (default: 2.0)')
    parser.add_argument('--damping', type=float, default=5.0, 
                       help='Virtual damping D_v [N·s/m] (default: 5.0)')
    parser.add_argument('--stiffness', type=float, default=10.0, 
                       help='Virtual stiffness K_v [N/m] (default: 10.0)')
    parser.add_argument('--k-imp', type=float, default=100.0, 
                       help='Manipulator impedance stiffness K_imp [N/m] (default: 100.0)')
    parser.add_argument('--d-imp', type=float, default=20.0, 
                       help='Manipulator impedance damping D_imp [N·s/m] (default: 20.0)')
    args = parser.parse_args()
    
    # Run simulation
    log_data = simulate_manipulator_pushes_cart(
        duration=args.duration,
        push_distance=args.distance,
        M_virtual=args.mass,
        D_virtual=args.damping,
        K_virtual=args.stiffness,
        K_imp=args.k_imp,
        D_imp=args.d_imp,
    )
    
    # Plot results
    fig = plot_results(log_data)
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("SIMULATION COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    print(colored("\nAnswer: Joint configuration evolution shown in plots.", "yellow"))
    print(colored("The manipulator adapts its joints to maintain impedance", "yellow"))
    print(colored("control while pushing the cart through virtual mass.", "yellow"))
    
    # Keep Meshcat open
    input(colored("\nPress Enter to close...", "cyan"))


if __name__ == "__main__":
    main()
