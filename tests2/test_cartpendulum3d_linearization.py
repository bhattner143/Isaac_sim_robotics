#!/usr/bin/env python3
"""
Test Script: Validate Numerical Linearization vs Drake's Auto-Diff Linearization

This script validates the finite difference linearization method against
Drake's automatic differentiation linearization for a cart-pendulum 3D system.

SYSTEM DESCRIPTION:
- Cart: 2 DOF (x, y position), mass = 1.0 kg
- Pendulum: 2 DOF (pitch, roll gimbal), mass = 0.5 kg, length = 0.2 m
- Total: 4 DOF, 8D state [x, y, pitch, roll, ẋ, ẏ, pitch_dot, roll_dot]
- Input: 2D forces on cart [F_x, F_y]
- Cart acceleration affects pendulum motion (coupled dynamics)

VALIDATION APPROACH:
1. Build a MultibodyPlant with cart + 3D pendulum (gimbal-mounted)
2. Compute "ground truth" A, B matrices using Drake's Linearize() (auto-diff)
3. Compute numerical A, B matrices using finite_difference_linearization()
4. Compare matrices and report differences (should be O(ε²) ≈ 1e-6)

TEST CASES:
- Equilibrium 1: Hanging down (cart at rest, θ=0°, φ=0°) - stable
- Equilibrium 2: Inverted (cart at rest, θ=180°, φ=0°) - unstable

NOTE:
- "Analytical" = Drake's Linearize() using automatic differentiation
- "Numerical" = Custom finite_difference_linearization() using central differences
- Both should agree to within O(ε²) where ε = 1e-6
"""

import numpy as np
import sys
from pathlib import Path
from termcolor import colored

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    Sphere,
    Cylinder,
    Linearize,
)

# Import configuration and classes from main script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.robot.robot_types import create_cart_pendulum_config, CartPendulumConfig


# Cart-Pendulum configuration (for testing/validation purposes)
CART_PENDULUM_CONFIG = create_cart_pendulum_config(
    cart_mass=1.0,
    cart_size=0.1,
    cart_damping=0.0,
    pendulum_mass=0.5,
    pendulum_length=0.2,
    pendulum_radius=0.05,
    pendulum_damping=0.0,
    attachment_offset=(0.0, 0.0, 0.0),
    initial_cart_x=0.0,
    initial_cart_y=0.0,
    initial_pitch=0.0,
    initial_roll=0.0,
    name="cart_pendulum"
)

# ============================================================================
# CART-PENDULUM 3D CLASS (INDEPENDENT - NO PENDULUM3D DEPENDENCY)
# ============================================================================

class CartPendulum3D:
    """
    Independent 3D Cart-Pendulum System with 2D cart motion and gimbal-mounted pendulum.
    
    This class is fully self-contained and does NOT depend on Pendulum3D.
    It creates both the cart and pendulum bodies internally.
    
    SYSTEM ARCHITECTURE:
    --------------------
    Cart: 2 DOF (x, y position in horizontal plane)
        - Actuated by forces [F_x, F_y]
        - Mass: configurable
        - Visualization: Optional (can be disabled)
    
    Pendulum: 2 DOF (pitch, roll gimbal angles)
        - Attached to cart at specified offset
        - Passive (no direct actuation)
        - Visualization: Sphere + rod (always visible)
    
    Total System: 4 DOF → 8D state [x, y, α, β, ẋ, ẏ, α̇, β̇]
    
    COUPLING:
    ---------
    - Cart acceleration affects pendulum motion (inertial coupling)
    - Pendulum motion creates reaction forces on cart
    - Full nonlinear coupled dynamics via MultibodyPlant
    
    COORDINATE SYSTEM:
    ------------------
    - x, y: Cart position in horizontal plane (m)
    - α (pitch): Pendulum rotation about Y-axis (rad)
    - β (roll): Pendulum rotation about X-axis (rad)
    - Zero angles: pendulum hanging down
    
    USE CASES:
    ----------
    - Testing linearization methods
    - OFC controller development
    - Coupled dynamics analysis
    """
    
    def __init__(
        self,
        config: CartPendulumConfig,
        visualize_cart: bool = False,
    ):
        """
        Initialize cart-pendulum system.
        
        Args:
            config: CartPendulumConfig with all system parameters
            visualize_cart: If True, add visual geometry to cart; if False, cart is invisible
        """
        self.config = config
        self.visualize_cart = visualize_cart
        
        # Will be populated during attach_to_plant()
        self.cart_body = None
        self.x_slider_body = None
        self.x_joint = None
        self.y_joint = None
        self.pitch_joint = None
        self.roll_joint = None
        self.pendulum_body = None
        self.pitch_body = None  # Intermediate body for gimbal
        
    def attach_to_plant(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True):
        """
        Attach cart-pendulum system to plant.
        
        Creates:
        1. Cart body with mass/inertia (optionally invisible)
        2. Prismatic joints for x and y motion
        3. Pendulum with gimbal joints (always visible if visuals enabled)
        4. Actuators for cart forces
        
        Args:
            plant: MultibodyPlant (before Finalize)
            model_instance: Model instance for all bodies
            register_visuals: If True, register visual geometry (requires scene graph)
        """
        from pydrake.all import Sphere, Cylinder, Rgba, RigidTransform, RevoluteJoint
        
        # ====================================================================
        # CREATE CART BODY
        # ====================================================================
        I_cart = (1.0/6.0) * self.config.cart_mass * (self.config.cart_size**2)
        cart_inertia = SpatialInertia(
            mass=self.config.cart_mass,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_cart, I_cart, I_cart)
        )
        
        self.cart_body = plant.AddRigidBody(
            f"{self.config.name}_cart",
            model_instance,
            cart_inertia
        )
        
        # Add visual geometry only if requested and scene graph is available
        if self.visualize_cart and register_visuals:
            from pydrake.all import Box
            cart_shape = Box(self.config.cart_size, self.config.cart_size, self.config.cart_size)
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                cart_shape,
                f"{self.config.name}_cart_visual",
                np.array([0.3, 0.3, 0.8, 0.5])  # Semi-transparent blue
            )
        
        # ====================================================================
        # CREATE INTERMEDIATE SLIDER FOR 2-DOF CART MOTION
        # ====================================================================
        # Chain: world --[x]--> x_slider --[y]--> cart
        x_slider_inertia = SpatialInertia(
            mass=0.001,  # Negligible mass
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.x_slider_body = plant.AddRigidBody(
            f"{self.config.name}_x_slider",
            model_instance,
            x_slider_inertia
        )
        
        # ====================================================================
        # ADD PRISMATIC JOINTS
        # ====================================================================
        # X-axis joint
        self.x_joint = plant.AddJoint(
            PrismaticJoint(
                name=f"{self.config.name}_x",
                frame_on_parent=plant.world_frame(),
                frame_on_child=self.x_slider_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.config.cart_damping
            )
        )
        
        # Y-axis joint
        self.y_joint = plant.AddJoint(
            PrismaticJoint(
                name=f"{self.config.name}_y",
                frame_on_parent=self.x_slider_body.body_frame(),
                frame_on_child=self.cart_body.body_frame(),
                axis=[0.0, 1.0, 0.0],
                damping=self.config.cart_damping
            )
        )
        
        # Add actuators
        plant.AddJointActuator(f"{self.config.name}_force_x", self.x_joint)
        plant.AddJointActuator(f"{self.config.name}_force_y", self.y_joint)
        
        # ====================================================================
        # CREATE PENDULUM BODIES AND JOINTS (GIMBAL MOUNT)
        # ====================================================================
        # Pendulum mass properties (cylinder)
        m_p = self.config.pendulum_mass
        L = self.config.pendulum_length
        r = self.config.pendulum_radius
        
        # Cylinder inertia about its own center (z-axis aligned)
        I_xx_com = I_yy_com = (1.0/12.0) * m_p * (3*r**2 + L**2)
        I_zz_com = 0.5 * m_p * r**2
        
        # Intermediate body for pitch rotation
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            f"{self.config.name}_pitch_body",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body - use SpatialInertia.MakeFromCentralInertia()
        # CRITICAL: COM must be offset from pivot to create pendulum dynamics!
        # For a hanging pendulum, COM is at -L/2 in the Z-direction (body frame)
        com_offset = np.array([0.0, 0.0, -L/2])  # COM at -L/2 below pivot
        
        # Inertia about COM (for a cylinder)
        I_com = RotationalInertia(I_xx_com, I_yy_com, I_zz_com)
        
        # Use MakeFromCentralInertia to automatically apply parallel axis theorem
        pendulum_inertia = SpatialInertia.MakeFromCentralInertia(
            mass=m_p,
            p_PScm_E=com_offset,  # COM offset from body origin P
            I_SScm_E=I_com  # Inertia about COM
        )
        self.pendulum_body = plant.AddRigidBody(
            f"{self.config.name}_pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Add visual geometry for pendulum (always visible if register_visuals=True)
        if register_visuals:
            # Position sphere at -L (bottom of pendulum in body frame)
            pendulum_sphere = Sphere(r)
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform(p=[0.0, 0.0, -L]),  # At bottom of pendulum
                pendulum_sphere,
                f"{self.config.name}_pendulum_visual",
                np.array([0.9, 0.1, 0.1, 1.0])  # Red
            )
            
            # Add cylinder rod (visual from origin to -L)
            pendulum_rod = Cylinder(radius=r/3, length=L)
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform(p=[0.0, 0.0, -L/2]),  # Cylinder center at -L/2
                pendulum_rod,
                f"{self.config.name}_rod_visual",
                np.array([0.6, 0.6, 0.6, 1.0])  # Gray
            )
        
        # ====================================================================
        # ADD GIMBAL JOINTS
        # ====================================================================
        # Pitch joint (rotation about Y-axis) - cart to pitch_body
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.config.name}_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0.0, 1.0, 0.0],  # Y-axis
                damping=self.config.pendulum_damping
            )
        )
        
        # Roll joint (rotation about X-axis) - pitch_body to pendulum
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.config.name}_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1.0, 0.0, 0.0],  # X-axis
                damping=self.config.pendulum_damping
            )
        )
        
        # ====================================================================
        # PRINT SUMMARY
        # ====================================================================
        print(colored(f"\n✓ Cart-Pendulum 3D System Created:", 'green', attrs=['bold']))
        print(colored(f"  Cart mass: {self.config.cart_mass} kg", 'cyan'))
        print(colored(f"  Cart DOF: 2 (x, y)", 'cyan'))
        print(colored(f"  Cart visualization: {'ON' if self.visualize_cart else 'OFF'}", 'cyan'))
        print(colored(f"  Pendulum mass: {self.config.pendulum_mass} kg", 'cyan'))
        print(colored(f"  Pendulum length: {self.config.pendulum_length} m", 'cyan'))
        print(colored(f"  Pendulum DOF: 2 (pitch, roll)", 'cyan'))
        print(colored(f"  Pendulum visualization: ON (always)", 'cyan'))
        print(colored(f"  Total DOF: 4", 'cyan'))
        print(colored(f"  Total state: 8D [x, y, α, β, ẋ, ẏ, α̇, β̇]", 'cyan'))
        print(colored(f"  Inputs: 2D [F_x, F_y]", 'cyan'))
    
    def set_cart_state(self, context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0):
        """Set cart position and velocity."""
        self.x_joint.set_translation(context, x)
        self.y_joint.set_translation(context, y)
        self.x_joint.set_translation_rate(context, x_dot)
        self.y_joint.set_translation_rate(context, y_dot)
    
    def set_pendulum_state(self, context, pitch=0.0, roll=0.0, pitch_dot=0.0, roll_dot=0.0):
        """Set pendulum angles and velocities."""
        self.pitch_joint.set_angle(context, pitch)
        self.roll_joint.set_angle(context, roll)
        self.pitch_joint.set_angular_rate(context, pitch_dot)
        self.roll_joint.set_angular_rate(context, roll_dot)
    
    def get_cart_state(self, context):
        """Get cart position and velocity."""
        x = self.x_joint.get_translation(context)
        y = self.y_joint.get_translation(context)
        x_dot = self.x_joint.get_translation_rate(context)
        y_dot = self.y_joint.get_translation_rate(context)
        return np.array([x, y, x_dot, y_dot])
    
    def get_pendulum_state(self, context):
        """Get pendulum angles and velocities."""
        pitch = self.pitch_joint.get_angle(context)
        roll = self.roll_joint.get_angle(context)
        pitch_dot = self.pitch_joint.get_angular_rate(context)
        roll_dot = self.roll_joint.get_angular_rate(context)
        return np.array([pitch, roll, pitch_dot, roll_dot])
    
    def get_full_state(self, context):
        """Get complete 8D state vector."""
        cart_state = self.get_cart_state(context)
        pend_state = self.get_pendulum_state(context)
        return np.concatenate([
            [cart_state[0], cart_state[1]],  # x, y
            [pend_state[0], pend_state[1]],  # pitch, roll
            [cart_state[2], cart_state[3]],  # x_dot, y_dot
            [pend_state[2], pend_state[3]],  # pitch_dot, roll_dot
        ])
    
    def finite_difference_linearization(self, plant, context, epsilon=1e-6):
        """
        Compute linearization A, B matrices using numerical finite differences.
        
        Uses central differences: f(x+ε) - f(x-ε) / (2ε) for each state and input dimension.
        
        Args:
            plant: MultibodyPlant
            context: Context at equilibrium
            epsilon: Perturbation size (default 1e-6)
        
        Returns:
            A: State matrix (8×8) - ∂ẋ/∂x at equilibrium
            B: Input matrix (8×2) - ∂ẋ/∂u at equilibrium
        """
        n = plant.num_multibody_states()  # Should be 8
        m = plant.num_actuators()  # Should be 2
        
        # Get equilibrium state and input
        x0 = plant.GetPositionsAndVelocities(context)
        u0 = plant.get_actuation_input_port().Eval(context)
        
        # Initialize matrices
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        
        # Compute A matrix (∂ẋ/∂x) using central differences
        for i in range(n):
            # Perturb state +epsilon
            context_plus = context.Clone()
            x_plus = x0.copy()
            x_plus[i] += epsilon
            plant.SetPositionsAndVelocities(context_plus, x_plus)
            plant.get_actuation_input_port().FixValue(context_plus, u0)
            xdot_plus = plant.EvalTimeDerivatives(context_plus).CopyToVector()
            
            # Perturb state -epsilon
            context_minus = context.Clone()
            x_minus = x0.copy()
            x_minus[i] -= epsilon
            plant.SetPositionsAndVelocities(context_minus, x_minus)
            plant.get_actuation_input_port().FixValue(context_minus, u0)
            xdot_minus = plant.EvalTimeDerivatives(context_minus).CopyToVector()
            
            # Central difference
            A[:, i] = (xdot_plus - xdot_minus) / (2 * epsilon)
        
        # Compute B matrix (∂ẋ/∂u) using central differences
        for j in range(m):
            # Perturb input +epsilon
            context_plus = context.Clone()
            plant.SetPositionsAndVelocities(context_plus, x0)
            u_plus = u0.copy()
            u_plus[j] += epsilon
            plant.get_actuation_input_port().FixValue(context_plus, u_plus)
            xdot_plus = plant.EvalTimeDerivatives(context_plus).CopyToVector()
            
            # Perturb input -epsilon
            context_minus = context.Clone()
            plant.SetPositionsAndVelocities(context_minus, x0)
            u_minus = u0.copy()
            u_minus[j] -= epsilon
            plant.get_actuation_input_port().FixValue(context_minus, u_minus)
            xdot_minus = plant.EvalTimeDerivatives(context_minus).CopyToVector()
            
            # Central difference
            B[:, j] = (xdot_plus - xdot_minus) / (2 * epsilon)
        
        return A, B

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfig:
    """Test parameters."""
    # Cart parameters
    CART_MASS = 1.0  # kg
    CART_SIZE = 0.1  # m (cube side length)
    
    # Pendulum parameters
    PENDULUM_MASS = 0.5  # kg
    PENDULUM_LENGTH = 0.2  # m
    PENDULUM_RADIUS = 0.05  # m
    PENDULUM_DAMPING = 0.0  # No damping for cleaner linearization
    
    # Linearization parameters
    EPSILON = 1e-6  # Finite difference step size
    
    # Comparison tolerances
    TOLERANCE_TIGHT = 1e-5  # For well-conditioned systems
    TOLERANCE_LOOSE = 1e-3  # For near-singular systems
    
    # Test equilibria (theta, phi) in degrees
    # Note: For cart-pendulum, θ=0°,φ=0° (hanging) and θ=180°,φ=0° (inverted) 
    # with cart at rest (x=0, y=0, ẋ=0, ẏ=0) are true equilibria
    TEST_EQUILIBRIA = [
        (0.0, 0.0, "Hanging Down (Stable Equilibrium)"),
        (180.0, 0.0, "Inverted (Unstable Equilibrium)"),
    ]


# ============================================================================
# BUILD TEST SYSTEM
# ============================================================================

def build_cart_pendulum_system():
    """
    Build a cart-pendulum 3D system with 2D cart motion (x, y) using CartPendulum3D class.
    
    System:
    - Cart: 2 DOF (x, y position), actuated by forces [F_x, F_y]
    - Pendulum: 2 DOF (pitch, roll), passive (no direct actuation)
    - Total: 4 DOF, 8D state, 2D input
    
    Returns:
        plant: MultibodyPlant with cart + pendulum
        cart_pendulum: CartPendulum3D instance
        context: Default context
    """
    print(colored("\n" + "="*70, "cyan"))
    print(colored("BUILDING TEST SYSTEM: Cart-Pendulum 3D", "cyan", attrs=["bold"]))
    print(colored("="*70, "cyan"))
    
    # Build plant WITHOUT scene graph to avoid port connection issues
    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=0.0)
    
    # Create model instance
    model_instance = plant.AddModelInstance("cart_pendulum_model")
    
    # ========================================================================
    # CREATE CART-PENDULUM SYSTEM USING CartPendulum3D CLASS
    # ========================================================================
    cart_pendulum_config = create_cart_pendulum_config(
        cart_mass=TestConfig.CART_MASS,
        cart_size=TestConfig.CART_SIZE,
        cart_damping=0.0,
        pendulum_mass=TestConfig.PENDULUM_MASS,
        pendulum_length=TestConfig.PENDULUM_LENGTH,
        pendulum_radius=TestConfig.PENDULUM_RADIUS,
        pendulum_damping=TestConfig.PENDULUM_DAMPING,
        attachment_offset=(0.0, 0.0, 0.0),
        initial_cart_x=0.0,
        initial_cart_y=0.0,
        initial_pitch=0.0,
        initial_roll=0.0,
        name="cart_pendulum"
    )
    
    cart_pendulum = CartPendulum3D(
        config=cart_pendulum_config,
        visualize_cart=False  # No cart visualization for test
    )
    
    # Attach to plant
    cart_pendulum.attach_to_plant(plant, model_instance, register_visuals=False)
    
    # Finalize plant
    plant.Finalize()
    
    # For standalone plant (no diagram needed for linearization test)
    context = plant.CreateDefaultContext()
    
    print(colored(f"\n✓ System built: {plant.num_positions()} DOF", "green"))
    print(colored(f"  Cart DOF: 2 (x, y)", "cyan"))
    print(colored(f"  Pendulum DOF: 2 (pitch, roll)", "cyan"))
    print(colored(f"  Total positions: {plant.num_positions()}", "cyan"))
    print(colored(f"  Total velocities: {plant.num_velocities()}", "cyan"))
    print(colored(f"  Total states: {plant.num_multibody_states()}", "cyan"))
    print(colored(f"  Inputs (forces): 2 (F_x, F_y)", "cyan"))
    
    return plant, cart_pendulum, context


# ============================================================================
# ANALYTICAL LINEARIZATION (GROUND TRUTH)
# ============================================================================

def compute_analytical_linearization(cart_pendulum, theta_eq_deg, phi_eq_deg, plant):
    """
    Compute analytical linearization using Drake's built-in Linearize() function.
    
    This serves as the "analytical ground truth" computed by Drake's automatic
    differentiation, which we'll compare against the finite difference method.
    
    For cart-pendulum system:
    - State: [x_cart, y_cart, pitch, roll, ẋ_cart, ẏ_cart, pitch_dot, roll_dot] (8D)
    - Input: [F_x, F_y] (2D)
    
    Args:
        cart_pendulum: CartPendulum3D instance
        theta_eq_deg: Equilibrium polar angle (degrees) - pitch in gimbal
        phi_eq_deg: Equilibrium azimuthal angle (degrees) - roll in gimbal
        plant: MultibodyPlant instance
    
    Returns:
        A_analytical: 8x8 state matrix
        B_analytical: 8x2 input matrix
    """
    from pydrake.all import Linearize
    
    # Create fresh context for this equilibrium
    context = plant.CreateDefaultContext()
    
    # Set equilibrium state for cart-pendulum
    pitch_rad = np.deg2rad(theta_eq_deg)
    roll_rad = np.deg2rad(phi_eq_deg)
    
    # Cart at origin, stationary
    cart_pendulum.set_cart_state(context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0)
    
    # Pendulum at specified angles, stationary
    cart_pendulum.set_pendulum_state(context, pitch=pitch_rad, roll=roll_rad, 
                                     pitch_dot=0.0, roll_dot=0.0)
    
    # Zero forces at equilibrium
    plant.get_actuation_input_port().FixValue(context, np.zeros(2))
    
    # Use Drake's automatic differentiation linearization
    linear_system = Linearize(
        plant,
        context,
        input_port_index=plant.get_actuation_input_port().get_index(),
        output_port_index=plant.get_state_output_port().get_index()
    )
    
    A_analytical = linear_system.A()
    B_analytical = linear_system.B()
    
    return A_analytical, B_analytical


# ============================================================================
# NUMERICAL LINEARIZATION (TEST METHOD)
# ============================================================================

def compute_numerical_linearization(plant, cart_pendulum, context, epsilon=1e-6):
    """
    Compute numerical linearization using finite differences.
    
    Args:
        plant: MultibodyPlant instance
        cart_pendulum: CartPendulum3D instance
        context: Context at equilibrium
        epsilon: Finite difference step size
    
    Returns:
        A_numerical: nxn state matrix
        B_numerical: nxm input matrix
    """
    # Use cart_pendulum's finite difference method (delegates to pendulum)
    A_numerical, B_numerical = cart_pendulum.finite_difference_linearization(
        plant, context, epsilon=epsilon
    )
    
    return A_numerical, B_numerical


# ============================================================================
# COMPARISON AND VALIDATION
# ============================================================================

def compare_matrices(A_analytical, A_numerical, B_analytical, B_numerical, 
                    test_name, tolerance):
    """
    Compare analytical and numerical linearization matrices.
    
    Args:
        A_analytical: Analytical state matrix
        A_numerical: Numerical state matrix
        B_analytical: Analytical input matrix
        B_numerical: Numerical input matrix
        test_name: Name of test case
        tolerance: Comparison tolerance
    
    Returns:
        passed: True if test passed
    """
    print(colored(f"\n{'─'*70}", "yellow"))
    print(colored(f"COMPARING: {test_name}", "yellow", attrs=["bold"]))
    print(colored(f"{'─'*70}", "yellow"))
    
    # Check dimensions
    if A_analytical.shape != A_numerical.shape:
        print(colored(f"✗ DIMENSION MISMATCH (A matrix):", "red", attrs=["bold"]))
        print(colored(f"  Analytical: {A_analytical.shape}", "red"))
        print(colored(f"  Numerical: {A_numerical.shape}", "red"))
        return False
    
    if B_analytical.shape != B_numerical.shape:
        print(colored(f"✗ DIMENSION MISMATCH (B matrix):", "red", attrs=["bold"]))
        print(colored(f"  Analytical: {B_analytical.shape}", "red"))
        print(colored(f"  Numerical: {B_numerical.shape}", "red"))
        return False
    
    # Compute differences
    A_diff = np.abs(A_analytical - A_numerical)
    B_diff = np.abs(B_analytical - B_numerical)
    
    A_max_error = np.max(A_diff)
    A_mean_error = np.mean(A_diff)
    A_rms_error = np.sqrt(np.mean(A_diff**2))
    
    B_max_error = np.max(B_diff)
    B_mean_error = np.mean(B_diff)
    B_rms_error = np.sqrt(np.mean(B_diff**2))
    
    # Print matrices for debugging
    if A_max_error > tolerance or B_max_error > tolerance:
        print(colored("\nDEBUG - Analytical A:", "yellow"))
        print(A_analytical)
        print(colored("\nDEBUG - Numerical A:", "yellow"))
        print(A_numerical)
        print(colored("\nDEBUG - Analytical B:", "yellow"))
        print(B_analytical)
        print(colored("\nDEBUG - Numerical B:", "yellow"))
        print(B_numerical)
    
    # Print A matrix comparison
    print(colored("\nA Matrix (State Dynamics):", "cyan", attrs=["bold"]))
    print(colored(f"  Shape: {A_analytical.shape}", "cyan"))
    print(colored(f"  Max error:  {A_max_error:.2e}", "cyan"))
    print(colored(f"  Mean error: {A_mean_error:.2e}", "cyan"))
    print(colored(f"  RMS error:  {A_rms_error:.2e}", "cyan"))
    
    if A_max_error < tolerance:
        print(colored(f"  ✓ PASS (within tolerance {tolerance:.2e})", "green"))
        A_passed = True
    else:
        print(colored(f"  ✗ FAIL (exceeds tolerance {tolerance:.2e})", "red"))
        A_passed = False
        
        # Print worst elements
        worst_idx = np.unravel_index(np.argmax(A_diff), A_diff.shape)
        print(colored(f"  Worst element at [{worst_idx[0]}, {worst_idx[1]}]:", "red"))
        print(colored(f"    Analytical: {A_analytical[worst_idx]:.6e}", "red"))
        print(colored(f"    Numerical:  {A_numerical[worst_idx]:.6e}", "red"))
        print(colored(f"    Difference: {A_diff[worst_idx]:.6e}", "red"))
    
    # Print B matrix comparison
    print(colored("\nB Matrix (Input Influence):", "cyan", attrs=["bold"]))
    print(colored(f"  Shape: {B_analytical.shape}", "cyan"))
    print(colored(f"  Max error:  {B_max_error:.2e}", "cyan"))
    print(colored(f"  Mean error: {B_mean_error:.2e}", "cyan"))
    print(colored(f"  RMS error:  {B_rms_error:.2e}", "cyan"))
    
    if B_max_error < tolerance:
        print(colored(f"  ✓ PASS (within tolerance {tolerance:.2e})", "green"))
        B_passed = True
    else:
        print(colored(f"  ✗ FAIL (exceeds tolerance {tolerance:.2e})", "red"))
        B_passed = False
        
        # Print worst elements
        worst_idx = np.unravel_index(np.argmax(B_diff), B_diff.shape)
        print(colored(f"  Worst element at [{worst_idx[0]}, {worst_idx[1]}]:", "red"))
        print(colored(f"    Analytical: {B_analytical[worst_idx]:.6e}", "red"))
        print(colored(f"    Numerical:  {B_numerical[worst_idx]:.6e}", "red"))
        print(colored(f"    Difference: {B_diff[worst_idx]:.6e}", "red"))
    
    return A_passed and B_passed


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all linearization validation tests."""
    print(colored("\n" + "="*70, "green"))
    print(colored("PENDULUM LINEARIZATION VALIDATION TEST SUITE", "green", attrs=["bold"]))
    print(colored("="*70, "green"))
    print(colored("\nValidating Drake auto-diff vs finite difference linearization", "white"))
    print(colored(f"Finite difference epsilon: {TestConfig.EPSILON:.2e}", "white"))
    
    # Build test system
    plant, cart_pendulum, base_context = build_cart_pendulum_system()
    
    # Run tests at different equilibria
    results = []
    
    for theta_deg, phi_deg, description in TestConfig.TEST_EQUILIBRIA:
        print(colored(f"\n{'='*70}", "magenta"))
        print(colored(f"TEST CASE: {description}", "magenta", attrs=["bold"]))
        print(colored(f"  θ = {theta_deg}°, φ = {phi_deg}°", "magenta"))
        print(colored(f"{'='*70}", "magenta"))
        
        # Clone context for this test
        context = base_context.Clone()
        
        # For cart-pendulum system: state = [x, y, pitch, roll, ẋ, ẏ, pitch_dot, roll_dot]
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)
        
        # Set cart at rest at origin
        cart_pendulum.set_cart_state(context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0)
        
        # Set pendulum angles at equilibrium
        cart_pendulum.set_pendulum_state(context, pitch=theta_rad, roll=phi_rad,
                                        pitch_dot=0.0, roll_dot=0.0)
        
        # Set zero forces at equilibrium
        plant.get_actuation_input_port().FixValue(context, np.zeros(2))
        
        # Compute analytical linearization
        print(colored("\n[1/2] Computing Drake's automatic differentiation linearization...", "cyan"))
        A_analytical, B_analytical = compute_analytical_linearization(
            cart_pendulum, theta_deg, phi_deg, plant
        )
        print(colored(f"  ✓ Analytical: A({A_analytical.shape}), B({B_analytical.shape})", "green"))
        
        # Compute numerical linearization
        print(colored("\n[2/2] Computing numerical linearization...", "cyan"))
        # Actually, the finite_difference_linearization expects actuation input
        # Let's check if pendulum has actuators
        
        try:
            A_numerical, B_numerical = compute_numerical_linearization(
                plant, cart_pendulum, context, epsilon=TestConfig.EPSILON
            )
            print(colored(f"  ✓ Numerical: A({A_numerical.shape}), B({B_numerical.shape})", "green"))
            
            # Compare matrices
            # All tests use natural equilibria (hanging or inverted) with zero input
            tolerance = TestConfig.TOLERANCE_TIGHT
            
            passed = compare_matrices(
                A_analytical, A_numerical,
                B_analytical, B_numerical,
                description,
                tolerance
            )
            
            results.append((description, passed))
            
        except Exception as e:
            print(colored(f"\n✗ ERROR during numerical linearization: {e}", "red", attrs=["bold"]))
            print(colored(f"  This may be due to missing actuators in test system", "yellow"))
            results.append((description, False))
    
    # Print summary
    print(colored(f"\n{'='*70}", "green"))
    print(colored("TEST SUMMARY", "green", attrs=["bold"]))
    print(colored(f"{'='*70}", "green"))
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "green" if passed else "red"
        print(colored(f"  {status}: {test_name}", color))
    
    print(colored(f"\nTotal: {passed_tests}/{total_tests} tests passed", "white", attrs=["bold"]))
    
    if passed_tests == total_tests:
        print(colored("\n🎉 ALL TESTS PASSED! 🎉", "green", attrs=["bold"]))
        return 0
    else:
        print(colored(f"\n⚠ {total_tests - passed_tests} TESTS FAILED", "red", attrs=["bold"]))
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(colored("\n" + "╔" + "═"*68 + "╗", "blue"))
    print(colored("║" + " "*68 + "║", "blue"))
    print(colored("║" + "  PENDULUM LINEARIZATION VALIDATION TEST".center(68) + "║", "blue", attrs=["bold"]))
    print(colored("║" + " "*68 + "║", "blue"))
    print(colored("╚" + "═"*68 + "╝", "blue"))
    
    exit_code = run_all_tests()
    
    print(colored("\n" + "─"*70, "white"))
    print(colored("Test complete.", "white"))
    print(colored("─"*70 + "\n", "white"))
    
    sys.exit(exit_code)
