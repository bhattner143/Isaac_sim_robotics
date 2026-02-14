"""
Cart-Pendulum System - Drake Controller Architecture

═══════════════════════════════════════════════════════════════════════════════
TWO-SYSTEM ARCHITECTURE EXPLANATION
═══════════════════════════════════════════════════════════════════════════════

Classic underactuated system: Cart on track with inverted pendulum

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DRAKE DIAGRAM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  state[x,θ,ẋ,θ̇]  ┌──────────────────┐              │
│  │                  │────────────────────>│                  │              │
│  │  MultibodyPlant  │                     │   Controller     │              │
│  │   (Physics)      │<────────────────────│   (Control Law)  │              │
│  │                  │  force[F]           │                  │              │
│  └──────────────────┘                     └──────────────────┘              │
│         │                                                                   │
│         │ geometry                                                          │
│         v                                                                   │
│  ┌──────────────────┐                                                      │
│  │   SceneGraph     │───────> MeshcatVisualizer                           │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

SYSTEM 1: MultibodyPlant (Physics)
──────────────────────────────────
• Cart on 1D track (prismatic joint)
• Inverted pendulum (revolute joint)
• 2 DOF: x (cart position), θ (pendulum angle)
• 1 actuator: horizontal force on cart
• Classic underactuated control problem

SYSTEM 2: Controller
─────────────────────
• PD: Simple position control
• Energy Shaping: Swing-up controller
• LQR: Balancing around upright
• Computed Torque: Trajectory tracking

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from dataclasses import dataclass, field
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
    RevoluteJoint,
    PrismaticJoint,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    Role,
    
    # Geometry
    Box,
    Cylinder,
    Sphere,
    Rgba,
    CoulombFriction,
    
    # Controllers
    LinearQuadraticRegulator,
    
    # Mathematical utilities
    RotationMatrix,
    RigidTransform,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Cart-Pendulum with Drake Controllers')
parser.add_argument('--mode', type=str, 
                    choices=['pd', 'energy-shaping', 'lqr', 'computed-torque', 'standard-lqr', 'finite-horizon-lqr', 'scene-viz', 'compare-models'],
                    default='finite-horizon-lqr', 
                    help='Controller type')
parser.add_argument('--visualize', action='store_true', default=True, help='Enable visualization')
parser.add_argument('--initial_theta', type=float, default=None, 
                    help='Initial pendulum angle (degrees, 180=up, 0=down)')
parser.add_argument('--use-model-plant', type=bool, default=True,
                    help='Use separate model plant for computed torque (True) or use real plant (False)')
parser.add_argument('--plant-type', type=str, default='equations',
                    choices=['multibody', 'equations', 'linearized'],
                    help='Plant type: multibody (MultibodyPlant), equations (nonlinear), or linearized (equations 2.1 & 2.2)')
args, _ = parser.parse_known_args()

# Interactive input for initial angle if not provided (only when running as main script)
if args.initial_theta is None and __name__ == "__main__":
    print("\n" + "="*70)
    print(colored("CART-PENDULUM INTERACTIVE SETUP", 'cyan', attrs=['bold']))
    print("="*70)
    print("\nEnter initial pendulum angle:")
    print(colored("  0°   = Hanging down", 'red'))
    print(colored("  45°  = Tilted from down", 'yellow'))
    print(colored("  90°  = Horizontal", 'yellow'))
    print(colored("  180° = Upright (balanced)", 'green'))
    print(colored("  225° = Tilted from up", 'yellow'))
    print("-"*70)
    
    while True:
        try:
            angle_input = input(colored("\nInitial angle (degrees, press Enter for 0°): ", 'cyan')).strip()
            if angle_input == '':
                args.initial_theta = 0.0
                print(colored("✓ Using default: 0° (hanging down)", 'green'))
                break
            args.initial_theta = float(angle_input)
            print(colored(f"✓ Using θ = {args.initial_theta}°", 'green'))
            break
        except ValueError:
            print(colored("❌ Invalid input. Please enter a number (e.g., 45, 90, 180)", 'red'))
    print("="*70 + "\n")
elif args.initial_theta is None:
    # When imported as module, default to 0.0
    args.initial_theta = 0.0
else:
    print(colored(f"\n✓ Using command-line angle: θ = {args.initial_theta}°\n", 'green'))

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class CartPendulumPhysicsConfig:
    """Parameters for cart-pendulum system dynamics (models)."""
    # Cart parameters
    mass_cart: float = 3.0  # kg (M)
    width_cart: float = 0.4  # m (visualization)
    height_cart: float = 0.3  # m (visualization)
    depth_cart: float = 0.3  # m (visualization)
    damping_cart: float = 0.0  # N·s/m
    
    # Pendulum parameters
    mass_pendulum: float = 0.3  # kg (m)
    length_pendulum: float = 0.5  # m (l - center of mass)
    radius_pendulum: float = 0.04  # m (visualization)
    damping_pendulum: float = 0.0  # N·s/m
    
    # System parameters
    coupling_gain: float = 5.0  # G parameter
    gravity: float = 9.81  # m/s²
    
    # Impedance/arm parameters (for linearized plant)
    mass_arm: float = 2.0  # kg (M_arm)
    
    # Motor parameters
    motor_time_constant: float = 0.030  # s (τ)
    motor_noise_mean: float = 1e-4  # N
    motor_noise_std: float = 5e-3  # N
    
    # Track parameters
    track_length: float = 4.0  # m
    track_limit: float = 2.0  # m (position limits ±2m)


@dataclass
class StandardLQRConfig:
    """Parameters for Standard LQR (Continuous-Time LQR) with linearized plant."""
    # Cost matrices for 5D state: [x, φ, ẋ, φ̇, F]
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 100.0, 1.0, 10.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    x_goal: np.ndarray = field(default_factory=lambda: np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
    
    # Control limits
    u_min: float = -100.0
    u_max: float = 100.0

@dataclass
class FiniteHorizonLQRConfig:
    """Parameters for Finite-Horizon LQR with linearized plant."""
    # Original paers values did not work well for swing-up, so these are tuned for better performance
    Q: np.ndarray = field(default_factory=lambda: np.diag([100., 1000.,   10.,  100.,  0.1]))
    QN : np.ndarray = field(default_factory=lambda: np.diag([100., 1000.,   10.,  100.,  0.1]))#field(default_factory=lambda: np.diag([1e5, 0, 1e5, 0, 0]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    x_goal: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.0, 0.0, 0.0, 0.0]))
    horizon: float = 10.0  # seconds
    timestep: float = 0.01  # seconds (for discretization)


@dataclass
class PDControllerConfig:
    """Parameters for PD Controller."""
    kp_cart: float = 50.0  # Cart position gain
    kd_cart: float = 10.0  # Cart velocity gain
    kp_pend: float = 25.0  # Pendulum angle gain
    kd_pend: float = 5.0   # Pendulum velocity gain


@dataclass
class ComputedTorqueConfig:
    """Parameters for Computed Torque Controller."""
    kp: np.ndarray = field(default_factory=lambda: np.diag([100.0, 50.0]))  # [cart, pendulum]
    kd: np.ndarray = field(default_factory=lambda: np.diag([20.0, 10.0]))   # [cart, pendulum]


@dataclass
class SimulationConfig:
    """Global simulation parameters."""
    # Simulation timing
    timestep: float = 0.001  # s (1 kHz)
    simulation_time: float = 10.0  # s
    realtime_rate: float = 1.0  # 1.0 = real-time
    
    # Logging
    print_interval: float = 0.5  # s
    logging_interval: float = 0.02  # s (50 Hz)
    
    # Trajectory configuration
    trajectory_mode: str = 'cart-motion'  # balance, track, or cart-motion
    cart_start_position: float = -0.5  # m
    cart_end_position: float = 0.5  # m
    cart_motion_duration: float = 3.0  # s
    pendulum_start_angle: float = 0.0  # degrees (0° = down, 180° = up)
    cart_settle_time: float = 0.5  # s
    
    # Noise parameters
    sensory_delay: float = 0.05  # s
    control_dependent_noise_std: float = 1.0
    state_dependent_sensory_noise_std: float = 0.0
    additive_process_noise_std: float = 1e-4
    additive_sensory_noise_cov: np.ndarray = field(default_factory=lambda: np.diag([1e-5, 1e-5, 1e-5, 1e-5]))
    internal_estimator_noise_cov: np.ndarray = field(default_factory=lambda: np.diag([1e-8, 1e-8, 1e-8, 1e-8]))
    target_hold_steps: int = 50


# ============================================================================
# CREATE GLOBAL CONFIG INSTANCES
# ============================================================================

# Physics/Dynamics configuration (used by all models)
PHYSICS_CONFIG = CartPendulumPhysicsConfig()

# Controller-specific configs (user can modify these)
STANDARD_LQR_CONFIG = StandardLQRConfig()  # Used by StandardLQRController
FINITE_HORIZON_LQR_CONFIG = FiniteHorizonLQRConfig()  # Used by FiniteHorizonLQRController
PD_CONTROLLER_CONFIG = PDControllerConfig()  # Used by PDController
COMPUTED_TORQUE_CONFIG = ComputedTorqueConfig()  # Used by ComputedTorqueController

# Global simulation config
SIM_CONFIG = SimulationConfig()

# ============================================================================
# BACKWARD COMPATIBILITY: EXPOSE COMMONLY USED PARAMS
# ============================================================================

# Physics parameters
CART_MASS = PHYSICS_CONFIG.mass_cart
CART_WIDTH = PHYSICS_CONFIG.width_cart
CART_HEIGHT = PHYSICS_CONFIG.height_cart
CART_DEPTH = PHYSICS_CONFIG.depth_cart
CART_DAMPING = PHYSICS_CONFIG.damping_cart

PENDULUM_MASS = PHYSICS_CONFIG.mass_pendulum
PENDULUM_LENGTH = PHYSICS_CONFIG.length_pendulum
PENDULUM_RADIUS = PHYSICS_CONFIG.radius_pendulum
PENDULUM_TOTAL_LENGTH = PHYSICS_CONFIG.length_pendulum
PENDULUM_DAMPING = PHYSICS_CONFIG.damping_pendulum

COUPLING_GAIN = PHYSICS_CONFIG.coupling_gain
GRAVITY = PHYSICS_CONFIG.gravity
ARM_MASS = PHYSICS_CONFIG.mass_arm
MOTOR_TIME_CONSTANT = PHYSICS_CONFIG.motor_time_constant

TRACK_LENGTH = PHYSICS_CONFIG.track_length
TRACK_LIMIT = PHYSICS_CONFIG.track_limit

# Simulation parameters
CONTROLLER_MODE = args.mode
USE_MODEL_PLANT = args.use_model_plant
PLANT_TYPE = args.plant_type
TIMESTEP = SIM_CONFIG.timestep
SIMULATION_TIME = SIM_CONFIG.simulation_time
PRINT_INTERVAL = SIM_CONFIG.print_interval
LOGGING_INTERVAL = SIM_CONFIG.logging_interval
REALTIME_RATE = SIM_CONFIG.realtime_rate

# Trajectory parameters
TRAJECTORY_MODE = SIM_CONFIG.trajectory_mode
CART_START_POSITION = SIM_CONFIG.cart_start_position
CART_END_POSITION = SIM_CONFIG.cart_end_position
CART_MOTION_DURATION = SIM_CONFIG.cart_motion_duration
PENDULUM_START_ANGLE = SIM_CONFIG.pendulum_start_angle
CART_SETTLE_TIME = SIM_CONFIG.cart_settle_time

# Standard LQR parameters
STANDARD_LQR_LINEARIZED_Q = STANDARD_LQR_CONFIG.Q
STANDARD_LQR_LINEARIZED_R = STANDARD_LQR_CONFIG.R
STANDARD_LQR_LINEARIZED_X_GOAL = STANDARD_LQR_CONFIG.x_goal

# Finite-Horizon LQR parameters
FINITE_HORIZON_LQR_Q = FINITE_HORIZON_LQR_CONFIG.Q
FINITE_HORIZON_LQR_QN = FINITE_HORIZON_LQR_CONFIG.QN
FINITE_HORIZON_LQR_R = FINITE_HORIZON_LQR_CONFIG.R
FINITE_HORIZON_LQR_X_GOAL = FINITE_HORIZON_LQR_CONFIG.x_goal
FINITE_HORIZON_LQR_T = FINITE_HORIZON_LQR_CONFIG.horizon
FINITE_HORIZON_LQR_DT = FINITE_HORIZON_LQR_CONFIG.timestep


# PD Controller parameters
PD_KP_CART = PD_CONTROLLER_CONFIG.kp_cart
PD_KD_CART = PD_CONTROLLER_CONFIG.kd_cart
PD_KP_PEND = PD_CONTROLLER_CONFIG.kp_pend
PD_KD_PEND = PD_CONTROLLER_CONFIG.kd_pend

# Computed Torque Controller parameters
CT_KP = COMPUTED_TORQUE_CONFIG.kp
CT_KD = COMPUTED_TORQUE_CONFIG.kd

# Legacy parameters (for compatibility)
MOTOR_NOISE_MEAN = PHYSICS_CONFIG.motor_noise_mean
MOTOR_NOISE_STD = PHYSICS_CONFIG.motor_noise_std
SENSORY_DELAY = SIM_CONFIG.sensory_delay
CONTROL_DEPENDENT_NOISE_STD = SIM_CONFIG.control_dependent_noise_std
STATE_DEPENDENT_SENSORY_NOISE_STD = SIM_CONFIG.state_dependent_sensory_noise_std
ADDITIVE_PROCESS_NOISE_STD = SIM_CONFIG.additive_process_noise_std
ADDITIVE_SENSORY_NOISE_COV = SIM_CONFIG.additive_sensory_noise_cov
INTERNAL_ESTIMATOR_NOISE_COV = SIM_CONFIG.internal_estimator_noise_cov
TARGET_HOLD_STEPS = SIM_CONFIG.target_hold_steps
# ============================================================================
# CART-PENDULUM DYNAMICS FROM EQUATIONS (Equations 2.1 & 2.2)
# ============================================================================

class CartPendulumSystemByEqns(LeafSystem):
    """
    Cart-pendulum system using FULL NONLINEAR equations.
    
    Implements the standard cart-pendulum dynamics (real physics):
        (M + m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F
        lφ̈ + ẍcos(φ) + g·sin(φ) = 0
    
    Where:
        M = cart mass
        m = pendulum mass  
        l = pendulum length (to center of mass)
        g = gravity
        φ = pendulum angle (measured from upright)
        x = cart position
        F = applied force (F_inter + F_pert)
    
    Note: Uses NONLINEAR dynamics (actual physics), NOT linearized.
    The paper's equations 2.1 & 2.2 are linearized approximations used
    for analysis, but this plant simulates the true dynamics.
    
    State: [x, φ, ẋ, φ̇]
    Input: [F_inter] (F_pert can be added as second input)
    """
    
    def __init__(self, M=CART_MASS, m=PENDULUM_MASS, l=PENDULUM_LENGTH, 
                 g=GRAVITY, G=COUPLING_GAIN):
        """
        Initialize cart-pendulum dynamics.
        
        Args:
            M: Cart mass (kg)
            m: Pendulum mass (kg)
            l: Pendulum length to COM (m)
            g: Gravity (m/s²)
            G: Coupling parameter
        """
        LeafSystem.__init__(self)
        
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.G = G
        
        # State: [x, φ, ẋ, φ̇]
        self.DeclareContinuousState(4)  # 4 state variables
        
        # Input: [F_inter] (can add F_pert later)
        self.DeclareVectorInputPort("force", BasicVector(1))
        
        # Output: full state (depends ONLY on continuous state, NOT on input)
        # This avoids algebraic loop - output doesn't directly depend on input
        self.DeclareVectorOutputPort(
            "state", 
            BasicVector(4), 
            self.CopyStateOut,
            {self.xc_ticket()}  # Depends only on continuous state
        )
        
        print(colored(f"✓ CartPendulumSystemByEqns initialized:", 'green'))
        print(colored(f"  M={M} kg, m={m} kg, l={l} m, g={g} m/s²", 'cyan'))
        print(colored(f"  Using NONLINEAR equations (real physics):", 'cyan'))
        print(colored(f"    (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F", 'cyan'))
        print(colored(f"    lφ̈ + ẍcos(φ) + g·sin(φ) = 0", 'cyan'))
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute state derivatives using FULL NONLINEAR cart-pendulum equations.
        
        Nonlinear equations (real physics):
            (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F
            lφ̈ + ẍcos(φ) + g·sin(φ) = 0
        
        This is a 2x2 linear system in [ẍ, φ̈]:
            [M+m    ml·cos(φ)] [ẍ  ]   [F + ml·φ̇²·sin(φ)]
            [cos(φ)    l     ] [φ̈ ] = [-g·sin(φ)        ]
        
        Solved using numpy.linalg.solve for numerical stability.
        """
        # State: [x, phi, x_dot, phi_dot]
        x, phi, x_dot, phi_dot = (
            context.get_continuous_state_vector().CopyToVector()
        )

        # Input force(s)
        F_inter = float(self.get_input_port(0).Eval(context)[0])
        F_pert = 0.0  # Can be added as second input if needed
        F = F_inter + F_pert

        # Parameters
        M = self.M  
        m = self.m
        l = self.l
        g = self.g
        G = self.G

        import numpy as np
        c = np.cos(phi)
        s = np.sin(phi)

        # Build linear system A * [x_ddot, phi_ddot]^T = b
        # From nonlinear equations:
        #   (M+m)ẍ + ml·φ̈·cos(φ) = F + ml·φ̇²·sin(φ)
        #   G·ẍ·cos(φ) + l·φ̈ = -g·sin(φ)
        A = np.array([
            [M + m,  m * l * c],
            [G * c,  l]
        ], dtype=float)

        b = np.array([
            F + m * l * (phi_dot ** 2) * s,
            -g * s
        ], dtype=float)

        x_ddot, phi_ddot = np.linalg.solve(A, b)

        # Time derivatives: [x_dot, phi_dot, x_ddot, phi_ddot]
        derivatives.get_mutable_vector().SetFromVector([
            x_dot,
            phi_dot,
            x_ddot,
            phi_ddot
        ])
    
    def CopyStateOut(self, context, output):
        """Output the full state vector."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)
    
    def SetInitialConditions(self, context, x0, phi0, x_dot0=0.0, phi_dot0=0.0):
        """
        Set initial state.
        
        Args:
            context: System context
            x0: Initial cart position (m)
            phi0: Initial pendulum angle (rad)
            x_dot0: Initial cart velocity (m/s)
            phi_dot0: Initial pendulum angular velocity (rad/s)
        """
        state = context.get_mutable_continuous_state_vector()
        state.SetFromVector([x0, phi0, x_dot0, phi_dot0])


# ============================================================================
# CART-PENDULUM LINEARIZED DYNAMICS WITH IMPEDANCE (Equation 2.7 from paper)
# ============================================================================

class CartPendulumSystemLinearizedWithMuscleDynamics(LeafSystem):
    """
    Linearized cart-pendulum with motor dynamics and impedance (Equation 2.7).
    
    State-space model:
        ẋ = Ax + Bu
        y = Hx
    
    State vector (5D):
        x = [x, φ, ẋ, φ̇, F]^T
    
    Where:
        x = cart position
        φ = pendulum angle
        F = control force (internal motor state)
        
    System matrices:
        A = 5×5 dynamics matrix (see equation 2.7, with F_pert=0)
        B = [0, 0, 0, 0, 1/τ]^T (motor input)
        H = I_5×5 (full state observation)
    
    Parameters:
        α = m + (M + M_arm) - mG
        τ = motor time constant
        M_arm = virtual arm/reference mass (enables impedance control)
    
    Key features:
    - First-order motor dynamics: Ḟ = -F/τ + u/τ
    - Impedance through M_arm parameter
    - No external perturbations (F_pert = 0 always)
    """
    
    def __init__(self, M=CART_MASS, m=PENDULUM_MASS, l=PENDULUM_LENGTH, 
                 g=GRAVITY, G=COUPLING_GAIN, tau=MOTOR_TIME_CONSTANT,
                 M_arm=ARM_MASS):
        """
        Initialize linearized cart-pendulum with impedance.
        
        Args:
            M: Cart mass (kg)
            m: Pendulum mass (kg)
            l: Pendulum length to COM (m)
            g: Gravity (m/s²)
            G: Coupling parameter
            tau: Motor time constant (s)
            M_arm: Virtual arm mass for impedance (kg)
        """
        LeafSystem.__init__(self)
        
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.G = G
        self.tau = tau
        self.M_arm = M_arm
        
        # Compute α parameter: α = m + (M + M_arm) - mG
        self.alpha = m + (M + M_arm) - m * G
        
        # Compute and store A and B matrices (constant for linear system)
        self.A, self.B = self._compute_system_matrices()
        
        # State: [x, φ, ẋ, φ̇, F] (5D)
        self.DeclareContinuousState(5)
        
        # Input: [u] - motor command (desired force rate)
        self.DeclareVectorInputPort("motor_command", BasicVector(1))
        
        # Output: state [x, φ, ẋ, φ̇, F]
        self.DeclareVectorOutputPort(
            "state",
            BasicVector(5),
            self.CopyStateOut,
            {self.xc_ticket()}  # Depends only on continuous state
        )
        
        print(colored(f"✓ CartPendulumSystemLinearizedWithMuscleDynamics initialized:", 'green'))
        print(colored(f"  M={M} kg, m={m} kg, l={l} m, g={g} m/s²", 'cyan'))
        print(colored(f"  G={G}, τ={tau} s, M_arm={M_arm} kg", 'cyan'))
        print(colored(f"  α = m + (M + M_arm) - mG = {self.alpha:.3f}", 'yellow'))
        print(colored(f"  Using equation 2.7 with MOTOR DYNAMICS and IMPEDANCE", 'yellow'))
        print(colored(f"    State: [x, φ, ẋ, φ̇, F]^T (5D, F_pert=0)", 'cyan'))
    
    def _compute_system_matrices(self):
        """
        Compute A and B matrices once (equation 2.7 with F_pert=0).
        
        A matrix (5×5):
            [0,          0,           1,  0,   0    ]
            [0,          0,           0,  1,   0    ]
            [0,        mg/α,          0,  0,  1/α   ]
            [0,  -g/l(1+Gm/α),       0,  0,  -G/lα ]
            [0,          0,           0,  0,  -1/τ  ]
        
        B matrix (5×1):
            [0, 0, 0, 0, 1/τ]^T
        
        Returns:
            A: 5×5 numpy array
            B: 5×1 numpy array
        """
        m = self.m
        g = self.g
        l = self.l
        G = self.G
        tau = self.tau
        alpha = self.alpha
        
        # Build A matrix according to equation 2.7 (5×5, no F_pert)
        A = np.zeros((5, 5))
        
        # Row 1: ẋ = ẋ
        A[0, 2] = 1.0
        
        # Row 2: φ̇ = φ̇
        A[1, 3] = 1.0
        
        # Row 3: ẍ = (mg/α)φ + (1/α)F
        A[2, 1] = m * g / alpha
        A[2, 4] = 1.0 / alpha
        
        # Row 4: φ̈ = -(g/l)(1 + Gm/α)φ - (G/lα)F
        A[3, 1] = -(g / l) * (1.0 + G * m / alpha)
        A[3, 4] = -G / (l * alpha)
        
        # Row 5: Ḟ = -(1/τ)F + (1/τ)u (motor dynamics)
        A[4, 4] = -1.0 / tau
        
        # Build B matrix
        B = np.zeros((5, 1))
        B[4, 0] = 1.0 / tau  # Motor input affects F
        
        return A, B 
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute ẋ = Ax + Bu using precomputed A and B matrices.
        
        This is efficient - matrices are computed once in __init__.
        """
        # Get current state: [x, φ, ẋ, φ̇, F, F_pert]
        state = context.get_continuous_state_vector().CopyToVector()
        
        # Get motor command input
        u = self.get_input_port(0).Eval(context)[0]
        
        # Compute state derivatives: ẋ = Ax + Bu
        x_dot_vec = self.A @ state + self.B.flatten() * u
        
        derivatives.get_mutable_vector().SetFromVector(x_dot_vec)
    
    def CopyStateOut(self, context, output):
        """Output the full state vector."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)
    
    def SetInitialConditions(self, context, x0=0.0, phi0=0.0, x_dot0=0.0, 
                           phi_dot0=0.0, F0=0.0):
        """
        Set initial state.
        
        Args:
            context: System context
            x0: Initial cart position (m)
            phi0: Initial pendulum angle (rad)
            x_dot0: Initial cart velocity (m/s)
            phi_dot0: Initial pendulum angular velocity (rad/s)
            F0: Initial control force (N)
        """
        state = context.get_mutable_continuous_state_vector()
        state.SetFromVector([x0, phi0, x_dot0, phi_dot0, F0])


# ============================================================================
# CART-PENDULUM SYSTEM
# ============================================================================

class CartPendulumSystem:
    """
    Cart-Pendulum Plant Builder.
    
    RESPONSIBILITY: Build the physics plant ONLY
    - Creates MultibodyPlant with cart and pendulum
    
    Does NOT handle:
    - Diagram building/wiring
    - Visualization
    - Simulation execution
    
    Those are handled by DrakeSceneManager.
    """
    
    def __init__(self, builder: DiagramBuilder):
        """Initialize plant builder.
        
        Args:
            builder: Drake DiagramBuilder (passed in from SceneManager)
        """
        self.builder = builder
        self.plant = None
        self.scene_graph = None
    
    def build_plant(self):
        """Build the cart-pendulum MultibodyPlant."""
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("Building Cart-Pendulum System", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        # Create plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=TIMESTEP
        )
        
        # Create cart (box sliding on track)
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = self.plant.AddRigidBody("cart", cart_inertia)
        
        # Add cart geometry
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        self.plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        self.plant.RegisterCollisionGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_collision",
            CoulombFriction(0.9, 0.8)
        )
        
        # Add prismatic joint (cart slides along x-axis)
        self.plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                self.plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],  # Slides along x-axis
                -TRACK_LIMIT,  # Lower limit
                TRACK_LIMIT,   # Upper limit
                damping=CART_DAMPING
            )
        )
        
        # Create pendulum using point mass at distance L from pivot
        # For a point mass m at distance L: I_pivot = m*L^2
        # Drake needs UnitInertia (inertia per unit mass)
        I_about_pivot = PENDULUM_LENGTH**2  # Per unit mass
        
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),  # COM location
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)  # Unit inertia about pivot
        )
        pendulum_body = self.plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Add pendulum geometry (cylinder)
        pendulum_shape = Cylinder(PENDULUM_RADIUS, PENDULUM_TOTAL_LENGTH)
        self.plant.RegisterVisualGeometry(
            pendulum_body, 
            RigidTransform(p=[0, 0, -PENDULUM_TOTAL_LENGTH/2]),
            pendulum_shape, "pendulum_visual",
            np.array([0.8, 0.1, 0.1, 1.0])
        )
        
        # Add ball at the end of pendulum for better visibility
        ball_shape = Sphere(0.08)
        self.plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(p=[0, 0, -PENDULUM_TOTAL_LENGTH]),
            ball_shape, "pendulum_ball",
            np.array([0.9, 0.2, 0.2, 1.0])
        )
        
        # Add revolute joint (pendulum rotates about y-axis)
        self.plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],  # Rotation axis (y-axis)
                damping=PENDULUM_DAMPING
            )
        )
        
        # Add track visualization (more visible)
        track_shape = Box(TRACK_LENGTH, 0.1, 0.1)
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            RigidTransform(p=[0, 0, -CART_HEIGHT/2 - 0.1]),
            track_shape, "track_visual",
            np.array([0.5, 0.5, 0.5, 0.8])
        )
        
        # Add actuator (force on cart)
        self.plant.AddJointActuator("cart_force", self.plant.GetJointByName("cart_slider"))
        
        # Finalize plant
        self.plant.Finalize()
        
        print(colored(f"✓ Cart-Pendulum plant created:", 'green'))
        print(colored(f"  Cart mass: {CART_MASS} kg", 'cyan'))
        print(colored(f"  Pendulum mass: {PENDULUM_MASS} kg", 'cyan'))
        print(colored(f"  Pendulum length: {PENDULUM_LENGTH} m", 'cyan'))
        print(colored(f"  DOF: {self.plant.num_positions()}", 'cyan'))
        print(colored(f"  Actuators: {self.plant.num_actuators()}", 'cyan'))
    
    def create_model_for_controller(self):
        """Create a separate dynamics model for controller (model-plant separation)."""
        print(colored("\nCreating Controller Model (separate from plant)", 'yellow', attrs=['bold']))
        
        # Create a separate plant for the controller's dynamics model
        model_plant = MultibodyPlant(time_step=0.0)  # Continuous-time for dynamics calculations
        
        # Create identical cart-pendulum system
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = model_plant.AddRigidBody("cart", cart_inertia)
        
        # Prismatic joint
        model_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                model_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -TRACK_LIMIT,
                TRACK_LIMIT,
                damping=CART_DAMPING
            )
        )
        
        # Pendulum (same configuration as main plant)
        I_about_pivot = PENDULUM_LENGTH**2  # Per unit mass
        
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = model_plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Revolute joint
        model_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=PENDULUM_DAMPING
            )
        )
        
        # Actuator
        model_plant.AddJointActuator("cart_force", model_plant.GetJointByName("cart_slider"))
        
        # Finalize model
        model_plant.Finalize()
        
        print(colored(f"✓ Controller model created (identical dynamics)", 'green'))
        print(colored(f"  Model DOF: {model_plant.num_positions()}", 'cyan'))
        print(colored(f"  Model purpose: Inverse dynamics calculations", 'cyan'))
        
        return model_plant  

# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
    Scene Manager for Drake simulation with Diagram-based controller.
    
    Manages:
    - Diagram construction (plant + controller wiring)
    - Controller creation and configuration
    - Simulator setup and execution
    - Visualization and data logging
    
    Pattern: Following cup manipulator architecture
    - CartPendulumSystem: Minimal class for plant setup only
    - DrakeSceneManager: Orchestrates all simulation aspects
    """
    
    def __init__(self, controller_mode: str = 'pd', plant_type: str = 'multibody', visualize: bool = True):
        """Initialize scene manager."""
        self.controller_mode = controller_mode
        self.plant_type = plant_type
        self.visualize = visualize
        
        # Drake objects
        self.builder = DiagramBuilder()
        self.plant = None
        self.scene_graph = None
        self.controller = None
        self.meshcat = None
        self.diagram = None
        self.simulator = None
        
        # Plant builder (physics only)
        self.system = CartPendulumSystem(self.builder)
        
        # Data logging
        self.time_log = []
        self.state_log = []
        self.force_log = []
        self.desired_state_log = []
        self.error_log = []
        
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("DrakeSceneManager Initialization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"  Controller Mode: {controller_mode}", 'yellow'))
        print(colored(f"  Plant Type: {plant_type}", 'yellow'))
        print(colored(f"  Visualization: {'Enabled' if visualize else 'Disabled'}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def setup_drake_system(self):
        """Step 1: Build the plant."""
        print(colored("\n[STEP 1/5] Setting up Drake system...", "yellow", attrs=["bold"]))
        
        # Build the cart-pendulum plant
        self.system.build_plant()
        
        # Get references to plant and scene graph
        self.plant = self.system.plant
        self.scene_graph = self.system.scene_graph
        
        print(colored("✓ Plant configured", "green"))
        print(colored(f"  DOF: {self.plant.num_positions()}", "cyan"))
        print(colored(f"  Actuators: {self.plant.num_actuators()}", "cyan"))
    
    def add_controller(self):
        """Step 2: Add controller to the diagram."""
        print(colored("\n[STEP 2/5] Adding controller...", "yellow", attrs=["bold"]))
        print(colored(f"  Controller: {self.controller_mode}", "cyan"))
        
        if self.controller_mode == 'scene-viz':
            from pydrake.systems.primitives import ConstantVectorSource
            self.controller = self.builder.AddSystem(
                ConstantVectorSource(np.zeros(1))
            )
            print(colored(f"✓ Zero-force controller (visualization only)", 'green'))
            self.builder.Connect(
                self.controller.get_output_port(0),
                self.plant.get_actuation_input_port()
            )
            return
        
        traj_gen = TrajectoryGenerator(mode='balance' if self.controller_mode in ['lqr', 'energy-shaping'] else 'swing')
        
        if self.controller_mode == 'pd':
            self.controller = self.builder.AddSystem(
                PDController(PD_KP_CART, PD_KD_CART, PD_KP_PEND, PD_KD_PEND, traj_gen)
            )
        
        elif self.controller_mode == 'computed-torque':
            model = self.system.create_model_for_controller()
            q_start = np.array([CART_START_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
            q_goal = np.array([CART_END_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
            
            traj_gen_ct = MinJerkTrajectoryGenerator(
                q_start=q_start,
                q_goal=q_goal,
                duration=CART_MOTION_DURATION,
                settle_time=CART_SETTLE_TIME
            )
            
            print(colored(f"  Trajectory: MinJerk cart motion", 'cyan'))
            print(colored(f"    Start: x={CART_START_POSITION:.1f}m, θ={PENDULUM_START_ANGLE:.0f}°", 'cyan'))
            print(colored(f"    Goal:  x={CART_END_POSITION:.1f}m, θ={PENDULUM_START_ANGLE:.0f}°", 'cyan'))
            print(colored(f"    Settle: {CART_SETTLE_TIME:.1f}s, Motion: {CART_MOTION_DURATION:.1f}s", 'cyan'))
            
            self.controller = self.builder.AddSystem(
                ComputedTorqueController(self.plant, model, CT_KP, CT_KD, traj_gen_ct, use_model=USE_MODEL_PLANT)
            )
        
        else:
            raise ValueError(f"Unknown controller mode: {self.controller_mode}")
        
        # Wire controller to plant
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.controller.get_input_port(0)
        )
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.plant.get_actuation_input_port()
        )
        
        print(colored(f"✓ Controller wired to plant", 'green'))
    
    def setup_visualization(self):
        """Step 3: Setup Meshcat visualization."""
        print(colored("\n[STEP 3/5] Setting up visualization...", "yellow", attrs=["bold"]))
        
        if not self.visualize:
            print(colored("✓ Visualization disabled", "green"))
            return
        
        if self.scene_graph is None:
            print(colored(f"⚠ Visualization not available (no geometry)", 'yellow'))
            return
        
        self.meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat
        )
        
        meshcat_url = self.meshcat.web_url()
        print(colored(f"✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {meshcat_url}", 'cyan'))
        print(colored(f"  👉 Open this URL in your browser!", "yellow", attrs=["bold"]))
        
        # Try to open in default browser
        try:
            import webbrowser
            webbrowser.open(meshcat_url)
            print(colored(f"  ✓ Attempting to open in browser...", "cyan"))
        except Exception as e:
            print(colored(f"  ⚠ Could not auto-open browser: {e}", "yellow"))
    
    def build_diagram(self):
        """Step 4: Wire systems and build diagram."""
        print(colored("\n[STEP 4/5] Building diagram...", "yellow", attrs=["bold"]))
        self.diagram = self.builder.Build()
        print(colored("✓ Diagram built and all systems wired", "green"))
    
    def create_simulator(self):
        """Step 5: Create and initialize simulator."""
        print(colored("\n[STEP 5/5] Creating simulator...", "yellow", attrs=["bold"]))
        
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(REALTIME_RATE)
        
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        initial_x = 0.0
        initial_theta = np.deg2rad(args.initial_theta)
        
        if CONTROLLER_MODE == 'computed-torque' and TRAJECTORY_MODE == 'cart-motion':
            initial_x = CART_START_POSITION
            initial_theta = np.deg2rad(PENDULUM_START_ANGLE)
            print(colored(f"  Cart-motion mode: starting at x={initial_x}m, θ={PENDULUM_START_ANGLE}°", 'yellow'))
        
        self.plant.SetPositions(plant_context, [initial_x, initial_theta])
        self.plant.SetVelocities(plant_context, [0.0, 0.0])
        
        self.diagram.ForcedPublish(context)
        
        print(colored(f"✓ Simulator created and initialized", 'green'))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("Running Simulation", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        t_next_print = 0.0
        t_next_log = 0.0
        t_next_viz = 0.0
        viz_interval = 0.05
        
        while context.get_time() < SIMULATION_TIME:
            self.simulator.AdvanceTo(context.get_time() + TIMESTEP)
            
            t = context.get_time()
            
            if t >= t_next_viz and self.meshcat is not None and hasattr(self, 'diagram'):
                self.diagram.ForcedPublish(context)
                t_next_viz += viz_interval
            
            if t >= t_next_print:
                state = self.plant.GetPositionsAndVelocities(plant_context)
                x, theta, x_dot, theta_dot = state
                theta_deg = np.rad2deg(theta)
                print(f"[{t:5.2f}s/{SIMULATION_TIME}s {int(100*t/SIMULATION_TIME):3d}%] "
                      f"x={x:6.3f}m θ={theta_deg:7.2f}° | "
                      f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(theta_dot):7.2f}°/s")
                
                t_next_print += PRINT_INTERVAL
            
            if t >= t_next_log:
                self.time_log.append(t)
                current_state = self.plant.GetPositionsAndVelocities(plant_context).copy()
                self.state_log.append(current_state)
                
                controller_context = self.controller.GetMyContextFromRoot(context)
                force = self.controller.get_output_port(0).Eval(controller_context)
                self.force_log.append(force[0])
                
                if CONTROLLER_MODE == 'computed-torque':
                    traj_gen = self.controller.trajectory_generator
                    q_d, v_d, a_d = traj_gen.compute_trajectory(t)
                    desired_state = np.concatenate([q_d, v_d])
                    self.desired_state_log.append(desired_state)
                    
                    q = current_state[:2]
                    v = current_state[2:]
                    theta = q[1]
                    theta_d = q_d[1]
                    theta_normalized = np.arctan2(np.sin(theta), np.cos(theta))
                    theta_d_normalized = np.arctan2(np.sin(theta_d), np.cos(theta_d))
                    
                    errors = np.array([
                        q_d[0] - q[0],
                        theta_d_normalized - theta_normalized,
                        v_d[0] - v[0],
                        v_d[1] - v[1]
                    ])
                    self.error_log.append(errors)
                
                t_next_log += LOGGING_INTERVAL
        
        print(colored(f"\n✓ Simulation completed successfully!", 'green', attrs=['bold']))
    
    def run_scene_viz(self):
        """Run interactive scene visualization."""
        if self.meshcat is None or self.scene_graph is None:
            print(colored("❌ Scene visualization requires MultibodyPlant geometry", 'red'))
            return
        
        print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
        print(colored("Interactive Scene Visualization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan', attrs=['bold']))
        
        print(colored("\nVisualization Mode: Interactive Static Scene", 'yellow'))
        print(colored("  - No physics simulation", 'yellow'))
        print(colored("  - Manual state control via terminal", 'yellow'))
        print(colored("  - Type 'q' to exit\n", 'yellow'))
        
        print(colored(f"\n✓ Meshcat URL: {self.meshcat.web_url()}", 'green', attrs=['bold']))
        print(colored("  👉 Open this URL in your browser to view the scene\n", 'yellow', attrs=['bold']))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        self.diagram.ForcedPublish(context)
        
        state = self.plant.GetPositionsAndVelocities(plant_context)
        x, theta, x_dot, theta_dot = state
        print(colored(f"\nInitial State:", 'magenta', attrs=['bold']))
        print(colored(f"  Cart position: {x:+.3f} m", 'cyan'))
        print(colored(f"  Pendulum angle: {np.rad2deg(theta):+.2f}° ({theta:+.4f} rad)", 'cyan'))
        print(colored(f"  Cart velocity: {x_dot:+.3f} m/s", 'cyan'))
        print(colored(f"  Pendulum velocity: {np.rad2deg(theta_dot):+.2f}°/s ({theta_dot:+.4f} rad/s)", 'cyan'))
        
        print("\n" + "=" * 70)
        print("Interactive State Control")
        print("=" * 70)
        print(f"\nEnter cart position and pendulum angle (space-separated):")
        print(f"  Format: <cart_x_meters> <pendulum_angle_degrees>")
        print(f"  Examples:")
        print(colored(f"    0 0      ", 'yellow') + "- Cart at center, pendulum hanging down")
        print(colored(f"    0.5 45   ", 'yellow') + "- Cart at 0.5m, pendulum at 45°")
        print(colored(f"    -1.0 180 ", 'yellow') + "- Cart at -1m, pendulum upright")
        print(f"\n  Type 'q' or 'quit' to exit")
        print("=" * 70 + "\n")
        
        try:
            while True:
                user_input = input(f"\nState [x(m), θ(°)]: ").strip()
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    break
                
                try:
                    values = [float(v.strip()) for v in user_input.split()]
                    if len(values) != 2:
                        print(colored(f"❌ Error: Expected 2 values, got {len(values)}", 'red'))
                        continue
                    
                    x_desired, theta_deg = values
                    theta_desired = np.deg2rad(theta_deg)
                    
                    self.plant.SetPositions(plant_context, [x_desired, theta_desired])
                    self.plant.SetVelocities(plant_context, [0.0, 0.0])
                    self.diagram.ForcedPublish(context)
                    
                    state = self.plant.GetPositionsAndVelocities(plant_context)
                    x_actual, theta_actual, _, _ = state
                    theta_actual_deg = np.rad2deg(theta_actual)
                    
                    print(colored(f"\n← Updated state:", 'cyan'))
                    print(colored(f"    Cart position: {x_actual:+.3f} m", 'cyan'))
                    print(colored(f"    Pendulum angle: {theta_actual_deg:+.2f}°", 'cyan'))
                    
                except ValueError:
                    print(colored(f"❌ Error: Invalid input", 'red'))
        
        except KeyboardInterrupt:
            print(colored("\n\n✓ Scene visualization closed by user", 'green'))
        
        print(colored("\n" + "="*70, 'green'))
        print(colored("Scene visualization complete!", 'green', attrs=['bold']))
        print(colored("="*70 + "\n", 'green'))
    
    def extract_data(self):
        """Extract and print data summary."""
        if len(self.time_log) == 0:
            print(colored("No data to extract", 'yellow'))
            return
        
        print(colored(f"Data logged: {len(self.time_log)} samples", 'cyan'))
    
    def plot_results(self):
        """Plot simulation results."""
        if len(self.time_log) == 0:
            return
        
        times = np.array(self.time_log)
        states = np.array(self.state_log)
        forces = np.array(self.force_log)
        
        if CONTROLLER_MODE == 'computed-torque' and len(self.desired_state_log) > 0:
            desired_states = np.array(self.desired_state_log)
            errors = np.array(self.error_log)
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            axes[0,0].plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
            axes[0,0].plot(times, desired_states[:, 0], 'r--', linewidth=2, label='Desired')
            axes[0,0].set_ylabel('Cart Position (m)', fontsize=12)
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            axes[0,0].set_title('Cart Position Tracking', fontweight='bold')
            
            axes[0,1].plot(times, errors[:, 0]*1000, 'r-', linewidth=2)
            axes[0,1].set_ylabel('Position Error (mm)', fontsize=12)
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0,1].set_title(f'Cart Error (RMS: {np.sqrt(np.mean(errors[:,0]**2))*1000:.2f} mm)', fontweight='bold')
            
            axes[1,0].plot(times, np.rad2deg(states[:, 1]), 'b-', linewidth=2, label='Actual')
            axes[1,0].plot(times, np.rad2deg(desired_states[:, 1]), 'r--', linewidth=2, label='Desired')
            axes[1,0].set_ylabel('Pendulum Angle (°)', fontsize=12)
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
            axes[1,0].set_title('Pendulum Angle Tracking', fontweight='bold')
            
            axes[1,1].plot(times, np.rad2deg(errors[:, 1]), 'r-', linewidth=2)
            axes[1,1].set_ylabel('Angle Error (°)', fontsize=12)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1,1].set_title(f'Pendulum Error (RMS: {np.sqrt(np.mean(errors[:,1]**2))*180/np.pi:.2f}°)', fontweight='bold')
            
            axes[2,0].plot(times, states[:, 2], 'b-', linewidth=2, label='Cart velocity')
            axes[2,0].plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2, label='Pendulum velocity')
            axes[2,0].set_ylabel('Velocities (m/s, °/s)', fontsize=12)
            axes[2,0].set_xlabel('Time (s)', fontsize=12)
            axes[2,0].grid(True, alpha=0.3)
            axes[2,0].legend(fontsize=9)
            axes[2,0].set_title('Velocities', fontweight='bold')
            
            axes[2,1].plot(times, forces, 'g-', linewidth=2)
            axes[2,1].set_ylabel('Control Force (N)', fontsize=12)
            axes[2,1].set_xlabel('Time (s)', fontsize=12)
            axes[2,1].grid(True, alpha=0.3)
            axes[2,1].set_title('Control Effort', fontweight='bold')
            
            print(colored("\n" + "="*70, 'cyan'))
            print(colored("TRACKING PERFORMANCE METRICS", 'cyan', attrs=['bold']))
            print(colored("="*70, 'cyan'))
            print(colored(f"Cart Position Error:", 'yellow'))
            print(colored(f"  RMS: {np.sqrt(np.mean(errors[:,0]**2))*1000:.2f} mm", 'cyan'))
            print(colored(f"Pendulum Angle Error:", 'yellow'))
            print(colored(f"  RMS: {np.sqrt(np.mean(errors[:,1]**2))*180/np.pi:.2f}°", 'cyan'))
            print(colored("="*70 + "\n", 'cyan'))
        
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            ax_cart_pos = fig.add_subplot(gs[0, 0])
            ax_cart_vel = fig.add_subplot(gs[1, 0])
            ax_pend_angle = fig.add_subplot(gs[0, 1])
            ax_pend_vel = fig.add_subplot(gs[1, 1])
            ax_control = fig.add_subplot(gs[2, :])
            
            ax_cart_pos.plot(times, states[:, 0], 'b-', linewidth=2.5, label='Position')
            ax_cart_pos.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
            ax_cart_pos.grid(True, alpha=0.3)
            ax_cart_pos.legend(loc='upper right', fontsize=11)
            ax_cart_pos.set_title('CART POSITION', fontsize=13, fontweight='bold', color='darkblue')
            
            ax_cart_vel.plot(times, states[:, 2], 'b-', linewidth=2.5, label='Velocity')
            ax_cart_vel.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
            ax_cart_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_cart_vel.grid(True, alpha=0.3)
            ax_cart_vel.legend(loc='upper right', fontsize=11)
            ax_cart_vel.set_title('CART VELOCITY', fontsize=13, fontweight='bold', color='darkblue')
            
            pend_angle_deg = np.rad2deg(states[:, 1])
            ax_pend_angle.plot(times, pend_angle_deg, 'r-', linewidth=2.5, label='Angle')
            ax_pend_angle.set_ylabel('Angle (°)', fontsize=12, fontweight='bold')
            ax_pend_angle.grid(True, alpha=0.3)
            ax_pend_angle.legend(loc='upper right', fontsize=11)
            ax_pend_angle.set_title('PENDULUM ANGLE', fontsize=13, fontweight='bold', color='darkred')
            
            ax_pend_vel.plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2.5, label='Angular Velocity')
            ax_pend_vel.set_ylabel('Angular Velocity (°/s)', fontsize=12, fontweight='bold')
            ax_pend_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_pend_vel.grid(True, alpha=0.3)
            ax_pend_vel.legend(loc='upper right', fontsize=11)
            ax_pend_vel.set_title('PENDULUM VELOCITY', fontsize=13, fontweight='bold', color='darkred')
            
            ax_control.plot(times, forces, 'g-', linewidth=2.5, label='Control Force')
            ax_control.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
            ax_control.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_control.grid(True, alpha=0.3)
            ax_control.legend(loc='upper right', fontsize=11)
            ax_control.set_title('CONTROL EFFORT', fontsize=13, fontweight='bold', color='darkgreen')
        
        plt.suptitle(f'Cart-Pendulum Simulation - {CONTROLLER_MODE.upper()} Controller', 
                     fontsize=14, fontweight='bold')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/cart_pendulum_{CONTROLLER_MODE}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(colored(f"✓ Plot saved: {filename}", 'green'))
        
        plt.show()
    
    def print_summary(self):
        """Print simulation summary."""
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("SIMULATION SUMMARY", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"Controller Mode: {self.controller_mode}", 'yellow'))
        print(colored(f"Simulation Duration: {SIMULATION_TIME}s", 'yellow'))
        print(colored(f"Data Samples: {len(self.time_log)}", 'yellow'))
        if self.meshcat is not None:
            print(colored(f"Meshcat URL: {self.meshcat.web_url()}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def run_full_simulation(self):
        """Execute complete simulation pipeline."""
        try:
            self.setup_drake_system()
            self.add_controller()
            self.setup_visualization()
            self.build_diagram()
            self.create_simulator()
            
            if self.controller_mode == 'scene-viz':
                self.run_scene_viz()
            else:
                self.run_simulation()
                self.extract_data()
                self.plot_results()
                self.print_summary()
        
        except KeyboardInterrupt:
            print(colored("\n\nSimulation interrupted by user", 'yellow'))
        except Exception as e:
            print(colored(f"\n\nError: {e}", 'red'))
            import traceback
            traceback.print_exc()


# ============================================================================
# MINIMUM JERK TRAJECTORY GENERATOR (from cup manipulator)
# ============================================================================

class MinJerkTrajectoryGenerator:
    """
    Minimum-jerk trajectory generator for smooth motion.
    
    Uses 5th-order polynomial (minimum jerk) for time scaling.
    Same implementation as cup manipulator for consistency.
    """
    
    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float, 
                 settle_time: float = 0.0):
        """
        Args:
            q_start: Starting configuration [x, θ]
            q_goal: Goal configuration [x, θ]
            duration: Motion duration (seconds)
            settle_time: Wait time before motion starts (seconds)
        """
        self.q_start = np.array(q_start, dtype=float)
        self.q_goal = np.array(q_goal, dtype=float)
        self.motion_duration = float(duration)
        self.settle_time = float(settle_time)
        self.total_duration = self.settle_time + self.motion_duration
    
    def _min_jerk_profile(self, s: float):
        """
        Compute minimum-jerk time scaling.
        
        Args:
            s: Normalized time [0, 1]
            
        Returns:
            (h, h_dot, h_ddot): position, velocity, acceleration scaling
        """
        s = np.clip(s, 0.0, 1.0)
        
        # 5th order polynomial: h(s) = 10s³ - 15s⁴ + 6s⁵
        h = 10*s**3 - 15*s**4 + 6*s**5
        h_dot = 30*s**2 - 60*s**3 + 30*s**4
        h_ddot = 60*s - 180*s**2 + 120*s**3
        
        return h, h_dot, h_ddot
    
    def compute_trajectory(self, t: float):
        """
        Compute desired state at time t.
        
        Args:
            t: Current time (seconds)
            
        Returns:
            q_d: [x, θ] desired positions
            qd_d: [ẋ, θ̇] desired velocities  
            qdd_d: [ẍ, θ̈] desired accelerations
        """
        # Settle phase
        if t < self.settle_time:
            return self.q_start, np.zeros(2), np.zeros(2)
        
        # Motion phase
        elif t < self.total_duration:
            t_motion = t - self.settle_time
            s = t_motion / self.motion_duration
            h, h_dot, h_ddot = self._min_jerk_profile(s)
            
            # Interpolate between start and goal
            q_d = self.q_start + (self.q_goal - self.q_start) * h
            qd_d = (self.q_goal - self.q_start) * h_dot / self.motion_duration
            qdd_d = (self.q_goal - self.q_start) * h_ddot / (self.motion_duration**2)
            
            return q_d, qd_d, qdd_d
        
        # Hold at goal
        else:
            return self.q_goal, np.zeros(2), np.zeros(2)


# ============================================================================
# TRAJECTORY GENERATOR (legacy modes)
# ============================================================================

class TrajectoryGenerator:
    """Generate reference trajectories for cart-pendulum."""
    
    def __init__(self, mode='balance'):
        """
        Args:
            mode: 'balance' (upright), 'swing' (sinusoidal), 'track' (position tracking)
        """
        self.mode = mode
        self.cart_amplitude = 0.3  # m
        self.cart_frequency = 0.5  # Hz
    
    def compute_trajectory(self, t):
        """
        Compute desired state at time t.
        
        Returns:
            q_desired: [x, θ] positions
            qd_desired: [ẋ, θ̇] velocities
            qdd_desired: [ẍ, θ̈] accelerations
        """
        if self.mode == 'balance':
            # Keep cart at origin, pendulum upright
            q_desired = np.array([0.0, 0.0])
            qd_desired = np.zeros(2)
            qdd_desired = np.zeros(2)
            
        elif self.mode == 'swing':
            # Sinusoidal cart motion, pendulum should balance
            omega = 2 * np.pi * self.cart_frequency
            q_desired = np.array([
                self.cart_amplitude * np.sin(omega * t),
                0.0  # Upright
            ])
            qd_desired = np.array([
                self.cart_amplitude * omega * np.cos(omega * t),
                0.0
            ])
            qdd_desired = np.array([
                -self.cart_amplitude * omega**2 * np.sin(omega * t),
                0.0
            ])
            
        elif self.mode == 'track':
            # Step input for cart position
            x_target = 0.5 if t > 2.0 else 0.0
            q_desired = np.array([x_target, 0.0])
            qd_desired = np.zeros(2)
            qdd_desired = np.zeros(2)
        
        return q_desired, qd_desired, qdd_desired


# ============================================================================
# PD CONTROLLER
# ============================================================================

class PDController(LeafSystem):
    """
    Simple PD controller for cart-pendulum.
    
    Control law: F = Kp_x·(x_d - x) + Kd_x·(ẋ_d - ẋ) 
                    + Kp_θ·(θ_d - θ) + Kd_θ·(θ̇_d - θ̇)
    
    Note: This is NOT optimal for balancing - use LQR instead.
    """
    
    def __init__(self, kp_cart, kd_cart, kp_pend, kd_pend, 
                 trajectory_generator: TrajectoryGenerator):
        LeafSystem.__init__(self)
        
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_pend = kp_pend
        self.kd_pend = kd_pend
        self.trajectory_generator = trajectory_generator
        
        # Input: state [x, θ, ẋ, θ̇]
        self.DeclareVectorInputPort("state", BasicVector(4))
        
        # Output: force [F]
        self.DeclareVectorOutputPort("force", BasicVector(1), self.CalcControlForce)
        
        print(colored(f"✓ PDController initialized:", 'green'))
        print(colored(f"  Cart gains: Kp={kp_cart}, Kd={kd_cart}", 'cyan'))
        print(colored(f"  Pendulum gains: Kp={kp_pend}, Kd={kd_pend}", 'cyan'))
    
    def CalcControlForce(self, context, output):
        """Compute PD control force."""
        # Get current state
        state = self.get_input_port(0).Eval(context)
        x, theta, x_dot, theta_dot = state
        
        # Get desired trajectory
        t = context.get_time()
        q_d, qd_d, _ = self.trajectory_generator.compute_trajectory(t)
        
        # Compute errors
        e_x = q_d[0] - x
        e_theta = q_d[1] - theta
        ed_x = qd_d[0] - x_dot
        ed_theta = qd_d[1] - theta_dot
        
        # PD control law
        F = (self.kp_cart * e_x + self.kd_cart * ed_x +
             self.kp_pend * e_theta + self.kd_pend * ed_theta)
        
        output.SetFromVector([F])




# ============================================================================
# COMPUTED TORQUE CONTROLLER
# ============================================================================

class ComputedTorqueController(LeafSystem):
    """
    Unified Computed Torque (Inverse Dynamics) controller.
    
    Supports two dynamics computation methods:
    1. MultibodyPlant-based: Uses Drake's CalcInverseDynamics
    2. Analytical equations: Uses closed-form inverse dynamics from equations 2.1 & 2.2
    
    Control law: F = M(q)·[q̈_d + Kp·e + Kd·ė] + C(q,q̇) + g(q)
    
    Key features:
    - Separate model for control calculations (optional)
    - Full nonlinear dynamics compensation
    - Trajectory tracking with feedforward + feedback
    """
    
    def __init__(self, plant=None, model=None, 
                 Kp=None, Kd=None, trajectory_generator=None, 
                 use_model: bool = True, use_analytical: bool = False):
        """
        Args:
            plant: Real MultibodyPlant (optional for analytical mode)
            model: Controller's internal MultibodyPlant model (optional for analytical mode)
            Kp: Position gain matrix (2x2)
            Kd: Velocity gain matrix (2x2)
            trajectory_generator: Trajectory generator object
            use_model: Use separate model plant (True) or real plant (False) - only for MultibodyPlant mode
            use_analytical: Use analytical inverse dynamics (True) or MultibodyPlant (False)
        """
        LeafSystem.__init__(self)
        
        self.use_analytical = use_analytical
        self.Kp = Kp
        self.Kd = Kd
        self.trajectory_generator = trajectory_generator
        
        if not use_analytical:
            # MultibodyPlant mode
            self.plant = plant
            self.model = model if use_model else plant
            self.model_context = self.model.CreateDefaultContext()
            self.use_model = use_model
        
        self.DeclareVectorInputPort("state", BasicVector(4))
        self.DeclareVectorOutputPort("force", BasicVector(1), self.CalcControlForce)
        
        print(colored(f"✓ ComputedTorqueController initialized:", 'green'))
        print(colored(f"  Kp = {np.diag(Kp)}", 'cyan'))
        print(colored(f"  Kd = {np.diag(Kd)}", 'cyan'))
        
        if use_analytical:
            print(colored(f"  Dynamics: ANALYTICAL (equations 2.1 & 2.2)", 'yellow'))
            print(colored(f"    Using closed-form inverse dynamics", 'cyan'))
        else:
            if use_model:
                print(colored(f"  Dynamics: MULTIBODY PLANT", 'yellow'))
                print(colored(f"  Model-Plant Separation: ENABLED", 'yellow'))
                print(colored(f"    Plant: Used for state observation", 'cyan'))
                print(colored(f"    Model: Used for inverse dynamics", 'cyan'))
            else:
                print(colored(f"  Dynamics: MULTIBODY PLANT", 'yellow'))
                print(colored(f"  Model-Plant Separation: DISABLED (using real plant)", 'yellow'))
                print(colored(f"    Plant: Used for both state and inverse dynamics", 'cyan'))
    
    def CalcControlForce(self, context, output):
        """Compute control using inverse dynamics."""
        # Get current state
        import numpy as np
        
        state = self.get_input_port(0).Eval(context)
        q = state[:2]
        v = state[2:]
        
        # Get desired trajectory
        t = context.get_time()
        q_d, v_d, a_d = self.trajectory_generator.compute_trajectory(t)
        
        if self.use_analytical:
            # Analytical mode: simple error calculation
            e = q_d - q
            ed = v_d - v
        else:
            # MultibodyPlant mode: normalize pendulum angle
            phi = q[1]
            phi_d = q_d[1]
            phi_normalized = np.arctan2(np.sin(phi), np.cos(phi))
            phi_d_normalized = np.arctan2(np.sin(phi_d), np.cos(phi_d))
            e = np.array([q_d[0] - q[0], phi_d_normalized - phi_normalized])
            ed = v_d - v
        
        # Commanded acceleration (feedback + feedforward)
        a_cmd = a_d + self.Kp @ e + self.Kd @ ed
        
        if self.use_analytical:
            # ═══════════════════════════════════════════════════════════════
            # ANALYTICAL NONLINEAR INVERSE DYNAMICS (APPROACH 2)
            # ═══════════════════════════════════════════════════════════════
            # UNDERACTUATED SYSTEM: Force F only acts on cart, not pendulum!
            # 
            # Full system equations:
            #   (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F  ... (cart equation)
            #   l·φ̈ + ẍ·cos(φ) + g·sin(φ) = 0         ... (pendulum equation)
            # 
            # APPROACH 2: Physically Consistent Inverse Dynamics
            # ──────────────────────────────────────────────────
            # 1. Use only ẍ_cmd from trajectory (cart is actuated)
            # 2. Compute φ̈ from passive constraint equation:
            #    φ̈ = (-ẍ_cmd·cos(φ) - g·sin(φ)) / l
            # 3. Compute required force F using cart equation
            # 
            # This GUARANTEES the commanded force produces accelerations
            # that satisfy BOTH equations simultaneously!
            # ═══════════════════════════════════════════════════════════════
            
            M = CART_MASS
            m = PENDULUM_MASS
            l = PENDULUM_LENGTH
            g = GRAVITY
            G = COUPLING_GAIN
            
            # Current state
            phi = q[1]
            phi_dot = v[1]
            
            # Commanded cart acceleration
            x_ddot_cmd = a_cmd[0]
            
            c = np.cos(phi)
            s = np.sin(phi)
            
            # Compute physically consistent pendulum acceleration
            # from constraint: l·φ̈ + G·ẍ·cos(φ) + g·sin(φ) = 0
            phi_ddot_consistent = (-G * x_ddot_cmd * c - g * s) / l
            
            # Inverse dynamics using cart equation
            # (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F
            F = (M + m) * x_ddot_cmd + m * l * phi_ddot_consistent * c - m * l * phi_dot**2 * s
        else:
            # MultibodyPlant inverse dynamics
            self.model.SetPositions(self.model_context, q)
            self.model.SetVelocities(self.model_context, v)
            
            from pydrake.multibody.tree import MultibodyForces
            tau = self.model.CalcInverseDynamics(
                self.model_context,
                a_cmd,
                MultibodyForces(self.model)
            )
            
            # Extract cart force (first actuator)
            F = tau[0]
        
        output.SetFromVector([F])

class ReferenceImpedanceController(LeafSystem):
    """
    Paper-style reference model + impedance coupling (Fig. 2B).

    Inputs:
      - state: [x, phi, x_dot, phi_dot]  (we only use x, x_dot)
      - u: scalar neural command

    Internal states:
      - F_m: muscle force
      - x_ref
      - v_ref

    Output:
      - F_int: interaction force to apply to the real plant
    """
    def __init__(self, M_ref, kp, kd, tau):
        super().__init__()
        self.M_ref = float(M_ref)
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau = float(tau)

        # Inputs
        self.DeclareVectorInputPort("state", BasicVector(4)) #Adds an input port for the state of the plant (x, phi, x_dot, phi_dot)
        self.DeclareVectorInputPort("u", BasicVector(1)) #Adds an input port for the neural command (u), generated from some policy or controller

        # Continuous state: [F_m, x_ref, v_ref]
        self.DeclareContinuousState(3) # Adds 3 continuous state variables: muscle force (F_m), reference position (x_ref), and reference velocity (v_ref)

        # Output force
        self.DeclareVectorOutputPort("force", BasicVector(1), self.CalcFint) # Adds an output port for the interaction force (F_int) that will be applied to the real plant, calculated by the CalcFint method

    def DoCalcTimeDerivatives(self, context, derivatives):
        # Plant state input
        x, phi, x_dot, phi_dot = self.get_input_port(0).Eval(context)

        # Control input u, generated from some policy or controller
        u = float(self.get_input_port(1).Eval(context)[0])

        # Internal states
        F_m, x_ref, v_ref = context.get_continuous_state_vector().CopyToVector()

        # Muscle dynamics: tau * Fdot = u - F
        F_m_dot = (u - F_m) / self.tau

        # Reference model dynamics:
        # M_ref * xref_ddot = kp(x - xref) + kd(xdot - vref) + F_m
        x_ref_ddot = (self.kp * (x - x_ref) + self.kd * (x_dot - v_ref) + F_m) / self.M_ref

        # State derivatives
        derivatives.get_mutable_vector().SetFromVector([
            F_m_dot,      # d/dt F_m
            v_ref,        # d/dt x_ref
            x_ref_ddot    # d/dt v_ref
        ])

    def CalcFint(self, context, output):
        # Plant state generated by the real plant
        x, phi, x_dot, phi_dot = self.get_input_port(0).Eval(context)

        # Internal states generated by the reference model
        F_m, x_ref, v_ref = context.get_continuous_state_vector().CopyToVector()

        # Interaction force applied to the plant
        F_int = self.kp * (x_ref - x) + self.kd * (v_ref - x_dot)
        output.SetFromVector([F_int])



class StandardLQRController(LeafSystem):
    """
    Standard lqr implemented as continuous-time LQR state feedback:
        u = -K (x - x_goal)

    Inputs:
      - x (plant state), dimension n

    Outputs:
      - u (motor command), dimension 1
    """
    def __init__(self, A, B, Q, R, x_goal=None, u_limits=None):
        super().__init__()
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).reshape((-1, 1))
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float).reshape((1, 1))

        n = self.A.shape[0]
        assert self.A.shape == (n, n)
        assert self.B.shape == (n, 1)
        assert self.Q.shape == (n, n)
        assert self.R.shape == (1, 1)

        self.K, self.P = self._dlqr(self.A, self.B, self.Q, self.R)

        self.x_goal = np.zeros(n) if x_goal is None else np.array(x_goal, dtype=float).reshape((n,))
        self.u_limits = u_limits  # tuple (umin, umax) or None

        self.DeclareVectorInputPort("x", BasicVector(n))
        self.DeclareVectorOutputPort("u", BasicVector(1), self.CalcU)

    def CalcU(self, context, output):
        x = self.get_input_port(0).Eval(context)
        x_err = x - self.x_goal
        u = float(-(self.K @ x_err.reshape((-1, 1)))[0, 0])

        if self.u_limits is not None:
            umin, umax = self.u_limits
            u = float(np.clip(u, umin, umax))

        output.SetFromVector([u])

    def _dlqr(self, A, B, Q, R, max_iters=1000, tol=1e-10):
        """
        Solve the continuous-time Algebraic Riccati Equation (CARE) and return K.
        
        Uses scipy.linalg.solve_continuous_are to solve:
            0 = A^T P + P A - P B R^{-1} B^T P + Q
        
        Then computes K = R^{-1} B^T P for the control law: u = -K(x - x_goal)
        """
        from scipy.linalg import solve_continuous_are
        
        # Solve continuous-time ARE
        P = solve_continuous_are(A, B, Q, R)
        
        # Compute gain: K = R^{-1} B^T P
        R_inv = np.linalg.inv(R)
        K = R_inv @ (B.T @ P)
        
        return K, P



class FiniteHorizonController(LeafSystem):
    """
    Finite-horizon, continuous-time LQR implemented as a time-varying state feedback:

        u(t) = -K(t) (x(t) - x_goal)

    This matches the paper-style cost with a running state weight Q(t) (= Q_t)
    and a terminal weight QN (= Q_N):

        J = ∫_0^T [ xᵀ Q x + uᵀ R u ] dt  +  x(T)ᵀ QN x(T)

    Implementation approach:
    - Discretize (A,B) with a chosen dt
    - Solve the finite-horizon discrete Riccati recursion backward to get K_k
    - At runtime, select K based on current time (piecewise-constant over dt bins)

    Inputs:
      - x (plant state), dimension n
    Outputs:
      - u (motor command), dimension 1
    """
    def __init__(self, A, B, Q, R, QN, T, dt,
                 x_goal=None, u_limits=None,
                 discretization="zoh"):
        super().__init__()

        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).reshape((-1, 1))
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float).reshape((1, 1))
        self.QN = np.array(QN, dtype=float)

        n = self.A.shape[0]
        assert self.A.shape == (n, n)
        assert self.B.shape == (n, 1)
        assert self.Q.shape == (n, n)
        assert self.QN.shape == (n, n)
        assert self.R.shape == (1, 1)

        assert dt > 0
        assert T > 0
        self.dt = float(dt)
        self.T = float(T)
        self.N = int(np.round(self.T / self.dt))
        if self.N < 1:
            raise ValueError("Horizon too short: N must be >= 1")
        # Make horizon exactly N*dt for indexing consistency
        self.T = self.N * self.dt

        self.x_goal = np.zeros(n) if x_goal is None else np.array(x_goal, dtype=float).reshape((n,))
        self.u_limits = u_limits  # tuple (umin, umax) or None

        # Discretize continuous-time (A,B) -> (Ad,Bd)
        self.Ad, self.Bd = self._discretize(self.A, self.B, self.dt, method=discretization)
        Qd = self.Q * self.dt
        Rd = self.R * self.dt
        QNd = self.QN  # Terminal cost is not multiplied by dt
        # Compute time-varying gains K_k, k = 0..N-1
        self.K_list, self.P_list = self._finite_horizon_dlqr(self.Ad, self.Bd, Qd, Rd, QNd, self.N)

        # Drake ports
        self.DeclareVectorInputPort("x", BasicVector(n))
        self.DeclareVectorOutputPort("u", BasicVector(1), self.CalcU)

    def CalcU(self, context, output):
        x = self.get_input_port(0).Eval(context)
        t = context.get_time()

        # Choose which gain to use based on time.
        # Clamp so after horizon we keep using the last gain (or you can switch to 0).
        k = int(np.floor(t / self.dt))
        k = int(np.clip(k, 0, self.N - 1))

        K = self.K_list[k]
        x_err = (x - self.x_goal).reshape((-1, 1))
        u = float(-(K @ x_err)[0, 0])

        if self.u_limits is not None:
            umin, umax = self.u_limits
            u = float(np.clip(u, umin, umax))

        output.SetFromVector([u])

    @staticmethod
    def _finite_horizon_dlqr(Ad, Bd, Q, R, QN, N):
        """
        Backward Riccati recursion for finite-horizon *discrete-time* LQR:

            x_{k+1} = Ad x_k + Bd u_k

            J = Σ_{k=0}^{N-1} (x_kᵀ Q x_k + u_kᵀ R u_k) + x_Nᵀ QN x_N

        Returns:
          K_list: length N, with u_k = -K_list[k] x_k
          P_list: length N+1, Riccati matrices P_k
        """
        n = Ad.shape[0]
        P_list = [None] * (N + 1)
        K_list = [None] * N

        P = QN.copy()
        P_list[N] = P

        for k in reversed(range(N)):
            S = R + Bd.T @ P @ Bd          # (m,m) here m=1
            # K = S^{-1} Bd^T P Ad
            K = np.linalg.solve(S, Bd.T @ P @ Ad)
            K_list[k] = K

            # P = Q + Ad^T P Ad - Ad^T P Bd K
            P = Q + Ad.T @ P @ (Ad - Bd @ K)
            P_list[k] = P

        # Sanity shapes
        assert all(K.shape == (1, n) for K in K_list)
        assert all(P.shape == (n, n) for P in P_list)
        return K_list, P_list

    @staticmethod
    def _discretize(A, B, dt, method="zoh"):
        """
        Discretize continuous-time linear system:
            xdot = A x + B u

        method:
          - "zoh": exact zero-order hold via matrix exponential
          - "euler": forward Euler (approx)
        """
        method = method.lower()
        if method == "euler":
            Ad = np.eye(A.shape[0]) + A * dt
            Bd = B * dt
            return Ad, Bd

        if method == "zoh":
            # Exact ZOH discretization using augmented matrix exponential:
            # exp([[A, B],
            #      [0, 0]] dt) = [[Ad, Bd],
            #                    [ 0,  I]]
            from scipy.linalg import expm

            n = A.shape[0]
            m = B.shape[1]
            M = np.zeros((n + m, n + m))
            M[:n, :n] = A
            M[:n, n:] = B
            Md = expm(M * dt)
            Ad = Md[:n, :n]
            Bd = Md[:n, n:]
            return Ad, Bd

        raise ValueError(f"Unknown discretization method: {method}")



# ============================================================================
# MODEL COMPARISON FUNCTION
# ============================================================================

def compare_models():
    """
    Compare CartPendulumSystemByEqns vs MultibodyPlant with same controller.
    Both use computed torque control for fair comparison.
    """
    print("\n" + "=" * 70)
    print(colored("MODEL COMPARISON: Equations vs MultibodyPlant", 'cyan', attrs=['bold']))
    print(colored("Both using Computed Torque Controller", 'cyan'))
    print("=" * 70 + "\n")
    
    # Create trajectory generator (same for both)
    q_start = np.array([CART_START_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
    q_goal = np.array([CART_END_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
    traj_gen = MinJerkTrajectoryGenerator(
        q_start=q_start,
        q_goal=q_goal,
        duration=CART_MOTION_DURATION,
        settle_time=CART_SETTLE_TIME
    )
    
    # ========== System 1: Equation-Based ==========
    print(colored("\n[1/2] Simulating Equation-Based System...", 'yellow', attrs=['bold']))
    
    builder1 = DiagramBuilder()
    
    # Create equation-based plant
    plant_eqn = builder1.AddSystem(CartPendulumSystemByEqns())
    
    # Create controller using analytical inverse dynamics
    controller_eqn = builder1.AddSystem(
        ComputedTorqueController(
            Kp=CT_KP, 
            Kd=CT_KD, 
            trajectory_generator=traj_gen,
            use_analytical=True
        )
    )
    
    # Wire connections
    builder1.Connect(plant_eqn.get_output_port(0), controller_eqn.get_input_port(0))
    builder1.Connect(controller_eqn.get_output_port(0), plant_eqn.get_input_port(0))
    
    # Build diagram
    diagram1 = builder1.Build()
    simulator1 = Simulator(diagram1)
    context1 = simulator1.get_mutable_context()
    
    # Set initial conditions
    plant_context1 = diagram1.GetMutableSubsystemContext(plant_eqn, context1)
    plant_eqn.SetInitialConditions(
        plant_context1,
        x0=CART_START_POSITION,
        phi0=np.deg2rad(args.initial_theta),
        x_dot0=0.0,
        phi_dot0=0.0
    )
    
    # Simulate
    time_log_eqn = []
    state_log_eqn = []
    force_log_eqn = []
    
    simulator1.set_target_realtime_rate(0.0)  # Run as fast as possible
    last_log_time = 0.0
    
    while context1.get_time() < SIMULATION_TIME:
        simulator1.AdvanceTo(context1.get_time() + TIMESTEP)
        
        if context1.get_time() - last_log_time >= LOGGING_INTERVAL:
            time_log_eqn.append(context1.get_time())
            state = plant_eqn.get_output_port(0).Eval(plant_context1)
            state_log_eqn.append(state.copy())
            force = controller_eqn.get_output_port(0).Eval(
                diagram1.GetMutableSubsystemContext(controller_eqn, context1)
            )
            force_log_eqn.append(force[0])
            last_log_time = context1.get_time()
    
    print(colored("  ✓ Equation-based simulation complete", 'green'))
    
    # ========== System 2: MultibodyPlant ==========
    print(colored("\n[2/2] Simulating MultibodyPlant System...", 'yellow', attrs=['bold']))
    
    # Temporarily set mode to computed-torque for proper initialization
    global CONTROLLER_MODE, TRAJECTORY_MODE
    saved_mode = CONTROLLER_MODE
    CONTROLLER_MODE = 'computed-torque'
    TRAJECTORY_MODE = 'cart-motion'
    
    system2 = CartPendulumSystem()
    system2.build_plant()
    system2.add_controller('computed-torque')
    system2.build_diagram()
    
    # Create simulator and manually set initial conditions to match equation-based system
    system2.simulator = Simulator(system2.diagram)
    system2.simulator.set_target_realtime_rate(REALTIME_RATE)
    context = system2.simulator.get_mutable_context()
    plant_context = system2.plant.GetMyMutableContextFromRoot(context)
    
    # Use same initial angle as equation-based system
    system2.plant.SetPositions(plant_context, [CART_START_POSITION, np.deg2rad(args.initial_theta)])
    system2.plant.SetVelocities(plant_context, [0.0, 0.0])
    system2.diagram.ForcedPublish(context)
    
    print(colored(f"✓ Simulator created", 'green'))
    print(colored(f"  Initial state: x={CART_START_POSITION:.1f} m, θ={args.initial_theta:.1f}°", 'cyan'))
    
    # Restore original mode
    CONTROLLER_MODE = saved_mode
    
    system2.run_simulation()
    
    print(colored("  ✓ MultibodyPlant simulation complete", 'green'))
    
    # ========== Plot Comparison ==========
    print(colored("\nGenerating comparison plots...", 'yellow'))
    
    time_eqn = np.array(time_log_eqn)
    states_eqn = np.array(state_log_eqn)
    forces_eqn = np.array(force_log_eqn)
    
    time_mbp = np.array(system2.time_log)
    states_mbp = np.array(system2.state_log)
    forces_mbp = np.array(system2.force_log)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cart position
    axes[0, 0].plot(time_eqn, states_eqn[:, 0], 'b-', linewidth=2, label='Equations', alpha=0.8)
    axes[0, 0].plot(time_mbp, states_mbp[:, 0], 'r--', linewidth=2, label='MultibodyPlant', alpha=0.8)
    axes[0, 0].set_ylabel('Cart Position (m)', fontsize=12)
    axes[0, 0].set_title('Cart Position Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pendulum angle
    axes[0, 1].plot(time_eqn, np.rad2deg(states_eqn[:, 1]), 'b-', linewidth=2, label='Equations', alpha=0.8)
    axes[0, 1].plot(time_mbp, np.rad2deg(states_mbp[:, 1]), 'r--', linewidth=2, label='MultibodyPlant', alpha=0.8)
    axes[0, 1].set_ylabel('Pendulum Angle (deg)', fontsize=12)
    axes[0, 1].set_title('Pendulum Angle Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cart velocity
    axes[1, 0].plot(time_eqn, states_eqn[:, 2], 'b-', linewidth=2, label='Equations', alpha=0.8)
    axes[1, 0].plot(time_mbp, states_mbp[:, 2], 'r--', linewidth=2, label='MultibodyPlant', alpha=0.8)
    axes[1, 0].set_ylabel('Cart Velocity (m/s)', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)', fontsize=12)
    axes[1, 0].set_title('Cart Velocity Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Control force
    axes[1, 1].plot(time_eqn, forces_eqn, 'b-', linewidth=2, label='Equations', alpha=0.8)
    axes[1, 1].plot(time_mbp, forces_mbp, 'r--', linewidth=2, label='MultibodyPlant', alpha=0.8)
    axes[1, 1].set_ylabel('Control Force (N)', fontsize=12)
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_title('Control Force Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison: Equations (2.1 & 2.2) vs MultibodyPlant', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/model_comparison_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Comparison plot saved: {filename}", 'green'))
    
    plt.show()
    
    # Compute and display differences
    print(colored("\n" + "="*70, 'cyan'))
    print(colored("DIFFERENCE ANALYSIS", 'cyan', attrs=['bold']))
    print(colored("="*70, 'cyan'))
    
    # Interpolate to common time base for comparison
    from scipy.interpolate import interp1d
    common_time = np.linspace(max(time_eqn[0], time_mbp[0]), 
                              min(time_eqn[-1], time_mbp[-1]), 500)
    
    x_eqn_interp = interp1d(time_eqn, states_eqn[:, 0])(common_time)
    x_mbp_interp = interp1d(time_mbp, states_mbp[:, 0])(common_time)
    
    phi_eqn_interp = interp1d(time_eqn, states_eqn[:, 1])(common_time)
    phi_mbp_interp = interp1d(time_mbp, states_mbp[:, 1])(common_time)
    
    x_diff_rms = np.sqrt(np.mean((x_eqn_interp - x_mbp_interp)**2))
    phi_diff_rms = np.sqrt(np.mean((phi_eqn_interp - phi_mbp_interp)**2))
    
    print(colored(f"Cart Position RMS Difference: {x_diff_rms*1000:.4f} mm", 'yellow'))
    print(colored(f"Pendulum Angle RMS Difference: {np.rad2deg(phi_diff_rms):.4f}°", 'yellow'))
    print(colored("="*70 + "\n", 'cyan'))


# ============================================================================
# STANDARD lqr WITH LINEARIZED PLANT
# ============================================================================

def run_standard_lqr_with_linearized_plant():
    """
    Run Standard lqr controller with CartPendulumSystemLinearizedWithMuscleDynamics.
    
    This mode uses:
    - Plant: Linearized 6D state-space model (equation 2.7)
    - Controller: StandardLQRController (continuous-time LQR via solve_continuous_are)
    - State: [x, φ, ẋ, φ̇, F, F_pert]
    """
    print("\n" + "=" * 70)
    print(colored("STANDARD lqr - LINEARIZED PLANT (CONTINUOUS-TIME LQR)", 'cyan', attrs=['bold']))
    print(colored("Plant: CartPendulumSystemLinearizedWithMuscleDynamics", 'cyan'))
    print(colored("Controller: StandardLQRController (Continuous-Time LQR via solve_continuous_are)", 'cyan'))
    print("=" * 70 + "\n")
    
    builder = DiagramBuilder()
    
    # ========== Create Linearized Plant (6D state) ==========
    plant = builder.AddSystem(
        CartPendulumSystemLinearizedWithMuscleDynamics(
            M=CART_MASS,
            m=PENDULUM_MASS,
            l=PENDULUM_LENGTH,
            g=GRAVITY,
            G=COUPLING_GAIN,
            tau=MOTOR_TIME_CONSTANT,
            M_arm=ARM_MASS
        )
    )
    
    # ========== Create Controller ==========
    print(colored(f"\nCreating Standard lqr Controller...", 'yellow', attrs=['bold']))
    
    # Get A, B matrices from the plant's method
    # (5D state: [x, φ, ẋ, φ̇, F])
    A, B = plant._compute_system_matrices()
    
    # Cost matrices (5D state: [x, φ, ẋ, φ̇, F])
    Q = STANDARD_LQR_LINEARIZED_Q
    R = STANDARD_LQR_LINEARIZED_R
    
    # Goal state: upright at origin
    x_goal = STANDARD_LQR_LINEARIZED_X_GOAL
    
    print(colored(f"  State dimension: 5D [x, φ, ẋ, φ̇, F]", 'cyan'))
    print(colored(f"  Goal: upright at origin", 'cyan'))
    
    controller = builder.AddSystem(
        StandardLQRController(
            A=A,
            B=B,
            Q=Q,
            R=R,
            x_goal=x_goal,
            u_limits=(-100.0, 100.0)
        )
    )
    
    print(colored(f"✓ StandardLQRController created", 'green'))
    
    # ========== Wire System ==========
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    
    print(colored(f"✓ System wired: plant -> controller -> plant", 'green'))
    
    # ========== Visualization ==========
    meshcat = None
    if args.visualize:
        print(colored(f"\nSetting up Meshcat visualization...", 'yellow', attrs=['bold']))
        
        # Create a separate MultibodyPlant for visualization only
        from pydrake.geometry import Sphere, Cylinder
        
        viz_plant, viz_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        
        # Cart
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = viz_plant.AddRigidBody("cart", cart_inertia)
        
        # Cart visuals
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        viz_plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        
        # Prismatic joint
        viz_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                viz_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -TRACK_LIMIT,
                TRACK_LIMIT,
                damping=CART_DAMPING
            )
        )
        
        # Pendulum
        I_about_pivot = PENDULUM_LENGTH**2
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = viz_plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Pendulum visuals
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH/2])),
            Cylinder(PENDULUM_RADIUS, PENDULUM_LENGTH),
            "pendulum_visual",
            np.array([0.8, 0.1, 0.1, 1.0])
        )
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH])),
            Sphere(PENDULUM_RADIUS * 1.5),
            "pendulum_tip",
            np.array([0.9, 0.2, 0.2, 1.0])
        )
        
        # Revolute joint
        viz_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=PENDULUM_DAMPING
            )
        )
        
        # Add actuator for the visualization plant
        viz_plant.AddJointActuator("cart_force", viz_plant.GetJointByName("cart_slider"))
        
        viz_plant.Finalize()
        
        # Add Meshcat visualizer
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, viz_scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration, prefix="standard_lqr")
        )
        
        print(colored("✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {meshcat.web_url()}", 'cyan'))
    
    # ========== Build Diagram ==========
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial conditions: [x, φ, ẋ, φ̇, F]
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    initial_angle_rad = np.deg2rad(args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE)
    plant.SetInitialConditions(
        plant_context,
        x0=0.0,
        phi0=initial_angle_rad,
        x_dot0=0.0,
        phi_dot0=0.0,
        F0=0.0
    )
    
    print(colored(f"\n✓ Initial conditions set:", 'green'))
    print(colored(f"  x = 0.0 m, φ = {args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°", 'cyan'))
    print(colored(f"  ẋ = 0.0 m/s, φ̇ = 0.0 rad/s", 'cyan'))
    print(colored(f"  F = 0.0 N", 'cyan'))
    
    # ========== Simulation ==========
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    
    print(colored(f"\nRunning simulation for {SIMULATION_TIME}s...", 'yellow', attrs=['bold']))
    
    # Data logging
    time_log = []
    state_log = []
    force_log = []
    
    LOGGING_INTERVAL = 0.01
    last_log_time = 0.0
    last_print_time = 0.0
    PRINT_INTERVAL = 1.0
    
    while context.get_time() < SIMULATION_TIME:
        simulator.AdvanceTo(context.get_time() + TIMESTEP)
        t = context.get_time()
        
        # Update visualization plant to match linearized plant state
        if meshcat is not None:
            state = plant.get_output_port(0).Eval(plant_context)
            x, phi = state[0], state[1]  # Extract positions from 6D state
            viz_plant_context = viz_plant.GetMyMutableContextFromRoot(context)
            viz_plant.SetPositions(viz_plant_context, [x, phi])
        
        # Print progress
        if t >= last_print_time + PRINT_INTERVAL:
            state = plant.get_output_port(0).Eval(plant_context)
            print(colored(f"  t={t:.1f}s: x={state[0]:.3f}m, φ={np.rad2deg(state[1]):.1f}°, F={state[4]:.2f}N", 'cyan'))
            last_print_time = t
        
        # Log data
        if t >= last_log_time + LOGGING_INTERVAL:
            time_log.append(t)
            state = plant.get_output_port(0).Eval(plant_context)
            state_log.append(state.copy())
            force = controller.get_output_port(0).Eval(
                diagram.GetMutableSubsystemContext(controller, context)
            )
            force_log.append(force[0])
            last_log_time = t
    
    print(colored("\n✓ Simulation complete!", 'green'))
    
    # ========== Plot Results ==========
    import matplotlib.pyplot as plt
    times = np.array(time_log)
    states = np.array(state_log)
    forces = np.array(force_log)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Cart position
    axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
    axes[0, 0].axhline(y=STANDARD_LQR_LINEARIZED_X_GOAL[0], color='r', linestyle='--', linewidth=2, alpha=0.7, label='Goal')
    axes[0, 0].set_ylabel('Cart Position (m)', fontsize=12)
    axes[0, 0].set_title('Cart Position', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Pendulum angle
    axes[0, 1].plot(times, np.rad2deg(states[:, 1]), 'r-', linewidth=2)
    axes[0, 1].set_ylabel('Pendulum Angle (°)', fontsize=12)
    axes[0, 1].set_title('Pendulum Angle', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    axes[0, 1].legend()
    
    # Cart velocity
    axes[1, 0].plot(times, states[:, 2], 'b-', linewidth=2)
    axes[1, 0].set_ylabel('Cart Velocity (m/s)', fontsize=12)
    axes[1, 0].set_title('Cart Velocity', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pendulum angular velocity
    axes[1, 1].plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2)
    axes[1, 1].set_ylabel('Angular Velocity (°/s)', fontsize=12)
    axes[1, 1].set_title('Pendulum Angular Velocity', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Control force F (state)
    axes[2, 0].plot(times, states[:, 4], 'm-', linewidth=2, label='F (state)')
    axes[2, 0].set_ylabel('Force State (N)', fontsize=12)
    axes[2, 0].set_xlabel('Time (s)', fontsize=12)
    axes[2, 0].set_title('Motor Force State', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Control command u
    axes[2, 1].plot(times, forces, 'g-', linewidth=2, label='u (command)')
    axes[2, 1].set_ylabel('Control Command', fontsize=12)
    axes[2, 1].set_xlabel('Time (s)', fontsize=12)
    axes[2, 1].set_title('Motor Command', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.suptitle(f'Standard lqr - Linearized Plant (θ₀={args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/standard_lqr_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved: {filename}", 'green'))
    
    plt.show()


# ============================================================================
# FINITE-HORIZON LQR WITH LINEARIZED PLANT
# ============================================================================

def run_finite_horizon_lqr_with_linearized_plant():
    """
    Run Finite-Horizon LQR controller with CartPendulumSystemLinearizedWithMuscleDynamics.
    
    This mode uses:
    - Plant: Linearized 5D state-space model (equation 2.7)
    - Controller: FiniteHorizonController (discrete-time time-varying LQR via backward Riccati)
    - State: [x, φ, ẋ, φ̇, F]
    - Time-varying gains computed over finite horizon
    """
    print("\n" + "=" * 70)
    print(colored("FINITE-HORIZON LQR - LINEARIZED PLANT (TIME-VARYING GAIN)", 'cyan', attrs=['bold']))
    print(colored("Plant: CartPendulumSystemLinearizedWithMuscleDynamics", 'cyan'))
    print(colored("Controller: FiniteHorizonController (Time-Varying LQR via Backward Riccati)", 'cyan'))
    print("=" * 70 + "\n")
    
    builder = DiagramBuilder()
    
    # ========== Create Linearized Plant (5D state) ==========
    plant = builder.AddSystem(
        CartPendulumSystemLinearizedWithMuscleDynamics(
            M=CART_MASS,
            m=PENDULUM_MASS,
            l=PENDULUM_LENGTH,
            g=GRAVITY,
            G=COUPLING_GAIN,
            tau=MOTOR_TIME_CONSTANT,
            M_arm=ARM_MASS
        )
    )
    
    # ========== Create Controller ==========
    print(colored(f"\nCreating Finite-Horizon LQR Controller...", 'yellow', attrs=['bold']))
    
    # Get A, B matrices from the plant's method
    # (5D state: [x, φ, ẋ, φ̇, F])
    A_cont, B_cont = plant._compute_system_matrices()
    
    # Cost matrices from config (5D state: [x, φ, ẋ, φ̇, F])
    Q = FINITE_HORIZON_LQR_Q
    QN = FINITE_HORIZON_LQR_QN
    R = FINITE_HORIZON_LQR_R
    x_goal = FINITE_HORIZON_LQR_X_GOAL
    T = FINITE_HORIZON_LQR_T
    dt = FINITE_HORIZON_LQR_DT
    
    print(colored(f"  State dimension: 5D [x, φ, ẋ, φ̇, F]", 'cyan'))
    print(colored(f"  Horizon: {T:.1f}s with discretization dt={dt:.4f}s", 'cyan'))
    print(colored(f"  Running cost Q: {np.diag(Q)}", 'cyan'))
    print(colored(f"  Terminal cost QN: {np.diag(QN)}", 'cyan'))
    print(colored(f"  Control cost R: {R[0,0]:.4f}", 'cyan'))
    print(colored(f"  Goal: x_goal = {x_goal}", 'cyan'))
    
    controller = builder.AddSystem(
        FiniteHorizonController(
            A=A_cont,
            B=B_cont,
            Q=Q,
            R=R,
            QN=QN,
            T=T,
            dt=dt,
            x_goal=x_goal,
            u_limits=(-100.0, 100.0),
            discretization="zoh"
        )
    )
    
    print(colored(f"✓ FiniteHorizonController created", 'green'))
    
    # ========== Wire System ==========
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    
    print(colored(f"✓ System wired: plant -> controller -> plant", 'green'))
    
    # ========== Visualization ==========
    meshcat = None
    viz_plant = None
    if args.visualize:
        print(colored(f"\nSetting up Meshcat visualization...", 'yellow', attrs=['bold']))
        
        # Create a separate MultibodyPlant for visualization only
        viz_plant, viz_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        
        # Cart
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = viz_plant.AddRigidBody("cart", cart_inertia)
        
        # Cart visuals
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        viz_plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        
        # Prismatic joint
        viz_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                viz_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -TRACK_LIMIT,
                TRACK_LIMIT,
                damping=CART_DAMPING
            )
        )
        
        # Pendulum
        I_about_pivot = PENDULUM_LENGTH**2
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = viz_plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Pendulum visuals
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH/2])),
            Cylinder(PENDULUM_RADIUS, PENDULUM_LENGTH),
            "pendulum_visual",
            np.array([0.8, 0.1, 0.1, 1.0])
        )
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH])),
            Sphere(PENDULUM_RADIUS * 1.5),
            "pendulum_tip",
            np.array([0.9, 0.2, 0.2, 1.0])
        )
        
        # Revolute joint
        viz_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=PENDULUM_DAMPING
            )
        )
        
        # Add actuator for the visualization plant
        viz_plant.AddJointActuator("cart_force", viz_plant.GetJointByName("cart_slider"))
        
        viz_plant.Finalize()
        
        # Add Meshcat visualizer
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, viz_scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration, prefix="finite_horizon_lqr")
        )
        
        print(colored("✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {meshcat.web_url()}", 'cyan'))
    
    # ========== Build Diagram ==========
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial conditions: [x, φ, ẋ, φ̇, F]
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    initial_angle_rad = np.deg2rad(args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE)
    plant.SetInitialConditions(
        plant_context,
        x0=0.0,
        phi0=initial_angle_rad,
        x_dot0=0.0,
        phi_dot0=0.0,
        F0=0.0
    )
    
    print(colored(f"\n✓ Initial conditions set:", 'green'))
    print(colored(f"  x = 0.0 m, φ = {args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°", 'cyan'))
    print(colored(f"  ẋ = 0.0 m/s, φ̇ = 0.0 rad/s", 'cyan'))
    print(colored(f"  F = 0.0 N", 'cyan'))
    
    # ========== Simulation ==========
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    
    print(colored(f"\nRunning simulation for {SIMULATION_TIME}s...", 'yellow', attrs=['bold']))
    
    # Data logging
    time_log = []
    state_log = []
    force_log = []
    
    LOGGING_INTERVAL = 0.01
    last_log_time = 0.0
    last_print_time = 0.0
    PRINT_INTERVAL = 1.0
    
    while context.get_time() < SIMULATION_TIME:
        simulator.AdvanceTo(context.get_time() + TIMESTEP)
        t = context.get_time()
        
        # Update visualization plant to match linearized plant state
        if meshcat is not None and viz_plant is not None:
            state = plant.get_output_port(0).Eval(plant_context)
            x, phi = state[0], state[1]  # Extract positions from 5D state
            viz_plant_context = viz_plant.GetMyMutableContextFromRoot(context)
            viz_plant.SetPositions(viz_plant_context, [x, phi])
        
        # Print progress
        if t >= last_print_time + PRINT_INTERVAL:
            state = plant.get_output_port(0).Eval(plant_context)
            print(colored(f"  t={t:.1f}s: x={state[0]:.3f}m, φ={np.rad2deg(state[1]):.1f}°, F={state[4]:.2f}N", 'cyan'))
            last_print_time = t
        
        # Log data
        if t >= last_log_time + LOGGING_INTERVAL:
            time_log.append(t)
            state = plant.get_output_port(0).Eval(plant_context)
            state_log.append(state.copy())
            force = controller.get_output_port(0).Eval(
                diagram.GetMutableSubsystemContext(controller, context)
            )
            force_log.append(force[0])
            last_log_time = t
    
    print(colored("\n✓ Simulation complete!", 'green'))
    
    # ========== Plot Results ==========
    import matplotlib.pyplot as plt
    times = np.array(time_log)
    states = np.array(state_log)
    forces = np.array(force_log)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Cart position
    axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
    axes[0, 0].axhline(y=FINITE_HORIZON_LQR_X_GOAL[0], color='r', linestyle='--', linewidth=2, alpha=0.7, label='Goal')
    axes[0, 0].set_ylabel('Cart Position (m)', fontsize=12)
    axes[0, 0].set_title('Cart Position', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Pendulum angle
    axes[0, 1].plot(times, np.rad2deg(states[:, 1]), 'r-', linewidth=2)
    axes[0, 1].axhline(y=np.rad2deg(FINITE_HORIZON_LQR_X_GOAL[1]), color='g', linestyle='--', linewidth=2, alpha=0.7, label='Goal')
    axes[0, 1].set_ylabel('Pendulum Angle (°)', fontsize=12)
    axes[0, 1].set_title('Pendulum Angle', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Cart velocity
    axes[1, 0].plot(times, states[:, 2], 'b-', linewidth=2)
    axes[1, 0].set_ylabel('Cart Velocity (m/s)', fontsize=12)
    axes[1, 0].set_title('Cart Velocity', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pendulum angular velocity
    axes[1, 1].plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2)
    axes[1, 1].set_ylabel('Angular Velocity (°/s)', fontsize=12)
    axes[1, 1].set_title('Pendulum Angular Velocity', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Control force F (state)
    axes[2, 0].plot(times, states[:, 4], 'm-', linewidth=2, label='F (state)')
    axes[2, 0].set_ylabel('Force State (N)', fontsize=12)
    axes[2, 0].set_xlabel('Time (s)', fontsize=12)
    axes[2, 0].set_title('Motor Force State', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Control command u
    axes[2, 1].plot(times, forces, 'g-', linewidth=2, label='u (command)')
    axes[2, 1].set_ylabel('Control Command', fontsize=12)
    axes[2, 1].set_xlabel('Time (s)', fontsize=12)
    axes[2, 1].set_title('Motor Command', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.suptitle(f'Finite-Horizon LQR - Linearized Plant (θ₀={args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/finite_horizon_lqr_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved: {filename}", 'green'))
    
    plt.show()


# ============================================================================
# EQUATION-BASED PLANT WITH COMPUTED TORQUE
# ============================================================================

def run_equations_plant_with_computed_torque():
    """Run equation-based plant with computed torque controller."""
    print("\n" + "="*70)
    print(colored("EQUATION-BASED PLANT with COMPUTED TORQUE CONTROLLER", 'cyan', attrs=['bold']))
    print("="*70 + "\n")
    
    builder = DiagramBuilder()
    
    # Create equation-based plant (pure dynamics, no geometry)
    plant = builder.AddSystem(CartPendulumSystemByEqns())
    
    # Create visualization plant (MultibodyPlant with geometry for Meshcat)
    if args.visualize:
        viz_plant, viz_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        
        # Build visualization plant (same geometry as main plant)
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = viz_plant.AddRigidBody("cart", cart_inertia)
        
        # Cart geometry
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        viz_plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        
        # Prismatic joint
        viz_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                viz_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -TRACK_LIMIT,
                TRACK_LIMIT,
                damping=CART_DAMPING
            )
        )
        
        # Pendulum
        I_about_pivot = PENDULUM_LENGTH**2
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = viz_plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Pendulum geometry
        pendulum_shape = Cylinder(PENDULUM_RADIUS, PENDULUM_TOTAL_LENGTH)
        viz_plant.RegisterVisualGeometry(
            pendulum_body, 
            RigidTransform(p=[0, 0, -PENDULUM_TOTAL_LENGTH/2]),
            pendulum_shape, "pendulum_visual",
            np.array([0.8, 0.1, 0.1, 1.0])
        )
        
        # Revolute joint
        viz_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=PENDULUM_DAMPING
            )
        )
        
        # Track visualization
        track_shape = Box(TRACK_LENGTH, 0.1, 0.1)
        viz_plant.RegisterVisualGeometry(
            viz_plant.world_body(),
            RigidTransform(p=[0, 0, -CART_HEIGHT/2 - 0.1]),
            track_shape, "track_visual",
            np.array([0.5, 0.5, 0.5, 0.8])
        )
        
        viz_plant.Finalize()
        
        # Add Meshcat visualizer
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, viz_scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration, prefix="analytical")
        )
        
        print(colored("✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {meshcat.web_url()}", 'cyan'))
    else:
        viz_plant = None
    
    # Create trajectory
    q_start = np.array([CART_START_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
    q_goal = np.array([CART_END_POSITION, np.deg2rad(PENDULUM_START_ANGLE)])
    traj_gen = MinJerkTrajectoryGenerator(
        q_start=q_start,
        q_goal=q_goal,
        duration=CART_MOTION_DURATION,
        settle_time=CART_SETTLE_TIME
    )
    
    print(colored(f"  Trajectory: MinJerk cart motion", 'cyan'))
    print(colored(f"    Start: x={CART_START_POSITION:.1f}m, θ={PENDULUM_START_ANGLE:.0f}°", 'cyan'))
    print(colored(f"    Goal:  x={CART_END_POSITION:.1f}m, θ={PENDULUM_START_ANGLE:.0f}°", 'cyan'))
    print(colored(f"    Settle: {CART_SETTLE_TIME:.1f}s, Motion: {CART_MOTION_DURATION:.1f}s", 'cyan'))
    
    # Create controller for equation-based system
    controller = builder.AddSystem(
        ComputedTorqueController(
            Kp=CT_KP, 
            Kd=CT_KD, 
            trajectory_generator=traj_gen,
            use_analytical=True
        )
    )
    
    # Wire connections
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial conditions
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    plant.SetInitialConditions(
        plant_context,
        x0=CART_START_POSITION,
        phi0=np.deg2rad(args.initial_theta),
        x_dot0=0.0,
        phi_dot0=0.0
    )
    
    # Get visualization plant context if visualization is enabled
    if viz_plant is not None:
        viz_plant_context = diagram.GetMutableSubsystemContext(viz_plant, context)
    
    # Run simulation
    simulator.set_target_realtime_rate(REALTIME_RATE)
    print(colored(f"\n⏱ Starting simulation (T={SIMULATION_TIME}s, dt={TIMESTEP}s)...", 'yellow'))
    print()  # Blank line before progress
    
    time_log = []
    state_log = []
    force_log = []
    last_log_time = 0.0
    last_print_time = 0.0
    
    while context.get_time() < SIMULATION_TIME:
        simulator.AdvanceTo(context.get_time() + TIMESTEP)
        
        t = context.get_time()
        
        # Get current state and control force for debugging
        state = plant.get_output_port(0).Eval(plant_context)
        controller_context = diagram.GetMutableSubsystemContext(controller, context)
        control_force = controller.get_output_port(0).Eval(controller_context)[0]
        
        # Update visualization plant state to match equation plant
        if viz_plant is not None:
            state = plant.get_output_port(0).Eval(plant_context)
            viz_plant.SetPositions(viz_plant_context, state[:2])
            viz_plant.SetVelocities(viz_plant_context, state[2:])
        
        # Print progress
        if t >= last_print_time + PRINT_INTERVAL:
            x, phi, x_dot, phi_dot = state
            phi_deg = np.rad2deg(phi)
            
            print(f"[{t:5.2f}s/{SIMULATION_TIME}s {int(100*t/SIMULATION_TIME):3d}%] "
                  f"x={x:6.3f}m θ={phi_deg:7.2f}° | "
                  f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(phi_dot):7.2f}°/s | "
                  f"F={control_force:7.2f}N")
            
            last_print_time = t
        
        # Log data
        if t >= last_log_time + LOGGING_INTERVAL:
            time_log.append(t)
            state = plant.get_output_port(0).Eval(plant_context)
            state_log.append(state.copy())
            force = controller.get_output_port(0).Eval(
                diagram.GetMutableSubsystemContext(controller, context)
            )
            force_log.append(force[0])
            last_log_time = t
    
    print(colored("\n✓ Simulation complete!", 'green'))
    
    # Plot results
    import matplotlib.pyplot as plt
    time_log = np.array(time_log)
    state_log = np.array(state_log)
    force_log = np.array(force_log)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(time_log, state_log[:, 0], 'b-', linewidth=2, label='Cart Position')
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].set_title('Equation-Based Plant - Computed Torque Control', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time_log, np.rad2deg(state_log[:, 1]), 'r-', linewidth=2, label='Pendulum Angle')
    axes[1].set_ylabel('Angle (deg)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(time_log, force_log, 'g-', linewidth=2, label='Control Force')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Force (N)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/equation_plant_ct_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved: {plot_path}", 'green'))
    plt.show()


# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
    Manages the Drake diagram for cart-pendulum system.
    
    Orchestrates:
    - Plant setup (MultibodyPlant with cart and pendulum)
    - Controller creation and wiring
    - Diagram building
    - Simulator creation and configuration
    - Simulation execution with data logging
    - Visualization setup
    - Post-simulation analysis and plotting
    
    Follows the template pattern from script_cup_manipulator_controller_drake.
    """
    
    def __init__(self, controller_mode: str = 'pd', plant_type: str = 'multibody', visualize: bool = True):
        """
        Initialize the scene manager.
        
        Args:
            controller_mode: Control mode ('pd', 'computed-torque', 'scene-viz', etc.)
            plant_type: Plant type ('multibody' or 'equations')
            visualize: Enable Meshcat visualization
        """
        self.controller_mode = controller_mode
        self.plant_type = plant_type
        self.visualize = visualize
        
        # Core Drake systems
        self.builder = DiagramBuilder()
        self.plant = None
        self.scene_graph = None
        self.controller = None
        self.meshcat = None
        self.diagram = None
        self.simulator = None
        self.context = None
        
        # Data logging
        self.time_log = []
        self.state_log = []
        self.force_log = []
        self.desired_state_log = []
        self.error_log = []
        
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("DrakeSceneManager Initialization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"  Controller Mode: {controller_mode}", 'yellow'))
        print(colored(f"  Plant Type: {plant_type}", 'yellow'))
        print(colored(f"  Visualization: {'Enabled' if visualize else 'Disabled'}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def setup_drake_system(self):
        """
        Build the Drake system: MultibodyPlant with cart and pendulum.
        
        Creates:
        - MultibodyPlant with physics timestep
        - Cart body (box with prismatic joint)
        - Pendulum body (point mass with revolute joint)
        - All geometry and collision
        """
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("Building Cart-Pendulum System", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        # Create plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=TIMESTEP
        )
        
        # Create cart (box sliding on track)
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = self.plant.AddRigidBody("cart", cart_inertia)
        
        # Add cart geometry
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        self.plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        self.plant.RegisterCollisionGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_collision",
            CoulombFriction(0.9, 0.8)
        )
        
        # Add prismatic joint (cart slides along x-axis)
        self.plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                self.plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],  # Slides along x-axis
                -TRACK_LIMIT,  # Lower limit
                TRACK_LIMIT,   # Upper limit
                damping=CART_DAMPING
            )
        )
        
        # Create pendulum using point mass at distance L from pivot
        I_about_pivot = PENDULUM_LENGTH**2  # Per unit mass
        
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),  # COM location
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = self.plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Add pendulum geometry (cylinder)
        pendulum_shape = Cylinder(PENDULUM_RADIUS, PENDULUM_TOTAL_LENGTH)
        self.plant.RegisterVisualGeometry(
            pendulum_body, RigidTransform([0, 0, -PENDULUM_LENGTH/2]), 
            pendulum_shape, "pendulum_visual",
            np.array([0.9, 0.1, 0.1, 1.0])
        )
        self.plant.RegisterCollisionGeometry(
            pendulum_body, RigidTransform([0, 0, -PENDULUM_LENGTH/2]),
            pendulum_shape, "pendulum_collision",
            CoulombFriction(0.1, 0.08)
        )
        
        # Add revolute joint (pendulum rotates about y-axis at cart pivot point)
        self.plant.AddJoint(
            RevoluteJoint(
                "pendulum_pivot",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],  # Rotates about y-axis
                damping=PENDULUM_DAMPING
            )
        )
        
        # Add gravity
        self.plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])
        
        # Finalize plant
        self.plant.Finalize()
        
        print(colored(f"✓ Plant built successfully", 'green'))
        print(colored(f"  Cart mass: {CART_MASS} kg", 'cyan'))
        print(colored(f"  Pendulum mass: {PENDULUM_MASS} kg, length: {PENDULUM_LENGTH} m", 'cyan'))
    
    def add_controller(self):
        """
        Add controller to the system.
        
        Wires:
        - Plant state output → Controller input
        - Controller output → Plant actuation input
        """
        print(colored(f"\nAdding Controller: {self.controller_mode}", 'yellow', attrs=['bold']))
        
        if self.controller_mode == 'scene-viz':
            # For visualization-only mode, add a zero-force controller
            from pydrake.systems.primitives import ConstantVectorSource
            self.controller = self.builder.AddSystem(
                ConstantVectorSource(np.zeros(1))
            )
            print(colored(f"✓ Zero-force controller (visualization only)", 'green'))
        
        elif self.controller_mode == 'pd':
            # PD controller with trajectory
            traj_gen = TrajectoryGenerator(mode='balance')
            self.controller = self.builder.AddSystem(
                PDController(PD_KP_CART, PD_KD_CART, PD_KP_PEND, PD_KD_PEND, traj_gen)
            )
            print(colored(f"✓ PD controller with trajectory generator", 'green'))
        
        elif self.controller_mode == 'computed-torque':
            # Computed torque controller
            traj_gen = TrajectoryGenerator(mode='swing')
            self.controller = self.builder.AddSystem(
                ComputedTorqueController(self.plant, traj_gen)
            )
            print(colored(f"✓ Computed torque controller", 'green'))
        
        elif self.controller_mode == 'energy-shaping':
            # Energy shaping swing-up + LQR balance
            traj_gen = TrajectoryGenerator(mode='swing')
            self.controller = self.builder.AddSystem(
                ReferenceImpedanceController(self.plant, traj_gen)
            )
            print(colored(f"✓ Energy shaping controller", 'green'))
        
        else:
            raise ValueError(f"Unknown controller mode: {self.controller_mode}")
        
        # Wire plant state to controller
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.controller.get_input_port(0)
        )
        
        # Wire controller output to plant actuation
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.plant.get_actuation_input_port()
        )
        
        print(colored(f"✓ Controller wired to plant", 'green'))
    
    def setup_visualization(self):
        """Setup Meshcat visualization."""
        if not self.visualize:
            return
        
        print(colored("\n📊 Setting up Meshcat visualization...", 'cyan'))
        
        self.meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat
        )
        
        print(colored(f"✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {self.meshcat.web_url()}", 'cyan'))
    
    def build_diagram(self):
        """Build the complete Drake diagram."""
        self.diagram = self.builder.Build()
        self.diagram.set_name("Cart-Pendulum Closed Loop System")
        print(colored(f"✓ Diagram built successfully", 'green'))
    
    def create_simulator(self):
        """Create and initialize the simulator."""
        print(colored("\n⚙️  Creating simulator...", 'cyan'))
        
        # Create context
        self.context = self.diagram.CreateDefaultContext()
        self.context.SetTime(0.0)
        
        # Set initial conditions
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Set initial cart position
        self.plant.GetJointByName("cart_slider").set_translation(plant_context, 0.0)
        
        # Set initial pendulum angle
        theta_initial = np.deg2rad(args.initial_theta)
        self.plant.GetJointByName("pendulum_pivot").set_angle(plant_context, theta_initial)
        
        # Create simulator
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.set_target_realtime_rate(1.0)
        
        print(colored(f"✓ Simulator created", 'green'))
        print(colored(f"   Initial pendulum angle: {args.initial_theta}°", 'cyan'))
        print(colored(f"   Realtime rate: 1.0x", 'cyan'))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored("\n▶️  Running Simulation", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        # Simulation loop
        t_next_print = 0.0
        t_next_log = 0.0
        
        while context.get_time() < SIMULATION_TIME:
            self.simulator.AdvanceTo(context.get_time() + TIMESTEP)
            
            t = context.get_time()
            
            # Print status
            if t >= t_next_print:
                state = self.plant.GetPositionsAndVelocities(plant_context)
                x, theta, x_dot, theta_dot = state
                theta_deg = np.rad2deg(theta)
                
                print(f"[{t:5.2f}s/{SIMULATION_TIME}s {int(100*t/SIMULATION_TIME):3d}%] "
                      f"x={x:6.3f}m θ={theta_deg:7.2f}° | "
                      f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(theta_dot):7.2f}°/s")
                
                t_next_print += PRINT_INTERVAL
            
            # Log data
            if t >= t_next_log:
                self.time_log.append(t)
                current_state = self.plant.GetPositionsAndVelocities(plant_context).copy()
                self.state_log.append(current_state)
                
                # Get control force
                controller_context = self.controller.GetMyContextFromRoot(context)
                force = self.controller.get_output_port(0).Eval(controller_context)
                self.force_log.append(force[0])
                
                t_next_log += LOGGING_INTERVAL
        
        print(colored("="*70, 'yellow'))
        print(colored(f"✓ Simulation completed successfully!", 'green', attrs=['bold']))
    
    def run_scene_viz(self):
        """Run interactive scene visualization."""
        print(colored("\n▶️  Running Scene Visualization", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        # Interactive visualization loop
        print(colored("\n🌐 Meshcat visualization active. Interact in the browser.", 'cyan'))
        print(colored("   Press Ctrl+C to exit", 'cyan'))
        
        try:
            t_next_update = 0.0
            while context.get_time() < SIMULATION_TIME:
                self.simulator.AdvanceTo(context.get_time() + 0.1)
                
                t = context.get_time()
                if t >= t_next_update:
                    self.diagram.ForcedPublish(context)
                    t_next_update += 0.05
        
        except KeyboardInterrupt:
            pass
        
        print(colored("\n" + "="*70, 'yellow'))
        print(colored(f"✓ Visualization complete!", 'green', attrs=['bold']))
    
    def extract_data(self):
        """Extract logged data from simulation."""
        print(colored("\n📈 Extracting simulation data...", 'cyan'))
        
        if self.time_log:
            print(colored(f"✓ Data extracted ({len(self.time_log)} samples)", 'green'))
        else:
            print(colored(f"⚠ No data logged", 'yellow'))
    
    def plot_results(self):
        """Generate comprehensive analysis plots."""
        print(colored("\n📊 Generating analysis plots...", 'cyan'))
        
        if not self.time_log:
            print(colored("⚠ No data to plot", 'yellow'))
            return
        
        time_log = np.array(self.time_log)
        state_log = np.array(self.state_log)
        force_log = np.array(self.force_log)
        
        # Extract state variables
        x = state_log[:, 0]
        theta = state_log[:, 1]
        x_dot = state_log[:, 2]
        theta_dot = state_log[:, 3]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(time_log, x, 'b-', linewidth=2, label='Cart Position')
        axes[0].set_ylabel('Position (m)', fontsize=12)
        axes[0].set_title('Cart-Pendulum System Response', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(time_log, np.rad2deg(theta), 'r-', linewidth=2, label='Pendulum Angle')
        axes[1].axhline(y=180, color='g', linestyle='--', alpha=0.5, label='Target (upright)')
        axes[1].set_ylabel('Angle (deg)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(time_log, force_log, 'g-', linewidth=2, label='Control Force')
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Force (N)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"plots/cart_pendulum_{self.controller_mode}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        print(colored(f"✓ Plot saved to: {plot_path}", 'green'))
    
    def print_summary(self):
        """Print simulation summary."""
        if not self.time_log:
            return
        
        print(colored("\n" + "="*70, 'green'))
        print(colored("SIMULATION SUMMARY", 'green', attrs=['bold']))
        print(colored("="*70, 'green'))
        
        # Final state
        state_array = np.array(self.state_log)
        if state_array.shape[0] > 0:
            final_state = state_array[-1]
            x_final = final_state[0]
            theta_final = final_state[1]
            
            print(colored(f"\n📊 Final State:", 'cyan'))
            print(f"   Cart position:    {x_final:.3f} m")
            print(f"   Pendulum angle:   {np.rad2deg(theta_final):.1f}°")
        
        # Simulation info
        print(colored(f"\n⏱️  Simulation Info:", 'cyan'))
        print(f"   Duration:         {self.time_log[-1]:.1f} s")
        print(f"   Samples:          {len(self.time_log)}")
        print(f"   Controller mode:  {self.controller_mode}")
        
        # Visualization info
        if self.visualize and self.meshcat:
            print(colored(f"\n🌐 Visualization:", 'cyan'))
            print(f"   Meshcat URL:      {self.meshcat.web_url()}")
        
        print(colored("="*70 + "\n", 'green'))
    
    def run_full_simulation(self):
        """Execute complete simulation pipeline."""
        try:
            self.setup_drake_system()
            self.add_controller()
            self.setup_visualization()
            self.build_diagram()
            self.create_simulator()
            
            if self.controller_mode == 'scene-viz':
                self.run_scene_viz()
            else:
                self.run_simulation()
                self.extract_data()
                self.plot_results()
                self.print_summary()
        
        except KeyboardInterrupt:
            print(colored("\n\nSimulation interrupted by user", 'yellow'))
        except Exception as e:
            print(colored(f"\n\nError: {e}", 'red'))
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function using DrakeSceneManager."""
    print("\n" + "=" * 70)
    print(colored("CART-PENDULUM SYSTEM - Drake Controller Architecture", 'cyan', attrs=['bold']))
    print(colored("Classic Underactuated Control Problem", 'cyan'))
    print("=" * 70)
    print(colored(f"Controller Mode: {CONTROLLER_MODE}", 'yellow', attrs=['bold']))
    print(colored(f"Plant Type: {PLANT_TYPE}", 'yellow'))
    print(colored(f"Initial Angle: {args.initial_theta}° (0=up, 180=down)", 'yellow'))
    print(colored(f"Time Step: {TIMESTEP} s", 'yellow'))
    print(colored(f"Duration: {SIMULATION_TIME} s", 'yellow'))
    print(colored(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}", 'yellow'))
    print("=" * 70 + "\n")
    
    # Special handling for modes that need dedicated functions
    if CONTROLLER_MODE == 'compare-models':
        compare_models()
        return
    
    if CONTROLLER_MODE == 'standard-lqr':
        run_standard_lqr_with_linearized_plant()
        return
    
    if CONTROLLER_MODE == 'finite-horizon-lqr':
        run_finite_horizon_lqr_with_linearized_plant()
        return
    
    if CONTROLLER_MODE == 'computed-torque' and PLANT_TYPE == 'equations':
        run_equations_plant_with_computed_torque()
        return
    
    # Use DrakeSceneManager for standard simulation modes
    manager = DrakeSceneManager(
        controller_mode=CONTROLLER_MODE,
        plant_type=PLANT_TYPE,
        visualize=args.visualize
    )
    
    manager.run_full_simulation()
    
    print(colored("\n" + "="*70, 'green'))
    print(colored("Execution Complete!", 'green', attrs=['bold']))
    print(colored("="*70 + "\n", 'green'))


if __name__ == "__main__":
    main()
