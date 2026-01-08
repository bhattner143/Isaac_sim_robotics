"""
=============================================================================
INVERTED PENDULUM LQR BALANCE CONTROL DEMONSTRATION
=============================================================================

This script demonstrates LQR (Linear Quadratic Regulator) control for balancing
an inverted pendulum at the upright position using Drake's LeafSystem framework.

TUTORIAL OBJECTIVES:
-------------------
1. Learn how to implement LQR control for unstable equilibria
2. Understand linearization around non-origin equilibrium points
3. Use Drake's simulation and control tools
4. Visualize phase portraits and convergence behavior

SYSTEM DESCRIPTION:
------------------
Simple Pendulum:
   - State: [θ, θ̇] where θ=0 is down, θ=π is up
   - Control input: torque τ applied at the base
   - Dynamics: θ̈ = -(g/l)sin(θ) - (b/m)θ̇ + τ/(ml²)
   - Upright equilibrium: θ=π, θ̇=0 (unstable without control)

LQR CONTROL:
-----------
Linearizes around upright position and designs optimal controller:
   - Cost: J = ∫(x'Qx + u'Ru)dt
   - Control law: u = -K*x where x = [θ-π, θ̇]
   - Q = diag(10, 1): Penalizes angle error
   - R = 1: Control effort penalty

AUTHOR: Tutorial for LQR Control
DATE: 2025
"""

# =============================================================================
# LAGRANGIAN MECHANICS FOR INVERTED PENDULUM
# =============================================================================
"""
DERIVATION OF EQUATIONS OF MOTION USING LAGRANGIAN MECHANICS
============================================================

SYSTEM CONFIGURATION:
--------------------
- Point mass: m (kg) at the end of a massless rigid rod
- Rod length: L (m) from pivot to mass
- Angle: θ (rad) measured from downward vertical
  * θ = 0: pendulum hangs downward (stable equilibrium)
  * θ = π: pendulum points upward (unstable equilibrium - inverted)
- External torque: τ (N·m) applied at the pivot
- Damping: b (N·m·s) viscous damping coefficient

COORDINATE SYSTEM:
-----------------
Let the pivot be at the origin. The position of the mass in Cartesian coordinates:
  x = L·sin(θ)
  z = -L·cos(θ)  (z-axis points up)

KINETIC ENERGY (T):
------------------
Velocity of the mass:
  ẋ = L·cos(θ)·θ̇
  ż = L·sin(θ)·θ̇

Kinetic energy:
  T = ½m(ẋ² + ż²)
    = ½m[L²cos²(θ)·θ̇² + L²sin²(θ)·θ̇²]
    = ½m·L²·θ̇²·[cos²(θ) + sin²(θ)]
    = ½m·L²·θ̇²
    = ½I·θ̇²

where I = m·L² is the moment of inertia about the pivot.

POTENTIAL ENERGY (V):
--------------------
Taking the pivot as the reference (V = 0):
  V = m·g·z = m·g·(-L·cos(θ)) = -m·g·L·cos(θ)

At θ = 0 (downward): V = -m·g·L (minimum energy)
At θ = π (upward):   V = +m·g·L (maximum energy)

LAGRANGIAN (L):
--------------
The Lagrangian is defined as:
  L = T - V = ½m·L²·θ̇² - (-m·g·L·cos(θ))
            = ½m·L²·θ̇² + m·g·L·cos(θ)

EULER-LAGRANGE EQUATION:
-----------------------
The Euler-Lagrange equation for a generalized coordinate q is:
  d/dt(∂L/∂q̇) - ∂L/∂q = Q

where Q is the generalized force (torque in our case).

For our system with q = θ:

1) ∂L/∂θ̇ = m·L²·θ̇

2) d/dt(∂L/∂θ̇) = m·L²·θ̈

3) ∂L/∂θ = -m·g·L·sin(θ)

4) Generalized force: Q = τ - b·θ̇
   (external torque minus damping torque)

Substituting into Euler-Lagrange equation:
  m·L²·θ̈ - (-m·g·L·sin(θ)) = τ - b·θ̇
  m·L²·θ̈ + m·g·L·sin(θ) = τ - b·θ̇
  m·L²·θ̈ + b·θ̇ + m·g·L·sin(θ) = τ

EQUATION OF MOTION:
------------------
Final nonlinear equation:
  m·L²·θ̈ + b·θ̇ + m·g·L·sin(θ) = τ

Solving for acceleration:
  θ̈ = [τ - b·θ̇ - m·g·L·sin(θ)] / (m·L²)
  θ̈ = τ/(m·L²) - (b/m·L²)·θ̇ - (g/L)·sin(θ)

LINEARIZATION AT UPRIGHT POSITION (θ = π):
-----------------------------------------
For small deviations θ̃ = θ - π from the upright position:
  sin(θ) = sin(π + θ̃) = -sin(θ̃) ≈ -θ̃  (for small θ̃)

Linearized equation:
  θ̈ ≈ τ/(m·L²) - (b/m·L²)·θ̇ + (g/L)·θ̃

State-space form with x = [θ̃, θ̇]ᵀ:
  ẋ = Ax + Bu

where:
  A = [    0         1    ]     B = [    0     ]
      [ g/L    -b/(m·L²) ]         [ 1/(m·L²) ]

STABILITY ANALYSIS:
------------------
Eigenvalues of A determine stability:
  - Downward (θ=0): Both eigenvalues negative → stable
  - Upward (θ=π):   One eigenvalue positive → unstable (requires control)

LQR CONTROL:
-----------
Design optimal state feedback u = -K·x that stabilizes the upright position
by minimizing the cost: J = ∫₀^∞ (x'Qx + u'Ru) dt
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Tuple, Optional
from copy import copy
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from IPython.display import display

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Linearize,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    Saturation,
    SceneGraph,
    Simulator,
    StartMeshcat,
    VectorLogSink,
    LogVectorOutput,
    wrap_to,
)
from pydrake.examples import PendulumGeometry, PendulumParams, PendulumPlant

# Set running_as_notebook to False for standalone script execution
running_as_notebook = False

# Start the Meshcat 3D visualizer for real-time animation
# This creates a web server that displays the pendulum simulation
meshcat = StartMeshcat()
print(f"Meshcat viewer available at: {meshcat.web_url()}")


# =============================================================================
# PARAMETER CLASSES
# =============================================================================
class LQRDesignParams:
    """Design parameters for LQR controller."""
    def __init__(self, Q: Optional[np.ndarray] = None, R: Optional[float] = None, saturation_limit: Optional[float] = None) -> None:
        # State cost matrix Q: penalize [θ-π, θ̇]
        self.Q: np.ndarray = Q if Q is not None else np.diag([10.0, 1.0])  # Heavy penalty on angle
        self.R: float = R if R is not None else 1.0  # Control cost
        self.saturation_limit: Optional[float] = saturation_limit  # Maximum control torque (N·m)


class SimulationParams:
    """Simulation parameters for demos."""
    def __init__(self) -> None:
        self.realtime_rate: float = 1.0  # Realtime simulation rate
        self.num_trials: int = 5  # Number of simulation trials for LQR demo
        self.sim_time: float = 4.0  # Simulation time per trial (seconds)
        self.print_dt: float = 0.5  # Print interval for state data (seconds)
        self.perturbation_scale: float = 0.3  # Scale for random initial perturbations


class VisualizationParams:
    """Visualization parameters for Meshcat display."""
    def __init__(self) -> None:
        self.realtime_rate: float = 1.0  # Visualization update rate (realtime multiplier)
        self.show_phase_plot: bool = True  # Whether to show phase portrait
        self.phase_plot_figsize: Tuple[int, int] = (10, 6)  # Figure size for phase plot
        self.save_plots: bool = True  # Whether to save plots to files


class LQRComputedParams:
    """LQR controller computed parameters and matrices."""
    def __init__(self, K: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: float) -> None:
        self.K: np.ndarray = K  # LQR gain matrix (1x2)
        self.S: np.ndarray = S  # Solution to Riccati equation (2x2)
        self.A: np.ndarray = A  # Linearized state matrix (2x2)
        self.B: np.ndarray = B  # Linearized input matrix (2x1)
        self.Q: np.ndarray = Q  # State cost matrix (2x2)
        self.R: float = R  # Control cost


# =============================================================================
# BASE SYSTEM CLASS - SHARED FUNCTIONALITY FOR STABILITY DEMOS
# =============================================================================
class StabilityDemo(LeafSystem):
    """
    Base class for stability demonstration systems.
    
    DRAKE LEAFSYSTEM OVERVIEW:
    -------------------------
    LeafSystem is Drake's fundamental building block for dynamical systems.
    Provides:
      - State variables (continuous and/or discrete)
      - Input ports (for external signals)
      - Output ports (for exposing internal state/measurements)
      - System dynamics (how state evolves over time)
    
    KEY METHODS TO IMPLEMENT:
    ------------------------
    1. __init__: Declare state, inputs, outputs, parameters
    2. DoCalcTimeDerivatives: Define continuous dynamics ẋ = f(x,u,t)
    3. Output port callbacks: Compute outputs from state
    
    WHY USE LEAFSYSTEM?
    -------------------
    - Automatic numerical integration
    - Built-in context management
    - Composability: Connect multiple systems into diagrams
    - Analysis tools: Linearization, optimization, etc.
    """
    
    def __init__(self, num_states, name="StabilitySystem"):
        """
        Initialize the base system.
        
        Args:
            num_states (int): Dimension of the continuous state vector
            name (str): System name for debugging/logging
        """
        LeafSystem.__init__(self)
        self.set_name(name)
        self._num_states = num_states
        
        # Declare continuous state vector
        self.DeclareContinuousState(num_states)
        
        # Declare output port that exposes the full state vector
        self.DeclareVectorOutputPort("state", BasicVector(num_states), self.CopyStateOut)
    
    def CopyStateOut(self, context, output):
        """
        Output port callback: Copy the continuous state to the output port.
        
        Args:
            context: Current system context (contains state, time, etc.)
            output: Output vector to be filled
        """
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)
    
    def simulate(self, initial_state, t0=0.0, tf=40.0, dt=0.01):
        """
        Simulate the dynamical system and return the state trajectory.
        
        Args:
            initial_state (np.array): Initial state vector x(0)
            t0 (float): Start time
            tf (float): Final simulation time
            dt (float): Logging timestep
        
        Returns:
            t (np.array): Time vector
            z (np.array): State trajectory, shape (n_samples, n_states)
        """
        diagram, system_in_diagram, logger = self._build_simulation_diagram()
        
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        system_context = diagram.GetMutableSubsystemContext(system_in_diagram, context)
        system_context.get_mutable_continuous_state_vector().SetFromVector(initial_state)
        
        simulator.Initialize()
        simulator.AdvanceTo(tf)
        
        log = logger.FindLog(context)
        t = log.sample_times()
        z = log.data().T
        
        return t, z
    
    def _build_simulation_diagram(self):
        """
        Build a Drake diagram for simulation (system + logger).
        
        Returns:
            tuple: (diagram, system_in_diagram, logger)
        """
        system = self._create_fresh_instance()
        
        builder = DiagramBuilder()
        system_in_diagram = builder.AddSystem(system)
        logger = LogVectorOutput(system_in_diagram.get_output_port(0), builder)
        diagram = builder.Build()
        
        return diagram, system_in_diagram, logger
    
    def _create_fresh_instance(self):
        """
        Create a fresh instance of this system.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _create_fresh_instance()")


# =============================================================================
# CART-POLE SYSTEM WITH LQR CONTROL
# =============================================================================
class CartPoleSystem(StabilityDemo):
    """
    Cart-Pole (Inverted Pendulum) System for LQR Balance Control.
    
    PHYSICAL INTERPRETATION:
    -----------------------
    An inverted pendulum mounted on a cart or pivot:
      - Angle θ: measured from downward vertical (θ=0 down, θ=π up)
      - Angular velocity θ̇
      - Upright equilibrium: (θ, θ̇) = (π, 0) - UNSTABLE
    
    DYNAMICS:
    --------
    Nonlinear: θ̈ = -(g/l)sin(θ) - (b/m)θ̇ + τ/(ml²)
    
    At upright position (θ=π), linearized around equilibrium:
      State: x = [θ-π, θ̇]
      Dynamics: ẋ = Ax + Bu
    
    STABILITY:
    ---------
    Without control: UNSTABLE (eigenvalues have positive real parts)
    With LQR control: ASYMPTOTICALLY STABLE (eigenvalues moved to left half-plane)
    
    CONTROL STRATEGY:
    ----------------
    LQR (Linear Quadratic Regulator):
      - Linearize around upright position
      - Design optimal state feedback: u = -K*x
      - Minimizes cost: J = ∫(x'Qx + u'Ru)dt
    """
    
    def __init__(self, pendulum_plant: Optional[PendulumPlant] = None, K: Optional[np.ndarray] = None) -> None:
        """
        Initialize the Cart-Pole system.
        
        Args:
            pendulum_plant: Drake PendulumPlant instance
            K: LQR gain matrix (1x2)
        """
        super().__init__(num_states=2, name="CartPole")
        
        # Store pendulum plant and LQR gains
        self.pendulum_plant: PendulumPlant = pendulum_plant if pendulum_plant else PendulumPlant()
        self.K: Optional[np.ndarray] = K
    
    def _create_fresh_instance(self):
        """Create a new CartPoleSystem instance."""
        return CartPoleSystem(pendulum_plant=self.pendulum_plant, K=self.K)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute time derivatives for the cart-pole system.
        
        This uses the PendulumPlant dynamics from Drake.
        
        Args:
            context: Current system context
            derivatives: Object to store computed derivatives
        """
        # Get current state
        state = context.get_continuous_state_vector().CopyToVector()
        
        # For now, this is a wrapper around PendulumPlant
        # In practice, we'd compute the pendulum dynamics here
        # θ̇ = ω
        # ω̇ = -(g/l)sin(θ) - (b/m)ω + τ/(ml²)
        
        theta = state[0]
        theta_dot = state[1]
        
        # Simple pendulum dynamics (normalized parameters)
        g = 9.81  # gravity
        l = 1.0   # length
        b = 0.1   # damping
        m = 1.0   # mass
        
        # Apply LQR control if available
        tau = 0.0
        if self.K is not None:
            # Wrap angle to [-π, π] and compute error from upright
            theta_wrapped = wrap_to(theta, 0, 2.0 * np.pi) - np.pi
            xbar = np.array([theta_wrapped, theta_dot])
            tau = -self.K.dot(xbar)[0]
            # Saturate control
            tau = np.clip(tau, -3.0, 3.0)
        
        # Compute derivatives
        theta_ddot = -(g/l) * np.sin(theta) - (b/m) * theta_dot + tau/(m*l**2)
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def PhasePlot(pendulum: PendulumPlant, viz_params: Optional[VisualizationParams] = None):
    """Create a phase portrait plot for the pendulum.
    
    The phase portrait shows θ (angle) vs θ̇ (angular velocity) centered
    around the upright position.
    
    Args:
        pendulum: PendulumPlant instance
        viz_params: VisualizationParams instance (uses defaults if None)
        
    Returns:
        matplotlib axes object
    """
    if viz_params is None:
        viz_params = VisualizationParams()
    
    phase_plot = plt.figure(figsize=viz_params.phase_plot_figsize)
    ax = phase_plot.gca()
    ax.set_xlabel('θ (rad)', fontsize=12)
    ax.set_ylabel('θ̇ (rad/s)', fontsize=12)
    ax.set_title('Phase Portrait: LQR Balancing at Upright Position', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Center plot around upright position (θ = π)
    theta_lim = [np.pi - 1.5, np.pi + 1.5]
    ax.set_xlim(theta_lim)
    ax.set_ylim(-5.0, 5.0)
    
    # Mark the equilibrium point
    ax.plot(np.pi, 0, 'r*', markersize=15, label='Equilibrium (upright)')
    ax.legend()

    return ax

# =============================================================================
# LQR CONTROLLER SYSTEM
# =============================================================================
class LQRController(LeafSystem):
    """
    LQR Controller for Balancing the Pendulum at Upright Position.
    
    CONTROL THEORY:
    --------------
    Linear Quadratic Regulator (LQR) is an optimal control technique that:
    1. Linearizes nonlinear dynamics around an equilibrium point
    2. Designs state feedback u = -K*x that minimizes a quadratic cost
    3. Provides guaranteed stability margins for the linearized system
    
    COST FUNCTION:
    -------------
    J = ∫₀^∞ (x'Qx + u'Ru) dt
    
    where:
      - Q: State penalty matrix (penalizes deviation from equilibrium)
      - R: Control penalty matrix (penalizes large control inputs)
      - x: State error from equilibrium
      - u: Control input
    
    CONTROL LAW:
    -----------
    u = -K*x where K is computed by solving the algebraic Riccati equation.
    
    For the inverted pendulum:
      - x = [θ-π, θ̇] (error from upright position)
      - u = τ (applied torque)
      - K is a 1×2 gain matrix
    
    STABILITY:
    ---------
    LQR guarantees:
      - Closed-loop system is asymptotically stable
      - Gain margin: ∞ in [0.5, ∞)
      - Phase margin: ≥ 60°
      - Robustness to modeling errors
    """
    
    def __init__(self, pendulum: PendulumPlant, lqr_design_params: Optional[LQRDesignParams] = None) -> None:
        """
        Initialize the LQR controller.
        
        Args:
            pendulum: PendulumPlant to control
            lqr_design_params: LQRDesignParams instance (uses defaults if None)
        """
        LeafSystem.__init__(self)
        
        # Input port: receives state [θ, θ̇] from pendulum
        self.DeclareVectorInputPort("state", 2)
        
        # Output port: sends control torque τ to pendulum
        self.DeclareVectorOutputPort("control", 1, self.DoCalcOutput)
        
        # Store reference to pendulum plant
        self.pendulum: PendulumPlant = pendulum
        
        # Use provided design params or create defaults
        if lqr_design_params is None:
            lqr_design_params = LQRDesignParams()
        self.design_params: LQRDesignParams = lqr_design_params
        
        # Design LQR controller for balancing
        # K: optimal gain matrix, S: Riccati solution
        self.computed_params = self._design_lqr_controller()
        self.K = self.computed_params.K
        self.S = self.computed_params.S
        
        print(f"\nLQR Controller Initialized:")
        print(f"  Q = diag({np.diag(self.design_params.Q)})")
        print(f"  R = {self.design_params.R}")
        print(f"  Gain matrix K = {self.K}")
        print(f"  Riccati solution S =\n{self.S}")
        if self.design_params.saturation_limit is not None:
            print(f"  Saturation limit = {self.design_params.saturation_limit} N·m")

    def DoCalcOutput(self, context, output):
        """
        Compute control torque using LQR feedback law.
        
        CONTROL COMPUTATION:
        -------------------
        1. Read current pendulum state [θ, θ̇]
        2. Wrap angle to [-π, π] to handle periodicity
        3. Compute error from upright: x = [θ-π, θ̇]
        4. Apply LQR law: u = -K*x
        
        Args:
            context: Current system context
            output: Output port to write control torque
        """
        # Read current state from input port
        pendulum_state = self.get_input_port(0).Eval(context)
        
        # Wrap angle to [-π, π] range for LQR
        # This handles the periodicity of the angle (θ and θ+2π are the same)
        # After wrapping, upright position is at 0 instead of π
        xbar = copy(pendulum_state)
        xbar[0] = wrap_to(xbar[0], 0, 2.0 * np.pi) - np.pi

        # LQR control law: u = -K*x drives the system to equilibrium
        # Negative sign because we want to oppose the error
        control_input = -self.K.dot(xbar)
        
        output.SetFromVector(control_input)
    
    def _design_lqr_controller(self) -> LQRComputedParams:
        """
        Design an LQR controller for balancing the pendulum at the upright position.
        
        LINEARIZATION:
        -------------
        The nonlinear pendulum dynamics are linearized around the upright equilibrium:
          Equilibrium point: (θ, θ̇) = (π, 0)
          Linearized system: ẋ = Ax + Bu
        
        LQR DESIGN:
        ----------
        Minimizes the infinite-horizon quadratic cost:
          J = ∫₀^∞ (x'Qx + u'Ru) dt
        
        where:
          Q = diag(10, 1): State cost matrix
                          - Heavy penalty on angle error (θ-π)
                          - Light penalty on velocity error (θ̇)
          R = 1: Control cost (torque penalty)
        
        SOLUTION:
        --------
        The optimal control law is: u = -K*x
        where K is computed by solving the Algebraic Riccati Equation (ARE):
          A'S + SA - SBR⁻¹B'S + Q = 0
        
        The gain is: K = R⁻¹B'S
        
        Returns:
            tuple: (K, S)
                - K: LQR gain matrix (1×2) such that u = -K*x
                - S: Solution to the Riccati equation (2×2 positive definite matrix)
                     Used for stability analysis and region of attraction estimation
        """
        # Create context for linearization
        context = self.pendulum.CreateDefaultContext()

        # Set equilibrium point: upright position with zero torque
        self.pendulum.get_input_port(0).FixValue(context, [0])
        context.SetContinuousState([np.pi, 0])  # θ = π (upright), θ̇ = 0

        # Get LQR cost matrices from design params
        Q = self.design_params.Q
        R = [self.design_params.R]

        # Linearize nonlinear dynamics around the upright position
        # This computes the Jacobian matrices A = ∂f/∂x and B = ∂f/∂u
        linearized_pendulum = Linearize(self.pendulum, context)
        A = linearized_pendulum.A()
        B = linearized_pendulum.B()
        
        # Solve LQR problem: returns gain K and Riccati solution S
        (K, S) = LinearQuadraticRegulator(A, B, Q, R)
        
        # Return all computed parameters
        return LQRComputedParams(K, S, A, B, Q, self.design_params.R)

    def DoCalcOutput(self, context, output):
        """
        Compute control torque using LQR feedback law.
        
        CONTROL COMPUTATION:
        -------------------
        1. Read current pendulum state [θ, θ̇]
        2. Wrap angle to [-π, π] to handle periodicity
        3. Compute error from upright: x = [θ-π, θ̇]
        4. Apply LQR law: u = -K*x
        
        Args:
            context: Current system context
            output: Output port to write control torque
        """
        # Read current state from input port
        pendulum_state = self.get_input_port(0).Eval(context)
        
        # Wrap angle to [-π, π] range for LQR
        # This handles the periodicity of the angle (θ and θ+2π are the same)
        # After wrapping, upright position is at 0 instead of π
        xbar = copy(pendulum_state)
        xbar[0] = wrap_to(xbar[0], 0, 2.0 * np.pi) - np.pi

        # LQR control law: u = -K*x drives the system to equilibrium
        # Negative sign because we want to oppose the error
        control_input = -self.K.dot(xbar)
        
        output.SetFromVector(control_input)


# =============================================================================
# HELPER FUNCTIONS FOR STATE PRINTING
# =============================================================================
def print_state_header() -> None:
    """Print table header for state data."""
    print("\n" + "="*80)
    print(f"{'Time (s)':>10} | {'θ (rad)':>12} | {'θ (deg)':>12} | {'θ̇ (rad/s)':>15} | {'τ (Nm)':>10}")
    print("="*80)

def print_state_row(time: float, theta: float, theta_dot: float, torque: float) -> None:
    """Print a single row of state data in tabular format."""
    theta_deg = np.rad2deg(theta)
    print(f"{time:10.3f} | {theta:12.4f} | {theta_deg:12.2f} | {theta_dot:15.4f} | {torque:10.4f}")

def print_state_summary(log_data, title: str = "Simulation Summary") -> None:
    """Print summary statistics of the simulation."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    theta_data = log_data.data()[0, :]
    theta_dot_data = log_data.data()[1, :]
    
    # Wrap angles to [-π, π] for better statistics
    theta_wrapped = np.array([wrap_to(t, 0, 2.0 * np.pi) - np.pi for t in theta_data])
    
    print(f"  Initial state: θ = {theta_wrapped[0]:.4f} rad ({np.rad2deg(theta_wrapped[0]):.2f}°), θ̇ = {theta_dot_data[0]:.4f} rad/s")
    print(f"  Final state:   θ = {theta_wrapped[-1]:.4f} rad ({np.rad2deg(theta_wrapped[-1]):.2f}°), θ̇ = {theta_dot_data[-1]:.4f} rad/s")
    print(f"  Max |θ| error from upright: {np.max(np.abs(theta_wrapped)):.4f} rad ({np.rad2deg(np.max(np.abs(theta_wrapped))):.2f}°)")
    print(f"  Max |θ̇|: {np.max(np.abs(theta_dot_data)):.4f} rad/s")
    print(f"  RMS θ error: {np.sqrt(np.mean(theta_wrapped**2)):.4f} rad")
    print("="*80)


# =============================================================================
# MAIN DEMONSTRATION FUNCTIONS
# =============================================================================
def no_control_demo(show: bool = False) -> None:
    """
    Run Pendulum Demo WITHOUT Control (Natural Behavior).
    
    DEMONSTRATION OVERVIEW:
    ----------------------
    This demo shows the natural dynamics of the inverted pendulum without any control:
    - Pendulum starts near upright position
    - Falls due to gravity (unstable equilibrium)
    - Oscillates around the downward position
    
    WHAT TO OBSERVE:
    ---------------
    - Unstable equilibrium at θ=π (upright)
    - Pendulum falls and swings
    - Natural oscillatory behavior around θ=0 (downward)
    - Energy conservation (no damping, only numerical errors)
    
    Args:
        show: If True, display plot interactively; if False, save to file
    """
    # Create visualization parameters
    viz_params = VisualizationParams()
    
    # Build the system diagram
    builder = DiagramBuilder()

    # Add pendulum plant (no control)
    pendulum = builder.AddSystem(PendulumPlant())
    
    # Create phase plot
    ax = PhasePlot(pendulum, viz_params)
    ax.set_title('Phase Portrait: NO CONTROL (Natural Dynamics)', fontsize=14)
    
    # Fix input to zero (no control torque)
    builder.ExportInput(pendulum.get_input_port(0), "torque")

    # Setup Meshcat 3D visualization
    scene_graph = builder.AddSystem(SceneGraph())
    PendulumGeometry.AddToBuilder(
        builder, pendulum.get_state_output_port(), scene_graph
    )
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Add state logger
    logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(pendulum.get_output_port(0), logger.get_input_port(0))

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Set realtime rate for real-time visualization
    simulator.set_target_realtime_rate(viz_params.realtime_rate)

    # Run simulation with small perturbation from upright
    print("\n" + "="*80)
    print("NO CONTROL DEMO - Observing Natural Pendulum Dynamics")
    print("="*80)
    print("The pendulum will fall from near-upright position (unstable equilibrium)")
    
    context.SetTime(0.0)
    # Small perturbation from upright position
    initial_state = np.array([np.pi - 0.1, 0.1])
    print(f"\nInitial state: θ = {initial_state[0]:.4f} rad ({np.rad2deg(initial_state[0]):.2f}°), θ̇ = {initial_state[1]:.4f} rad/s")
    context.SetContinuousState(initial_state)
    
    # Fix control input to zero
    diagram.get_input_port(0).FixValue(context, [0.0])
    
    simulator.Initialize()
    
    # Print header for state table
    print_state_header()
    
    # Simulate for 5 seconds
    sim_time = 5.0
    dt = 0.5  # Print every 0.5 seconds
    for i in range(int(sim_time/dt) + 1):
        t = i * dt
        if t > 0:
            simulator.AdvanceTo(t)
        state = pendulum.get_output_port(0).Eval(pendulum.GetMyContextFromRoot(context))
        print_state_row(t, state[0], state[1], 0.0)
    
    # Plot trajectory
    log = logger.FindLog(context)
    ax.plot(log.data()[0, :], log.data()[1, :], linewidth=2, label='No Control', color='red')
    ax.legend()
    
    # Print summary
    print_state_summary(log, "NO CONTROL - Summary")
    
    # Save plot
    if not show and viz_params.save_plots:
        plt.savefig('pendulum_no_control_phase_plot.png', dpi=150, bbox_inches='tight')
        print("\nPhase plot saved as 'pendulum_no_control_phase_plot.png'")


def lqr_balance_demo(show: bool = False) -> None:
    """
    Run LQR Balance Controller Demonstration with Visualization.
    
    DEMONSTRATION OVERVIEW:
    ----------------------
    This demo illustrates LQR control stabilizing an inverted pendulum:
    1. Linearizes pendulum dynamics around upright position
    2. Designs optimal LQR controller
    3. Runs multiple simulations with random perturbations
    4. Visualizes convergence in phase space
    
    WHAT TO OBSERVE:
    ---------------
    - Phase portrait trajectories spiral inward to equilibrium
    - All trajectories converge to (θ=π, θ̇=0) regardless of initial condition
    - Control torque adapts to bring pendulum back to upright
    - Demonstrates asymptotic stability of closed-loop system
    
    SIMULATION SETUP:
    ----------------
    - Number of trials: 5
    - Initial conditions: Small random perturbations from upright
    - Simulation time: 4 seconds per trial
    - Control saturation: ±3 Nm (realistic actuator limits)
    
    Args:
        show: If True, display plot interactively; if False, save to file
    
    EXPECTED BEHAVIOR:
    -----------------
    Phase plot should show:
      - Multiple trajectories starting near (π, 0)
      - Spiraling convergence to equilibrium point
      - Exponential decay of state error over time
    """
    # Create parameter objects
    lqr_design = LQRDesignParams(
        Q=np.diag([10.0, 1.0]),
        R=1.0,
        saturation_limit=3.0
    )
    sim_params = SimulationParams()
    viz_params = VisualizationParams()
    
    # Build the system diagram
    builder = DiagramBuilder()

    # Add pendulum plant
    pendulum = builder.AddSystem(PendulumPlant())
    
    # Create phase plot
    ax = PhasePlot(pendulum, viz_params)
    
    # Add saturation to limit control torque
    saturation = builder.AddSystem(Saturation(
        min_value=[-lqr_design.saturation_limit], 
        max_value=[lqr_design.saturation_limit]
    ))
    builder.Connect(saturation.get_output_port(0), pendulum.get_input_port(0))
    
    # Add LQR controller with design parameters
    controller = builder.AddSystem(LQRController(pendulum, lqr_design))
    builder.Connect(pendulum.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))

    # Setup Meshcat 3D visualization
    scene_graph = builder.AddSystem(SceneGraph())
    PendulumGeometry.AddToBuilder(
        builder, pendulum.get_state_output_port(), scene_graph
    )
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Add state logger
    logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(pendulum.get_output_port(0), logger.get_input_port(0))

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Set realtime rate for real-time visualization
    simulator.set_target_realtime_rate(viz_params.realtime_rate)

    # Run multiple simulations with small perturbations from upright
    for i in range(sim_params.num_trials):
        print("\n" + "="*80)
        print(f"LQR Balance Demo - Simulation {i+1}/{sim_params.num_trials}")
        print("="*80)
        context.SetTime(0.0)
        # Small random perturbation from upright position
        initial_state = np.array([np.pi, 0.0]) + sim_params.perturbation_scale * np.random.randn(2)
        print(f"Initial state: θ = {initial_state[0]:.4f} rad ({np.rad2deg(initial_state[0]):.2f}°), θ̇ = {initial_state[1]:.4f} rad/s")
        context.SetContinuousState(initial_state)
        simulator.Initialize()
        
        # Print header for state table
        print_state_header()
        
        # Simulate with state printing
        sim_time = sim_params.sim_time
        dt = sim_params.print_dt
        for j in range(int(sim_time/dt) + 1):
            t = j * dt
            if t > 0:
                simulator.AdvanceTo(t)
            state = pendulum.get_output_port(0).Eval(pendulum.GetMyContextFromRoot(context))
            
            # Get control torque from controller
            controller_context = controller.GetMyContextFromRoot(context)
            torque_output = controller.get_output_port(0).Eval(controller_context)
            torque = np.clip(torque_output[0], -lqr_design.saturation_limit, lqr_design.saturation_limit)  # Apply saturation
            
            print_state_row(t, state[0], state[1], torque)
        
        # Plot trajectory - should converge to (π, 0)
        log = logger.FindLog(context)
        ax.plot(log.data()[0, :], log.data()[1, :], linewidth=2, label=f'Trial {i+1}' if i < 3 else '')
        
        # Print summary for this trial
        print_state_summary(log, f"LQR CONTROL - Trial {i+1} Summary")
        
        log.Clear()
        print(f"\n✓ Simulation {i+1} complete!")

    # Set phase plot limits centered around upright position
    ax.set_xlim(np.pi - 1.5, np.pi + 1.5)
    ax.set_ylim(-5.0, 5.0)
    ax.legend()
    
    # Save or display the phase plot
    if show:
        display(mpld3.display())
    elif viz_params.save_plots:
        plt.savefig('pendulum_lqr_phase_plot.png', dpi=150, bbox_inches='tight')
        print("\n" + "="*80)
        print("Phase plot saved as 'pendulum_lqr_phase_plot.png'")
        print("="*80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Main Function to Run Pendulum Demonstrations.
    
    DEMONSTRATION OPTIONS:
    ---------------------
    Option 1: NO CONTROL - Observe natural pendulum dynamics
              - Shows unstable equilibrium at upright position
              - Pendulum falls and oscillates naturally
              
    Option 2: LQR CONTROL - Stabilize pendulum at upright position
              - Demonstrates optimal feedback control
              - Shows asymptotic stability and convergence
    
    LEARNING OUTCOMES:
    -----------------
    After running this demo, you should understand:
      - Why upright position is unstable without control
      - How LQR stabilizes unstable equilibria
      - The role of Q and R matrices in LQR design
      - Phase portrait interpretation for closed-loop systems
      - Practical considerations (saturation, wrapping angles)
    
    CONTROLS:
    --------
      - Choose demo option (1 or 2)
      - View 3D animation at Meshcat URL
      - Press Ctrl+C to exit when done
    """
    print("\n" + "="*80)
    print("INVERTED PENDULUM CONTROL DEMONSTRATION")
    print("="*80)
    print("\nTUTORIAL OBJECTIVES:")
    print("  • Compare natural vs. controlled pendulum behavior")
    print("  • Demonstrate LQR control for unstable equilibrium stabilization")
    print("  • Visualize phase portrait convergence behavior")
    print("  • Illustrate optimal control theory in practice")
    print("\nSYSTEM DETAILS:")
    print("  • Inverted pendulum (1-DOF)")
    print("  • Upright equilibrium: θ=π (unstable)")
    print("  • Downward equilibrium: θ=0 (stable)")
    print("  • Control: LQR with Q=diag(10,1), R=1")
    print("  • Saturation: τ ∈ [-3, 3] Nm")
    print("="*80)
    
    print("\nDEMONSTRATION OPTIONS:")
    print("  1) NO CONTROL - Observe natural pendulum dynamics (falls from upright)")
    print("  2) LQR CONTROL - Stabilize pendulum at upright position")
    print("="*80)
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        print("\n" + "="*80)
        print("RUNNING: NO CONTROL DEMO")
        print("="*80)
        no_control_demo(show=False)
        output_file = "pendulum_no_control_phase_plot.png"
    else:
        print("\n" + "="*80)
        print("RUNNING: LQR CONTROL DEMO")
        print("="*80)
        lqr_balance_demo(show=False)
        output_file = "pendulum_lqr_phase_plot.png"
    
    # Keep the program running so Meshcat stays active
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\n  Meshcat visualization: {meshcat.web_url()}")
    print(f"  Phase portrait saved: {output_file}")
    print("\nThe Meshcat viewer will remain active for inspection.")
    print("Press Ctrl+C to exit when done...")
    print("="*80)
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nExiting demo... Goodbye!")


if __name__ == "__main__":
    main()