"""
=============================================================================
SIMPLE PENDULUM LQR CONTROL - BUILT FROM SCRATCH
=============================================================================

This script demonstrates LQR control for a simple pendulum system built
entirely from scratch using Drake's LeafSystem framework.

TUTORIAL OBJECTIVES:
-------------------
1. Implement simple pendulum dynamics from first principles
2. Design LQR controller for stabilizing the inverted pendulum
3. Compare controlled vs. uncontrolled behavior
4. Understand the complete control pipeline from dynamics to stabilization

SYSTEM DESCRIPTION:
------------------
Simple Pendulum (matching PDF notation):
   - Mass: m (point mass at end of massless rod)
   - Length: L (from pivot to center of mass)
   - State: [θ, θ̇] where:
     * θ: pendulum angle measured from downward vertical
     * θ̇: angular velocity
   - Control input: τ (torque applied at pivot)
   - Upright equilibrium: θ=π, θ̇=0 (UNSTABLE)
   - Downward equilibrium: θ=0, θ̇=0 (STABLE)

DYNAMICS (Derived from Lagrangian - PDF Equation 15):
-----------------------------------------------------
Lagrangian:
   L = T - V = ½mL²θ̇² - (-mgLcos(θ)) = ½mL²θ̇² + mgLcos(θ)

Euler-Lagrange equation:
   d/dt(∂L/∂θ̇) - ∂L/∂θ = τ

Results in equation of motion (PDF Eq. 15):
   mL²θ̈ + mgLsin(θ) + bθ̇ = τ

Simplified (normalized):
   θ̈ = -(g/L)sin(θ) - (b/mL²)θ̇ + τ/(mL²)

LQR CONTROL:
-----------
Linearizes around upright position and designs optimal controller:
   - Cost: J = ∫(x'Qx + u'Ru)dt
   - Control law: u = -K*x where x = [θ-π, θ̇]
   - Q: State penalty matrix
   - R: Control effort penalty

AUTHOR: Tutorial for Simple Pendulum Control from Scratch
DATE: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Simulator,
    VectorLogSink,
    LogVectorOutput,
    wrap_to,
    RigidTransform,  # <-- Added import
)

# Import for diagram visualization
try:
    import pydot
    PYDOT_AVAILABLE = True
except ImportError:
    PYDOT_AVAILABLE = False
    print("Warning: pydot not available. Diagram visualization disabled.")

# Meshcat imports for 3D visualization
try:
    from pydrake.all import (
        MeshcatVisualizer,
        StartMeshcat,
        Meshcat,
    )
    from pydrake.geometry import (
        MeshcatVisualizerParams,
        Role,
        Sphere,
        Cylinder,
    )
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    print("Warning: Meshcat not available. Visualization disabled.")

# Meshcat imports for 3D visualization
try:
    from pydrake.all import (
        MeshcatVisualizer,
        StartMeshcat,
        Meshcat,
    )
    from pydrake.geometry import (
        MeshcatVisualizerParams,
        Role,
        Sphere,
        Cylinder,
    )
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    print("Warning: Meshcat not available. Visualization disabled.")

# Physical parameters for pendulum system
class PendulumParams:
    """Physical parameters for the simple pendulum system."""
    def __init__(self) -> None:
        self.m: float = 1.0      # Pendulum mass (kg)
        self.L: float = 1.0      # Pendulum length (m) - using L notation from PDF
        self.L: float = self.L   # Alias for compatibility
        self.g: float = 9.81     # Gravity (m/s²)
        self.b: float = 0.1      # Damping coefficient (N·m·s)


class LQRDesignParams:
    """Design parameters for LQR controller."""
    def __init__(self, Q: Optional[np.ndarray] = None, R: Optional[float] = None, saturation_limit: Optional[float] = None) -> None:
        # State cost matrix Q: penalize [θ-π, θ̇]
        self.Q: np.ndarray = Q if Q is not None else np.diag([10.0, 1.0])  # Heavy penalty on angle
        self.R: float = R if R is not None else 1.0  # Control cost
        self.saturation_limit: Optional[float] = saturation_limit  # Maximum control torque (N·m)


class VisualizationParams:
    """Visualization parameters for Meshcat display."""
    def __init__(self) -> None:
        self.pivot_radius: float = 0.05      # Radius of pivot sphere (m)
        self.rod_radius: float = 0.02        # Radius of pendulum rod (m)
        self.bob_radius: float = 0.08        # Radius of bob sphere (m)
        self.target_radius: float = 0.06     # Radius of target indicator (m)
        self.update_rate: float = 30.0       # Visualization update rate (Hz)


class LQRParams:
    """LQR controller computed parameters and design matrices."""
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: float, K: np.ndarray, S: np.ndarray, eigenvalues: np.ndarray) -> None:
        self.A: np.ndarray = A                    # State matrix (2x2)
        self.B: np.ndarray = B                    # Input matrix (2x1)
        self.Q: np.ndarray = Q                    # State cost matrix (2x2)
        self.R: float = R                    # Control cost (scalar or 1x1)
        self.K: np.ndarray = K                    # LQR gain matrix (1x2)
        self.S: np.ndarray = S                    # Solution to Riccati equation (2x2)
        self.eigenvalues: np.ndarray = eigenvalues  # Closed-loop eigenvalues


# =============================================================================
# MESHCAT VISUALIZER FOR SIMPLE PENDULUM
# =============================================================================
class PendulumVisualizer(LeafSystem):
    """
    Meshcat-based 3D visualizer for the simple pendulum.
    
    Creates an interactive 3D visualization showing:
      - Pivot point (fixed)
      - Pendulum rod (thin cylinder)
      - Mass (sphere at end)
      - Reference frame axes
    
    The visualization updates in real-time during simulation.
    
    Args:
        meshcat: Meshcat instance for visualization
        plant: SimplePendulumSystem instance (for accessing parameters)
        show_target: If True, show target upright position
    """
    
    def __init__(self, meshcat, plant: 'SimplePendulumSystem', show_target: bool = False, viz_params: Optional[VisualizationParams] = None) -> None:
        LeafSystem.__init__(self)
        
        self.meshcat = meshcat
        self.plant: 'SimplePendulumSystem' = plant
        self.params: PendulumParams = plant.params
        self.show_target: bool = show_target
        self.L: float = plant.params.L
        
        # Use provided visualization params or create defaults
        if viz_params is None:
            viz_params = VisualizationParams()
        self.viz_params: VisualizationParams = viz_params
        
        # Input: pendulum state [θ, θ̇]
        self.DeclareVectorInputPort("state", 2)
        
        # Input: pendulum bob position [x, z]
        self.DeclareVectorInputPort("bob_position", 2)
        
        # Input: applied torque (for debugging)
        self.DeclareVectorInputPort("applied_torque", 1)
        
        # Declare periodic update for visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=1.0/self.viz_params.update_rate,
            offset_sec=0.0,
            publish=self.UpdateVisualization
        )
        
        # Setup meshcat scene with proper hierarchy
        from pydrake.all import Rgba, RigidTransform
        
        # Pivot point (small red sphere at origin)
        self.meshcat.SetObject(
            "pendulum/pivot", 
            Sphere(self.viz_params.pivot_radius), 
            Rgba(0.8, 0.1, 0.1, 1.0)
        )
        
        # Pendulum rod (thin gray cylinder)
        self.meshcat.SetObject(
            "pendulum/rod",
            Cylinder(self.viz_params.rod_radius, self.L),
            Rgba(0.5, 0.5, 0.5, 1.0)
        )
        
        # Pendulum mass (blue sphere at end)
        self.meshcat.SetObject(
            "pendulum/bob",
            Sphere(self.viz_params.bob_radius),
            Rgba(0.2, 0.5, 0.9, 1.0)
        )
        
        # Target position indicator (if enabled)
        if self.show_target:
            self.meshcat.SetObject(
                "pendulum/target",
                Sphere(self.viz_params.target_radius),
                Rgba(0.2, 0.9, 0.2, 0.4)  # Semi-transparent green
            )
            # Position at upright (θ=π): [0, 0, L]
            target_pose = RigidTransform([0.0, 0.0, self.L])
            self.meshcat.SetTransform("pendulum/target", target_pose)
        
        # Initialize poses with identity transforms
        identity_pose = RigidTransform()
        self.meshcat.SetTransform("pendulum/pivot", identity_pose)
        self.meshcat.SetTransform("pendulum/rod", identity_pose)
        self.meshcat.SetTransform("pendulum/bob", identity_pose)
    
    def UpdateVisualization(self, context):
        """Update visualization based on current state."""
        # ---- Read state ----
        state = self.GetInputPort("state").Eval(context)
        theta, theta_dot = state[0], state[1]
        
        # ---- Bob position ----
        bob_x, bob_z = self.GetInputPort("bob_position").Eval(context)
        
        # Consistent naming for transforms
        T_W_pivot = self.plant.CalcPivotTransformFromState()
        T_W_rod = self.plant.CalcRodTransformFromState(theta)
        T_W_bob = self.plant.CalcBobTransformFromState(theta)
        
        # ---- Read controller torque for debugging ----
        applied_torque = self.GetInputPort("applied_torque").Eval(context)[0]
        
        # ---- Concise debug print ----
        time = context.get_time()
        print(
            f"t={time:5.2f}s θ={np.rad2deg(theta):6.1f}° θ̇={np.rad2deg(theta_dot):6.1f}°/s "
            f"bob=({bob_x:5.2f},{bob_z:5.2f}) τ={applied_torque:6.2f}Nm"
        )
        
        # ---- Send transforms to Meshcat ----
        self.meshcat.SetTransform("pendulum/pivot", T_W_pivot)
        self.meshcat.SetTransform("pendulum/rod", T_W_rod)
        self.meshcat.SetTransform("pendulum/bob", T_W_bob)
    
    @staticmethod
    def print_state_header():
        """Print table header for state data."""
        print("\n" + "="*90)
        print(f"{'Time (s)':>10} | {'θ (rad)':>12} | {'θ (deg)':>12} | {'θ̇ (rad/s)':>15} | {'τ (Nm)':>10} | {'Energy (J)':>12}")
        print("="*90)
    
    @staticmethod
    def print_state_row(time, state, torque, energy):
        """Print a single row of state data in tabular format."""
        theta, theta_dot = state
        theta_deg = np.rad2deg(theta)
        print(f"{time:10.3f} | {theta:12.4f} | {theta_deg:12.2f} | {theta_dot:15.4f} | {torque:10.4f} | {energy:12.4f}")
    
    @staticmethod
    def print_state_summary(t, z, torques, energies, title="Simulation Summary"):
        """Print summary statistics of the simulation."""
        print("\n" + "="*90)
        print(f"{title}")
        print("="*90)
        
        # Wrap angles to [-π, π] for statistics
        theta_wrapped = np.array([wrap_to(theta, 0, 2.0 * np.pi) - np.pi for theta in z[:, 0]])
        
        print(f"  Duration: {t[-1]:.2f} seconds")
        print(f"  Initial state: θ = {theta_wrapped[0]:.4f} rad ({np.rad2deg(theta_wrapped[0]):.2f}°), θ̇ = {z[0,1]:.4f} rad/s")
        print(f"  Final state:   θ = {theta_wrapped[-1]:.4f} rad ({np.rad2deg(theta_wrapped[-1]):.2f}°), θ̇ = {z[-1,1]:.4f} rad/s")
        print(f"  Max |θ| error from upright: {np.max(np.abs(theta_wrapped)):.4f} rad ({np.rad2deg(np.max(np.abs(theta_wrapped))):.2f}°)")
        print(f"  Max |θ̇|: {np.max(np.abs(z[:,1])):.4f} rad/s")
        print(f"  RMS θ error: {np.sqrt(np.mean(theta_wrapped**2)):.4f} rad")
        if torques is not None:
            print(f"  Max control torque: {np.max(np.abs(torques)):.4f} N·m")
            print(f"  RMS control torque: {np.sqrt(np.mean(torques**2)):.4f} N·m")
        if energies is not None:
            print(f"  Initial energy: {energies[0]:.4f} J")
            print(f"  Final energy: {energies[-1]:.4f} J")
            print(f"  Energy change: {energies[-1] - energies[0]:.4f} J")
        print("="*90)


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
    
    def __init__(self, num_states: int, num_inputs: int = 0, name: str = "StabilitySystem") -> None:
        """
        Initialize the base system.
        
        Args:
            num_states (int): Dimension of the continuous state vector
            num_inputs (int): Dimension of the input vector
            name (str): System name for debugging/logging
        """
        LeafSystem.__init__(self)
        self.set_name(name)
        self._num_states: int = num_states
        self._num_inputs: int = num_inputs
        
        # Declare continuous state vector
        self.DeclareContinuousState(num_states)
        
        # Declare input port if needed
        if num_inputs > 0:
            self.DeclareVectorInputPort("control", num_inputs)
        
        # Declare output port that exposes the full state vector
        # Set prerequisites_of_calc to indicate no direct feedthrough from input to state output
        self.DeclareVectorOutputPort(
            "state", 
            BasicVector(num_states), 
            self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()}  # Only depends on continuous state, not inputs
        )
        
        # Bob position output (for visualization) - only for pendulum systems
        if name == "SimplePendulum":
            self.DeclareVectorOutputPort(
                "bob_position", 
                BasicVector(2), 
                self.CalcBobPosition,
                prerequisites_of_calc={self.xc_ticket()}  # Only depends on continuous state
            )
    
    def CopyStateOut(self, context, output):
        """
        Output port callback: Copy the continuous state to the output port.
        
        Args:
            context: Current system context (contains state, time, etc.)
            output: Output vector to be filled
        """
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)
    
    def simulate(self, initial_state: np.ndarray, t0: float = 0.0, tf: float = 10.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
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
# SIMPLE PENDULUM SYSTEM - BUILT FROM SCRATCH
# =============================================================================
class SimplePendulumSystem(StabilityDemo):
    """
    Simple Pendulum System with Dynamics Implemented from First Principles.
    
    PHYSICAL SYSTEM:
    ---------------
    A point mass m is attached to a massless rigid rod of length L.
    The rod pivots freely about a fixed point (frictionless pivot).
    
    STATE VECTOR: x = [θ, θ̇]
      - θ: Pendulum angle from downward vertical (rad)
      - θ̇: Angular velocity (rad/s)
    
    CONTROL INPUT: u = τ
      - τ: Torque applied at the pivot (N·m)
    
    DYNAMICS (from Euler-Lagrange):
    -------------------------------
    The equation of motion is:
    
    mL²θ̈ + bθ̇ + mgLsin(θ) = τ
    
    Solving for acceleration:
      θ̈ = [τ - bθ̇ - mgLsin(θ)] / (mL²)
    
    Or in normalized form (divide by mL²):
      θ̈ = -(g/L)sin(θ) - (b/mL²)θ̇ + τ/(mL²)
    
    EQUILIBRIA:
    ----------
    1. Downward (stable): θ = 0, θ̇ = 0
       - Natural resting position
       - Small perturbations lead to oscillation around this point
    
    2. Upward (unstable): θ = π, θ̇ = 0
       - Inverted pendulum position
       - Any small perturbation causes it to fall
       - Requires active control to maintain
    
    LINEARIZATION at upright position θ = π:
    ----------------------------------------
    Let θ̃ = θ - π (deviation from upright)
    For small θ̃: sin(θ̃) ≈ θ̃, cos(θ̃) ≈ 1
    
    Linearized dynamics: θ̈ ≈ (g/L)θ̃ - (b/mL²)θ̇ + τ/(mL²)
    
    State space: ẋ = Ax + Bu where x = [θ̃, θ̇]
      A = [    0        1    ]
          [ g/L   -b/(mL²) ]
      
      B = [    0      ]
          [ 1/(mL²)  ]
    
    ENERGY (matching PDF notation):
    ------------------------------
    Kinetic energy: T = ½mL²θ̇²
    Potential energy: V = -mgLcos(θ) (measured from horizontal)
    Total mechanical energy:
      E = T + V = ½mL²θ̇² - mgLcos(θ)
    
    At downward position (θ=0): E = -mgL
    At upward position (θ=π): E = mgL
    """
    
    def __init__(self, params: Optional[PendulumParams] = None, control_gain: Optional[np.ndarray] = None) -> None:
        """
        Initialize the simple pendulum system.
        
        Args:
            params: PendulumParams instance (uses defaults if None)
            control_gain: LQR gain matrix K (1x2) for control law u = -K*x
        """
        # Initialize with 2 states and 1 input
        super().__init__(num_states=2, num_inputs=1, name="SimplePendulum")
        
        # Store physical parameters
        self.params: PendulumParams = params if params is not None else PendulumParams()
        
        # Store LQR gain (if provided)
        self.K: Optional[np.ndarray] = control_gain
        
        print(f"\nSimple Pendulum System Initialized:")
        print(f"  Mass m = {self.params.m} kg")
        print(f"  Length L = {self.params.L} m")
        print(f"  Gravity g = {self.params.g} m/s²")
        print(f"  Damping b = {self.params.b} N·m·s")
        if self.K is not None:
            print(f"  LQR Control: ENABLED")
            print(f"  Gain matrix K = {self.K.flatten()}")
        else:
            print(f"  LQR Control: DISABLED")
    
    def CalcBobPosition(self, context, output):
        """Calculate pendulum bob position [x, z] in world frame."""
        state = context.get_continuous_state_vector().CopyToVector()
        theta = state[0]  # Angle from downward vertical
        
        # Bob position: x = L*sin(θ), z = -L*cos(θ)
        x = self.params.L * np.sin(theta)
        z = -self.params.L * np.cos(theta)
        output.SetFromVector([x, z])
    
    # ------------------------------------------------------------------
    # KINEMATICS HELPER METHODS (for visualization)
    # ------------------------------------------------------------------
    def CalcBobPositionFromTheta(self, theta):
        """Calculate bob position from angle."""
        bx = self.params.L * np.sin(theta)
        bz = -self.params.L * np.cos(theta)
        return bx, bz
    
    def CalcRodTransformFromState(self, theta):
        """Calculate rod transform from pendulum angle."""
        from pydrake.math import RotationMatrix
        
        bx, bz = self.CalcBobPositionFromTheta(theta)
        # Rod center is at midpoint between pivot (0,0,0) and bob
        p = np.array([bx/2, 0, bz/2])
        
        # Direction of rod (pivot -> bob)
        rod_dir = np.array([bx, 0, bz])
        rod_dir = rod_dir / np.linalg.norm(rod_dir)
        
        # Meshcat cylinder aligned along +Z axis
        R = RotationMatrix.MakeFromOneVector(rod_dir, axis_index=2)
        return RigidTransform(R, p)
    
    def CalcBobTransformFromState(self, theta):
        """Calculate bob transform from pendulum angle."""
        bx, bz = self.CalcBobPositionFromTheta(theta)
        return RigidTransform([bx, 0, bz])
    
    def CalcPivotTransformFromState(self):
        """Calculate pivot transform (always at origin)."""
        return RigidTransform([0.0, 0.0, 0.0])
    
    def _create_fresh_instance(self):
        """Create a new SimplePendulumSystem instance."""
        return SimplePendulumSystem(params=self.params, control_gain=self.K)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute time derivatives for the simple pendulum system.
        
        Implements the full nonlinear dynamics:
          θ̇ = ω (angular velocity)
          ω̇ = -(g/L)sin(θ) - (b/mL²)ω + τ/(mL²)
        
        Args:
            context: Current system context
            derivatives: Object to store computed derivatives
        """
        # Get current state: [θ, θ̇]
        state = context.get_continuous_state_vector().CopyToVector()
        theta = state[0]      # Pendulum angle
        theta_dot = state[1]  # Angular velocity
        
        # Get control input (torque)
        if self._num_inputs > 0:
            u = self.get_input_port(0).Eval(context)
            tau = u[0]
        else:
            tau = 0.0
        
        # Extract parameters
        m = self.params.m
        L = self.params.L
        g = self.params.g
        b = self.params.b
        
        # Compute angular acceleration using equation of motion:
        # θ̈ = [τ - bθ̇ - mgLsin(θ)] / (mL²)
        inertia = m * L**2
        theta_ddot = (tau - b*theta_dot - m*g*L*np.sin(theta)) / inertia
        
        # Set derivatives: [θ̇, θ̈]
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def energy(self, state: np.ndarray) -> float:
        """
        Matching PDF notation:
          T (Kinetic Energy) = ½mL²θ̇²
          V (Potential Energy) = -mgLcos(θ)
          E = T + V = ½mL²θ̇² - mgLcos(θ)
        
        Args:
            state: [θ, θ̇]
        
        Returns:
            Total energy (Joules)
        """
        theta = state[0]
        theta_dot = state[1]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        I = m * L**2
        
        T = 0.5 * I * theta_dot**2              # Kinetic energy
        V = -m * g * L * np.cos(theta)          # Potential energy (PDF notation)
        
        return T + V * (1 - np.cos(theta))
        
        return KE + PE


# =============================================================================
# LQR CONTROLLER SYSTEM
# =============================================================================
class LQRController(LeafSystem):
    """
    LQR Controller for Simple Pendulum System.
    
    CONTROL LAW:
    -----------
    u = -K*x_error
    
    where:
      x_error = [θ - π, θ̇]
      K = LQR gain matrix (1x2)
      u = control torque
    
    The controller computes the optimal control torque to drive the pendulum
    to the upright equilibrium while minimizing a quadratic cost function.
    """
    
    def __init__(self, pendulum: SimplePendulumSystem, pendulum_params: PendulumParams, lqr_design_params: Optional[LQRDesignParams] = None) -> None:
        """
        Initialize the LQR controller.
        
        Args:
            pendulum: SimplePendulumSystem instance
            pendulum_params: PendulumParams instance for linearization
            lqr_design_params: LQRDesignParams instance (uses defaults if None)
        """
        LeafSystem.__init__(self)
        
        # Input: pendulum state [θ, θ̇]
        self.DeclareVectorInputPort("state", 2)
        
        # Output: control torque τ
        self.DeclareVectorOutputPort("control", 1, self.DoCalcOutput)

        self.params: PendulumParams = pendulum_params
        
        # Use provided design params or create defaults
        if lqr_design_params is None:
            lqr_design_params = LQRDesignParams()
        self.design_params: LQRDesignParams = lqr_design_params
        self.saturation_limit: Optional[float] = lqr_design_params.saturation_limit

        # Get Q and R from design params
        Q = self.design_params.Q
        R = self.design_params.R
        
        print("\nDesigning LQR controller...")
        print(f"  Q = diag({np.diag(Q)})")
        print(f"  R = {R}")
        
        K, S = self.design_lqr_controller(self.params, Q, R)
        
        print(f"\nLQR Solution:")
        print(f"  Gain matrix K = {K.flatten()}")
        
        # Verify closed-loop stability
        A, B = LQRController.get_linearization_at_upright(self.params)

        eigenvalues = np.linalg.eigvals(A - B @ K)

        # Create LQR parameters object
        self.params_lqr = LQRParams(A, B, Q, R, K, S, eigenvalues)
        
        # Store controller parameters (after computing K)
        self.K = K
        
        print(f"\nLQR Controller Initialized:")
        print(f"  Gain matrix K = {self.K.flatten()}")
        print(f"  Saturation limit = {self.saturation_limit} N·m" if self.saturation_limit else "  No saturation")
    
    @staticmethod
    def get_linearization_at_upright(params: PendulumParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytical linearization at the upright equilibrium.
        
        At θ = π (upright), the linearized dynamics are:
        
        ẋ = Ax + Bu
        
        where x = [θ-π, θ̇] and u = τ
        
        Args:
            params: PendulumParams instance
        
        Returns:
            A (2x2 matrix): State matrix
            B (2x1 matrix): Input matrix
        """
        m = params.m
        L = params.L
        g = params.g
        b = params.b
        
        # Moment of inertia
        I = m * L**2
        
        # Linearized state matrix A
        # At θ=π: sin(θ) ≈ -(θ-π), cos(θ) ≈ -1
        # So: d/dθ[mgLsin(θ)]|_{θ=π} = -mgL
        A = np.array([
            [0, 1],              # θ̇ = θ̇
            [g/L, -b/I]          # θ̈ = (g/L)θ̃ - (b/I)θ̇ + τ/I
        ])
        
        # Linearized input matrix B
        B = np.array([
            [0],
            [1/I]
        ])
        
        return A, B
    

    def design_lqr_controller(self, params: PendulumParams, Q: np.ndarray, R: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design LQR controller for the upright equilibrium.
        
        Solves the continuous-time algebraic Riccati equation (CARE):
          A'S + SA - SBR⁻¹B'S + Q = 0
        
        The optimal gain is: K = R⁻¹B'S
        
        Args:
            params: PendulumParams instance
            Q (2x2 matrix): State cost matrix
            R (scalar or 1x1 matrix): Control cost
        
        Returns:
            K (1x2 matrix): LQR gain matrix
            S (2x2 matrix): Solution to Riccati equation
        """
        from scipy.linalg import solve_continuous_are
        
        # Get linearized dynamics at upright position
        A, B = LQRController.get_linearization_at_upright(params)
        
        # Ensure R is properly shaped
        if np.isscalar(R):
            R = np.array([[R]])
        
        # Solve continuous-time algebraic Riccati equation
        S = solve_continuous_are(A, B, Q, R)
        
        # Compute optimal gain: K = R⁻¹B'S
        K = np.linalg.solve(R, B.T @ S)
        
        return K, S
    
    def DoCalcOutput(self, context, output):
        """
        Compute control torque using LQR feedback law.
        
        Args:
            context: Current system context
            output: Output port to write control torque
        """
        # Read current state [θ, θ̇]
        state = self.get_input_port(0).Eval(context)
        
        # Compute error from target equilibrium (upright position)
        # Wrap angle to [-π, π] to handle periodicity
        theta_wrapped = wrap_to(state[0], 0, 2.0 * np.pi) - np.pi
        
        x_error = np.array([
            theta_wrapped,  # Angle error from upright
            state[1]        # Angular velocity (target is 0)
        ])
        
        # LQR control law: u = -K*x_error
        control_torque = -self.K @ x_error
        
        # Apply saturation if specified
        if self.saturation_limit is not None:
            control_torque = np.clip(control_torque, -self.saturation_limit, self.saturation_limit)
        
        output.SetFromVector(control_torque.flatten())


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================
def no_control_demo():
    """
    Demonstrate pendulum behavior WITHOUT control.
    
    Shows that the upright equilibrium is unstable - the pendulum falls
    when perturbed from the upright position.
    """
    print("\n" + "="*90)
    print("SIMPLE PENDULUM WITHOUT CONTROL - Natural Dynamics")
    print("="*90)
    print("The pendulum will fall from near-upright position (unstable equilibrium)")
    
    # Create pendulum system without controller
    pendulum = SimplePendulumSystem()
    
    # Build diagram
    builder = DiagramBuilder()
    system = builder.AddSystem(pendulum)
    
    # Add Meshcat visualizer
    meshcat = None
    if MESHCAT_AVAILABLE:
        try:
            meshcat = StartMeshcat()
            visualizer = builder.AddSystem(
                PendulumVisualizer(meshcat, pendulum, show_target=False)
            )
            # Connect state output to visualizer state input
            builder.Connect(
                system.GetOutputPort("state"),
                visualizer.GetInputPort("state")
            )
            # Connect bob position output to visualizer bob_position input
            builder.Connect(
                system.GetOutputPort("bob_position"),
                visualizer.GetInputPort("bob_position")
            )
            print(f"\n✓ Meshcat visualization started")
            print(f"  Open browser to: {meshcat.web_url()}")
        except Exception as e:
            print(f"\n⚠ Could not start Meshcat visualization: {e}")
            meshcat = None
    
    # Connect zero torque to system and visualizer
    from pydrake.systems.primitives import ConstantVectorSource
    zero_torque = builder.AddSystem(ConstantVectorSource([0.0]))
    builder.Connect(
        zero_torque.get_output_port(0),
        system.get_input_port(0)
    )
    
    # Connect torque to visualizer if available
    if meshcat is not None:
        builder.Connect(
            zero_torque.get_output_port(0),
            visualizer.GetInputPort("applied_torque")
        )
    
    # Add logger
    logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(system.get_output_port(0), logger.get_input_port(0))
    
    # Build and simulate
    diagram = builder.Build()
    
    # Save block diagram as PNG
    if PYDOT_AVAILABLE:
        try:
            import pydot
            graphviz_str = diagram.GetGraphvizString()
            (graph,) = pydot.graph_from_dot_data(graphviz_str)
            graph.write_png('pendulum_no_control_block_diagram.png')
            print(f"\n✓ Block diagram saved to: pendulum_no_control_block_diagram.png")
        except Exception as e:
            print(f"\n⚠ Could not save block diagram: {e}")
    
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial condition: small perturbation from upright
    initial_state = np.array([np.pi - 0.1, 0.0])  # [θ, θ̇]
    print(f"\nInitial state: θ={initial_state[0]:.4f}rad ({np.rad2deg(initial_state[0]):.2f}°), θ̇={initial_state[1]:.4f}rad/s")
    
    # Get the system context from the diagram and set initial state
    system_context = system.GetMyContextFromRoot(context)
    system_context.SetContinuousState(initial_state)
    
    # Set real-time rate for visualization (0.5 = half speed for better viewing)
    if meshcat is not None:
        simulator.set_target_realtime_rate(0.5)
        print("\n⏱ Simulation running at 0.5x real-time for better visualization")
    
    simulator.Initialize()
    
    # Print state header
    PendulumVisualizer.print_state_header()
    
    # Simulate
    sim_time = 10.0
    dt = 0.1
    torques = []
    energies = []
    
    for i in range(int(sim_time/dt) + 1):
        t = i * dt
        if t > 0:
            simulator.AdvanceTo(t)
        
        # Get state
        sys_context = system.GetMyContextFromRoot(context)
        state = system.get_output_port(0).Eval(sys_context)
        torque = 0.0
        energy = pendulum.energy(state)
        
        torques.append(torque)
        energies.append(energy)
        
        PendulumVisualizer.print_state_row(t, state, torque, energy)
    
    # Get logged data
    log = logger.FindLog(context)
    t_log = log.sample_times()
    z_log = log.data().T
    
    # Compute energies for all logged data
    energies_log = np.array([pendulum.energy(z_log[i, :]) for i in range(len(t_log))])
    torques_log = np.zeros(len(t_log))
    
    # Print summary
    PendulumVisualizer.print_state_summary(t_log, z_log, torques_log, energies_log, "NO CONTROL - Summary")
    
    # Generate phase portrait
    plot_phase_portrait_with_trajectory(
        t_log, z_log, pendulum.params,
        title="Simple Pendulum - No Control - Phase Portrait",
        filename="pendulum_no_control_phase_portrait.png"
    )
    
    return t_log, z_log, torques_log, energies_log


def lqr_control_demo(meshcat_instance=None):
    """
    Demonstrate pendulum with LQR control.
    
    Designs an LQR controller and shows that it can stabilize the
    inverted pendulum at the upright position.
    
    Args:
        meshcat_instance: Optional shared Meshcat instance for reuse
    """
    print("\n" + "="*90)
    print("SIMPLE PENDULUM WITH LQR CONTROL")
    print("="*90)
    
    # Create pendulum system
    pendulum = SimplePendulumSystem()
    
    # Create LQR design parameters
    lqr_design = LQRDesignParams(
        Q=np.diag([10.0, 1.0]),  # State cost matrix
        R=1.0,                    # Control cost
        saturation_limit=3.0      # Maximum torque
    )
    
    # Create controller
    controller = LQRController(pendulum, pendulum.params, lqr_design)


    print(f"\nClosed-loop eigenvalues: {controller.params_lqr.eigenvalues}")
    print(f"  Max real part: {np.max(controller.params_lqr.eigenvalues.real):.4f}")
    if np.max(controller.params_lqr.eigenvalues.real) < 0:
        print("  ✓ System is asymptotically stable!")
    else:
        print("  ✗ Warning: System may be unstable!")
    
    
    
    # Build diagram
    builder = DiagramBuilder()
    system = builder.AddSystem(pendulum)
    ctrl = builder.AddSystem(controller)
    
    # Connect controller to system
    builder.Connect(system.get_output_port(0), ctrl.get_input_port(0))
    builder.Connect(ctrl.get_output_port(0), system.get_input_port(0))
    
    # Add Meshcat visualizer
    meshcat = meshcat_instance
    if MESHCAT_AVAILABLE:
        try:
            if meshcat is None:
                meshcat = StartMeshcat()
            else:
                # Clear previous visualization
                meshcat.Delete("pendulum")
            
            visualizer = builder.AddSystem(
                PendulumVisualizer(meshcat, pendulum, show_target=True)
            )
            # Connect state output to visualizer state input
            builder.Connect(
                system.GetOutputPort("state"),
                visualizer.GetInputPort("state")
            )
            # Connect bob position output to visualizer bob_position input
            builder.Connect(
                system.GetOutputPort("bob_position"),
                visualizer.GetInputPort("bob_position")
            )
            # Connect torque output to visualizer applied_torque input
            builder.Connect(
                ctrl.get_output_port(0),
                visualizer.GetInputPort("applied_torque")
            )
            print(f"\n✓ Meshcat visualization started")
            print(f"  Open browser to: {meshcat.web_url()}")
            print(f"  Green sphere shows target upright position")
        except Exception as e:
            print(f"\n⚠ Could not start Meshcat visualization: {e}")
            meshcat = None
    
    # Add logger
    logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(system.get_output_port(0), logger.get_input_port(0))
    
    # Build and simulate
    diagram = builder.Build()
    
    # Save block diagram as PNG
    if PYDOT_AVAILABLE:
        try:
            import pydot
            graphviz_str = diagram.GetGraphvizString()
            (graph,) = pydot.graph_from_dot_data(graphviz_str)
            graph.write_png('pendulum_lqr_control_block_diagram.png')
            print(f"\n✓ Block diagram saved to: pendulum_lqr_control_block_diagram.png")
        except Exception as e:
            print(f"\n⚠ Could not save block diagram: {e}")
    
    # Create Drake simulator for the diagram
    simulator = Simulator(diagram)
    # Get mutable context (read/write access) for setting initial conditions, etc.
    context = simulator.get_mutable_context()
    
    # Set initial condition: perturbation from upright
    # Randomize initial perturbation from upright (θ = π)
    perturbation = np.random.uniform(-0.3, 0.3)
    initial_state = np.array([np.pi + perturbation, 0.0])  # [θ, θ̇]
    print(f"\nInitial state: θ={initial_state[0]:.4f}rad ({np.rad2deg(initial_state[0]):.2f}°), θ̇={initial_state[1]:.4f}rad/s")
    
    context.SetContinuousState(initial_state)
    
    # Set real-time rate for visualization (0.5 = half speed for better viewing)
    if meshcat is not None:
        simulator.set_target_realtime_rate(0.5)
        print("\n⏱ Simulation running at 0.5x real-time for better visualization")
    
    simulator.Initialize()
    
    # Print state header
    PendulumVisualizer.print_state_header()
    
    # Simulate
    sim_time = 4.0
    dt = 0.5
    torques = []
    energies = []
    
    for i in range(int(sim_time/dt) + 1):
        t = i * dt
        if t > 0:
            simulator.AdvanceTo(t)
        
        # Get state and control
        sys_context = system.GetMyContextFromRoot(context)
        ctrl_context = ctrl.GetMyContextFromRoot(context)
        
        state = system.get_output_port(0).Eval(sys_context)
        torque = ctrl.get_output_port(0).Eval(ctrl_context)[0]
        energy = pendulum.energy(state)
        
        torques.append(torque)
        energies.append(energy)
        
        PendulumVisualizer.print_state_row(t, state, torque, energy)
    
    # Get logged data
    log = logger.FindLog(context)
    t_log = log.sample_times()
    z_log = log.data().T
    
    # Compute torques and energies for all logged data
    torques_log = np.zeros(len(t_log))
    energies_log = np.zeros(len(t_log))
    
    for i in range(len(t_log)):
        state = z_log[i, :]
        theta_wrapped = wrap_to(state[0], 0, 2.0 * np.pi) - np.pi
        x_error = np.array([theta_wrapped, state[1]])
        torque = -controller.K @ x_error
        torques_log[i] = np.clip(torque[0], -3.0, 3.0)
        energies_log[i] = pendulum.energy(state)
    
    # Print summary
    PendulumVisualizer.print_state_summary(t_log, z_log, torques_log, energies_log, "LQR CONTROL - Summary")
    
    # Generate phase portrait
    plot_phase_portrait_with_trajectory(
        t_log, z_log, pendulum.params,
        title="Simple Pendulum - LQR Control - Phase Portrait",
        filename="pendulum_lqr_control_phase_portrait.png"
    )
    
    return t_log, z_log, torques_log, energies_log


# =============================================================================
# PHASE PORTRAIT PLOTTING
# =============================================================================
def plot_phase_portrait_with_trajectory(t_log, z_log, params, title="Phase Portrait", 
                                       filename="phase_portrait.png"):
    """
    Create phase portrait showing:
    1. Vector field of pendulum dynamics
    2. Multiple sample trajectories
    3. Actual simulated trajectory
    4. Equilibrium points
    
    Args:
        t_log: Time array from simulation
        z_log: State trajectory [theta, theta_dot] from simulation
        params: PendulumParams object
        title: Plot title
        filename: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # =========================================================================
    # TOP PLOT: POTENTIAL ENERGY WELL
    # =========================================================================
    theta_range = np.linspace(-np.pi, np.pi, 500)
    
    # Potential energy: PE = -m*g*L*cos(theta)
    # Normalized: PE/(m*g*L) = -cos(theta)
    PE_normalized = -np.cos(theta_range)
    
    ax1.plot(theta_range, PE_normalized, 'b-', linewidth=2, label='Potential Energy')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.axvline(x=-np.pi, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Unstable Eq.')
    ax1.axvline(x=0, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Stable Eq.')
    ax1.axvline(x=np.pi, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_ylabel(r'Potential Energy $V(\theta)/(mgL)$', fontsize=12)
    ax1.set_title('Potential Energy Well', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    # =========================================================================
    # BOTTOM PLOT: PHASE PORTRAIT
    # =========================================================================
    
    # Create grid for vector field
    theta_vec = np.linspace(-np.pi, np.pi, 20)
    theta_dot_vec = np.linspace(-5, 5, 20)
    Theta, Theta_dot = np.meshgrid(theta_vec, theta_dot_vec)
    
    # Compute derivatives at each grid point (no control, u=0)
    m, L, g, b = params.m, params.L, params.g, params.b
    Theta_ddot = (-g/L * np.sin(Theta) - b/(m*L**2) * Theta_dot)
    
    # Vector field (quiver plot)
    ax2.quiver(Theta, Theta_dot, Theta_dot, Theta_ddot, 
              alpha=0.4, color='gray', scale=50, width=0.003)
    
    # Mark equilibrium points
    # Stable: theta = 0 (downward)
    ax2.plot(0, 0, 'go', markersize=12, label='Stable Equilibrium', zorder=5)
    # Unstable: theta = ±π (upright)
    ax2.plot(-np.pi, 0, 'rx', markersize=12, markeredgewidth=3, 
            label='Unstable Equilibrium', zorder=5)
    ax2.plot(np.pi, 0, 'rx', markersize=12, markeredgewidth=3, zorder=5)
    
    # Plot sample trajectories from different initial conditions
    sample_ics = [
        [0.5, 0.0],   # Small angle near stable
        [1.0, 0.0],   # Moderate angle
        [2.0, 0.0],   # Larger angle
        [np.pi - 0.3, 0.0],  # Near upright
        [0.0, 2.0],   # Initial velocity
        [0.0, -2.0],  # Initial velocity opposite
    ]
    
    for ic in sample_ics:
        # Simple forward Euler integration for trajectory
        dt = 0.01
        t_max = 3.0
        n_steps = int(t_max / dt)
        traj = np.zeros((n_steps, 2))
        traj[0] = ic
        
        for i in range(1, n_steps):
            theta, theta_dot = traj[i-1]
            theta_ddot = -g/L * np.sin(theta) - b/(m*L**2) * theta_dot
            traj[i, 0] = theta + theta_dot * dt
            traj[i, 1] = theta_dot + theta_ddot * dt
            
            # Wrap theta to [-π, π]
            traj[i, 0] = wrap_to(traj[i, 0], -np.pi, np.pi)
        
        ax2.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    # Plot actual simulated trajectory (HIGHLIGHTED)
    theta_wrapped = np.array([wrap_to(z_log[i, 0], -np.pi, np.pi) 
                             for i in range(len(z_log))])
    ax2.plot(theta_wrapped, z_log[:, 1], 'r-', linewidth=2.5, 
            label='Simulated Trajectory', zorder=4)
    
    # Mark start and end points
    ax2.plot(theta_wrapped[0], z_log[0, 1], 'mo', markersize=10, 
            label='Start', zorder=6)
    ax2.plot(theta_wrapped[-1], z_log[-1, 1], 'cs', markersize=10, 
            label='End', zorder=6)
    
    # Labels and formatting
    ax2.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax2.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax2.set_title('Phase Portrait', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_ylim([-5, 5])
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Phase portrait saved to: {filename}")
    
    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """
    Main function to run pendulum demonstrations.
    """
    print("\n" + "="*90)
    print("SIMPLE PENDULUM LQR CONTROL DEMONSTRATION - BUILT FROM SCRATCH")
    print("="*90)
    print("\nTUTORIAL OBJECTIVES:")
    print("  • Implement simple pendulum dynamics from first principles")
    print("  • Design LQR controller for inverted pendulum stabilization")
    print("  • Compare controlled vs. uncontrolled behavior")
    print("  • Understand complete control pipeline")
    print("\nSYSTEM DETAILS:")
    print("  • Simple pendulum (1-DOF)")
    print("  • State: [θ, θ̇]")
    print("  • Control: Torque at pivot")
    print("  • Upright equilibrium: θ=π (unstable)")
    print("  • Downward equilibrium: θ=0 (stable)")
    print("="*90)
    
    print("\nDEMONSTRATION OPTIONS:")
    print("  1) NO CONTROL - Observe natural dynamics (pendulum falls)")
    print("  2) LQR CONTROL - Stabilize inverted pendulum")
    print("  3) LQR CONTROL - Run 5 times in a loop")
    print("="*90)
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == '1':
        no_control_demo()
    elif choice == '2':
        lqr_control_demo()
    else:
        # Run LQR demo 5 times in a loop
        print("\n" + "="*90)
        print("RUNNING LQR CONTROL DEMO 5 TIMES")
        print("="*90)
        
        # Create shared Meshcat instance for all runs
        meshcat_shared = None
        if MESHCAT_AVAILABLE:
            try:
                meshcat_shared = StartMeshcat()
                print(f"\n✓ Meshcat visualization started at http://localhost:7001")
            except Exception as e:
                print(f"\n⚠ Could not start Meshcat: {e}")
        
        for i in range(5):
            print(f"\n{'='*90}")
            print(f"RUN {i+1} of 5")
            print(f"{'='*90}")
            lqr_control_demo(meshcat_instance=meshcat_shared)
            if i < 4:
                input("\nPress Enter to continue to next run...")
    
    print("\n" + "="*90)
    print("DEMONSTRATION COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
