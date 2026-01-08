"""
Cart-Pendulum LQR Control Simulation (Textbook Version with Drake)

Uses dynamics from pendcart.m and LQR parameters from pendcart_system_matrices.m

COORDINATE SYSTEM:
    X-axis: Horizontal (cart motion direction, positive to the right)
    Y-axis: Horizontal (into/out of page, perpendicular to cart motion)
    Z-axis: Vertical (positive upward)
    
GRAVITY:
    Acts in the negative Z direction (downward) with g = 9.81 m/s²
    
ANGLE CONVENTION:
    theta = 0:    Pendulum hanging straight down (stable equilibrium)
    theta = π:    Pendulum pointing straight up (unstable equilibrium, target)
    theta > 0:    Pendulum rotated counterclockwise when viewed from positive Y-axis

VISUAL REPRESENTATION:
              ↑ Y (up)
              |
              |  ● (pendulum bob)
              | / θ (angle from upright)
    [========] ← cart
    ────────────→ track
    ────────────→ X (horizontal)
    
    Gravity: g ↓ (downward)
    
    The torque equation τ = -m*g*l*sin(theta) creates:
    - Restoring torque when pendulum deviates from upright (theta ≈ π)
    - Destabilizing torque when hanging down (theta ≈ 0)
    
STATES:
    [x, x_dot, theta, theta_dot]
    x:         Cart position along X-axis (m)
    x_dot:     Cart velocity (m/s)
    theta:     Pendulum angle from downward vertical (rad)
    theta_dot: Pendulum angular velocity (rad/s)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from pydrake.all import (
    plot_system_graphviz,
    DiagramBuilder,
    Simulator,
    LeafSystem,
    LogVectorOutput,
    StartMeshcat,
    RigidTransform,
    RollPitchYaw,
    Cylinder,
    Sphere,
)
from pydrake.geometry import Box, Rgba


# =============================================================================
# PARAMETER CLASS - All simulation parameters in one place
# =============================================================================

class CartPendulumParameters:
    """Configuration parameters for cart-pendulum system."""
    
    # Physical Parameters
    # mass_cart = 0.5         # Cart mass (kg)
    # mass_arm = 0.2          # Pendulum mass (kg)
    # length_arm = 0.3        # Pendulum length (m)
    # cart_friction = 0.1     # Cart friction coefficient (N/m/s)
    # gravity = 9.8           # Gravity (m/s²)

    mass_cart = 5.0         # Cart mass (kg) - from textbook
    mass_arm = 1.0          # Pendulum mass (kg) - from textbook
    length_arm = 2.0        # Pendulum length (m) - from textbook
    cart_friction = 1.0     # Damping coefficient (N/(m/s)) - from textbook
    gravity = -9.8         # Gravity (m/s^2) - from textbook (g = -10, negative!)
    # moment_inertia = (1.0/3.0) * mass_arm * length_arm**2  # Moment of inertia

    # # For compatibility with rest of the code
    # I = moment_inertia
    
    # Initial Conditions [x, x_dot, theta, theta_dot] - from textbook
    x_initial = -1.0        # Initial cart position (m)
    x_dot_initial = 0.0     # Initial cart velocity (m/s)
    theta_initial = np.pi + 0.1  # Initial pendulum angle from vertical (rad)
    theta_dot_initial = 0.0 # Initial angular velocity (rad/s)
    
    # Target state (for error calculation) - from textbook
    x_target = 1.0          # Target cart position (m)
    theta_target = np.pi    # Target pendulum angle (upright = π rad)
    
    # Simulation Settings
    t_final = 10.0          # Total simulation time (s) - textbook
    dt_control = 0.01       # Controller timestep (s)
    
    # Controller Parameters
    # Energy shaping
    energy_gain = 0.08      # Energy pumping gain (reduced for smoother swing-up)
    cart_gain = 20.0        # Cart position penalty (increased for tighter centering)
    force_limit = 500.0     # Maximum control force (N) - increased for textbook LQR
    
    # LQR weight matrices - from textbook (pendcart_system_matrices.m)
    # State weights: [x, x_dot, theta, theta_dot]
    Q_lqr = np.eye(4)                       # Identity matrix (textbook)
    R_lqr = np.array([[0.0001]])            # Control effort penalty (textbook)
    
    # LQR gains (computed from linear state space, or can be manually set)
    K_lqr = None  # Manual gains
    # K_lqr = np.array([-22.36, -14.28, 50.96, 9.975]) # Set to None to compute automatically from Q and R using control.lqr()   
    
    # Switching thresholds
    e_thresh_factor = 0.10      # Energy threshold factor
    theta_thresh_deg = 18       # Angle threshold (degrees)
    theta_dot_thresh_deg = 40   # Angular velocity threshold (deg/s)
    x_thresh = 1.5              # Cart position threshold (m)
    xdot_thresh = 1.5           # Cart velocity threshold (m/s)
    switch_margin = 0.6         # Hysteresis margin
    
    @classmethod
    def get_initial_state(cls):
        """Return initial state as numpy array."""
        return np.array([
            cls.x_initial,
            cls.x_dot_initial,
            cls.theta_initial,
            cls.theta_dot_initial
        ])
    
    @classmethod
    def print_summary(cls):
        """Print parameter summary."""
        print("=" * 70)
        print("CART-PENDULUM SIMULATION PARAMETERS")
        print("=" * 70)
        print("\n📐 Physical Parameters:")
        print(f"   Cart mass:        {cls.mass_cart} kg")
        print(f"   Arm mass:         {cls.mass_arm} kg")
        print(f"   Arm length:       {cls.length_arm} m")
        print(f"   Cart friction:    {cls.cart_friction} N/m/s")
        print(f"   Gravity:          {cls.gravity} m/s²")
        
        print("\n🎯 Initial Conditions:")
        print(f"   Cart position:    {cls.x_initial} m")
        print(f"   Cart velocity:    {cls.x_dot_initial} m/s")
        print(f"   Arm angle:        {cls.theta_initial} rad ({np.rad2deg(cls.theta_initial):.1f}°)")
        print(f"   Arm ang. vel:     {cls.theta_dot_initial} rad/s")
        
        print("\n⏱️  Simulation Settings:")
        print(f"   Duration:         {cls.t_final} s")
        print(f"   Control timestep: {cls.dt_control} s")
        
        print("\n🎮 Controller Parameters:")
        print(f"   Energy gain:      {cls.energy_gain}")
        print(f"   Cart gain:        {cls.cart_gain}")
        print(f"   Force limit:      {cls.force_limit} N")
        print(f"   LQR gains K:      {cls.K_lqr}")
        
        print("=" * 70)


# =============================================================================
# VISUALIZER PARAMETERS
# =============================================================================

class VisualizerParameters:
    """Configuration parameters for CartPendulumVisualizer"""
    
    # Visualization settings
    update_rate = 30.0          # Update frequency (Hz)
    
    # Cart geometry
    cart_width = 0.4            # Cart width (m)
    cart_height = 0.1           # Cart height (m)
    cart_depth = 0.1            # Cart depth (m)
    cart_color = Rgba(0.2, 0.6, 0.9, 1.0)  # Blue
    
    # Pendulum rod
    rod_radius = 0.01           # Rod radius (m)
    rod_color = Rgba(0.9, 0.1, 0.1, 1.0)   # Red
    
    # Pendulum bob
    bob_radius = 0.05           # Bob radius (m)
    bob_color = Rgba(0.1, 0.1, 0.9, 1.0)   # Dark blue
    
    # Pivot point
    pivot_radius = 0.03         # Pivot radius (m)
    pivot_color = Rgba(0.3, 0.3, 0.3, 1.0)  # Gray

class CartPendulum(LeafSystem):
    # State vector: [x, xdot, theta, thetadot]
    # x: cart position (horizontal)
    # xdot: cart velocity
    # theta: pendulum angle (measured from vertical, counterclockwise positive)
    # thetadot: pendulum angular velocity
    # Input: horizontal force on cart
    

    def __init__(self, dt=0.01, params=None):   
        LeafSystem.__init__(self)
        
        # Load parameters
        if params is None:
            params = CartPendulumParameters

        # Input: horizontal force on cart
        self.DeclareVectorInputPort("u", 1)

        # State: [x, xdot, theta, thetadot]
        self.state_index = self.DeclareDiscreteState(4)
        self.DeclareStateOutputPort("y", self.state_index)

        # Bob position output (for visualization)
        self.DeclareVectorOutputPort("bob_position", 2, self.CalcBobPosition)

        # Parameters from CartPendulumParameters class
        self.m = params.mass_arm        # pendulum mass
        self.M = params.mass_cart       # cart mass
        self.b = params.cart_friction   # pendulum damping
        self.g = params.gravity
        self.l = params.length_arm
        self.dt = dt

        # Periodic discrete update
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dt,
            offset_sec=0.0,
            update=self.CalcForwardDynamicsDiscreteUpdate
        )

    # ------------------------------------------------------------------
    # KINEMATICS
    # ------------------------------------------------------------------
    def CalcBobPosition(self, context, output):
        x, xdot, theta, thetadot = context.get_discrete_state_vector().CopyToVector()
        bx = x + self.l * np.sin(theta)
        bz = -self.l * np.cos(theta)
        output.SetFromVector([bx, bz])

    def CalcBobPositionFromTheta(self, x, theta):
        bx = x + self.l * np.sin(theta)
        bz = -self.l * np.cos(theta)
        return bx, bz

    def CalcRodCenterFromState(self, x, theta):
        bx, bz = self.CalcBobPositionFromTheta(x, theta)
        return np.array([ (x + bx)/2, 0, bz/2 ])

    def CalcRodTransformFromState(self, x, theta):
        from pydrake.math import RotationMatrix

        bx, bz = self.CalcBobPositionFromTheta(x, theta)
        p = np.array([ (x + bx)/2, 0, bz/2 ])

        # Direction of rod (pivot -> bob)
        rod_dir = np.array([ bx - x, 0, bz ])
        rod_dir = rod_dir / np.linalg.norm(rod_dir)

        # Meshcat cylinder aligned along +Z axis
        R = RotationMatrix.MakeFromOneVector(rod_dir, axis_index=2)
        return RigidTransform(R, p)

    def CalcBobTransformFromState(self, x, theta):
        bx, bz = self.CalcBobPositionFromTheta(x, theta)
        return RigidTransform([bx, 0, bz])

    def CalcCartTransformFromState(self, x):
        return RigidTransform([x, 0, 0])

    def CalcCartAcceleration(self, state, force):
        """Utility to compute cart horizontal acceleration."""
        x, xdot, theta, thetadot = state
        m = self.m
        M = self.M
        l = self.l
        g = self.g

        s = np.sin(theta)
        c = np.cos(theta)
        denom = M + m * s * s
        xddot = (force + m * s * (l * thetadot * thetadot + g * c)) / denom
        return xddot

    # ------------------------------------------------------------------
    # DYNAMICS (discrete update) - Using pendcart.m formulation
    # ------------------------------------------------------------------
    def CalcForwardDynamicsDiscreteUpdate(self, context, discrete_state):
        state = context.get_discrete_state(self.state_index).CopyToVector()
        x, xdot, theta, thetadot = state
        u = self.get_input_port().Eval(context)[0]

        # Dynamics matching pendcart.m exactly
        # From Code 8.2: pendcart.m
        # 
        # function dx = pendcart(x, m, M, L, g, d, u)
        #   Sx = sin(x(3));
        #   Cx = cos(x(3));
        #   D = m*L*L*(M + m*(1 - Cx^2));
        #   
        #   dx(1,1) = x(2);
        #   dx(2,1) = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*x(4)^2*Sx - d*x(2))) + m*L*L*(1/D)*u;
        #   dx(3,1) = x(4);
        #   dx(4,1) = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*x(4)^2*Sx - d*x(2))) - m*L*Cx*(1/D)*u;
        
        M = self.M
        m = self.m
        L = self.l  # Note: using self.l for length
        g = self.g
        d = self.b  # Note: using self.b for damping
        
        # Trigonometric functions
        Sx = np.sin(theta)
        Cx = np.cos(theta)
        
        # Common denominator (from pendcart.m)
        D = m * L * L * (M + m * (1 - Cx**2))
        
        # State derivatives (matching pendcart.m line by line)
        x_dot = xdot
        x_ddot = (1/D) * (-m**2 * L**2 * g * Cx * Sx + 
                          m * L**2 * (m * L * thetadot**2 * Sx - d * xdot)) + \
                 m * L * L * (1/D) * u
        theta_dot = thetadot
        theta_ddot = (1/D) * ((m + M) * m * g * L * Sx - 
                              m * L * Cx * (m * L * thetadot**2 * Sx - d * xdot)) - \
                     m * L * Cx * (1/D) * u

        # Forward Euler integration
        x_new = x + x_dot * self.dt
        xdot_new = xdot + x_ddot * self.dt
        theta_new = theta + theta_dot * self.dt
        thetadot_new = thetadot + theta_ddot * self.dt

        # Update discrete state
        discrete_state.get_mutable_vector(self.state_index).SetFromVector(
            np.array([x_new, xdot_new, theta_new, thetadot_new])
        )

    # ------------------------------------------------------------------
    # ENERGY
    # ------------------------------------------------------------------
    def EvalPotentialEnergy(self, context):
        x, xdot, theta, thetadot = context.get_discrete_state_vector().CopyToVector()
        return self.m * self.g * self.l * (1 - np.cos(theta))

    def EvalKineticEnergy(self, context):
        x, xdot, theta, thetadot = context.get_discrete_state_vector().CopyToVector()
        # cart KE + pendulum rotational KE + pendulum translational KE
        v_bob_x = xdot + self.l * thetadot * np.cos(theta)
        v_bob_z = self.l * thetadot * np.sin(theta)
        KE_cart = 0.5 * self.M * xdot**2
        KE_bob  = 0.5 * self.m * (v_bob_x**2 + v_bob_z**2)
        return KE_cart + KE_bob

    def EvalTotalEnergy(self, context):
        return (self.EvalKineticEnergy(context)
                + self.EvalPotentialEnergy(context))
    
    # ------------------------------------------------------------------
    # LINEAR STATE SPACE (from plant.py)
    # ------------------------------------------------------------------
    def linear_state_space(self):
        """
        Return linearized A, B, C, D matrices around upright equilibrium.
        Matches pendcart_system_matrices.m (Code 8.3) exactly.
        """
        M = self.M
        m = self.m
        L = self.l  # Using L for clarity (textbook notation)
        d = self.b  # Damping coefficient
        g = self.g
        b = 1       # Pendulum up (b=1)
        
        # Linearized system matrices from Code 8.3
        # A = [0  1  0  0;
        #      0  -d/M  b*m*g/M  0;
        #      0  0  0  1;
        #      0  -b*d/(M*L)  -b*(m+M)*g/(M*L)  0];
        
        A = np.array([
            [0,  1,  0,  0],
            [0,  -d/M,  b*m*g/M,  0],
            [0,  0,  0,  1],
            [0,  -b*d/(M*L),  -b*(m+M)*g/(M*L),  0]
        ])
        
        # B = [0; 1/M; 0; b*1/(M*L)];
        B = np.array([[0], [1/M], [0], [b/(M*L)]])
        
        C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        D = np.array([[0], [0]])
        
        return A, B, C, D

class CartPendulumSwingUpController(LeafSystem):

    def __init__(self, plant, params=None):
        
        super().__init__()
        self.plant = plant
        
        # Load parameters
        if params is None:
            params = CartPendulumParameters

        self.energy_gain = params.energy_gain
        self.cart_gain = params.cart_gain
        self.force_limit = params.force_limit
        
        # Compute LQR gains from linear state space if not provided
        if params.K_lqr is None:
            self.K = self.compute_lqr_gains_manual(plant, params.Q_lqr, params.R_lqr)
            print(f"\n🎯 Computed LQR gains: K = {self.K}")
        else:
            self.K = params.K_lqr
            print(f"\n🎯 Using manual LQR gains: K = {self.K}")

        self.state_in = self.DeclareVectorInputPort("state", 4)
        self.force_out = self.DeclareVectorOutputPort("force", 1, self.CalcForceOutWithLQR)

        self.eval_context = plant.CreateDefaultContext()

        # Target energy (upright)
        upright = np.array([0.0, 0.0, np.pi, 0.0])
        ctx = plant.CreateDefaultContext()
        ctx.get_mutable_discrete_state(plant.state_index).SetFromVector(upright)
        self.E_target = plant.EvalTotalEnergy(ctx)

        # switching thresholds from params
        self.e_thresh = params.e_thresh_factor * plant.m * plant.g * plant.l
        self.theta_thresh = np.deg2rad(params.theta_thresh_deg)
        self.theta_dot_thresh = np.deg2rad(params.theta_dot_thresh_deg)
        self.x_thresh = params.x_thresh
        self.xdot_thresh = params.xdot_thresh

        # hysteresis (helps remove chattering)
        self.switch_margin = params.switch_margin
        self.use_lqr = False

    def compute_lqr_gains(self, plant, Q, R):
        """Compute LQR gains using linear state space around upright equilibrium."""
        try:
            import control
            
            # Get linearized state space
            A, B, C, D = plant.linear_state_space()
            
            # Solve continuous-time algebraic Riccati equation
            K, S, E = control.lqr(A, B, Q, R)
            
            print("\n" + "="*70)
            print("LQR CONTROLLER DESIGN")
            print("="*70)
            print(f"\nState-space A matrix eigenvalues (open-loop):")
            print(f"  {np.linalg.eigvals(A)}")
            print(f"\nClosed-loop eigenvalues (A - B*K):")
            print(f"  {E}")
            print(f"\nQ matrix (state cost):")
            print(f"  diag({np.diag(Q)})")
            print(f"\nR matrix (control cost):")
            print(f"  {R[0,0]}")
            print(f"\nComputed LQR gain matrix K:")
            print(f"  [x, x_dot, theta, theta_dot]")
            print(f"  {K[0]}")
            print("="*70)
            
            return K[0]  # Return as 1D array
            
        except ImportError:
            print("⚠️  Warning: 'control' library not found. Using default gains.")
            print("   Install with: pip install control")

            return np.array([100.0, 50.0, 200.0, 20.0])
        
    def compute_lqr_gains_manual(self, plant, Q, R):
        """Alternative: manually compute LQR gains using Riccati equation (scipy)."""
        try:
            from scipy.linalg import solve_continuous_are
            
            # Get linearized state space
            A, B, _, _ = plant.linear_state_space()
            
            # Solve continuous-time algebraic Riccati equation
            P = solve_continuous_are(A, B, Q, R)
            
            # Compute LQR gain: K = R^(-1) * B^T * P
            K = np.linalg.inv(R) @ B.T @ P
            
            # Compute closed-loop eigenvalues
            E = np.linalg.eigvals(A - B @ K)
            
            print("\n" + "="*70)
            print("LQR CONTROLLER DESIGN (Manual - scipy.linalg)")
            print("="*70)
            print(f"\nState-space A matrix eigenvalues (open-loop):")
            print(f"  {np.linalg.eigvals(A)}")
            print(f"\nClosed-loop eigenvalues (A - B*K):")
            print(f"  {E}")
            print(f"\nQ matrix (state cost):")
            print(f"  diag({np.diag(Q)})")
            print(f"\nR matrix (control cost):")
            print(f"  {R[0,0]}")
            print(f"\nComputed LQR gain matrix K:")
            print(f"  [x, x_dot, theta, theta_dot]")
            print(f"  {K[0]}")
            print("="*70)
            
            return K[0]  # Return as 1D array
            
        except ImportError:
            print("⚠️  Warning: 'scipy.linalg' not found. Using default gains.")
            print("   Install with: pip install scipy")
            return np.array([100.0, 50.0, 200.0, 20.0])
    
    def wrap_theta_error(self, theta):
        return np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))

    def near_upright(self, state, E_diff):
        x, xdot, theta, theta_dot = state
        theta_err = self.wrap_theta_error(theta)

        energy_ok = abs(E_diff) < self.e_thresh
        angle_ok  = abs(theta_err) < self.theta_thresh
        angvel_ok = abs(theta_dot) < self.theta_dot_thresh
        x_ok      = abs(x) < self.x_thresh
        xdot_ok   = abs(xdot) < self.xdot_thresh

        return energy_ok and angle_ok and angvel_ok and x_ok and xdot_ok

    def CalcOut(self, context, output):
        state = self.state_in.Eval(context)
        x, xdot, theta, theta_dot = state
        theta_err = self.wrap_theta_error(theta)

        # energy term with cart position penalty and velocity damping
        self.eval_context.get_mutable_discrete_state(self.plant.state_index).SetFromVector(state)
        E = self.plant.EvalTotalEnergy(self.eval_context)
        E_diff = E - self.E_target
        u_energy = -self.energy_gain * theta_dot * E_diff - self.cart_gain * x - 3.0 * xdot

        # LQR term
        u_lqr = -self.K @ np.array([x, xdot, theta_err, theta_dot])

        # smooth switching with hysteresis
        near = self.near_upright(state, E_diff)
        if near:
            self.use_lqr = True
        elif not near and abs(E_diff) > self.e_thresh * self.switch_margin:
            self.use_lqr = False

        # blending
        alpha = 1.0 if self.use_lqr else 0.0
        u = alpha * u_lqr + (1 - alpha) * u_energy

        # saturation
        u = np.clip(u, -self.force_limit, self.force_limit)
        output.SetFromVector([u])

    # def CalcForceOutWithLQR(self, context, output):
    #     state = self.state_in.Eval(context)
    #     x, xdot, theta, theta_dot = state
    #     theta_err = self.wrap_theta_error(theta)

    #     # LQR control only
    #     u_lqr = -self.K @ np.array([x, xdot, theta_err, theta_dot])
    #     u = np.clip(u_lqr, -self.force_limit, self.force_limit)
    #     output.SetFromVector([u])

    def CalcForceOutWithLQR(self, context, output):
        """
        Calculate control force using LQR law: u = -K*(x - xr)
        
        Matches pendcart_system_matrices.m: u = @(x) -K*(x - wr)
        """
        state = self.GetInputPort("state").Eval(context)
        
        # Target/reference state from parameters
        x_target = CartPendulumParameters.x_target
        theta_target = CartPendulumParameters.theta_target
        target_state = np.array([x_target, 0.0, theta_target, 0.0])
        
        # LQR control law: u = -K*(x - xr) (textbook formulation)
        control_force = -self.K @ (state - target_state)
        
        # Apply saturation limits
        # control_force = np.clip(control_force, -self.force_limit, self.force_limit)
        
        output.SetFromVector([control_force])



class CartPendulumVisualizer(LeafSystem):
    """Simple visualizer for pendulum using Meshcat"""
    
    def __init__(self, meshcat, plant, vis_params=None):
        LeafSystem.__init__(self)
        
        if vis_params is None:
            vis_params = VisualizerParameters
        
        self.plant = plant
        self.meshcat = meshcat
        self.vis_params = vis_params
        
        # Extract plant parameters
        self.l = plant.l
        self.b = plant.b
        self.m = plant.m
        self.g = plant.g
        self.plant_params = {
            "m": self.m,
            "l": self.l,
            "b": self.b,    
            "g": self.g,
        }
        
        # Input: cart-pendulum state [x, xdot, theta, theta_dot]
        self.DeclareVectorInputPort("state", 4)

        # Input: pendulum bob position [x, z]
        self.DeclareVectorInputPort("bob_position", 2)

        # Input: applied force (for debugging)
        self.DeclareVectorInputPort("applied_force", 1)
        
        # Declare periodic update for visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=1.0/vis_params.update_rate,
            offset_sec=0.0,
            publish=self.UpdateVisualization
        )
        
        # Set up meshcat scene using parameters
        self.meshcat.SetObject(
            "cart/rigid_body", 
            Box(vis_params.cart_width, vis_params.cart_height, vis_params.cart_depth), 
            vis_params.cart_color
        )
        self.meshcat.SetObject(
            "pendulum/rod", 
            Cylinder(vis_params.rod_radius, self.l), 
            vis_params.rod_color
        )
        self.meshcat.SetObject(
            "pendulum/bob", 
            Sphere(vis_params.bob_radius), 
            vis_params.bob_color
        )
        self.meshcat.SetObject(
            "pendulum/pivot", 
            Sphere(vis_params.pivot_radius), 
            vis_params.pivot_color
        )

        # Initialize poses
        identity_pose = RigidTransform()
        self.meshcat.SetTransform("cart/rigid_body", identity_pose)
        self.meshcat.SetTransform("pendulum/rod", identity_pose)
        self.meshcat.SetTransform("pendulum/bob", identity_pose)
        self.meshcat.SetTransform("pendulum/pivot", identity_pose)
    
    
    def UpdateVisualization(self, context):
        """Update the visualization based on current state"""
        # ---- Read state ----
        x, xdot, theta, theta_dot = self.GetInputPort("state").Eval(context)

        # ---- Bob position ----
        bob_x, bob_z = self.GetInputPort("bob_position").Eval(context)
        
        # Consistent naming for transforms
        T_W_cart = self.plant.CalcCartTransformFromState(x)
        T_W_rod = self.plant.CalcRodTransformFromState(x, theta)
        T_W_bob = self.plant.CalcBobTransformFromState(x, theta)
        T_W_pivot = RigidTransform([x, 0, 0])

        # ---- Read controller input for debugging ----
        applied_force = self.GetInputPort("applied_force").Eval(context)[0]
        state_vec = np.array([x, xdot, theta, theta_dot])
        xddot = self.plant.CalcCartAcceleration(state_vec, applied_force)

        # ---- Concise debug print ----
        time = context.get_time()
        print(
            f"t={time:5.2f}s x={x:5.2f}m ẋ={xdot:6.2f}m/s ẍ={xddot:6.2f}m/s² "
            f"θ={np.rad2deg(theta):6.1f}° θ̇={np.rad2deg(theta_dot):6.1f}°/s "
            f"bob=({bob_x:5.2f},{bob_z:5.2f}) F={applied_force:6.2f}"
        )
        # ---- Send transforms to Meshcat ----
        self.meshcat.SetTransform("cart/rigid_body", T_W_cart)
        self.meshcat.SetTransform("pendulum/rod", T_W_rod)
        self.meshcat.SetTransform("pendulum/bob", T_W_bob)
        self.meshcat.SetTransform("pendulum/pivot", T_W_pivot)


class ForceTrajectorySource(LeafSystem):
    """Provides a single brief impulse then uses strong braking to stop cart completely."""

    def __init__(self, impulse_duration=0.15, impulse_force=2.0):
        super().__init__()
        self.impulse_duration = impulse_duration
        self.impulse_force = impulse_force
        self.brake_gain = 50.0  # Strong proportional brake to stop cart
        self.DeclareVectorInputPort("state", 4)  # Need state for velocity feedback
        self.DeclareVectorOutputPort("force", 1, self.CalcOutput)

    def CalcOutput(self, context, output):
        t = context.get_time()
        state = self.get_input_port().Eval(context)
        x, xdot, theta, thetadot = state
        
        # Phase 1: Brief impulse (0 to impulse_duration)
        if t < self.impulse_duration:
            force = self.impulse_force * np.sin(np.pi * t / self.impulse_duration)
        
        # Phase 2: Active braking to stop cart quickly
        elif t < 1.0:  # Brake for about 1 second after impulse
            # Strong velocity-based braking
            force = -self.brake_gain * xdot
        
        # Phase 3: Cart stopped, zero force - pendulum falls naturally
        else:
            force = 0.0
            
        output.SetFromVector([force])

        
if __name__ == "__main__":
    
    # Print parameter summary
    CartPendulumParameters.print_summary()
    
    # Start Meshcat for visualization
    meshcat = StartMeshcat()
    
    # create the diagram
    builder = DiagramBuilder()
    
    plant = builder.AddSystem(CartPendulum(dt=CartPendulumParameters.dt_control))
    plant.set_name("CartPendulum Plant")

    use_open_loop_force = False

    if use_open_loop_force:
        actuator = builder.AddSystem(ForceTrajectorySource(impulse_duration=0.15, impulse_force=2.0))
        builder.Connect(plant.GetOutputPort("y"), actuator.get_input_port())  # Connect state to actuator
        builder.Connect(actuator.get_output_port(), plant.get_input_port())
        mode_description = "Brief impulse → Strong braking → Cart stops, pendulum falls"
    else:
        actuator = builder.AddSystem(CartPendulumSwingUpController(plant, params=CartPendulumParameters))
        builder.Connect(plant.GetOutputPort("y"), actuator.GetInputPort("state"))
        builder.Connect(actuator.get_output_port(), plant.get_input_port())
        mode_description = "Swing-up energy shaping + LQR balance controller"
    
    # Add custom visualizer
    visualizer = builder.AddSystem(CartPendulumVisualizer(meshcat, plant))

    # pendulum state -> visualizer state input
    builder.Connect(
        plant.GetOutputPort("y"),              # state output port
        visualizer.GetInputPort("state")        # visualizer state input port
    )

    # pendulum bob position -> visualizer bob position input
    builder.Connect(
        plant.GetOutputPort("bob_position"),    # bob position output
        visualizer.GetInputPort("bob_position")  # visualizer bob position input
    )

    # applied force -> visualizer debugging input
    builder.Connect(
        actuator.get_output_port(),
        visualizer.GetInputPort("applied_force")
    )
    
    logger_state = LogVectorOutput(plant.GetOutputPort("y"), builder)
    logger_state.set_name("state logger")
    
    logger_force = LogVectorOutput(actuator.get_output_port(), builder)
    logger_force.set_name("force logger")
    
    diagram = builder.Build()
    diagram.set_name("Closed Loop System (Solution)")
    
    plot_system_graphviz(diagram)
    # plt.show()
    
    # set initial conditions
    context = diagram.CreateDefaultContext()
    context.SetTime(0.0)
    
    # Set discrete state for the pendulum plant using parameters
    plant_context = plant.GetMyContextFromRoot(context)
    initial_state = CartPendulumParameters.get_initial_state()
    plant_context.get_mutable_discrete_state(plant.state_index).SetFromVector(initial_state)
    
    # create the simulator
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    
    print(f"\n{'='*70}")
    print(f"🌐 Meshcat URL: {meshcat.web_url()}")
    print(f"{'='*70}")
    print(f"\n🎯 Cart-Pendulum Demo")
    print(f"   Initial angle: {np.rad2deg(CartPendulumParameters.theta_initial):.1f}°")
    print(f"   Mode: {mode_description}")
    print(f"{'='*70}\n")
    
    # run the simulation with a loop
    print("Running simulation...")
    sim_time = 0.0
    end_time = CartPendulumParameters.t_final
    dt = 0.1  # Step size
    
    while sim_time < end_time:
        simulator.AdvanceTo(sim_time + dt)
        sim_time += dt
    
    print("\nSimulation complete!")
    
    # create plots
    log_state = logger_state.FindLog(context)
    log_force = logger_force.FindLog(context)

    # Extract state variables
    x = log_state.data()[0,:]
    x_dot = log_state.data()[1,:]
    theta = log_state.data()[2,:]
    theta_dot = log_state.data()[3,:]
    time = log_state.sample_times()
    
    # Extract force
    force = log_force.data()[0,:]
    force_time = log_force.sample_times()
    
    # Calculate errors
    x_desired = CartPendulumParameters.x_target
    theta_desired = CartPendulumParameters.theta_target
    position_error = x - x_desired
    angle_error = theta - theta_desired
    
    # Create comprehensive subplot figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle('Cart-Pole System Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Cart Position (x)
    axes[0, 0].plot(time, x, 'b-', linewidth=2)
    axes[0, 0].axhline(y=x_desired, color='r', linestyle='--', label='Target')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Cart Velocity (x_dot)
    axes[0, 1].plot(time, x_dot, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Target')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Cart Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Pendulum Angle (theta)
    axes[1, 0].plot(time, np.rad2deg(theta), 'purple', linewidth=2)
    axes[1, 0].axhline(y=np.rad2deg(theta_desired), color='r', linestyle='--', label='Target')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].set_title('Pendulum Angle')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Pendulum Angular Velocity (theta_dot)
    axes[1, 1].plot(time, np.rad2deg(theta_dot), 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', label='Target')
    axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[1, 1].set_title('Pendulum Angular Velocity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 5: Position Error
    axes[2, 0].plot(time, position_error, 'c-', linewidth=2)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2, 0].set_ylabel('Error (m)')
    axes[2, 0].set_title('Position Error')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Angle Error
    axes[2, 1].plot(time, np.rad2deg(angle_error), 'magenta', linewidth=2)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2, 1].set_ylabel('Error (degrees)')
    axes[2, 1].set_title('Angle Error')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 7: Applied Force
    axes[3, 0].plot(force_time, force, 'r-', linewidth=2)
    axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Force (N)')
    axes[3, 0].set_title('Applied Force')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Plot 8: Phase Portrait (theta vs theta_dot)
    axes[3, 1].plot(np.rad2deg(theta), np.rad2deg(theta_dot), 'b-', linewidth=1, alpha=0.6)
    axes[3, 1].plot(np.rad2deg(theta[0]), np.rad2deg(theta_dot[0]), 'go', markersize=8, label='Start')
    axes[3, 1].plot(np.rad2deg(theta[-1]), np.rad2deg(theta_dot[-1]), 'rs', markersize=8, label='End')
    axes[3, 1].axvline(x=np.rad2deg(theta_desired), color='r', linestyle='--', alpha=0.5)
    axes[3, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[3, 1].set_xlabel('Angle (degrees)')
    axes[3, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[3, 1].set_title('Phase Portrait')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("cart_pole_analysis.png", dpi=150, bbox_inches='tight')
    print("Saved comprehensive analysis plot to: cart_pole_analysis.png")
    
    print(f"\n{'='*70}")
    print(f"🌐 Meshcat URL: {meshcat.web_url()}")
    print(f"   Visit this URL in your browser to see the 3D visualization")
    print(f"{'='*70}\n")

    
    