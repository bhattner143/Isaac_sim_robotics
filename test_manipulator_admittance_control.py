#!/usr/bin/env python3
"""
Test Script: Admittance Control for Manipulator

═══════════════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

This script demonstrates:
1. External force applied to manipulator end effector
2. Manipulator moves compliantly in response to force (admittance control)
3. Similar to manually guiding a robot arm by hand

Key Difference: Impedance vs Admittance Control
• Impedance: Force input → Position output (actuates force, measures position)
• Admittance: Position input → Force output (actuates position, measures force)

In this implementation:
• We apply external force F_ext to the end effector
• Manipulator complies by moving to reduce the force
• Creates virtual "springy" behavior at end effector

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

1. ADMITTANCE DYNAMICS (Virtual Mass-Spring-Damper)
───────────────────────────────────────────────────

The end effector behaves as if it has virtual dynamics:

    M_v ẍ_des + D_v ẋ_des + K_v (x_des - x₀) = F_ext

Where:
    x_des ∈ ℝ²   : Desired end effector position [m]
    ẋ_des ∈ ℝ²   : Desired end effector velocity [m/s]
    ẍ_des ∈ ℝ²   : Desired end effector acceleration [m/s²]
    F_ext ∈ ℝ²   : External force applied to end effector [N]
    M_v ∈ ℝ      : Virtual mass [kg]
    D_v ∈ ℝ      : Virtual damping [N·s/m]
    K_v ∈ ℝ      : Virtual stiffness [N/m]
    x₀ ∈ ℝ²      : Equilibrium position [m]

Solving for acceleration:

    ẍ_des = M_v⁻¹ (F_ext - D_v ẋ_des - K_v (x_des - x₀))

This second-order ODE is integrated as a state-space system:

    State: s = [x_des, ẋ_des]ᵀ ∈ ℝ⁴

    Dynamics:
    ẋ_des = ẋ_des
    v̇_des = M_v⁻¹ (F_ext - D_v ẋ_des - K_v (x_des - x₀))

Physical Interpretation:
• M_v ↑ → More inertia, slower response to force
• D_v ↑ → More damping, less oscillation
• K_v ↑ → Stronger return to equilibrium when force removed
• K_v = 0 → Pure inertia/damping, no restoring force


2. TASK-SPACE CONTROLLER (Tracking Desired Motion)
───────────────────────────────────────────────────

Goal: Make actual end effector x follow desired trajectory x_des

PD Control in Task Space:

    F_task = K_p (x_des - x) + K_d (ẋ_des - ẋ)

Where:
    F_task ∈ ℝ²  : Task-space control force [N]
    K_p ∈ ℝ      : Position gain [N/m]
    K_d ∈ ℝ      : Velocity gain [N·s/m]
    x ∈ ℝ²       : Actual end effector position [m]
    ẋ ∈ ℝ²       : Actual end effector velocity [m/s]


3. JACOBIAN TRANSPOSE MAPPING
───────────────────────────────────────────────────

Map task-space force to joint torques:

    τ = J^T F_task

Where:
    τ ∈ ℝⁿ       : Joint torques [N·m] (n = number of joints)
    J ∈ ℝ²ˣⁿ     : Jacobian matrix (maps joint vel → EE vel)
    J^T ∈ ℝⁿˣ²   : Jacobian transpose (maps EE force → joint torque)

Jacobian Definition:
    ẋ = J(q) q̇

    J = ∂x/∂q = [∂x/∂q₁  ∂x/∂q₂]
                 [∂y/∂q₁  ∂y/∂q₂]

Virtual Work Principle:
    δW = F_task^T δx = τ^T δq
    
    Since δx = J δq:
    F_task^T J δq = τ^T δq
    
    Therefore: τ = J^T F_task

This ensures static force balance between task space and joint space.


4. COMPLETE CONTROL LOOP
───────────────────────────────────────────────────

Step 1: Read external force F_ext (provided by environment/user)

Step 2: Admittance dynamics computes desired motion
    ẍ_des = M_v⁻¹ (F_ext - D_v ẋ_des - K_v (x_des - x₀))
    
    Integrate to get: x_des, ẋ_des

Step 3: Forward kinematics computes actual EE state
    x = FK(q)
    ẋ = J(q) q̇

Step 4: Task-space PD control
    F_task = K_p (x_des - x) + K_d (ẋ_des - ẋ)

Step 5: Map to joint torques
    τ = J^T(q) F_task

Step 6: Apply torques to robot
    Plant dynamics: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ


═══════════════════════════════════════════════════════════════════════════════
DESIGN PARAMETERS & TUNING
═══════════════════════════════════════════════════════════════════════════════

Admittance Parameters (affect compliance):
    M_v : 0.5 - 5.0 kg
        Low  → Fast response, feels light
        High → Slow response, feels heavy
    
    D_v : 2.0 - 20.0 N·s/m
        Low  → Less damping, may oscillate
        High → More damping, sluggish feel
    
    K_v : 0 - 50.0 N/m
        0    → No restoring force (integration behavior)
        Low  → Weak return to equilibrium
        High → Strong return (stiff spring)

    Typical Ratios:
        Critical damping: D_v = 2√(M_v K_v)
        Underdamped: D_v < 2√(M_v K_v) → oscillatory
        Overdamped: D_v > 2√(M_v K_v) → slow return

Task Controller Gains (affect tracking):
    K_p : 100 - 500 N/m
        Controls position tracking stiffness
    
    K_d : 20 - 100 N·s/m
        Controls velocity tracking damping


═══════════════════════════════════════════════════════════════════════════════
ALTERNATIVE CONTROL LAW: PURE DAMPING CONTROL
═══════════════════════════════════════════════════════════════════════════════

What if we use pure velocity feedback instead?

    τ = -K_q q̇

Where:
    τ ∈ ℝⁿ     : Joint torques [N·m]
    K_q ∈ ℝ    : Velocity gain (damping coefficient) [N·m·s/rad]
    q̇ ∈ ℝⁿ     : Joint velocities [rad/s]


BEHAVIOR ANALYSIS
─────────────────────────────────────────────────────────────────────────────

Robot dynamics:
    M(q)q̈ + C(q,q̇)q̇ + g(q) = τ

Substituting τ = -K_q q̇:
    M(q)q̈ + C(q,q̇)q̇ + g(q) = -K_q q̇
    M(q)q̈ + (C(q,q̇) + K_q I)q̇ + g(q) = 0

This creates PASSIVE DAMPING:

✓ What happens:
    1. Robot resists motion proportional to velocity
    2. Energy is dissipated (system loses kinetic energy)
    3. Robot eventually comes to REST at some equilibrium position
    4. NO active position control - doesn't track a desired position
    5. Acts like adding viscous friction to joints

✗ What DOESN'T happen:
    1. No position regulation - won't return to a specific pose
    2. Won't move to follow commands (unless you push it)
    3. Gravity still pulls robot down (unless K_q very high)


EQUILIBRIUM ANALYSIS
─────────────────────────────────────────────────────────────────────────────

At steady state: q̈ = 0, q̇ = 0

    g(q_eq) = 0

The robot settles where:
    • Joint velocities are zero
    • Gravitational torques are balanced by mechanical constraints
    • NOT necessarily at a desired configuration!

For a planar manipulator with gravity:
    • Will droop/sag under its own weight
    • K_q only resists MOTION, not static forces
    • To hold position against gravity, need K_p term!


ENERGY PERSPECTIVE
─────────────────────────────────────────────────────────────────────────────

Power dissipated by damping:
    P = τ^T q̇ = -K_q q̇^T q̇ = -K_q ||q̇||² ≤ 0

This is ALWAYS NEGATIVE (energy removed from system):
    • Robot slows down over time
    • Kinetic energy → heat
    • System is PASSIVE and STABLE
    • Cannot add energy to system


COMPARISON: τ = -K_p q - K_q q̇  (Full PD Control)
─────────────────────────────────────────────────────────────────────────────

With position feedback added:
    τ = -K_p (q - q_des) - K_q (q̇ - q̇_des)

Now the robot:
    ✓ Actively moves toward q_des
    ✓ Resists deviations from q_des (spring-like)
    ✓ Damps oscillations (damper-like)
    ✓ Can hold position against gravity
    ✓ Can track trajectories

Without K_p (pure damping τ = -K_q q̇):
    ✗ No position target
    ✗ Just resists motion
    ✗ Sags under gravity
    ✗ Like a robot with "loose" joints that resist being moved


PRACTICAL APPLICATIONS OF PURE DAMPING
─────────────────────────────────────────────────────────────────────────────

When τ = -K_q q̇ is useful:

1. Gravity Compensation Mode:
    τ = g(q) - K_q q̇
    → Robot floats weightlessly, resists fast motion
    → Used for manual teaching/demonstration

2. Safety/Compliance:
    τ = τ_desired - K_q q̇
    → Adds artificial damping to smooth out jerky commands
    → Prevents oscillations

3. Energy Dissipation:
    → Slow down robot before switching controllers
    → Brake/coast to stop

4. Testing:
    → Verify velocity sensing works
    → Measure actual joint friction


NUMERICAL EXAMPLE: τ = -K_q q̇
─────────────────────────────────────────────────────────────────────────────

Given:
    K_q = 10 N·m·s/rad
    q̇ = [1.0, -0.5] rad/s  (joint velocities)

Torque output:
    τ = -K_q q̇
      = -10 × [1.0, -0.5]
      = [-10, 5] N·m

Interpretation:
    • Joint 1 moving at +1.0 rad/s → torque -10 N·m (opposes motion)
    • Joint 2 moving at -0.5 rad/s → torque +5 N·m (opposes motion)
    • Both torques act to SLOW DOWN the joints
    • No torque when joints are stationary (q̇ = 0)


STABILITY ANALYSIS
─────────────────────────────────────────────────────────────────────────────

Lyapunov function (kinetic energy):
    V = ½ q̇^T M(q) q̇ ≥ 0

Time derivative:
    V̇ = q̇^T M(q) q̈ + ½ q̇^T Ṁ(q) q̇
      = q̇^T [τ - C(q,q̇)q̇ - g(q)] + ½ q̇^T Ṁ(q) q̇

Using τ = -K_q q̇ and the property Ṁ - 2C is skew-symmetric:
    V̇ = -K_q ||q̇||² ≤ 0

Since V̇ ≤ 0:
    ✓ System is STABLE (energy decreases)
    ✓ Robot CANNOT speed up on its own
    ✓ Eventually reaches q̇ = 0
    ✓ PASSIVE system (safe!)


SUMMARY: τ = -K_q q̇
─────────────────────────────────────────────────────────────────────────────

Effect:         Damping/friction on joints
Behavior:       Resists motion, dissipates energy
Steady State:   q̇ = 0 (stationary, but arbitrary position)
Stability:      Stable (passive)
Energy:         Always dissipates, never adds

Use Cases:
    ✓ Add safety damping
    ✓ Smooth out motion
    ✓ Coast to stop
    
NOT suitable for:
    ✗ Position control
    ✗ Trajectory tracking
    ✗ Holding against gravity
    
To control position: MUST add K_p term!
    τ = -K_p (q - q_des) - K_q q̇


═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: 20N FORCE IN +X DIRECTION
═══════════════════════════════════════════════════════════════════════════════

Given: M_v = 1.0 kg, D_v = 5.0 N·s/m, K_v = 10.0 N/m, F_ext = 20N in +X

At steady state (ẍ = 0, ẋ = 0):
    0 = F_ext - K_v (x_des - x₀)
    x_des = x₀ + F_ext/K_v = x₀ + 20/10 = x₀ + 2.0 m

The end effector will displace 2.0 m in the +X direction!

Transient response (natural frequency & damping ratio):
    ωₙ = √(K_v/M_v) = √(10/1) = 3.16 rad/s
    ζ = D_v/(2√(M_v K_v)) = 5/(2√10) = 0.79 (underdamped)
    
    Rise time ≈ 1.8/ωₙ ≈ 0.57 s
    Settling time ≈ 4/(ζωₙ) ≈ 1.6 s

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    SceneGraph,
    AddMultibodyPlantSceneGraph,
)
from pydrake.multibody.tree import JacobianWrtVariable
from termcolor import colored

# Import from main script
from robot_types import create_cup_manipulator_config
from script_cup_manipulator_controller_ofc import CupManipulator


class ExternalForceSource(LeafSystem):
    """
    Generates time-varying external force to apply to end effector.
    
    Force profile:
    - Ramp up from 0 to F_max over t=[0, 1]s
    - Hold at F_max for t=[1, 5]s
    - Ramp down from F_max to 0 over t=[5, 6]s
    - Zero for t > 6s
    
    Outputs:
    - External force (2D): [Fx, Fy] in world frame
    """
    
    def __init__(self, F_max=20.0, direction=np.array([1.0, 0.0])):
        """
        Initialize external force source.
        
        Args:
            F_max: Maximum force magnitude [N]
            direction: Force direction (2D unit vector)
        """
        LeafSystem.__init__(self)
        
        self.F_max = F_max
        self.direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Output port
        self.DeclareVectorOutputPort(
            "external_force",
            BasicVector(2),
            self.CalcForce
        )
    
    def CalcForce(self, context, output):
        """Generate time-varying force profile."""
        t = context.get_time()
        
        # Piecewise force profile
        if t < 1.0:
            # Ramp up
            magnitude = self.F_max * (t / 1.0)
        elif t < 5.0:
            # Hold
            magnitude = self.F_max
        elif t < 6.0:
            # Ramp down
            magnitude = self.F_max * (1.0 - (t - 5.0) / 1.0)
        else:
            # Zero
            magnitude = 0.0
        
        force = magnitude * self.direction
        output.SetFromVector(force)


class AdmittanceController(LeafSystem):
    """
    Admittance controller: external force -> desired end effector motion.
    
    ═══════════════════════════════════════════════════════════════════════
    MATHEMATICAL MODEL
    ═══════════════════════════════════════════════════════════════════════
    
    Implements virtual mass-spring-damper dynamics at the end effector:
    
        M_v ẍ_des + D_v ẋ_des + K_v (x_des - x₀) = F_ext
    
    Rearranged as state-space form:
    
        State: s = [x_des, y_des, vx_des, vy_des]ᵀ ∈ ℝ⁴
        
        Dynamics:
            ẋ_des = v_des                                    (velocity)
            v̇_des = M_v⁻¹ (F_ext - D_v v_des - K_v (x_des - x₀))  (acceleration)
    
    ═══════════════════════════════════════════════════════════════════════
    PARAMETERS
    ═══════════════════════════════════════════════════════════════════════
    
    M_v : Virtual mass [kg]
        - Physical interpretation: inertia felt when pushing the end effector
        - Higher M_v → slower response, feels "heavier"
        - Typical: 0.5 - 5.0 kg
    
    D_v : Virtual damping [N·s/m]
        - Resists velocity, dissipates energy
        - Higher D_v → more sluggish, less oscillation
        - Typical: 2.0 - 20.0 N·s/m
    
    K_v : Virtual stiffness [N/m]
        - Restoring force toward equilibrium x₀
        - K_v = 0 → pure integrator (no restoring force)
        - Higher K_v → stronger return to x₀ when force removed
        - Typical: 0 - 50.0 N/m
    
    x₀ : Equilibrium position [m]
        - Position where end effector rests when F_ext = 0
        - Usually set to initial EE position
    
    ═══════════════════════════════════════════════════════════════════════
    STEADY-STATE RESPONSE
    ═══════════════════════════════════════════════════════════════════════
    
    For constant force F_ext (steady state: ẍ = 0, ẋ = 0):
    
        K_v (x_des - x₀) = F_ext
        
        x_des = x₀ + F_ext / K_v
    
    Example: K_v = 10 N/m, F_ext = 20 N → x_des = x₀ + 2.0 m
    
    ═══════════════════════════════════════════════════════════════════════
    TRANSIENT RESPONSE
    ═══════════════════════════════════════════════════════════════════════
    
    Natural frequency:
        ωₙ = √(K_v/M_v)  [rad/s]
    
    Damping ratio:
        ζ = D_v / (2√(M_v K_v))
    
    Behavior:
        ζ < 1 : Underdamped (oscillatory)
        ζ = 1 : Critically damped (fastest non-oscillatory)
        ζ > 1 : Overdamped (slow, no oscillation)
    
    Critical damping:
        D_v_critical = 2√(M_v K_v)
    
    ═══════════════════════════════════════════════════════════════════════
    INPUTS/OUTPUTS
    ═══════════════════════════════════════════════════════════════════════
    
    Inputs:
        - External force (2D): [Fx, Fy]  [N]
        - Current EE position (2D): [x, y]  [m] (not used in current implementation)
        - Current EE velocity (2D): [vx, vy]  [m/s] (not used in current implementation)
    
    Outputs:
        - Desired EE position (2D): [x_des, y_des]  [m]
        - Desired EE velocity (2D): [vx_des, vy_des]  [m/s]
    
    State:
        - [x_des, y_des, vx_des, vy_des] (4D continuous state)
    """
    
    def __init__(self, M_virtual=1.0, D_virtual=5.0, K_virtual=10.0, x0=None):
        """
        Initialize admittance controller.
        
        Args:
            M_virtual: Virtual mass [kg]
            D_virtual: Virtual damping [N·s/m]
            K_virtual: Virtual stiffness [N/m] (restoring force)
            x0: Equilibrium position [m] (if None, uses initial EE position)
        """
        LeafSystem.__init__(self)
        
        self.M_virtual = M_virtual
        self.D_virtual = D_virtual
        self.K_virtual = K_virtual
        self.x0 = x0 if x0 is not None else np.zeros(2)
        
        # State: [x_des, y_des, vx_des, vy_des]
        self.DeclareContinuousState(4)
        
        # Input ports
        self.force_input = self.DeclareVectorInputPort("external_force", BasicVector(2))
        self.ee_pos_input = self.DeclareVectorInputPort("ee_position", BasicVector(2))
        self.ee_vel_input = self.DeclareVectorInputPort("ee_velocity", BasicVector(2))
        
        # Output ports
        self.DeclareVectorOutputPort(
            "desired_ee_position",
            BasicVector(2),
            self.OutputDesiredPosition
        )
        self.DeclareVectorOutputPort(
            "desired_ee_velocity",
            BasicVector(2),
            self.OutputDesiredVelocity
        )
    
    def SetDefaultState(self, context, state):
        """Initialize state to current EE position and zero velocity."""
        # Start at equilibrium with zero velocity
        state.SetFromVector([self.x0[0], self.x0[1], 0.0, 0.0])
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute state derivatives using admittance dynamics.
        
        ẋ_des = v_des
        v̇_des = M^-1 * (F_ext - D * v_des - K * (x_des - x_0))
        """
        state = context.get_continuous_state_vector().CopyToVector()
        x_des = state[:2]
        v_des = state[2:]
        
        # Get external force
        F_ext = self.force_input.Eval(context)
        
        # Compute acceleration using admittance dynamics
        # a_des = M^-1 * (F_ext - D * v_des - K * (x_des - x_0))
        spring_force = -self.K_virtual * (x_des - self.x0)
        damping_force = -self.D_virtual * v_des
        a_des = (F_ext + spring_force + damping_force) / self.M_virtual
        
        # State derivatives: [v_des, a_des]
        derivatives.SetFromVector(np.concatenate([v_des, a_des]))
    
    def OutputDesiredPosition(self, context, output):
        """Output desired position."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[:2])
    
    def OutputDesiredVelocity(self, context, output):
        """Output desired velocity."""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state[2:])


class EndEffectorKinematics(LeafSystem):
    """
    Compute end effector position and velocity using manipulator's CalcPosition.
    
    Inputs:
    - Manipulator state (4D): [q1, q2, q1_dot, q2_dot]
    
    Outputs:
    - End effector position (2D): [x_ee, y_ee]
    - End effector velocity (2D): [vx_ee, vy_ee]
    """
    
    def __init__(self, plant: MultibodyPlant, manipulator: CupManipulator):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        # Create context for plant queries
        self.plant_context = plant.CreateDefaultContext()
        
        # Input port
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output ports
        self.DeclareVectorOutputPort("ee_position", BasicVector(2), self.CalcPosition)
        self.DeclareVectorOutputPort("ee_velocity", BasicVector(2), self.CalcVelocity)
    
    def CalcPosition(self, context, output):
        """Compute end effector position using manipulator's CalcPosition method."""
        state = self.state_input.Eval(context)
        q = state[:2]
        
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        ee_pos_world = self.manipulator.CalcPosition(self.plant, self.plant_context)
        
        output.SetFromVector([ee_pos_world[0], ee_pos_world[1]])
    
    def CalcVelocity(self, context, output):
        """Compute end effector velocity using Drake's Jacobian."""
        state = self.state_input.Eval(context)
        q = state[:2]
        v = state[2:]
        
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        self.plant.SetVelocities(self.plant_context, self.model_instance, v)
        
        # Compute Jacobian at simple_ball offset
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            self.manipulator.EE_OFFSET,
            self.world_frame,
            self.world_frame
        )
        
        # Extract linear velocity part and project to X-Y
        J_translational = J_spatial[3:6, :]
        J_manip = J_translational[:, :2]
        v_ee_world = J_manip @ v
        
        output.SetFromVector([v_ee_world[0], v_ee_world[1]])


class TaskSpaceController(LeafSystem):
    """
    Task-space controller: desired EE motion -> joint torques.
    
    ═══════════════════════════════════════════════════════════════════════
    CONTROL LAW: JACOBIAN TRANSPOSE METHOD
    ═══════════════════════════════════════════════════════════════════════
    
    Step 1: Task-space PD control force
    ────────────────────────────────────
    
        F_task = K_p (x_des - x) + K_d (ẋ_des - ẋ)
    
    Where:
        F_task ∈ ℝ² : Control force in task space (end effector) [N]
        x_des ∈ ℝ²  : Desired end effector position [m]
        x ∈ ℝ²      : Actual end effector position [m]
        ẋ_des ∈ ℝ²  : Desired end effector velocity [m/s]
        ẋ ∈ ℝ²      : Actual end effector velocity [m/s]
        K_p ∈ ℝ     : Position gain [N/m]
        K_d ∈ ℝ     : Velocity gain [N·s/m]
    
    This creates a virtual spring-damper that pulls the EE toward x_des.
    
    
    Step 2: Map task force to joint torques using Jacobian transpose
    ─────────────────────────────────────────────────────────────────
    
        τ = J^T(q) F_task
    
    Where:
        τ ∈ ℝⁿ      : Joint torques [N·m] (n = number of joints)
        J(q) ∈ ℝ²ˣⁿ : Manipulator Jacobian (task vel = J × joint vel)
        J^T ∈ ℝⁿˣ²  : Jacobian transpose
    
    ═══════════════════════════════════════════════════════════════════════
    WHY JACOBIAN TRANSPOSE?
    ═══════════════════════════════════════════════════════════════════════
    
    The Jacobian J relates joint velocities to end effector velocities:
    
        ẋ = J(q) q̇
    
    For a 2-DOF planar manipulator:
    
        [ẋ]   [∂x/∂q₁  ∂x/∂q₂] [q̇₁]
        [ẏ] = [∂y/∂q₁  ∂y/∂q₂] [q̇₂]
    
    
    Virtual Work Principle:
    ───────────────────────
    
    For static equilibrium, virtual work done by forces must be equal:
    
        δW_task = δW_joint
        F_task^T δx = τ^T δq
    
    Since δx = J δq:
    
        F_task^T (J δq) = τ^T δq
        (J^T F_task)^T δq = τ^T δq
    
    Therefore:
        τ = J^T F_task
    
    This ensures that:
    • Force at EE properly maps to torques at joints
    • Energy is conserved (virtual work is consistent)
    • Control is stable (passive mapping)
    
    
    Advantages of Jacobian Transpose:
    ──────────────────────────────────
    
    ✓ Simple to compute (no matrix inversion)
    ✓ Always well-defined (even near singularities)
    ✓ Passive/stable mapping
    ✓ Natural force transmission
    
    Compare to Jacobian Inverse (τ = J⁻¹ a_des):
    ✗ Requires matrix inversion
    ✗ Undefined at singularities
    ✗ May amplify noise
    
    ═══════════════════════════════════════════════════════════════════════
    GAIN TUNING
    ═══════════════════════════════════════════════════════════════════════
    
    K_p (Position Gain):
        - Higher → stiffer tracking, faster response
        - Too high → oscillation, instability
        - Typical: 100 - 500 N/m
    
    K_d (Velocity Gain):
        - Higher → more damping, less overshoot
        - Too high → sluggish response
        - Typical: 20 - 100 N·s/m
        - Rule of thumb: K_d ≈ 0.2 K_p for good damping
    
    ═══════════════════════════════════════════════════════════════════════
    INPUTS/OUTPUTS
    ═══════════════════════════════════════════════════════════════════════
    
    Inputs:
        - Manipulator state (4D): [q₁, q₂, q̇₁, q̇₂]
        - Desired EE position (2D): [x_des, y_des]  [m]
        - Desired EE velocity (2D): [vx_des, vy_des]  [m/s]
    
    Outputs:
        - Joint torques (2D): [τ₁, τ₂]  [N·m]
    
    ═══════════════════════════════════════════════════════════════════════
    EXAMPLE CALCULATION
    ═══════════════════════════════════════════════════════════════════════
    
    Given:
        x_des = [1.0, 0.5] m
        x = [0.9, 0.4] m  
        ẋ_des = [0.1, 0.0] m/s
        ẋ = [0.05, -0.01] m/s
        K_p = 200 N/m
        K_d = 40 N·s/m
    
    Step 1: Compute errors
        e_pos = x_des - x = [0.1, 0.1] m
        e_vel = ẋ_des - ẋ = [0.05, 0.01] m/s
    
    Step 2: Task force
        F_task = K_p e_pos + K_d e_vel
               = 200[0.1, 0.1] + 40[0.05, 0.01]
               = [20, 20] + [2, 0.4]
               = [22, 20.4] N
    
    Step 3: Map to joints (assuming J^T = [[0.8, 0.6], [0.3, 0.9]])
        τ = J^T F_task
          = [[0.8, 0.6],  [22  ]
             [0.3, 0.9]]  [20.4]
          = [0.8×22 + 0.6×20.4, 0.3×22 + 0.9×20.4]
          = [17.6 + 12.24, 6.6 + 18.36]
          = [29.84, 24.96] N·m
    
    The controller outputs τ₁ = 29.84 N·m and τ₂ = 24.96 N·m to the joints.
    """
    
    def __init__(self, plant: MultibodyPlant, manipulator: CupManipulator, 
                 kp=100.0, kd=20.0):
        """
        Initialize task-space controller.
        
        Args:
            plant: MultibodyPlant instance
            manipulator: CupManipulator instance
            kp: Task-space position gain [N/m]
            kd: Task-space velocity gain [N·s/m]
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        self.kp = kp
        self.kd = kd
        
        # Create context for plant queries
        self.plant_context = plant.CreateDefaultContext()
        
        # Input ports
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        self.des_pos_input = self.DeclareVectorInputPort("desired_ee_position", BasicVector(2))
        self.des_vel_input = self.DeclareVectorInputPort("desired_ee_velocity", BasicVector(2))
        
        # Output port
        self.DeclareVectorOutputPort("torque_output", BasicVector(2), self.CalcTorque)
    
    def CalcTorque(self, context, output):
        """Compute joint torques using Jacobian transpose control."""
        state = self.state_input.Eval(context)
        q = state[:2]
        v = state[2:]
        
        des_pos = self.des_pos_input.Eval(context)
        des_vel = self.des_vel_input.Eval(context)
        
        # Set plant state
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        self.plant.SetVelocities(self.plant_context, self.model_instance, v)
        
        # Get current EE position and velocity
        ee_pos = self.manipulator.CalcPosition(self.plant, self.plant_context)
        
        # Compute Jacobian
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            self.manipulator.EE_OFFSET,
            self.world_frame,
            self.world_frame
        )
        J_translational = J_spatial[3:6, :]
        J_manip = J_translational[:, :2]
        
        ee_vel = J_manip @ v
        
        # Task-space PD control
        pos_error = des_pos[:2] - np.array([ee_pos[0], ee_pos[1]])
        vel_error = des_vel - ee_vel[:2]
        
        F_task = self.kp * pos_error + self.kd * vel_error
        
        # Jacobian transpose mapping: τ = J^T * F_task
        torque = J_manip[:2, :].T @ F_task
        
        output.SetFromVector(torque)


def test_admittance_control(
    duration=10.0,
    F_max=20.0,
    force_direction=np.array([1.0, 0.0]),
    M_virtual=10.0,
    D_virtual=5.0,
    K_virtual=10.0,
):
    """
    Test admittance control: external force -> compliant motion.
    
    Args:
        duration: Simulation duration [s]
        F_max: Maximum external force [N]
        force_direction: Force direction (2D unit vector)
        M_virtual: Virtual mass [kg]
        D_virtual: Virtual damping [N·s/m]
        K_virtual: Virtual stiffness [N/m]
        
    Returns:
        log_data: Dictionary with logged data
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("ADMITTANCE CONTROL: FORCE -> COMPLIANT MOTION", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Start Meshcat server
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat server started at: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Create manipulator configuration
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),  # Initial pose
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
    
    # Add actuators
    joint1 = plant.GetJointByName("link1_base", manipulator.model_instance)
    joint2 = plant.GetJointByName("link2_link1", manipulator.model_instance)
    plant.AddJointActuator("joint1_actuator", joint1)
    plant.AddJointActuator("joint2_actuator", joint2)
    
    # Finalize plant
    plant.Finalize()
    
    # Get initial EE position for equilibrium
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, manipulator.model_instance, 
                      np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    x0 = manipulator.CalcPosition(plant, temp_context)[:2]  # X-Y only
    
    print(colored(f"\n✓ System created", "green"))
    print(colored(f"  Manipulator DOF: {plant.num_positions()}", "cyan"))
    print(colored(f"  Initial EE position: x={x0[0]:.3f} m, y={x0[1]:.3f} m", "cyan"))
    print(colored(f"\nAdmittance Parameters:", "yellow", attrs=["bold"]))
    print(colored(f"  Virtual mass: {M_virtual:.2f} kg", "cyan"))
    print(colored(f"  Virtual damping: {D_virtual:.2f} N·s/m", "cyan"))
    print(colored(f"  Virtual stiffness: {K_virtual:.2f} N/m", "cyan"))
    print(colored(f"\nExternal Force:", "yellow", attrs=["bold"]))
    print(colored(f"  Maximum: {F_max:.1f} N", "cyan"))
    print(colored(f"  Direction: [{force_direction[0]:.2f}, {force_direction[1]:.2f}]", "cyan"))
    
    # Create systems
    force_source = builder.AddSystem(ExternalForceSource(F_max, force_direction))
    
    admittance = builder.AddSystem(AdmittanceController(
        M_virtual=M_virtual,
        D_virtual=D_virtual,
        K_virtual=K_virtual,
        x0=x0
    ))
    
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    
    task_controller = builder.AddSystem(TaskSpaceController(
        plant, manipulator, kp=200.0, kd=40.0
    ))
    
    # Create demux for plant state
    from pydrake.systems.primitives import Demultiplexer
    state_demux = builder.AddSystem(Demultiplexer([2, 2]))  # [positions, velocities]
    
    # Connect external force source to admittance controller
    builder.Connect(
        force_source.get_output_port(),
        admittance.GetInputPort("external_force")
    )
    
    # Connect plant state to EE kinematics
    builder.Connect(
        plant.get_state_output_port(),
        ee_kinematics.GetInputPort("manipulator_state")
    )
    
    # Connect EE kinematics to admittance controller
    builder.Connect(
        ee_kinematics.GetOutputPort("ee_position"),
        admittance.GetInputPort("ee_position")
    )
    builder.Connect(
        ee_kinematics.GetOutputPort("ee_velocity"),
        admittance.GetInputPort("ee_velocity")
    )
    
    # Connect admittance output and plant state to task controller
    builder.Connect(
        admittance.GetOutputPort("desired_ee_position"),
        task_controller.GetInputPort("desired_ee_position")
    )
    builder.Connect(
        admittance.GetOutputPort("desired_ee_velocity"),
        task_controller.GetInputPort("desired_ee_velocity")
    )
    builder.Connect(
        plant.get_state_output_port(),
        task_controller.GetInputPort("manipulator_state")
    )
    
    # Connect task controller to plant actuation
    builder.Connect(
        task_controller.get_output_port(),
        plant.get_actuation_input_port()
    )
    
    # Add Meshcat visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Add loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(force_source.get_output_port(), force_logger.get_input_port())
    
    ee_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_pos_logger.get_input_port())
    
    des_pos_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(admittance.GetOutputPort("desired_ee_position"), des_pos_logger.get_input_port())
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    # Start recording
    visualizer.StartRecording()
    
    print(colored(f"\nSimulating for {duration} s...", "yellow"))
    simulator.AdvanceTo(duration)
    print(colored("✓ Simulation complete", "green"))
    
    # Publish recording
    visualizer.PublishRecording()
    print(colored(f"\n🎬 Animation published to Meshcat: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    force_log = force_logger.FindLog(context)
    ee_pos_log = ee_pos_logger.FindLog(context)
    des_pos_log = des_pos_logger.FindLog(context)
    
    time_data = state_log.sample_times()
    state_data = state_log.data()
    force_data = force_log.data()
    ee_pos_data = ee_pos_log.data()
    des_pos_data = des_pos_log.data()
    
    # Parse state data
    q1 = state_data[0, :]
    q2 = state_data[1, :]
    q1_dot = state_data[2, :]
    q2_dot = state_data[3, :]
    
    # Plot results
    print(colored(f"\n📈 Generating plots...", "yellow"))
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Plot 1: External force vs time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_data, force_data[0, :], 'b-', linewidth=2, label='Fx')
    ax1.plot(time_data, force_data[1, :], 'r-', linewidth=2, label='Fy')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Force [N]', fontweight='bold')
    ax1.set_title('External Force Profile', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: EE position vs time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_data, ee_pos_data[0, :], 'b-', linewidth=2, label='Actual X')
    ax2.plot(time_data, des_pos_data[0, :], 'b--', linewidth=2, alpha=0.7, label='Desired X')
    ax2.plot(time_data, ee_pos_data[1, :], 'r-', linewidth=2, label='Actual Y')
    ax2.plot(time_data, des_pos_data[1, :], 'r--', linewidth=2, alpha=0.7, label='Desired Y')
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('Position [m]', fontweight='bold')
    ax2.set_title('End Effector Position', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Joint angles vs time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_data, np.rad2deg(q1), 'b-', linewidth=2, label='q₁')
    ax3.plot(time_data, np.rad2deg(q2), 'r-', linewidth=2, label='q₂')
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Angle [deg]', fontweight='bold')
    ax3.set_title('Joint Angles', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: EE trajectory in X-Y plane
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ee_pos_data[0, :], ee_pos_data[1, :], 'purple', linewidth=2.5, alpha=0.7, label='Actual')
    ax4.plot(des_pos_data[0, :], des_pos_data[1, :], 'orange', linewidth=2, alpha=0.5, linestyle='--', label='Desired')
    ax4.plot(ee_pos_data[0, 0], ee_pos_data[1, 0], 'go', markersize=12, label='Start')
    ax4.plot(ee_pos_data[0, -1], ee_pos_data[1, -1], 'ro', markersize=12, label='End')
    ax4.set_xlabel('X Position [m]', fontweight='bold')
    ax4.set_ylabel('Y Position [m]', fontweight='bold')
    ax4.set_title('End Effector Path (X-Y Plane)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.legend()
    
    # Plot 5: Displacement vs time
    ax5 = fig.add_subplot(gs[1, 1])
    displacement = np.sqrt((ee_pos_data[0, :] - ee_pos_data[0, 0])**2 + 
                          (ee_pos_data[1, :] - ee_pos_data[1, 0])**2)
    ax5.plot(time_data, displacement, 'purple', linewidth=2.5)
    ax5.set_xlabel('Time [s]', fontweight='bold')
    ax5.set_ylabel('Displacement [m]', fontweight='bold')
    ax5.set_title('EE Displacement from Initial Position', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Joint velocities
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(time_data, np.rad2deg(q1_dot), 'b-', linewidth=2, label='q̇₁')
    ax6.plot(time_data, np.rad2deg(q2_dot), 'r-', linewidth=2, label='q̇₂')
    ax6.set_xlabel('Time [s]', fontweight='bold')
    ax6.set_ylabel('Angular Velocity [deg/s]', fontweight='bold')
    ax6.set_title('Joint Velocities', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Force vs displacement
    ax7 = fig.add_subplot(gs[2, 0])
    force_magnitude = np.sqrt(force_data[0, :]**2 + force_data[1, :]**2)
    ax7.plot(displacement, force_magnitude, 'green', linewidth=2.5)
    ax7.set_xlabel('Displacement [m]', fontweight='bold')
    ax7.set_ylabel('Force Magnitude [N]', fontweight='bold')
    ax7.set_title('Force vs Displacement', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Tracking error
    ax8 = fig.add_subplot(gs[2, 1])
    error_x = des_pos_data[0, :] - ee_pos_data[0, :]
    error_y = des_pos_data[1, :] - ee_pos_data[1, :]
    error_magnitude = np.sqrt(error_x**2 + error_y**2)
    ax8.plot(time_data, error_magnitude * 1000, 'red', linewidth=2)  # mm
    ax8.set_xlabel('Time [s]', fontweight='bold')
    ax8.set_ylabel('Tracking Error [mm]', fontweight='bold')
    ax8.set_title('Desired vs Actual EE Position Error', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Phase plot (position vs velocity in X)
    ax9 = fig.add_subplot(gs[2, 2])
    ee_vel_x = np.gradient(ee_pos_data[0, :], time_data)
    ax9.plot(ee_pos_data[0, :] - ee_pos_data[0, 0], ee_vel_x, 'blue', linewidth=2)
    ax9.plot(ee_pos_data[0, 0] - ee_pos_data[0, 0], ee_vel_x[0], 'go', markersize=10, label='Start')
    ax9.plot(ee_pos_data[0, -1] - ee_pos_data[0, 0], ee_vel_x[-1], 'ro', markersize=10, label='End')
    ax9.set_xlabel('X Displacement [m]', fontweight='bold')
    ax9.set_ylabel('X Velocity [m/s]', fontweight='bold')
    ax9.set_title('Phase Plot (X-axis)', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    plt.suptitle('Admittance Control: Force-Driven Compliant Motion', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print(colored("✓ Plots generated", "green"))
    
    # Print summary statistics
    print(colored(f"\n📊 Summary Statistics:", "yellow", attrs=["bold"]))
    print(colored(f"  Maximum force: {np.max(force_magnitude):.2f} N", "cyan"))
    print(colored(f"  Maximum displacement: {np.max(displacement):.4f} m ({np.max(displacement)*1000:.2f} mm)", "cyan"))
    print(colored(f"  Maximum tracking error: {np.max(error_magnitude)*1000:.2f} mm", "cyan"))
    print(colored(f"  Final EE position: X={ee_pos_data[0, -1]:.4f} m, Y={ee_pos_data[1, -1]:.4f} m", "cyan"))
    print(colored(f"  EE displacement: ΔX={ee_pos_data[0, -1] - ee_pos_data[0, 0]:.4f} m, ΔY={ee_pos_data[1, -1] - ee_pos_data[1, 0]:.4f} m", "cyan"))
    
    return {
        'time': time_data,
        'q1': q1,
        'q2': q2,
        'ee_position': ee_pos_data,
        'desired_position': des_pos_data,
        'force': force_data,
        'displacement': displacement,
    }


def main():
    """Main function with different test scenarios."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test admittance control with external forces')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration [s]')
    parser.add_argument('--force', type=float, default=10.0, help='Maximum force [N]')
    parser.add_argument('--direction', type=str, default='x', choices=['x', 'y', 'xy'],
                       help='Force direction')
    parser.add_argument('--mass', type=float, default=2.0, help='Virtual mass [kg]')
    parser.add_argument('--damping', type=float, default=5.0, help='Virtual damping [N·s/m]')
    parser.add_argument('--stiffness', type=float, default=10.0, help='Virtual stiffness [N/m]')
    
    args = parser.parse_args()
    
    # Parse force direction
    if args.direction == 'x':
        force_dir = np.array([1.0, 0.0])
    elif args.direction == 'y':
        force_dir = np.array([0.0, 1.0])
    else:  # xy
        force_dir = np.array([1.0, 1.0])
    
    # Run test
    result = test_admittance_control(
        duration=args.duration,
        F_max=args.force,
        force_direction=force_dir,
        M_virtual=args.mass,
        D_virtual=args.damping,
        K_virtual=args.stiffness,
    )
    
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("ADMITTANCE CONTROL TEST COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    
    input(colored("\nPress Enter to close Meshcat and exit...", "yellow"))


if __name__ == "__main__":
    main()
