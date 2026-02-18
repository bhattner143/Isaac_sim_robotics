"""
Joint-Space Optimal Feedback Control Implementation
Based on Section C.2 from Razavian et al. (2021)

This implements the simplified joint-space formulation with:
- d = 0 (no disturbance)
- ε = 0 (no error dynamics)  
- ω = 0 (no oscillator dynamics)

The core equations are:
1. Impedance dynamics (eq. C.13-C.15):
   Ma·ÿ + kp·(y - y_zf) + kd·(ẏ - ẏ_zf) = 0
   
2. Robot dynamics:
   M(q)·q̈ + C(q,v)·v + G(q) = τ
   
3. Control law:
   τ = -K·(x - x_desired)
   
Where:
- y: actual joint position (from plant)
- y_zf: zero-force trajectory position
- Ma: virtual/impedance mass
- kp, kd: impedance spring and damper
- τ: control torque
- K: LQR gain matrix

OPTIMIZATION FORMULATION:
────────────────────────────────────────────────────────────────────────────
This implementation uses CONTINUOUS-TIME INFINITE-HORIZON LQR:
   min ∫₀^∞ [x^T·Q·x + u^T·R·u] dt
   s.t. ẋ = A·x + B·u

This differs from DISCRETE-TIME FINITE-HORIZON (typical trajectory optimization):
   J = Σₜ₌₀^(N-1) [xₜ^T·Qₜ·xₜ + uₜ^T·Rₜ·uₜ] + x_N^T·Q_N·x_N

Key differences:
- No terminal cost Q_N (infinite horizon → steady-state regulator)
- Solves continuous algebraic Riccati equation (CARE), not Riccati recursion
- Time-invariant gains K (not time-varying Kₜ)
- Suitable for stabilization, not finite-time trajectory tracking

For finite-horizon trajectory optimization with Q_N, use DirectCollocation or DDP.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from pydrake.all import LeafSystem, BasicVector, LinearQuadraticRegulator
from termcolor import colored


class JointSpaceOFC(LeafSystem):
    """
    Simplified Joint-Space Optimal Feedback Controller.
    
    Section C.2 Implementation with d=ε=ω=0:
    
    State vector (for each joint):
    - Effort mode: x = [q, q̇, F, y_zf, ẏ_zf]
      where F is the driving force
      
    - Smoothness mode: x = [q, q̇, y_zf, ẏ_zf, ÿ_zf]
      where jerk is the control input
    
    Dynamics:
    1. Robot: q̈ = M^{-1}·(τ - C·v - G)
    2. Impedance: Ma·ÿ = kp·(y_zf - y) + kd·(ẏ_zf - ẏ) + F
    3. Zero-force trajectory dynamics (depends on mode)
    
    Control:
    τ = kp·(y_zf - y) + kd·(ẏ_zf - ẏ) + F
    
    where (y_zf, ẏ_zf, F or ÿ_zf) come from LQR optimization.
    """
    
    def __init__(self, plant, q_start, q_goal, duration,
                 mode='effort',
                 Q_position=None, Q_velocity=None, R=None,
                 Q_pendulum=None, Q_pendulum_vel=None,
                 Ma=1.0, kp=100.0, kd=20.0, tau_filter=0.01,
                 include_pendulum=True):
        """
        Initialize joint-space OFC with optional pendulum states.
        
        Args:
            plant: MultibodyPlant for dynamics
            q_start: Initial configuration [4] (2 manip + 2 pendulum)
            q_goal: Goal configuration [4]
            duration: Motion duration [s]
            mode: 'effort' or 'smoothness'
            Q_position: State cost for manipulator positions [2]
            Q_velocity: State cost for manipulator velocities [2]
            Q_pendulum: State cost for pendulum angles [2]
            Q_pendulum_vel: State cost for pendulum velocities [2]
            R: Control cost [2]
            Ma: Virtual mass [kg]
            kp: Spring stiffness [N/m]
            kd: Damping coefficient [N·s/m]
            tau_filter: Time constant for F-dot filter [s]
            include_pendulum: Include pendulum states in LQR (True = full state)
        """
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.mode = mode
        self.include_pendulum = include_pendulum
        self.q_start = np.array(q_start)
        self.q_goal = np.array(q_goal)
        self.duration = duration
        
        # Impedance parameters (eq. C.13-C.15)
        self.Ma = Ma
        self.kp = kp
        self.kd = kd
        self.tau_filter = tau_filter  # F-dot filter time constant
        
        # Cost matrices
        self.Q_pos = Q_position if Q_position is not None else np.array([100.0, 100.0])
        self.Q_vel = Q_velocity if Q_velocity is not None else np.array([10.0, 10.0])
        self.Q_pend = Q_pendulum if Q_pendulum is not None else np.array([500.0, 500.0])
        self.Q_pend_vel = Q_pendulum_vel if Q_pendulum_vel is not None else np.array([50.0, 50.0])
        self.R = R if R is not None else np.array([0.1, 0.1])
        
        # Dimensions
        self.n_joints = 2  # Manipulator actuators
        self.n_pos = plant.num_positions()  # 4 (2 arm + 2 pendulum)
        self.n_vel = plant.num_velocities()  # 4
        
        # Input port: state from plant
        self.DeclareVectorInputPort("state", BasicVector(self.n_pos + self.n_vel))
        
        # Output port: control torques
        self.DeclareVectorOutputPort("control", BasicVector(self.n_joints),
                                     self.CalcControl)
        
        # Zero-force trajectory state (per joint)
        # Effort mode: [F, y_zf, ẏ_zf] per joint → 6 states
        # Smoothness mode: [y_zf, ẏ_zf, ÿ_zf] per joint → 6 states  
        self.zf_state_dim = 3 * self.n_joints
        
        # Initialize zero-force trajectory states
        self.y_zf = self.q_start[:2].copy()
        self.ydot_zf = np.zeros(2)
        self.F = np.zeros(2)  # Driving forces (effort mode)
        self.yddot_zf = np.zeros(2)  # Accelerations (smoothness mode)
        
        # Linearize and compute LQR gains
        self._compute_lqr_gains()
        
        state_type = "FULL-STATE (manip + pendulum)" if include_pendulum else "MANIPULATOR-ONLY"
        print(colored(f"✓ Joint-Space OFC Initialized (Section C.2)", 'green', attrs=['bold']))
        print(colored(f"  Type: {state_type}", 'green'))
        print(colored(f"  Mode: {mode}", 'green'))
        print(colored(f"  Impedance: Ma={Ma}, kp={kp}, kd={kd}", 'green'))
        print(colored(f"  F-dot filter: τ={tau_filter}s", 'green'))
        print(colored(f"  LQR gain K: {self.K.shape}", 'green'))
    
    def _compute_lqr_gains(self):
        """
        Compute LQR gains for the augmented system.
        
        For joint space with impedance dynamics (eq. C.13-C.15):
        
        State (manipulator-only): x = [q_m, q̇_m, (F or ÿ_zf), y_zf, ẏ_zf] per joint
        State (full): x = [q_m, q_p, q̇_m, q̇_p, (F or ÿ_zf), y_zf, ẏ_zf] per joint
        
        Linearized dynamics around equilibrium:
        ẋ = A·x + B·u
        
        where u = [F_dot] for effort mode or [y_zf_jerk] for smoothness mode
        """
        print(colored(f"\n⏳ Computing LQR gains for joint-space OFC...", 'yellow'))
        
        state_type = "FULL-STATE (4 DOF)" if self.include_pendulum else "MANIPULATOR-ONLY (2 DOF)"
        print(colored(f"  Configuration: {state_type}", 'yellow'))
        
        # Get equilibrium dynamics
        context = self.plant.CreateDefaultContext()
        
        # Build full equilibrium configuration based on actual number of DOFs
        # The actual configuration depends on plant structure (could be 2, 4, or more DOF)
        n_pos = self.plant.num_positions()
        n_vel = self.plant.num_velocities()
        
        # Start with zeros
        q_eq = np.zeros(n_pos)
        v_eq = np.zeros(n_vel)
        
        # Set full goal configuration
        q_eq[:] = self.q_goal  # Use full q_goal (4 elements: 2 manip + 2 pend)
        
        # EXPLANATION OF EXTRA DOFs (11 positions, 10 velocities):
        # The asymmetry indicates quaternion parameterization of floating bodies.
        # Typical breakdown when base is not welded:
        #   [0:2]   - Manipulator joints (2 revolute) = 2 pos, 2 vel
        #   [2:5]   - Base translation (x,y,z) = 3 pos, 3 vel  
        #   [5:9]   - Base orientation (quaternion) = 4 pos, 3 vel (angular velocity)
        #   [9:11]  - Pendulum (pitch, roll) = 2 pos, 2 vel
        #   TOTAL: 11 positions, 10 velocities
        #
        # NOTE: This controller was designed for 4-DOF (2 arm + 2 pendulum).
        # The extra 7 floating-base DOFs are uncontrolled, causing poor tracking.
        # FIX: Weld the manipulator base to world in the setup script.
        
        # If there are pendulum joints (positions 2-3 for 4-DOF system, more for gimbal)
        # Set them to hanging down - but only if they exist
        if n_pos == 4:
            q_eq[2] = 0.0  # pitch
            q_eq[3] = np.deg2rad(180.0)  # roll
        elif n_pos > 4:
            # For systems with floating base or more complex structure
            # Leave extra DOFs as zeros (may need adjustment for quaternions)
            pass
        
        print(colored(f"  Equilibrium config (n_pos={n_pos}): q_eq = {q_eq}", 'cyan'))
        
        self.plant.SetPositions(context, q_eq)
        self.plant.SetVelocities(context, v_eq)
        
        # Get full mass matrix and inverse
        M_full = self.plant.CalcMassMatrix(context)
        
        # Get Coriolis and gravity terms at equilibrium
        C_full = self.plant.CalcBiasTerm(context)  # Includes Coriolis + gravity
        
        print(colored(f"  Mass matrix M_full:", 'cyan'))
        print(colored(f"    shape = {M_full.shape}", 'cyan'))
        print(colored(f"    diag(M) = {np.diag(M_full)}", 'cyan'))
        
        # Build augmented state-space model
        if self.mode == 'effort':
            A, B = self._build_effort_mode_dynamics(M_full, C_full)
        else:
            A, B = self._build_smoothness_mode_dynamics(M_full, C_full)
        
        # ═══════════════════════════════════════════════════════════════════════
        # LQR COST MATRIX DESIGN - FULL STATE WITH PENDULUM
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.include_pendulum:
            # FULL-STATE (PAPER-INSPIRED STRUCTURE): 
            # Primary objective: PENDULUM stabilization (task-space goal)
            # Secondary: Minimal damping for detectability
            #
            # Note: Paper uses task-space OFC where object is directly controlled.
            # In joint-space, pendulum is passive, so we need small velocity weights
            # to ensure (Q,A) detectability for LQR solvability.
            #
            # State: [q_m1, q_m2, q_p_pitch, q_p_roll, v_m1, v_m2, v_p_pitch, v_p_roll,
            #         F1/ÿ_zf1, y_zf1, ẏ_zf1, F2/ÿ_zf2, y_zf2, ẏ_zf2]  (14-dim)
            Q = np.diag([
                0.0,                    # q_m1: Manipulator position (not penalized)
                0.0,                    # q_m2: Manipulator position (not penalized)
                self.Q_pend[0],         # q_p_pitch: PENDULUM ANGLE (primary objective!)
                self.Q_pend[1],         # q_p_roll: PENDULUM ANGLE (primary objective!)
                self.Q_vel[0] * 0.01,   # v_m1: Minimal damping for detectability
                self.Q_vel[1] * 0.01,   # v_m2: Minimal damping for detectability
                self.Q_pend_vel[0],     # v_p_pitch: Pendulum velocity damping
                self.Q_pend_vel[1],     # v_p_roll: Pendulum velocity damping
                0.0,                    # F1 or ÿ_zf1: No force penalty
                0.0,                    # y_zf1: No ZFT position penalty
                0.0,                    # ẏ_zf1: No ZFT velocity penalty
                0.0,                    # F2 or ÿ_zf2: No force penalty
                0.0,                    # y_zf2: No ZFT position penalty
                0.0,                    # ẏ_zf2: No ZFT velocity penalty
            ])
            print(colored(f"  Q (OBJECT-CENTRIC STRUCTURE):", 'cyan'))
            print(colored(f"    PRIMARY: Pendulum angle Q_pend = {self.Q_pend}", 'cyan'))
            print(colored(f"    DAMPING: Pendulum velocity Q_pend_vel = {self.Q_pend_vel}", 'cyan'))
            print(colored(f"    SECONDARY: Manipulator damping (0.01×Q_vel) for detectability", 'cyan'))
        else:
            # MANIPULATOR-ONLY: [q_m1, q_m2, v_m1, v_m2, F1/ÿ_zf1, y_zf1, ẏ_zf1, F2/ÿ_zf2, y_zf2, ẏ_zf2]  (10-dim)
            Q = np.diag([
                self.Q_pos[0],  # q₁: Position tracking (joint 1)
                self.Q_pos[1],  # q₂: Position tracking (joint 2)
                self.Q_vel[0],  # q̇₁: Velocity damping (joint 1)
                self.Q_vel[1],  # q̇₂: Velocity damping (joint 2)
                1.0,            # F₁ or ÿ_zf₁: Effort/smoothness penalty (joint 1)
                self.Q_pos[0],  # y_zf₁: Zero-force trajectory position (joint 1)
                self.Q_vel[0],  # ẏ_zf₁: Zero-force trajectory velocity (joint 1)
                1.0,            # F₂ or ÿ_zf₂: Effort/smoothness penalty (joint 2)
                self.Q_pos[1],  # y_zf₂: Zero-force trajectory position (joint 2)
                self.Q_vel[1],  # ẏ_zf₂: Zero-force trajectory velocity (joint 2)
            ])
            print(colored(f"  Q (manip-only): diag([Q_manip, Q_vel_manip, Q_ZFT])", 'cyan'))
        
        R_mat = np.diag(self.R)  # Control cost: penalizes u = [Ḟ₁, Ḟ₂] or [jerk₁, jerk₂]
        
        print(colored(f"  A: {A.shape}, B: {B.shape}", 'cyan'))
        print(colored(f"  Q: diag({np.diag(Q)})", 'cyan'))
        print(colored(f"  R: diag({self.R})", 'cyan'))
        
        # Solve LQR
        self.K, S = LinearQuadraticRegulator(A, B, Q, R_mat)
        
        print(colored(f"✓ LQR solved", 'green'))
        print(colored(f"  K: {self.K.shape}", 'green'))
        print(colored(f"  Position feedback: {self.K[:, 0:2]}", 'green'))
        print(colored(f"  Velocity feedback: {self.K[:, 2:4]}", 'green'))
    
    def _build_effort_mode_dynamics(self, M_full, C_full):
        """
        Build linearized dynamics for effort-minimizing mode with F-dot filter.
        
        Section C.2 equations with F-dot first-order filter:
        
        F-dot filter (from paper screenshot):
           Ḟ = -(1/τ)·F + (1/τ)·u
        
        State (manipulator-only): x = [q_m, v_m, F, y_zf, ẏ_zf] → 10-dim
        State (full): x = [q_m, q_p, v_m, v_p, F, y_zf, ẏ_zf] → 14-dim
        Control: u = [u1, u2] (desired force rate)
        
        Dynamics:
        1. Robot: M·q̈ + C = τ
        2. Control: τ = [kp·(y_zf - q_m) + kd·(ẏ_zf - v_m) + F; 0]
        3. F-dot filter: Ḟ = -(1/τ)·F + (1/τ)·u
        4. ZFT: ẏ_zf = ẏ_zf, ÿ_zf = 0
        """
        if self.include_pendulum:
            dim = 14  # [q_m(2), q_p(2), v_m(2), v_p(2), ZFT(6)]
        else:
            dim = 10  # [q_m(2), v_m(2), ZFT(6)]
        
        A = np.zeros((dim, dim))
        B = np.zeros((dim, 2))
        
        # Compute M_inv for dynamics
        M_inv = np.linalg.inv(M_full)
        
        if self.include_pendulum:
            # FULL STATE: [q_m1, q_m2, q_p1, q_p2, v_m1, v_m2, v_p1, v_p2, F1, y_zf1, ẏ_zf1, F2, y_zf2, ẏ_zf2]
            # Position derivatives: q̇ = v
            A[0:4, 4:8] = np.eye(4)
            
            # Velocity derivatives: v̇ = M^{-1}·(τ - C)
            # τ = [kp·(y_zf - q_m) + kd·(ẏ_zf - v_m) + F; 0, 0]
            
            for i in range(2):  # Manipulator joints
                # Indices
                q_m_i = i  # 0, 1
                v_m_i = 4 + i  # 4, 5
                F_i = 8 + 3*i  # 8, 11
                y_zf_i = 9 + 3*i  # 9, 12
                ydot_zf_i = 10 + 3*i  # 10, 13
                
                # Torque on manipulator: τ_i = kp·(y_zf - q_m) + kd·(ẏ_zf - v_m) + F
                # Acceleration: v̇ = M_inv·[τ; 0]
                
                # Effect on manipulator velocities (coupled through M_inv)
                for j in range(4):
                    v_j = 4 + j
                    A[v_j, q_m_i] += M_inv[j, i] * (-self.kp)
                    A[v_j, v_m_i] += M_inv[j, i] * (-self.kd)
                    A[v_j, y_zf_i] += M_inv[j, i] * self.kp
                    A[v_j, ydot_zf_i] += M_inv[j, i] * self.kd
                    A[v_j, F_i] += M_inv[j, i]
                
                # F-dot filter: Ḟ = -(1/τ)·F + (1/τ)·u
                A[F_i, F_i] = -1.0 / self.tau_filter
                B[F_i, i] = 1.0 / self.tau_filter
                
                # ZFT dynamics
                A[y_zf_i, ydot_zf_i] = 1.0  # ẏ_zf = ẏ_zf
                # ÿ_zf = 0 (no dynamics)
        else:
            # MANIPULATOR-ONLY: [q_m1, q_m2, v_m1, v_m2, F1, y_zf1, ẏ_zf1, F2, y_zf2, ẏ_zf2]
            M = M_full[0:2, 0:2]
            M_inv_manip = np.linalg.inv(M)
            
            # Position derivatives
            A[0:2, 2:4] = np.eye(2)
            
            for i in range(2):
                q_m_i = i
                v_m_i = 2 + i
                F_i = 4 + 3*i
                y_zf_i = 5 + 3*i
                ydot_zf_i = 6 + 3*i
                
                # Velocity derivatives (only manipulator)
                for j in range(2):
                    v_j = 2 + j
                    A[v_j, q_m_i] += M_inv_manip[j, i] * (-self.kp)
                    A[v_j, v_m_i] += M_inv_manip[j, i] * (-self.kd)
                    A[v_j, y_zf_i] += M_inv_manip[j, i] * self.kp
                    A[v_j, ydot_zf_i] += M_inv_manip[j, i] * self.kd
                    A[v_j, F_i] += M_inv_manip[j, i]
                
                # F-dot filter
                A[F_i, F_i] = -1.0 / self.tau_filter
                B[F_i, i] = 1.0 / self.tau_filter
                
                # ZFT dynamics
                A[y_zf_i, ydot_zf_i] = 1.0
        
        return A, B
    
    def _build_smoothness_mode_dynamics(self, M_full, C_full):
        """
        Build linearized dynamics for smoothness-minimizing mode.
        
        State (manipulator-only): x = [q_m, v_m, ÿ_zf, y_zf, ẏ_zf] → 10-dim
        State (full): x = [q_m, q_p, v_m, v_p, ÿ_zf, y_zf, ẏ_zf] → 14-dim
        Control: u = [y_zf_jerk1, y_zf_jerk2]
        
        Dynamics:
           q̇ = v
           v̇ = M^{-1}·(τ - C)
           τ = [kp·(y_zf - q_m) + kd·(ẏ_zf - v_m); 0]
           y_zf_jerk = u (control)
           ẏ_zf = ẏ_zf + ÿ_zf·dt
           ÿ_zf = ÿ_zf + jerk·dt
        """
        n_q = 4 if self.include_pendulum else 2
        n_zft = 6  # 3 states per ZFT joint (2 manipulator joints)
        dim = n_q + n_zft  # Total: 4+6=10 (manip-only) or 4+6=10... WAIT NO!
        # For full state: [q_m, q_p, v_m, v_p, ZFT] = [2, 2, 2, 2, 6] = 14
        # For manip-only: [q_m, v_m, ZFT] = [2, 2, 6] = 10
        
        if self.include_pendulum:
            dim = 14  # [q_m(2), q_p(2), v_m(2), v_p(2), ZFT(6)]
        else:
            dim = 10  # [q_m(2), v_m(2), ZFT(6)]
        
        A = np.zeros((dim, dim))
        B = np.zeros((dim, 2))
        
        M_inv = np.linalg.inv(M_full)
        
        if self.include_pendulum:
            # Position derivatives
            A[0:4, 4:8] = np.eye(4)
            
            for i in range(2):
                q_m_i = i
                v_m_i = 4 + i
                yddot_zf_i = 8 + 3*i
                y_zf_i = 9 + 3*i
                ydot_zf_i = 10 + 3*i
                
                # Velocity derivatives (coupled through M_inv)
                for j in range(4):
                    v_j = 4 + j
                    A[v_j, q_m_i] += M_inv[j, i] * (-self.kp)
                    A[v_j, v_m_i] += M_inv[j, i] * (-self.kd)
                    A[v_j, y_zf_i] += M_inv[j, i] * self.kp
                    A[v_j, ydot_zf_i] += M_inv[j, i] * self.kd
                
                # ZFT jerk control: y_zf_jerk = u
                B[yddot_zf_i, i] = 1.0
                
                # ZFT integration
                A[y_zf_i, ydot_zf_i] = 1.0  # ẏ_zf = ẏ_zf
                A[ydot_zf_i, yddot_zf_i] = 1.0  # ÿ_zf derivative contributes to ẏ_zf
        else:
            M = M_full[0:2, 0:2]
            M_inv_manip = np.linalg.inv(M)
            
            # Position derivatives
            A[0:2, 2:4] = np.eye(2)
            
            for i in range(2):
                q_m_i = i
                v_m_i = 2 + i
                yddot_zf_i = 4 + 3*i
                y_zf_i = 5 + 3*i
                ydot_zf_i = 6 + 3*i
                
                # Velocity derivatives
                for j in range(2):
                    v_j = 2 + j
                    A[v_j, q_m_i] += M_inv_manip[j, i] * (-self.kp)
                    A[v_j, v_m_i] += M_inv_manip[j, i] * (-self.kd)
                    A[v_j, y_zf_i] += M_inv_manip[j, i] * self.kp
                    A[v_j, ydot_zf_i] += M_inv_manip[j, i] * self.kd
                
                # ZFT jerk control
                B[yddot_zf_i, i] = 1.0
                
                # ZFT integration
                A[y_zf_i, ydot_zf_i] = 1.0
                A[ydot_zf_i, yddot_zf_i] = 1.0
        
        return A, B
    
    def CalcControl(self, context, output):
        """
        Compute optimal control torque.
        
        Control law (eq. C.13-C.15):
        τ = kp·(y_zf - y) + kd·(ẏ_zf - ẏ) + F  (effort mode)
        τ = kp·(y_zf - y) + kd·(ẏ_zf - ẏ)     (smoothness mode)
        
        where (y_zf, ẏ_zf, F or ÿ_zf) are optimized via LQR.
        """
        # Get current state from plant
        state = self.get_input_port(0).Eval(context)
        q = state[0:self.n_pos]
        v = state[self.n_pos:]
        
        # Extract manipulator and pendulum states
        q_manip = q[0:2]
        v_manip = v[0:2]
        
        if self.include_pendulum:
            q_pend = q[2:4]
            v_pend = v[2:4]
            
            # Build full augmented state vector (14-dim)
            if self.mode == 'effort':
                x_aug = np.array([
                    q_manip[0], q_manip[1], q_pend[0], q_pend[1],  # Positions
                    v_manip[0], v_manip[1], v_pend[0], v_pend[1],  # Velocities
                    self.F[0], self.y_zf[0], self.ydot_zf[0],      # ZFT joint 1
                    self.F[1], self.y_zf[1], self.ydot_zf[1]       # ZFT joint 2
                ])
            else:  # smoothness
                x_aug = np.array([
                    q_manip[0], q_manip[1], q_pend[0], q_pend[1],
                    v_manip[0], v_manip[1], v_pend[0], v_pend[1],
                    self.yddot_zf[0], self.y_zf[0], self.ydot_zf[0],
                    self.yddot_zf[1], self.y_zf[1], self.ydot_zf[1]
                ])
            
            # Desired state (goal equilibrium)
            x_desired = np.array([
                self.q_goal[0], self.q_goal[1],  # Manipulator at goal
                0.0, np.deg2rad(180.0),           # Pendulum hanging down (pitch=0, roll=180°)
                0.0, 0.0, 0.0, 0.0,              # Zero velocities
                0.0, self.q_goal[0], 0.0,        # ZFT joint 1: F=0 or ÿ=0, y=goal, ẏ=0
                0.0, self.q_goal[1], 0.0         # ZFT joint 2
            ])
        else:
            # Build manipulator-only augmented state (10-dim)
            if self.mode == 'effort':
                x_aug = np.array([
                    q_manip[0], q_manip[1],
                    v_manip[0], v_manip[1],
                    self.F[0], self.y_zf[0], self.ydot_zf[0],
                    self.F[1], self.y_zf[1], self.ydot_zf[1]
                ])
            else:
                x_aug = np.array([
                    q_manip[0], q_manip[1],
                    v_manip[0], v_manip[1],
                    self.yddot_zf[0], self.y_zf[0], self.ydot_zf[0],
                    self.yddot_zf[1], self.y_zf[1], self.ydot_zf[1]
                ])
            
            # Desired state
            x_desired = np.array([
                self.q_goal[0], self.q_goal[1],
                0.0, 0.0,
                0.0, self.q_goal[0], 0.0,
                0.0, self.q_goal[1], 0.0
            ])
        
        # LQR control: u = -K·(x - x_desired)
        x_error = x_aug - x_desired
        u_opt = -self.K @ x_error
        
        # Update internal states (integrate control)
        dt = 0.001  # Match discrete update rate
        
        if self.mode == 'effort':
            # u = [u1, u2] (desired force rate from filter input)
            # F-dot filter: Ḟ = -(1/τ)·F + (1/τ)·u
            # Already built into A matrix, but we integrate here for internal state
            F_dot = (-1.0/self.tau_filter) * self.F + (1.0/self.tau_filter) * u_opt
            self.F += F_dot * dt
        else:
            # u = [y_zf_jerk1, y_zf_jerk2]
            self.yddot_zf += u_opt * dt
        
        # Integrate ZFT dynamics
        self.y_zf += self.ydot_zf * dt
        if self.mode == 'smoothness':
            self.ydot_zf += self.yddot_zf * dt
        
        # Compute control torque (impedance law)
        # τ = kp·(y_zf - q) + kd·(ẏ_zf - q̇) + F
        tau = self.kp * (self.y_zf - q_manip) + self.kd * (self.ydot_zf - v_manip)
        
        if self.mode == 'effort':
            tau += self.F
        
        output.SetFromVector(tau)
    
    def _update_internal_state(self, context, discrete_state):
        """Update internal zero-force trajectory states."""
        # This is called periodically but actual integration happens in CalcControl
        # Just maintain state consistency
        pass


def demonstrate_joint_space_ofc():
    """
    Demonstration of joint-space OFC implementation.
    Shows the key equations from Section C.2.
    """
    print("="*70)
    print("JOINT-SPACE OPTIMAL FEEDBACK CONTROL (Section C.2)")
    print("Simplified formulation with d=ε=ω=0")
    print("="*70)
    
    print("\n📐 Key Equations:")
    print("\n1. Impedance Dynamics (eq. C.13-C.15):")
    print("   Ma·ÿ + kp·(y - y_zf) + kd·(ẏ - ẏ_zf) = 0")
    print("   where:")
    print("   - y: actual joint position")
    print("   - y_zf: zero-force trajectory")
    print("   - Ma: virtual mass")
    print("   - kp, kd: spring and damper coefficients")
    
    print("\n2. Control Law:")
    print("   τ = kp·(y_zf - y) + kd·(ẏ_zf - ẏ) + F")
    print("   where F is optimized via LQR in effort mode")
    
    print("\n3. State-Space Form:")
    print("   Effort mode: x = [q, q̇, F, y_zf, ẏ_zf]")
    print("   Smoothness mode: x = [q, q̇, ÿ_zf, y_zf, ẏ_zf]")
    
    print("\n4. LQR Optimization:")
    print("   min ∫[x^T·Q·x + u^T·R·u] dt")
    print("   subject to: ẋ = A·x + B·u")
    
    print("\n✓ Implementation complete!")
    print("  Use JointSpaceOFC class in your Drake simulation")


if __name__ == "__main__":
    demonstrate_joint_space_ofc()
