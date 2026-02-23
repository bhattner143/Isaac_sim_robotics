"""
Utility functions for cart-pendulum & manipulator simulations.

Moved from script_cup_manipulator_pendulam_lqr_min_effort_2d.py to reduce
file size and improve reusability.

Functions
---------
- build_linearized_system_2d
- build_linearized_for_complete_system_2d
- check_trajectory_feasibility
- test_and_visualize_ik_feasibility
"""

import numpy as np
from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    RigidTransform,
    JacobianWrtVariable,
)
from pydrake.systems.primitives import Linearize


# ============================================================================
# LINEARIZATION FUNCTIONS
# ============================================================================

def build_linearized_system_2d(
    physics_config,
    impedance_config,
    zft_config,
    muscle_config,
):
    """
    Build linearized 14D system matrices.

    Uses Drake's Linearize() for cart-pendulum (8D), then assembles
    with muscle dynamics (2D) and ZFT dynamics (4D).

    Parameters
    ----------
    physics_config : CartPendulumPhysicsConfig
    impedance_config : ImpedanceForceConfig
    zft_config : ZFTReferenceMassConfig
    muscle_config : MuscleDynamicsConfig

    Returns
    -------
    A : np.ndarray, shape (14, 14)
    B : np.ndarray, shape (14, 2)
    """
    # Lazy import to avoid circular dependency at module load time.
    from script_cup_manipulator_pendulam_lqr_min_effort_2d import (
        CartPendulum2DExtended,
        create_cart_pendulum_config,
    )

    K_imp = impedance_config.K_imp
    D_imp = impedance_config.D_imp
    M_ref = zft_config.M_ref
    muscle_tau = muscle_config.muscle_tau
    M_cart = physics_config.mass_cart

    # Temporary continuous-time plant for linearization.
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)
    temp_model = temp_plant.AddModelInstance("cart_temp")

    temp_cart = CartPendulum2DExtended(physics_config)
    temp_cart.build_plant(temp_plant, temp_model, register_visuals=False)

    temp_plant.Finalize()
    temp_builder.AddSystem(temp_plant)
    temp_diagram = temp_builder.Build()

    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_plant.GetMyContextFromRoot(temp_context)

    temp_plant.SetPositions(temp_plant_context, np.zeros(4))
    temp_plant.SetVelocities(temp_plant_context, np.zeros(4))

    input_port = temp_plant.get_actuation_input_port()
    input_port.FixValue(temp_plant_context, np.zeros(2))
    output_port = temp_plant.get_state_output_port()

    linear_sys = Linearize(
        temp_plant, temp_plant_context,
        input_port_index=input_port.get_index(),
        output_port_index=output_port.get_index(),
    )

    A_cp = linear_sys.A()
    B_cp = linear_sys.B()

    # Muscle dynamics (2D): Ḟ = (-F + u) / τ
    A_muscle = -np.eye(2) / muscle_tau
    B_muscle = np.eye(2) / muscle_tau

    # Assemble 14×14 A matrix
    A = np.zeros((14, 14))

    # Cart-pendulum block (8×8)
    A[0:8, 0:8] = A_cp

    # Coupling: cart-pendulum affected by impedance force
    # ẍ += (K*(x_ref - x) + D*(ẋ_ref - ẋ) + F_x) / M
    # ÿ += (K*(y_ref - y) + D*(ẏ_ref - ẏ) + F_y) / M
    A[4, 0]  = -K_imp / M_cart   # ẍ ← -K*x/M
    A[4, 4]  = -D_imp / M_cart   # ẍ ← -D*ẋ/M
    A[4, 8]  =  1.0 / M_cart     # ẍ ← F_x/M
    A[4, 10] =  K_imp / M_cart   # ẍ ← K*x_ref/M
    A[4, 12] =  D_imp / M_cart   # ẍ ← D*ẋ_ref/M

    A[5, 1]  = -K_imp / M_cart   # ÿ ← -K*y/M
    A[5, 5]  = -D_imp / M_cart   # ÿ ← -D*ẏ/M
    A[5, 9]  =  1.0 / M_cart     # ÿ ← F_y/M
    A[5, 11] =  K_imp / M_cart   # ÿ ← K*y_ref/M
    A[5, 13] =  D_imp / M_cart   # ÿ ← D*ẏ_ref/M

    # Muscle dynamics block (2×2)
    A[8:10, 8:10] = A_muscle

    # ZFT dynamics block (4×4)
    A[10, 12] = 1.0
    A[11, 13] = 1.0

    A[12, 0]  =  K_imp / M_ref
    A[12, 4]  =  D_imp / M_ref
    A[12, 8]  =  1.0 / M_ref
    A[12, 10] = -K_imp / M_ref
    A[12, 12] = -D_imp / M_ref

    A[13, 1]  =  K_imp / M_ref
    A[13, 5]  =  D_imp / M_ref
    A[13, 9]  =  1.0 / M_ref
    A[13, 11] = -K_imp / M_ref
    A[13, 13] = -D_imp / M_ref

    # Assemble 14×2 B matrix
    B = np.zeros((14, 2))
    B[8:10, 0:2] = B_muscle

    return A, B


def build_linearized_for_complete_system_2d(
    plant: MultibodyPlant,
    manipulator,
    cart_model,
    physics_config,
    muscle_config=None,
    zft_config=None,
    Kp_ct: float = 200.0,
    Kd_ct: float = 60.0,
):
    """
    Build linearized system matrices for the welded cart-pendulum-manipulator.

    Without optional configs → returns 8D plant matrices (backward compat).
    With muscle_config + zft_config → returns augmented 14D matrices:

        x = [q_arm(2), q_pend(2), v_arm(2), v_pend(2),   ← plant (8D)
              F_x, F_y,                                    ← muscle (2D)
              x_ref, y_ref, ẋ_ref, ẏ_ref]                 ← ZFT ref (4D)

    The 14D augmentation appends:
      • Muscle dynamics  : Ḟ = -F/τ + u/τ
      • ZFT dynamics     : p̈_ref = K/M*(J·q_arm - p_ref) + D/M*(J·v_arm - ṗ_ref) + F/M
      • CT coupling      : adds Kp/Kd joint-space stiffness/damping through J_inv
        (captures the computed-torque controller feedback in the linear model)

    Parameters
    ----------
    plant : MultibodyPlant
    manipulator : CupManipulator
    cart_model : ModelInstanceIndex
    physics_config : CartPendulumPhysicsConfig
    muscle_config : MuscleDynamicsConfig, optional
    zft_config : ZFTReferenceMassConfig, optional
    Kp_ct : float  — computed-torque proportional gain (must match controller)
    Kd_ct : float  — computed-torque derivative gain (must match controller)

    Returns
    -------
    A : np.ndarray, shape (8, 8)  or  (14, 14)
    B : np.ndarray, shape (8, 2)  or  (14, 2)
    """
    # Lazy imports to avoid circular dependency at module load time.
    from script_cup_manipulator_pendulam_lqr_min_effort_2d import (
        CupManipulator,
        CartPendulum2DExtended,
    )

    # Temporary continuous-time plant (no SceneGraph needed).
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)

    # Rebuild the welded system in the temporary plant.
    temp_parser = Parser(temp_plant)
    temp_manip = CupManipulator(manipulator.config, enable_visualization=False)
    temp_manip.load_urdf_to_plant(temp_plant, temp_parser)
    temp_manip.weld_base_to_world(temp_plant, orientation=np.array([0.0, 0.0, 0.0]))
    temp_manip.add_joint_actuators(temp_plant)
    temp_manip.add_end_effector_frame(temp_plant)

    temp_cart_model = temp_plant.AddModelInstance("cart_pendulum")
    temp_cart_pend = CartPendulum2DExtended(physics_config, z_offset=0.0)
    cart_body = temp_cart_pend.build_plant_welded(
        temp_plant, temp_cart_model, register_visuals=False
    )

    ee_frame = temp_manip.get_end_effector_frame(temp_plant)
    temp_plant.WeldFrames(
        frame_on_parent_F=ee_frame,
        frame_on_child_M=cart_body.body_frame(),
        X_FM=RigidTransform(),
    )

    temp_plant.Finalize()
    temp_builder.AddSystem(temp_plant)
    temp_diagram = temp_builder.Build()

    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_plant.GetMyContextFromRoot(temp_context)

    # Equilibrium: manipulator at [0°, 20°], pendulum hanging (α=0, β=0).
    initial_q = np.array([np.deg2rad(0.0), np.deg2rad(20.0)])
    temp_manip.set_positions_user_order(temp_plant, temp_plant_context, {
        "link1_base": initial_q[0],
        "link2_link1": initial_q[1],
    })
    temp_plant.SetPositions(temp_plant_context, temp_cart_model, np.array([0.0, 0.0]))
    temp_plant.SetVelocities(temp_plant_context, np.zeros(temp_plant.num_velocities()))

    manip_input_port = temp_plant.get_actuation_input_port(temp_manip.model_instance)
    state_output_port = temp_plant.get_state_output_port()
    manip_input_port.FixValue(temp_plant_context, np.zeros(2))

    linear_sys = Linearize(
        temp_plant,
        temp_plant_context,
        input_port_index=manip_input_port.get_index(),
        output_port_index=state_output_port.get_index(),
    )

    A_plant = linear_sys.A()   # (8×8)
    B_plant = linear_sys.B()   # (8×2)

    print(colored(
        f"  - Linearized around: q1={np.rad2deg(initial_q[0]):.1f}°, "
        f"q2={np.rad2deg(initial_q[1]):.1f}°, α=0°, β=0°", "cyan"
    ))
    print(colored(
        f"  - Plant state dimension: {A_plant.shape[0]} (q1, q2, α, β, q̇1, q̇2, α̇, β̇)", "cyan"
    ))
    print(colored(f"  - Control dimension: {B_plant.shape[1]} (τ1, τ2)", "cyan"))

    # -----------------------------------------------------------------------
    # Early exit: return the 8D plant matrices if no augmentation requested
    # -----------------------------------------------------------------------
    if muscle_config is None or zft_config is None:
        return A_plant, B_plant

    # -----------------------------------------------------------------------
    # AUGMENT TO 14D
    # State: [q_arm(0:2), q_pend(2:4), v_arm(4:6), v_pend(6:8),
    #          F(8:10), p_ref(10:12), pdot_ref(12:14)]
    # Control u (2D): neural command → muscle
    # -----------------------------------------------------------------------
    muscle_tau = muscle_config.muscle_tau
    M_z  = zft_config.M_ref
    K_z  = zft_config.K_imp
    D_z  = zft_config.D_imp

    # ------------------------------------------------------------------
    # Compute EE Jacobian J_arm (2×2) mapping [v_{q1}, v_{q2}] → [ẋ_ee, ẏ_ee]
    # at the equilibrium configuration (already set in temp_plant_context).
    # ------------------------------------------------------------------
    ee_frame = temp_manip.get_end_effector_frame(temp_plant)
    world_frame_tmp = temp_plant.world_frame()
    jt1_tmp = temp_manip.get_joint_by_name(temp_plant, temp_manip.JT1_NAME)
    jt2_tmp = temp_manip.get_joint_by_name(temp_plant, temp_manip.JT2_NAME)
    manip_vel_indices = [jt1_tmp.velocity_start(), jt2_tmp.velocity_start()]

    J_full = temp_plant.CalcJacobianTranslationalVelocity(
        temp_plant_context,
        JacobianWrtVariable.kQDot,
        ee_frame,
        np.zeros(3),
        world_frame_tmp,
        world_frame_tmp,
    )  # shape (3, n_v_total)
    J_arm = J_full[0:2, manip_vel_indices]   # (2×2): XY rows, arm vel cols
    try:
        J_inv = np.linalg.inv(J_arm)
    except np.linalg.LinAlgError:
        J_inv = np.linalg.pinv(J_arm)

    print(colored(
        f"  - J_arm at eq:\n    {np.round(J_arm, 4)}\n"
        f"  - J_inv:\n    {np.round(J_inv, 4)}", "cyan"
    ))

    # ------------------------------------------------------------------
    # Assemble A_aug (14×14) and B_aug (14×2)
    # ------------------------------------------------------------------
    A_aug = np.zeros((14, 14))
    B_aug = np.zeros((14, 2))

    # == Plant block (rows 0:8) — open-loop Drake linearization ==
    A_aug[0:8, 0:8] = A_plant

    # == CT coupling: computed-torque adds joint-space stiffness/damping ==
    # The CT controller (in joint space) linearizes to:
    #   δv̇_arm ≈ Kp_ct * J_inv * δp_ref  - Kp_ct * δq_arm
    #            + Kd_ct * J_inv * δṗ_ref - Kd_ct * δv_arm
    # These are ADDITIONAL to A_plant (Drake linearized with τ=0).
    A_aug[4:6, 0:2]   -= Kp_ct * np.eye(2)           # -Kp * δq_arm → v̇_arm
    A_aug[4:6, 4:6]   -= Kd_ct * np.eye(2)           # -Kd * δv_arm → v̇_arm
    A_aug[4:6, 10:12] += Kp_ct * J_inv               # +Kp * J⁻¹ * δp_ref → v̇_arm
    A_aug[4:6, 12:14] += Kd_ct * J_inv               # +Kd * J⁻¹ * δṗ_ref → v̇_arm

    # == Muscle block (rows 8:10): Ḟ = -F/τ + u/τ ==
    A_aug[8:10, 8:10] = -np.eye(2) / muscle_tau
    B_aug[8:10, :]    =  np.eye(2) / muscle_tau

    # == ZFT block (rows 10:14): [ṗ_ref, p̈_ref] ==
    # ṗ_ref = pdot_ref
    A_aug[10, 12] = 1.0
    A_aug[11, 13] = 1.0

    # p̈_ref = K/M*(J·q_arm - p_ref) + D/M*(J·v_arm - pdot_ref) + F/M
    #   x-component (row 12):
    A_aug[12, 0:2] =  K_z / M_z * J_arm[0, :]    # K/M * J[0,:] * δq_arm
    A_aug[12, 4:6] =  D_z / M_z * J_arm[0, :]    # D/M * J[0,:] * δv_arm
    A_aug[12, 8]   =  1.0 / M_z                   # F_x / M
    A_aug[12, 10]  = -K_z / M_z                   # -K/M * δp_ref_x
    A_aug[12, 12]  = -D_z / M_z                   # -D/M * δpdot_ref_x
    #   y-component (row 13):
    A_aug[13, 0:2] =  K_z / M_z * J_arm[1, :]    # K/M * J[1,:] * δq_arm
    A_aug[13, 4:6] =  D_z / M_z * J_arm[1, :]    # D/M * J[1,:] * δv_arm
    A_aug[13, 9]   =  1.0 / M_z                   # F_y / M
    A_aug[13, 11]  = -K_z / M_z                   # -K/M * δp_ref_y
    A_aug[13, 13]  = -D_z / M_z                   # -D/M * δpdot_ref_y

    print(colored(
        f"  - Augmented state (14D): [q_arm(2), q_pend(2), v_arm(2), v_pend(2), F(2), p_ref(4)]",
        "cyan"
    ))

    return A_aug, B_aug


# ============================================================================
# INVERSE KINEMATICS FEASIBILITY CHECK
# ============================================================================

def check_trajectory_feasibility(manipulator, plant, trajectory_points, q_init=None):
    """
    Check if the manipulator can reach all points in the trajectory using IK.

    Parameters
    ----------
    manipulator : CupManipulator
    plant : MultibodyPlant
    trajectory_points : np.ndarray, shape (N, 2)
        Array of [x, y] target positions.
    q_init : np.ndarray, optional
        Initial joint configuration [q1, q2] in radians.
        Defaults to [-10°, 20°].

    Returns
    -------
    feasible : np.ndarray of bool, shape (N,)
    joint_solutions : np.ndarray, shape (N, 2)
    stats : dict
    """
    from scipy.optimize import minimize

    if q_init is None:
        q_init = np.deg2rad([-10, 20])

    N = len(trajectory_points)
    feasible = np.zeros(N, dtype=bool)
    joint_solutions = np.zeros((N, 2))

    ee_frame = plant.GetFrameByName(manipulator.LINK2_NAME, manipulator.model_instance)
    world_frame = plant.world_frame()
    EE_OFFSET = manipulator.EE_OFFSET

    def forward_kinematics(q):
        context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, context, {
            "link1_base": q[0],
            "link2_link1": q[1],
        })
        ee_pos = plant.CalcPointsPositions(
            context, ee_frame, EE_OFFSET.reshape(3, 1), world_frame
        ).flatten()
        return ee_pos[:2]

    def ik_cost(q, target_xy):
        error = target_xy - forward_kinematics(q)
        return np.sum(error ** 2)

    q_prev = q_init.copy()
    for i, target_xy in enumerate(trajectory_points):
        result = minimize(
            ik_cost,
            q_prev,
            args=(target_xy,),
            method='SLSQP',
            bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
            options={'ftol': 1e-6, 'maxiter': 100},
        )
        final_error = np.sqrt(result.fun)
        if final_error < 0.01:   # 10 mm threshold
            feasible[i] = True
            joint_solutions[i] = result.x
            q_prev = result.x
        else:
            feasible[i] = False
            joint_solutions[i] = np.nan

    stats = {
        'n_total': N,
        'n_feasible': int(np.sum(feasible)),
        'n_infeasible': int(N - np.sum(feasible)),
        'feasibility_rate': float(np.sum(feasible) / N * 100),
        'max_joint_range_deg': np.rad2deg([
            np.nanmax(joint_solutions[:, 0]) - np.nanmin(joint_solutions[:, 0]),
            np.nanmax(joint_solutions[:, 1]) - np.nanmin(joint_solutions[:, 1]),
        ]).tolist() if np.sum(feasible) > 0 else [0, 0],
    }

    return feasible, joint_solutions, stats


def test_and_visualize_ik_feasibility(
    manipulator,
    plant,
    duration,
    dt=0.001,
    trajectory_func=None,
    x_target=None,
    y_target=None,
):
    """
    Test IK feasibility for the entire trajectory and print results.

    Parameters
    ----------
    manipulator : CupManipulator
    plant : MultibodyPlant
    duration : float
        Simulation duration [s].
    dt : float
        Time step [s].
    trajectory_func : callable, optional
        Function ``f(t) -> (x, y)`` for a custom trajectory.
        If None, a linear ramp from a hardcoded initial position to
        ``(x_target, y_target)`` is used.
    x_target, y_target : float, optional
        Target position (used when ``trajectory_func`` is None).

    Returns
    -------
    feasible : np.ndarray of bool
    joint_solutions : np.ndarray, shape (N_sampled, 2)
    trajectory_points : np.ndarray, shape (N, 2)
    stats : dict
    """
    print(colored("\n🔍 Testing Inverse Kinematics Feasibility...", "cyan"))

    t_points = np.arange(0, duration, dt)
    N = len(t_points)
    trajectory_points = np.zeros((N, 2))

    if trajectory_func is not None:
        for i, t in enumerate(t_points):
            trajectory_points[i] = trajectory_func(t)
    else:
        x0, y0 = -2.174, 0.052   # Typical initial EE position
        for i, t in enumerate(t_points):
            alpha = min(t / duration, 1.0)
            trajectory_points[i, 0] = x0 + alpha * (x_target - x0)
            trajectory_points[i, 1] = y0 + alpha * (y_target - y0)

    # Sample every 10 steps for faster IK evaluation.
    sample_indices = np.arange(0, N, 10)
    sampled_points = trajectory_points[sample_indices]
    sampled_times = t_points[sample_indices]

    feasible, joint_solutions, stats = check_trajectory_feasibility(
        manipulator, plant, sampled_points
    )

    print(colored("📊 IK Feasibility Analysis:", "cyan"))
    print(f"   Total points checked: {stats['n_total']}")
    print(f"   Feasible points:      {stats['n_feasible']} ({stats['feasibility_rate']:.1f}%)")
    print(f"   Infeasible points:    {stats['n_infeasible']}")

    if stats['n_feasible'] > 0:
        print(f"   Joint 1 range:        {stats['max_joint_range_deg'][0]:.1f}°")
        print(f"   Joint 2 range:        {stats['max_joint_range_deg'][1]:.1f}°")
        q1_min = np.rad2deg(np.nanmin(joint_solutions[:, 0]))
        q1_max = np.rad2deg(np.nanmax(joint_solutions[:, 0]))
        q2_min = np.rad2deg(np.nanmin(joint_solutions[:, 1]))
        q2_max = np.rad2deg(np.nanmax(joint_solutions[:, 1]))
        print(f"   Joint 1 limits:       [{q1_min:+6.1f}°, {q1_max:+6.1f}°]")
        print(f"   Joint 2 limits:       [{q2_min:+6.1f}°, {q2_max:+6.1f}°]")

    if stats['n_infeasible'] > 0:
        infeasible_times = sampled_times[~feasible]
        print(colored(
            f"\n⚠️  Warning: {stats['n_infeasible']} points are unreachable!", "yellow"
        ))
        if len(infeasible_times) <= 10:
            print(f"   Infeasible times: {infeasible_times}")
        else:
            print(f"   First infeasible time: {infeasible_times[0]:.3f}s")
            print(f"   Last infeasible time:  {infeasible_times[-1]:.3f}s")
    else:
        print(colored("✓ All trajectory points are reachable!", "green"))

    return feasible, joint_solutions, trajectory_points, stats
