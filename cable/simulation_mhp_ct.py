"""Drake computed-torque simulation for the MHP cable manipulator."""
from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    LogVectorOutput,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    Parser,
    Role,
)
from pydrake.multibody.tree import RevoluteSpring
from pydrake.geometry import Rgba

from configs.robot.mhp_configs import create_mhp_config
from robots.mhp_manipulator import MHPManipulator
from controller.mhp_ct_controller import MHPComputedTorqueController
from controller.trajectory_drake import build_trajectory, build_move_to_start, PreambleSrc
from cable.cables_mhp import build_lower_cable_config, build_upper_cable_config
from cable.meshcat_mhp import _configure_urdf_alpha, update_mhp_cable_tubes

_DT = 0.002
_M_PATCH = SpatialInertia(
    mass=0.3, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)
_DEFAULT_VIS_PREFIX = "visualizer"


def _patch_zero_mass(plant, plant_ctx, model_instance) -> list[str]:
    patched = []
    for idx in plant.GetBodyIndices(model_instance):
        body = plant.get_body(idx)
        if body.default_mass() < 1e-6:
            body.SetSpatialInertiaInBodyFrame(plant_ctx, _M_PATCH)
            patched.append(body.name())
    return patched


def run_simulation(meshcat, args) -> dict:
    """Build, run, and log an MHP CT trajectory-tracking simulation."""
    print(colored(f"\n{'=' * 68}", "cyan"))
    print(colored("  MHP — Computed Torque + Cable Routing", "cyan", attrs=["bold"]))
    print(colored(f"{'=' * 68}", "cyan"))

    manip_config = create_mhp_config(
        joint_angles={
            "jt_upper_base": np.deg2rad(0.0),
            "jt_lower_upper": np.deg2rad(0.0),
        },
        damping=tuple(args.joint_damping),
        stiffness=tuple(args.joint_stiffness),
        tilt_roll_deg=args.tilt_roll,
        tilt_pitch_deg=args.tilt_pitch,
    )

    lower_cable = build_lower_cable_config()
    upper_cable = build_upper_cable_config()

    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=_DT)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = MHPManipulator(manip_config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(
        plant,
        position=np.zeros(3),
        orientation=np.deg2rad([args.tilt_roll, args.tilt_pitch, 0.0]),
    )
    manipulator.add_joint_actuators(plant)
    manipulator.set_joint_properties(plant)
    for jt_name, cfg in manip_config.joint_configs.items():
        if cfg.stiffness and cfg.stiffness > 0.0:
            jt = manipulator.get_joint_by_name(plant, jt_name)
            plant.AddForceElement(RevoluteSpring(jt, nominal_angle=0.0, stiffness=cfg.stiffness))
    manipulator.add_end_effector_frame(plant)
    plant.Finalize()
    builder.AddSystem(plant)

    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    visualizer = None
    if meshcat is not None:
        vis_params = MeshcatVisualizerParams(role=Role.kIllustration)
        vis_params.prefix = _DEFAULT_VIS_PREFIX
        _configure_urdf_alpha(vis_params, args.urdf_alpha)
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, vis_params
        )

    traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt = build_trajectory(manipulator, plant, args)
    if args.move_duration > 0.0:
        move_traj, move_vel, move_acc, q_init = build_move_to_start(
            manipulator, plant, traj, traj_vel, args.move_duration
        )
    else:
        move_traj = move_vel = move_acc = None
        L1, L2 = manipulator.ik.get_link_lengths(plant)
        p0 = traj.value(0.0).ravel()
        q_init, _ = manipulator.ik._solve_2r_core(L1, L2, p0, np.zeros(2))

    ct = builder.AddSystem(
        MHPComputedTorqueController(
            plant, manipulator,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        )
    )
    ct.set_name("MHP_CT")

    ee_src = builder.AddSystem(
        PreambleSrc(move_traj, args.move_duration, traj, args.duration)
    )
    vel_src = builder.AddSystem(
        PreambleSrc(move_vel, args.move_duration, traj_vel, args.duration)
    )
    acc_src = builder.AddSystem(
        PreambleSrc(move_acc, args.move_duration, traj_acc, args.duration)
    )

    builder.Connect(ee_src.get_output_port(), ct.GetInputPort("desired_ee_pos"))
    builder.Connect(vel_src.get_output_port(), ct.GetInputPort("ee_vel_ref"))
    builder.Connect(acc_src.get_output_port(), ct.GetInputPort("ee_acc_ref"))
    builder.Connect(plant.get_state_output_port(), ct.GetInputPort("plant_state"))
    builder.Connect(ct.GetOutputPort("actuation"), plant.get_actuation_input_port())

    log_state = LogVectorOutput(plant.get_state_output_port(), builder)
    log_act = LogVectorOutput(ct.GetOutputPort("actuation"), builder)
    log_qdes = LogVectorOutput(ct.GetOutputPort("joint_positions"), builder)
    log_tau = LogVectorOutput(ct.GetOutputPort("torques_raw"), builder)
    log_ref = LogVectorOutput(ee_src.get_output_port(), builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    sim_ctx = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    patched = _patch_zero_mass(plant, plant_ctx, manipulator.model_instance)
    if patched:
        print(colored(f"  ✓ Patched zero-mass bodies: {patched}", "yellow"))

    manipulator.set_positions_user_order(plant, plant_ctx, q_init)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    ee0 = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored(
        f"  ✓ Init q=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE=({ee0[0]*1e3:.1f}, {ee0[1]*1e3:.1f}) mm",
        "green",
    ))

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    if visualizer is not None and args.record:
        visualizer.StartRecording()

    _ee_trail: list = []
    _TRAIL_PATH = "/ee_trail"
    _TRAIL_COLOR = Rgba(0.0, 0.9, 1.0, 0.85)

    def _viz_tick():
        if meshcat is None:
            return
        ctx = simulator.get_mutable_context()
        pc = plant.GetMyMutableContextFromRoot(ctx)
        q = manipulator.get_positions_user_order(plant, pc)
        update_mhp_cable_tubes(meshcat, lower_cable, upper_cable, q[0], q[1])
        ee_pos = manipulator.get_end_effector_position(plant, pc)
        _ee_trail.append(ee_pos.copy())
        if len(_ee_trail) >= 2:
            meshcat.SetLine(_TRAIL_PATH, np.column_stack(_ee_trail), 2.5, _TRAIL_COLOR)

    wn = np.sqrt(args.ct_kp)
    zeta = np.where(wn > 0, args.ct_kd / (2.0 * wn), 0.0)
    print(colored(
        f"\n▶  CT  lap={args.duration:.1f}s  Kp={args.ct_kp}  Kd={args.ct_kd}  "
        f"ωn={wn}  ζ={zeta}",
        "yellow",
    ))
    if meshcat is not None:
        print(colored(f"  Meshcat → {meshcat.web_url()}", "green"))

    t_end = args.move_duration + args.duration * max(args.num_laps, 1)
    if args.num_laps == 0:
        t_end = float("inf")

    _viz_tick()
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while simulator.get_context().get_time() < t_end:
            simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)
            _viz_tick()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)

    if visualizer is not None and args.record:
        visualizer.StopRecording()
        visualizer.PublishRecording()

    def _get(log):
        obj = log.FindLog(sim_ctx)
        return obj.sample_times(), obj.data()

    t_log, state_data = _get(log_state)
    _, act_data = _get(log_act)
    _, qdes_data = _get(log_qdes)
    _, tau_data = _get(log_tau)
    _, ref_data = _get(log_ref)

    return {
        "t": t_log,
        "state": state_data,
        "actuation": act_data,
        "q_des": qdes_data,
        "tau_raw": tau_data,
        "ref": ref_data,
        "ee_x_tgt": ee_x_tgt,
        "ee_y_tgt": ee_y_tgt,
        "manipulator": manipulator,
        "plant": plant,
    }


def plot_results(logs: dict, args) -> None:
    """Quick 4-panel summary of the CT run."""
    state = logs["state"]
    q_des = logs["q_des"]
    tau = logs["tau_raw"]
    ref = logs["ref"]
    t = logs["t"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("MHP Computed Torque", fontsize=12, fontweight="bold")

    axes[0, 0].plot(t, np.rad2deg(state[0]), label="q1 act")
    axes[0, 0].plot(t, np.rad2deg(q_des[0]), "--", label="q1 des")
    axes[0, 0].plot(t, np.rad2deg(state[1]), label="q2 act")
    axes[0, 0].plot(t, np.rad2deg(q_des[1]), "--", label="q2 des")
    axes[0, 0].set_ylabel("Joint [deg]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, tau[0], label="τ1")
    axes[0, 1].plot(t, tau[1], label="τ2")
    axes[0, 1].set_ylabel("Torque [Nm]")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, ref[0], "--", label="x ref")
    axes[1, 0].plot(t, ref[1], "--", label="y ref")
    axes[1, 0].set_ylabel("EE ref [m]")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(logs["ee_x_tgt"], logs["ee_y_tgt"], "k--", lw=0.8, label="target")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlabel("X [m]")
    axes[1, 1].set_ylabel("Y [m]")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / f"mhp_ct_{args.traj_type}_{int(time.time())}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(colored(f"\n  📊 Saved {out}", "green"))
    if not args.no_show:
        plt.show(block=True)
