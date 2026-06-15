"""MHP Plant A simulation: full cable framework (CT → W(q) → T → MIT → Plant A)."""
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
from controller.mhp_cable_framework import MHPCableFramework
from controller.cable_wrench_mhp import CableWrenchConfig
from controller.trajectory_drake import build_trajectory, build_move_to_start, PreambleSrc
from cable.cables_mhp import build_lower_cable_config, build_upper_cable_config
from cable.meshcat_mhp import _configure_urdf_alpha, update_mhp_cable_tubes

_DT = 0.002
_M_PATCH = SpatialInertia(
    mass=0.3, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)
_DEFAULT_VIS_PREFIX = "visualizer"
_PLANT_A_NAME = "PlantA"


def _patch_zero_mass(plant, plant_ctx, model_instance) -> list[str]:
    patched = []
    for idx in plant.GetBodyIndices(model_instance):
        body = plant.get_body(idx)
        if body.default_mass() < 1e-6:
            body.SetSpatialInertiaInBodyFrame(plant_ctx, _M_PATCH)
            patched.append(body.name())
    return patched


def _build_plant_a(args) -> tuple[DiagramBuilder, MultibodyPlant, MHPManipulator, object, object]:
    """Build Plant A (passive MHP multibody) + scene graph inside a builder."""
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

    builder = DiagramBuilder()
    plant_a = MultibodyPlant(time_step=_DT)
    plant_a.set_name(_PLANT_A_NAME)
    scene_graph = builder.AddSystem(SceneGraph())
    plant_a.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = MHPManipulator(manip_config, enable_visualization=True)
    parser_urdf = Parser(plant_a)
    manipulator.load_urdf_to_plant(plant_a, parser_urdf)
    manipulator.weld_base_to_world(
        plant_a,
        position=np.zeros(3),
        orientation=np.deg2rad([args.tilt_roll, args.tilt_pitch, 0.0]),
    )
    manipulator.add_joint_actuators(plant_a)
    manipulator.set_joint_properties(plant_a)
    for jt_name, cfg in manip_config.joint_configs.items():
        if cfg.stiffness and cfg.stiffness > 0.0:
            jt = manipulator.get_joint_by_name(plant_a, jt_name)
            plant_a.AddForceElement(
                RevoluteSpring(jt, nominal_angle=0.0, stiffness=cfg.stiffness),
            )
    manipulator.add_end_effector_frame(plant_a)
    plant_a.Finalize()
    builder.AddSystem(plant_a)

    builder.Connect(
        plant_a.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant_a.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant_a.get_geometry_query_input_port(),
    )

    return builder, plant_a, manipulator, scene_graph, manip_config


def run_simulation(meshcat, args) -> dict:
    """Build and run Plant A with the full cable framework."""
    print(colored(f"\n{'=' * 68}", "cyan"))
    print(colored("  MHP Plant A — Shoulder Direct + Elbow Antagonistic Cable", "cyan", attrs=["bold"]))
    print(colored(f"{'=' * 68}", "cyan"))

    lower_cable = build_lower_cable_config()
    upper_cable = build_upper_cable_config()

    builder, plant_a, manipulator, scene_graph, _ = _build_plant_a(args)

    visualizer = None
    if meshcat is not None:
        vis_params = MeshcatVisualizerParams(role=Role.kIllustration)
        vis_params.prefix = _DEFAULT_VIS_PREFIX
        _configure_urdf_alpha(vis_params, args.urdf_alpha)
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, vis_params,
        )

    traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt = build_trajectory(manipulator, plant_a, args)
    if args.move_duration > 0.0:
        move_traj, move_vel, move_acc, q_init = build_move_to_start(
            manipulator, plant_a, traj, traj_vel, args.move_duration,
        )
    else:
        move_traj = move_vel = move_acc = None
        L1, L2 = manipulator.ik.get_link_lengths(plant_a)
        p0 = traj.value(0.0).ravel()
        q_init, _ = manipulator.ik._solve_2r_core(L1, L2, p0, np.zeros(2))

    # ── Layer 1: Computed torque → τ_req ─────────────────────────────────────
    ct = builder.AddSystem(
        MHPComputedTorqueController(
            plant_a, manipulator,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        ),
    )
    ct.set_name("CT")

    # ── Layer 2: Cable framework → MIT motors → Plant A actuation ────────────
    wrench_cfg = CableWrenchConfig()
    cable_fw = builder.AddSystem(
        MHPCableFramework(
            plant_a, manipulator,
            wrench_cfg=wrench_cfg,
            mit_kp=tuple(args.mit_kp),
            mit_kd=tuple(args.mit_kd),
            tension_kp=float(args.tension_kp),
            elbow_ff_from_cable=bool(args.elbow_ff_from_cable),
            tau_max=args.ct_tau_max,
            dt=_DT,
            use_motor_dynamics=args.mit_dynamics,
            tension_noise_std=args.tension_noise,
        ),
    )
    cable_fw.set_name("CableFramework")

    ee_src = builder.AddSystem(
        PreambleSrc(move_traj, args.move_duration, traj, args.duration),
    )
    vel_src = builder.AddSystem(
        PreambleSrc(move_vel, args.move_duration, traj_vel, args.duration),
    )
    acc_src = builder.AddSystem(
        PreambleSrc(move_acc, args.move_duration, traj_acc, args.duration),
    )

    builder.Connect(ee_src.get_output_port(), ct.GetInputPort("desired_ee_pos"))
    builder.Connect(vel_src.get_output_port(), ct.GetInputPort("ee_vel_ref"))
    builder.Connect(acc_src.get_output_port(), ct.GetInputPort("ee_acc_ref"))
    builder.Connect(plant_a.get_state_output_port(), ct.GetInputPort("plant_state"))

    builder.Connect(ct.GetOutputPort("actuation"), cable_fw.GetInputPort("tau_req"))
    builder.Connect(plant_a.get_state_output_port(), cable_fw.GetInputPort("plant_state"))
    builder.Connect(cable_fw.GetOutputPort("actuation"), plant_a.get_actuation_input_port())

    log_state = LogVectorOutput(plant_a.get_state_output_port(), builder)
    log_tau_req = LogVectorOutput(ct.GetOutputPort("torques_raw"), builder)
    log_tau_out = LogVectorOutput(cable_fw.GetOutputPort("actuation"), builder)
    log_tensions = LogVectorOutput(cable_fw.GetOutputPort("tensions"), builder)
    log_t_meas = LogVectorOutput(cable_fw.GetOutputPort("tensions_meas"), builder)
    log_tau_ff = LogVectorOutput(cable_fw.GetOutputPort("tau_ff"), builder)
    log_cable_cmd = LogVectorOutput(cable_fw.GetOutputPort("cable_cmd"), builder)
    log_qdes = LogVectorOutput(ct.GetOutputPort("joint_positions"), builder)
    log_ref = LogVectorOutput(ee_src.get_output_port(), builder)
    log_diag = LogVectorOutput(cable_fw.GetOutputPort("diagnostics"), builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    sim_ctx = simulator.get_mutable_context()
    plant_ctx = plant_a.GetMyMutableContextFromRoot(sim_ctx)

    patched = _patch_zero_mass(plant_a, plant_ctx, manipulator.model_instance)
    if patched:
        print(colored(f"  ✓ Patched zero-mass bodies: {patched}", "yellow"))

    manipulator.set_positions_user_order(plant_a, plant_ctx, q_init)
    plant_a.SetVelocities(plant_ctx, np.zeros(plant_a.num_velocities()))
    ee0 = manipulator.get_end_effector_position(plant_a, plant_ctx)
    print(colored(
        f"  ✓ Plant A init q=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
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
        pc = plant_a.GetMyMutableContextFromRoot(ctx)
        q = manipulator.get_positions_user_order(plant_a, pc)
        update_mhp_cable_tubes(meshcat, lower_cable, upper_cable, q[0], q[1])
        ee_pos = manipulator.get_end_effector_position(plant_a, pc)
        _ee_trail.append(ee_pos.copy())
        if len(_ee_trail) >= 2:
            meshcat.SetLine(_TRAIL_PATH, np.column_stack(_ee_trail), 2.5, _TRAIL_COLOR)

    wn = np.sqrt(args.ct_kp)
    zeta = np.where(wn > 0, args.ct_kd / (2.0 * wn), 0.0)
    print(colored(
        f"\n▶  CT→Cable  lap={args.duration:.1f}s  "
        f"shoulder=direct  elbow=antagonistic  "
        f"elbow_ff={'r_p·F_cmd' if args.elbow_ff_from_cable else 'τ₂_req'}  "
        f"Kp={args.ct_kp}  Kd={args.ct_kd}  ωn={wn}  ζ={zeta}",
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
    _, tau_req_data = _get(log_tau_req)
    _, tau_out_data = _get(log_tau_out)
    _, tensions_data = _get(log_tensions)
    _, t_meas_data = _get(log_t_meas)
    _, tau_ff_data = _get(log_tau_ff)
    _, cable_cmd_data = _get(log_cable_cmd)
    _, qdes_data = _get(log_qdes)
    _, ref_data = _get(log_ref)
    _, diag_data = _get(log_diag)

    return {
        "t": t_log,
        "state": state_data,
        "tau_req": tau_req_data,
        "tau_out": tau_out_data,
        "tensions": tensions_data,
        "tensions_meas": t_meas_data,
        "tau_ff": tau_ff_data,
        "cable_cmd": cable_cmd_data,
        "q_des": qdes_data,
        "ref": ref_data,
        "diag": diag_data,
        "ee_x_tgt": ee_x_tgt,
        "ee_y_tgt": ee_y_tgt,
        "manipulator": manipulator,
        "plant": plant_a,
    }


def plot_results(logs: dict, args) -> None:
    """Six-panel summary: joints, torques, cable tensions."""
    state = logs["state"]
    q_des = logs["q_des"]
    tau_req = logs["tau_req"]
    tau_out = logs["tau_out"]
    tensions = logs["tensions"]
    t_meas = logs["tensions_meas"]
    ref = logs["ref"]
    t = logs["t"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.suptitle("MHP Plant A — Shoulder Direct + Elbow Antagonistic Cable", fontsize=12, fontweight="bold")

    axes[0, 0].plot(t, np.rad2deg(state[0]), label="q1")
    axes[0, 0].plot(t, np.rad2deg(q_des[0]), "--", label="q1 des")
    axes[0, 0].plot(t, np.rad2deg(state[1]), label="q2")
    axes[0, 0].plot(t, np.rad2deg(q_des[1]), "--", label="q2 des")
    axes[0, 0].set_ylabel("Joint [deg]")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, ref[0], "--", label="x ref")
    axes[0, 1].plot(t, ref[1], "--", label="y ref")
    axes[0, 1].set_ylabel("EE ref [m]")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, tau_req[0], label="τ1 req")
    axes[1, 0].plot(t, tau_out[0], "--", label="τ1 out")
    axes[1, 0].plot(t, tau_req[1], label="τ2 req")
    axes[1, 0].plot(t, tau_out[1], "--", label="τ2 out")
    axes[1, 0].set_ylabel("Torque [Nm]")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, tensions[0], label="T lower (+Y)")
    axes[1, 1].plot(t, tensions[1], label="T upper (−Y)")
    axes[1, 1].plot(t, t_meas[0], ":", alpha=0.7, label="meas lower")
    axes[1, 1].plot(t, t_meas[1], ":", alpha=0.7, label="meas upper")
    axes[1, 1].set_ylabel("Tension [N]")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(logs["ee_x_tgt"], logs["ee_y_tgt"], "k--", lw=0.8)
    axes[2, 0].set_aspect("equal")
    axes[2, 0].set_xlabel("X [m]")
    axes[2, 0].set_ylabel("Y [m]")
    axes[2, 0].set_title("Target path")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(t, logs["tau_ff"][0], label="τ_ff shoulder")
    axes[2, 1].plot(t, logs["tau_ff"][1], label="τ_ff elbow")
    axes[2, 1].plot(t, tau_req[1], ":", alpha=0.6, label="τ₂ req")
    if args.elbow_ff_from_cable and float(args.tension_kp) > 0.0:
        axes[2, 1].plot(t, logs["cable_cmd"][1] * CableWrenchConfig().r_elbow, "--",
                        alpha=0.8, label="r_p·F_cmd")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("MIT τ_ff [Nm]")
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / f"mhp_cable_fw_{args.traj_type}_{int(time.time())}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(colored(f"\n  📊 Saved {out}", "green"))
    if not args.no_show:
        plt.show(block=True)
