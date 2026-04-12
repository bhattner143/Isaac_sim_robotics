"""
rl/eval_ppo_residual.py
───────────────────────
Evaluate a trained residual-RL PPO policy on the manipulator.

Runs one episode with the trained policy and produces a comparison plot:
  - CT-only (no RL residual) vs CT+RL tracking error
  - Residual torques applied by the RL agent
  - SEA spring extension

Usage::

    conda activate env_isaacsim

    # Evaluate latest checkpoint
    python rl/eval_ppo_residual.py

    # Evaluate a specific checkpoint (inside a run folder)
    python rl/eval_ppo_residual.py --model rl/checkpoints/run_20260412_153000/best_model

    # With rendering
    python rl/eval_ppo_residual.py --render native

    # Save plots to a specific directory
    python rl/eval_ppo_residual.py --plots-dir rl/checkpoints/run_20260412_153000/plots

    # Log plots to Weights & Biases
    python rl/eval_ppo_residual.py --wandb --model rl/checkpoints/run_20260412_153000/best_model
"""

import os
import sys
import math
import json
import argparse
import time
from pathlib import Path

import numpy as np

# ── Project root ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Pre-parse --render ───────────────────────────────────────────────────────
_render_mode = "headless"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]

# ── SimulationApp — MUST be first Isaac Sim import ───────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1280,
    "height": 720,
    "hide_ui": True,
})

from omni.isaac.core import World

# ── Project imports ──────────────────────────────────────────────────────────
from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
    solve_2r_ik,
    forward_kinematics_2r,
)
from controller.computed_torque_isaacsim import (
    ComputedTorqueController,
    ik_to_joint_space_references,
)
from controller.trajectory import (
    CircleTrajectory,
    RectTrajectory,
    PreambleTrajectorySource,
    build_move_to_start,
)
from actuators.sea_isaacsim import SEACableActuatorNP
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES
from rl.envs.manipulator_residual_env import ManipulatorResidualEnv

from stable_baselines3 import PPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import signal
_stop_requested = False
_orig_sigint = signal.getsignal(signal.SIGINT)

def _sigint_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[eval] Ctrl+C — finishing and plotting…")

signal.signal(signal.SIGINT, _sigint_handler)

# ── Optional W&B ─────────────────────────────────────────────────────────────
_WANDB_AVAILABLE = False
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Motor defaults
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_motor_defaults(args):
    """Resolve motor config and apply motor-derived defaults.

    Args:
        args: parsed Namespace

    Returns:
        motor:      MotorConfig dataclass
        motor_mode: MotorMode enum
    """
    motor = get_motor(args.motor)
    motor_mode = MotorMode(args.sea_mode)
    if args.ct_tau_max is None:
        args.ct_tau_max = motor.peak_torque_joint
    return motor, motor_mode


def build_scene(args, motor, motor_mode):
    """Build Isaac Sim scene for evaluation.

    Sets up the manipulator, physics world, CT controller, SEA, and trajectory.
    This mirrors the training scene setup but for a single evaluation run.

    Args:
        args:       parsed Namespace with all configuration parameters
        motor:      MotorConfig dataclass
        motor_mode: MotorMode enum

    Returns:
        Tuple: (robot, world, ct, sea, traj_source, L1, L2, r_p)
    """
    # Load and initialize the manipulator model
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    URDF_PATH = str(
        PROJECT_ROOT / "model_using_onshape_to_robot"
        / "manipulator_cable" / "manipulator_cable_obj.urdf"
    )
    config = create_cable_manipulator_config(
        urdf_path=URDF_PATH,
        joint_angles={
            "link1_base": math.radians(10.0),
            "link2_link1": math.radians(-10.0),
        },
        damping=(0.05, 0.05), stiffness=(0.0, 0.0),
    )
    robot = CupManipulatorTendonIsaac(config, enable_visualization=False)
    robot.prepare_usd()

    # Create physics world (same setup as training, but typically shorter duration)
    world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
    world.scene.add_default_ground_plane()
    robot.load_urdf(world)
    robot.weld_base_to_world(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.deg2rad([0.0, 0.0, 0.0]),
    )
    robot.add_end_effector_frame()
    robot.set_joint_properties()
    robot.add_joint_actuators()
    world.reset()
    robot.initialize_state()
    robot.initialize_dynamics_view(world)
    robot.set_initial_positions()

    # Get manipulator geometry parameters
    L1, L2 = robot._get_link_lengths()
    r_p = robot.r_p

    # Build trajectory — lap_duration matches training (episode_steps * dt)
    lap_dur = args.lap_duration if args.lap_duration is not None else args.duration
    if args.traj_type == "circle":
        main_traj = CircleTrajectory(cx=0.4, cy=0.00, radius=0.1, lap_duration=lap_dur, N=60)
    elif args.traj_type == "rect":
        main_traj = RectTrajectory(
            x_range=(0.49, 0.51), y_range=(-0.08, 0.08),
            lap_duration=lap_dur, N=60, v_max=0.9, v_corner=0.05, corner_blend=0.35,
        )
    else:
        from controller.trajectory import LineTrajectory
        main_traj = LineTrajectory(cx=0.4, cy=0.0, radius=0.1, lap_duration=lap_dur, N=60)

    # Build preamble to move EE to trajectory start
    q_cur = robot.get_positions_user_order()
    ee_cur = forward_kinematics_2r(L1, L2, *q_cur)
    preamble = build_move_to_start(
        p_start=ee_cur, p_end=main_traj.eval_position(0.0),
        v_end=main_traj.eval_velocity(0.0), duration=args.move_duration,
    )
    traj_source = PreambleTrajectorySource(preamble, main_traj)

    # Initialize CT controller and SEA actuator (same as training)
    ct = ComputedTorqueController(Kp=args.ct_kp, Kd=args.ct_kd,
                                  tau_max=args.ct_tau_max, pulley_radius=r_p)

    sea = SEACableActuatorNP(
        r_p=r_p, k_s=args.spring_stiffness, b_c=args.cable_damping,
        tau_max=args.ct_tau_max, dt=args.dt,
        motor_mode=motor_mode, motor_cfg=motor,
        omega_m=args.motor_bandwidth, motor_substeps=args.motor_substeps,
    )
    sea.initialize(q_cur[1])

    return robot, world, ct, sea, traj_source, L1, L2, r_p


def run_episode(env, model=None, max_steps=2000, label="RL"):
    """
    Run one episode and log all state/action/reward data.
    
    If model is None, uses zero residual torques (CT-only baseline).
    If model is provided, uses the trained PPO policy for residual torques.
    
    Args:
      env: ManipulatorResidualEnv instance
      model: PPO model, or None for zero residual (CT-only)
      max_steps: Max steps per episode
      label: Label for this run ("CT only" or "CT + RL")
    
    Returns:
      Dict with logged trajectories:
        - t: time array [s]
        - ee_err_mm: end-effector tracking error [mm]
        - ee_actual, ee_ref: EE positions [m]
        - tau_residual, tau_ct, tau_applied: torques [Nm]
        - delta: SEA spring extension [rad]
        - tau_motor, F_cable, l_m, l_m_des: detailed SEA states
        - label: run label
    """
    obs, info = env.reset()
    
    # Initialize logging buffers
    log_ee_err = []
    log_tau_res = []
    log_tau_ct = []
    log_tau_applied = []
    log_delta = []
    log_t = []
    log_ee_actual = []
    log_ee_ref = []
    log_tau_motor = []
    log_F_cable = []
    log_l_m = []
    log_l_m_des = []

    # Simulate one episode
    for step in range(max_steps):
        if _stop_requested:
            break

        # Get action from policy or zero (CT-only)
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)  # Use policy (no exploration noise)
        else:
            action = np.zeros(env.action_space.shape)  # Zero residual → CT only

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Log data from info dict
        log_t.append(info.get("t", step * 0.01))
        log_ee_err.append(info.get("ee_error_mm", 0.0))
        log_tau_res.append(info.get("tau_residual", np.zeros(2)).copy())
        log_tau_ct.append(info.get("tau_ct", np.zeros(2)).copy())
        log_tau_applied.append(info.get("tau_applied", np.zeros(2)).copy())
        log_ee_actual.append(info.get("ee_actual", np.zeros(2)).copy())
        log_ee_ref.append(info.get("ee_ref", np.zeros(2)).copy())

        # Log detailed SEA diagnostics (if available)
        diag = env.unwrapped._last_sea_diag
        if diag:
            log_delta.append(diag.delta)
            log_tau_motor.append(diag.tau_motor)
            log_F_cable.append(diag.F_cable)
            log_l_m.append(diag.l_m)
            log_l_m_des.append(diag.l_m_des)
        else:
            log_delta.append(0.0)
            log_tau_motor.append(0.0)
            log_F_cable.append(0.0)
            log_l_m.append(0.0)
            log_l_m_des.append(0.0)

        if terminated or truncated:
            break

    # Package results
    return {
        "t": np.array(log_t),
        "ee_err_mm": np.array(log_ee_err),
        "ee_actual": np.array(log_ee_actual),
        "ee_ref": np.array(log_ee_ref),
        "tau_residual": np.array(log_tau_res),
        "tau_ct": np.array(log_tau_ct),
        "tau_applied": np.array(log_tau_applied),
        "delta": np.array(log_delta),
        "tau_motor": np.array(log_tau_motor),
        "F_cable": np.array(log_F_cable),
        "l_m": np.array(log_l_m),
        "l_m_des": np.array(log_l_m_des),
        "label": label,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plots directory resolution
# ═════════════════════════════════════════════════════════════════════════════

def resolve_plots_dir(args):
    """Determine plots output directory.

    Priority:
      1. Explicit --plots-dir
      2. Auto-detect: if model lives inside a run folder (run_*/), use run_*/plots/
      3. Fallback: <project_root>/plots/

    Args:
        args: parsed Namespace

    Returns:
        plots_dir: Path (created if needed)
    """
    if args.plots_dir is not None:
        plots_dir = Path(_PROJECT_ROOT) / args.plots_dir
    else:
        model_path = Path(_PROJECT_ROOT) / args.model
        for parent in [model_path.parent, model_path.parent.parent]:
            if parent.name.startswith("run_") and (parent / "args.json").exists():
                plots_dir = parent / "plots"
                break
        else:
            plots_dir = Path(_PROJECT_ROOT) / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


# ═════════════════════════════════════════════════════════════════════════════
# W&B setup (eval)
# ═════════════════════════════════════════════════════════════════════════════

def setup_wandb(args):
    """Initialize W&B for eval logging (if --wandb flag is set).

    Loads API key from rl/.env.wandb, creates a wandb run tagged "eval".

    Args:
        args: parsed Namespace

    Returns:
        wandb_run: active wandb.Run or None
    """
    if not args.wandb or not _WANDB_AVAILABLE:
        return None

    env_file = Path(__file__).resolve().parent / ".env.wandb"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        tags=["eval"],
        job_type="eval",
    )
    print(f"[eval] W&B run: {wandb_run.url}")
    return wandb_run


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

_COLORS = {"CT only": "tab:red", "CT + RL": "tab:blue"}


def create_overview_plot(results, args, motor, plots_dir, stamp):
    """Generate overview figure (3×2): side-by-side comparison of all runs.

    Layout:
      [0,0] End-effector XY path (actual vs reference)
      [0,1] Tracking error vs time (post-preamble mean/max)
      [1,0] Joint-2 torque: CT-desired vs SEA-applied
      [1,1] SEA spring extension
      [2,0] Motor-side torque (saturation limits)
      [2,1] RL residual torques

    Args:
        results:   list of episode result dicts
        args:      parsed Namespace
        motor:     MotorConfig (for saturation limits)
        plots_dir: Path to output directory
        stamp:     timestamp string

    Returns:
        (fig, fname): matplotlib Figure and saved path
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(
        f"Eval Residual RL  —  {args.traj_type}  "
        f"(Kp={args.ct_kp}, Kd={args.ct_kd}, "
        f"k_s={args.spring_stiffness}, ω_m={args.motor_bandwidth})",
        fontsize=13, fontweight="bold",
    )

    # [0,0] — XY PATH
    ax = axes[0, 0]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["ee_ref"][:, 0] * 1e3, r["ee_ref"][:, 1] * 1e3,
                "--", color="gray", lw=1.0, alpha=0.5)
        ax.plot(r["ee_actual"][:, 0] * 1e3, r["ee_actual"][:, 1] * 1e3,
                "-", color=c, lw=1.5, label=r["label"])
        ax.plot(r["ee_actual"][0, 0] * 1e3, r["ee_actual"][0, 1] * 1e3,
                "o", color=c, ms=6)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("EE Path  (solid=actual, dashed=ref)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.4)

    # [0,1] — TRACKING ERROR vs TIME
    ax = axes[0, 1]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], r["ee_err_mm"], "-", color=c, lw=1.2, label=r["label"])
    ax.axvline(args.move_duration, color="k", lw=0.8, ls="--", label="preamble end")
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        mask = r["t"] > args.move_duration
        if mask.any():
            m = r["ee_err_mm"][mask].mean()
            ax.axhline(m, color=c, lw=0.9, ls=":",
                       label=f'{r["label"]} mean={m:.1f} mm')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("EE error [mm]")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

    # [1,0] — τ₂ DESIRED vs SEA-APPLIED
    ax = axes[1, 0]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], r["tau_ct"][:, 1], "-", color=c, lw=1.0, alpha=0.5)
        ax.plot(r["t"], r["tau_applied"][:, 1], "--", color=c, lw=1.2,
                label=r["label"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("τ₂ [Nm]")
    ax.set_title("Joint-2 Torque  (solid=CT desired, dashed=SEA applied)")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

    # [1,1] — SPRING EXTENSION δ
    ax = axes[1, 1]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], np.array(r["delta"]) * 1e3, "-", color=c, lw=1.2,
                label=r["label"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("δ [mm]")
    ax.set_title("SEA Spring Extension δ")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

    # [2,0] — MOTOR-SIDE TORQUE
    _peak_motor = motor.peak_torque_joint / motor.gear_ratio
    ax = axes[2, 0]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], r["tau_motor"], "-", color=c, lw=1.2, label=r["label"])
    ax.axhline(+_peak_motor, color="k", ls="--", lw=1.0,
               label=f"±peak = {_peak_motor:.2f} Nm")
    ax.axhline(-_peak_motor, color="k", ls="--", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("τ_motor [Nm]")
    ax.set_title(f"Motor-Side Torque  (N={motor.gear_ratio})")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

    # [2,1] — RL RESIDUAL TORQUES
    ax = axes[2, 1]
    rl = next((r for r in results if r["label"] == "CT + RL"), results[-1])
    ax.plot(rl["t"], rl["tau_residual"][:, 0], label="Δτ₁ (RL)", lw=1.0,
            color="tab:blue")
    ax.plot(rl["t"], rl["tau_residual"][:, 1], label="Δτ₂ (RL)", lw=1.0,
            color="tab:cyan")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Residual τ [Nm]")
    ax.set_title("RL Residual Torques")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fname = plots_dir / f"eval_ppo_overview_{stamp}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    print(f"\n[eval] Overview figure saved: {fname}")
    return fig, fname


def create_detail_plot(results, args, plots_dir, stamp):
    """Generate per-mode detail figure (N_runs × 3).

    For each run (CT-only, CT+RL), plot:
      [i,0] End-effector XY path
      [i,1] Tracking error vs time (with mean/max post-preamble)
      [i,2] Motor cable length (desired vs actual)

    Args:
        results:   list of episode result dicts
        args:      parsed Namespace
        plots_dir: Path to output directory
        stamp:     timestamp string

    Returns:
        (fig, fname): matplotlib Figure and saved path
    """
    n_runs = len(results)
    fig, axes = plt.subplots(n_runs, 3, figsize=(18, 4 * n_runs), squeeze=False)
    fig.suptitle(
        f"Per-Mode Detail  —  {args.traj_type}  "
        f"(Kp={args.ct_kp}, Kd={args.ct_kd}, "
        f"k_s={args.spring_stiffness}, ω_m={args.motor_bandwidth})",
        fontsize=13, fontweight="bold",
    )

    preamble_steps = int(args.move_duration / args.dt)

    for i, r in enumerate(results):
        c = _COLORS.get(r["label"], "tab:blue")

        # Column 0: XY PATH
        ax_xy = axes[i, 0]
        ax_xy.plot(r["ee_ref"][:, 0] * 1e3, r["ee_ref"][:, 1] * 1e3,
                   "--", color="gray", lw=1.0, alpha=0.7, label="ref")
        ax_xy.plot(r["ee_actual"][:, 0] * 1e3, r["ee_actual"][:, 1] * 1e3,
                   "-", color=c, lw=1.5, label="actual")
        ax_xy.plot(r["ee_actual"][0, 0] * 1e3, r["ee_actual"][0, 1] * 1e3,
                   "o", color=c, ms=6)
        ax_xy.set_xlabel("X [mm]"); ax_xy.set_ylabel("Y [mm]")
        ax_xy.set_title(f'{r["label"]} — EE Path')
        ax_xy.set_aspect("equal")
        ax_xy.legend(fontsize=8, loc="best"); ax_xy.grid(True, alpha=0.4)

        # Column 1: TRACKING ERROR vs TIME
        ax_err = axes[i, 1]
        ax_err.plot(r["t"], r["ee_err_mm"], "-", color=c, lw=1.2)
        ax_err.axvline(args.move_duration, color="k", lw=0.8, ls="--",
                       label="preamble end")
        n_logged = len(r["t"])
        if n_logged > preamble_steps:
            _mean_err = r["ee_err_mm"][preamble_steps:].mean()
            _max_err = r["ee_err_mm"][preamble_steps:].max()
            ax_err.axhline(_mean_err, color=c, lw=0.9, ls=":",
                           label=f"mean={_mean_err:.1f} mm  max={_max_err:.1f} mm")
        ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("EE error [mm]")
        ax_err.set_title(f'{r["label"]} — Tracking Error')
        ax_err.legend(fontsize=8, loc="best"); ax_err.grid(True, alpha=0.4)

        # Column 2: MOTOR CABLE LENGTH l_m vs desired l_m_des
        ax_sea = axes[i, 2]
        ax_sea.plot(r["t"], np.array(r["l_m"]) * 1e3, "b-", lw=1.2,
                    label="l_m (motor)")
        ax_sea.plot(r["t"], np.array(r["l_m_des"]) * 1e3, "b--", lw=1.0,
                    label="l_m_des")
        ax_sea.axhline(0, color="k", lw=0.5)
        ax_sea.set_xlabel("Time [s]"); ax_sea.set_ylabel("[mm]")
        ax_sea.set_title(f'{r["label"]} — Motor Cable l_m')
        ax_sea.legend(fontsize=8, loc="best"); ax_sea.grid(True, alpha=0.4)

    fig.tight_layout()
    fname = plots_dir / f"eval_ppo_detail_{stamp}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    print(f"  Detail figure saved: {fname}")
    return fig, fname


def log_plots_to_wandb(wandb_run, fig_overview, fig_detail, summary_stats):
    """Upload eval figures and summary metrics to W&B.

    Args:
        wandb_run:     active wandb.Run (or None)
        fig_overview:  matplotlib Figure (overview)
        fig_detail:    matplotlib Figure (detail)
        summary_stats: dict of {label: {mean, max, std}}
    """
    if wandb_run is None:
        return

    wandb_run.log({
        "eval/overview": wandb.Image(fig_overview),
        "eval/detail": wandb.Image(fig_detail),
    })

    for label, stats in summary_stats.items():
        prefix = "eval/" + label.replace(" ", "_").lower()
        wandb_run.log({
            f"{prefix}/mean_err_mm": stats["mean"],
            f"{prefix}/max_err_mm": stats["max"],
            f"{prefix}/std_err_mm": stats["std"],
        })

    print("[eval] Plots and metrics logged to W&B.")


def print_summary(results, move_duration):
    """Print evaluation summary statistics.

    Args:
        results:       list of episode result dicts
        move_duration: preamble duration [s]

    Returns:
        summary_stats: dict of {label: {mean, max, std}}
    """
    summary_stats = {}
    print(f"\n[eval] Summary (post-preamble, t > {move_duration}s):")
    for r in results:
        t = r["t"]
        err = r["ee_err_mm"]
        mask = t > move_duration
        if mask.any():
            stats = {
                "mean": float(err[mask].mean()),
                "max": float(err[mask].max()),
                "std": float(err[mask].std()),
            }
            summary_stats[r["label"]] = stats
            print(f"  {r['label']:12s}  mean={stats['mean']:.2f} mm  "
                  f"max={stats['max']:.2f} mm  std={stats['std']:.2f} mm")
    return summary_stats


def cleanup(env, wandb_run=None):
    """Close environment, simulation, and W&B.

    Args:
        env:       ManipulatorResidualEnv
        wandb_run: active wandb.Run or None
    """
    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    simulation_app.close()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main(args):
    """Evaluation pipeline.

    1. Build identical scene as used during training
    2. Load the trained PPO policy
    3. Run CT-only baseline (if --compare-baseline)
    4. Run CT+RL (learned policy)
    5. Generate comparison plots
    6. Log to W&B (if enabled)
    7. Print summary statistics

    Args:
        args: parsed Namespace
    """
    print("\n[eval] Setting up scene…")

    # ── Motor + scene ────────────────────────────────────────────────────────
    motor, motor_mode = resolve_motor_defaults(args)
    robot, world, ct, sea, traj_source, L1, L2, r_p = build_scene(
        args, motor, motor_mode,
    )

    # ── Plots + W&B ─────────────────────────────────────────────────────────
    plots_dir = resolve_plots_dir(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    wandb_run = setup_wandb(args)

    # ── Environment + model ──────────────────────────────────────────────────
    max_steps = int(args.duration / args.dt)
    env = ManipulatorResidualEnv(
        robot=robot, world=world, ct_controller=ct, sea_actuator=sea,
        traj_source=traj_source, L1=L1, L2=L2, dt=args.dt,
        max_episode_steps=max_steps, residual_max=args.residual_max,
        render_mode=args.render,
        solve_ik_fn=solve_2r_ik, fk_fn=forward_kinematics_2r,
        ik_to_jt_fn=ik_to_joint_space_references,
    )

    model_path = Path(_PROJECT_ROOT) / args.model
    print(f"[eval] Loading model: {model_path}")
    model = PPO.load(str(model_path), env=env)

    # ── Run episodes ─────────────────────────────────────────────────────────
    results = []

    if args.compare_baseline:
        print(f"[eval] Running CT-only baseline ({max_steps} steps)…")
        baseline_data = run_episode(env, model=None, max_steps=max_steps,
                                    label="CT only")
        results.append(baseline_data)
        sea.initialize(robot.get_positions_user_order()[1])

    print(f"[eval] Running CT+RL episode ({max_steps} steps)…")
    rl_data = run_episode(env, model=model, max_steps=max_steps,
                          label="CT + RL")
    results.append(rl_data)

    signal.signal(signal.SIGINT, _orig_sigint)

    # ── Plots ────────────────────────────────────────────────────────────────
    fig_overview, fname_overview = create_overview_plot(
        results, args, motor, plots_dir, stamp,
    )
    fig_detail, fname_detail = create_detail_plot(
        results, args, plots_dir, stamp,
    )

    # ── Summary + W&B ────────────────────────────────────────────────────────
    summary_stats = print_summary(results, args.move_duration)
    log_plots_to_wandb(wandb_run, fig_overview, fig_detail, summary_stats)

    plt.close(fig_overview)
    plt.close(fig_detail)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cleanup(env, wandb_run)

    # Try to open the overview plot in default image viewer
    import subprocess
    try:
        subprocess.Popen(["eog", str(fname_overview)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    print("[eval] Done.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate residual-RL PPO agent")

    # Model and rendering
    parser.add_argument("--render", default=_render_mode,
                        choices=["native", "headless"],
                        help="'native' shows 3D window, 'headless' runs faster")
    parser.add_argument("--model", type=str,
                        default="rl/checkpoints/ppo_residual_latest",
                        help="Path to trained PPO model (without .zip extension)")

    # Episode configuration
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Episode duration [s]")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Physics timestep [s]")
    parser.add_argument("--move-duration", type=float, default=3.0,
                        help="Duration of preamble trajectory [s]")
    parser.add_argument("--lap-duration", type=float, default=None,
                        help="Duration of one trajectory lap [s]. Default: matches --duration (same as training episode_steps*dt)")

    # Baseline comparison
    parser.add_argument("--compare-baseline", action="store_true", default=True,
                        help="Run CT-only baseline for comparison (default: True)")
    parser.add_argument("--no-compare-baseline", dest="compare_baseline",
                        action="store_false",
                        help="Skip CT-only baseline run")

    # CT controller gains (should match training)
    parser.add_argument("--ct-kp", type=float, default=100.0,
                        help="CT proportional gain [Nm·s/rad]")
    parser.add_argument("--ct-kd", type=float, default=40.0,
                        help="CT derivative gain [Nm·s²/rad]")
    parser.add_argument("--ct-tau-max", type=float, default=None,
                        help="Torque saturation [Nm]. Default: motor peak torque")

    # Motor and SEA (should match training)
    parser.add_argument("--motor", choices=MOTOR_CHOICES,
                        default="AK60_6_KV80_Config",
                        help="Motor model")
    parser.add_argument("--sea-mode", choices=["torque", "position"],
                        default="torque",
                        help="Motor control mode")
    parser.add_argument("--spring-stiffness", type=float, default=30.0,
                        help="SEA spring stiffness [Nm/rad]")
    parser.add_argument("--cable-damping", type=float, default=2.0,
                        help="SEA cable damping [Nm·s/rad]")
    parser.add_argument("--motor-bandwidth", type=float, default=100.0,
                        help="Motor servo bandwidth [rad/s]")
    parser.add_argument("--motor-substeps", type=int, default=None,
                        help="Motor integration sub-steps")
    parser.add_argument("--residual-max", type=float, default=5.0,
                        help="Max RL residual torque magnitude [Nm]")

    # Trajectory (should match training)
    parser.add_argument("--traj-type", default="rect",
                        choices=["circle", "rect", "line"],
                        help="End-effector trajectory shape")

    # Output
    parser.add_argument("--plots-dir", type=str, default=None,
                        help="Directory for eval plots. Auto-detected from "
                             "model path if inside a run_* folder")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str,
                        default="isaac_sim_robotics",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str,
                        default="dbha483-transgp",
                        help="W&B entity (team or username)")

    main(parser.parse_args())
