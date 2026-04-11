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

    # Evaluate a specific checkpoint
    python rl/eval_ppo_residual.py --model rl/checkpoints/ppo_residual_final_20260410_150000

    # With rendering
    python rl/eval_ppo_residual.py --render native --model rl/checkpoints/ppo_residual_latest

    # Compare against CT-only baseline
    python rl/eval_ppo_residual.py --compare-baseline
"""

import os
import sys
import math
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

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate residual-RL PPO agent")
parser.add_argument("--render", default=_render_mode, choices=["native", "headless"])
parser.add_argument("--model", type=str, default="rl/checkpoints/ppo_residual_latest",
                    help="Path to trained model (without .zip)")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--compare-baseline", action="store_true", default=True,
                    help="Also run CT-only for comparison (default: True)")
parser.add_argument("--no-compare-baseline", dest="compare_baseline",
                    action="store_false",
                    help="Skip CT-only baseline run")
parser.add_argument("--move-duration", type=float, default=3.0)
parser.add_argument("--ct-kp", type=float, default=100.0)
parser.add_argument("--ct-kd", type=float, default=40.0)
parser.add_argument("--ct-tau-max", type=float, default=None,
                    help="Torque saturation [Nm]. Default: motor peak.")
parser.add_argument("--motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config")
parser.add_argument("--sea-mode", choices=["torque", "position"], default="torque")
parser.add_argument("--spring-stiffness", type=float, default=30.0)
parser.add_argument("--cable-damping", type=float, default=2.0)
parser.add_argument("--motor-bandwidth", type=float, default=100.0)
parser.add_argument("--motor-substeps", type=int, default=None)
parser.add_argument("--residual-max", type=float, default=5.0)
parser.add_argument("--traj-type", default="rect", choices=["circle", "rect", "line"])
args = parser.parse_args()

# Motor-derived defaults
_motor = get_motor(args.motor)
_motor_mode = MotorMode(args.sea_mode)
if args.ct_tau_max is None:
    args.ct_tau_max = _motor.peak_torque_joint


def build_scene(render_mode, args):
    """Build Isaac Sim scene and return all components."""
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

    L1, L2 = robot._get_link_lengths()
    r_p = robot.r_p

    if args.traj_type == "circle":
        main_traj = CircleTrajectory(cx=0.4, cy=0.00, radius=0.1, lap_duration=8.0, N=60)
    elif args.traj_type == "rect":
        main_traj = RectTrajectory(
            x_range=(0.49, 0.51), y_range=(-0.08, 0.08),
            lap_duration=8.0, N=60, v_max=0.9, v_corner=0.05, corner_blend=0.35,
        )
    else:
        from controller.trajectory import LineTrajectory
        main_traj = LineTrajectory(cx=0.4, cy=0.0, radius=0.1, lap_duration=8.0, N=60)

    q_cur = robot.get_positions_user_order()
    ee_cur = forward_kinematics_2r(L1, L2, *q_cur)
    preamble = build_move_to_start(
        p_start=ee_cur, p_end=main_traj.eval_position(0.0),
        v_end=main_traj.eval_velocity(0.0), duration=args.move_duration,
    )
    traj_source = PreambleTrajectorySource(preamble, main_traj)

    ct = ComputedTorqueController(Kp=args.ct_kp, Kd=args.ct_kd,
                                  tau_max=args.ct_tau_max, pulley_radius=r_p)

    sea = SEACableActuatorNP(
        r_p=r_p, k_s=args.spring_stiffness, b_c=args.cable_damping,
        tau_max=args.ct_tau_max, dt=args.dt,
        motor_mode=_motor_mode, motor_cfg=_motor,
        omega_m=args.motor_bandwidth, motor_substeps=args.motor_substeps,
    )
    sea.initialize(q_cur[1])

    return robot, world, ct, sea, traj_source, L1, L2, r_p


def run_episode(env, model=None, max_steps=2000, label="RL"):
    """Run one episode. If model is None, uses zero residual (CT-only)."""
    obs, info = env.reset()
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

    for step in range(max_steps):
        if _stop_requested:
            break

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)

        log_t.append(info.get("t", step * 0.01))
        log_ee_err.append(info.get("ee_error_mm", 0.0))
        log_tau_res.append(info.get("tau_residual", np.zeros(2)).copy())
        log_tau_ct.append(info.get("tau_ct", np.zeros(2)).copy())
        log_tau_applied.append(info.get("tau_applied", np.zeros(2)).copy())
        log_ee_actual.append(info.get("ee_actual", np.zeros(2)).copy())
        log_ee_ref.append(info.get("ee_ref", np.zeros(2)).copy())

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
# Main
# ═════════════════════════════════════════════════════════════════════════════
print("\n[eval] Setting up scene…")
robot, world, ct, sea, traj_source, L1, L2, r_p = build_scene(args.render, args)

max_steps = int(args.duration / args.dt)

env = ManipulatorResidualEnv(
    robot=robot, world=world, ct_controller=ct, sea_actuator=sea,
    traj_source=traj_source, L1=L1, L2=L2, dt=args.dt,
    max_episode_steps=max_steps, residual_max=args.residual_max,
    render_mode=args.render,
    solve_ik_fn=solve_2r_ik, fk_fn=forward_kinematics_2r,
    ik_to_jt_fn=ik_to_joint_space_references,
)

# Load trained model
model_path = Path(_PROJECT_ROOT) / args.model
print(f"[eval] Loading model: {model_path}")
model = PPO.load(str(model_path), env=env)

results = []

# ── Run CT-only baseline first (if comparing) ──────────────────────────────
if args.compare_baseline:
    print(f"[eval] Running CT-only baseline ({max_steps} steps)…")
    baseline_data = run_episode(env, model=None, max_steps=max_steps, label="CT only")
    results.append(baseline_data)
    # Re-initialize SEA for next run
    sea.initialize(robot.get_positions_user_order()[1])

# ── Run with RL policy ──────────────────────────────────────────────────────
print(f"[eval] Running CT+RL episode ({max_steps} steps)…")
rl_data = run_episode(env, model=model, max_steps=max_steps, label="CT + RL")
results.append(rl_data)

signal.signal(signal.SIGINT, _orig_sigint)

# ═════════════════════════════════════════════════════════════════════════════
# Colours: CT-only = gray/red, CT+RL = blue/green
# ═════════════════════════════════════════════════════════════════════════════
_COLORS = {"CT only": "tab:red", "CT + RL": "tab:blue"}

_peak_motor = _motor.peak_torque_joint / _motor.gear_ratio

plots_dir = Path(_PROJECT_ROOT) / "plots"
plots_dir.mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Overview (3 × 2)  — matches multi-instance layout
# ═════════════════════════════════════════════════════════════════════════════
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
# Add mean lines (post-preamble)
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
ax = axes[2, 0]
for r in results:
    c = _COLORS.get(r["label"], "tab:blue")
    ax.plot(r["t"], r["tau_motor"], "-", color=c, lw=1.2, label=r["label"])
ax.axhline(+_peak_motor, color="k", ls="--", lw=1.0,
           label=f"±peak = {_peak_motor:.2f} Nm")
ax.axhline(-_peak_motor, color="k", ls="--", lw=1.0)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Time [s]"); ax.set_ylabel("τ_motor [Nm]")
ax.set_title(f"Motor-Side Torque  (N={_motor.gear_ratio})")
ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.4)

# [2,1] — RL RESIDUAL TORQUES (or cable force if no RL)
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
fname_overview = plots_dir / f"eval_ppo_overview_{stamp}.png"
fig.savefig(str(fname_overview), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[eval] Overview figure saved: {fname_overview}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-mode detail  (N_runs × 3)  — matches per-robot layout
# ═════════════════════════════════════════════════════════════════════════════
n_runs = len(results)
fig2, axes2 = plt.subplots(n_runs, 3, figsize=(18, 4 * n_runs), squeeze=False)
fig2.suptitle(
    f"Per-Mode Detail  —  {args.traj_type}  "
    f"(Kp={args.ct_kp}, Kd={args.ct_kd}, "
    f"k_s={args.spring_stiffness}, ω_m={args.motor_bandwidth})",
    fontsize=13, fontweight="bold",
)

preamble_steps = int(args.move_duration / args.dt)

for i, r in enumerate(results):
    c = _COLORS.get(r["label"], "tab:blue")

    # Column 0: XY PATH
    ax_xy = axes2[i, 0]
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

    # Column 1: TRACKING ERROR
    ax_err = axes2[i, 1]
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

    # Column 2: MOTOR CABLE l_m vs l_m_des
    ax_sea = axes2[i, 2]
    ax_sea.plot(r["t"], np.array(r["l_m"]) * 1e3, "b-", lw=1.2,
                label="l_m (motor)")
    ax_sea.plot(r["t"], np.array(r["l_m_des"]) * 1e3, "b--", lw=1.0,
                label="l_m_des")
    ax_sea.axhline(0, color="k", lw=0.5)
    ax_sea.set_xlabel("Time [s]"); ax_sea.set_ylabel("[mm]")
    ax_sea.set_title(f'{r["label"]} — Motor Cable l_m')
    ax_sea.legend(fontsize=8, loc="best"); ax_sea.grid(True, alpha=0.4)

fig2.tight_layout()
fname_detail = plots_dir / f"eval_ppo_detail_{stamp}.png"
fig2.savefig(str(fname_detail), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  📊 Detail figure saved: {fname_detail}")

# ── Summary stats ────────────────────────────────────────────────────────────
for r in results:
    t = r["t"]
    err = r["ee_err_mm"]
    mask = t > args.move_duration
    if mask.any():
        print(f"  {r['label']:12s}  mean={err[mask].mean():.2f} mm  "
              f"max={err[mask].max():.2f} mm  std={err[mask].std():.2f} mm")

env.close()
simulation_app.close()

# Open plot
import subprocess
try:
    subprocess.Popen(["eog", str(fname_overview)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

print("[eval] Done.")
