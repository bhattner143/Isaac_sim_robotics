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
parser.add_argument("--compare-baseline", action="store_true",
                    help="Also run CT-only for comparison")
parser.add_argument("--move-duration", type=float, default=3.0)
parser.add_argument("--ct-kp", type=float, default=800.0)
parser.add_argument("--ct-kd", type=float, default=40.0)
parser.add_argument("--ct-tau-max", type=float, default=50.0)
parser.add_argument("--spring-stiffness", type=float, default=200.0)
parser.add_argument("--cable-damping", type=float, default=2.0)
parser.add_argument("--motor-bandwidth", type=float, default=30.0)
parser.add_argument("--residual-max", type=float, default=5.0)
parser.add_argument("--traj-type", default="circle", choices=["circle", "rect"])
args = parser.parse_args()


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
        damping=(0.05, 0.05), stiffness=(0.5, 0.5),
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
    robot.set_joint_properties()
    robot.add_joint_actuators()
    world.reset()
    robot.initialize_state()
    robot.initialize_dynamics_view(world, reset=False)
    world.reset()
    robot.initialize_state()
    robot.finalize_dynamics_view(world)

    L1, L2 = robot._get_link_lengths()
    r_p = robot.r_p

    if args.traj_type == "circle":
        main_traj = CircleTrajectory(cx=0.42, cy=0.00, radius=0.09, lap_duration=8.0, N=60)
    else:
        main_traj = RectTrajectory(
            x_range=(0.38, 0.52), y_range=(-0.10, 0.10),
            lap_duration=8.0, N=40, v_max=0.10, v_corner=0.03, corner_blend=0.35,
        )

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
        omega_m=args.motor_bandwidth, tau_max=args.ct_tau_max, dt=args.dt,
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
        if env.unwrapped._last_sea_diag:
            log_delta.append(env.unwrapped._last_sea_diag.delta)
        else:
            log_delta.append(0.0)

        if terminated or truncated:
            break

    return {
        "t": np.array(log_t),
        "ee_err_mm": np.array(log_ee_err),
        "tau_residual": np.array(log_tau_res),
        "tau_ct": np.array(log_tau_ct),
        "tau_applied": np.array(log_tau_applied),
        "delta": np.array(log_delta),
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

# Run with RL policy
print(f"[eval] Running CT+RL episode ({max_steps} steps)…")
rl_data = run_episode(env, model=model, max_steps=max_steps, label="CT + RL")
results.append(rl_data)

# Optionally run CT-only baseline
if args.compare_baseline:
    print(f"[eval] Running CT-only baseline ({max_steps} steps)…")
    # Re-initialize SEA
    sea.initialize(robot.get_positions_user_order()[1])
    baseline_data = run_episode(env, model=None, max_steps=max_steps, label="CT only")
    results.append(baseline_data)

signal.signal(signal.SIGINT, _orig_sigint)

# ═════════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════════
n_rows = 3 if not args.compare_baseline else 4
fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=True)
fig.suptitle("Residual RL Evaluation", fontsize=14, fontweight="bold")

# 1. Tracking error
ax = axes[0]
for r in results:
    ax.plot(r["t"], r["ee_err_mm"], label=r["label"], lw=1.2)
ax.set_ylabel("EE Error [mm]")
ax.set_title("End-Effector Tracking Error")
ax.legend()
ax.grid(True, alpha=0.4)

# 2. Residual torques (RL only)
ax = axes[1]
rl = results[0]
ax.plot(rl["t"], rl["tau_residual"][:, 0], label="Δτ₁", lw=1.0)
ax.plot(rl["t"], rl["tau_residual"][:, 1], label="Δτ₂", lw=1.0)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Residual τ [Nm]")
ax.set_title("RL Residual Torques")
ax.legend()
ax.grid(True, alpha=0.4)

# 3. Spring extension
ax = axes[2]
for r in results:
    ax.plot(r["t"], np.array(r["delta"]) * 1e3, label=r["label"], lw=1.0)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("δ [mm]")
ax.set_title("SEA Spring Extension")
ax.legend()
ax.grid(True, alpha=0.4)

# 4. Error comparison (if baseline)
if args.compare_baseline and len(results) > 1:
    ax = axes[3]
    for r in results:
        ax.plot(r["t"], r["ee_err_mm"], label=r["label"], lw=1.2)
    ax.set_ylabel("EE Error [mm]")
    ax.set_title("CT-only vs CT+RL Comparison")
    ax.legend()
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()

plots_dir = Path(_PROJECT_ROOT) / "plots"
plots_dir.mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
fname = plots_dir / f"eval_ppo_residual_{stamp}.png"
fig.savefig(str(fname), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[eval] Plot saved: {fname}")

# Print summary stats
for r in results:
    t = r["t"]
    err = r["ee_err_mm"]
    # Skip preamble (first 3s)
    mask = t > 3.0
    if mask.any():
        print(f"  {r['label']:12s}  mean={err[mask].mean():.2f} mm  "
              f"max={err[mask].max():.2f} mm  std={err[mask].std():.2f} mm")

env.close()
simulation_app.close()

# Open plot
import subprocess
try:
    subprocess.Popen(["eog", str(fname)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

print("[eval] Done.")
