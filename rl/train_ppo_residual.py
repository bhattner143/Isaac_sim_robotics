"""
rl/train_ppo_residual.py
────────────────────────
Train a residual-RL PPO agent on top of the CT+SEA manipulator in Isaac Sim.

The CT controller provides a baseline torque.  The PPO policy learns a
small correction Δτ that compensates for SEA lag, model error, and
trajectory-dependent dynamics.

Usage::

    conda activate env_isaacsim

    # Default training (headless, circle trajectory, 500k steps)
    python rl/train_ppo_residual.py

    # Custom SEA + longer training
    python rl/train_ppo_residual.py \\
        --total-timesteps 2000000 \\
        --spring-stiffness 100 \\
        --motor-bandwidth 20

    # Resume from checkpoint
    python rl/train_ppo_residual.py --resume rl/checkpoints/ppo_residual_latest

    # With rendering (slow, for debugging)
    python rl/train_ppo_residual.py --render native --total-timesteps 50000
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

# ── Pre-parse --render before SimulationApp ──────────────────────────────────
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

# ── Isaac Sim imports (safe after SimulationApp) ─────────────────────────────
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

# ── RL imports ───────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# ── SIGINT handling ──────────────────────────────────────────────────────────
import signal
_stop_requested = False
_orig_sigint = signal.getsignal(signal.SIGINT)

def _sigint_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[train] Ctrl+C — saving checkpoint and exiting…")

signal.signal(signal.SIGINT, _sigint_handler)

# ── Argument parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train residual-RL PPO for manipulator")
parser.add_argument("--render", default=_render_mode,
                    choices=["native", "headless"])
parser.add_argument("--total-timesteps", type=int, default=500_000)
parser.add_argument("--episode-steps", type=int, default=2000,
                    help="Max steps per episode (default: 2000 = 20s at 100Hz)")
parser.add_argument("--dt", type=float, default=1.0 / 100.0)
parser.add_argument("--residual-max", type=float, default=5.0,
                    help="Max residual torque magnitude [Nm]")
parser.add_argument("--move-duration", type=float, default=3.0)

# CT gains
parser.add_argument("--ct-kp", type=float, default=800.0)
parser.add_argument("--ct-kd", type=float, default=40.0)
parser.add_argument("--ct-tau-max", type=float, default=50.0)

# SEA parameters
parser.add_argument("--spring-stiffness", type=float, default=200.0)
parser.add_argument("--cable-damping", type=float, default=2.0)
parser.add_argument("--motor-bandwidth", type=float, default=30.0)

# PPO hyperparameters
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--n-steps", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.0)

# Reward weights
parser.add_argument("--w-tracking", type=float, default=100.0)
parser.add_argument("--w-effort", type=float, default=0.01)
parser.add_argument("--w-smoothness", type=float, default=0.001)

# Trajectory
parser.add_argument("--traj-type", default="circle",
                    choices=["circle", "rect"])

# Checkpointing
parser.add_argument("--save-freq", type=int, default=10_000,
                    help="Save checkpoint every N steps")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to model zip to resume training from")
parser.add_argument("--log-dir", type=str, default="rl/logs")
parser.add_argument("--checkpoint-dir", type=str, default="rl/checkpoints")

args = parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Build Isaac Sim scene
# ═════════════════════════════════════════════════════════════════════════════
print("\n[train] Setting up Isaac Sim scene…")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = str(
    PROJECT_ROOT
    / "model_using_onshape_to_robot"
    / "manipulator_cable"
    / "manipulator_cable_obj.urdf"
)

config = create_cable_manipulator_config(
    urdf_path=URDF_PATH,
    joint_angles={
        "link1_base":  math.radians(10.0),
        "link2_link1": math.radians(-10.0),
    },
    damping=(0.05, 0.05),
    stiffness=(0.5, 0.5),
)

robot = CupManipulatorTendonIsaac(config, enable_visualization=False)
robot.prepare_usd()

world = World(
    stage_units_in_meters=1.0,
    physics_dt=args.dt,
    rendering_dt=args.dt,
)
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
print(f"  L1={L1*1e3:.1f} mm  L2={L2*1e3:.1f} mm  r_p={r_p*1e3:.2f} mm")

# ── Trajectory ───────────────────────────────────────────────────────────────
if args.traj_type == "circle":
    main_traj = CircleTrajectory(
        cx=0.42, cy=0.00, radius=0.09, lap_duration=8.0, N=60,
    )
elif args.traj_type == "rect":
    main_traj = RectTrajectory(
        x_range=(0.38, 0.52), y_range=(-0.10, 0.10),
        lap_duration=8.0, N=40, v_max=0.10, v_corner=0.03, corner_blend=0.35,
    )

q_cur = robot.get_positions_user_order()
ee_cur = forward_kinematics_2r(L1, L2, *q_cur)
first_ee = main_traj.eval_position(0.0)
first_vel = main_traj.eval_velocity(0.0)

preamble = build_move_to_start(
    p_start=ee_cur, p_end=first_ee, v_end=first_vel,
    duration=args.move_duration,
)
traj_source = PreambleTrajectorySource(preamble, main_traj)

# ── CT Controller ────────────────────────────────────────────────────────────
ct = ComputedTorqueController(
    Kp=args.ct_kp, Kd=args.ct_kd,
    tau_max=args.ct_tau_max, pulley_radius=r_p,
)

# ── SEA Actuator ─────────────────────────────────────────────────────────────
sea = SEACableActuatorNP(
    r_p=r_p,
    k_s=args.spring_stiffness,
    b_c=args.cable_damping,
    omega_m=args.motor_bandwidth,
    tau_max=args.ct_tau_max,
    dt=args.dt,
)
sea.initialize(q_cur[1])

# ═════════════════════════════════════════════════════════════════════════════
# Build Gymnasium environment
# ═════════════════════════════════════════════════════════════════════════════
print("[train] Building Gymnasium environment…")

env = ManipulatorResidualEnv(
    robot=robot,
    world=world,
    ct_controller=ct,
    sea_actuator=sea,
    traj_source=traj_source,
    L1=L1, L2=L2,
    dt=args.dt,
    max_episode_steps=args.episode_steps,
    residual_max=args.residual_max,
    reward_weights={
        "tracking": args.w_tracking,
        "effort": args.w_effort,
        "smoothness": args.w_smoothness,
    },
    render_mode=args.render,
    solve_ik_fn=solve_2r_ik,
    fk_fn=forward_kinematics_2r,
    ik_to_jt_fn=ik_to_joint_space_references,
)
env = Monitor(env)

# ═════════════════════════════════════════════════════════════════════════════
# PPO setup
# ═════════════════════════════════════════════════════════════════════════════
print("[train] Configuring PPO…")

log_dir = Path(_PROJECT_ROOT) / args.log_dir
ckpt_dir = Path(_PROJECT_ROOT) / args.checkpoint_dir
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

ppo_kwargs = dict(
    policy="MlpPolicy",
    env=env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    ),
    device="cpu",  # Isaac Sim owns the GPU; policy runs on CPU
)

if args.resume:
    print(f"[train] Resuming from {args.resume}")
    model = PPO.load(args.resume, env=env, **{
        k: v for k, v in ppo_kwargs.items()
        if k not in ("policy", "env")
    })
else:
    model = PPO(**ppo_kwargs)

# Logger
_log_formats = ["stdout", "csv"]
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    _log_formats.append("tensorboard")
except ImportError:
    print("[train] tensorboard not installed — skipping TB logging")
new_logger = configure(str(log_dir), _log_formats)
model.set_logger(new_logger)

# Callbacks
checkpoint_cb = CheckpointCallback(
    save_freq=args.save_freq,
    save_path=str(ckpt_dir),
    name_prefix="ppo_residual",
)

# ═════════════════════════════════════════════════════════════════════════════
# Train
# ═════════════════════════════════════════════════════════════════════════════
_stamp = time.strftime("%Y%m%d_%H%M%S")
print(f"\n[train] Starting PPO training — {args.total_timesteps} steps")
print(f"  residual_max = {args.residual_max} Nm")
print(f"  SEA: k_s={args.spring_stiffness}  b_c={args.cable_damping}  ω_m={args.motor_bandwidth}")
print(f"  CT:  Kp={args.ct_kp}  Kd={args.ct_kd}")
print(f"  Reward: track={args.w_tracking}  effort={args.w_effort}  smooth={args.w_smoothness}")
print(f"  Log dir: {log_dir}")
print(f"  Checkpoint dir: {ckpt_dir}\n")

try:
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )
except KeyboardInterrupt:
    print("\n[train] Training interrupted by user.")

# Save final model
final_path = ckpt_dir / f"ppo_residual_final_{_stamp}"
model.save(str(final_path))
print(f"\n[train] Final model saved: {final_path}.zip")

# Also save as "latest" for easy resume
latest_path = ckpt_dir / "ppo_residual_latest"
model.save(str(latest_path))
print(f"[train] Latest model saved: {latest_path}.zip")

# ── Cleanup ──────────────────────────────────────────────────────────────────
signal.signal(signal.SIGINT, _orig_sigint)
env.close()
simulation_app.close()
print("[train] Done.")
