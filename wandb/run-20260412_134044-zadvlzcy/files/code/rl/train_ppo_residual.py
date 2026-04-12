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
import json
import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe inside Isaac Sim
import matplotlib.pyplot as plt

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
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES
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

# ── Weights & Biases (optional) ──────────────────────────────────────────────
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── SIGINT handling ──────────────────────────────────────────────────────────
import signal
_stop_requested = False
_orig_sigint = signal.getsignal(signal.SIGINT)

def _sigint_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[train] Ctrl+C — saving checkpoint and exiting…")

signal.signal(signal.SIGINT, _sigint_handler)


def setup_world(args):
    """Create and return the Isaac Sim physics world.

    Must be called after args are parsed so that args.dt is available.

    Args:
        args: parsed argparse Namespace

    Returns:
        world: initialized World instance with ground plane
    """
    # Create physics simulation world with specified timesteps
    world = World(
        stage_units_in_meters=1.0,   # Spatial scale (Omniverse engine parameter)
        physics_dt=args.dt,           # Physics integration step duration [s]
        rendering_dt=args.dt,         # Rendering frame update duration [s]
    )
    world.scene.add_default_ground_plane()  # Add ground for collision
    return world


def setup_robot(config, world, args):
    """Load, wire up, and initialize the manipulator inside an existing world.

    Args:
        config: ManipulatorConfig dataclass (URDF path, initial joint angles, etc.)
        world:  Isaac Sim World instance (already has ground plane)
        args:   parsed argparse Namespace (used for print only)

    Returns:
        robot: CupManipulatorTendonIsaac loaded and wired into world
               (not yet reset — call initialize_simulation() next)
    """
    # Instantiate the manipulator robot object (wrapper around URDF)
    robot = CupManipulatorTendonIsaac(config, enable_visualization=False)
    robot.prepare_usd()  # Convert URDF meshes to USD for Isaac Sim GPU rendering

    # Load robot URDF into the world scene
    robot.load_urdf(world)
    robot.weld_base_to_world(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.deg2rad([0.0, 0.0, 0.0]),  # Fixed to ground at origin
    )

    # Add end-effector frame (virtual reference frame at tool tip)
    robot.add_end_effector_frame()

    # Configure joint damping/friction and constraint stiffness
    robot.set_joint_properties()

    # Add ActuationPortIndex for each joint (required for motor commands in Isaac Sim)
    robot.add_joint_actuators()

    return robot


def initialize_simulation(world, robot):
    """Reset the physics world and extract robot geometric parameters.

    Must be called after setup_robot() and before any trajectory or controller
    setup, because it performs the first world.reset() that activates the
    ArticulationView GPU tensors.

    Args:
        world: Isaac Sim World instance (already has robot loaded)
        robot: CupManipulatorTendonIsaac (already loaded into world)

    Returns:
        L1:  proximal link length [m]
        L2:  distal link length [m]
        r_p: pulley radius [m]
    """
    # Reset world to activate GPU physics tensors and ArticulationView
    world.reset()
    robot.initialize_state()
    robot.initialize_dynamics_view(world)  # Prepare GPU view for fast joint/link operations
    robot.set_initial_positions()

    # Read geometric parameters needed for kinematics and cable-torque conversion
    L1, L2 = robot._get_link_lengths()        # Link lengths [m]
    r_p = robot.r_p                           # Pulley radius [m]
    print(f"  L1={L1*1e3:.1f} mm  L2={L2*1e3:.1f} mm  r_p={r_p*1e3:.2f} mm")

    return L1, L2, r_p


def setup_trajectory(args, robot, L1, L2):
    """Build the end-effector reference trajectory for training.

    Constructs the main repeating trajectory (circle / rect / line) and prepends
    a smooth preamble that moves the EE from its current rest position to the
    trajectory start, avoiding discontinuous velocity jumps at episode start.

    Args:
        args:   parsed argparse Namespace (traj_type, episode_steps, dt, move_duration)
        robot:  initialized CupManipulatorTendonIsaac (needed for current joint angles)
        L1:     proximal link length [m]
        L2:     distal link length [m]

    Returns:
        traj_source: PreambleTrajectorySource that yields (pos, vel) reference at any t
    """
    lap_duration = args.episode_steps * args.dt  # Total time for one lap [s]

    # Build the main repeating trajectory shape
    if args.traj_type == "circle":
        main_traj = CircleTrajectory(
            cx=0.4, cy=0.00, radius=0.1, lap_duration=lap_duration, N=60,
        )
    elif args.traj_type == "rect":
        main_traj = RectTrajectory(
            x_range=(0.49, 0.51), y_range=(-0.08, 0.08),
            lap_duration=lap_duration, N=60,
            v_max=0.9, v_corner=0.05, corner_blend=0.35,
        )
    else:  # line
        from controller.trajectory import LineTrajectory
        main_traj = LineTrajectory(
            cx=0.4, cy=0.0, radius=0.1, lap_duration=lap_duration, N=60,
        )

    # Build a smooth preamble to move the EE from its current position to the
    # trajectory start, matching the initial velocity of the main trajectory.
    # This prevents discontinuous velocity jumps and large transient errors
    # at the beginning of each episode.
    q_cur  = robot.get_positions_user_order()       # Current joint angles [q1, q2]
    ee_cur = forward_kinematics_2r(L1, L2, *q_cur)  # Current EE position from FK
    first_ee  = main_traj.eval_position(0.0)         # EE position at t=0 of main traj
    first_vel = main_traj.eval_velocity(0.0)         # Desired EE velocity at t=0

    preamble = build_move_to_start(
        p_start=ee_cur, p_end=first_ee, v_end=first_vel,
        duration=args.move_duration,  # Time to reach starting point [s]
    )

    # PreambleTrajectorySource plays the preamble once (t=0 → move_duration),
    # then loops main_traj indefinitely.
    return PreambleTrajectorySource(preamble, main_traj)


def setup_controller_and_sea(args, robot, motor, motor_mode):
    """Instantiate the CT controller and SEA cable actuator.

    The Computed Torque (CT) controller provides model-based feedforward/feedback
    torques.  The SEA actuator then applies those torques through a compliant cable
    spring, introducing realistic force-transmission lag that the PPO policy must
    learn to compensate for.

    Args:
        args:       parsed argparse Namespace (CT gains, SEA parameters, dt, etc.)
        robot:      initialized CupManipulatorTendonIsaac (provides initial joint state)
        motor:      motor config object from catalog (inertia, peak torque, gear ratio)
        motor_mode: MotorMode enum — "torque" (2nd-order) or "position" (1st-order servo)

    Returns:
        ct:  ComputedTorqueController
        sea: SEACableActuatorNP (initialized at current joint-2 position)
    """
    # ── CT Controller ─────────────────────────────────────────────────────────
    # Computed Torque uses feedback linearization:
    #   τ_des = M(q)·a_ref + C(q,qdot)·qdot + G(q)
    # where  a_ref = Kp·e_pos + Kd·e_vel  (PD tracking error)
    # The result is the desired joint torque before SEA dynamics.
    ct = ComputedTorqueController(
        Kp=args.ct_kp,           # Proportional gain [Nm·s/rad]
        Kd=args.ct_kd,           # Derivative gain   [Nm·s²/rad]
        tau_max=args.ct_tau_max, # Torque saturation limit [Nm]
        pulley_radius=robot.r_p, # Cable-to-torque conversion factor
    )

    # ── SEA Actuator ──────────────────────────────────────────────────────────
    # Series Elastic Actuator transmits force through a compliant cable spring:
    #   - Motor rotates → cable pulls → spring compresses → torque = k_s * δ
    #   - Motor dynamics add lag between desired and applied torque
    #   - Two modes: 2nd-order torque tracking or 1st-order position servo
    # This lag is what the PPO residual policy learns to predict and compensate.
    sea = SEACableActuatorNP(
        r_p=robot.r_p,                              # Pulley radius [m]
        k_s=args.spring_stiffness,            # Spring stiffness  [Nm/rad]
        b_c=args.cable_damping,               # Cable damping     [Nm·s/rad]
        tau_max=args.ct_tau_max,              # Motor output limit [Nm]
        dt=args.dt,                           # Integration step  [s]
        motor_mode=motor_mode,                # "torque" or "position"
        motor_cfg=motor,                      # Motor catalog entry (inertia, peak, gear)
        omega_m=args.motor_bandwidth,         # Servo bandwidth [rad/s] (position mode)
        motor_substeps=args.motor_substeps,   # Sub-steps per physics step (auto if None)
    )
    # Warm-start motor state at the current joint-2 angle to avoid a transient
    # at the very start of the first episode.
    q_cur = robot.get_positions_user_order()
    sea.initialize(q_cur[1])

    return ct, sea




# ═════════════════════════════════════════════════════════════════════════════
# Scene helpers
# ═════════════════════════════════════════════════════════════════════════════

def resolve_motor_defaults(args):
    """Resolve motor catalog entry and fill in CT torque-saturation default.

    If --ct-tau-max was not provided on the CLI, sets it to the motor's peak
    joint-side torque (after gear reduction).

    Args:
        args: parsed Namespace (motor, sea_mode, ct_tau_max)

    Returns:
        motor:      motor config object from catalog
        motor_mode: MotorMode enum
    """
    motor      = get_motor(args.motor)
    motor_mode = MotorMode(args.sea_mode)
    if args.ct_tau_max is None:
        args.ct_tau_max = motor.peak_torque_joint
    return motor, motor_mode


def build_robot_config():
    """Construct the manipulator URDF config from the workspace model folder.

    Returns:
        config: ManipulatorConfig dataclass (frozen, JSON-serialisable)
    """
    urdf_path = str(
        Path(__file__).resolve().parent.parent
        / "model_using_onshape_to_robot"
        / "manipulator_cable"
        / "manipulator_cable_obj.urdf"
    )
    return create_cable_manipulator_config(
        urdf_path=urdf_path,
        joint_angles={
            "link1_base":  math.radians(10.0),   # Initial joint-1 angle
            "link2_link1": math.radians(-10.0),  # Initial joint-2 angle
        },
        damping=(0.05, 0.05),    # Small passive damping (models friction)
        stiffness=(0.0, 0.0),    # No passive springs (cable provides active restoring)
    )


# ═════════════════════════════════════════════════════════════════════════════
# Run directory & args persistence
# ═════════════════════════════════════════════════════════════════════════════

def create_run_dir(args):
    """Create a timestamped run directory with sub-folders.

    Layout::

        rl/checkpoints/run_<YYYYMMDD_HHMMSS>/
            args.json          # frozen CLI arguments
            best_model.zip     # best checkpoint (updated by EvalCallback)
            checkpoints/       # periodic checkpoints
            plots/             # evaluation plots

    Also writes args.json immediately so the run’s config is recorded even
    if training crashes.

    Args:
        args: parsed Namespace

    Returns:
        run_dir: Path to the run directory
        stamp:   timestamp string (YYYYmmdd_HHMMSS)
    """
    stamp   = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(_PROJECT_ROOT) / args.checkpoint_dir / f"run_{stamp}"

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Persist all arguments as JSON for reproducibility
    args_path = run_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[train] Run directory : {run_dir}")
    print(f"[train] Args saved to : {args_path}")

    return run_dir, stamp


# ═════════════════════════════════════════════════════════════════════════════
# Weights & Biases
# ═════════════════════════════════════════════════════════════════════════════

def setup_wandb(args, run_name=None):
    """Initialise a W&B run if --wandb is set.

    API key resolution order:
      1. WANDB_API_KEY environment variable (already set by shell / CI)
      2. .env.wandb file in the project root  (local secret file, git-ignored)
      3. Previously stored key from `wandb login`

    Logs all CLI arguments as the run config so every hyperparameter is
    recorded and searchable in the W&B dashboard.

    Args:
        args:     parsed Namespace (wandb, wandb_project, wandb_entity, …)
        run_name: optional display name for this run in the W&B dashboard

    Returns:
        wandb.Run or None (if wandb disabled or unavailable)
    """
    if not args.wandb:
        print("[train] W&B disabled — pass --wandb to enable logging")
        return None
    if not _WANDB_AVAILABLE:
        print("[train] --wandb requested but `wandb` is not installed. "
              "Run: pip install wandb")
        return None

    # Load API key from .env.wandb if not already in the environment
    if not os.environ.get("WANDB_API_KEY"):
        env_file = Path(__file__).resolve().parent / ".env.wandb"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("WANDB_API_KEY=") and not line.startswith("#"):
                    os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()
                    print("[train] W&B API key loaded from .env.wandb")
                    break
        else:
            print("[train] WARNING: .env.wandb not found and WANDB_API_KEY not set")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        sync_tensorboard=True,   # auto-upload TB logs
        save_code=True,          # snapshot of this script
    )
    print("\n" + "=" * 60)
    print(f"  W&B run name : {run.name}")
    print(f"  W&B URL      : {run.url}")
    print("=" * 60 + "\n")
    return run


# ═════════════════════════════════════════════════════════════════════════════
# RL helpers
# ═════════════════════════════════════════════════════════════════════════════

def setup_env(args, robot, world, ct, sea, traj_source, L1, L2):
    """Wrap the Isaac Sim simulation in a Gymnasium environment + Monitor.

    Args:
        args:        parsed Namespace (dt, episode_steps, residual_max, reward weights, render)
        robot:       initialized CupManipulatorTendonIsaac
        world:       Isaac Sim World
        ct:          ComputedTorqueController
        sea:         SEACableActuatorNP
        traj_source: PreambleTrajectorySource
        L1, L2:      link lengths [m]

    Returns:
        env: Monitor-wrapped ManipulatorResidualEnv
    """
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
            "tracking":        args.w_tracking,
            "effort":          args.w_effort,
            "smoothness":      args.w_smoothness,
            "torque_tracking": args.w_torque_tracking,
        },
        render_mode=args.render,
        solve_ik_fn=solve_2r_ik,
        fk_fn=forward_kinematics_2r,
        ik_to_jt_fn=ik_to_joint_space_references,
    )
    return Monitor(env)


def setup_ppo(args, env, run_dir):
    """Configure PPO: either resume from checkpoint or start fresh; attach logger.

    Args:
        args:    parsed Namespace (all PPO hyperparams, resume path)
        env:     Monitor-wrapped Gymnasium environment
        run_dir: Path to the run directory (TB logs written here)

    Returns:
        model: stable_baselines3 PPO instance ready for model.learn()
    """
    print("[train] Configuring PPO…")
    log_dir = run_dir / "tb_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="cpu",  # Isaac Sim owns the GPU; run policy on CPU to avoid contention
    )

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, **{
            k: v for k, v in ppo_kwargs.items() if k not in ("policy", "env")
        })
    else:
        model = PPO(**ppo_kwargs)

    # Logger: always stdout + CSV; add tensorboard if available (required for
    # wandb sync_tensorboard to pick up SB3 scalars automatically).
    _log_formats = ["stdout", "csv"]
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        _log_formats.append("tensorboard")
    except ImportError:
        if args.wandb:
            print("[train] tensorboard not installed — wandb sync_tensorboard will not work")
        print("[train] tensorboard not installed — skipping TB logging")
    model.set_logger(configure(str(log_dir), _log_formats))
    return model


def setup_callbacks(args, run_dir, wandb_run=None):
    """Build training callbacks: checkpoint saver, best-model tracker, optional W&B.

    Checkpoints are saved under ``run_dir/checkpoints/``.  The best model
    (by mean episode reward) is kept at ``run_dir/best_model``.

    Args:
        args:      parsed Namespace (save_freq)
        run_dir:   Path to the run directory
        wandb_run: active wandb.Run or None

    Returns:
        CallbackList wrapping all active callbacks
    """
    ckpt_sub = run_dir / "checkpoints"
    ckpt_sub.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(ckpt_sub),
            name_prefix="ppo_residual",
        ),
    ]

    if wandb_run is not None and _WANDB_AVAILABLE:
        callbacks.append(WandbCallback(
            model_save_path=str(ckpt_sub),
            model_save_freq=args.save_freq,
            verbose=2,
        ))

    return CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]


# ═════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═════════════════════════════════════════════════════════════════════════════

def run_training(args, model, callback):
    """Run the PPO training loop and print a startup banner.

    Args:
        args:     parsed Namespace (total_timesteps + all hyperparams for banner)
        model:    configured PPO model
        callback: CheckpointCallback (or CallbackList)
    """
    print(f"\n[train] Starting PPO training — {args.total_timesteps} steps")
    print(f"  residual_max = {args.residual_max} Nm")
    print(f"  Motor : {args.motor}  mode={args.sea_mode}  substeps={args.motor_substeps or 'auto'}")
    print(f"  CT    : Kp={args.ct_kp}  Kd={args.ct_kd}  τ_max={args.ct_tau_max:.2f} Nm")
    print(f"  SEA   : k_s={args.spring_stiffness}  b_c={args.cable_damping}")
    print(f"  Reward: track={args.w_tracking}  effort={args.w_effort}  "
          f"smooth={args.w_smoothness}  τ_track={args.w_torque_tracking}\n")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[train] Training interrupted by user.")


def save_model(model, run_dir):
    """Save the final model into the run directory + a 'best_model' copy.

    Args:
        model:   trained PPO model
        run_dir: Path to the run directory
    """
    final_path  = run_dir / "final_model"
    best_path   = run_dir / "best_model"
    model.save(str(final_path))
    print(f"\n[train] Final model saved : {final_path}.zip")
    model.save(str(best_path))
    print(f"[train] Best model saved  : {best_path}.zip")


# ═════════════════════════════════════════════════════════════════════════════
# Post-training evaluation
# ═════════════════════════════════════════════════════════════════════════════

def _collect_eval_episode(model, use_model: bool, n_steps: int, dt: float) -> dict:
    """Run one deterministic episode through the already-trained model's env.

    Uses ``model.env`` (SB3's internal DummyVecEnv) so no second Isaac Sim
    instance or environment reset is needed — the same GPU physics session
    that trained the policy is reused.

    Args:
        model:     trained SB3 PPO model
        use_model: True = run trained policy, False = zero residuals (CT-only)
        n_steps:   max steps to simulate
        dt:        physics timestep [s] (for time axis)

    Returns:
        dict of numpy arrays: t, ee_err_mm, ee_actual, ee_ref,
                              tau_residual, tau_ct, delta, label
    """
    vec_env = model.env
    obs = vec_env.reset()
    action_dim = vec_env.action_space.shape[0]

    log = {k: [] for k in ("t", "ee_err_mm", "ee_actual", "ee_ref",
                            "tau_residual", "tau_ct", "delta")}

    for step in range(n_steps):
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros((1, action_dim), dtype=np.float32)

        # DummyVecEnv.step returns (obs, reward, dones, infos)
        obs, _, dones, infos = vec_env.step(action)
        info = infos[0]

        log["t"].append(info.get("t", step * dt))
        log["ee_err_mm"].append(info.get("ee_error_mm", 0.0))
        log["ee_actual"].append(info.get("ee_actual", np.zeros(2)).copy())
        log["ee_ref"].append(info.get("ee_ref", np.zeros(2)).copy())
        log["tau_residual"].append(info.get("tau_residual", np.zeros(2)).copy())
        log["tau_ct"].append(info.get("tau_ct", np.zeros(2)).copy())

        raw = vec_env.envs[0].env.unwrapped
        diag = getattr(raw, "_last_sea_diag", None)
        log["delta"].append(diag.delta if diag else 0.0)

        if dones[0]:
            break

    return {k: np.array(v) for k, v in log.items()}


def run_post_training_eval(args, model, run_dir, wandb_run):
    """Run CT-only and CT+RL eval episodes after training; upload plots to W&B.

    Reuses the existing DummyVecEnv from ``model.env`` — no second
    SimulationApp or new environment is created.  Two episodes are run:

    1. **CT-only** — zero residual actions (baseline)
    2. **CT+RL**   — trained policy (deterministic)

    A 2×2 overview figure is saved to ``run_dir/plots/`` and optionally
    uploaded to W&B as ``eval/overview``.

    Args:
        args:      parsed Namespace (eval_steps, dt, move_duration, wandb)
        model:     trained PPO model (env already attached)
        run_dir:   Path to the run directory
        wandb_run: active wandb.Run or None
    """
    print("\n[train] Running post-training eval…")

    # ── CT-only baseline ──────────────────────────────────────────────────────
    print("  [eval] CT-only baseline…")
    baseline = _collect_eval_episode(model, use_model=False,
                                     n_steps=args.eval_steps, dt=args.dt)
    baseline["label"] = "CT only"

    # Re-initialise SEA spring so CT+RL starts from the same mechanical state
    raw_env = model.env.envs[0].env.unwrapped
    raw_env.sea.initialize(raw_env.robot.get_positions_user_order()[1])

    # ── CT+RL ─────────────────────────────────────────────────────────────────
    print("  [eval] CT+RL policy…")
    rl_result = _collect_eval_episode(model, use_model=True,
                                      n_steps=args.eval_steps, dt=args.dt)
    rl_result["label"] = "CT + RL"

    results = [baseline, rl_result]
    _COLORS = {"CT only": "tab:red", "CT + RL": "tab:blue"}

    # ── Summary statistics ────────────────────────────────────────────────────
    preamble_steps = int(args.move_duration / args.dt)
    summary = {}
    print(f"  [eval] Summary (post-preamble, t > {args.move_duration}s):")
    for r in results:
        idx  = np.arange(len(r["ee_err_mm"]))
        mask = idx > preamble_steps
        if mask.any():
            mean_e = float(r["ee_err_mm"][mask].mean())
            max_e  = float(r["ee_err_mm"][mask].max())
            summary[r["label"]] = {"mean": mean_e, "max": max_e}
            print(f"    {r['label']:12s}  mean={mean_e:.2f} mm  max={max_e:.2f} mm")

    # ── Figure (2 × 2) ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title_parts = [f"{lbl}: {s['mean']:.1f} mm" for lbl, s in summary.items()]
    fig.suptitle(
        "Post-Training Eval   |   " + "   |   ".join(title_parts),
        fontsize=11, fontweight="bold",
    )

    # [0,0] EE XY path
    ax = axes[0, 0]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["ee_ref"][:, 0] * 1e3, r["ee_ref"][:, 1] * 1e3,
                "--", color="gray", lw=0.8, alpha=0.5)
        ax.plot(r["ee_actual"][:, 0] * 1e3, r["ee_actual"][:, 1] * 1e3,
                "-", color=c, lw=1.4, label=r["label"])
        ax.plot(r["ee_actual"][0, 0] * 1e3, r["ee_actual"][0, 1] * 1e3,
                "o", color=c, ms=5)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("EE Path  (solid=actual, dashed=ref)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # [0,1] Tracking error vs time
    ax = axes[0, 1]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], r["ee_err_mm"], "-", color=c, lw=1.2, label=r["label"])
    ax.axvline(args.move_duration, color="k", lw=0.8, ls="--", label="preamble end")
    for lbl, s in summary.items():
        c = _COLORS.get(lbl, "tab:blue")
        ax.axhline(s["mean"], color=c, lw=0.9, ls=":",
                   label=f"{lbl} mean={s['mean']:.1f} mm")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("EE error [mm]")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # [1,0] CT desired torque joint-2
    ax = axes[1, 0]
    for r in results:
        c = _COLORS.get(r["label"], "tab:blue")
        ax.plot(r["t"], r["tau_ct"][:, 1], "-", color=c, lw=1.0,
                alpha=0.8, label=r["label"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("τ₂ [Nm]")
    ax.set_title("CT Desired Torque (joint 2)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # [1,1] RL residual torques
    ax = axes[1, 1]
    rl = next((r for r in results if r["label"] == "CT + RL"), results[-1])
    ax.plot(rl["t"], rl["tau_residual"][:, 0], lw=1.0,
            color="tab:blue", label="Δτ₁")
    ax.plot(rl["t"], rl["tau_residual"][:, 1], lw=1.0,
            color="tab:cyan", label="Δτ₂")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Residual τ [Nm]")
    ax.set_title("RL Residual Torques")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.tight_layout()

    # ── Save locally ──────────────────────────────────────────────────────────
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = plots_dir / f"eval_final_{stamp}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"  [eval] Plot saved: {plot_path}")

    # ── Upload to W&B ─────────────────────────────────────────────────────────
    if wandb_run is not None and _WANDB_AVAILABLE:
        log_dict = {"eval/overview": wandb.Image(fig)}
        for lbl, s in summary.items():
            prefix = "eval/" + lbl.replace(" ", "_").lower()
            log_dict[f"{prefix}/mean_err_mm"] = s["mean"]
            log_dict[f"{prefix}/max_err_mm"]  = s["max"]
        if "CT only" in summary and "CT + RL" in summary:
            base = summary["CT only"]["mean"]
            if base > 0:
                log_dict["eval/improvement_pct"] = (
                    100.0 * (base - summary["CT + RL"]["mean"]) / base
                )
        wandb_run.log(log_dict)
        print("  [eval] Plots and metrics uploaded to W&B.")

    plt.close(fig)


def cleanup(env, wandb_run=None):
    """Close the environment, restore the SIGINT handler, and shut down Isaac Sim.

    Args:
        env:       Monitor-wrapped environment to close
        wandb_run: active wandb.Run or None (will call wandb.finish())
    """
    if wandb_run is not None:
        wandb_run.finish()
        print("[train] W&B run finished.")
    signal.signal(signal.SIGINT, _orig_sigint)
    env.close()
    simulation_app.close()
    print("[train] Done.")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main(args):
    """Orchestrate the full training pipeline.

    Pipeline
    ────────
    1.  resolve_motor_defaults   — fill CT τ_max from motor catalog
    2.  build_robot_config       — URDF path + joint / passive-dynamics config
    3.  setup_world              — Isaac Sim World + ground plane
    4.  setup_robot              — load URDF, weld base, add EE frame + actuators
    5.  initialize_simulation    — world.reset(), GPU tensors, geometry params
    6.  setup_trajectory         — preamble + repeating EE reference path
    7.  setup_controller_and_sea — CT controller + SEA cable actuator
    8.  setup_env                — Gymnasium wrapper + Monitor
    9.  setup_ppo                — PPO config, logger, optional resume
    10. setup_callbacks          — periodic checkpoint saver
    11. run_training             — model.learn() loop  →  returns timestamp
    12. save_model               — timestamped final + 'latest' checkpoint
    13. cleanup                  — close env + Isaac Sim

    Args:
        args: parsed argparse Namespace from parse_args()
    """
    print("\n[train] Setting up Isaac Sim scene…")

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir, stamp      = create_run_dir(args)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run           = setup_wandb(args, run_name=f"run_{stamp}")

    # ── Motor ─────────────────────────────────────────────────────────────────
    motor, motor_mode   = resolve_motor_defaults(args)

    # ── Scene ─────────────────────────────────────────────────────────────────
    config              = build_robot_config()
    world               = setup_world(args)
    robot               = setup_robot(config, world, args)
    L1, L2, r_p         = initialize_simulation(world, robot)

    # ── Control ───────────────────────────────────────────────────────────────
    traj_source         = setup_trajectory(args, robot, L1, L2)
    ct, sea             = setup_controller_and_sea(args, robot, motor, motor_mode)

    # ── RL ────────────────────────────────────────────────────────────────────
    env                 = setup_env(args, robot, world, ct, sea, traj_source, L1, L2)
    model               = setup_ppo(args, env, run_dir)
    callback            = setup_callbacks(args, run_dir, wandb_run)

    # ── Train + persist ───────────────────────────────────────────────────────
    run_training(args, model, callback)
    save_model(model, run_dir)

    # ── Post-training eval ────────────────────────────────────────────────────
    if args.eval_after_training:
        run_post_training_eval(args, model, run_dir, wandb_run)

    cleanup(env, wandb_run)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Argument parser
    # ═════════════════════════════════════════════════════════════════════════════
    """Build and return the CLI argument parser for PPO training."""
    parser = argparse.ArgumentParser(description="Train residual-RL PPO for manipulator")

    # Rendering and simulation
    parser.add_argument("--render", default=_render_mode,
                        choices=["native", "headless"],
                        help="Rendering mode: 'native' shows 3D window, 'headless' runs CPU-only (faster)")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training steps. Approx episodes = steps / 2000")
    parser.add_argument("--episode-steps", type=int, default=2000,
                        help="Max steps per episode (default: 2000 ≈ 20s at dt=0.01s)")
    parser.add_argument("--dt", type=float, default=1.0 / 100.0,
                        help="Physics simulation timestep [s] (default: 0.01s = 100Hz)")
    parser.add_argument("--move-duration", type=float, default=3.0,
                        help="Duration of preamble trajectory to move EE to start [s]")

    # CT (Computed Torque) controller gains
    parser.add_argument("--ct-kp", type=float, default=100.0,
                        help="CT proportional gain [Nm·s/rad] for trajectory tracking")
    parser.add_argument("--ct-kd", type=float, default=40.0,
                        help="CT derivative gain [Nm·s²/rad] for damping")
    parser.add_argument("--ct-tau-max", type=float, default=None,
                        help="Torque saturation [Nm]. Default: motor peak_torque_joint")

    # Motor model selection and dynamics
    parser.add_argument("--motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config",
                        help="CubeMars motor model for the elbow joint")
    parser.add_argument("--sea-mode", choices=["torque", "position"], default="torque",
                        help="Motor control mode: 'torque' (2nd-order model) or 'position' (1st-order servo)")

    # SEA (Series Elastic Actuator) parameters
    parser.add_argument("--spring-stiffness", type=float, default=30.0,
                        help="Cable spring stiffness [Nm/rad]")
    parser.add_argument("--cable-damping", type=float, default=2.0,
                        help="Cable damping coefficient [Nm·s/rad]")
    parser.add_argument("--motor-bandwidth", type=float, default=100.0,
                        help="Motor servo bandwidth [rad/s] — only used in position servo mode")
    parser.add_argument("--residual-max", type=float, default=5.0,
                        help="Max magnitude of PPO residual torque command [Nm]")
    parser.add_argument("--motor-substeps", type=int, default=None,
                        help="Motor integration sub-steps per physics step (None=auto)")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Adam optimizer learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per rollout before policy update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size for policy gradient updates")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of passes through rollout buffer per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for return estimation")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clipping range for surrogate objective (20%)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy regularization coefficient (default=0 = no entropy bonus)")

    # Reward function weights
    parser.add_argument("--w-tracking", type=float, default=100.0,
                        help="Weight on end-effector tracking error penalty")
    parser.add_argument("--w-effort", type=float, default=0.01,
                        help="Weight on torque effort penalty")
    parser.add_argument("--w-smoothness", type=float, default=0.001,
                        help="Weight on torque jerk (smoothness) penalty")
    parser.add_argument("--w-torque-tracking", type=float, default=1.0,
                        help="Weight on SEA torque (desired vs applied) tracking penalty")

    # Trajectory type
    parser.add_argument("--traj-type", default="rect",
                        choices=["circle", "rect", "line"],
                        help="End-effector trajectory shape: circle, rectangle, or line")

    # Checkpointing and logging
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Save model checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model .zip to resume training from (without extension)")
    parser.add_argument("--eval-after-training", action="store_true", default=True,
                        help="Run CT-only vs CT+RL eval after training and upload plots to W&B")
    parser.add_argument("--no-eval-after-training", dest="eval_after_training",
                        action="store_false",
                        help="Skip post-training eval")
    parser.add_argument("--eval-steps", type=int, default=2000,
                        help="Steps per eval episode (default: same as episode_steps)")
    parser.add_argument("--log-dir", type=str, default="rl/logs",
                        help="Directory for tensorboard/CSV logs")
    parser.add_argument("--checkpoint-dir", type=str, default="rl/checkpoints",
                        help="Directory to save model checkpoints")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="isaac_sim_robotics",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="dbha483-transgp",
                        help="W&B entity (team or username)")

    main(parser.parse_args())
