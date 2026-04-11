"""
rl/envs/manipulator_residual_env.py
────────────────────────────────────
Gymnasium environment for **Residual RL** on a 2-DOF cable-driven
manipulator with SEA actuation in Isaac Sim.

Architecture
────────────
    ┌──────────┐    τ_ct     ┌─────────┐   τ_ct + Δτ   ┌─────┐   τ_sea   ┌───────┐
    │    CT    │──────────▶│  RL Add  │──────────────▶│ SEA │──────────▶│ Plant │
    │Controller│            │  Δτ      │               │Cable│           │PhysX  │
    └──────────┘            └─────────┘               └─────┘           └───────┘
         ▲                       ▲                                          │
         │                       │           observation                    │
         └───────────────────────┴──────────────────────────────────────────┘

The CT controller runs the standard inverse-dynamics law.  The RL agent
outputs a *small* residual torque Δτ ∈ [-δ_max, δ_max] that is *added*
to the CT output *before* the SEA spring-damper model.  This lets the
RL agent compensate for:
  - SEA phase lag & spring compliance
  - Model inaccuracies in M(q), h(q,q̇)
  - Trajectory-dependent dynamics

Observation (14-D)
──────────────────
    [q₁, q₂, q̇₁, q̇₂, ee_err_x, ee_err_y,
     δ, δ̇, F_cable, τ₂_ct, τ₂_sea, τ_motor,
     (τ₁_des − τ₁_applied), (τ₂_des − τ₂_applied)]

Action (2-D)
────────────
    [Δτ₁, Δτ₂]  residual torques  ∈ [-residual_max, residual_max]

Reward
──────
    r = -α·‖ee_err‖² - β·‖Δτ‖² - γ·‖q̈‖² - λ·(τ_des − τ_applied)²
      tracking     effort       smoothness    torque tracking

Episode terminates after ``max_episode_steps`` or if joint limits exceeded.
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any


class ManipulatorResidualEnv(gym.Env):
    """Residual-RL Gymnasium wrapper around the Isaac Sim 2-DOF manipulator.

    This environment is designed to be instantiated **after** Isaac Sim's
    ``SimulationApp`` and ``World`` are already running.  It does NOT create
    its own SimulationApp — it receives pre-built robot / controller / SEA
    objects and drives the existing World.

    Parameters
    ----------
    robot : CupManipulatorTendonIsaac
        Initialized robot with dynamics view ready.
    world : omni.isaac.core.World
        Isaac Sim world (already reset).
    ct_controller : ComputedTorqueController
        Baseline CT controller.
    sea_actuator : SEACableActuatorNP
        SEA cable actuator (initialized).
    traj_source : PreambleTrajectorySource
        Trajectory provider (preamble + main).
    L1, L2 : float
        Link lengths [m].
    dt : float
        Physics timestep [s].
    max_episode_steps : int
        Steps per episode.
    residual_max : float
        Maximum residual torque magnitude [Nm].
    reward_weights : dict
        Keys: 'tracking', 'effort', 'smoothness'.
    render_mode : str
        'native', 'headless', etc.  Passed to world.step().
    """

    metadata = {"render_modes": ["native", "headless"]}

    def __init__(
        self,
        robot,
        world,
        ct_controller,
        sea_actuator,
        traj_source,
        L1: float,
        L2: float,
        dt: float = 0.01,
        max_episode_steps: int = 2000,
        residual_max: float = 5.0,
        reward_weights: Optional[dict] = None,
        render_mode: str = "headless",
        solve_ik_fn=None,
        fk_fn=None,
        ik_to_jt_fn=None,
    ):
        super().__init__()

        self.robot = robot
        self.world = world
        self.ct = ct_controller
        self.sea = sea_actuator
        self.traj_source = traj_source
        self.L1, self.L2 = L1, L2
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.residual_max = residual_max
        self.render_mode = render_mode
        self.solve_ik_fn = solve_ik_fn
        self.fk_fn = fk_fn
        self.ik_to_jt_fn = ik_to_jt_fn

        # Reward weights
        w = reward_weights or {}
        self.w_track = w.get("tracking", 100.0)
        self.w_effort = w.get("effort", 0.01)
        self.w_smooth = w.get("smoothness", 0.001)
        self.w_torque_track = w.get("torque_tracking", 1.0)

        # ── Observation space (14-D) ────────────────────────────────────────
        #   [q1, q2, q1_dot, q2_dot, ee_err_x, ee_err_y,
        #    delta, delta_dot, F_cable, tau2_ct, tau2_sea, tau_motor,
        #    tau1_tracking_err, tau2_tracking_err]
        obs_high = np.array([
            np.pi,       # q1
            np.pi,       # q2
            10.0,        # q1_dot
            10.0,        # q2_dot
            0.5,         # ee_err_x
            0.5,         # ee_err_y
            0.1,         # delta (spring extension)
            5.0,         # delta_dot (spring extension rate)
            500.0,       # F_cable
            50.0,        # tau2_ct
            50.0,        # tau2_sea
            50.0,        # tau_motor
            50.0,        # tau1_tracking_err
            50.0,        # tau2_tracking_err
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32,
        )

        # ── Action space (2-D): residual torques ────────────────────────────
        self.action_space = spaces.Box(
            low=-residual_max,
            high=residual_max,
            shape=(2,),
            dtype=np.float32,
        )

        # ── Internal state ──────────────────────────────────────────────────
        self._step_count = 0
        self._t = 0.0
        self._q_seed = None
        self._prev_tau = np.zeros(2)
        self._prev_delta = 0.0
        self._last_sea_diag = None
        self._last_ct_out = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_count = 0
        self._t = 0.0
        self._prev_tau = np.zeros(2)
        self._prev_delta = 0.0

        # Randomize initial joint angles slightly for domain randomization
        jc = self.robot.config.joint_configs
        q_nom = np.array([
            jc["link1_base"].position if "link1_base" in jc else math.radians(10.0),
            jc["link2_link1"].position if "link2_link1" in jc else math.radians(-10.0),
        ])
        if self.np_random is not None:
            q_init = q_nom + self.np_random.uniform(-0.05, 0.05, size=2)
        else:
            q_init = q_nom

        # Reset robot state
        self.robot.set_positions_user_order(q_init)
        self.robot.set_velocities_user_order(np.zeros(2))

        # Reset SEA motor position
        self.sea.initialize(q_init[1])

        # IK seed at first trajectory point
        first_ee = self.traj_source.eval_position(0.0)
        q_wp, ok = self.solve_ik_fn(self.L1, self.L2, first_ee, q_init)
        self._q_seed = q_wp if ok else q_init.copy()

        # Zero-out diagnostics
        self._last_sea_diag = None
        self._last_ct_out = None

        # Step world once to settle physics
        self.world.step(render=(self.render_mode == "native"))

        obs = self._get_obs()
        info = {"t": 0.0, "ee_error": 0.0}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64).clip(
            -self.residual_max, self.residual_max,
        )

        # ── 1. Read robot state ─────────────────────────────────────────────
        q = self.robot.get_positions_user_order()
        q_dot = self.robot.get_velocities_user_order()

        # ── 2. Trajectory reference ─────────────────────────────────────────
        ee_ref = self.traj_source.eval_position(self._t)
        ee_vel = self.traj_source.eval_velocity(self._t)
        ee_acc = self.traj_source.eval_acceleration(self._t)

        # ── 3. IK → joint-space references ──────────────────────────────────
        q_des, q_dot_ref, q_ddot_ref, ik_ok = self.ik_to_jt_fn(
            ee_ref, ee_vel, ee_acc,
            self.L1, self.L2,
            self._q_seed,
            self.solve_ik_fn,
        )
        if ik_ok:
            self._q_seed = q_des.copy()

        # ── 4. Dynamics ─────────────────────────────────────────────────────
        M = self.robot.get_mass_matrix()
        h = self.robot.get_bias_forces()

        # ── 5. CT baseline torque ───────────────────────────────────────────
        ct_out = self.ct.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)
        self._last_ct_out = ct_out

        # ── 6. Add residual RL torque ───────────────────────────────────────
        tau_combined = ct_out.tau_raw + action

        # ── 7. SEA actuator ─────────────────────────────────────────────────
        tau_applied, sea_diag = self.sea.step(tau_combined, q, q_dot)
        self._last_sea_diag = sea_diag

        # ── 8. Apply to plant ───────────────────────────────────────────────
        self.robot.set_joint_torques(tau_applied)

        # ── 9. Physics step ─────────────────────────────────────────────────
        self.world.step(render=(self.render_mode == "native"))

        # ── 10. Compute reward ──────────────────────────────────────────────
        q_new = self.robot.get_positions_user_order()
        ee_actual = self.fk_fn(self.L1, self.L2, q_new[0], q_new[1])
        ee_err = ee_actual - ee_ref
        ee_err_norm = np.linalg.norm(ee_err)

        # Reward components
        r_track = -self.w_track * ee_err_norm ** 2
        r_effort = -self.w_effort * np.sum(action ** 2)
        tau_diff = tau_applied - self._prev_tau
        r_smooth = -self.w_smooth * np.sum(tau_diff ** 2)
        # Torque tracking: penalize gap between CT desired and SEA actual
        tau_track_err = ct_out.tau_raw - tau_applied
        r_torque_track = -self.w_torque_track * np.sum(tau_track_err ** 2)
        reward = float(r_track + r_effort + r_smooth + r_torque_track)

        self._prev_tau = tau_applied.copy()

        # ── 11. Termination ─────────────────────────────────────────────────
        self._step_count += 1
        self._t += self.dt

        # Terminate on joint limit violation
        q_abs = np.abs(q_new)
        terminated = bool(np.any(q_abs > np.deg2rad(150)))

        # Truncation: max steps reached
        truncated = self._step_count >= self.max_episode_steps

        # ── 12. Observation & info ──────────────────────────────────────────
        obs = self._get_obs()
        info = {
            "t": self._t,
            "ee_error_mm": ee_err_norm * 1e3,
            "ee_actual": ee_actual.copy(),
            "ee_ref": ee_ref.copy(),
            "r_track": r_track,
            "r_effort": r_effort,
            "r_smooth": r_smooth,
            "r_torque_track": r_torque_track,
            "tau_ct": ct_out.tau_raw.copy(),
            "tau_residual": action.copy(),
            "tau_applied": tau_applied.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build the 14-D observation vector."""
        q = self.robot.get_positions_user_order()
        q_dot = self.robot.get_velocities_user_order()

        # EE error
        ee_actual = self.fk_fn(self.L1, self.L2, q[0], q[1])
        ee_ref = self.traj_source.eval_position(self._t)
        ee_err = ee_actual - ee_ref

        # SEA state
        diag = self._last_sea_diag
        delta = diag.delta if diag else 0.0
        delta_dot = (delta - self._prev_delta) / self.dt if self.dt > 0 else 0.0
        self._prev_delta = delta
        F_cable = diag.F_cable if diag else 0.0
        tau2_ct = self._last_ct_out.tau_raw[1] if self._last_ct_out else 0.0
        tau2_sea = diag.tau_sea if diag else 0.0
        tau_motor = diag.tau_motor if diag else 0.0

        # Torque tracking error: τ_des − τ_applied
        tau_ct_raw = self._last_ct_out.tau_raw if self._last_ct_out else np.zeros(2)
        tau1_track_err = tau_ct_raw[0] - self._prev_tau[0]
        tau2_track_err = tau_ct_raw[1] - (diag.tau_sea if diag else 0.0)

        obs = np.array([
            q[0], q[1],
            q_dot[0], q_dot[1],
            ee_err[0], ee_err[1],
            delta,
            delta_dot,
            F_cable,
            tau2_ct,
            tau2_sea,
            tau_motor,
            tau1_track_err,
            tau2_track_err,
        ], dtype=np.float32)

        return obs

    def render(self):
        pass  # Rendering handled by world.step(render=True)

    def close(self):
        pass  # World/SimulationApp lifecycle managed externally
