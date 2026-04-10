"""
controller/trajectory.py
────────────────────────
Pure-NumPy trajectory generation — no Drake dependency.

Provides C² (continuous second derivative) cubic spline trajectories for
the cup manipulator controllers.  Works with both Isaac Sim and PyDrake
simulation loops.

Usage::

    from controller.trajectory import RectTrajectory, build_move_to_start

    traj = RectTrajectory(
        x_range=(0.49, 0.51), y_range=(-0.08, 0.08),
        N=60, lap_duration=10.0,
    )
    pos = traj.eval_position(t)   # (2,)  [x, y]
    vel = traj.eval_velocity(t)   # (2,)  [ẋ, ẏ]
    acc = traj.eval_acceleration(t)  # (2,)  [ẍ, ÿ]
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import CubicSpline


class LoopingCubicTrajectory:
    """C² cubic spline trajectory that loops with period T.

    Wraps scipy.interpolate.CubicSpline with periodic boundary conditions
    to produce smooth position, velocity, and acceleration at any time.
    """

    def __init__(self, t_breaks: np.ndarray, waypoints: np.ndarray, period: float):
        """
        Parameters
        ----------
        t_breaks : (N+1,) time stamps for waypoints (0 .. period)
        waypoints : (N+1, 2) XY positions — last == first for closure
        period : loop duration [s]
        """
        self.period = period
        # CubicSpline with periodic BC gives C² continuity at the wrap
        self._cs = CubicSpline(t_breaks, waypoints, bc_type='periodic')
        self._cs_vel = self._cs.derivative(1)
        self._cs_acc = self._cs.derivative(2)

    def eval_position(self, t: float) -> np.ndarray:
        t_wrap = t % self.period
        return self._cs(t_wrap)

    def eval_velocity(self, t: float) -> np.ndarray:
        t_wrap = t % self.period
        return self._cs_vel(t_wrap)

    def eval_acceleration(self, t: float) -> np.ndarray:
        t_wrap = t % self.period
        return self._cs_acc(t_wrap)


class RectTrajectory:
    """Rectangle trajectory with velocity profiling (slow corners, fast straights).

    Matches the PyDrake version's non-uniform time stamps and CubicWithContinuousSecondDerivatives.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (0.49, 0.51),
        y_range: Tuple[float, float] = (-0.08, 0.08),
        N: int = 60,
        lap_duration: float = 10.0,
        v_max: float = 0.08,
        v_corner: float = 0.02,
        corner_blend: float = 0.35,
    ):
        self.N = N
        self.lap_duration = lap_duration

        x_min, x_max = x_range
        y_min, y_max = y_range
        W = x_max - x_min
        H = y_max - y_min
        P = 2.0 * (W + H)  # perimeter

        # Corner arc-length positions
        cs = np.array([0.0, W, W + H, 2.0 * W + H])

        def s_to_xy(s):
            s = s % P
            if s <= W:
                return x_min + s, y_min
            elif s <= W + H:
                return x_max, y_min + (s - W)
            elif s <= 2.0 * W + H:
                return x_max - (s - W - H), y_max
            else:
                return x_min, y_max - (s - 2.0 * W - H)

        def dist_corner(s):
            s = s % P
            d = np.abs(s - cs)
            return float(np.minimum(d, P - d).min())

        d_blend = corner_blend * min(W, H)

        def speed(s):
            t = np.clip(dist_corner(s) / d_blend, 0.0, 1.0)
            return v_corner + (v_max - v_corner) * t * t * (3.0 - 2.0 * t)

        # Arc-length uniform samples
        s_vals = np.linspace(0.0, P, N + 1, endpoint=True)
        speeds = np.array([speed(s) for s in s_vals])

        # Non-uniform time stamps
        ds = P / N
        t_raw = np.zeros(N + 1)
        for i in range(N):
            t_raw[i + 1] = t_raw[i] + ds / (0.5 * (speeds[i] + speeds[i + 1]))
        t_wp = t_raw * (lap_duration / t_raw[-1])

        # Waypoints
        xy = np.array([s_to_xy(s) for s in s_vals])  # (N+1, 2)
        self.ee_x_tgt = xy[:N, 0]
        self.ee_y_tgt = xy[:N, 1]

        # Build looping C² cubic spline
        self._traj = LoopingCubicTrajectory(t_wp, xy, lap_duration)

    def eval_position(self, t: float) -> np.ndarray:
        return self._traj.eval_position(t)

    def eval_velocity(self, t: float) -> np.ndarray:
        return self._traj.eval_velocity(t)

    def eval_acceleration(self, t: float) -> np.ndarray:
        return self._traj.eval_acceleration(t)


class CircleTrajectory:
    """Circular trajectory in XY plane."""

    def __init__(
        self,
        cx: float = 0.4,
        cy: float = 0.0,
        radius: float = 0.1,
        N: int = 60,
        lap_duration: float = 10.0,
    ):
        self.N = N
        self.lap_duration = lap_duration

        angles = np.linspace(0, 2 * np.pi, N + 1, endpoint=True)
        ee_x = cx + radius * np.cos(angles)
        ee_y = cy + radius * np.sin(angles)

        self.ee_x_tgt = ee_x[:N]
        self.ee_y_tgt = ee_y[:N]

        t_wp = np.linspace(0.0, lap_duration, N + 1)
        xy = np.column_stack([ee_x, ee_y])  # (N+1, 2)
        self._traj = LoopingCubicTrajectory(t_wp, xy, lap_duration)

    def eval_position(self, t: float) -> np.ndarray:
        return self._traj.eval_position(t)

    def eval_velocity(self, t: float) -> np.ndarray:
        return self._traj.eval_velocity(t)

    def eval_acceleration(self, t: float) -> np.ndarray:
        return self._traj.eval_acceleration(t)


class LineTrajectory:
    """Back-and-forth line trajectory."""

    def __init__(
        self,
        cx: float = 0.4,
        cy: float = 0.0,
        radius: float = 0.1,
        N: int = 60,
        lap_duration: float = 10.0,
    ):
        self.N = N
        self.lap_duration = lap_duration

        ee_x = np.linspace(cx - radius, cx + radius, N)
        ee_y = np.full(N, cy)

        self.ee_x_tgt = ee_x
        self.ee_y_tgt = ee_y

        # Build closed loop: forward + reverse
        ee_x_full = np.concatenate([ee_x, ee_x[::-1], [ee_x[0]]])
        ee_y_full = np.concatenate([ee_y, ee_y[::-1], [ee_y[0]]])
        t_wp = np.linspace(0.0, lap_duration, len(ee_x_full))
        xy = np.column_stack([ee_x_full, ee_y_full])
        self._traj = LoopingCubicTrajectory(t_wp, xy, lap_duration)

    def eval_position(self, t: float) -> np.ndarray:
        return self._traj.eval_position(t)

    def eval_velocity(self, t: float) -> np.ndarray:
        return self._traj.eval_velocity(t)

    def eval_acceleration(self, t: float) -> np.ndarray:
        return self._traj.eval_acceleration(t)


def build_move_to_start(
    p_start: np.ndarray,   # (2,) start EE position [m]
    p_end: np.ndarray,     # (2,) target EE position (first waypoint) [m]
    v_end: np.ndarray,     # (2,) velocity at target (from trajectory) [m/s]
    duration: float,       # move duration [s]
) -> Optional["MoveToStartSpline"]:
    """Build a cubic Hermite spline from pre-home to first waypoint.

    Start: v=0 (rest), End: matches trajectory initial velocity for C¹ continuity.
    Returns None if duration <= 0.
    """
    if duration <= 0.0:
        return None
    return MoveToStartSpline(p_start, p_end, v_end, duration)


class MoveToStartSpline:
    """Cubic Hermite approach trajectory (v=0 → v_end over duration)."""

    def __init__(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        v_end: np.ndarray,
        duration: float,
    ):
        self.duration = duration
        t = np.array([0.0, duration])
        # CubicHermite: match position & velocity at both endpoints
        from scipy.interpolate import CubicHermiteSpline
        positions = np.array([p_start, p_end])  # (2, 2)
        velocities = np.array([np.zeros(2), v_end])  # (2, 2)
        self._cs = CubicHermiteSpline(t, positions, velocities)
        self._cs_vel = self._cs.derivative(1)
        self._cs_acc = self._cs.derivative(2)

    def eval_position(self, t: float) -> np.ndarray:
        return self._cs(np.clip(t, 0, self.duration))

    def eval_velocity(self, t: float) -> np.ndarray:
        return self._cs_vel(np.clip(t, 0, self.duration))

    def eval_acceleration(self, t: float) -> np.ndarray:
        return self._cs_acc(np.clip(t, 0, self.duration))


class PreambleTrajectorySource:
    """Combines move-to-start with the main looping trajectory.

    During t < move_duration: outputs the approach spline.
    After:  wraps into the main looping trajectory.
    """

    def __init__(self, move_spline: Optional[MoveToStartSpline], main_traj):
        self._move = move_spline
        self._main = main_traj
        self.move_duration = move_spline.duration if move_spline else 0.0

    def eval_position(self, t: float) -> np.ndarray:
        if self._move is not None and t < self.move_duration:
            return self._move.eval_position(t)
        t_main = max(0.0, t - self.move_duration)
        return self._main.eval_position(t_main)

    def eval_velocity(self, t: float) -> np.ndarray:
        if self._move is not None and t < self.move_duration:
            return self._move.eval_velocity(t)
        t_main = max(0.0, t - self.move_duration)
        return self._main.eval_velocity(t_main)

    def eval_acceleration(self, t: float) -> np.ndarray:
        if self._move is not None and t < self.move_duration:
            return self._move.eval_acceleration(t)
        t_main = max(0.0, t - self.move_duration)
        return self._main.eval_acceleration(t_main)
