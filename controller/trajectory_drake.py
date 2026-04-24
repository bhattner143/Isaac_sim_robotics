"""
controller/trajectory_drake.py
───────────────────────────────
Drake PiecewisePolynomial-based EE trajectory builders for the cup manipulator.

Requires pydrake.  Each builder takes ``(manip, plant, args)`` and returns
``(traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt)`` where the first three are
Drake PiecewisePolynomial objects.

Also provides :class:`PreambleSrc`, a Drake ``LeafSystem`` that wraps any of
the above trajectories with an optional move-to-start preamble.

Usage::

    from controller.trajectory_drake import (
        build_trajectory,
        build_move_to_start,
        PreambleSrc,
    )

    traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt = build_trajectory(manip, plant, args)
    move_traj, move_vel, move_acc, q_init = build_move_to_start(
        manip, plant, traj, traj_vel, args.move_duration
    )

    ee_src = builder.AddSystem(
        PreambleSrc(move_traj, args.move_duration, traj, args.duration)
    )
"""

import numpy as np
from pydrake.all import LeafSystem, PiecewisePolynomial


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clamp_to_reach(
    ee_x: np.ndarray,
    ee_y: np.ndarray,
    r_max: float,
    r_min: float,
) -> tuple:
    """Clamp EE waypoints to the robot's reachable annulus [r_min, r_max]."""
    _r = np.hypot(ee_x, ee_y)
    _far = _r > r_max
    if _far.any():
        ee_x[_far] *= r_max * 0.97 / _r[_far]
        ee_y[_far] *= r_max * 0.97 / _r[_far]
    _close = _r < r_min + 0.01
    if _close.any():
        _r_c = np.maximum(_r[_close], 1e-6)
        ee_x[_close] *= (r_min + 0.01) / _r_c
        ee_y[_close] *= (r_min + 0.01) / _r_c
    return ee_x, ee_y


# ─── Trajectory builders ──────────────────────────────────────────────────────

def build_rect_trajectory(manip, plant, args):
    """C² rectangular EE trajectory with speed-profiled corners.

    Returns (traj_ref, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max  = L1 + L2
    r_min  = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    N   = args.traj_n
    W   = x_max - x_min
    H   = y_max - y_min
    P   = 2.0 * (W + H)
    _cs = np.array([0.0, W, W + H, 2.0 * W + H])

    def _s_to_xy(s):
        s = s % P
        if   s <= W:            return x_min + s,            y_min
        elif s <= W + H:        return x_max,                 y_min + (s - W)
        elif s <= 2.0 * W + H:  return x_max - (s - W - H),  y_max
        else:                   return x_min,                 y_max - (s - 2.0 * W - H)

    def _corner_dist(s):
        s = s % P
        d = np.abs(s - _cs)
        return float(np.minimum(d, P - d).min())

    _v_max    = args.traj_v_max
    _v_corner = args.traj_v_corner
    _d_blend  = args.traj_corner_blend * min(W, H)

    def _speed(s):
        t = np.clip(_corner_dist(s) / max(_d_blend, 1e-9), 0.0, 1.0)
        return _v_corner + (_v_max - _v_corner) * t * t * (3.0 - 2.0 * t)

    _s_vals = np.linspace(0.0, P, N + 1, endpoint=True)
    _speeds = np.array([_speed(s) for s in _s_vals])
    _ds     = P / N
    _t_raw  = np.zeros(N + 1)
    for i in range(N):
        _t_raw[i + 1] = _t_raw[i] + _ds / (0.5 * (_speeds[i] + _speeds[i + 1]))
    t_wp = _t_raw * (args.duration / _t_raw[-1])
    _xy  = np.array([_s_to_xy(s) for s in _s_vals])
    ee_x = _xy[:N, 0]
    ee_y = _xy[:N, 1]
    ee_x, ee_y = _clamp_to_reach(ee_x, ee_y, r_max, r_min)

    wp = np.column_stack([np.append(ee_x, ee_x[0]),
                          np.append(ee_y, ee_y[0])]).T
    traj     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def build_circle_trajectory(manip, plant, args):
    """Circular EE trajectory.

    Returns (traj_ref, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max  = L1 + L2
    r_min  = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    if args.traj_radius is not None:
        R = float(args.traj_radius)
    else:
        R = 0.5 * min(x_max - x_min, y_max - y_min)
    R = max(R, 1e-4)

    N = max(int(args.traj_n), 16)
    s = np.linspace(0.0, 2.0 * np.pi, N + 1, endpoint=True)
    ee_x = cx + R * np.cos(s[:-1])
    ee_y = cy + R * np.sin(s[:-1])
    ee_x, ee_y = _clamp_to_reach(ee_x.copy(), ee_y.copy(), r_max, r_min)

    t_wp = np.linspace(0.0, args.duration, N + 1)
    wp   = np.column_stack([np.append(ee_x, ee_x[0]),
                            np.append(ee_y, ee_y[0])]).T
    traj     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def build_figure8_trajectory(manip, plant, args):
    """Lemniscate-of-Gerono (figure-8) EE trajectory.

    Returns (traj_ref, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max  = L1 + L2
    r_min  = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    if args.traj_radius is not None:
        Ax = float(args.traj_radius)
        Ay = float(args.traj_radius)
    else:
        Ax = 0.5 * (x_max - x_min)
        Ay = 0.5 * (y_max - y_min)
    Ax = max(Ax, 1e-4); Ay = max(Ay, 1e-4)

    N = max(int(args.traj_n), 32)
    s = np.linspace(0.0, 2.0 * np.pi, N + 1, endpoint=True)
    # Lemniscate-of-Gerono: x = A*sin(s), y = B*sin(s)*cos(s)
    ee_x = cx + Ax * np.sin(s[:-1])
    ee_y = cy + Ay * np.sin(s[:-1]) * np.cos(s[:-1])
    ee_x, ee_y = _clamp_to_reach(ee_x.copy(), ee_y.copy(), r_max, r_min)

    t_wp = np.linspace(0.0, args.duration, N + 1)
    wp   = np.column_stack([np.append(ee_x, ee_x[0]),
                            np.append(ee_y, ee_y[0])]).T
    traj     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def build_line_trajectory(manip, plant, args):
    """Back-and-forth sinusoidal line trajectory along y at fixed x = mean(x_range).

    Returns (traj_ref, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max  = L1 + L2
    r_min  = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    cx = 0.5 * (x_min + x_max)

    N = max(int(args.traj_n), 8)
    # one full period: y_min -> y_max -> y_min
    s = np.linspace(0.0, 2.0 * np.pi, N + 1, endpoint=True)
    ee_x = np.full(N, cx)
    ee_y = 0.5 * (y_min + y_max) + 0.5 * (y_max - y_min) * np.sin(s[:-1])
    ee_x, ee_y = _clamp_to_reach(ee_x.copy(), ee_y.copy(), r_max, r_min)

    t_wp = np.linspace(0.0, args.duration, N + 1)
    wp   = np.column_stack([np.append(ee_x, ee_x[0]),
                            np.append(ee_y, ee_y[0])]).T
    traj     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def build_trajectory(manip, plant, args):
    """Dispatch to the appropriate trajectory builder based on ``args.traj_type``.

    Supported types: ``rect`` (default), ``circle``, ``figure8``, ``line``.
    """
    kind = getattr(args, "traj_type", "rect")
    if kind == "circle":
        return build_circle_trajectory(manip, plant, args)
    if kind == "figure8":
        return build_figure8_trajectory(manip, plant, args)
    if kind == "line":
        return build_line_trajectory(manip, plant, args)
    return build_rect_trajectory(manip, plant, args)


def build_move_to_start(manip, plant, traj, traj_vel, move_duration):
    """Cubic-Hermite approach from near the first waypoint (zero initial velocity).

    Returns (move_traj, move_traj_vel, move_traj_acc, q_end).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    p_end  = traj.value(0.0).ravel()
    seed   = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
    q_end, ok = manip.ik._solve_2r_core(L1, L2, p_end, seed)
    if not ok:
        q_end = seed.copy()

    # Pre-home: small q-space offset for a meaningful approach
    q_pre   = q_end + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    tmp_ctx = plant.CreateDefaultContext()
    manip.set_positions_user_order(plant, tmp_ctx, q_pre)
    p_start = manip.get_end_effector_position(plant, tmp_ctx)[:2]
    v_end   = traj_vel.value(0.0).ravel()

    t_br  = np.array([0.0, move_duration])
    smp   = np.column_stack([p_start, p_end])
    smp_d = np.column_stack([np.zeros(2), v_end])
    move_traj     = PiecewisePolynomial.CubicHermite(t_br, smp, smp_d)
    move_traj_vel = move_traj.derivative(1)
    move_traj_acc = move_traj.derivative(2)
    return move_traj, move_traj_vel, move_traj_acc, q_end


# ─── Drake LeafSystem wrapper ─────────────────────────────────────────────────

class PreambleSrc(LeafSystem):
    """Drake LeafSystem: trajectory source with optional move-to-start preamble.

    During ``t < move_duration``: outputs the approach spline (CubicHermite).
    After:  wraps into the main periodic trajectory (phase = (t - move_duration) % period).

    Parameters
    ----------
    mv            : move-to-start PiecewisePolynomial (or None to skip preamble)
    move_duration : duration of the preamble [s]
    main_traj     : main looping PiecewisePolynomial
    period        : main trajectory loop period [s]
    """

    def __init__(self, mv, move_duration: float, main_traj, period: float):
        super().__init__()
        self._mv     = mv
        self._md     = float(move_duration)
        self._main   = main_traj
        self._period = float(period)
        self.DeclareVectorOutputPort("out", main_traj.rows(), self._calc)

    def _calc(self, ctx, out):
        t = ctx.get_time()
        if self._mv is not None and t < self._md:
            out.SetFromVector(self._mv.value(t).ravel())
        else:
            tw = max(0.0, t - self._md) % self._period
            out.SetFromVector(self._main.value(tw).ravel())
