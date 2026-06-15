"""Pure-numpy cable routing geometry for MHP (tangents, helix, arcs)."""
from __future__ import annotations

import numpy as np

def compute_tangent(
    c1, r1: float,
    c2, r2: float,
    branch: int = +1 or -1,
    kind: str = "external" or "internal",
) -> tuple:
    """One external/internal tangent between circles (c1, r1) and (c2, r2).

    Works in the XY plane; Z of each centre is preserved in the output points.

    Parameters
    ----------
    branch : +1 or -1  — selects one of the two parallel tangent branches.
    kind   : 'external' (both circles on same side) or 'internal' (crosses between).

    Returns (T1, T2) — tangent contact points on circle 1 and circle 2.
    """
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    p1, p2 = c1[:2], c2[:2]
    d = p2 - p1
    D = np.linalg.norm(d)
    if D < 1e-9:
        return c1.copy(), c2.copy()
    d_hat = d / D
    perp  = np.array([-d_hat[1], d_hat[0]])
    branch = 1 if branch >= 0 else -1

    if kind == "internal":
        cos_a = (r1 + r2) / D
        sin_a = np.sqrt(max(0.0, 1.0 - cos_a ** 2))
        n  = cos_a * d_hat + branch * sin_a * perp
        T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
        T2 = np.array([p2[0] - r2 * n[0], p2[1] - r2 * n[1], c2[2]])
    else:  # external
        cos_a = (r1 - r2) / D
        sin_a = np.sqrt(max(0.0, 1.0 - cos_a ** 2))
        n  = cos_a * d_hat + branch * sin_a * perp
        T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
        T2 = np.array([p2[0] + r2 * n[0], p2[1] + r2 * n[1], c2[2]])
    return T1, T2


def _helical_wrap_xy(ax, center, radius: float, z_start: float, z_end: float,
                     T_start, T_end, branch: int = +1,
                     color: str = "#333", lw: float = 2.5, n_turns: int = 8,
                     pts_per_turn: int = 32, zorder: int = 7) -> None:
    """Draw a helical wrap around a spool/cylinder from T_start to T_end.

    The helix winds around the cylinder and terminates *exactly* on T_end so the
    cable stays continuous with the next segment. Wrap direction is set by branch:
    branch > 0 → CCW (increasing angle), else CW (decreasing angle).

    Only draws the XY projection (top-view axes).
    """
    cx, cy = float(center[0]), float(center[1])
    a_start  = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1])   - cy, float(T_end[0])   - cx)
    n_extra  = max(0, int(round(n_turns)) - 1)   # full extra turns before landing

    if branch > 0:                                # CCW — increasing angle
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:                                         # CW — decreasing angle
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi

    n_pts  = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    ax.plot(xs, ys, '-', color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round', antialiased=False)


def _helical_wrap_3d(ax3d, center, radius: float, z_start: float, z_end: float,
                     T_start, T_end, branch: int = +1,
                     color: str = "#333", lw: float = 2.5, n_turns: int = 8,
                     pts_per_turn: int = 32, zorder: int = 7) -> None:
    """Draw a 3-D helix around a spool, terminating exactly on T_end."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_start  = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1])   - cy, float(T_end[0])   - cx)
    n_extra  = max(0, int(round(n_turns)) - 1)

    if branch > 0:                                # CCW
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:                                         # CW
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi

    n_pts  = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    zs = np.linspace(z_start, z_end, n_pts)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    ax3d.plot(xs, ys, zs, '-', color=color, linewidth=lw, zorder=zorder,
              solid_capstyle='round', solid_joinstyle='round')


def _wrap_arc_xy(ax, center, radius: float, T_in, T_out,
                 color: str, lw: float = 3.0, n: int = 56, zorder: int = 7,
                 direction: str = 'auto') -> None:
    """Draw a wrap arc from T_in to T_out around *center* in the XY plane.

    direction : 'auto' (cross-product), 'ccw' (force anticlockwise), 'cw' (force clockwise).
    """
    cx, cy = float(center[0]), float(center[1])
    a_in  = np.arctan2(float(T_in[1])  - cy, float(T_in[0])  - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:
        cross_z = (
            (float(T_in[0])  - cx) * (float(T_out[1]) - cy)
            - (float(T_in[1]) - cy) * (float(T_out[0]) - cx)
        )
        if cross_z > 0:        # CCW
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:                   # CW
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    ax.plot(cx + radius * np.cos(ang), cy + radius * np.sin(ang),
            color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round', antialiased=False)


def _wrap_arc_3d(ax3d, center, radius: float, T_in, T_out,
                 color: str, lw: float = 3.0, n: int = 56, zorder: int = 7,
                 direction: str = 'auto') -> None:
    """Draw a wrap arc in 3D — arc lives in the XY plane at Z = center[2].

    direction : 'auto' (cross-product), 'ccw' (force anticlockwise), 'cw' (force clockwise).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_in  = np.arctan2(float(T_in[1])  - cy, float(T_in[0])  - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:
        cross_z = (
            (float(T_in[0])  - cx) * (float(T_out[1]) - cy)
            - (float(T_in[1]) - cy) * (float(T_out[0]) - cx)
        )
        if cross_z > 0:
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    xs, ys = cx + radius * np.cos(ang), cy + radius * np.sin(ang)
    ax3d.plot(xs, ys, np.full_like(xs, cz), color=color, linewidth=lw, zorder=zorder,
              solid_capstyle='round', solid_joinstyle='round')


def _seg_xy(ax, P1, P2, color: str, lw: float = 3.0, zorder: int = 7) -> None:
    """Draw a straight cable segment in the XY top-view axes."""
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], '-',
            color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round', antialiased=False)


def _seg_3d(ax3d, P1, P2, color: str, lw: float = 3.0, zorder: int = 7) -> None:
    """Draw a straight cable segment in 3D."""
    ax3d.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], '-',
              color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
              solid_joinstyle='round')



# ─── Meshcat visualization ────────────────────────────────────────────────────

def helix_pts_3d(center, radius: float, z_start: float, z_end: float,
                  T_start, T_end, branch: int = +1,
                  n_turns: int = 2, pts_per_turn: int = 48) -> np.ndarray:
    """Return (N,3) helix points — same math as _helical_wrap_3d but returns array."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_start  = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1])   - cy, float(T_end[0])   - cx)
    n_extra  = max(0, int(round(n_turns)) - 1)
    if branch > 0:
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi
    n_pts  = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    zs     = np.linspace(z_start, z_end, n_pts)
    xs     = cx + radius * np.cos(angles)
    ys     = cy + radius * np.sin(angles)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def Rz(theta: float) -> np.ndarray:
    """3×3 rotation matrix about the Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def arc_pts_3d(center, radius: float, T_in, T_out,
                n: int = 56, direction: str = 'auto') -> np.ndarray:
    """Return (N,3) arc points in the XY plane at Z=center[2] — same math as _wrap_arc_3d.

    direction : 'auto' (cross-product), 'ccw' (force anticlockwise), 'cw' (force clockwise).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_in  = np.arctan2(float(T_in[1])  - cy, float(T_in[0])  - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:  # auto
        cross_z = ((float(T_in[0]) - cx) * (float(T_out[1]) - cy)
                   - (float(T_in[1]) - cy) * (float(T_out[0]) - cx))
        if cross_z > 0:
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    xs  = cx + radius * np.cos(ang)
    ys  = cy + radius * np.sin(ang)
    return np.column_stack([xs, ys, np.full(n, cz)]).astype(np.float32)

# ─── OBJ loading utilities ─────────────────────────────────────────────────────
