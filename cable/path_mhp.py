"""Compute world-frame cable path geometry for MHP routes."""
from __future__ import annotations

from typing import Literal

import numpy as np

from cable.geometry_mhp import arc_pts_3d, compute_tangent, helix_pts_3d
from cable.types_mhp import CablePathData, CableRouteConfig, MHPKinematics

def compute_cable_path(route: "CableRouteConfig",
                       kin: "MHPKinematics",
                       cable_loc: Literal['lower', 'upper']) -> CablePathData:
    """Compute all cable path geometry for *route* at the current FK pose.

    Returns a :class:`CablePathData` consumed identically by the matplotlib
    and Meshcat renderers — there is no other place where these tangents are
    computed.

    Tangent kinds match the physical geometry:
      * Spool → GP1, GP1 → GP2  : external tangent (both pulleys same side)
      * GP2 → GP3, GP3 → Roller : internal tangent (cable crosses between them)
    """
    path_w = [kin.to_world(c.pos_in_link, c.link) for c in route.path]
    phys_w = [kin.to_world(c.pos_in_link, c.link) for c in route.physical]
    phys_r = [c.diameter_mm * 0.5e-3              for c in route.physical]
    helix_branch = +1 if cable_loc == 'upper' else -1

    # physical indices:
    #   [0] cable_anchor  — clamp point on spool rim (no radius)
    #   [1] spool drum    — rotation axis, r=20 mm
    #   [2] GP1, [3] GP2, [4] GP3
    #   [5] elbow roller
    #   [6] cable endpoint

    branch_sign_seq      = route.branch_sign_seq # [+1/-1 for each tangent segment]
    kind_seq             = route.kind_seq# ['external'/'internal' for each tangent segment]
    elbow_roller_arc_dir = route.elbow_roller_arc_dir# 'cw' or 'ccw' arc direction for wrap around elbow roller

    # Compute tangents between spool and pulleys.
    T_spool_exit, T_gp1_entry = compute_tangent(
        phys_w[1], phys_r[1], phys_w[2], phys_r[2], branch=branch_sign_seq[0],  kind=kind_seq[0])
    T_gp1_exit,   T_gp2_entry = compute_tangent(
        phys_w[2], phys_r[2], phys_w[3], phys_r[3], branch=branch_sign_seq[1],  kind=kind_seq[1])
    T_gp2_exit,   T_gp3_entry = compute_tangent(
        phys_w[3], phys_r[3], phys_w[4], phys_r[4], branch=branch_sign_seq[2],  kind=kind_seq[2])
    T_gp3_exit,   T_roller_in = compute_tangent(
        phys_w[4], phys_r[4], phys_w[5], phys_r[5], branch=branch_sign_seq[3],  kind=kind_seq[3])
    T_roller_out, _            = compute_tangent(
        phys_w[5], phys_r[5], phys_w[6], 0.0,       branch=branch_sign_seq[4], kind=kind_seq[4])

    # Override T_spool_exit Z so the helix has a real axial span.
    # compute_tangent preserves Z of the spool centre (phys_w[1][2]), making z_start==z_end
    # which produces a flat circle instead of a helix.
    # We advance Z by (n_turns × pitch) away from the guide pulleys:
    #   lower cable (helix_branch=-1) exits upward  → z_end = spool_Z + span
    #   upper cable (helix_branch=+1) exits downward → z_end = spool_Z - span
    _spool_pitch = route.spool_pitch_mm * 1e-3
    _z_helix_end = float(phys_w[1][2]) - helix_branch * route.n_spool_turns * _spool_pitch
    T_spool_exit = np.array([T_spool_exit[0], T_spool_exit[1], _z_helix_end], dtype=float)

    # Build the full piecewise path as a list of (N,3) arrays for each segment.
    pieces = [
        # 1. Cable anchor (phys_w[0]) is fixed on the spool rim.
        #    Helix wraps around spool centre (phys_w[1]) starting from that anchor angle.
        np.vstack([
            phys_w[0].reshape(1, 3),
            helix_pts_3d(phys_w[1], phys_r[1],
                          float(phys_w[0][2]), float(T_spool_exit[2]),
                          phys_w[0], T_spool_exit,
                          branch=helix_branch, n_turns=route.n_spool_turns, pts_per_turn=48),
        ]).astype(np.float32),
        # 2. Spool exit → GP1 entry
        np.array([T_spool_exit, T_gp1_entry], dtype=np.float32),
        # 3. GP1 arc
        arc_pts_3d(phys_w[2], phys_r[2], T_gp1_entry, T_gp1_exit),
        # 4. GP1 exit → GP2 entry
        np.array([T_gp1_exit, T_gp2_entry], dtype=np.float32),
        # 5. GP2 arc
        arc_pts_3d(phys_w[3], phys_r[3], T_gp2_entry, T_gp2_exit),
        # 6. GP2 exit → GP3 entry
        np.array([T_gp2_exit, T_gp3_entry], dtype=np.float32),
        # 7. GP3 arc
        arc_pts_3d(phys_w[4], phys_r[4], T_gp3_entry, T_gp3_exit),
        # 8. GP3 exit → roller entry
        np.array([T_gp3_exit, T_roller_in], dtype=np.float32),
        # 9. Roller arc (direction is CCW for lower, CW for upper)
        arc_pts_3d(phys_w[5], phys_r[5], T_roller_in, T_roller_out, direction=elbow_roller_arc_dir),
        # 10. Roller exit → cable endpoint (direct tangent)
        np.array([T_roller_out, phys_w[6]], dtype=np.float32),
    ]

    return CablePathData(
        route=route, path_w=path_w, phys_w=phys_w, phys_r=phys_r,
        T_spool_exit=T_spool_exit, T_gp1_entry=T_gp1_entry,
        T_gp1_exit=T_gp1_exit,    T_gp2_entry=T_gp2_entry,
        T_gp2_exit=T_gp2_exit,    T_gp3_entry=T_gp3_entry,
        T_gp3_exit=T_gp3_exit,    T_roller_in=T_roller_in,
        T_roller_out=T_roller_out, helix_branch=helix_branch,
        pieces=pieces,
    )

