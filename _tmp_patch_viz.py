#!/usr/bin/env python3
"""
Apply targeted text substitutions to utils/viz_cables.py:
1. Update _compute_all_tangents signature + body to use local vars
2. Update draw_cables signature + body to use cable_routes kwarg and local pulley vars
"""
from pathlib import Path

path = Path("utils/viz_cables.py")
text = path.read_text()

# ─── 1. _compute_all_tangents ─────────────────────────────────────────────────

OLD_TANGENTS_SIG = \
    "def _compute_all_tangents(plant, plant_context, manipulator) -> None:\n" \
    "    \"\"\"Compute all inter-pulley external tangent contacts using Drake FK.\n" \
    "\n" \
    "    Works uniformly for any pair of pulleys — same body frame or different\n" \
    "    body frames.  Delegates per-pair computation to\n" \
    "    :meth:`PulleyBase.tangent_in_world_frame`, which uses\n" \
    "    :meth:`PulleyBase.tangent` for the pure 2-D geometry.\n" \
    "\n" \
    "    Call this once after the Drake plant is built with joint positions set,\n" \
    "    and again whenever joint angles change (because cross-frame pairs like\n" \
    "    Idler↔BigPulley move relative to each other with q2).\n" \
    "    \"\"\"\n" \
    "    twf = PulleyBase.tangent_in_world_frame   # alias for brevity"

NEW_TANGENTS_SIG = \
    "def _compute_all_tangents(plant, plant_context, manipulator,\n" \
    "                          drive_pulley=None, idler_r=None, idler_l=None,\n" \
    "                          pulley_big=None, cable_end_point_l=None,\n" \
    "                          cable_end_point_r=None,\n" \
    "                          tangent_drive_b_r=None, tangent_idler_a_r=None,\n" \
    "                          tangent_idler_b_r=None, tangent_big_a_l=None,\n" \
    "                          tangent_drive_b_l=None, tangent_idler_a_l=None,\n" \
    "                          tangent_idler_b_l=None, tangent_big_a_r=None,\n" \
    "                          tangent_big_b_l=None, tangent_big_b_r=None) -> None:\n" \
    "    \"\"\"Compute all inter-pulley external tangent contacts using Drake FK.\n" \
    "\n" \
    "    Works uniformly for any pair of pulleys — same body frame or different\n" \
    "    body frames.  Delegates per-pair computation to\n" \
    "    :meth:`PulleyBase.tangent_in_world_frame`, which uses\n" \
    "    :meth:`PulleyBase.tangent` for the pure 2-D geometry.\n" \
    "\n" \
    "    Call this once after the Drake plant is built with joint positions set,\n" \
    "    and again whenever joint angles change (because cross-frame pairs like\n" \
    "    Idler↔BigPulley move relative to each other with q2).\n" \
    "\n" \
    "    All pulley/tangent-point arguments default to the module-level instances\n" \
    "    so existing call sites need no changes.\n" \
    "    \"\"\"\n" \
    "    if drive_pulley      is None: drive_pulley      = DRIVE_PULLEY\n" \
    "    if idler_r           is None: idler_r           = IDLER_R\n" \
    "    if idler_l           is None: idler_l           = IDLER_L\n" \
    "    if pulley_big        is None: pulley_big        = PULLEY_BIG\n" \
    "    if cable_end_point_l is None: cable_end_point_l = CABLE_END_POINT_L\n" \
    "    if cable_end_point_r is None: cable_end_point_r = CABLE_END_POINT_R\n" \
    "    if tangent_drive_b_r is None: tangent_drive_b_r = TANGENT_DRIVE_B_R\n" \
    "    if tangent_idler_a_r is None: tangent_idler_a_r = TANGENT_IDLER_A_R\n" \
    "    if tangent_idler_b_r is None: tangent_idler_b_r = TANGENT_IDLER_B_R\n" \
    "    if tangent_big_a_l   is None: tangent_big_a_l   = TANGENT_BIG_A_L\n" \
    "    if tangent_drive_b_l is None: tangent_drive_b_l = TANGENT_DRIVE_B_L\n" \
    "    if tangent_idler_a_l is None: tangent_idler_a_l = TANGENT_IDLER_A_L\n" \
    "    if tangent_idler_b_l is None: tangent_idler_b_l = TANGENT_IDLER_B_L\n" \
    "    if tangent_big_a_r   is None: tangent_big_a_r   = TANGENT_BIG_A_R\n" \
    "    if tangent_big_b_l   is None: tangent_big_b_l   = TANGENT_BIG_B_L\n" \
    "    if tangent_big_b_r   is None: tangent_big_b_r   = TANGENT_BIG_B_R\n" \
    "    twf = PulleyBase.tangent_in_world_frame   # alias for brevity"

assert OLD_TANGENTS_SIG in text, "Could not find _compute_all_tangents signature"
text = text.replace(OLD_TANGENTS_SIG, NEW_TANGENTS_SIG, 1)

# ─── 2. Tangent body — replace UPPER_CASE pulley refs with locals ────────────

UPPER_CASES = [
    # Green cable calls
    ("DRIVE_PULLEY.B_R, IDLER_R.A_R    = twf(plant, plant_context, manipulator, DRIVE_PULLEY, IDLER_R, kind=\"external\", branch=-1)",
     "drive_pulley.B_R, idler_r.A_R    = twf(plant, plant_context, manipulator, drive_pulley, idler_r, kind=\"external\", branch=-1)"),
    ("IDLER_R.B_R,      PULLEY_BIG.A_L = twf(plant, plant_context, manipulator, IDLER_R,      PULLEY_BIG, kind=\"internal\", branch=-1)",
     "idler_r.B_R,      pulley_big.A_L = twf(plant, plant_context, manipulator, idler_r,      pulley_big, kind=\"internal\", branch=-1)"),
    ("PULLEY_BIG.B_L, CABLE_END_POINT_L.A_L = twf(plant, plant_context, manipulator, PULLEY_BIG, CABLE_END_POINT_L, kind=\"external\", branch=+1)",
     "pulley_big.B_L, cable_end_point_l.A_L = twf(plant, plant_context, manipulator, pulley_big, cable_end_point_l, kind=\"external\", branch=+1)"),
    # Red cable calls
    ("DRIVE_PULLEY.B_L, IDLER_L.A_L    = twf(plant, plant_context, manipulator, DRIVE_PULLEY, IDLER_L,    kind=\"external\", branch=+1)",
     "drive_pulley.B_L, idler_l.A_L    = twf(plant, plant_context, manipulator, drive_pulley, idler_l,    kind=\"external\", branch=+1)"),
    ("IDLER_L.B_L,      PULLEY_BIG.A_R = twf(plant, plant_context, manipulator, IDLER_L,      PULLEY_BIG, kind=\"internal\", branch=+1)",
     "idler_l.B_L,      pulley_big.A_R = twf(plant, plant_context, manipulator, idler_l,      pulley_big, kind=\"internal\", branch=+1)"),
    ("PULLEY_BIG.B_R, CABLE_END_POINT_R.A_R = twf(plant, plant_context, manipulator, PULLEY_BIG, CABLE_END_POINT_R, kind=\"external\", branch=-1)",
     "pulley_big.B_R, cable_end_point_r.A_R = twf(plant, plant_context, manipulator, pulley_big, cable_end_point_r, kind=\"external\", branch=-1)"),
    # FixedBodyPoint assignments
    ("TANGENT_DRIVE_B_R._tangent_point  = DRIVE_PULLEY.B_R",
     "tangent_drive_b_r._tangent_point  = drive_pulley.B_R"),
    ("TANGENT_IDLER_A_R._tangent_point  = IDLER_R.A_R",
     "tangent_idler_a_r._tangent_point  = idler_r.A_R"),
    ("TANGENT_IDLER_B_R._tangent_point  = IDLER_R.B_R",
     "tangent_idler_b_r._tangent_point  = idler_r.B_R"),
    ("TANGENT_BIG_A_L._tangent_point    = PULLEY_BIG.A_L",
     "tangent_big_a_l._tangent_point    = pulley_big.A_L"),
    ("TANGENT_DRIVE_B_L._tangent_point  = DRIVE_PULLEY.B_L",
     "tangent_drive_b_l._tangent_point  = drive_pulley.B_L"),
    ("TANGENT_IDLER_A_L._tangent_point  = IDLER_L.A_L",
     "tangent_idler_a_l._tangent_point  = idler_l.A_L"),
    ("TANGENT_IDLER_B_L._tangent_point  = IDLER_L.B_L",
     "tangent_idler_b_l._tangent_point  = idler_l.B_L"),
    ("TANGENT_BIG_A_R._tangent_point    = PULLEY_BIG.A_R",
     "tangent_big_a_r._tangent_point    = pulley_big.A_R"),
    ("TANGENT_BIG_B_L._tangent_point    = PULLEY_BIG.B_L",
     "tangent_big_b_l._tangent_point    = pulley_big.B_L"),
    ("TANGENT_BIG_B_R._tangent_point    = PULLEY_BIG.B_R",
     "tangent_big_b_r._tangent_point    = pulley_big.B_R"),
]

for old, new in UPPER_CASES:
    if old not in text:
        print(f"  WARNING: could not find: {old[:60]!r}")
    else:
        text = text.replace(old, new, 1)

# ─── 3. draw_cables — update signature + replace globals in body ──────────────

OLD_DRAW_SIG = \
    "def draw_cables(meshcat, plant, plant_context, manipulator,\n" \
    "                cable_radius: float = 0.0005, n_arc_pts: int = 32) -> None:\n" \
    "    \"\"\"Draw both tendon cables in Meshcat: straight segments and pulley wrap arcs.\n" \
    "\n" \
    "    Straight segments between consecutive contact points are rendered as\n" \
    "    cylinders.  Chord segments that span a pulley wrap (listed in each\n" \
    "    route's ``skip_chord_segments``) are replaced by a smooth arc drawn\n" \
    "    in the plane perpendicular to each pulley's shaft axis.\n" \
    "\n" \
    "    Parameters\n" \
    "    ----------\n" \
    "    cable_radius : float\n" \
    "        Tube radius in metres.\n" \
    "    n_arc_pts : int\n" \
    "        Number of sample points along each wrap arc (more = smoother).\n" \
    "    \"\"\""

NEW_DRAW_SIG = \
    "def draw_cables(meshcat, plant, plant_context, manipulator,\n" \
    "                cable_radius: float = 0.0005, n_arc_pts: int = 32,\n" \
    "                cable_routes: list | None = None,\n" \
    "                drive_pulley=None, idler_r=None, idler_l=None,\n" \
    "                pulley_big=None) -> None:\n" \
    "    \"\"\"Draw both tendon cables in Meshcat: straight segments and pulley wrap arcs.\n" \
    "\n" \
    "    Straight segments between consecutive contact points are rendered as\n" \
    "    cylinders.  Chord segments that span a pulley wrap (listed in each\n" \
    "    route's ``skip_chord_segments``) are replaced by a smooth arc drawn\n" \
    "    in the plane perpendicular to each pulley's shaft axis.\n" \
    "\n" \
    "    Parameters\n" \
    "    ----------\n" \
    "    cable_radius : float\n" \
    "        Tube radius in metres.\n" \
    "    n_arc_pts : int\n" \
    "        Number of sample points along each wrap arc (more = smoother).\n" \
    "    cable_routes : list | None\n" \
    "        CableRoute list to draw.  Defaults to module-level CABLE_ROUTES.\n" \
    "    drive_pulley, idler_r, idler_l, pulley_big :\n" \
    "        Pulley instances for wrap arcs.  Default to module-level instances.\n" \
    "    \"\"\"\n" \
    "    if cable_routes is None: cable_routes = CABLE_ROUTES\n" \
    "    if drive_pulley is None: drive_pulley = DRIVE_PULLEY\n" \
    "    if idler_r      is None: idler_r      = IDLER_R\n" \
    "    if idler_l      is None: idler_l      = IDLER_L\n" \
    "    if pulley_big   is None: pulley_big   = PULLEY_BIG"

assert OLD_DRAW_SIG in text, "Could not find draw_cables signature"
text = text.replace(OLD_DRAW_SIG, NEW_DRAW_SIG, 1)

# Replace body globals in draw_cables section
DRAW_BODY_REPLACEMENTS = [
    ("    for route in CABLE_ROUTES:\n",
     "    for route in cable_routes:\n"),
]
for old, new in DRAW_BODY_REPLACEMENTS:
    if old not in text:
        print(f"  WARNING: could not find draw_cables body: {old!r}")
    else:
        text = text.replace(old, new, 1)

# Replace the wraps list (it uses DRIVE_PULLEY etc.)
OLD_WRAPS = """\
    wraps = [
        # Green cable — Drive → IdlerR → BigPulley
        (DRIVE_PULLEY, DRIVE_PULLEY.A_R, DRIVE_PULLEY.B_R, "/wrap/drive/green",  _G),
        (IDLER_R,      IDLER_R.A_R,      IDLER_R.B_R,      "/wrap/idlerR/green", _G),
        (PULLEY_BIG,   PULLEY_BIG.A_L,   PULLEY_BIG.B_L,   "/wrap/big/green",    _G),
        # Red cable — Drive → IdlerL → BigPulley
        (DRIVE_PULLEY, DRIVE_PULLEY.A_L, DRIVE_PULLEY.B_L, "/wrap/drive/red",    _R),
        (IDLER_L,      IDLER_L.A_L,      IDLER_L.B_L,      "/wrap/idlerL/red",   _R),
        (PULLEY_BIG,   PULLEY_BIG.A_R,   PULLEY_BIG.B_R,   "/wrap/big/red",      _R),
    ]"""
NEW_WRAPS = """\
    wraps = [
        # Green cable — Drive → IdlerR → BigPulley
        (drive_pulley, drive_pulley.A_R, drive_pulley.B_R, "/wrap/drive/green",  _G),
        (idler_r,      idler_r.A_R,      idler_r.B_R,      "/wrap/idlerR/green", _G),
        (pulley_big,   pulley_big.A_L,   pulley_big.B_L,   "/wrap/big/green",    _G),
        # Red cable — Drive → IdlerL → BigPulley
        (drive_pulley, drive_pulley.A_L, drive_pulley.B_L, "/wrap/drive/red",    _R),
        (idler_l,      idler_l.A_L,      idler_l.B_L,      "/wrap/idlerL/red",   _R),
        (pulley_big,   pulley_big.A_R,   pulley_big.B_R,   "/wrap/big/red",      _R),
    ]"""
if OLD_WRAPS not in text:
    print(f"  WARNING: could not find wraps list")
else:
    text = text.replace(OLD_WRAPS, NEW_WRAPS, 1)

# ─── Write and verify ─────────────────────────────────────────────────────────
path.write_text(text)

import ast
try:
    ast.parse(text)
    print("AST OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")

print(f"Lines: {text.count(chr(10))}")
