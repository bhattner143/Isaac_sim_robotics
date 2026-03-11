#!/usr/bin/env python3
"""
Rewrite utils/viz_cables.py to contain ONLY the drawing/viz functions.
All classes stay in test_drive_pulley.py.
Functions accept their data (cable_routes, cable_waypoints, pulleys) as explicit args.
"""
from pathlib import Path

viz_src = Path("utils/viz_cables.py")
with open(viz_src) as f:
    lines = f.readlines()

# Find where the functions start (after all class/instance definitions)
# _compute_all_tangents starts at the first line that begins "def _compute_all_tangents"
fn_start = next(i for i, l in enumerate(lines) if l.startswith("def _compute_all_tangents"))

# The functions block is lines[fn_start:]
functions_block = lines[fn_start:]

# Remove the "if X is None: X = GLOBAL" fallback lines — they reference globals that
# won't exist in viz_cables.py anymore.  Since all callers will pass the args explicitly,
# the functions become simple: just remove those fallback guards.
# We keep the full bodies but drop the None-guard blocks from _compute_all_tangents and draw_cables.

functions_text = "".join(functions_block)

# Remove the None-guard block from _compute_all_tangents (17 lines)
NULL_GUARD_TANGENTS = """\
    if drive_pulley      is None: drive_pulley      = DRIVE_PULLEY
    if idler_r           is None: idler_r           = IDLER_R
    if idler_l           is None: idler_l           = IDLER_L
    if pulley_big        is None: pulley_big        = PULLEY_BIG
    if cable_end_point_l is None: cable_end_point_l = CABLE_END_POINT_L
    if cable_end_point_r is None: cable_end_point_r = CABLE_END_POINT_R
    if tangent_drive_b_r is None: tangent_drive_b_r = TANGENT_DRIVE_B_R
    if tangent_idler_a_r is None: tangent_idler_a_r = TANGENT_IDLER_A_R
    if tangent_idler_b_r is None: tangent_idler_b_r = TANGENT_IDLER_B_R
    if tangent_big_a_l   is None: tangent_big_a_l   = TANGENT_BIG_A_L
    if tangent_drive_b_l is None: tangent_drive_b_l = TANGENT_DRIVE_B_L
    if tangent_idler_a_l is None: tangent_idler_a_l = TANGENT_IDLER_A_L
    if tangent_idler_b_l is None: tangent_idler_b_l = TANGENT_IDLER_B_L
    if tangent_big_a_r   is None: tangent_big_a_r   = TANGENT_BIG_A_R
    if tangent_big_b_l   is None: tangent_big_b_l   = TANGENT_BIG_B_L
    if tangent_big_b_r   is None: tangent_big_b_r   = TANGENT_BIG_B_R
"""
assert NULL_GUARD_TANGENTS in functions_text, "Cannot find tangent null-guards"
functions_text = functions_text.replace(NULL_GUARD_TANGENTS, "", 1)

# Remove the "All pulley/tangent-point arguments default..." doc paragraph
DOC_EXTRA = "    All pulley/tangent-point arguments default to the module-level instances\n    so existing call sites need no changes.\n"
functions_text = functions_text.replace(DOC_EXTRA, "", 1)

# Remove the None-guard block from draw_cables (5 lines)
NULL_GUARD_DRAW = """\
    if cable_routes is None: cable_routes = CABLE_ROUTES
    if drive_pulley is None: drive_pulley = DRIVE_PULLEY
    if idler_r      is None: idler_r      = IDLER_R
    if idler_l      is None: idler_l      = IDLER_L
    if pulley_big   is None: pulley_big   = PULLEY_BIG
"""
assert NULL_GUARD_DRAW in functions_text, "Cannot find draw_cables null-guards"
functions_text = functions_text.replace(NULL_GUARD_DRAW, "", 1)

# Remove the "if routes is None: routes = CABLE_ROUTES" fallback in print_cable_routing_points
NULL_GUARD_PRINT = "    if routes is None:\n        routes = CABLE_ROUTES\n"
assert NULL_GUARD_PRINT in functions_text, "Cannot find print_cable null-guard"
functions_text = functions_text.replace(NULL_GUARD_PRINT, "", 1)

# Remove the routes doc default mention update
functions_text = functions_text.replace(
    "        List of CableRoute objects to report.  Defaults to CABLE_ROUTES.\n",
    "        List of CableRoute objects to report.\n", 1
)

# Also remove the "cable_routes ... Defaults to module-level CABLE_ROUTES" doc lines
functions_text = functions_text.replace(
    "    cable_routes : list | None\n        CableRoute list to draw.  Defaults to module-level CABLE_ROUTES.\n",
    "    cable_routes : list\n        CableRoute list to draw.\n", 1
)
functions_text = functions_text.replace(
    "    drive_pulley, idler_r, idler_l, pulley_big :\n        Pulley instances for wrap arcs.  Default to module-level instances.\n",
    "    drive_pulley, idler_r, idler_l, pulley_big :\n        Pulley instances for wrap arcs.\n", 1
)

# Keep the list | None = None signature (required args will just always be passed explicitly)
# No change needed — the None defaults are fine; we only removed the fallback guards.

# Remove separating "─" comment between draw_cables end and visualize_cable_routing_3d
SEP_COMMENT = "# ──────────────────────────────────────────────────────────────────────────────\n\ndef visualize_cable_routing_3d"
functions_text = functions_text.replace(SEP_COMMENT,
    "\n# ──────────────────────────────────────────────────────────────────────────────\n\ndef visualize_cable_routing_3d", 1)

# Also remove PulleyBase reference in _compute_all_tangents since PulleyBase comes from caller's scope
# Wait — PulleyBase.tangent_in_world_frame is referenced inside _compute_all_tangents.
# We need to pass it in OR import PulleyBase. The cleanest approach: accept pulley_base_cls arg.
# But that's complex. Simpler: the caller passes a `twf` function or we import it from test_drive_pulley.
# BEST: accept pulley_base_cls as parameter OR make tangent_in_world_frame a standalone function.
# For now: the simplest fix is to just accept `tangent_fn` as an argument.
functions_text = functions_text.replace(
    "    twf = PulleyBase.tangent_in_world_frame   # alias for brevity\n",
    "    twf = tangent_in_world_frame\n", 1
)

# Update _compute_all_tangents signature to include tangent_in_world_frame
functions_text = functions_text.replace(
    "def _compute_all_tangents(plant, plant_context, manipulator,\n"
    "                          drive_pulley=None, idler_r=None, idler_l=None,\n",
    "def _compute_all_tangents(plant, plant_context, manipulator,\n"
    "                          tangent_in_world_frame,\n"
    "                          drive_pulley=None, idler_r=None, idler_l=None,\n", 1
)

# Add "Call this once..." but remove reference to Idler↔BigPulley — it's fine to keep as is.

# Now write the new viz_cables.py
new_header = """\
#!/usr/bin/env python3
\"\"\"
viz_cables.py
─────────────
Drawing and visualization functions for the cable manipulator.

All data (cable_routes, cable_waypoints, pulley instances) is passed as explicit
arguments — no module-level state is defined here.
Classes and module-level instances live in test_drive_pulley.py.
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)
from pydrake.geometry import Rgba, Cylinder
from termcolor import colored

"""

with open(viz_src, "w") as f:
    f.write(new_header)
    f.write(functions_text)

import ast
src = open(viz_src).read()
try:
    ast.parse(src)
    print(f"AST OK  ({src.count(chr(10))} lines)")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
