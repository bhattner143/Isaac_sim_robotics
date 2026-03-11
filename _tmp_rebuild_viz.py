#!/usr/bin/env python3
"""
Rebuild utils/viz_cables.py from the canonical content in test_drive_pulley.py,
with function signatures updated so globals become explicit keyword arguments.
"""
import re
from pathlib import Path

src = Path("test_drive_pulley.py")
dst = Path("utils/viz_cables.py")

with open(src) as f:
    content = f.read()
    lines = content.splitlines(keepends=True)

# Lines 57-953 (1-indexed) = indices 56-952 (0-indexed)
# Contains: classes, instances, _compute_all_tangents, print_cable_routing_points,
#           _Xw, draw_cables (but NOT build_plant / main)
cable_block = lines[56:953]

header = """\
#!/usr/bin/env python3
\"\"\"
viz_cables.py
─────────────
Cable routing visualization utilities for the cable manipulator.

Provides pulley geometry classes, cable route definitions, tangent computation,
and Meshcat / matplotlib visualization helpers.  Imported by test_drive_pulley.py.
\"\"\"

import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)
from pydrake.geometry import Rgba, Cylinder
from termcolor import colored

"""

# Read visualize functions from the EXISTING on-disk viz_cables.py
# They're the correct versions (lines 63-356 of existing file)
with open(dst) as f:
    viz_lines = f.readlines()

# The two viz functions are in the existing file — grab them
viz_3d_start    = next(i for i, l in enumerate(viz_lines) if l.startswith("def visualize_cable_routing_3d"))
viz_top_start   = next(i for i, l in enumerate(viz_lines) if l.startswith("def visualize_cable_routing_top_view"))
viz_3d_block    = viz_lines[viz_3d_start:viz_top_start]
viz_top_block   = viz_lines[viz_top_start:]

with open(dst, "w") as f:
    f.write(header)
    f.writelines(cable_block)
    f.write("\n")
    f.writelines(viz_3d_block)
    f.write("\n")
    f.writelines(viz_top_block)

total = sum(1 for _ in open(dst))
print(f"Written {total} lines to {dst}")
