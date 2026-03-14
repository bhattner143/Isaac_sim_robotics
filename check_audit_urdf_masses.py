import re
import numpy as np

with open('model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf') as f:
    txt = f.read()

# ── Link masses / inertias ────────────────────────────────────────────────────
links = re.findall(r'<link name="([^"]+)">(.*?)</link>', txt, re.DOTALL)
for name, body in links:
    mass_m    = re.search(r'<mass value="([^"]+)"', body)
    inertia_m = re.search(r'<inertia ixx="([^"]+)" ixy="([^"]+)" ixz="([^"]+)" iyy="([^"]+)" iyz="([^"]+)" izz="([^"]+)"', body)
    origin_m  = re.search(r'<inertial>\s*<origin xyz="([^"]+)"', body, re.DOTALL)
    vis_parts = re.findall(r'<!--\s*Part\s+(.*?)\s*-->', body)

    mass = mass_m.group(1) if mass_m else '???'
    com  = origin_m.group(1) if origin_m else '???'
    zero = float(mass) < 1e-6 if mass != '???' else True

    status = "❌ ZERO/DUMMY" if zero else "✅"
    print(f"LINK: {name}   {status}")
    print(f"  mass = {mass} kg")
    if inertia_m:
        ixx, ixy, ixz, iyy, iyz, izz = inertia_m.groups()
        print(f"  Ixx={ixx}  Iyy={iyy}  Izz={izz}")
        print(f"  Ixy={ixy}  Ixz={ixz}  Iyz={iyz}")
    print(f"  CoM = [{com}]")
    print(f"  Parts inside this link:")
    for p in vis_parts:
        print(f"    • {p.strip()}")
    print()

# ── Link lengths from joint origins ──────────────────────────────────────────
# Each revolute joint's <origin xyz="..."> gives the translation from the
# parent link's frame to the child joint frame.
#
# For a horizontal SCARA arm the joints rotate about Z, so the kinematically
# meaningful "arm length" is the XY-plane projection of the joint offset.
# The Z component is a vertical height difference between joint planes and
# does NOT contribute to horizontal reach or IK.
print("=" * 60)
print("JOINT ORIGINS  (= link lengths along kinematic chain)")
print("=" * 60)

joints = re.findall(
    r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', txt, re.DOTALL
)
total_reach_xy = 0.0
for jname, jtype, jbody in joints:
    origin_m = re.search(r'<origin xyz="([^"]+)"', jbody)
    parent_m = re.search(r'<parent link="([^"]+)"', jbody)
    child_m  = re.search(r'<child link="([^"]+)"', jbody)
    axis_m   = re.search(r'<axis xyz="([^"]+)"', jbody)

    if not origin_m:
        continue
    xyz    = np.fromstring(origin_m.group(1), sep=' ')
    len_3d = np.linalg.norm(xyz)
    len_xy = np.linalg.norm(xyz[:2])   # horizontal arm length (XY projection)
    len_z  = abs(xyz[2])               # vertical height offset
    parent = parent_m.group(1) if parent_m else '?'
    child  = child_m.group(1)  if child_m  else '?'
    axis   = axis_m.group(1)   if axis_m   else 'N/A'

    marker = "  🔵 REVOLUTE" if jtype == 'revolute' else f"  ({jtype})"
    print(f"\nJoint: {jname}{marker}")
    print(f"  {parent}  →  {child}")
    print(f"  origin xyz      = [{origin_m.group(1)}]")
    print(f"  arm length (XY) = {len_xy*1e3:.2f} mm  ← horizontal reach contribution")
    print(f"  Z height offset = {len_z*1e3:.2f} mm  ← vertical (not reach)")
    print(f"  3D magnitude    = {len_3d*1e3:.2f} mm  (hypotenuse — not kinematically meaningful)")
    print(f"  axis            = [{axis}]")
    if jtype == 'revolute':
        total_reach_xy += len_xy

print()
print("=" * 60)
print(f"Max horizontal reach (sum of XY arm lengths): {total_reach_xy*1e3:.2f} mm  ({total_reach_xy:.6f} m)")
print("(= L1 + L2 when all joints fully extended in-plane)")
print("=" * 60)
