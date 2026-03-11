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
# parent link's frame to the child joint frame, which equals the link length.
print("=" * 60)
print("JOINT ORIGINS  (= link lengths along kinematic chain)")
print("=" * 60)

joints = re.findall(
    r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', txt, re.DOTALL
)
total_reach = 0.0
for jname, jtype, jbody in joints:
    origin_m = re.search(r'<origin xyz="([^"]+)"', jbody)
    parent_m = re.search(r'<parent link="([^"]+)"', jbody)
    child_m  = re.search(r'<child link="([^"]+)"', jbody)
    axis_m   = re.search(r'<axis xyz="([^"]+)"', jbody)

    if not origin_m:
        continue
    xyz = np.fromstring(origin_m.group(1), sep=' ')
    length = np.linalg.norm(xyz)
    parent = parent_m.group(1) if parent_m else '?'
    child  = child_m.group(1)  if child_m  else '?'
    axis   = axis_m.group(1)   if axis_m   else 'N/A'

    marker = "  🔵 REVOLUTE" if jtype == 'revolute' else f"  ({jtype})"
    print(f"\nJoint: {jname}{marker}")
    print(f"  {parent}  →  {child}")
    print(f"  origin xyz = [{origin_m.group(1)}]")
    print(f"  |length|   = {length*1e3:.2f} mm  ({length:.6f} m)")
    print(f"  axis       = [{axis}]")
    if jtype == 'revolute':
        total_reach += length

print()
print("=" * 60)
print(f"Sum of revolute joint offsets: {total_reach*1e3:.2f} mm  ({total_reach:.6f} m)")
print("(max reach ≈ L1 + L2, computed via Drake FK in practice)")
print("=" * 60)
