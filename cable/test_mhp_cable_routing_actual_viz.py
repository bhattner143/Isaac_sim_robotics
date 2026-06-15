#!/usr/bin/env python3
"""
test_mhp_cable_routing_actual_viz.py
────────────────────────────────────
Interactive MHP (manipulator hybrid planar) cable routing visualization.

Modules:
  types_mhp.py        — dataclasses, FK (MHPKinematics)
  geometry_mhp.py     — tangent / helix / arc math
  cables_lower_mhp.py — lower (+Y shoulder) cable config
  cables_upper_mhp.py — upper (-Y elbow) cable config
  path_mhp.py         — compute_cable_path()
  cable_viz_mhp.py    — matplotlib plot_cable_routing()
  meshcat_mhp.py      — Drake / raw Meshcat scene
  simulation_mhp_ct.py — computed-torque Drake simulation

Computed-torque simulation (separate entry):
    python script_mhp_manipulator_ct_pydrake.py
    python cable/test_mhp_ct_pydrake.py

Usage:
    cd /Volumes/Data/Isaac_sim_robotics
    conda activate pydrake_cursor
    python cable/test_mhp_cable_routing_actual_viz.py
    python cable/test_mhp_cable_routing_actual_viz.py --q1 30 --q2 -20
    python cable/test_mhp_cable_routing_actual_viz.py --save out.png --show
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python cable/test_mhp_cable_routing_actual_viz.py` or via debugger.
# The script directory must not stay on sys.path: it shadows the `cable` package
# (Python would import cable/cable.py instead of cable/__init__.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CABLE_DIR = Path(__file__).resolve().parent
_repo = str(_REPO_ROOT)
_cable_dir = str(_CABLE_DIR)
while _cable_dir in sys.path:
    sys.path.remove(_cable_dir)
while _repo in sys.path:
    sys.path.remove(_repo)
sys.path.insert(0, _repo)

import numpy as np

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):  # type: ignore
        return text

from cable.cables_mhp import build_lower_cable_config, build_upper_cable_config
from cable.cable_viz_mhp import plot_cable_routing
from cable.meshcat_mhp import build_meshcat_scene, update_meshcat_pose
from cable.types_mhp import MHPKinematics

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MHP cable routing visualisation — 3D + XY top view"
    )
    ap.add_argument("--q1", type=float, default=0.0,
                    help="Shoulder joint angle q1 [deg]  (default 0)")
    ap.add_argument("--q2", type=float, default=0,
                    help="Elbow joint angle q2 [deg]     (default 0)")
    ap.add_argument("--elev", type=float, default=28,
                    help="3D view elevation angle [deg] (default 28)")
    ap.add_argument("--azim", type=float, default=-55,
                    help="3D view azimuth angle [deg] (default -55)")
    ap.add_argument("--save", type=str, default=None,
                    help="Save figure to this path (PNG/PDF) instead of showing")
    ap.add_argument("--show", action="store_true",
                    help="Show interactive matplotlib window")
    ap.add_argument("--no-matplotlib", action="store_true",
                    help="Skip matplotlib; go straight to Meshcat + interactive loop")
    ap.add_argument("--urdf-alpha", type=float, default=0.1,
                    help="URDF robot transparency [0=invisible, 1=opaque] (default 0.3)")
    args = ap.parse_args()

    q1_rad = np.deg2rad(args.q1)
    q2_rad = np.deg2rad(args.q2)

    lower_cable = build_lower_cable_config()
    upper_cable = build_upper_cable_config()

    print(f"\nLower cable ({lower_cable.name}): "
          f"{len(lower_cable.physical)} physical components, "
          f"{len(lower_cable.path)} path markers")
    print(f"Upper cable ({upper_cable.name}): "
          f"{len(upper_cable.physical)} physical components, "
          f"{len(upper_cable.path)} path markers")

    kin = MHPKinematics(q1_rad, q2_rad)
    print(f"\nFK at q1={args.q1:.1f}°, q2={args.q2:.1f}°:")
    print(f"  J1 (shoulder) = {kin.J1}")
    print(f"  J2 (elbow)    = {kin.J2}")

    if not args.no_matplotlib:
        fig = plot_cable_routing(
            lower_cable, upper_cable, q1=q1_rad, q2=q2_rad,
            view_elev=args.elev, view_azim=args.azim,
        )
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"\nSaved → {args.save}")
            if args.show:
                plt.show()
        elif args.show:
            print("\nClose the figure window to continue to Meshcat.\n")
            plt.show()

    scene = build_meshcat_scene(lower_cable, upper_cable, urdf_alpha=args.urdf_alpha)
    if scene is None:
        sys.exit(0)

    update_meshcat_pose(scene, lower_cable, upper_cable, q1_rad, q2_rad)
    backend_url = (scene["drake_mc"].web_url() if scene["mode"] == "drake"
                   else scene["vis"].url())
    print(f"\n  Open browser → {backend_url}")
    print(colored(
        "\nEnter joint angles in degrees  (e.g.  30  -15)  →  robot + cables update.",
        "yellow"))
    print(colored("Press Enter on an empty line or Ctrl+C to exit.\n", "yellow"))

    if not sys.stdin.isatty():
        print("\nNo interactive TTY — Meshcat server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        try:
            while True:
                raw = input(colored("q1  q2 [deg]: ", "cyan")).strip()
                if not raw:
                    break
                parts = raw.split()
                if len(parts) != 2:
                    print(colored("  ✗ Expected exactly two values: q1 q2", "red"))
                    continue
                try:
                    q1_deg, q2_deg = float(parts[0]), float(parts[1])
                except ValueError:
                    print(colored("  ✗ Invalid numbers. Enter two floats: q1 q2", "red"))
                    continue

                update_meshcat_pose(
                    scene, lower_cable, upper_cable,
                    np.deg2rad(q1_deg), np.deg2rad(q2_deg),
                )
                kin = MHPKinematics(np.deg2rad(q1_deg), np.deg2rad(q2_deg))
                ee = kin.to_world(np.array([0.2, 0., 0.]), "lower_arm")
                print(colored(
                    f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                    f"→  EE≈({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m",
                    "green"))
        except (KeyboardInterrupt, EOFError):
            pass
        print(colored("\n✓ Stopped.", "green"))


if __name__ == "__main__":
    main()
