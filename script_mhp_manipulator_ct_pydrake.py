#!/usr/bin/env python3
"""
script_mhp_manipulator_ct_pydrake.py
====================================
MHP (manipulator hybrid planar) computed-torque trajectory tracking with
analytical cable routing overlay in Meshcat.

Follows the import/layout conventions of
``script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py`` but targets the
real MHP URDF and the ``cable/*_mhp.py`` routing modules.

USAGE
─────
  python script_mhp_manipulator_ct_pydrake.py
  python script_mhp_manipulator_ct_pydrake.py --traj-type circle --duration 8
  python script_mhp_manipulator_ct_pydrake.py --ct-kp 400 400 --ct-kd 40 40 --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
import platform as _platform

if _platform.system() == "Darwin":
    try:
        matplotlib.use("MacOSX")
    except Exception:
        matplotlib.use("TkAgg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from termcolor import colored

from pydrake.all import StartMeshcat

sys.path.insert(0, str(Path(__file__).parent))

from cable.simulation_mhp_ct import run_simulation, plot_results

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="MHP computed-torque simulation with cable routing overlay",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

_ct = parser.add_argument_group("computed-torque gains")
_ct.add_argument("--ct-kp", type=float, nargs="+", default=[400.0, 400.0],
                 metavar="KP", help="CT position gain [1/s²] per joint.")
_ct.add_argument("--ct-kd", type=float, nargs="+", default=[40.0, 40.0],
                 metavar="KD", help="CT velocity gain [1/s] per joint.")
_ct.add_argument("--ct-tau-max", type=float, default=10.0,
                 help="Torque saturation [Nm].")

_sim = parser.add_argument_group("simulation")
_sim.add_argument("--duration", type=float, default=10.0, help="Lap duration [s]")
_sim.add_argument("--num-laps", type=int, default=1,
                  help="Laps before stopping (0 = run until Ctrl-C).")
_sim.add_argument("--move-duration", type=float, default=3.0,
                  help="Move-to-start preamble [s]. 0 to disable.")
_sim.add_argument("--no-meshcat", action="store_true",
                  help="Disable Meshcat 3-D visualisation.")
_sim.add_argument("--record", action="store_true", default=False,
                  help="Record Meshcat body poses for replay.")
_sim.add_argument("--no-show", action="store_true",
                  help="Skip blocking plt.show().")
_sim.add_argument("--urdf-alpha", type=float, default=0.25,
                  help="URDF transparency [0–1].")

_rob = parser.add_argument_group("robot mount")
_rob.add_argument("--tilt-roll", type=float, default=0.0)
_rob.add_argument("--tilt-pitch", type=float, default=0.0)
_rob.add_argument("--joint-damping", type=float, nargs=2, default=[0.05, 0.05],
                  metavar=("D1", "D2"))
_rob.add_argument("--joint-stiffness", type=float, nargs=2, default=[0.0, 0.0],
                  metavar=("K1", "K2"))

_traj = parser.add_argument_group("trajectory")
_traj.add_argument("--traj-type", choices=["rect", "circle", "figure8", "line"],
                   default="rect")
_traj.add_argument("--traj-x-range", type=float, nargs=2, default=[0.50, 0.62],
                   metavar=("X_MIN", "X_MAX"))
_traj.add_argument("--traj-y-range", type=float, nargs=2, default=[-0.06, 0.10],
                   metavar=("Y_MIN", "Y_MAX"))
_traj.add_argument("--traj-radius", type=float, default=None)
_traj.add_argument("--traj-n", type=int, default=60)
_traj.add_argument("--traj-v-max", type=float, default=0.35)
_traj.add_argument("--traj-v-corner", type=float, default=0.05)
_traj.add_argument("--traj-corner-blend", type=float, default=0.35)

args = parser.parse_args()

if len(args.ct_kp) == 1:
    args.ct_kp = np.array([args.ct_kp[0], args.ct_kp[0]])
elif len(args.ct_kp) == 2:
    args.ct_kp = np.array(args.ct_kp)
else:
    parser.error("--ct-kp expects 1 or 2 values")

if len(args.ct_kd) == 1:
    args.ct_kd = np.array([args.ct_kd[0], args.ct_kd[0]])
elif len(args.ct_kd) == 2:
    args.ct_kd = np.array(args.ct_kd)
else:
    parser.error("--ct-kd expects 1 or 2 values")


def main() -> None:
    meshcat = None if args.no_meshcat else StartMeshcat()
    if meshcat is not None:
        print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    logs = run_simulation(meshcat, args)
    plot_results(logs, args)


if __name__ == "__main__":
    main()
