#!/usr/bin/env python3
"""
script_mhp_manipulator_cable_framework_pydrake.py
=================================================
MHP Plant A — practical cable controller with correct hardware topology::

    Shoulder (q1)  — DIRECT DRIVE MIT motor
    Elbow    (q2)  — ONE motor, antagonistic lower (+Y) / upper (−Y) cables
                     (one taut, one slack)

    EE traj → CT → τ_req → tension split → MIT motors → Plant A

USAGE
─────
  python script_mhp_manipulator_cable_framework_pydrake.py
  python script_mhp_manipulator_cable_framework_pydrake.py --no-meshcat --duration 4
  python script_mhp_manipulator_cable_framework_pydrake.py --no-elbow-ff-from-cable
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

from termcolor import colored
from pydrake.all import StartMeshcat

sys.path.insert(0, str(Path(__file__).parent))

from cable.simulation_mhp_cable_framework import run_simulation, plot_results

parser = argparse.ArgumentParser(
    description="MHP Plant A — shoulder direct + elbow antagonistic cable",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

_ct = parser.add_argument_group("computed-torque gains")
_ct.add_argument("--ct-kp", type=float, nargs="+", default=[400.0, 400.0], metavar="KP")
_ct.add_argument("--ct-kd", type=float, nargs="+", default=[40.0, 40.0], metavar="KD")
_ct.add_argument("--ct-tau-max", type=float, default=10.0, help="Torque saturation [Nm].")

_mit = parser.add_argument_group("dummy MIT motors")
_mit.add_argument("--mit-kp", type=float, nargs=2, default=[30.0, 15.0],
                  metavar=("KP_SHO", "KP_ELB"),
                  help="MIT Kp: [shoulder direct, elbow cable motor].")
_mit.add_argument("--mit-kd", type=float, nargs=2, default=[1.5, 0.5],
                  metavar=("KD_SHO", "KD_ELB"),
                  help="MIT Kd: [shoulder direct, elbow cable motor].")
_mit.add_argument("--mit-dynamics", action="store_true",
                  help="Use 2nd-order rotor dynamics (default: algebraic MIT).")
_mit.add_argument("--tension-kp", type=float, default=0.5,
                  metavar="KF", help="Elbow cable inner-loop gain on F_net [Nm/N].")
_mit.add_argument("--tension-noise", type=float, default=0.0,
                  help="Simulated load-cell noise std on taut cable [N].")
_mit.add_argument(
    "--elbow-ff-from-cable",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Elbow MIT τ_ff source: enabled → τ_ff₂ = r_p·F_cmd (after tension inner loop); "
         "disabled (--no-elbow-ff-from-cable) → τ_ff₂ = τ₂_req from CT.",
)

_sim = parser.add_argument_group("simulation")
_sim.add_argument("--duration", type=float, default=10.0)
_sim.add_argument("--num-laps", type=int, default=1)
_sim.add_argument("--move-duration", type=float, default=3.0)
_sim.add_argument("--no-meshcat", action="store_true")
_sim.add_argument("--record", action="store_true", default=False)
_sim.add_argument("--no-show", action="store_true")
_sim.add_argument("--urdf-alpha", type=float, default=0.25)

_rob = parser.add_argument_group("robot mount")
_rob.add_argument("--tilt-roll", type=float, default=0.0)
_rob.add_argument("--tilt-pitch", type=float, default=0.0)
_rob.add_argument("--joint-damping", type=float, nargs=2, default=[0.05, 0.05])
_rob.add_argument("--joint-stiffness", type=float, nargs=2, default=[0.0, 0.0])

_traj = parser.add_argument_group("trajectory")
_traj.add_argument("--traj-type", choices=["rect", "circle", "figure8", "line"], default="rect")
_traj.add_argument("--traj-x-range", type=float, nargs=2, default=[0.50, 0.62])
_traj.add_argument("--traj-y-range", type=float, nargs=2, default=[-0.06, 0.10])
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
