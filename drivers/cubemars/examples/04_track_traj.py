"""04 — MIT mode + computed-torque feed-forward.

Wire the existing `ComputedTorqueControllerNP` (from the simulation side) into
the hardware via MIT `tau_ff`. This is the *real* control loop we'll use in
experiments.

    python -m drivers.cubemars.examples.04_track_traj --duration 6

NOTE: `ComputedTorqueControllerNP` and the trajectory class signatures may
need light adaptation; this script shows the *pattern*, not a runnable
turn-key demo. Provide the URDF / pulley info before flipping the power on.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from drivers.cubemars import TwoJointArm

# Pull in the simulation-side computed torque (pure NumPy, engine-agnostic).
# from controller.computed_torque_isaacsim import ComputedTorqueControllerNP
# from controller.trajectory import CircleTrajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel",  default="can0")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--rate-hz",  type=float, default=500.0)
    args = parser.parse_args()

    # ct = ComputedTorqueControllerNP(...)  # same gains as sim
    # traj = CircleTrajectory(...)

    dt = 1.0 / args.rate_hz

    with TwoJointArm(args.channel) as arm:
        time.sleep(0.1)
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < args.duration:
            q  = arm.q()
            qd = arm.qd()

            # --- replace these with the actual CT call ---
            t_now = time.time() - t0
            q_des  = q                       # placeholder: hold pose
            qd_des = np.zeros(2)
            qdd_des = np.zeros(2)
            tau_ff = np.zeros(2)             # ct.compute(q, qd, q_des, qd_des, qdd_des)
            # ---------------------------------------------

            arm.command(q_des, qd_des, tau_ff)

            if not arm.healthy():
                print("CAN timeout - aborting")
                break

            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    main()
