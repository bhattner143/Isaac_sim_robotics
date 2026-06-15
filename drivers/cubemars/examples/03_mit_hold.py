"""03 — MIT-mode 'soft hold' at the current pose.

Push the arm by hand: it should resist gently and return.
This is the canonical MIT-mode sanity check.

    python -m drivers.cubemars.examples.03_mit_hold --duration 10 --kp 5 --kd 0.5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from drivers.cubemars import TwoJointArm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel",  default="can0")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.5)
    args = parser.parse_args()

    with TwoJointArm(args.channel) as arm:
        # Capture starting pose as the hold point
        time.sleep(0.1)
        q_hold = arm.q()
        print(f"Holding at q = {q_hold}")

        rate_hz = 200.0
        dt = 1.0 / rate_hz
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < args.duration:
            arm.command(q_des=q_hold,
                        qd_des=np.zeros(2),
                        tau_ff=np.zeros(2),
                        kp=[args.kp, args.kp],
                        kd=[args.kd, args.kd])
            if not arm.healthy():
                print("CAN timeout!")
                break
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    main()
