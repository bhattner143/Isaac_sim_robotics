"""01 — Ping each motor with a single MIT 'hold' command and print feedback.

Run on the Jetson after `sudo ip link set can0 up`.

    python -m drivers.cubemars.examples.01_ping --duration 5
"""
from __future__ import annotations

import argparse
import time

from drivers.cubemars import TwoJointArm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    with TwoJointArm(args.channel) as arm:
        t0 = time.time()
        while time.time() - t0 < args.duration:
            q  = arm.q()
            qd = arm.qd()
            print(f"t={time.time()-t0:5.2f}s  "
                  f"q=[{q[0]:+.3f}, {q[1]:+.3f}] rad  "
                  f"qd=[{qd[0]:+.3f}, {qd[1]:+.3f}] rad/s  "
                  f"healthy={arm.healthy()}")
            # Hold pose with default low gains
            arm.command(q_des=q, qd_des=[0.0, 0.0], tau_ff=[0.0, 0.0])
            time.sleep(0.05)


if __name__ == "__main__":
    main()
