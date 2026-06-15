"""02 — Servo position-loop step (no impedance), one motor at a time.

This avoids MIT mode for the very first power-up. Use the R-Link tool
beforehand to confirm motor parameters and CAN ID.

    python -m drivers.cubemars.examples.02_position_step --motor shoulder --angle-deg 5
"""
from __future__ import annotations

import argparse
import struct
import time

import can

from drivers.cubemars.config import AK60_6, AK80_8, MotorMode
from drivers.cubemars import protocol as proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor", choices=["shoulder", "elbow"],
                        default="shoulder")
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--hold", type=float, default=2.0)
    args = parser.parse_args()

    cfg = AK80_8 if args.motor == "shoulder" else AK60_6
    bus = can.interface.Bus(channel=args.channel, bustype="socketcan",
                            bitrate=1_000_000)

    can_id = proto.servo_can_id(MotorMode.POSITION, cfg.can_id)
    payload = proto.encode_position_loop(args.angle_deg)
    msg = can.Message(arbitration_id=can_id, data=payload,
                      is_extended_id=True)

    print(f"Sending POSITION {args.angle_deg:+.2f} deg to {cfg.name} "
          f"(can_id=0x{can_id:X})")
    bus.send(msg)
    time.sleep(args.hold)

    # Bring back to zero, then disable
    bus.send(can.Message(arbitration_id=can_id,
                         data=proto.encode_position_loop(0.0),
                         is_extended_id=True))
    time.sleep(args.hold)
    bus.send(can.Message(
        arbitration_id=proto.servo_can_id(MotorMode.DISABLE, cfg.can_id),
        data=b"", is_extended_id=True))
    bus.shutdown()


if __name__ == "__main__":
    main()
