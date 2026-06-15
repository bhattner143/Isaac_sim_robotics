"""Byte-accurate CubeMars V3 driver protocol.

References:
    AK Series Manual V3.2.0
        S4.1 Servo-mode CAN frames
        S4.2 MIT (force-control) mode prelude / payload
        S4.3 Feedback frame layouts

All multi-byte ints are big-endian within the CAN payload.
Float <-> uint quantization follows the standard MIT-Cheetah scheme.
"""
from __future__ import annotations

import struct
from typing import Tuple

from .config import MotorConfig


# ----------------------------------------------------------------------------
# Float <-> bit-field helpers
# ----------------------------------------------------------------------------
def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    span = x_max - x_min
    x = max(min(x, x_max), x_min)
    return int((x - x_min) * (((1 << bits) - 1) / span))


def uint_to_float(u: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return float(u) * span / ((1 << bits) - 1) + x_min


# ----------------------------------------------------------------------------
# MIT mode (function-ID 8, but uses driver_id directly as arbitration ID
# for both command and feedback — see manual S4.2)
# ----------------------------------------------------------------------------
ENTER_MIT = bytes([0xFF] * 7 + [0xFC])
EXIT_MIT  = bytes([0xFF] * 7 + [0xFD])
ZERO_POS  = bytes([0xFF] * 7 + [0xFE])


def encode_mit(cfg: MotorConfig,
               p_des: float, v_des: float, tau_ff: float,
               kp: float, kd: float) -> bytes:
    """Pack 8-byte MIT command. Layout (manual S4.2 table):

        byte 0 : kp[15:8]
        byte 1 : kp[7:0] | kd[15:12]
        byte 2 : kd[11:4]
        byte 3 : pos[15:8]
        byte 4 : pos[7:0]
        byte 5 : vel[11:4]
        byte 6 : vel[3:0] | tau[11:8]
        byte 7 : tau[7:0]

    Note: the manual reuses 12-bit fields for kp / kd; the encoding above
    treats kp/kd as 12-bit unsigned in [0, kp_max] / [0, kd_max].
    """
    p   = float_to_uint(p_des,  cfg.p_min,   cfg.p_max,   16)
    v   = float_to_uint(v_des,  cfg.v_min,   cfg.v_max,   12)
    t   = float_to_uint(tau_ff, cfg.tau_min, cfg.tau_max, 12)
    kp_ = float_to_uint(kp,     0.0,         cfg.kp_max,  12)
    kd_ = float_to_uint(kd,     0.0,         cfg.kd_max,  12)

    return bytes([
        (kp_ >> 4) & 0xFF,
        ((kp_ & 0x0F) << 4) | ((kd_ >> 8) & 0x0F),
        kd_ & 0xFF,
        (p >> 8) & 0xFF,
        p & 0xFF,
        (v >> 4) & 0xFF,
        ((v & 0x0F) << 4) | ((t >> 8) & 0x0F),
        t & 0xFF,
    ])


def decode_mit_feedback(cfg: MotorConfig, data: bytes
                        ) -> Tuple[int, float, float, float]:
    """Decode 6-byte MIT feedback frame.

    Returns
    -------
    motor_id, position [rad], velocity [rad/s], current [A]
    """
    if len(data) < 6:
        raise ValueError(f"MIT feedback needs 6 bytes, got {len(data)}")

    motor_id = data[0]
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | ((data[4] >> 4) & 0x0F)
    i_int = ((data[4] & 0x0F) << 8) | data[5]

    p = uint_to_float(p_int, cfg.p_min, cfg.p_max, 16)
    v = uint_to_float(v_int, cfg.v_min, cfg.v_max, 12)
    # current is signed +-60 A in the V3 firmware (manual S4.3)
    i = uint_to_float(i_int, -60.0, 60.0, 12)
    return motor_id, p, v, i


# ----------------------------------------------------------------------------
# Servo mode (extended CAN ID = (function_id << 8) | driver_id)
# Useful only for bring-up; payloads are simple big-endian ints.
# ----------------------------------------------------------------------------
def servo_can_id(function_id: int, driver_id: int) -> int:
    return ((function_id & 0xFF) << 8) | (driver_id & 0xFF)


def encode_position_loop(angle_deg: float) -> bytes:
    """Servo function ID 4: position in 0.0001 deg steps, 4-byte int32 BE."""
    raw = int(round(angle_deg * 10_000.0))
    return struct.pack(">i", raw)


def encode_current_loop(current_a: float) -> bytes:
    """Servo function ID 1: current in mA, 4-byte int32 BE."""
    raw = int(round(current_a * 1000.0))
    return struct.pack(">i", raw)


def encode_velocity_loop(speed_erpm: float) -> bytes:
    """Servo function ID 3: ERPM (motor side), 4-byte int32 BE."""
    return struct.pack(">i", int(round(speed_erpm)))


def decode_servo_feedback(data: bytes) -> dict:
    """Decode the 8-byte feedback packet emitted under arbitration ID
    (0x29 << 8) | driver_id when servo modes run with feedback enabled."""
    if len(data) < 8:
        raise ValueError(f"servo feedback needs 8 bytes, got {len(data)}")
    pos_raw  = struct.unpack(">h", data[0:2])[0]   # 0.1 deg
    spd_raw  = struct.unpack(">h", data[2:4])[0]   # ERPM / 10
    cur_raw  = struct.unpack(">h", data[4:6])[0]   # 0.01 A
    temp_c   = data[6]
    err_code = data[7]
    return {
        "position_deg": pos_raw * 0.1,
        "speed_erpm":   spd_raw * 10.0,
        "current_a":    cur_raw * 0.01,
        "temperature":  temp_c,
        "error_code":   err_code,
    }
