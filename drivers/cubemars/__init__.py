"""CubeMars AK-series CAN driver package.

Hardware target: NVIDIA Jetson Orin Nano + SN65HVD230 transceiver +
CubeMars AK80-8-KV60 (shoulder, ID 0x01) + AK60-6-KV80 (elbow, ID 0x02)
on a single 1 Mbps CAN bus.

See: notes_all/notes_cup_manipulator_tendon/hardware/
     CubeMars_Hardware_Implementation.tex
"""
from .config import MotorConfig, MotorMode, AK80_8, AK60_6
from .can_iface import CanBus
from .motor import CubeMarsMotor, MotorState
from .arm import TwoJointArm
from . import protocol

__all__ = [
    "MotorConfig", "MotorMode", "AK80_8", "AK60_6",
    "CanBus", "CubeMarsMotor", "MotorState", "TwoJointArm",
    "protocol",
]
