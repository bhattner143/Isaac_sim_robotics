"""Two-joint planar manipulator controller for the cup-manipulator project.

Shoulder = AK80-8 (CAN id 0x01)
Elbow    = AK60-6 (CAN id 0x02)

Bridges the existing simulation-side `ComputedTorqueControllerNP` to the
hardware MIT-mode interface.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np

from .can_iface import CanBus
from .config import AK60_6, AK80_8, MotorConfig
from .motor import CubeMarsMotor


class TwoJointArm:
    """High-level wrapper. Owns the CAN bus, both motors and the rx thread."""

    def __init__(self,
                 channel: str = "can0",
                 shoulder_cfg: MotorConfig = AK80_8,
                 elbow_cfg:    MotorConfig = AK60_6):
        self.bus = CanBus(channel)
        self.shoulder = CubeMarsMotor(self.bus, shoulder_cfg)
        self.elbow    = CubeMarsMotor(self.bus, elbow_cfg)

        self._rx_stop = threading.Event()
        self._rx_thr: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._rx_stop.clear()
        self._rx_thr = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thr.start()

        self.shoulder.enable_mit()
        self.elbow.enable_mit()
        time.sleep(0.05)   # give first feedback a chance to arrive
        # Hold whatever pose we are in
        q0 = self.q()
        self.command(q0, np.zeros(2), np.zeros(2))

    def stop(self) -> None:
        try:
            self.shoulder.safe_stop()
            self.elbow.safe_stop()
            time.sleep(0.05)
            self.shoulder.disable()
            self.elbow.disable()
        finally:
            self._rx_stop.set()
            if self._rx_thr is not None:
                self._rx_thr.join(timeout=0.5)
            self.bus.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ----------------------------------------------------------------- state
    def q(self) -> np.ndarray:
        return np.array([self.shoulder.state.position,
                         self.elbow.state.position])

    def qd(self) -> np.ndarray:
        return np.array([self.shoulder.state.velocity,
                         self.elbow.state.velocity])

    def tau_estimate(self) -> np.ndarray:
        """Joint-side torque inferred from current * Kt."""
        return np.array([
            self.shoulder.cfg.kt * self.shoulder.state.current,
            self.elbow.cfg.kt    * self.elbow.state.current,
        ])

    # --------------------------------------------------------------- command
    def command(self,
                q_des:  Sequence[float],
                qd_des: Sequence[float],
                tau_ff: Sequence[float],
                kp: Optional[Sequence[float]] = None,
                kd: Optional[Sequence[float]] = None) -> None:
        kp = (None, None) if kp is None else kp
        kd = (None, None) if kd is None else kd
        self.shoulder.send_mit(q_des[0], qd_des[0], tau_ff[0], kp[0], kd[0])
        self.elbow.send_mit(   q_des[1], qd_des[1], tau_ff[1], kp[1], kd[1])

    # ---------------------------------------------------------------- timeout
    def healthy(self, timeout: float = 0.1) -> bool:
        return (not self.shoulder.timed_out(timeout)
                and not self.elbow.timed_out(timeout))

    # ------------------------------------------------------------------ rx
    def _rx_loop(self) -> None:
        sho_id = self.shoulder.cfg.can_id
        elb_id = self.elbow.cfg.can_id
        while not self._rx_stop.is_set():
            msg = self.bus.recv(timeout=0.005)
            if msg is None:
                continue
            if msg.arbitration_id == sho_id:
                self.shoulder.consume_feedback(msg)
            elif msg.arbitration_id == elb_id:
                self.elbow.consume_feedback(msg)
