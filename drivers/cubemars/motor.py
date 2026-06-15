"""Single-motor CubeMars wrapper: state, MIT command, feedback parsing,
timeout / safe-stop helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import protocol as proto
from .config import MotorConfig


@dataclass
class MotorState:
    position: float = 0.0   # rad
    velocity: float = 0.0   # rad/s
    current:  float = 0.0   # A
    last_rx:  float = 0.0   # epoch s
    enabled:  bool  = False
    fault:    Optional[str] = None


class CubeMarsMotor:
    """One motor on a shared CAN bus."""

    def __init__(self, bus, cfg: MotorConfig):
        self.bus = bus
        self.cfg = cfg
        self.state = MotorState()

    # ------------------------------------------------------------------ mode
    def enable_mit(self) -> None:
        self.bus.send(self.cfg.can_id, proto.ENTER_MIT, extended=False)
        self.state.enabled = True

    def disable(self) -> None:
        self.bus.send(self.cfg.can_id, proto.EXIT_MIT, extended=False)
        self.state.enabled = False

    def zero_here(self) -> None:
        """Set the current shaft position as the new origin (MIT prelude)."""
        self.bus.send(self.cfg.can_id, proto.ZERO_POS, extended=False)

    # --------------------------------------------------------------- command
    def send_mit(self,
                 p_des: float,
                 v_des: float,
                 tau_ff: float,
                 kp: Optional[float] = None,
                 kd: Optional[float] = None) -> None:
        kp = self.cfg.kp_default if kp is None else kp
        kd = self.cfg.kd_default if kd is None else kd

        # safety clamp on host before encoding
        tau_lim = self.cfg.clamped_tau()
        if tau_ff > tau_lim:
            tau_ff = tau_lim
        elif tau_ff < -tau_lim:
            tau_ff = -tau_lim

        data = proto.encode_mit(self.cfg, p_des, v_des, tau_ff, kp, kd)
        self.bus.send(self.cfg.can_id, data, extended=False)

    # --------------------------------------------------------------- feedback
    def consume_feedback(self, msg) -> None:
        """Call from the rx loop when arbitration_id matches self.cfg.can_id."""
        try:
            _, p, v, i = proto.decode_mit_feedback(self.cfg, bytes(msg.data))
        except ValueError:
            return
        self.state.position = p
        self.state.velocity = v
        self.state.current  = i
        self.state.last_rx  = time.time()

    # ---------------------------------------------------------------- safety
    def timed_out(self, timeout: float = 0.1) -> bool:
        return (time.time() - self.state.last_rx) > timeout

    def safe_stop(self) -> None:
        """Hold current position with zero feed-forward and gentle damping."""
        self.send_mit(self.state.position, 0.0, 0.0, kp=0.0, kd=0.5)
