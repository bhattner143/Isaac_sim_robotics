"""SocketCAN wrapper around python-can for the Jetson Orin Nano TTCAN."""
from __future__ import annotations

from typing import Optional

try:
    import can  # python-can
except ImportError:  # graceful failure when developing on macOS
    can = None  # type: ignore


class CanBus:
    """Thin wrapper. Only exposes what we actually need."""

    def __init__(self, channel: str = "can0", bitrate: int = 1_000_000):
        if can is None:
            raise RuntimeError(
                "python-can is not installed. Run `pip install python-can` "
                "on the Jetson before instantiating CanBus."
            )
        self.channel = channel
        self.bitrate = bitrate
        self.bus = can.interface.Bus(
            channel=channel,
            bustype="socketcan",
            bitrate=bitrate,
        )

    # ------------------------------------------------------------------
    def send(self, can_id: int, data: bytes, extended: bool = False) -> None:
        msg = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=extended,
            is_remote_frame=False,
            is_error_frame=False,
        )
        # 5 ms tx timeout; if we exceed it, bus is in error-passive
        self.bus.send(msg, timeout=0.005)

    def recv(self, timeout: float = 0.001):
        return self.bus.recv(timeout=timeout)

    def shutdown(self) -> None:
        try:
            self.bus.shutdown()
        except Exception:
            pass

    # context-manager sugar -------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
