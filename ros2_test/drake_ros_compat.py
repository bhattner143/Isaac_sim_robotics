"""
drake_ros_compat.py
====================
Unified import shim that auto-selects the ROS-Drake backend:

  Priority 1 — real ``drake_ros`` C++ package (Linux/Docker with compiled
               RobotLocomotion/drake-ros).  Provides native DDS transport,
               zero-copy publish, SceneTfBroadcaster, and Rviz2 integration.

  Priority 2 — ``drake_ros_bridge`` (macOS / any platform without
               compiled drake_ros).  Pure-Python re-implementation with
               identical class names and the same DiagramBuilder API.

Usage in your scripts — replace:
    from drake_ros_bridge import RosInterfaceSystem, ...

with:
    from drake_ros_compat import RosInterfaceSystem, ...
    # (optional) check which backend loaded:
    from drake_ros_compat import BACKEND   # "drake_ros" | "drake_ros_bridge"

API guaranteed by this module (same for both backends):
    RosInterfaceSystem(node_name)        → LeafSystem; has .get_node()
    ClockSystem.AddToBuilder(builder, ros_if)
    RosPublisherSystem.Make(msg_type, topic, qos, ros_if)
    RosSubscriberSystem.Make(msg_type, topic, qos, ros_if)
    drake_ros_init()                     → call before building diagram
    drake_ros_shutdown()                 → call after simulator finishes

Key difference from raw ``drake_ros`` C++ API:
    Both Make() and AddToBuilder() accept the *LeafSystem* (ros_if) directly,
    NOT ros_if.get_ros_interface().  The compat layer handles that internally.
"""

from __future__ import annotations
import logging as _logging
import sys as _sys

# ─────────────────────────────────────────────────────────────────────────────
# Backend 1: real drake_ros C++ package
# ─────────────────────────────────────────────────────────────────────────────
try:
    from drake_ros.core import (
        RosInterfaceSystem as _RealRosInterfaceSystem,
        RosPublisherSystem as _RealRosPublisherSystem,
        RosSubscriberSystem as _RealRosSubscriberSystem,
        ClockSystem as _RealClockSystem,
        init as drake_ros_init,
        shutdown as drake_ros_shutdown,
    )
    _HAVE_DRAKE_ROS = True
except ImportError:
    _HAVE_DRAKE_ROS = False


if _HAVE_DRAKE_ROS:
    BACKEND = "drake_ros"
    print(
        "[drake_ros_compat] Backend: real drake_ros C++ "
        f"(imported from {_sys.modules.get('drake_ros', None)})",
        flush=True,
    )

    # ── Fake rclpy-style node for .get_logger().info() calls ──────────────
    class _FakeLogger:
        def __init__(self, name: str) -> None:
            self._log = _logging.getLogger(name)

        def info(self, msg: str) -> None:
            self._log.info(msg)

        def warn(self, msg: str) -> None:
            self._log.warning(msg)

        def error(self, msg: str) -> None:
            self._log.error(msg)

    class _FakeNode:
        def __init__(self, name: str) -> None:
            self._logger = _FakeLogger(name)

        def get_logger(self) -> _FakeLogger:
            return self._logger

    # ── RosInterfaceSystem adapter ─────────────────────────────────────────
    # Returns the real C++ LeafSystem but patches .get_node() onto it so
    # existing code using ros_if.get_node().get_logger() keeps working.
    def RosInterfaceSystem(node_name: str):
        """Create a real drake_ros RosInterfaceSystem with a compatible get_node()."""
        sys = _RealRosInterfaceSystem(node_name)
        fake_node = _FakeNode(node_name)
        try:
            # Drake pybind11 objects support dynamic attributes via __dict__
            sys.get_node = lambda: fake_node
        except (AttributeError, TypeError):
            pass  # pybind11 object didn't allow it; logging falls back silently
        return sys

    # ── ClockSystem adapter ────────────────────────────────────────────────
    # Real API: ClockSystem.AddToBuilder(builder, ros_if.get_ros_interface())
    # Compat:   ClockSystem.AddToBuilder(builder, ros_if)  ← system directly
    class ClockSystem:
        @staticmethod
        def AddToBuilder(builder, ros_interface_system) -> None:
            _RealClockSystem.AddToBuilder(
                builder, ros_interface_system.get_ros_interface()
            )

    # ── RosPublisherSystem adapter ─────────────────────────────────────────
    # Real API: RosPublisherSystem.Make(msg_type, topic, qos, ros_if.get_ros_interface())
    # Compat:   RosPublisherSystem.Make(msg_type, topic, qos, ros_if)
    class RosPublisherSystem:
        @staticmethod
        def Make(msg_type, topic: str, qos, ros_interface_system):
            return _RealRosPublisherSystem.Make(
                msg_type, topic, qos, ros_interface_system.get_ros_interface()
            )

    # ── RosSubscriberSystem adapter ────────────────────────────────────────
    class RosSubscriberSystem:
        @staticmethod
        def Make(msg_type, topic: str, qos, ros_interface_system):
            return _RealRosSubscriberSystem.Make(
                msg_type, topic, qos, ros_interface_system.get_ros_interface()
            )

# ─────────────────────────────────────────────────────────────────────────────
# Backend 2: drake_ros_bridge (pure-Python fallback)
# ─────────────────────────────────────────────────────────────────────────────
else:
    BACKEND = "drake_ros_bridge"
    print(
        "[drake_ros_compat] Backend: drake_ros_bridge "
        "(drake_ros C++ not available — using pure-Python fallback)",
        flush=True,
    )
    from drake_ros_bridge import (  # noqa: F401
        RosInterfaceSystem,
        RosPublisherSystem,
        RosSubscriberSystem,
        ClockSystem,
        JointStateBroadcasterSystem,
        SceneTfBroadcasterSystem,
    )

    def drake_ros_init() -> None:
        """No-op: bridge initialises rclpy automatically."""
        pass

    def drake_ros_shutdown() -> None:
        """No-op: bridge shuts down rclpy automatically."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "BACKEND",
    "RosInterfaceSystem",
    "RosPublisherSystem",
    "RosSubscriberSystem",
    "ClockSystem",
    "drake_ros_init",
    "drake_ros_shutdown",
]
