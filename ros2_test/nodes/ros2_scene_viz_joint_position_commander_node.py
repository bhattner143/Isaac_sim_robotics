"""
Joint Position Commander — ROS 2 Node
=======================================

Sends joint position commands to the scene-viz plant node and receives
the applied joint state back.

Published topics:
    /joint_position_command  (std_msgs/Float64MultiArray) — [q1, q2] rad

Subscribed topics:
    /joint_states            (sensor_msgs/JointState)     — echo from plant
    /ee_position             (geometry_msgs/Point)        — EE from plant

Architecture
────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  ros2_joint_position_commander_node.py                       │
  │                                                              │
  │   /joint_position_command  ──►  (plant node)                 │
  │                                                              │
  │   /joint_states     ◄──  (plant echo)                        │
  │   /ee_position      ◄──  (plant echo)                        │
  └──────────────────────────────────────────────────────────────┘

Trajectory modes:
    hold      — hold fixed joint angles (default: 0, 0)
    sine      — sinusoidal sweep on both joints
    step      — step changes between two poses

Usage:
    conda activate pydrake_ros2
    python ros2_test/ros2_joint_position_commander_node.py --traj sine
"""

from __future__ import annotations

import argparse
import math
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

# ── Constants ────────────────────────────────────────────────────────────────
JOINT_NAMES = ["link1_base", "link2_link1"]
NUM_JOINTS = len(JOINT_NAMES)


# ─── Trajectory generators ──────────────────────────────────────────────────
class HoldTrajectory:
    """Constant position."""

    def __init__(self, q_hold: np.ndarray):
        self._q = np.asarray(q_hold, dtype=float)

    def evaluate(self, t: float):
        return self._q.copy()


class SineTrajectory:
    """Independent sinusoidal sweep per joint."""

    def __init__(
        self,
        amplitudes_deg: tuple = (30.0, 20.0),
        frequencies_hz: tuple = (0.25, 0.35),
        offsets_deg: tuple = (0.0, 0.0),
    ):
        self._amp = np.deg2rad(amplitudes_deg)
        self._freq = np.asarray(frequencies_hz, dtype=float)
        self._off = np.deg2rad(offsets_deg)

    def evaluate(self, t: float):
        return self._off + self._amp * np.sin(2.0 * np.pi * self._freq * t)


class StepTrajectory:
    """Alternates between two poses at a fixed period."""

    def __init__(
        self,
        q_a_deg: tuple = (0.0, 0.0),
        q_b_deg: tuple = (30.0, -20.0),
        period_s: float = 4.0,
    ):
        self._qa = np.deg2rad(q_a_deg)
        self._qb = np.deg2rad(q_b_deg)
        self._half = period_s / 2.0

    def evaluate(self, t: float):
        phase = t % (2.0 * self._half)
        return self._qa.copy() if phase < self._half else self._qb.copy()


# ═════════════════════════════════════════════════════════════════════════════
class JointPositionCommanderNode(Node):
    """Publishes joint position commands; logs echoed state from plant."""

    def __init__(
        self,
        trajectory,
        command_rate_hz: float = 30.0,
        log_rate_hz: float = 2.0,
    ):
        super().__init__("joint_position_commander")

        self.trajectory = trajectory
        self._t0 = None

        # ── Latest feedback (protected by lock) ─────────────────────────
        self._lock = threading.Lock()
        self._q_fb = np.zeros(NUM_JOINTS)
        self._ee_fb = np.zeros(3)
        self._fb_received = False

        # ── Publishers ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, "/joint_position_command", 10
        )

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self.create_subscription(
            Point, "/ee_position", self._ee_cb, 10
        )

        # ── Timers ───────────────────────────────────────────────────────
        # Small one-shot delay before starting commands — gives ROS2 DDS
        # time to discover the plant node's /joint_position_command subscriber.
        self._ready = False
        self._ready_timer = self.create_timer(1.5, self._mark_ready)
        self.create_timer(1.0 / command_rate_hz, self._command_tick)
        self._log_period = max(1, int(command_rate_hz / log_rate_hz))
        self._tick_count = 0

        self.get_logger().info(
            f"Joint position commander ready  |  "
            f"rate={command_rate_hz} Hz  |  "
            f"trajectory={type(trajectory).__name__}"
        )

    # ── callbacks ────────────────────────────────────────────────────────
    def _mark_ready(self):
        """One-shot timer fires after 1.5s — enables command publishing."""
        self._ready = True
        self._ready_timer.cancel()   # fire only once
        self.get_logger().info("Commander armed — starting trajectory")

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) != NUM_JOINTS:
            return
        with self._lock:
            if not self._fb_received:
                self._fb_received = True
                self.get_logger().info("Received first /joint_states from plant")
            for i, name in enumerate(msg.name):
                if name in JOINT_NAMES:
                    idx = JOINT_NAMES.index(name)
                    self._q_fb[idx] = msg.position[i]

    def _ee_cb(self, msg: Point):
        with self._lock:
            self._ee_fb = np.array([msg.x, msg.y, msg.z])

    def _command_tick(self):
        if not self._ready:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0

        q_cmd = self.trajectory.evaluate(t)

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.cmd_pub.publish(msg)

        # Periodic log
        self._tick_count += 1
        if self._tick_count % self._log_period == 0:
            with self._lock:
                q_fb = self._q_fb.copy()
                ee = self._ee_fb.copy()
            self.get_logger().info(
                f"t={t:6.2f}s  cmd=[{np.rad2deg(q_cmd[0]):+7.2f}, {np.rad2deg(q_cmd[1]):+7.2f}]°  "
                f"fb=[{np.rad2deg(q_fb[0]):+7.2f}, {np.rad2deg(q_fb[1]):+7.2f}]°  "
                f"EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})"
            )


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Joint Position Commander — ROS 2 Node"
    )
    ap.add_argument(
        "--traj", type=str, default="sine",
        choices=["hold", "sine", "step"],
        help="Trajectory mode  (default: sine)",
    )
    ap.add_argument(
        "--rate", type=float, default=30.0,
        help="Command publish rate (Hz)  [default: 30]",
    )
    ap.add_argument(
        "--hold-q1", type=float, default=0.0,
        help="Hold: q1 degrees  [default: 0]",
    )
    ap.add_argument(
        "--hold-q2", type=float, default=0.0,
        help="Hold: q2 degrees  [default: 0]",
    )
    ap.add_argument(
        "--sine-amp", type=float, nargs=2, default=[30.0, 20.0],
        metavar=("A1", "A2"),
        help="Sine: amplitude degrees  [default: 30 20]",
    )
    ap.add_argument(
        "--sine-freq", type=float, nargs=2, default=[0.25, 0.35],
        metavar=("F1", "F2"),
        help="Sine: frequency Hz  [default: 0.25 0.35]",
    )
    ap.add_argument(
        "--step-a", type=float, nargs=2, default=[0.0, 0.0],
        metavar=("Q1A", "Q2A"),
        help="Step: pose A degrees  [default: 0 0]",
    )
    ap.add_argument(
        "--step-b", type=float, nargs=2, default=[30.0, -20.0],
        metavar=("Q1B", "Q2B"),
        help="Step: pose B degrees  [default: 30 -20]",
    )
    ap.add_argument(
        "--step-period", type=float, default=4.0,
        help="Step: full period seconds  [default: 4]",
    )
    args = ap.parse_args()

    # Build trajectory
    if args.traj == "hold":
        traj = HoldTrajectory(np.deg2rad([args.hold_q1, args.hold_q2]))
    elif args.traj == "sine":
        traj = SineTrajectory(
            amplitudes_deg=tuple(args.sine_amp),
            frequencies_hz=tuple(args.sine_freq),
        )
    elif args.traj == "step":
        traj = StepTrajectory(
            q_a_deg=tuple(args.step_a),
            q_b_deg=tuple(args.step_b),
            period_s=args.step_period,
        )
    else:
        raise ValueError(f"Unknown trajectory: {args.traj}")

    rclpy.init()
    node = JointPositionCommanderNode(
        trajectory=traj,
        command_rate_hz=args.rate,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
