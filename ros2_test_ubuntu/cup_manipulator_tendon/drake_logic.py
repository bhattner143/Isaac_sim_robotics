#!/usr/bin/env python3
"""
Drake Manipulator Commander — conda Python 3.11
================================================
Generates manipulator commands (joint positions or EE targets) using
PyDrake and prints them to stdout to be piped to the ROS 2 publisher.

Two modes (set via --mode):
  joint_command — Publish 'JOINT_CMD:q1,q2' lines
  ee_command    — Publish 'EE_CMD:x,y' lines

Architecture:
    drake_logic.py (conda 3.11)
        └─ stdout pipe ──► ros2_publisher.py (system 3.12)
                                └─ publishes ──► /manip/joint_command OR /manip/ee_command
                                                        └─► cup_manipulator_tendon_isaac_sim.py

Usage:
    bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh
"""

import argparse
import sys
import time
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Simulator,
)


# ============================================================================
# DRAKE LEAF SYSTEM: Joint trajectory generator
# ============================================================================
class JointTrajectoryCommander(LeafSystem):
    """
    Generates sinusoidal joint angle trajectories for the 2-DOF manipulator.

    Prints 'JOINT_CMD:q1,q2' to stdout on each periodic publish.
    q1, q2 in radians.
    """

    def __init__(self, q1_center: float, q2_center: float,
                 q1_amp: float, q2_amp: float,
                 freq: float, period_sec: float):
        super().__init__()
        self._q1_center = q1_center
        self._q2_center = q2_center
        self._q1_amp = q1_amp
        self._q2_amp = q2_amp
        self._freq = freq
        self._step_count = 0

        self.DeclareDiscreteState(np.array([q1_center, q2_center]))

        self.DeclareVectorOutputPort(
            "joint_command", BasicVector(2), self._output,
        )

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=period_sec, offset_sec=0.0, update=self._update,
        )

        self.DeclarePeriodicPublishEvent(
            period_sec=period_sec, offset_sec=0.0, publish=self._publish,
        )

    def _output(self, context, output):
        output.SetFromVector(context.get_discrete_state_vector().CopyToVector())

    def _update(self, context, discrete_state):
        t = context.get_time()
        q1 = self._q1_center + self._q1_amp * np.sin(2 * np.pi * self._freq * t)
        q2 = self._q2_center + self._q2_amp * np.sin(2 * np.pi * self._freq * t + np.pi / 3)
        discrete_state.set_value(np.array([q1, q2]))

    def _publish(self, context):
        state = context.get_discrete_state_vector().CopyToVector()
        q1, q2 = state[0], state[1]
        self._step_count += 1

        print(f"JOINT_CMD:{q1:.6f},{q2:.6f}", flush=True)

        if self._step_count % 100 == 0:
            print(
                f"# [Drake] Step {self._step_count} | "
                f"q1={np.rad2deg(q1):+.1f}° q2={np.rad2deg(q2):+.1f}°",
                file=sys.stderr,
            )


# ============================================================================
# DRAKE LEAF SYSTEM: End-effector trajectory generator
# ============================================================================
class EETrajectoryCommander(LeafSystem):
    """
    Generates a circular EE trajectory in the XZ plane.

    Prints 'EE_CMD:x,y' to stdout on each periodic publish.
    """

    def __init__(self, cx: float, cy: float, radius: float,
                 freq: float, period_sec: float):
        super().__init__()
        self._cx = cx
        self._cy = cy
        self._radius = radius
        self._freq = freq
        self._step_count = 0

        self.DeclareDiscreteState(np.array([cx + radius, cy]))

        self.DeclareVectorOutputPort(
            "ee_command", BasicVector(2), self._output,
        )

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=period_sec, offset_sec=0.0, update=self._update,
        )

        self.DeclarePeriodicPublishEvent(
            period_sec=period_sec, offset_sec=0.0, publish=self._publish,
        )

    def _output(self, context, output):
        output.SetFromVector(context.get_discrete_state_vector().CopyToVector())

    def _update(self, context, discrete_state):
        t = context.get_time()
        x = self._cx + self._radius * np.cos(2 * np.pi * self._freq * t)
        y = self._cy + self._radius * np.sin(2 * np.pi * self._freq * t)
        discrete_state.set_value(np.array([x, y]))

    def _publish(self, context):
        state = context.get_discrete_state_vector().CopyToVector()
        x, y = state[0], state[1]
        self._step_count += 1

        print(f"EE_CMD:{x:.6f},{y:.6f}", flush=True)

        if self._step_count % 100 == 0:
            print(
                f"# [Drake] Step {self._step_count} | "
                f"EE=({x:.4f}, {y:.4f})",
                file=sys.stderr,
            )


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Drake Manipulator Commander — sends joint/EE commands via stdout pipe"
    )
    parser.add_argument(
        '--mode', choices=('joint_command', 'ee_command'), default='joint_command',
        help='joint_command: sine wave on joints | ee_command: circular EE path',
    )
    parser.add_argument('--period', type=float, default=0.005,
                        help='Update period [s] (default: 0.005 = 200 Hz)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Total simulation duration [s]')
    parser.add_argument('--freq', type=float, default=0.2,
                        help='Trajectory frequency [Hz] (default: 0.2)')

    # Joint mode params
    parser.add_argument('--q1-center', type=float, default=10.0,
                        help='q1 center angle [deg] (default: 10)')
    parser.add_argument('--q2-center', type=float, default=-10.0,
                        help='q2 center angle [deg] (default: -10)')
    parser.add_argument('--q1-amp', type=float, default=20.0,
                        help='q1 amplitude [deg] (default: 20)')
    parser.add_argument('--q2-amp', type=float, default=15.0,
                        help='q2 amplitude [deg] (default: 15)')

    # EE mode params
    parser.add_argument('--ee-cx', type=float, default=0.15,
                        help='EE circle center X [m]')
    parser.add_argument('--ee-cy', type=float, default=0.35,
                        help='EE circle center Y [m]')
    parser.add_argument('--ee-radius', type=float, default=0.05,
                        help='EE circle radius [m]')

    args = parser.parse_args()

    print(f"# Drake Manipulator Commander starting — mode={args.mode}", file=sys.stderr)
    print(f"#   period    : {args.period} s ({1/args.period:.0f} Hz)", file=sys.stderr)
    print(f"#   duration  : {args.duration} s", file=sys.stderr)
    print(f"#   frequency : {args.freq} Hz", file=sys.stderr)

    builder = DiagramBuilder()

    if args.mode == 'joint_command':
        system = builder.AddSystem(
            JointTrajectoryCommander(
                q1_center=np.deg2rad(args.q1_center),
                q2_center=np.deg2rad(args.q2_center),
                q1_amp=np.deg2rad(args.q1_amp),
                q2_amp=np.deg2rad(args.q2_amp),
                freq=args.freq,
                period_sec=args.period,
            )
        )
        system.set_name("joint_commander")
        print(f"#   q1 = {args.q1_center}° ± {args.q1_amp}°", file=sys.stderr)
        print(f"#   q2 = {args.q2_center}° ± {args.q2_amp}°", file=sys.stderr)
    else:
        system = builder.AddSystem(
            EETrajectoryCommander(
                cx=args.ee_cx,
                cy=args.ee_cy,
                radius=args.ee_radius,
                freq=args.freq,
                period_sec=args.period,
            )
        )
        system.set_name("ee_commander")
        print(
            f"#   circle center=({args.ee_cx}, {args.ee_cy}) "
            f"radius={args.ee_radius}m",
            file=sys.stderr,
        )

    diagram = builder.Build()
    diagram.set_name("drake_manip_commander")

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    print(f"# Publishing {args.mode} to stdout → ROS 2", file=sys.stderr)
    simulator.AdvanceTo(args.duration)

    print(f"# Drake Manipulator Commander finished.", file=sys.stderr)


if __name__ == '__main__':
    main()
