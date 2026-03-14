"""
Drake Cube Commander — conda Python 3.11
=========================================
Uses PyDrake to compute cube target positions (1cm step along X-axis
every second) and prints them to stdout to be piped to the ROS 2
cube commander node.

Architecture:
    drake_logic.py (conda pydrake 3.11)
        └─ stdout pipe ──► ros2_publisher.py (system Python 3.12)
                                └─ publishes ──► /cube_target_pos (geometry_msgs/Point)
                                                        └─► ros2_subscriber.py
                                                                └─ stdout ──► isaac_sim.py

Cube motion:
    Starts at x=0.0m, moves +1cm every second along X-axis
    e.g. 0.00 → 0.01 → 0.02 → ... → 0.10m (10 steps)

Usage:
    bash ros2_test_ubuntu/cube_commander/run_commander.sh
"""

import argparse
import sys
import time
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    AbstractValue,
    Simulator,
)


# ============================================================================
# DRAKE LEAF SYSTEM: Cube position generator
# ============================================================================
class CubePositionCommander(LeafSystem):
    """
    Drake LeafSystem that generates cube target positions.

    Increments cube X-position by step_size every period_sec.
    Prints position as 'x,y,z' to stdout → piped to ROS 2 node.

    State:  discrete [x, y, z]
    Output: prints 'CUBE_POS:x,y,z' to stdout
    """

    def __init__(self, step_size: float = 0.001, period_sec: float = 0.005,
                 num_steps: int = 1000):  # 200 Hz, 0.1 cm/step → 1 m total
        super().__init__()
        self.step_size  = step_size
        self.num_steps  = num_steps
        self._step_count = 0

        # Discrete state: [x, y, z] — z=0.1 keeps cube sitting on the ground
        # (cube scale=0.2m → half-height=0.1m → ground contact at z=0.1)
        self.DeclareDiscreteState(np.array([0.0, 0.0, 0.1]))

        # Output port: current position
        self.DeclareVectorOutputPort(
            "cube_position",
            BasicVector(3),
            self._output_position,
        )

        # Periodic update: increment position every period_sec
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec = period_sec,
            offset_sec = 0.0,
            update     = self._update_position,
        )

        # Periodic publish to stdout (same rate)
        self.DeclarePeriodicPublishEvent(
            period_sec = period_sec,
            offset_sec = 0.0,
            publish    = self._publish_position,
        )

    def _output_position(self, context, output):
        state = context.get_discrete_state_vector().CopyToVector()
        output.SetFromVector(state)

    def _update_position(self, context, discrete_state):
        """Increment X by step_size each tick."""
        current = context.get_discrete_state_vector().CopyToVector()
        self._step_count += 1

        if self._step_count <= self.num_steps:
            current[0] += self.step_size  # +1cm along X

        discrete_state.set_value(current)

    def _publish_position(self, context):
        """Print current position to stdout → pipe → ROS 2 node."""
        state = context.get_discrete_state_vector().CopyToVector()
        x, y, z = state[0], state[1], state[2]

        msg = f"CUBE_POS:{x:.4f},{y:.4f},{z:.4f}"
        print(msg, flush=True)

        print(
            f"# [Drake] Step {self._step_count}/{self.num_steps} | "
            f"Cube X = {x*100:.1f} cm",
            file=sys.stderr
        )


# ============================================================================
# BUILD DRAKE DIAGRAM
# ============================================================================
def build_diagram(step_size: float, period_sec: float, num_steps: int):
    """Build Drake diagram with CubePositionCommander."""
    builder = DiagramBuilder()

    commander = builder.AddSystem(
        CubePositionCommander(step_size, period_sec, num_steps)
    )
    commander.set_name("cube_commander")

    diagram = builder.Build()
    diagram.set_name("drake_cube_control")
    return diagram, commander


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Drake Cube Commander — moves cube 0.2cm/step along X via ROS 2"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.001,
        help="Position increment per step in metres (default: 0.001 = 0.1cm @ 200Hz → 20cm/s)"
    )
    parser.add_argument(
        "--period", type=float, default=0.005,
        help="Time between steps in seconds (default: 0.005 = 200 Hz — impedance control rate)"
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps (default: 1000 = 1m total @ 200Hz over 5s)"
    )
    args = parser.parse_args()

    total_distance = args.step_size * args.steps * 100  # in cm
    duration       = args.period * (args.steps + 1)

    print(f"# Drake Cube Commander starting", file=sys.stderr)
    print(f"#   step size : {args.step_size*100:.1f} cm", file=sys.stderr)
    print(f"#   period    : {args.period} s", file=sys.stderr)
    print(f"#   steps     : {args.steps}", file=sys.stderr)
    print(f"#   total     : {total_distance:.1f} cm along X-axis", file=sys.stderr)
    print(f"# Publishing CUBE_POS to stdout → ROS 2 /cube_target_pos", file=sys.stderr)

    # Build and run
    diagram, commander = build_diagram(args.step_size, args.period, args.steps)

    simulator = Simulator(diagram)
    context   = simulator.get_mutable_context()

    # Initial position: x=0, y=0, z=0.1 (cube resting on ground, half of 0.2m scale)
    commander_context = commander.GetMyMutableContextFromRoot(context)
    commander_context.get_mutable_discrete_state_vector().SetFromVector(
        [0.0, 0.0, 0.1]
    )

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(duration)

    print(f"# Drake Cube Commander finished. Cube moved {total_distance:.1f}cm.", file=sys.stderr)


if __name__ == '__main__':
    main()
