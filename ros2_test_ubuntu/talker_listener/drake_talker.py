"""
Drake Hello World Talker — conda Python 3.11
=============================================
Uses PyDrake to simulate a simple pendulum and publish its state
as "Hello from Drake" messages via ROS 2.

Runs in conda (pydrake) and pipes stdout to ros2_drake_talker_node.py
which runs in system Python 3.12 (where rclpy is installed).

Usage (run from repo root):
    conda activate pydrake
    python ros2_test_ubuntu/script_drake_talker.py | \\
        /usr/bin/python3 -c "
import sys; sys.path.insert(0, '.'); 
exec(open('ros2_test_ubuntu/ros2_drake_talker_node.py').read())
" 

Or use the convenience script:
    bash ros2_test_ubuntu/run_talker.sh

System:
    Pendulum: 1 DOF revolute joint, 1 kg bob, 0.5 m length
    State:    [theta (rad), theta_dot (rad/s)]
    Output:   Hello from Drake messages at --rate Hz for --duration seconds
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import sys
import os
import time
import numpy as np

# Drake imports (requires conda activate pydrake)
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Simulator,
    ScalarType,
)
from pydrake.systems.primitives import Integrator


# ============================================================================
# DRAKE LEAF SYSTEM: Simple pendulum dynamics
# ============================================================================
class SimplePendulumDynamics(LeafSystem):
    """
    Simple pendulum dynamics as a Drake LeafSystem.

    State:   [theta, theta_dot]  (continuous)
    Input:   none (unactuated)
    Output:  [theta, theta_dot]

    EOM: theta_ddot = -(g/L) * sin(theta) - b * theta_dot
    """

    def __init__(self, mass: float = 1.0, length: float = 0.5,
                 damping: float = 0.1, gravity: float = 9.81):
        super().__init__()
        self.mass    = mass
        self.length  = length
        self.damping = damping
        self.gravity = gravity

        # Continuous state: [theta, theta_dot]
        self.DeclareContinuousState(2)

        # Output: [theta, theta_dot]
        self.DeclareVectorOutputPort(
            "pendulum_state",
            BasicVector(2),
            self._output_state,
        )

    def _output_state(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)

    def DoCalcTimeDerivatives(self, context, derivatives):
        x         = context.get_continuous_state_vector().CopyToVector()
        theta     = x[0]
        theta_dot = x[1]

        # theta_ddot = -(g/L)*sin(theta) - b*theta_dot
        theta_ddot = (
            -(self.gravity / self.length) * np.sin(theta)
            - self.damping * theta_dot
        )

        derivatives.get_mutable_vector().SetFromVector(
            [theta_dot, theta_ddot]
        )


# ============================================================================
# DRAKE LEAF SYSTEM: Hello World message generator
# ============================================================================
class DrakeHelloPublisher(LeafSystem):
    """
    Drake LeafSystem that formats pendulum state as Hello World messages
    and prints them to stdout to be piped to the ROS 2 talker node.

    Input:  pendulum_state [theta, theta_dot]
    Output: prints to stdout at publish_rate_hz
    """

    def __init__(self, publish_rate_hz: float = 1.0):
        super().__init__()
        self._count = 0

        # Input: pendulum state from SimplePendulumDynamics
        self.DeclareVectorInputPort(
            "pendulum_state", BasicVector(2)
        )

        # Periodic publish event at specified rate
        self.DeclarePeriodicPublishEvent(
            period_sec = 1.0 / publish_rate_hz,
            offset_sec = 0.0,
            publish    = self._publish_hello,
        )

    def _publish_hello(self, context, event):
        """Format and print Drake state message → stdout → ROS 2."""
        state     = self.GetInputPort("pendulum_state").Eval(context)
        theta     = state[0]
        theta_dot = state[1]
        t         = context.get_time()
        self._count += 1

        message = (
            f"Hello from Drake [{self._count}] | "
            f"t={t:.2f}s | "
            f"theta={np.rad2deg(theta):.3f}deg | "
            f"theta_dot={theta_dot:.4f}rad/s"
        )

        # Flush immediately so the pipe receives it promptly
        print(message, flush=True)


# ============================================================================
# BUILD DRAKE DIAGRAM
# ============================================================================
def build_diagram(publish_rate_hz: float):
    """Build Drake diagram: SimplePendulum → HelloPublisher."""
    builder = DiagramBuilder()

    # Pendulum dynamics system
    pendulum = builder.AddSystem(SimplePendulumDynamics(
        mass=1.0, length=0.5, damping=0.1, gravity=9.81
    ))
    pendulum.set_name("simple_pendulum")

    # Hello message publisher
    hello_pub = builder.AddSystem(DrakeHelloPublisher(publish_rate_hz))
    hello_pub.set_name("hello_publisher")

    # Connect pendulum state → hello publisher input
    builder.Connect(
        pendulum.GetOutputPort("pendulum_state"),
        hello_pub.GetInputPort("pendulum_state"),
    )

    diagram = builder.Build()
    diagram.set_name("drake_hello_world")
    return diagram, pendulum


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Drake Hello World Talker — pipes state to ROS 2"
    )
    parser.add_argument(
        "--rate", type=float, default=1.0,
        help="Publish rate in Hz (default: 1.0)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Simulation duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--angle", type=float, default=30.0,
        help="Initial pendulum angle in degrees (default: 30.0)"
    )
    args = parser.parse_args()

    # Log to stderr so it doesn't pollute the stdout pipe
    print(f"# Drake Talker starting", file=sys.stderr)
    print(f"#   rate     = {args.rate} Hz", file=sys.stderr)
    print(f"#   duration = {args.duration} s", file=sys.stderr)
    print(f"#   angle    = {args.angle} deg", file=sys.stderr)
    print(f"# Messages publishing to stdout → ROS 2 /drake_hello", file=sys.stderr)

    # Build diagram
    diagram, pendulum = build_diagram(publish_rate_hz=args.rate)

    # Create simulator
    simulator = Simulator(diagram)
    context   = simulator.get_mutable_context()

    # Set initial pendulum state
    pendulum_context = pendulum.GetMyMutableContextFromRoot(context)
    pendulum_context.get_mutable_continuous_state_vector().SetFromVector(
        [np.deg2rad(args.angle), 0.0]   # [theta0, theta_dot0]
    )

    # Run at real-time rate so ROS 2 subscriber can keep up
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(args.duration)

    print(f"# Drake Talker finished after {args.duration}s.", file=sys.stderr)


if __name__ == '__main__':
    main()
