"""
Drake-ROS Publisher / Subscriber Round-Trip Test
=================================================
Uses the *real* `drake_ros.core` C++ extension (NOT the pure-Python bridge).
Designed to run inside the drake-ros Docker devcontainer where `drake_ros` is
built with Bazel and the ROS 2 environment is sourced.

Diagram
-------
                     ┌─────────────────────────────────────────┐
                    │           Drake DiagramBuilder           │
                    │                                          │
  ┌──────────────┐  │  ┌─────────┐    ┌──────────────────┐   │
  │ StringSource │──┼─▶│  Pub    │─▶  /drake_test/echo   │   │
  │  (LeafSys)   │  │  │ (drake) │     (ROS 2 topic)      │   │
  └──────────────┘  │  └─────────┘          │             │   │
                    │                        ▼             │   │
                    │  ┌─────────┐    ┌──────────────────┐│   │
  ┌──────────────┐  │  │  Sub    │◀─  /drake_test/echo   ││   │
  │   SinkPrint  │◀─┼──│ (drake) │     (same ROS topic)  ││   │
  │  (LeafSys)   │  │  └─────────┘    └──────────────────┘│   │
  └──────────────┘  │       ▲ Memory (breaks algebraic loop)  │
                    └─────────────────────────────────────────┘

Topics
------
  /drake_test/echo    String   (published by Drake, echoed back via ROS)
  /drake_test/status  String   (one-way publish for monitoring with ros2 topic echo)

Usage (inside Docker container)
--------------------------------
  # Quick run and stop at N seconds:
  python drake_ros_pubsub_test.py --duration 10.0

  # Run forever (Ctrl-C to quit):
  python drake_ros_pubsub_test.py

  # Override topic prefix:
  python drake_ros_pubsub_test.py --prefix /my_robot --duration 5.0

External monitoring (separate terminal inside container)
---------------------------------------------------------
  ros2 topic echo /drake_test/echo
  ros2 topic echo /drake_test/status
  ros2 topic pub  /drake_test/echo std_msgs/msg/String "data: 'hello from ROS'"
"""
import argparse
import sys

# ── real drake_ros (requires a Bazel-built / pip-installed drake_ros package) ──
try:
    import drake_ros.core
    from drake_ros.core import ClockSystem
    from drake_ros.core import RosInterfaceSystem
    from drake_ros.core import RosPublisherSystem
    from drake_ros.core import RosSubscriberSystem
except ImportError:
    print(
        "\n[ERROR] Could not import drake_ros.core.\n"
        "This script requires the real drake_ros C++ package.\n"
        "Run it inside the drake-ros Docker container after building with Bazel.\n"
        "See: drake-ros/.devcontainer/README.md for setup instructions.\n",
        file=sys.stderr,
    )
    sys.exit(1)

from pydrake.common.value import AbstractValue
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem, UnrestrictedUpdateEvent

from rclpy.qos import QoSProfile
from std_msgs.msg import String


# ─────────────────────────────────────────────────────────────────────────────
# Helper leaf systems
# ─────────────────────────────────────────────────────────────────────────────

class StringSource(LeafSystem):
    """Publishes a String message every Drake time step.

    The message payload is a human-readable counter + elapsed-time string so
    that the subscriber can confirm the round-trip without needing external
    tools.
    """

    def __init__(self, prefix: str = ""):
        super().__init__()
        self._prefix = prefix
        self._counter = 0

        self.DeclareAbstractOutputPort(
            "msg",
            lambda: AbstractValue.Make(String()),
            self._calc_output,
        )

        # Advance counter once per step (unrestricted update)
        self.DeclarePerStepEvent(
            UnrestrictedUpdateEvent(self._increment_counter)
        )

    def _increment_counter(self, context, event, state):
        self._counter += 1

    def _calc_output(self, context, output):
        t = context.get_time()
        msg = String()
        msg.data = (
            f"{self._prefix}[#{self._counter:04d} | t={t:.3f}s] "
            "Hello from Drake-ROS!"
        )
        output.get_mutable_value().data = msg.data


class MemoryString(LeafSystem):
    """One-step delay for String messages.

    Needed to break the algebraic loop that would occur when the publisher
    and subscriber share the same ROS topic inside the same diagram.
    """

    def __init__(self):
        super().__init__()
        initial = String()

        self._input_port = self.DeclareAbstractInputPort(
            "in", AbstractValue.Make(initial)
        )
        self.DeclareAbstractState(AbstractValue.Make(initial))
        self.DeclareAbstractOutputPort(
            "out",
            lambda: AbstractValue.Make(String()),
            self._calc_output,
            {self.all_state_ticket()},
        )
        self.DeclarePerStepEvent(
            UnrestrictedUpdateEvent(self._latch)
        )

    def _latch(self, context, event, state):
        val = self._input_port.Eval(context)
        stored = String()
        stored.data = val.data
        state.get_mutable_abstract_state().get_mutable_value(0).SetFrom(
            AbstractValue.Make(stored)
        )

    def _calc_output(self, context, output):
        stored = context.get_abstract_state().get_value(0).get_value()
        out_msg = String()
        out_msg.data = stored.data
        output.get_mutable_value().data = out_msg.data


class SinkPrint(LeafSystem):
    """Receives a String message and prints it to stdout along with a counter."""

    def __init__(self, label: str = "RX"):
        super().__init__()
        self._label = label
        self._rx_count = 0

        self._input_port = self.DeclareAbstractInputPort(
            "msg", AbstractValue.Make(String())
        )
        # Force evaluation every step via a per-step publish event
        self.DeclarePerStepEvent(
            UnrestrictedUpdateEvent(self._print_msg)
        )

    def _print_msg(self, context, event, state):
        msg = self._input_port.Eval(context)
        if msg.data:  # skip empty initial messages
            self._rx_count += 1
            print(f"  [{self._label} #{self._rx_count:04d}] {msg.data}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_diagram(prefix: str, qos: QoSProfile):
    """Build and return the Drake diagram for the pub/sub test."""
    builder = DiagramBuilder()

    # ── ROS interface (manages the rclcpp node inside Drake) ──────────────────
    sys_ros = builder.AddSystem(RosInterfaceSystem("drake_pubsub_test_node"))
    ros_if = sys_ros.get_ros_interface()

    # Publish the ROS /clock topic so ROS tools see Drake sim time
    ClockSystem.AddToBuilder(builder, ros_if)

    topic_echo = f"{prefix}/echo"
    topic_status = f"{prefix}/status"

    # ── Publisher / Subscriber systems ────────────────────────────────────────
    sys_pub_echo = builder.AddSystem(
        RosPublisherSystem.Make(String, topic_echo, qos, ros_if)
    )
    sys_sub_echo = builder.AddSystem(
        RosSubscriberSystem.Make(String, topic_echo, qos, ros_if)
    )
    # Status topic is publish-only (external tools can monitor)
    sys_pub_status = builder.AddSystem(
        RosPublisherSystem.Make(String, topic_status, qos, ros_if)
    )

    # ── Custom leaf systems ───────────────────────────────────────────────────
    sys_source = builder.AddSystem(StringSource(prefix=f"[{topic_echo}]"))
    sys_memory = builder.AddSystem(MemoryString())       # algebraic-loop breaker
    sys_sink = builder.AddSystem(SinkPrint(label="ECHO-RX"))

    # ── Wiring ────────────────────────────────────────────────────────────────
    #  StringSource → pub(echo)               (Drake → ROS)
    builder.Connect(sys_source.get_output_port(0), sys_pub_echo.get_input_port(0))
    #  StringSource → pub(status)             (status feed for external observers)
    builder.Connect(sys_source.get_output_port(0), sys_pub_status.get_input_port(0))
    #  sub(echo) → Memory → SinkPrint         (ROS → Drake, delayed by 1 step)
    builder.Connect(sys_sub_echo.get_output_port(0), sys_memory.get_input_port(0))
    builder.Connect(sys_memory.get_output_port(0), sys_sink.get_input_port(0))

    return builder.Build()


def main():
    parser = argparse.ArgumentParser(
        description="Drake-ROS publisher/subscriber round-trip test"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float("inf"),
        help="Simulation duration in seconds (default: run forever)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="/drake_test",
        help="ROS topic prefix (default: /drake_test)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
        help="Simulation timestep in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="QoS queue depth (default: 10)",
    )
    args = parser.parse_args()

    # Normalise prefix (strip trailing slash, ensure leading slash)
    prefix = args.prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix

    print("=" * 60)
    print("  Drake-ROS Publisher/Subscriber Round-Trip Test")
    print("=" * 60)
    print(f"  Echo topic  : {prefix}/echo")
    print(f"  Status topic: {prefix}/status")
    print(f"  Timestep    : {args.timestep} s")
    print(f"  Duration    : {'∞' if args.duration == float('inf') else f'{args.duration} s'}")
    print("-" * 60)
    print("  External monitoring:")
    print(f"    ros2 topic echo {prefix}/echo")
    print(f"    ros2 topic echo {prefix}/status")
    print(f"    ros2 topic pub  {prefix}/echo std_msgs/msg/String \"data: 'hi'\"")
    print("=" * 60)

    # Initialise drake_ros (creates the rclcpp context)
    drake_ros.core.init()

    try:
        qos = QoSProfile(depth=args.depth)
        diagram = build_diagram(prefix, qos)

        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()

        context = simulator.get_mutable_context()

        # Run in fixed timestep increments so we can catch KeyboardInterrupt
        t = 0.0
        step = args.timestep
        max_t = args.duration

        print("\n  Running… (Ctrl-C to stop)\n")
        while t < max_t:
            t_next = min(t + step, max_t)
            simulator.AdvanceTo(t_next)
            t = t_next

    except KeyboardInterrupt:
        print("\n\n  [Interrupted by user]")
    finally:
        drake_ros.core.shutdown()
        print("  drake_ros shutdown complete.")


if __name__ == "__main__":
    main()
