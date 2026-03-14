"""
Drake-ROS Publisher / Subscriber Round-Trip Test  (Python / macOS)
===================================================================
Uses ``drake_ros_compat`` which auto-selects the backend:
  • macOS / no compiled drake_ros  →  drake_ros_bridge  (pure Python + rclpy)
  • Linux / Docker with compiled drake_ros  →  real drake_ros C++

Run
---
    conda activate pydrake_ros2
    python ros2_test/drake_ros_pubsub_test_bridge.py
    python ros2_test/drake_ros_pubsub_test_bridge.py --duration 20.0
    python ros2_test/drake_ros_pubsub_test_bridge.py --prefix /robot --timestep 0.05

Monitor (separate terminal with ROS 2 sourced)
----------------------------------------------
    ros2 topic echo /drake_test/echo
    ros2 topic echo /drake_test/status
    ros2 topic pub  /drake_test/echo std_msgs/msg/String "data: 'hi'"

Diagram
-------
  StringSource ──▶ RosPublisherSystem[/drake_test/echo]
                             │
                     (ROS 2 loopback)
                             │
               RosSubscriberSystem[/drake_test/echo]
                             │
                       MemoryString   ← one-step delay (breaks algebraic loop)
                             │
                        SinkPrint     ← prints round-tripped messages
"""
import argparse

from drake_ros_compat import (
    ClockSystem,
    RosInterfaceSystem,
    RosPublisherSystem,
    RosSubscriberSystem,
    drake_ros_init,
    drake_ros_shutdown,
)

from pydrake.common.value import AbstractValue
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem, UnrestrictedUpdateEvent

from rclpy.qos import QoSProfile
from std_msgs.msg import String


# ─────────────────────────────────────────────────────────────────────────────
# Leaf systems
# ─────────────────────────────────────────────────────────────────────────────

class StringSource(LeafSystem):
    """Emits a std_msgs/String every Drake time step (counter + sim time)."""

    def __init__(self, label: str = ""):
        super().__init__()
        self._label = label
        self._counter = 0

        self.DeclareAbstractOutputPort(
            "msg",
            lambda: AbstractValue.Make(String()),
            self._calc_output,
        )
        self.DeclarePerStepEvent(UnrestrictedUpdateEvent(self._tick))

    def _tick(self, context, event, state):
        self._counter += 1

    def _calc_output(self, context, output):
        t = context.get_time()
        output.get_mutable_value().data = (
            f"{self._label}[#{self._counter:04d} t={t:.3f}s] Hello from Drake!"
        )


class MemoryString(LeafSystem):
    """One-step delay for String messages (breaks algebraic loop)."""

    def __init__(self):
        super().__init__()
        blank = String()

        self._in = self.DeclareAbstractInputPort("in", AbstractValue.Make(blank))
        self.DeclareAbstractState(AbstractValue.Make(blank))
        self.DeclareAbstractOutputPort(
            "out",
            lambda: AbstractValue.Make(String()),
            self._calc_output,
            {self.all_state_ticket()},
        )
        self.DeclarePerStepEvent(UnrestrictedUpdateEvent(self._latch))

    def _latch(self, context, event, state):
        val = self._in.Eval(context)
        s = String()
        s.data = val.data
        state.get_mutable_abstract_state().get_mutable_value(0).SetFrom(
            AbstractValue.Make(s)
        )

    def _calc_output(self, context, output):
        s = context.get_abstract_state().get_value(0).get_value()
        out = String()
        out.data = s.data
        output.get_mutable_value().data = out.data


class SinkPrint(LeafSystem):
    """Prints every received String message to stdout."""

    def __init__(self, label: str = "RX"):
        super().__init__()
        self._label = label
        self._count = 0

        self._in = self.DeclareAbstractInputPort("msg", AbstractValue.Make(String()))
        self.DeclarePerStepEvent(UnrestrictedUpdateEvent(self._print))

    def _print(self, context, event, state):
        msg = self._in.Eval(context)
        if msg.data:
            self._count += 1
            print(f"  [{self._label} #{self._count:04d}] {msg.data}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram
# ─────────────────────────────────────────────────────────────────────────────

def build_diagram(prefix: str, qos: QoSProfile):
    builder = DiagramBuilder()

    sys_ros = builder.AddSystem(RosInterfaceSystem("drake_pubsub_bridge_node"))
    ClockSystem.AddToBuilder(builder, sys_ros)

    topic_echo   = f"{prefix}/echo"
    topic_status = f"{prefix}/status"

    sys_pub_echo   = builder.AddSystem(RosPublisherSystem.Make(String, topic_echo,   qos, sys_ros))
    sys_pub_status = builder.AddSystem(RosPublisherSystem.Make(String, topic_status, qos, sys_ros))
    sys_sub_echo   = builder.AddSystem(RosSubscriberSystem.Make(String, topic_echo,  qos, sys_ros))

    sys_source = builder.AddSystem(StringSource(label=f"[{topic_echo}]"))
    sys_memory = builder.AddSystem(MemoryString())
    sys_sink   = builder.AddSystem(SinkPrint(label="ECHO-RX"))

    builder.Connect(sys_source.get_output_port(0), sys_pub_echo.get_input_port(0))
    builder.Connect(sys_source.get_output_port(0), sys_pub_status.get_input_port(0))
    builder.Connect(sys_sub_echo.get_output_port(0), sys_memory.get_input_port(0))
    builder.Connect(sys_memory.get_output_port(0),   sys_sink.get_input_port(0))

    return builder.Build()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drake-ROS pub/sub round-trip test (Python bridge)"
    )
    parser.add_argument("--duration", type=float, default=float("inf"),
                        help="Sim duration in seconds (default: ∞)")
    parser.add_argument("--prefix",   type=str,   default="/drake_test",
                        help="ROS topic prefix (default: /drake_test)")
    parser.add_argument("--timestep", type=float, default=0.1,
                        help="Sim timestep in seconds (default: 0.1)")
    parser.add_argument("--depth",    type=int,   default=10,
                        help="QoS queue depth (default: 10)")
    args = parser.parse_args()

    prefix = args.prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix

    print("=" * 60)
    print("  Drake-ROS Pub/Sub Test  (Python bridge)")
    print("=" * 60)
    print(f"  Echo topic  : {prefix}/echo")
    print(f"  Status topic: {prefix}/status")
    print(f"  Timestep    : {args.timestep} s")
    dur_str = "∞" if args.duration == float("inf") else f"{args.duration} s"
    print(f"  Duration    : {dur_str}")
    print("-" * 60)
    print("  Monitor (separate terminal):")
    print(f"    ros2 topic echo {prefix}/echo")
    print(f"    ros2 topic echo {prefix}/status")
    print(f"    ros2 topic pub  {prefix}/echo std_msgs/msg/String \"data: 'hi'\"")
    print("=" * 60)

    drake_ros_init()
    try:
        qos     = QoSProfile(depth=args.depth)
        diagram = build_diagram(prefix, qos)

        sim = Simulator(diagram)
        sim.set_target_realtime_rate(1.0)
        sim.Initialize()

        t     = 0.0
        step  = args.timestep
        max_t = args.duration

        print("\n  Running… (Ctrl-C to stop)\n")
        while t < max_t:
            t_next = min(t + step, max_t)
            sim.AdvanceTo(t_next)
            t = t_next

    except KeyboardInterrupt:
        print("\n\n  [Interrupted]")
    finally:
        drake_ros_shutdown()
        print("  Shutdown complete.")


if __name__ == "__main__":
    main()
