"""
drake_ros_bridge.py
====================

Pure-Python implementation of the core **drake-ros** bridge API
(https://github.com/RobotLocomotion/drake-ros).

Provides Drake ``LeafSystem`` blocks that act as the ROS 2 / Drake boundary,
mirroring the C++ ``drake_ros`` package so that the same *diagram-builder*
pattern can be used on platforms (e.g. macOS) where the C++ bindings are not
yet available.

Public API
----------
RosInterfaceSystem
    Creates and owns a ``rclpy`` node; spins it in a background daemon thread.
    Call ``get_ros_interface()`` to obtain the accessor object passed to
    publisher/subscriber factories.

RosPublisherSystem.Make(msg_type, topic, qos, ros_interface) → LeafSystem
    Publishes the value on its abstract input port to a ROS 2 topic every
    Drake simulation step.

RosSubscriberSystem.Make(msg_type, topic, qos, ros_interface) → LeafSystem
    Outputs the most-recently received ROS 2 message on an abstract output
    port; updated every simulation step from a thread-safe internal buffer.

ClockSystem.AddToBuilder(builder, ros_interface, period_sec) → LeafSystem
    Adds a periodic publisher that sends the Drake simulation time on
    ``/clock`` (``rosgraph_msgs/Clock``).  Enables ``use_sim_time`` in tools
    like ``ros2 topic echo`` and RViz.

JointStateBroadcasterSystem(plant, model_instance, ros_interface, ...)
    Periodically publishes ``sensor_msgs/JointState`` for the joints of a
    given model instance.

SceneTfBroadcasterSystem(plant, ros_interface, ...)
    Periodically publishes ``/tf`` for every body of the plant.

Usage (see ros2_drakeROS_flip_flop.py for a complete working example):

    from drake_ros_bridge import RosInterfaceSystem, RosPublisherSystem, RosSubscriberSystem
    builder = DiagramBuilder()
    ros_if = builder.AddSystem(RosInterfaceSystem("my_node"))
    pub     = builder.AddSystem(RosPublisherSystem.Make(std_msgs.msg.String, "/out", 10, ros_if))
    sub     = builder.AddSystem(RosSubscriberSystem.Make(std_msgs.msg.String, "/in",  10, ros_if))
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
import rclpy.node
import tf2_ros
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import (
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    UnrestrictedUpdateEvent,
)

__all__ = [
    "RosInterfaceSystem",
    "RosPublisherSystem",
    "RosSubscriberSystem",
    "ClockSystem",
    "JointStateBroadcasterSystem",
    "SceneTfBroadcasterSystem",
]


# ─────────────────────────────────────────────────────────────────────────────
def _drake_time_to_ros(t: float) -> RosTime:
    """Convert a Drake simulation time (float seconds) to builtin_interfaces/Time."""
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    msg = RosTime()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


# ─────────────────────────────────────────────────────────────────────────────
class RosInterfaceSystem(LeafSystem):
    """Drake ``LeafSystem`` that owns an ``rclpy`` node.

    Creates the node and immediately starts a daemon thread that spins
    rclpy so that subscribers are serviced while the Drake simulator runs.

    Parameters
    ----------
    node_name:
        Name of the ``rclpy`` node to create.  Each simulator in a process
        should use a unique name.
    """

    def __init__(self, node_name: str) -> None:
        LeafSystem.__init__(self)
        if not rclpy.ok():
            rclpy.init()
        self._node: rclpy.node.Node = rclpy.create_node(node_name)
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

    # Public helpers ----------------------------------------------------------
    def get_ros_interface(self) -> "RosInterfaceSystem":
        """Return *self* — the accessor object expected by publisher/subscriber factories."""
        return self

    def get_node(self) -> rclpy.node.Node:
        """Return the underlying ``rclpy`` node."""
        return self._node

    def __del__(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
class RosPublisherSystem(LeafSystem):
    """Drake ``LeafSystem`` that publishes a ROS 2 message each simulation step.

    Do **not** instantiate directly — use the factory method:

        ``RosPublisherSystem.Make(msg_type, topic, qos, ros_interface)``

    The system has one abstract input port named ``"message"`` that accepts
    an instance of ``msg_type``.  The value on that port is published to the
    ROS 2 topic every time Drake calls a publish event (i.e. every step).

    Parameters
    ----------
    msg_type:
        The ROS 2 message class (e.g. ``std_msgs.msg.Bool``).
    topic:
        ROS 2 topic name (e.g. ``"/Q"``).
    qos:
        ``rclpy`` QoS profile or integer queue depth.
    ros_interface:
        The ``RosInterfaceSystem`` instance owning the rclpy node.
    """

    @staticmethod
    def Make(
        msg_type,
        topic: str,
        qos,
        ros_interface: RosInterfaceSystem,
    ) -> "RosPublisherSystem":
        """Factory matching the ``drake_ros`` C++ API."""
        return RosPublisherSystem(msg_type, topic, qos, ros_interface)

    def __init__(
        self,
        msg_type,
        topic: str,
        qos,
        ros_interface: RosInterfaceSystem,
    ) -> None:
        LeafSystem.__init__(self)
        self._pub = ros_interface.get_node().create_publisher(msg_type, topic, qos)
        # One abstract input port that holds messages of msg_type
        self._input_port = self.DeclareAbstractInputPort(
            "message", AbstractValue.Make(msg_type())
        )
        self.DeclarePerStepPublishEvent(self._do_publish)

    def _do_publish(self, context) -> EventStatus:
        self._pub.publish(self._input_port.Eval(context))
        return EventStatus.Succeeded()


# ─────────────────────────────────────────────────────────────────────────────
class RosSubscriberSystem(LeafSystem):
    """Drake ``LeafSystem`` that outputs the latest ROS 2 message received.

    Do **not** instantiate directly — use the factory method:

        ``RosSubscriberSystem.Make(msg_type, topic, qos, ros_interface)``

    The system has one abstract output port named ``"message"`` that emits the
    most-recently received message of ``msg_type``.  Before any message arrives
    the default-constructed ``msg_type()`` is returned.

    Thread safety
    ~~~~~~~~~~~~~
    The ROS 2 subscription callback (rclpy spin thread) writes to a
    ``threading.Lock``-protected buffer.  A Drake per-step unrestricted update
    event copies that buffer into the abstract state, which is then read by the
    output-port callback — all without holding the lock during Drake evaluation.

    Parameters
    ----------
    msg_type, topic, qos, ros_interface:
        Same as ``RosPublisherSystem``.
    """

    @staticmethod
    def Make(
        msg_type,
        topic: str,
        qos,
        ros_interface: RosInterfaceSystem,
    ) -> "RosSubscriberSystem":
        """Factory matching the ``drake_ros`` C++ API."""
        return RosSubscriberSystem(msg_type, topic, qos, ros_interface)

    def __init__(
        self,
        msg_type,
        topic: str,
        qos,
        ros_interface: RosInterfaceSystem,
    ) -> None:
        LeafSystem.__init__(self)

        # Thread-safe incoming message buffer
        self._lock = threading.Lock()
        self._latest = msg_type()

        # Subscribe via the shared rclpy node
        ros_interface.get_node().create_subscription(
            msg_type, topic, self._ros_callback, qos
        )

        # Abstract state index 0 holds the committed (Drake-side) message
        self.DeclareAbstractState(AbstractValue.Make(msg_type()))

        # Output port reads from abstract state
        self.DeclareAbstractOutputPort(
            "message",
            lambda: AbstractValue.Make(msg_type()),
            self._calc_output,
            {self.all_state_ticket()},
        )

        # Per-step unrestricted update: copy buffer → state
        self.DeclarePerStepEvent(
            UnrestrictedUpdateEvent(self._transfer_to_state)
        )

    # ROS callback (runs in rclpy spin thread) --------------------------------
    def _ros_callback(self, msg) -> None:
        with self._lock:
            self._latest = msg

    # Drake callbacks ---------------------------------------------------------
    def _transfer_to_state(self, context, event, state) -> EventStatus:
        with self._lock:
            msg = self._latest
        state.get_mutable_abstract_state().get_mutable_value(0).SetFrom(
            AbstractValue.Make(msg)
        )
        return EventStatus.Succeeded()

    def _calc_output(self, context, output) -> None:
        output.SetFrom(context.get_abstract_state().get_value(0))


# ─────────────────────────────────────────────────────────────────────────────
class _ClockPublisher(LeafSystem):
    """Internal: periodic ``/clock`` publisher."""

    def __init__(self, ros_interface: RosInterfaceSystem, period_sec: float) -> None:
        LeafSystem.__init__(self)
        self._pub = ros_interface.get_node().create_publisher(Clock, "/clock", 10)
        self.DeclarePeriodicPublishEvent(period_sec=period_sec, offset_sec=0.0,
                                        publish=self._publish_clock)

    def _publish_clock(self, context) -> EventStatus:
        msg = Clock()
        msg.clock = _drake_time_to_ros(context.get_time())
        self._pub.publish(msg)
        return EventStatus.Succeeded()


class ClockSystem:
    """Helper that mirrors ``drake_ros::ClockSystem::AddToBuilder``."""

    @staticmethod
    def AddToBuilder(
        builder: DiagramBuilder,
        ros_interface: RosInterfaceSystem,
        period_sec: float = 1.0 / 32.0,
    ) -> "_ClockPublisher":
        """Add a ``/clock`` publisher to *builder* and return the system."""
        sys = builder.AddSystem(_ClockPublisher(ros_interface, period_sec))
        return sys


# ─────────────────────────────────────────────────────────────────────────────
class JointStateBroadcasterSystem(LeafSystem):
    """Publish ``sensor_msgs/JointState`` for a Drake ``MultibodyPlant`` model.

    Connects to the plant's continuous-state vector input and publishes
    joint positions and velocities on ``/joint_states`` at the requested rate.

    Parameters
    ----------
    plant:
        Finalised ``MultibodyPlant``.
    model_instance:
        The model instance whose joints to broadcast.
    ros_interface:
        Node owner.
    joint_names:
        Ordered list of joint names (user-facing order).  If ``None``, all
        actuated revolute joints are broadcast in Drake's internal order.
    topic:
        ROS 2 topic name (default ``"/joint_states"``).
    publish_period_sec:
        How often to publish (seconds).
    """

    def __init__(
        self,
        plant,
        model_instance,
        ros_interface: RosInterfaceSystem,
        joint_names: Optional[list] = None,
        topic: str = "/joint_states",
        publish_period_sec: float = 1.0 / 50.0,
    ) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._mi = model_instance
        self._pub = ros_interface.get_node().create_publisher(JointState, topic, 10)

        # Determine joint names and velocity indices
        if joint_names is not None:
            self._joint_names = joint_names
            self._q_idx = []
            self._v_idx = []
            for jn in joint_names:
                j = plant.GetJointByName(jn, model_instance)
                self._q_idx.append(j.position_start())
                self._v_idx.append(j.velocity_start())
        else:
            self._joint_names = []
            self._q_idx = []
            self._v_idx = []
            for idx in plant.GetJointIndices(model_instance):
                j = plant.get_joint(idx)
                if j.num_positions() == 1:
                    self._joint_names.append(j.name())
                    self._q_idx.append(j.position_start())
                    self._v_idx.append(j.velocity_start())

        # State input from MultibodyPlant
        self._state_input = self.DeclareVectorInputPort(
            "state", plant.num_multibody_states(model_instance)
        )
        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period_sec,
            offset_sec=0.0,
            publish=self._publish_joint_state,
        )

    def _publish_joint_state(self, context) -> EventStatus:
        state = self._state_input.Eval(context)
        nq = self._plant.num_positions(self._mi)
        q_all = state[:nq]
        v_all = state[nq:]

        msg = JointState()
        msg.header.stamp = _drake_time_to_ros(context.get_time())
        msg.name = self._joint_names
        msg.position = [float(q_all[i]) for i in self._q_idx]
        msg.velocity = [float(v_all[i]) for i in self._v_idx]
        msg.effort = [0.0] * len(self._joint_names)
        self._pub.publish(msg)
        return EventStatus.Succeeded()


# ─────────────────────────────────────────────────────────────────────────────
class SceneTfBroadcasterSystem(LeafSystem):
    """Broadcast ``/tf`` for every body of a Drake ``MultibodyPlant``.

    Mirrors ``drake_ros::tf2::SceneTfBroadcasterSystem``.  Uses
    ``plant.EvalBodyPoseInWorld()`` to retrieve each body's world-frame pose
    and publishes it as a ``geometry_msgs/TransformStamped`` via
    ``tf2_ros.TransformBroadcaster``.

    The plant's *context* is passed in through the scalar input port
    ``"plant_context_dummy"`` — Drake requires a connection to participate in
    the simulation loop.  The actual plant context is obtained via
    ``plant.GetMyContextFromRoot(root_context)`` in the publish callback.

    Parameters
    ----------
    plant:
        Finalised ``MultibodyPlant``.
    ros_interface:
        Node owner.
    world_frame_id:
        TF frame name for the world (default ``"world"``).
    publish_period_sec:
        Broadcast rate.
    """

    def __init__(
        self,
        plant,
        ros_interface: RosInterfaceSystem,
        world_frame_id: str = "world",
        publish_period_sec: float = 1.0 / 32.0,
    ) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._world_id = world_frame_id
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(ros_interface.get_node())

        # Collect all non-world bodies
        world_body = plant.world_body()
        self._bodies = [
            plant.get_body(idx)
            for idx in plant.GetBodyIndices()
            if plant.get_body(idx).index() != world_body.index()
        ]

        # Dummy vector input so the system participates in the diagram ordering
        self._dummy = self.DeclareVectorInputPort("dummy_in", 1)

        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period_sec,
            offset_sec=0.0,
            publish=self._broadcast_tf,
        )

    def _broadcast_tf(self, context) -> EventStatus:
        t_ros = _drake_time_to_ros(context.get_time())
        transforms = []
        plant_ctx = self._plant.GetMyContextFromRoot(context)
        for body in self._bodies:
            pose: RigidTransform = self._plant.EvalBodyPoseInWorld(plant_ctx, body)
            ts = TransformStamped()
            ts.header.stamp = t_ros
            ts.header.frame_id = self._world_id
            ts.child_frame_id = body.name()
            p = pose.translation()
            ts.transform.translation.x = float(p[0])
            ts.transform.translation.y = float(p[1])
            ts.transform.translation.z = float(p[2])
            q = pose.rotation().ToQuaternion()
            ts.transform.rotation.w = float(q.w())
            ts.transform.rotation.x = float(q.x())
            ts.transform.rotation.y = float(q.y())
            ts.transform.rotation.z = float(q.z())
            transforms.append(ts)
        self._tf_broadcaster.sendTransform(transforms)
        return EventStatus.Succeeded()
