"""
ros2_drakeROS_plant_node.py
============================

Drake-ROS-style **cable-manipulator plant** — all ROS 2 I/O is wired as
``LeafSystem`` blocks inside a **single Drake Diagram** that is advanced by a
``Simulator`` at real-time rate.

This re-implements ``ros2_drake_plant_node.py`` with the same topology used by
the official ``drake_ros_examples`` (multirobot, rs_flip_flop):

   DiagramBuilder
   ├── RosInterfaceSystem          (rclpy node + spin-thread)
   ├── ClockSystem                 (/clock publisher)
   ├── MultibodyPlant + SceneGraph (cable-manipulator physics)
   ├── MeshcatVisualizer           (optional)
   ├── RosSubscriberSystem         (/torque_command or /joint_position_command)
   ├── TorqueDispatcher / PosDispatcher  (bridge sub → plant)
   ├── JointStateBroadcaster       (/joint_states publisher)
   └── EEPositionBroadcaster       (/ee_position publisher)

Modes
-----
  dynamics     Discrete-time plant, receives torque commands, publishes state.
  scene-viz    Continuous-time plant, receives joint positions, publishes state.

ROS 2 Topics
-------------
  (dynamics)
      IN:  /torque_command         (Float64MultiArray)  — [τ1, τ2]
      OUT: /joint_states           (JointState)
      OUT: /ee_position            (Point)
      OUT: /clock                  (Clock)

  (scene-viz)
      IN:  /joint_position_command (Float64MultiArray)  — [q1, q2]
      OUT: /joint_states           (JointState)
      OUT: /ee_position            (Point)
      OUT: /clock                  (Clock)

Usage
-----
    conda activate pydrake_ros2

    # Dynamics mode (physics simulation)
    python ros2_test/ros2_drakeROS_plant_node.py --mode dynamics --timestep 0.002

    # Scene-viz mode (position-driven Meshcat)
    python ros2_test/ros2_drakeROS_plant_node.py --mode scene-viz

    # Pair with the controller:
    # Terminal 1:
    python ros2_test/ros2_drakeROS_plant_node.py --mode dynamics --timestep 0.002
    # Terminal 2:
    python ros2_test/ros2_drakeROS_controller_node.py --mode min-jerk --q-goal 60,-120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Project path ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "ros2_test"))

from drake_ros_compat import (
    BACKEND,
    ClockSystem,
    RosInterfaceSystem,
    RosPublisherSystem,
    RosSubscriberSystem,
    drake_ros_init,
    drake_ros_shutdown,
)

from pydrake.all import (
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)
from pydrake.common.value import AbstractValue
from pydrake.multibody.tree import RevoluteSpring
from pydrake.systems.framework import EventStatus, LeafSystem
from pydrake.systems.primitives import ConstantVectorSource

from rclpy.qos import QoSProfile
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robots.cup_manipulator_tendon import (
    CupManipulatorTendon,
    create_cable_manipulator_config,
)
from project_utils.viz_cables import draw_cables

# ── Constants ────────────────────────────────────────────────────────────────
URDF_PATH = str(
    WORKSPACE
    / "model_using_onshape_to_robot"
    / "manipulator_cable"
    / "manipulator_cable_obj.urdf"
)
JOINT_NAMES = ["link1_base", "link2_link1"]  # user order
NUM_JOINTS = len(JOINT_NAMES)

_M_PATCH = SpatialInertia(
    mass=0.3,
    p_PScm_E=np.zeros(3),
    G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)


# ═════════════════════════════════════════════════════════════════════════════
# Custom LeafSystems that live inside the Drake diagram
# ═════════════════════════════════════════════════════════════════════════════

class TorqueDispatcher(LeafSystem):
    """Reads ``Float64MultiArray`` from a ROS subscriber output and converts
    it to a vector output compatible with the plant's actuation input port.

    The subscriber provides an abstract port (message); this system extracts
    ``msg.data`` and outputs a numeric vector of size ``nu``.
    """

    def __init__(self, nu: int) -> None:
        LeafSystem.__init__(self)
        self._nu = nu
        self._msg_input = self.DeclareAbstractInputPort(
            "torque_msg", AbstractValue.Make(Float64MultiArray())
        )
        self.DeclareVectorOutputPort("torque", nu, self._calc_output)

    def _calc_output(self, context, output) -> None:
        msg = self._msg_input.Eval(context)
        if len(msg.data) == self._nu:
            output.SetFromVector(np.array(msg.data, dtype=float))
        else:
            output.SetFromVector(np.zeros(self._nu))


class PositionDispatcher(LeafSystem):
    """Reads ``Float64MultiArray`` from a ROS subscriber and outputs
    a vector of joint positions + zero velocities suitable for direct
    position-setting in scene-viz mode.

    Outputs a (2*nq)-vector: [q1, q2, 0, 0] — positions then velocities.
    """

    def __init__(self, nq: int) -> None:
        LeafSystem.__init__(self)
        self._nq = nq
        self._msg_input = self.DeclareAbstractInputPort(
            "pos_msg", AbstractValue.Make(Float64MultiArray())
        )
        self.DeclareVectorOutputPort("state_cmd", 2 * nq, self._calc_output)

    def _calc_output(self, context, output) -> None:
        msg = self._msg_input.Eval(context)
        q = np.array(msg.data, dtype=float) if len(msg.data) == self._nq else np.zeros(self._nq)
        output.SetFromVector(np.concatenate([q, np.zeros(self._nq)]))


class JointStateBroadcaster(LeafSystem):
    """Reads the plant state and publishes ``sensor_msgs/JointState``
    on the abstract output port connected to a ``RosPublisherSystem``.
    """

    def __init__(self, plant, model_instance, manipulator) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._mi = model_instance
        self._manip = manipulator
        nstates = plant.num_multibody_states()
        self._state_input = self.DeclareVectorInputPort("state", nstates)
        self.DeclareAbstractOutputPort(
            "joint_state_msg",
            lambda: AbstractValue.Make(JointState()),
            self._calc_output,
        )

    def _calc_output(self, context, output) -> None:
        state = self._state_input.Eval(context)
        nq = self._plant.num_positions()
        q_all = state[:nq]
        v_all = state[nq:]

        # Map Drake-order to user-order
        q_user = np.zeros(NUM_JOINTS)
        v_user = np.zeros(NUM_JOINTS)
        for i, jn in enumerate(JOINT_NAMES):
            jt = self._plant.GetJointByName(jn, self._mi)
            q_user[i] = q_all[jt.position_start()]
            v_user[i] = v_all[jt.velocity_start()]

        msg = output.get_mutable_value()
        msg.name = list(JOINT_NAMES)
        msg.position = q_user.tolist()
        msg.velocity = v_user.tolist()
        msg.effort = [0.0] * NUM_JOINTS


class EEPositionBroadcaster(LeafSystem):
    """Reads plant state, computes FK to find the EE position, and outputs
    a ``geometry_msgs/Point`` on the abstract output port.
    """

    def __init__(self, plant, model_instance, manipulator) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._mi = model_instance
        self._manip = manipulator
        self._nq = plant.num_positions()
        # Private context for FK evaluation (plant is already finalized)
        self._fk_ctx = plant.CreateDefaultContext()
        nstates = plant.num_multibody_states()
        self._state_input = self.DeclareVectorInputPort("state", nstates)
        self.DeclareAbstractOutputPort(
            "ee_point_msg",
            lambda: AbstractValue.Make(Point()),
            self._calc_output,
        )

    def _calc_output(self, context, output) -> None:
        state = self._state_input.Eval(context)
        self._plant.SetPositions(self._fk_ctx, state[:self._nq])
        self._plant.SetVelocities(self._fk_ctx, state[self._nq:])
        ee = self._manip.get_end_effector_position(self._plant, self._fk_ctx)
        msg = output.get_mutable_value()
        msg.x = float(ee[0])
        msg.y = float(ee[1])
        msg.z = float(ee[2])


class CableVisPublisher(LeafSystem):
    """Periodically updates cable visualization in Meshcat."""

    def __init__(self, plant, manipulator, rig, meshcat_handle, period_sec: float = 1.0 / 30.0) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._manip = manipulator
        self._rig = rig
        self._meshcat = meshcat_handle
        self._nq = plant.num_positions()
        # Private context for cable FK evaluation
        self._cable_ctx = plant.CreateDefaultContext()
        self._state_input = self.DeclareVectorInputPort(
            "state", plant.num_multibody_states()
        )
        self.DeclarePeriodicPublishEvent(
            period_sec=period_sec, offset_sec=0.0, publish=self._update_cables,
        )

    def _update_cables(self, context) -> EventStatus:
        try:
            state = self._state_input.Eval(context)
            self._plant.SetPositions(self._cable_ctx, state[:self._nq])
            self._plant.SetVelocities(self._cable_ctx, state[self._nq:])
            self._manip.compute_tangents(self._plant, self._cable_ctx)
            draw_cables(
                self._meshcat, self._plant, self._cable_ctx,
                self._manip, self._rig,
            )
        except Exception:
            pass
        return EventStatus.Succeeded()


# ═════════════════════════════════════════════════════════════════════════════
# Main builder
# ═════════════════════════════════════════════════════════════════════════════

def build_dynamics_diagram(
    sim_timestep: float,
    enable_meshcat: bool,
    joint_damping: tuple,
    joint_stiffness: tuple,
):
    """Build diagram for dynamics mode: physics simulation with torque input."""
    config = create_cable_manipulator_config(
        urdf_path=URDF_PATH,
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=joint_damping,
        stiffness=joint_stiffness,
    )

    builder = DiagramBuilder()
    qos = QoSProfile(depth=10)

    # ── ROS interface ────────────────────────────────────────────────────
    ros_if = builder.AddSystem(RosInterfaceSystem("drake_plant_node"))
    ClockSystem.AddToBuilder(builder, ros_if)

    # ── Physics plant ────────────────────────────────────────────────────
    plant = MultibodyPlant(time_step=sim_timestep)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulatorTendon(config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(plant, position=np.zeros(3))
    manipulator.add_joint_actuators(plant)
    manipulator.set_joint_properties(plant)

    # Passive springs
    for jt_name in JOINT_NAMES:
        jt_cfg = config.joint_configs.get(jt_name)
        K = jt_cfg.stiffness if jt_cfg is not None else 0.0
        if K > 0.0:
            jt = manipulator.get_joint_by_name(plant, jt_name)
            plant.AddForceElement(
                RevoluteSpring(jt, nominal_angle=0.0, stiffness=K)
            )

    manipulator.add_end_effector_frame(plant)
    plant.Finalize()

    manipulator.init_cable_rig(URDF_PATH)
    rig = manipulator.rig

    builder.AddSystem(plant)
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    # ── Meshcat ──────────────────────────────────────────────────────────
    meshcat = None
    if enable_meshcat:
        meshcat = StartMeshcat()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        # Cable visualizer
        cable_viz = builder.AddSystem(
            CableVisPublisher(plant, manipulator, rig, meshcat)
        )
        builder.Connect(
            plant.get_state_output_port(),
            cable_viz.get_input_port(0),
        )

    # ── ROS subscriber: /torque_command → plant actuation ────────────────
    nu = plant.num_actuated_dofs()
    torque_sub = builder.AddSystem(
        RosSubscriberSystem.Make(Float64MultiArray, "/torque_command", qos, ros_if)
    )
    dispatcher = builder.AddSystem(TorqueDispatcher(nu))
    builder.Connect(torque_sub.get_output_port(0), dispatcher.get_input_port(0))
    builder.Connect(dispatcher.get_output_port(0), plant.get_actuation_input_port())

    # ── ROS publisher: /joint_states ─────────────────────────────────────
    js_broadcaster = builder.AddSystem(
        JointStateBroadcaster(plant, manipulator.model_instance, manipulator)
    )
    builder.Connect(plant.get_state_output_port(), js_broadcaster.get_input_port(0))
    js_pub = builder.AddSystem(
        RosPublisherSystem.Make(JointState, "/joint_states", qos, ros_if)
    )
    builder.Connect(js_broadcaster.get_output_port(0), js_pub.get_input_port(0))

    # ── ROS publisher: /ee_position ──────────────────────────────────────
    ee_broadcaster = builder.AddSystem(
        EEPositionBroadcaster(plant, manipulator.model_instance, manipulator)
    )
    builder.Connect(plant.get_state_output_port(), ee_broadcaster.get_input_port(0))
    ee_pub = builder.AddSystem(
        RosPublisherSystem.Make(Point, "/ee_position", qos, ros_if)
    )
    builder.Connect(ee_broadcaster.get_output_port(0), ee_pub.get_input_port(0))

    # ── Build ────────────────────────────────────────────────────────────
    diagram = builder.Build()

    return diagram, plant, manipulator, meshcat, ros_if


def build_scene_viz_diagram(
    enable_meshcat: bool,
    joint_damping: tuple,
    joint_stiffness: tuple,
):
    """Build diagram for scene-viz mode: position-driven, no physics."""
    config = create_cable_manipulator_config(
        urdf_path=URDF_PATH,
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=joint_damping,
        stiffness=joint_stiffness,
    )

    builder = DiagramBuilder()
    qos = QoSProfile(depth=10)

    # ── ROS interface ────────────────────────────────────────────────────
    ros_if = builder.AddSystem(RosInterfaceSystem("scene_viz_plant_node"))
    ClockSystem.AddToBuilder(builder, ros_if)

    # ── Plant (continuous, no dynamics stepping needed) ───────────────────
    plant = MultibodyPlant(time_step=0.0)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulatorTendon(config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(plant, position=np.zeros(3))
    manipulator.add_joint_actuators(plant)
    manipulator.add_end_effector_frame(plant)
    plant.Finalize()

    manipulator.init_cable_rig(URDF_PATH)
    rig = manipulator.rig

    builder.AddSystem(plant)
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    # Zero actuation (scene-viz doesn't use dynamics)
    nu = plant.num_actuated_dofs()
    zero_torque = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))
    builder.Connect(zero_torque.get_output_port(0), plant.get_actuation_input_port())

    # ── Meshcat ──────────────────────────────────────────────────────────
    meshcat = StartMeshcat() if enable_meshcat else None
    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        cable_viz = builder.AddSystem(
            CableVisPublisher(plant, manipulator, rig, meshcat)
        )
        builder.Connect(
            plant.get_state_output_port(),
            cable_viz.get_input_port(0),
        )

    # ── ROS subscriber: /joint_position_command ──────────────────────────
    pos_sub = builder.AddSystem(
        RosSubscriberSystem.Make(Float64MultiArray, "/joint_position_command", qos, ros_if)
    )

    # ── ROS publisher: /joint_states ─────────────────────────────────────
    js_broadcaster = builder.AddSystem(
        JointStateBroadcaster(plant, manipulator.model_instance, manipulator)
    )
    builder.Connect(plant.get_state_output_port(), js_broadcaster.get_input_port(0))
    js_pub = builder.AddSystem(
        RosPublisherSystem.Make(JointState, "/joint_states", qos, ros_if)
    )
    builder.Connect(js_broadcaster.get_output_port(0), js_pub.get_input_port(0))

    # ── ROS publisher: /ee_position ──────────────────────────────────────
    ee_broadcaster = builder.AddSystem(
        EEPositionBroadcaster(plant, manipulator.model_instance, manipulator)
    )
    builder.Connect(plant.get_state_output_port(), ee_broadcaster.get_input_port(0))
    ee_pub = builder.AddSystem(
        RosPublisherSystem.Make(Point, "/ee_position", qos, ros_if)
    )
    builder.Connect(ee_broadcaster.get_output_port(0), ee_pub.get_input_port(0))

    # ── Build ────────────────────────────────────────────────────────────
    diagram = builder.Build()

    return diagram, plant, manipulator, meshcat, ros_if, pos_sub


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", choices=["dynamics", "scene-viz"], default="dynamics",
        help="Operating mode [default: dynamics]",
    )
    ap.add_argument(
        "--timestep", type=float, default=0.002,
        help="Sim timestep (s) — dynamics mode only [default: 0.002]",
    )
    ap.add_argument(
        "--no-meshcat", action="store_true",
        help="Disable Meshcat (dynamics only; scene-viz always shows it)",
    )
    ap.add_argument(
        "--joint-damping", type=float, nargs=2, default=[0.05, 0.05],
        metavar=("D1", "D2"), help="Joint damping [Nm·s/rad]",
    )
    ap.add_argument(
        "--joint-stiffness", type=float, nargs=2, default=[2.5, 2.5],
        metavar=("K1", "K2"), help="Passive spring stiffness [Nm/rad]",
    )
    ap.add_argument(
        "--simulation_sec", type=float, default=float("inf"),
        help="How many seconds to run (default: forever)",
    )
    args = ap.parse_args()

    enable_meshcat = (not args.no_meshcat) or (args.mode == "scene-viz")

    print(f"[plant] drake-ros backend: {BACKEND}")
    drake_ros_init()

    if args.mode == "dynamics":
        diagram, plant, manipulator, meshcat, ros_if = build_dynamics_diagram(
            sim_timestep=args.timestep,
            enable_meshcat=enable_meshcat,
            joint_damping=tuple(args.joint_damping),
            joint_stiffness=tuple(args.joint_stiffness),
        )
    else:
        diagram, plant, manipulator, meshcat, ros_if, pos_sub = build_scene_viz_diagram(
            enable_meshcat=enable_meshcat,
            joint_damping=tuple(args.joint_damping),
            joint_stiffness=tuple(args.joint_stiffness),
        )

    # ── Create simulator ─────────────────────────────────────────────────
    simulator = Simulator(diagram)
    sim_ctx = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Patch zero-mass bodies
    patched = []
    for idx in plant.GetBodyIndices(manipulator.model_instance):
        body = plant.get_body(idx)
        if body.default_mass() < 1e-6:
            body.SetSpatialInertiaInBodyFrame(plant_ctx, _M_PATCH)
            patched.append(body.name())

    # Set home position
    manipulator.set_positions_user_order(plant, plant_ctx, np.zeros(NUM_JOINTS))
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    if args.mode == "dynamics":
        simulator.Initialize()
    else:
        # Push initial state to Meshcat
        diagram.ForcedPublish(sim_ctx)
        manipulator.compute_tangents(plant, plant_ctx)
        if meshcat:
            draw_cables(meshcat, plant, plant_ctx, manipulator, manipulator.rig)

    simulator.set_target_realtime_rate(1.0)

    node = ros_if.get_node()
    if patched:
        node.get_logger().warn(f"Patched zero-mass bodies (0.3 kg): {patched}")
    if meshcat:
        node.get_logger().info(f"Meshcat: {meshcat.web_url()}")
    node.get_logger().info(
        f"drake-ros plant ready  |  mode={args.mode}  |  "
        f"joints={JOINT_NAMES}  |  sim_time={args.simulation_sec}s\n"
        f"  Topics IN:  {'torque_command' if args.mode == 'dynamics' else 'joint_position_command'}\n"
        f"  Topics OUT: /joint_states, /ee_position, /clock"
    )

    # ── Simulation loop ──────────────────────────────────────────────────
    step = 0.1  # same step size as drake-ros examples
    try:
        if args.mode == "scene-viz":
            # Scene-viz: we need to manually set joint positions from the
            # subscriber each step, then force-publish to Meshcat.
            while sim_ctx.get_time() < args.simulation_sec:
                # Read latest position command from the subscriber
                sub_ctx = pos_sub.GetMyContextFromRoot(sim_ctx)
                msg = pos_sub.get_output_port(0).Eval(sub_ctx)
                if len(msg.data) == NUM_JOINTS:
                    q_cmd = np.array(msg.data, dtype=float)
                    manipulator.set_positions_user_order(plant, plant_ctx, q_cmd)
                    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

                next_t = min(sim_ctx.get_time() + step, args.simulation_sec)
                simulator.AdvanceTo(next_t)
        else:
            # Dynamics: just advance the simulator — everything is wired
            while sim_ctx.get_time() < args.simulation_sec:
                next_t = min(sim_ctx.get_time() + step, args.simulation_sec)
                simulator.AdvanceTo(next_t)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        drake_ros_shutdown()


if __name__ == "__main__":
    main()
