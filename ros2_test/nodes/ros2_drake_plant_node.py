"""
Drake Cable-Manipulator Plant — ROS 2 Node (Unified)
=====================================================

Single plant node for the 2-DOF **cable (tendon) manipulator** with two
operating modes selected via ``--mode``:

  dynamics   (default)
      Discrete-time MultibodyPlant with full physics simulation.
      Receives joint torques, steps the simulator each tick.
      Subscribes:  /torque_command          (std_msgs/Float64MultiArray)
      Publishes:   /joint_states, /ee_position

  scene-viz
      Continuous-time plant (time_step=0), no dynamics stepping.
      Positions are set directly from ROS 2 commands; ForcedPublish
      drives Meshcat — identical to the ``scene-viz-q`` mode in the
      standalone scripts.
      Subscribes:  /joint_position_command  (std_msgs/Float64MultiArray)
      Publishes:   /joint_states, /ee_position

Architecture
────────────
  ┌──────────────────────────────────────────────────────────────────┐
  │  ros2_drake_plant_node.py                                        │
  │                                                                  │
  │  [dynamics]   MultibodyPlant ←── /torque_command                 │
  │               (discrete, AdvanceTo)                              │
  │                                                                  │
  │  [scene-viz]  MultibodyPlant ←── /joint_position_command         │
  │               (continuous, ForcedPublish)                        │
  │                                                                  │
  │               ├──► /joint_states   (pub)                         │
  │               └──► /ee_position    (pub)                         │
  │               └──► Meshcat + cable viz                           │
  └──────────────────────────────────────────────────────────────────┘

Usage:
    conda activate pydrake_ros2

    # Physics simulation (torque control)
    python ros2_test/ros2_drake_plant_node.py --mode dynamics --timestep 0.002 --rate 500

    # Position-driven Meshcat visualization
    python ros2_test/ros2_drake_plant_node.py --mode scene-viz --rate 30
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    Simulator,
    Parser,
    MeshcatVisualizer,
    StartMeshcat,
    SpatialInertia,
    UnitInertia,
)
from pydrake.multibody.tree import RevoluteSpring

# ── Project imports ──────────────────────────────────────────────────────────
import sys

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

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
JOINT_NAMES = ["link1_base", "link2_link1"]   # user order
NUM_JOINTS = len(JOINT_NAMES)

_M_PATCH = SpatialInertia(
    mass=0.3,
    p_PScm_E=np.zeros(3),
    G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)


# ═════════════════════════════════════════════════════════════════════════════
class DrakePlantNode(Node):
    """Unified ROS 2 node for the Drake cable-manipulator plant.

    Parameters
    ----------
    mode : {"dynamics", "scene-viz"}
    sim_timestep : float
        Simulation timestep (s) — only used in ``dynamics`` mode.
    publish_rate_hz : float
        Timer rate for stepping/publishing.
    enable_meshcat : bool
        Show Meshcat. Always True in ``scene-viz`` mode.
    """

    def __init__(
        self,
        mode: str = "dynamics",
        sim_timestep: float = 0.002,
        publish_rate_hz: float = 500.0,
        enable_meshcat: bool = True,
        joint_damping: tuple = (0.05, 0.05),
        joint_stiffness: tuple = (2.5, 2.5),
    ):
        node_name = "scene_viz_plant_node" if mode == "scene-viz" else "drake_plant_node"
        super().__init__(node_name)
        self._mode = mode
        self.get_logger().info(f"Building Drake plant  [mode={mode}] …")

        # ── Config ───────────────────────────────────────────────────────
        config = create_cable_manipulator_config(
            urdf_path=URDF_PATH,
            joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
            damping=joint_damping,
            stiffness=joint_stiffness,
        )

        # ── Drake diagram ────────────────────────────────────────────────
        builder = DiagramBuilder()
        time_step = 0.0 if mode == "scene-viz" else sim_timestep
        self.plant = MultibodyPlant(time_step=time_step)
        scene_graph = builder.AddSystem(SceneGraph())
        self.plant.RegisterAsSourceForSceneGraph(scene_graph)

        self.manipulator = CupManipulatorTendon(config, enable_visualization=True)
        parser_urdf = Parser(self.plant)
        self.manipulator.load_urdf_to_plant(self.plant, parser_urdf)
        self.manipulator.weld_base_to_world(self.plant, position=np.zeros(3))
        self.manipulator.add_joint_actuators(self.plant)

        if mode == "dynamics":
            self.manipulator.set_joint_properties(self.plant)
            # Passive joint springs
            for jt_name in JOINT_NAMES:
                jt_cfg = config.joint_configs.get(jt_name)
                K = jt_cfg.stiffness if jt_cfg is not None else 0.0
                if K > 0.0:
                    jt = self.manipulator.get_joint_by_name(self.plant, jt_name)
                    self.plant.AddForceElement(
                        RevoluteSpring(jt, nominal_angle=0.0, stiffness=K)
                    )

        self.manipulator.add_end_effector_frame(self.plant)
        self.plant.Finalize()

        self.manipulator.init_cable_rig(URDF_PATH)
        self.rig = self.manipulator.rig

        builder.AddSystem(self.plant)
        builder.Connect(
            self.plant.get_geometry_pose_output_port(),
            scene_graph.get_source_pose_port(self.plant.get_source_id()),
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port(),
        )

        # Meshcat — scene-viz always needs it; dynamics makes it optional
        self.meshcat = None
        if enable_meshcat or mode == "scene-viz":
            self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)
            self.get_logger().info(f"Meshcat: {self.meshcat.web_url()}")

        self.diagram = builder.Build()

        # ── Simulator ────────────────────────────────────────────────────
        self.simulator = Simulator(self.diagram)
        self.sim_context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(
            self.sim_context
        )

        # Patch zero-mass bodies (Onshape parts with no material → mass=0)
        patched = []
        for idx in self.plant.GetBodyIndices(self.manipulator.model_instance):
            body = self.plant.get_body(idx)
            if body.default_mass() < 1e-6:
                body.SetSpatialInertiaInBodyFrame(self.plant_context, _M_PATCH)
                patched.append(body.name())
        if patched:
            self.get_logger().warn(
                f"Patched zero-mass bodies (0.3 kg nominal): {patched}"
            )

        # Set home position (q = 0, 0)
        self.manipulator.set_positions_user_order(
            self.plant, self.plant_context, np.zeros(NUM_JOINTS)
        )
        self.plant.SetVelocities(
            self.plant_context, np.zeros(self.plant.num_velocities())
        )

        if mode == "dynamics":
            self.simulator.Initialize()
        else:
            # Scene-viz: push home pose to Meshcat immediately
            self.diagram.ForcedPublish(self.sim_context)
            self.manipulator.compute_tangents(self.plant, self.plant_context)
            draw_cables(
                self.meshcat, self.plant, self.plant_context,
                self.manipulator, self.rig,
            )

        # ── ROS 2 publishers (common to both modes) ──────────────────────
        self.joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self.ee_pub = self.create_publisher(Point, "/ee_position", 10)

        # ── Mode-specific subscriber + timer ─────────────────────────────
        if mode == "dynamics":
            self._torque_cmd = np.zeros(NUM_JOINTS)
            self._torque_lock = threading.Lock()
            self.create_subscription(
                Float64MultiArray, "/torque_command", self._torque_cb, 10
            )
            self.sim_dt = sim_timestep
            self.create_timer(1.0 / publish_rate_hz, self._step_and_publish)
        else:
            self._q_cmd = np.zeros(NUM_JOINTS)
            self._cmd_lock = threading.Lock()
            self._cmd_updated = False
            self.create_subscription(
                Float64MultiArray, "/joint_position_command",
                self._position_cmd_cb, 10,
            )
            self.create_timer(1.0 / publish_rate_hz, self._publish_tick)

        self.get_logger().info(
            f"Plant ready  |  mode={mode}  |  robot=manipulator_cable  |  "
            f"joints={JOINT_NAMES}  |  pub@{publish_rate_hz} Hz"
            + (f"  |  dt={sim_timestep}" if mode == "dynamics" else "")
        )

    # ── dynamics callbacks ────────────────────────────────────────────────
    def _torque_cb(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f"Expected {NUM_JOINTS} torques, got {len(msg.data)}"
            )
            return
        with self._torque_lock:
            if not hasattr(self, "_torque_received"):
                self._torque_received = True
                self.get_logger().info("Received first /torque_command")
            self._torque_cmd = np.array(msg.data, dtype=float)

    def _step_and_publish(self):
        with self._torque_lock:
            tau = self._torque_cmd.copy()

        self.plant.get_actuation_input_port().FixValue(self.plant_context, tau)

        target_time = self.sim_context.get_time() + self.sim_dt
        self.simulator.AdvanceTo(target_time)

        if self.meshcat is not None and self.rig is not None:
            try:
                self.manipulator.compute_tangents(self.plant, self.plant_context)
                draw_cables(
                    self.meshcat, self.plant, self.plant_context,
                    self.manipulator, self.rig,
                )
            except Exception:
                pass

        q = self.manipulator.get_positions_user_order(self.plant, self.plant_context)
        qd = self.manipulator.get_velocities_user_order(self.plant, self.plant_context)
        self._publish_joint_state(q, qd, effort=tau)
        self._publish_ee()

    # ── scene-viz callbacks ───────────────────────────────────────────────
    def _position_cmd_cb(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f"Expected {NUM_JOINTS} positions, got {len(msg.data)}"
            )
            return
        with self._cmd_lock:
            if not hasattr(self, "_first_cmd_received"):
                self._first_cmd_received = True
                self.get_logger().info("Received first /joint_position_command")
            self._q_cmd = np.array(msg.data, dtype=float)
            self._cmd_updated = True

    def _publish_tick(self):
        with self._cmd_lock:
            q = self._q_cmd.copy()
            updated = self._cmd_updated
            self._cmd_updated = False

        if updated:
            self.manipulator.set_positions_user_order(
                self.plant, self.plant_context, q
            )
            self.plant.SetVelocities(
                self.plant_context, np.zeros(self.plant.num_velocities())
            )

        # Always push to Meshcat every tick
        self.diagram.ForcedPublish(self.sim_context)

        if updated:
            try:
                self.manipulator.compute_tangents(self.plant, self.plant_context)
                draw_cables(
                    self.meshcat, self.plant, self.plant_context,
                    self.manipulator, self.rig,
                )
            except Exception:
                pass

        q_actual = self.manipulator.get_positions_user_order(
            self.plant, self.plant_context
        )
        self._publish_joint_state(q_actual, np.zeros(NUM_JOINTS), effort=np.zeros(NUM_JOINTS))
        self._publish_ee()

    # ── common helpers ────────────────────────────────────────────────────
    def _publish_joint_state(self, q: np.ndarray, qd: np.ndarray, effort: np.ndarray):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(JOINT_NAMES)
        js.position = q.tolist()
        js.velocity = qd.tolist()
        js.effort = effort.tolist()
        self.joint_state_pub.publish(js)

    def _publish_ee(self):
        ee = self.manipulator.get_end_effector_position(
            self.plant, self.plant_context
        )
        self.ee_pub.publish(Point(x=float(ee[0]), y=float(ee[1]), z=float(ee[2])))


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Drake Cable-Manipulator Plant — ROS 2 Node (Unified)"
    )
    ap.add_argument(
        "--mode", choices=["dynamics", "scene-viz"], default="scene-viz",
        help="Operating mode  [default: dynamics]",
    )
    ap.add_argument(
        "--timestep", type=float, default=0.002,
        help="Sim timestep (s) — dynamics mode only  [default: 0.002]",
    )
    ap.add_argument(
        "--rate", type=float, default=None,
        help="Publish rate (Hz)  [default: 500 for dynamics, 30 for scene-viz]",
    )
    ap.add_argument(
        "--no-meshcat", action="store_true",
        help="Disable Meshcat — dynamics mode only (scene-viz always shows Meshcat)",
    )
    ap.add_argument(
        "--joint-damping", type=float, nargs=2, default=[0.05, 0.05],
        metavar=("D1", "D2"), help="Joint damping [Nm·s/rad]",
    )
    ap.add_argument(
        "--joint-stiffness", type=float, nargs=2, default=[2.5, 2.5],
        metavar=("K1", "K2"), help="Passive spring stiffness [Nm/rad]",
    )
    args = ap.parse_args()

    default_rate = 30.0 if args.mode == "scene-viz" else 500.0
    rate = args.rate if args.rate is not None else default_rate

    rclpy.init()
    node = DrakePlantNode(
        mode=args.mode,
        sim_timestep=args.timestep,
        publish_rate_hz=rate,
        enable_meshcat=not args.no_meshcat,
        joint_damping=tuple(args.joint_damping),
        joint_stiffness=tuple(args.joint_stiffness),
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
