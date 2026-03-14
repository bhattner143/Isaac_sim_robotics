"""
ros2_drakeROS_controller_node.py
=================================

Drake-ROS-style **computed-torque controller** — all ROS 2 I/O is wired as
``LeafSystem`` blocks inside a **single Drake Diagram** that is advanced by
a ``Simulator`` at real-time rate.

This re-implements ``ros2_computed_torque_controller_node.py`` with the same
topology used by the official ``drake_ros_examples``:

   DiagramBuilder
   ├── RosInterfaceSystem          (rclpy node + spin-thread)
   ├── ClockSystem                 (/clock publisher)
   ├── RosSubscriberSystem         (/joint_states subscriber)
   ├── ComputedTorqueSystem        (computed-torque control law)
   └── RosPublisherSystem          (/torque_command publisher)

Control Law (feedback linearisation)
─────────────────────────────────────
    a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_des − q̇)
    τ     = M(q)·a_des + h(q, q̇)

where h(q,q̇) = C(q,q̇)·q̇ + g(q) is the Drake bias term.

Joint-2 torque is decomposed into non-negative cable tensions:
    F_net   = τ2 / r_p
    T_green = max(F_net,  0)
    T_red   = max(−F_net, 0)

ROS 2 Topics
─────────────
    IN:  /joint_states   (sensor_msgs/JointState)        — plant feedback
    OUT: /torque_command  (std_msgs/Float64MultiArray)    — [τ1, τ2]
    OUT: /clock           (rosgraph_msgs/Clock)

Usage
-----
    conda activate pydrake_ros2

    # Terminal 1: start plant
    python ros2_test/ros2_drakeROS_plant_node.py --mode dynamics --timestep 0.002

    # Terminal 2: start controller
    python ros2_test/ros2_drakeROS_controller_node.py \\
        --kp 10000 --kd 400 --tau-max 10 \\
        --mode min-jerk --q-goal 60,-120 --duration 3.0

    # Hold at current position
    python ros2_test/ros2_drakeROS_controller_node.py --mode hold --q-start 0,0
"""

from __future__ import annotations

import argparse
import sys
import time as _time
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
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    SpatialInertia,
    UnitInertia,
)
from pydrake.common.value import AbstractValue
from pydrake.multibody.tree import MultibodyForces, RevoluteSpring
from pydrake.systems.analysis import Simulator as _Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem

from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robots.cup_manipulator_tendon import (
    CupManipulatorTendon,
    create_cable_manipulator_config,
)

# ── Constants ────────────────────────────────────────────────────────────────
URDF_PATH = str(
    WORKSPACE
    / "model_using_onshape_to_robot"
    / "manipulator_cable"
    / "manipulator_cable_obj.urdf"
)
JOINT_NAMES = ["link1_base", "link2_link1"]
NUM_JOINTS = len(JOINT_NAMES)

_M_PATCH = SpatialInertia(
    mass=0.3,
    p_PScm_E=np.zeros(3),
    G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory generators
# ═════════════════════════════════════════════════════════════════════════════

class MinJerkTrajectory:
    """5th-order polynomial minimum-jerk trajectory in joint space."""

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float):
        self.q0 = np.asarray(q_start, dtype=float)
        self.qf = np.asarray(q_goal, dtype=float)
        self.T = float(duration)

    def evaluate(self, t: float):
        if t >= self.T:
            return self.qf.copy(), np.zeros_like(self.qf), np.zeros_like(self.qf)
        s = np.clip(t / self.T, 0.0, 1.0)
        h = 10 * s**3 - 15 * s**4 + 6 * s**5
        hd = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.T
        hdd = (60 * s - 180 * s**2 + 120 * s**3) / self.T**2
        dq = self.qf - self.q0
        return self.q0 + dq * h, dq * hd, dq * hdd


class HoldTrajectory:
    """Hold constant joint angles."""

    def __init__(self, q_hold: np.ndarray):
        self.q = np.asarray(q_hold, dtype=float)

    def evaluate(self, _t: float):
        z = np.zeros_like(self.q)
        return self.q.copy(), z, z


# ═════════════════════════════════════════════════════════════════════════════
# Computed Torque LeafSystem (lives inside the Drake diagram)
# ═════════════════════════════════════════════════════════════════════════════

class ComputedTorqueSystem(LeafSystem):
    """Drake LeafSystem implementing the computed-torque control law.

    Input ports:
        [0] joint_state_msg  — ``sensor_msgs/JointState`` (abstract)

    Output ports:
        [0] torque_msg       — ``std_msgs/Float64MultiArray`` (abstract)

    Internally maintains a separate lightweight Drake ``MultibodyPlant`` for
    dynamics queries (mass-matrix, bias forces).
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        tau_max: float,
        trajectory,
        joint_damping: tuple,
        joint_stiffness: tuple,
    ) -> None:
        LeafSystem.__init__(self)

        self._Kp = float(kp)
        self._Kd = float(kd)
        self._tau_max = float(tau_max)
        self._trajectory = trajectory
        self._t0 = None  # wall-clock reference when first sensor data arrives
        self._sensor_received = False

        # ── Build private Drake plant for dynamics queries ───────────────
        config = create_cable_manipulator_config(
            urdf_path=URDF_PATH,
            joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
            damping=joint_damping,
            stiffness=joint_stiffness,
        )
        self._ctrl_plant = MultibodyPlant(time_step=0.0)
        sg = SceneGraph()
        self._ctrl_plant.RegisterAsSourceForSceneGraph(sg)

        self._manip = CupManipulatorTendon(config, enable_visualization=False)
        ctrl_parser = Parser(self._ctrl_plant)
        self._manip.load_urdf_to_plant(self._ctrl_plant, ctrl_parser)
        self._manip.weld_base_to_world(self._ctrl_plant, position=np.zeros(3))
        self._manip.add_joint_actuators(self._ctrl_plant)
        self._manip.set_joint_properties(self._ctrl_plant)

        # Passive springs (must match plant node)
        for jt_name in JOINT_NAMES:
            jt_cfg = config.joint_configs.get(jt_name)
            K = jt_cfg.stiffness if jt_cfg is not None else 0.0
            if K > 0.0:
                jt = self._manip.get_joint_by_name(self._ctrl_plant, jt_name)
                self._ctrl_plant.AddForceElement(
                    RevoluteSpring(jt, nominal_angle=0.0, stiffness=K)
                )

        self._manip.add_end_effector_frame(self._ctrl_plant)
        self._ctrl_plant.Finalize()
        self._ctrl_context = self._ctrl_plant.CreateDefaultContext()

        # Patch zero-mass bodies
        for idx in self._ctrl_plant.GetBodyIndices(self._manip.model_instance):
            body = self._ctrl_plant.get_body(idx)
            if body.default_mass() < 1e-6:
                body.SetSpatialInertiaInBodyFrame(self._ctrl_context, _M_PATCH)

        self._forces = MultibodyForces(self._ctrl_plant)
        self._r_p = self._manip.PULLEY_RADIUS

        # Velocity-vector indices for user-order joints
        j1 = self._manip.get_joint_by_name(self._ctrl_plant, JOINT_NAMES[0])
        j2 = self._manip.get_joint_by_name(self._ctrl_plant, JOINT_NAMES[1])
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # ── Drake ports ──────────────────────────────────────────────────
        self._js_input = self.DeclareAbstractInputPort(
            "joint_state_msg", AbstractValue.Make(JointState())
        )
        self.DeclareAbstractOutputPort(
            "torque_msg",
            lambda: AbstractValue.Make(Float64MultiArray()),
            self._calc_torque,
        )

    def _calc_torque(self, context, output) -> None:
        js_msg = self._js_input.Eval(context)

        # Parse joint state from message
        q_user = np.zeros(NUM_JOINTS)
        qd_user = np.zeros(NUM_JOINTS)

        if len(js_msg.position) != NUM_JOINTS:
            # No valid data yet — output zero torque
            msg = output.get_mutable_value()
            msg.data = [0.0] * NUM_JOINTS
            return

        for i, name in enumerate(js_msg.name):
            if name in JOINT_NAMES:
                idx = JOINT_NAMES.index(name)
                q_user[idx] = js_msg.position[i]
                qd_user[idx] = js_msg.velocity[i]

        # Time reference: use simulation time from context
        t = context.get_time()

        # Desired trajectory
        q_des, qd_des, qdd_des = self._trajectory.evaluate(t)

        # Set state in dynamics-only plant
        self._manip.set_positions_user_order(
            self._ctrl_plant, self._ctrl_context, q_user
        )
        self._manip.set_velocities_user_order(
            self._ctrl_plant, self._ctrl_context, qd_user
        )

        # Computed-torque law
        a_des_user = qdd_des + self._Kp * (q_des - q_user) + self._Kd * (qd_des - qd_user)

        nv = self._ctrl_plant.num_velocities()
        a_des_drake = np.zeros(nv)
        a_des_drake[self._v_idx[0]] = a_des_user[0]
        a_des_drake[self._v_idx[1]] = a_des_user[1]

        # τ = M(q)·a_des + h(q, q̇)
        self._forces.SetZero()
        tau_full = self._ctrl_plant.CalcInverseDynamics(
            self._ctrl_context, a_des_drake, self._forces
        )

        tau1 = float(tau_full[self._v_idx[0]])
        tau2 = float(tau_full[self._v_idx[1]])

        # Cable tension decomposition for joint 2
        F_net = tau2 / self._r_p
        T_green = max(F_net, 0.0)
        T_red = max(-F_net, 0.0)
        tau2_cmd = (T_green - T_red) * self._r_p

        tau_clip = np.clip(
            np.array([tau1, tau2_cmd]), -self._tau_max, self._tau_max
        )

        msg = output.get_mutable_value()
        msg.data = tau_clip.tolist()


# ═════════════════════════════════════════════════════════════════════════════
# Main builder
# ═════════════════════════════════════════════════════════════════════════════

def build_controller_diagram(
    kp: float,
    kd: float,
    tau_max: float,
    trajectory,
    joint_damping: tuple,
    joint_stiffness: tuple,
):
    """Build the Drake diagram for the computed-torque controller."""
    builder = DiagramBuilder()
    qos = QoSProfile(depth=10)

    # ── ROS interface ────────────────────────────────────────────────────
    ros_if = builder.AddSystem(RosInterfaceSystem("computed_torque_controller"))
    ClockSystem.AddToBuilder(builder, ros_if)

    # ── ROS subscriber: /joint_states ────────────────────────────────────
    js_sub = builder.AddSystem(
        RosSubscriberSystem.Make(JointState, "/joint_states", qos, ros_if)
    )

    # ── Computed-torque controller ───────────────────────────────────────
    ctrl = builder.AddSystem(
        ComputedTorqueSystem(
            kp=kp, kd=kd, tau_max=tau_max,
            trajectory=trajectory,
            joint_damping=joint_damping,
            joint_stiffness=joint_stiffness,
        )
    )
    builder.Connect(js_sub.get_output_port(0), ctrl.get_input_port(0))

    # ── ROS publisher: /torque_command ───────────────────────────────────
    tau_pub = builder.AddSystem(
        RosPublisherSystem.Make(Float64MultiArray, "/torque_command", qos, ros_if)
    )
    builder.Connect(ctrl.get_output_port(0), tau_pub.get_input_port(0))

    # ── Build ────────────────────────────────────────────────────────────
    diagram = builder.Build()
    return diagram, ros_if


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--kp", type=float, default=10000.0,
                    help="Position gain Kp [s⁻²] (default: 10000 → ωn=100)")
    ap.add_argument("--kd", type=float, default=400.0,
                    help="Velocity gain Kd [s⁻¹] (default: 400 → ζ=2)")
    ap.add_argument("--tau-max", type=float, default=10.0,
                    help="Torque saturation [Nm] (default: 10)")
    ap.add_argument("--mode", choices=["hold", "min-jerk"], default="min-jerk",
                    help="Trajectory mode")
    ap.add_argument("--q-start", type=str, default="0,0",
                    help="Start angles in degrees (comma-separated)")
    ap.add_argument("--q-goal", type=str, default="60,-120",
                    help="Goal angles in degrees (comma-separated)")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="Trajectory duration (s)")
    ap.add_argument("--simulation_sec", type=float, default=float("inf"),
                    help="How many seconds to run (default: forever)")
    ap.add_argument("--joint-damping", type=float, nargs=2, default=[0.05, 0.05],
                    metavar=("D1", "D2"), help="Joint damping [Nm·s/rad]")
    ap.add_argument("--joint-stiffness", type=float, nargs=2, default=[2.5, 2.5],
                    metavar=("K1", "K2"), help="Passive spring stiffness [Nm/rad]")
    args = ap.parse_args()

    q_start = np.deg2rad(np.array([float(x) for x in args.q_start.split(",")]))
    q_goal = np.deg2rad(np.array([float(x) for x in args.q_goal.split(",")]))

    if args.mode == "hold":
        traj = HoldTrajectory(q_start)
    else:
        traj = MinJerkTrajectory(q_start, q_goal, args.duration)

    print(f"[controller] drake-ros backend: {BACKEND}")
    drake_ros_init()

    diagram, ros_if = build_controller_diagram(
        kp=args.kp,
        kd=args.kd,
        tau_max=args.tau_max,
        trajectory=traj,
        joint_damping=tuple(args.joint_damping),
        joint_stiffness=tuple(args.joint_stiffness),
    )

    # ── Simulate ─────────────────────────────────────────────────────────
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    sim_ctx = simulator.get_mutable_context()

    node = ros_if.get_node()
    wn = np.sqrt(args.kp) if args.kp > 0 else 0.0
    zeta = args.kd / (2.0 * wn) if wn > 0 else 0.0
    node.get_logger().info(
        f"drake-ros computed-torque controller ready\n"
        f"  Kp={args.kp}  Kd={args.kd}  τ_max={args.tau_max}\n"
        f"  ωn={wn:.1f} rad/s  ζ={zeta:.2f}\n"
        f"  mode={args.mode}  q_start={np.rad2deg(q_start)}°  "
        f"q_goal={np.rad2deg(q_goal)}°  T={args.duration}s\n"
        f"  Topics IN:  /joint_states\n"
        f"  Topics OUT: /torque_command, /clock"
    )

    step = 0.1
    try:
        while sim_ctx.get_time() < args.simulation_sec:
            next_t = min(sim_ctx.get_time() + step, args.simulation_sec)
            simulator.AdvanceTo(next_t)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        drake_ros_shutdown()


if __name__ == "__main__":
    main()
