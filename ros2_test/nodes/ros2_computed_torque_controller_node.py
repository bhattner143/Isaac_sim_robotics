"""
Computed-Torque Controller — ROS 2 Node
========================================

Implements a computed-torque (inverse-dynamics + PD) controller for the
2-DOF cable (tendon) manipulator, matching the control law in
``script_cup_manipulator_pendulam_with_spring_damper.py``.

Control Law  (feedback linearisation)
─────────────────────────────────────
    a_des = q̈_ref  +  Kp · (q_des − q)  +  Kd · (q̇_ref − q̇)
    τ     = M(q) · a_des  +  h(q, q̇)

where  h(q, q̇) = C(q,q̇)·q̇ + g(q)  is the Drake bias term from
CalcInverseDynamics(ctx, vdot=0).

Joint-2 torque is decomposed into non-negative cable tensions:
    F_net   = τ2 / r_p
    T_green = max(F_net,  0)    (retracting side)
    T_red   = max(−F_net, 0)   (extending side)

Subscribed topics:
    /joint_states   (sensor_msgs/JointState)    — plant feedback [q, q̇]

Published topics:
    /torque_command (std_msgs/Float64MultiArray) — [τ1, τ2] Nm

Architecture
────────────
  ┌──────────────────────────────────────────────────────────┐
  │ ros2_computed_torque_controller_node.py                  │
  │                                                          │
  │  /joint_states (sub) ──►  Computed Torque  ──► /torque   │
  │                           Controller            _command │
  └──────────────────────────────────────────────────────────┘

Usage:
    conda activate pydrake_ros2
    python ros2_test/ros2_computed_torque_controller_node.py \\
        --kp 10000 --kd 400 --tau-max 10 --rate 500 \\
        --mode min-jerk --q-start 0,0 --q-goal 60,-120 --duration 3.0
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

from pydrake.all import (
    MultibodyPlant,
    SceneGraph,
    Parser,
    RigidTransform,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)
from pydrake.multibody.tree import MultibodyForces, RevoluteSpring

# ── Project imports ──────────────────────────────────────────────────────────
import sys

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

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
class ComputedTorqueControllerNode(Node):
    """Computed-torque controller using CalcInverseDynamics (same as the
    spring-damper script's ComputedTorqueController LeafSystem)."""

    def __init__(
        self,
        kp: float,
        kd: float,
        tau_max: float,
        trajectory,
        control_rate_hz: float = 500.0,
        joint_damping: tuple = (0.05, 0.05),
        joint_stiffness: tuple = (2.5, 2.5),
    ):
        super().__init__("computed_torque_controller")

        # ── Build a light-weight Drake plant for dynamics queries ─────────
        config = create_cable_manipulator_config(
            urdf_path=URDF_PATH,
            joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
            damping=joint_damping,
            stiffness=joint_stiffness,
        )
        self.ctrl_plant = MultibodyPlant(time_step=0.0)  # continuous for dynamics
        scene_graph = SceneGraph()
        self.ctrl_plant.RegisterAsSourceForSceneGraph(scene_graph)

        self.manipulator = CupManipulatorTendon(config, enable_visualization=False)
        ctrl_parser = Parser(self.ctrl_plant)
        self.manipulator.load_urdf_to_plant(self.ctrl_plant, ctrl_parser)
        self.manipulator.weld_base_to_world(self.ctrl_plant, position=np.zeros(3))
        self.manipulator.add_joint_actuators(self.ctrl_plant)
        self.manipulator.set_joint_properties(self.ctrl_plant)

        # Passive springs (must match plant node)
        for jt_name in JOINT_NAMES:
            jt_cfg = config.joint_configs.get(jt_name)
            K = jt_cfg.stiffness if jt_cfg is not None else 0.0
            if K > 0.0:
                jt = self.manipulator.get_joint_by_name(self.ctrl_plant, jt_name)
                self.ctrl_plant.AddForceElement(
                    RevoluteSpring(jt, nominal_angle=0.0, stiffness=K)
                )

        self.manipulator.add_end_effector_frame(self.ctrl_plant)
        self.ctrl_plant.Finalize()
        self.ctrl_context = self.ctrl_plant.CreateDefaultContext()

        # Patch zero-mass bodies in the dynamics plant too
        for idx in self.ctrl_plant.GetBodyIndices(self.manipulator.model_instance):
            body = self.ctrl_plant.get_body(idx)
            if body.default_mass() < 1e-6:
                body.SetSpatialInertiaInBodyFrame(self.ctrl_context, _M_PATCH)

        # Pre-allocated forces object (reused each tick)
        self._forces = MultibodyForces(self.ctrl_plant)

        # Pulley radius for cable tension decomposition
        self._r_p = self.manipulator.PULLEY_RADIUS

        # Velocity-vector indices for user-order joints in Drake's nv
        j1 = self.manipulator.get_joint_by_name(self.ctrl_plant, JOINT_NAMES[0])
        j2 = self.manipulator.get_joint_by_name(self.ctrl_plant, JOINT_NAMES[1])
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # Link lengths for analytical Jacobian
        self._L1, self._L2 = self.manipulator.ik.get_link_lengths(self.ctrl_plant)

        # ── Gains ────────────────────────────────────────────────────────
        self._Kp = float(kp)
        self._Kd = float(kd)
        self._tau_max = float(tau_max)
        self.trajectory = trajectory
        self._t0 = None  # set on first sensor callback

        # ── Sensor state (protected by lock) ─────────────────────────────
        self._q = np.zeros(NUM_JOINTS)
        self._qd = np.zeros(NUM_JOINTS)
        self._sensor_received = False
        self._lock = threading.Lock()

        # ── ROS 2 plumbing ───────────────────────────────────────────────
        self.torque_pub = self.create_publisher(
            Float64MultiArray, "/torque_command", 10
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self.create_timer(1.0 / control_rate_hz, self._control_loop)

        wn = np.sqrt(self._Kp) if self._Kp > 0 else 0.0
        zeta = self._Kd / (2.0 * wn) if wn > 0 else 0.0
        self.get_logger().info(
            f"Computed-torque controller ready  |  "
            f"Kp={kp}  Kd={kd}  τ_max={tau_max}  "
            f"ωn={wn:.1f} rad/s  ζ={zeta:.2f}  "
            f"rate={control_rate_hz} Hz  "
            f"r_p={self._r_p*1e3:.1f} mm"
        )

    # ── callbacks ────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) != NUM_JOINTS:
            return
        with self._lock:
            first = not self._sensor_received
            # Map from message order to internal arrays
            for i, name in enumerate(msg.name):
                if name in JOINT_NAMES:
                    idx = JOINT_NAMES.index(name)
                    self._q[idx] = msg.position[i]
                    self._qd[idx] = msg.velocity[i]
            self._sensor_received = True
        if first:
            self.get_logger().info("Received first /joint_states")

    # ── control loop ─────────────────────────────────────────────────────
    def _control_loop(self):
        with self._lock:
            if not self._sensor_received:
                return
            q = self._q.copy()
            qd = self._qd.copy()

        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0

        # Desired trajectory  →  q_des, q̇_des, q̈_des
        q_des, qd_des, qdd_des = self.trajectory.evaluate(t)

        # ── Set state in dynamics-only plant ─────────────────────────────
        self.manipulator.set_positions_user_order(
            self.ctrl_plant, self.ctrl_context, q
        )
        self.manipulator.set_velocities_user_order(
            self.ctrl_plant, self.ctrl_context, qd
        )

        # ── Computed-torque law ──────────────────────────────────────────
        # a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_des − q̇)
        a_des_user = qdd_des + self._Kp * (q_des - q) + self._Kd * (qd_des - qd)

        # Map user-order acceleration to Drake velocity-vector order
        nv = self.ctrl_plant.num_velocities()
        a_des_drake = np.zeros(nv)
        a_des_drake[self._v_idx[0]] = a_des_user[0]
        a_des_drake[self._v_idx[1]] = a_des_user[1]

        # τ = M(q)·a_des + h(q, q̇)  via CalcInverseDynamics
        self._forces.SetZero()
        tau_full = self.ctrl_plant.CalcInverseDynamics(
            self.ctrl_context, a_des_drake, self._forces
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

        # ── Publish ──────────────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = tau_clip.tolist()
        self.torque_pub.publish(msg)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Computed-Torque Controller — ROS 2 Node"
    )
    ap.add_argument("--kp", type=float, default=10000.0,
                    help="Position gain Kp [s⁻²]  (default: 10000 → ωn=100 rad/s)")
    ap.add_argument("--kd", type=float, default=400.0,
                    help="Velocity gain Kd [s⁻¹]  (default: 400 → ζ=2)")
    ap.add_argument("--tau-max", type=float, default=10.0,
                    help="Torque saturation [Nm]  (default: 10)")
    ap.add_argument("--rate", type=float, default=500.0,
                    help="Control rate (Hz)")
    ap.add_argument("--mode", choices=["hold", "min-jerk"], default="min-jerk",
                    help="Trajectory mode")
    ap.add_argument("--q-start", type=str, default="0,0",
                    help="Start angles in degrees (comma-separated)")
    ap.add_argument("--q-goal", type=str, default="60,-120",
                    help="Goal angles in degrees (comma-separated)")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="Trajectory duration (s)")
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

    rclpy.init()
    node = ComputedTorqueControllerNode(
        kp=args.kp,
        kd=args.kd,
        tau_max=args.tau_max,
        trajectory=traj,
        control_rate_hz=args.rate,
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
