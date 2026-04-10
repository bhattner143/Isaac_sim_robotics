"""
robots/cup_manipulator_cable.py

Cable-driven (tendon) 2-DOF manipulator for Drake -- basic version
(no motor model, no IK solver).

Used by interactive cable visualization and DrakeCablePlant.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    Parser,
    RigidTransform,
    RevoluteJoint,
    PrismaticJoint,
)
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.math import RollPitchYaw
from termcolor import colored

from robots.cup_manipulator import RobotBase
from configs.robot.robot_types import ManipulatorConfig, JointConfig, Pose
from cable.pulley import PulleyBase, _parse_urdf_part_origins
from cable.routing import CableRig


# ═══════════════════════════════════════════════════════════════════════════════
# CUP MANIPULATOR — cable-driven 2-DOF robot adapter for Drake
# ═══════════════════════════════════════════════════════════════════════════════
# Lives here (not in robots/cup_manipulator_tendon.py) so that test_drive_pulley
# is fully self-contained.  robots/cup_manipulator_tendon.py re-exports this class
# as CupManipulatorTendon for backward compatibility with other scripts.

class CupManipulator(RobotBase):
    """Cable-driven (tendon) 2-DOF manipulator for Drake.

    Wraps manipulator_cable.urdf which uses a belt/pulley transmission.
    Joint names:
        JT1_NAME = "link1_base"   (q1)
        JT2_NAME = "link2_link1"  (q2)
    """

    JT1_NAME  = "link1_base"
    JT2_NAME  = "link2_link1"
    ACT1_NAME = f"tau_{JT1_NAME}"
    ACT2_NAME = f"tau_{JT2_NAME}"

    BASE_LINK_NAME = "base_mate"
    LINK2_NAME     = "link2_tendon"

    EE_XYZ_LINK2  = np.array([0.19, 0.0, 0.0515])
    EE_RPY_LINK2  = np.array([0.0, 0.0, 0.0])
    EE_FRAME_NAME = "tendon_ee"
    EE_OFFSET     = EE_XYZ_LINK2

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names: List[str]    = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization      = enable_visualization
        self.rig                       = None  # CableRig — set via init_cable_rig()

    # ── URDF loading ────────────────────────────────────────────────────────

    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        model_instance = super().load_urdf_to_plant(plant, parser)
        self.JT1_NAME    = "link1_base"
        self.JT2_NAME    = "link2_link1"
        self.ACT1_NAME   = f"tau_{self.JT1_NAME}"
        self.ACT2_NAME   = f"tau_{self.JT2_NAME}"
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        print(colored(
            f"✓ CupManipulator: joints confirmed: [{self.JT1_NAME}, {self.JT2_NAME}]",
            'green'
        ))
        return model_instance

    # ── End-effector frame ──────────────────────────────────────────────────

    def add_end_effector_frame(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")
        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
        X_L2_EE    = RigidTransform(RollPitchYaw(self.EE_RPY_LINK2), self.EE_XYZ_LINK2)
        try:
            return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
        except Exception:
            pass
        return plant.AddFrame(
            FixedOffsetFrame(
                self.EE_FRAME_NAME,
                link2_body.body_frame(),
                X_L2_EE,
                self.model_instance,
            )
        )

    def get_end_effector_frame(self, plant: MultibodyPlant):
        return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)

    # ── Joint actuators ─────────────────────────────────────────────────────

    def add_joint_actuators(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        plant.AddJointActuator(self.ACT1_NAME, jt1)
        plant.AddJointActuator(self.ACT2_NAME, jt2)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ Added actuators: {self.ACT1_NAME}, {self.ACT2_NAME}", 'green'))

    # ── EE kinematics ───────────────────────────────────────────────────────

    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        ee_frame = self.get_end_effector_frame(plant)
        X_WE     = plant.CalcRelativeTransform(context, plant.world_frame(), ee_frame)
        return X_WE.translation()

    def CalcPosition(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_end_effector_position(plant, context)

    # ── State helpers ───────────────────────────────────────────────────────

    def get_state_from_plant(self, plant: MultibodyPlant, context) -> np.ndarray:
        return plant.GetPositionsAndVelocities(context, self.model_instance)

    def set_state_in_plant(self, plant: MultibodyPlant, context, user_state: np.ndarray):
        q1, q2, q1_dot, q2_dot = user_state
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_positions_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return np.array(self.get_jt([self.JT1_NAME, self.JT2_NAME], plant, context))

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions):
        if isinstance(user_positions, dict):
            for joint_name, angle in user_positions.items():
                self.set_jt([joint_name], plant, context, [angle])
        else:
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities):
        if isinstance(user_velocities, dict):
            for joint_name, velocity in user_velocities.items():
                self.set_jt_velocity([joint_name], plant, context, [velocity])
        else:
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_joint_positions(self, plant: MultibodyPlant, context) -> dict:
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions

    def get_joint_velocities(self, plant: MultibodyPlant, context) -> dict:
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities

    # ── Joint helpers ───────────────────────────────────────────────────────

    def get_joint_by_name(self, plant: MultibodyPlant, joint_name: str):
        return plant.GetJointByName(joint_name, self.model_instance)

    def get_jt(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angle(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angle(context)

    def set_jt(self, joint_name, plant: MultibodyPlant, context, angle):
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            for name, ang in zip(joint_name, angles):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            self.get_joint_by_name(plant, joint_name).set_angle(context, float(angle))

    def get_jt_velocity(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angular_rate(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angular_rate(context)

    def set_jt_velocity(self, joint_name, plant: MultibodyPlant, context, velocity):
        if isinstance(joint_name, list):
            velocities = np.atleast_1d(velocity)
            for name, vel in zip(joint_name, velocities):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            self.get_joint_by_name(plant, joint_name).set_angular_rate(context, float(velocity))

    # ── Inverse kinematics ──────────────────────────────────────────────────

    def solve_initial_pose_via_ik(
        self,
        plant,
        target_xy,
        q_seed,
        pos_tol: float = 1e-3,
        verbose: bool = False,
        ee_frame_name: Optional[str] = None,
        target_z: Optional[float] = None,
    ):
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve

        target_xy = np.asarray(target_xy).reshape(2,)
        q_seed    = np.asarray(q_seed).reshape(2,)
        ik         = InverseKinematics(plant)
        ik_context = ik.context()
        self.set_positions_user_order(plant, ik_context, q_seed)
        world = plant.world_frame()

        if ee_frame_name is None:
            ee_frame_name = self.EE_FRAME_NAME
        try:
            ee_frame = plant.GetFrameByName(ee_frame_name, self.model_instance)
            p_BQ = np.zeros(3)
        except Exception:
            link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
            ee_frame   = link2_body.body_frame()
            p_BQ       = np.asarray(self.EE_XYZ_LINK2).reshape(3,)

        ee_pos_seed = plant.CalcPointsPositions(
            ik_context, ee_frame, p_BQ.reshape(3, 1), world
        ).ravel()
        z_target = target_z if target_z is not None else ee_pos_seed[2]

        if verbose:
            print(f"  Seed EE: ({ee_pos_seed[0]:.3f}, {ee_pos_seed[1]:.3f}, {ee_pos_seed[2]:.3f})")
            print(f"  Target:  ({target_xy[0]:.3f}, {target_xy[1]:.3f}, {z_target:.3f})")
            print(f"  Tol:     ±{pos_tol:.6f} m")

        lower = np.array([target_xy[0], target_xy[1], z_target]) - pos_tol
        upper = np.array([target_xy[0], target_xy[1], z_target]) + pos_tol
        ik.AddPositionConstraint(
            frameB=ee_frame, p_BQ=p_BQ,
            frameA=world, p_AQ_lower=lower, p_AQ_upper=upper,
        )
        prog   = ik.prog()
        q_vars = ik.q()
        q0_all = plant.GetPositions(ik_context)
        prog.AddQuadraticErrorCost(1000.0 * np.eye(len(q0_all)), q0_all, q_vars)
        prog.SetInitialGuess(q_vars, q0_all)
        result = Solve(prog)

        if verbose:
            print(f"  Solver: {result.get_solver_id().name()}, success={result.is_success()}")
        if not result.is_success():
            return q_seed, False

        q_sol_all    = result.GetSolution(q_vars)
        temp_context = plant.CreateDefaultContext()
        plant.SetPositions(temp_context, q_sol_all)
        q_sol_user   = plant.GetPositions(temp_context, self.model_instance)
        return np.asarray(q_sol_user), True

    # ── Cable rig ───────────────────────────────────────────────────────────

    def init_cable_rig(self, urdf_path: str = None, assets_dir: str = None,
                       springs_enabled: bool = True) -> None:
        """Initialize the cable rig.  Call after the plant is built.

        Args:
            springs_enabled: If True, add compliant springs at End-point L/R.
        """
        if urdf_path is None:
            urdf_path = self.config.urdf_path
        if assets_dir is None:
            assets_dir = str(Path(urdf_path).parent / "assets")
        PulleyBase._urdf_origins = _parse_urdf_part_origins(urdf_path)
        PulleyBase.assets_dir    = assets_dir
        self.rig = CableRig(springs_enabled=springs_enabled)

    def compute_tangents(self, plant, plant_context) -> None:
        """Recompute all cable tangent contacts at the current joint configuration."""
        if self.rig is None:
            raise RuntimeError("init_cable_rig() must be called before compute_tangents()")
        self.rig.compute_tangents(plant, plant_context, self)

    # ── Weld base ───────────────────────────────────────────────────────────

    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position:    np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")
        base_body = plant.GetBodyByName(self.BASE_LINK_NAME, self.model_instance)
        X_WB      = RigidTransform(RollPitchYaw(orientation), position)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), X_WB)
        print(colored(
            f"✓ Welded '{self.BASE_LINK_NAME}' to world at pos={position}, rpy={orientation}",
            'green'
        ))


def create_cable_manipulator_config(
    urdf_path: str = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
    joint_angles: Optional[dict] = None,
    damping:    tuple = (0.1, 0.1),
    stiffness:  tuple = (0.0, 0.0),
    friction:   tuple = (0.0, 0.0),
) -> ManipulatorConfig:
    """Factory for the cable (tendon) manipulator configuration."""
    urdf_dir    = str(Path(urdf_path).parent)
    joint_names = ["link1_base", "link2_link1"]
    if joint_angles is None:
        joint_angles = {n: 0.0 for n in joint_names}
    joint_configs = {}
    for i, name in enumerate(joint_names):
        joint_configs[name] = JointConfig(
            position=joint_angles.get(name, 0.0),
            damping=damping[i],
            stiffness=stiffness[i],
            friction=friction[i],
        )
    return ManipulatorConfig(
        name="manipulator_cable",
        urdf_path=urdf_path,
        joint_configs=joint_configs,
        base_pose=Pose(),
        package_map={"assets": urdf_dir + "/assets/"},
    )


# ──────────────────────────────────────────────────────────────────────────────
def build_plant(manipulator_config):
    """Build DiagramBuilder + MultibodyPlant containing only the manipulator."""
    builder     = DiagramBuilder()
    plant       = MultibodyPlant(time_step=0.0)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulator(manipulator_config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(plant, position=np.zeros(3), orientation=np.zeros(3))
    manipulator.add_joint_actuators(plant)
    manipulator.add_end_effector_frame(plant)
    plant.Finalize()

    builder.AddSystem(plant)
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    return builder, plant, scene_graph, manipulator



# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    import matplotlib.pyplot as plt
    from pydrake.all import MeshcatVisualizer, StartMeshcat, Simulator
    from project_utils.viz_cables import (
        print_cable_routing_points,
        draw_cables,
        visualize_cable_routing_top_view,
        visualize_cable_routing_3d,
    )
    ap = argparse.ArgumentParser(description="Cable routing visualization.")
    ap.add_argument("--no-springs", action="store_true",
                    help="Disable endpoint springs (default: springs enabled)")
    args = ap.parse_args()
    springs_enabled = not args.no_springs

    # ── Configuration ─────────────────────────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path="model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=(0.1, 0.1),
    )

    # ── Meshcat ───────────────────────────────────────────────────────────────
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    # ── Plant ─────────────────────────────────────────────────────────────────
    builder, plant, scene_graph, manipulator = build_plant(config)

    # ── Cable rig — owned by manipulator, mirrors physical assembly ───────────
    manipulator.init_cable_rig(springs_enabled=springs_enabled)
    rig = manipulator.rig  # local alias for draw_cables / viz helpers

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)

    # ── Home pose ─────────────────────────────────────────────────────────────
    current_q = np.array([0.0, 0.0])
    manipulator.set_positions_user_order(plant, plant_ctx, {
        "link1_base":  current_q[0],
        "link2_link1": current_q[1],
    })
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    diagram.ForcedPublish(context)
    manipulator.compute_tangents(plant, plant_ctx)  # FK-based, all pairs
    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)    # straight segments + wrap arcs
    print_cable_routing_points(plant, plant_ctx, manipulator, rig)

    # Figure 1 — top view (XY)
    _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, 0.0, 0.0, rig)
    plt.show(block=False)
    plt.pause(0.05)

    # # Figure 2 — 3-D view with OBJ meshes
    # _viz_fig, _ = visualize_cable_routing_3d(
    #     plant, plant_ctx, manipulator, PulleyBase.assets_dir, 0.0, 0.0
    # )
    # plt.show(block=False)
    # plt.pause(0.1)
    _viz_fig = None  # created on first interactive update

    ee = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored("Cable route: drive_pulley → 623zz (A) → 623zz_2 (B, other side) → pulley_big", "yellow"))
    print(colored(f"Home:  q1=0°  q2=0°  →  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m\n", "cyan"))
    print(colored("Enter joint angles in degrees  (e.g.  30  -15)  or Ctrl+C to exit.\n", "yellow"))

    # ── Interactive loop ───────────────────────────────────────────────────────
    try:
        while True:
            raw = input(colored("q1  q2 [deg]: ", "cyan")).strip()
            if not raw:
                continue
            try:
                parts = raw.split()
                if len(parts) != 2:
                    print(colored("  ✗ Expected exactly two values: q1 q2", "red"))
                    continue
                q1_deg, q2_deg = float(parts[0]), float(parts[1])
                current_q = np.deg2rad([q1_deg, q2_deg])

                manipulator.set_positions_user_order(plant, plant_ctx, {
                    "link1_base":  current_q[0],
                    "link2_link1": current_q[1],
                })
                plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

                # Update robot geometry
                diagram.ForcedPublish(context)

                # Recompute all tangents in world frame (q1 and q2 may have changed)
                manipulator.compute_tangents(plant, plant_ctx)

                # Recompute and redraw cable at new pose
                draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
                # Update Figure 1 (top view) and Figure 2 (3-D)
                plt.close(_top_fig)
                _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                if _viz_fig is not None:
                    plt.close(_viz_fig)
                _viz_fig, _ = visualize_cable_routing_3d(plant, plant_ctx, manipulator, PulleyBase.assets_dir, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                ee = manipulator.get_end_effector_position(plant, plant_ctx)
                print(colored(
                    f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                    f"→  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m",
                    "green",
                ))
            except ValueError:
                print(colored("  ✗ Invalid numbers. Enter two floats: q1 q2", "red"))
    except KeyboardInterrupt:
        print(colored("\n✓ Stopped.", "green"))


# ============================================================================
# DRAKE CABLE PLANT — headless FK wrapper for cable tangent computation
# ============================================================================



if __name__ == "__main__":
    main()
