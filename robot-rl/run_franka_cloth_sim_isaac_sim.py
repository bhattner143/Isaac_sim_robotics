#!/usr/bin/env python3
###########################################################################
# Franka Cloth Manipulation — Newton Physics + Isaac Sim Rendering
#
# Runs the same Newton VBD cloth + Featherstone robot physics from
# run_franka_cloth_sim.py, but renders in the Isaac Sim GUI instead of
# Newton's GL/Viser viewers.
#
# Architecture:
#   Newton (VBD + Featherstone) — physics engine (cm scale)
#   Isaac Sim (PhysX disabled) — rendering only (m scale)
#
# Each frame:
#   1. Newton steps physics (cloth, robot)
#   2. Cloth particle positions → UsdGeom.Mesh vertices (cm→m)
#   3. Robot body transforms → Articulation joint positions (cm→m)
#
# Usage:
#   # Activate env with Isaac Sim 5.1 sourced + Newton installed
#   source /path/to/isaacsim/setup_conda_env.sh
#   conda activate env_isaacsim
#
#   python run_franka_cloth_sim_isaac_sim.py                    # native GUI
#   python run_franka_cloth_sim_isaac_sim.py --render websocket # WebRTC
#   python run_franka_cloth_sim_isaac_sim.py --render headless  # no display
#   python run_franka_cloth_sim_isaac_sim.py --num-frames 5000
###########################################################################

from __future__ import annotations

import sys
import os

# ── PRE-PARSE --render BEFORE SimulationApp ──────────────────────────────
_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "native"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}, got '{_render_mode}'")
            sys.exit(1)
        break

_num_frames = 3850
for _i, _arg in enumerate(sys.argv):
    if _arg == "--num-frames" and _i + 1 < len(sys.argv):
        _num_frames = int(sys.argv[_i + 1])
        break

os.environ.setdefault("CARB_LOG_LEVEL", "error")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Import warp BEFORE SimulationApp ─────────────────────────────────────
# Isaac Sim 5.1 bundles Warp 1.8.2, but Newton requires Warp ≥1.12.1
# (which has DeviceLike). Import from conda env first so it's cached in
# sys.modules before Isaac Sim adds its older bundled version to sys.path.
import warp as wp
_warp_conda_path = wp.__path__[:]  # save conda warp's __path__

# ── Isaac Sim startup (MUST be after warp, before other omni imports) ────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1920,
    "height": 1080,
    "hide_ui": False,
})

if _render_mode == "websocket":
    from isaacsim.core.utils.extensions import enable_extension
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/livestream/proto", "websocket")
    enable_extension("omni.kit.livestream.webrtc")

# ── Fix Warp path after SimulationApp ────────────────────────────────────
# omni.warp.core-1.8.2 extension injects Isaac Sim's bundled warp into
# warp.__path__ during startup. Restore conda's warp path so submodules
# (warp.fem, warp.sparse etc.) load from the correct 1.12.1 version.
wp.__path__[:] = _warp_conda_path
# Also purge any warp submodules that Isaac Sim may have partially loaded
# from the old version during extension startup.
import sys as _sys
_bad_warp_mods = [k for k in _sys.modules if k.startswith("warp.") and "isaacsim" in str(getattr(_sys.modules[k], "__file__", ""))]
for _mod in _bad_warp_mods:
    del _sys.modules[_mod]

# ── Post-SimulationApp imports ───────────────────────────────────────────
import argparse
import numpy as np
import omni.usd
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf, Usd, Vt

# warp already imported above (before SimulationApp) to avoid version conflict
import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.solvers import SolverFeatherstone, SolverVBD


# ─── Warp kernels (identical to run_franka_cloth_sim.py) ─────────────────

@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(
        pos_diff[0], pos_diff[1], pos_diff[2],
        ang_diff[0], ang_diff[1], ang_diff[2],
    )


# ─── Newton Physics Scene ───────────────────────────────────────────────

class NewtonPhysics:
    """Encapsulates Newton VBD+Featherstone physics (cm scale)."""

    def __init__(self):
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viz_scale = 0.01  # cm → m

        # contact params (cm scale)
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2
        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2
        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5
        self.self_contact_friction = 0.25

        # elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6
        self.bending_ke = 5
        self.bending_kd = 1e-2

        self.scene = ModelBuilder(gravity=-981.0)

        # Robot
        franka = ModelBuilder()
        self._create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count
        self.dof_q_per_world = franka.joint_coord_count
        self.dof_qd_per_world = franka.joint_dof_count

        # Table (cm scale)
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(self.table_pos_cm, wp.quat_identity()),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # Cloth (T-shirt)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        self.cloth_mesh_points = shirt_mesh.vertices
        self.cloth_mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in self.cloth_mesh_points]

        self.scene.add_cloth_mesh(
            vertices=vertices,
            indices=self.cloth_mesh_indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 70.0, 30.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=1.0,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )
        self.scene.color()
        self.scene.add_ground_plane()

        # Finalize model
        self.model = self.scene.finalize(requires_grad=False)

        # Material properties
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device)

        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        # Collision pipeline
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Solvers
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self._setup_control()

        self.model.edge_rest_angle.zero_()
        self.cloth_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.soft_contact_ke,
        )

        # Gravity arrays for toggling during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # FK eval
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # CUDA graph capture
        self._capture()

    def _create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, -10.0), wp.quat_identity()),
            floating=False,
            scale=100,  # m → cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close = 0.1
        clamp_open = 0.8

        self.robot_key_poses = np.array([
            [2.5, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, 26.0, -60.0, 26.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, 12.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [3, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, 15.0, -33.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, 15.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [3, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, -18.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [3, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_open],
            [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [3, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, 0.0, -20.0, 30.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_close],
            [2, 0.0, -20.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1.5, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close],
            [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_open],
            [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open],
        ], dtype=np.float32)

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def _setup_control(self):
        self.control = self.model.control()
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            return wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_qd: wp.array[wp.spatial_vector], body_out: wp.array[float]):
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def _capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _compute_body_jacobian(self, joint_q, joint_qd):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        out_dim = 6

        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(self.compute_body_out_kernel, 1,
                      inputs=[self.temp_state_for_jacobian.body_qd],
                      outputs=[self.body_out])

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim:(i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def _generate_control(self):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        wp.launch(
            compute_ee_delta, dim=1,
            inputs=[
                self.state_0.body_q, self.endeffector_offset,
                self.endeffector_id, self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self._compute_body_jacobian(self.state_0.joint_q, self.state_0.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = self.state_0.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null

        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    def step(self):
        """Advance one frame of Newton physics."""
        self._generate_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def _simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def get_particle_positions_m(self) -> np.ndarray:
        """Return cloth particle positions in metres."""
        pts_cm = self.state_0.particle_q.numpy()
        return pts_cm * self.viz_scale

    def get_body_transforms_m(self) -> np.ndarray:
        """Return body transforms with positions in metres (7-vectors: px,py,pz, qx,qy,qz,qw)."""
        xforms = self.state_0.body_q.numpy()  # shape (N, 7)
        xforms_m = xforms.copy()
        xforms_m[:, :3] *= self.viz_scale
        return xforms_m

    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions (radians)."""
        return self.state_0.joint_q.numpy()


# ─── Isaac Sim Scene Builder ────────────────────────────────────────────

def setup_lighting(stage):
    """Add dome light and distant light for good visibility."""
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)

    dist = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist.CreateIntensityAttr(3000.0)
    dist.CreateAngleAttr(0.53)
    xform = UsdGeom.Xformable(dist)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))


def create_table_prim(stage, pos_m, half_extents_m):
    """Create a visual-only box for the table."""
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)  # unit cube, scaled below
    sx, sy, sz = half_extents_m
    UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3d(sx * 2, sy * 2, sz * 2))
    UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(*pos_m))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
    return table


def create_ground_prim(stage):
    """Create a large ground plane."""
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    s = 5.0  # 5m half-extent
    ground.CreatePointsAttr([
        Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0),
        Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    ground.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.3)])
    return ground


def create_cloth_mesh_prim(stage, indices):
    """Create UsdGeom.Mesh for the cloth — topology set once, points updated per frame."""
    cloth = UsdGeom.Mesh.Define(stage, "/World/Cloth")
    num_tris = len(indices) // 3
    cloth.CreateFaceVertexCountsAttr([3] * num_tris)
    cloth.CreateFaceVertexIndicesAttr(list(indices))
    cloth.CreateSubdivisionSchemeAttr("none")
    # Orange-ish cloth color
    cloth.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.5, 0.2)])
    # Double-sided so visible from both sides
    cloth.CreateDoubleSidedAttr(True)
    return cloth


def update_cloth_mesh(cloth_prim, points_m):
    """Update cloth mesh vertex positions (metres)."""
    vt_points = Vt.Vec3fArray.FromNumpy(points_m.astype(np.float32))
    cloth_prim.GetPointsAttr().Set(vt_points)


def load_franka_urdf(stage):
    """Load Franka URDF via Isaac Sim's URDF importer, return prim path."""
    from isaacsim.asset.importer.urdf import _urdf

    # Use the same URDF Newton downloads
    asset_path = newton.utils.download_asset("franka_emika_panda")
    urdf_path = str(asset_path / "urdf" / "fr3_franka_hand.urdf")

    config = _urdf.ImportConfig()
    config.convex_decomp = False
    config.fix_base = True
    config.make_default_prim = False
    config.self_collision = False
    config.distance_scale = 1.0
    # Don't create physics — Newton handles it
    config.create_physics_scene = False

    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=config,
        dest_path="",
    )
    print(f"[Isaac Sim] Franka loaded at: {prim_path}")
    return prim_path


def set_franka_root_transform(stage, prim_path, pos_m):
    """Set the Franka base transform in metres."""
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    # Clear existing ops and set translate
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*pos_m))


def update_robot_joints(stage, prim_path, joint_positions):
    """
    Update robot visual joint angles by setting revolute joint targets.
    joint_positions: numpy array of joint positions in radians from Newton.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    # Walk all descendant prims looking for revolute/prismatic joints
    for desc in Usd.PrimRange(prim):
        if desc.HasAPI(UsdGeom.XformableAPI) or desc.IsA(UsdGeom.Xform):
            continue


def sync_robot_body_transforms(stage, prim_path, newton_physics):
    """
    Sync Newton body transforms to Isaac Sim Xform prims.
    Uses the body names from Newton model to find matching prims.
    """
    body_transforms = newton_physics.get_body_transforms_m()
    model = newton_physics.model

    # Get the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    # Build a mapping: prim name → prim path for all Xform descendants
    prim_map = {}
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Xform) or desc.IsA(UsdGeom.Mesh):
            name = desc.GetName()
            prim_map[name] = desc

    for body_idx in range(model.body_count):
        body_name = None
        for name, idx in model.body_name.items():
            if idx == body_idx:
                body_name = name
                break
        if body_name is None:
            continue

        # Try to find matching prim
        target_prim = prim_map.get(body_name)
        if target_prim is None:
            # Try without prefix
            short_name = body_name.split("/")[-1] if "/" in body_name else body_name
            target_prim = prim_map.get(short_name)
        if target_prim is None:
            continue

        xf = body_transforms[body_idx]
        pos = Gf.Vec3d(float(xf[0]), float(xf[1]), float(xf[2]))
        quat = Gf.Quatd(float(xf[6]), float(xf[3]), float(xf[4]), float(xf[5]))  # w, x, y, z

        xformable = UsdGeom.Xformable(target_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(pos)
        xformable.AddOrientOp().Set(Gf.Quatf(quat))


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"[Isaac Sim] Render mode: {_render_mode}")
    print(f"[Isaac Sim] Num frames: {_num_frames}")
    print()

    # ── 1. Initialize Newton physics ─────────────────────────────────────
    print("[Newton] Building physics scene...")
    physics = NewtonPhysics()
    print(f"[Newton] Bodies: {physics.model.body_count}, "
          f"Particles: {physics.model.particle_count}, "
          f"Joints: {physics.model.joint_count}")

    # ── 2. Set up Isaac Sim world + stage ────────────────────────────────
    world = World(stage_units_in_meters=1.0)

    stage = omni.usd.get_context().get_stage()

    # Y-up → Z-up (Newton uses Z-up)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Lighting
    setup_lighting(stage)

    # Ground plane
    create_ground_prim(stage)

    # Table (convert cm → m)
    table_pos_m = (
        float(physics.table_pos_cm[0]) * physics.viz_scale,
        float(physics.table_pos_cm[1]) * physics.viz_scale,
        float(physics.table_pos_cm[2]) * physics.viz_scale,
    )
    table_half_m = (
        physics.table_hx_cm * physics.viz_scale,
        physics.table_hy_cm * physics.viz_scale,
        physics.table_hz_cm * physics.viz_scale,
    )
    create_table_prim(stage, table_pos_m, table_half_m)

    # Cloth mesh (topology from Newton)
    cloth_prim = create_cloth_mesh_prim(stage, physics.cloth_mesh_indices)
    # Set initial positions
    init_pts = physics.get_particle_positions_m()
    update_cloth_mesh(cloth_prim, init_pts)

    # Franka robot
    franka_prim_path = load_franka_urdf(stage)

    # Robot base position (Newton uses (-50, -50, -10) cm → (-0.5, -0.5, -0.1) m)
    franka_base_m = (-0.5, -0.5, -0.1)
    set_franka_root_transform(stage, franka_prim_path, franka_base_m)

    # ── 3. Initialize world ──────────────────────────────────────────────
    world.reset()

    # Warm up renderer
    for _ in range(5):
        world.step(render=True)
        simulation_app.update()

    print("[Isaac Sim] Scene ready. Starting simulation loop...")
    print(f"[Isaac Sim] Total key-pose duration: {physics.robot_key_poses_time[-1]:.1f}s "
          f"({int(physics.robot_key_poses_time[-1] * physics.fps)} frames)")

    # ── 4. Main simulation loop ──────────────────────────────────────────
    frame = 0
    try:
        while frame < _num_frames and simulation_app.is_running():
            # Step Newton physics
            physics.step()

            # Sync cloth particles → Isaac Sim mesh
            cloth_pts = physics.get_particle_positions_m()
            update_cloth_mesh(cloth_prim, cloth_pts)

            # Sync robot body transforms → Isaac Sim
            sync_robot_body_transforms(stage, franka_prim_path, physics)

            # Render
            world.step(render=True)
            simulation_app.update()

            frame += 1
            if frame % 60 == 0:
                print(f"  Frame {frame}/{_num_frames}  "
                      f"sim_time={physics.sim_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[Isaac Sim] Interrupted by user.")

    print(f"[Isaac Sim] Done. Rendered {frame} frames.")

    # Keep window open for inspection
    if _render_mode == "native":
        print("[Isaac Sim] Press Ctrl+C to exit...")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()


if __name__ == "__main__":
    main()
