#!/usr/bin/env python3
"""
franka_cloth_newton.py
======================
Franka robot manipulating a cloth (t-shirt) — governed entirely by Newton physics.

Uses Newton's solvers:
  - SolverVBD for cloth simulation (implicit, supports self-collision)
  - SolverFeatherstone for Franka articulation dynamics

Visualization via Newton's built-in GL viewer (no Isaac Sim dependency required,
but runs fine inside conda env_isaacsim).

Based on: newton/newton/examples/cloth/example_cloth_franka.py

Usage:
    # Activate env with Newton + Warp installed
    conda activate env_isaacsim

    # GUI mode (default)
    python franka_cloth_newton.py

    # Headless mode
    python franka_cloth_newton.py --headless

    # Custom frames / viewer
    python franka_cloth_newton.py --num-frames 5000 --viewer gl

    # Use custom t-shirt OBJ
    python franka_cloth_newton.py --obj-file dataset_mesh_gat/t_shirt_l3/CAD/t_shirt_l3.obj
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.solvers import SolverFeatherstone, SolverVBD

# ─────────────────────────────────────────────────────────────────────────────
# Warp kernels
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Cloth mesh loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_tshirt_obj() -> str | None:
    """Search workspace for the t_shirt_l3 OBJ file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "dataset_mesh_gat", "t_shirt_l3", "CAD", "t_shirt_l3.obj"),
        os.path.join(script_dir, "..", "dataset_mesh_gat", "t_shirt_l3", "CAD", "t_shirt_l3.obj"),
    ]
    for p in candidates:
        rp = os.path.realpath(p)
        if os.path.isfile(rp):
            return rp
    return None


def load_cloth_from_obj(filepath: str, cm_scale: float = 100.0):
    """Load OBJ, return vertices (list[wp.vec3]) and face indices (list[int]) in cm scale."""
    import trimesh
    mesh = trimesh.load(filepath, process=False, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32) * cm_scale

    # Centre on bounding-box midpoint
    bbox_mid = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts -= bbox_mid

    # Rotate +90° about X (OBJ shirts are often X-Z spread)
    rot_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    verts = verts @ rot_x.T

    vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    faces = np.array(mesh.faces, dtype=np.int32)
    indices = faces.ravel().tolist()
    return vertices, indices


def load_cloth_from_usd(usd_path: str):
    """Load the bundled Newton shirt USD, return vertices + indices."""
    usd_stage = Usd.Stage.Open(usd_path)
    usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
    shirt_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
    indices = shirt_mesh.indices
    return vertices, indices


# ─────────────────────────────────────────────────────────────────────────────
# Main Example class (follows Newton examples pattern)
# ─────────────────────────────────────────────────────────────────────────────

class FrankaClothNewton:
    """Franka robot + cloth manipulation using Newton VBD + Featherstone solvers."""

    def __init__(self, viewer, args):
        self.viewer = viewer

        # ── Simulation parameters (centimetre scale for VBD stability) ───
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Visualize in metres, simulate in cm
        self.viz_scale = 0.01

        # Contact parameters (cm scale)
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

        # Cloth elasticity (cm scale)
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6
        self.bending_ke = 5
        self.bending_kd = 1e-2

        # ── Build scene ──────────────────────────────────────────────────
        self.scene = ModelBuilder(gravity=-981.0)  # cm/s²

        # Robot
        self._add_franka()

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

        # Cloth
        self._add_cloth(args)

        # Ground plane
        self.scene.add_ground_plane()

        # ── Finalize model ───────────────────────────────────────────────
        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering (render manually at metre scale)
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Metre-scale table viz data
        self.table_viz_xform = wp.array(
            [wp.transform(
                (
                    float(self.table_pos_cm[0]) * self.viz_scale,
                    float(self.table_pos_cm[1]) * self.viz_scale,
                    float(self.table_pos_cm[2]) * self.viz_scale,
                ),
                wp.quat_identity(),
            )],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        # Contact material properties
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

        # ── Solver initialisation ────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        # Collision pipeline
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Robot solver (Featherstone)
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self._setup_control()

        # Cloth solver (VBD)
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

        # ── Viewer setup ─────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Visualization state (metre scale)
        self.viz_state = self.model.state()

        # Pre-compute scaled shapes for viz
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale viewer's cached shape instance data
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # Gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # Ensure FK evaluation
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Graph capture for GPU acceleration
        self._capture()

    # ── Scene construction ───────────────────────────────────────────────

    def _add_franka(self):
        """Add Franka Panda robot via URDF (downloads if needed)."""
        franka = ModelBuilder()
        asset_path = newton.utils.download_asset("franka_emika_panda")

        franka.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, -10.0), wp.quat_identity()),
            floating=False,
            scale=100,  # URDF metres → cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        franka.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count
        self.dof_q_per_world = franka.joint_coord_count
        self.dof_qd_per_world = franka.joint_dof_count

        # Key poses: [duration, x, y, z, qw, qx, qy, qz, gripper_activation]
        clamp_close = 0.1
        clamp_open = 0.8
        self.robot_key_poses = np.array([
            # Top left: approach, grasp, lift, drag, release
            [2.5,  31.0, -60.0, 23.0, 1, 0, 0, 0, clamp_open],
            [2.0,  31.0, -60.0, 23.0, 1, 0, 0, 0, clamp_close],
            [2.0,  26.0, -60.0, 26.0, 1, 0, 0, 0, clamp_close],
            [2.0,  12.0, -60.0, 31.0, 1, 0, 0, 0, clamp_close],
            [3.0,  -6.0, -60.0, 31.0, 1, 0, 0, 0, clamp_close],
            [1.0,  -6.0, -60.0, 31.0, 1, 0, 0, 0, clamp_open],
            # Bottom left
            [2.0,  15.0, -33.0, 31.0, 1, 0, 0, 0, clamp_open],
            [3.0,  15.0, -33.0, 21.0, 1, 0, 0, 0, clamp_open],
            [3.0,  15.0, -33.0, 21.0, 1, 0, 0, 0, clamp_close],
            [2.0,  15.0, -33.0, 28.0, 1, 0, 0, 0, clamp_close],
            [3.0,  -2.0, -33.0, 28.0, 1, 0, 0, 0, clamp_close],
            [1.0,  -2.0, -33.0, 28.0, 1, 0, 0, 0, clamp_open],
            # Top right
            [2.0, -28.0, -60.0, 28.0, 1, 0, 0, 0, clamp_open],
            [2.0, -28.0, -60.0, 20.0, 1, 0, 0, 0, clamp_open],
            [2.0, -28.0, -60.0, 20.0, 1, 0, 0, 0, clamp_close],
            [2.0, -18.0, -60.0, 31.0, 1, 0, 0, 0, clamp_close],
            [3.0,   5.0, -60.0, 31.0, 1, 0, 0, 0, clamp_close],
            [1.0,   5.0, -60.0, 31.0, 1, 0, 0, 0, clamp_open],
            # Bottom right
            [3.0, -18.0, -30.0, 20.5, 1, 0, 0, 0, clamp_open],
            [3.0, -18.0, -30.0, 20.5, 1, 0, 0, 0, clamp_close],
            [2.0,  -3.0, -30.0, 31.0, 1, 0, 0, 0, clamp_close],
            [3.0,  -3.0, -30.0, 31.0, 1, 0, 0, 0, clamp_close],
            [2.0,  -3.0, -30.0, 31.0, 1, 0, 0, 0, clamp_open],
            # Bottom centre: pick-lift-move-release
            [2.0,   0.0, -20.0, 30.0, 1, 0, 0, 0, clamp_open],
            [2.0,   0.0, -20.0, 19.5, 1, 0, 0, 0, clamp_open],
            [2.0,   0.0, -20.0, 19.5, 1, 0, 0, 0, clamp_close],
            [2.0,   0.0, -20.0, 35.0, 1, 0, 0, 0, clamp_close],
            [1.0,   0.0, -30.0, 35.0, 1, 0, 0, 0, clamp_close],
            [1.5,   0.0, -30.0, 35.0, 1, 0, 0, 0, clamp_close],
            [1.5,   0.0, -40.0, 35.0, 1, 0, 0, 0, clamp_close],
            [1.5,   0.0, -40.0, 35.0, 1, 0, 0, 0, clamp_open],
            [2.0, -28.0, -60.0, 28.0, 1, 0, 0, 0, clamp_open],
        ], dtype=np.float32)

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        self.endeffector_id = franka.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def _add_cloth(self, args):
        """Add cloth mesh (custom OBJ or bundled Newton shirt USD)."""
        obj_file = getattr(args, "obj_file", None)

        if obj_file and os.path.isfile(obj_file):
            print(f"Loading cloth from OBJ: {obj_file}")
            vertices, indices = load_cloth_from_obj(obj_file)
        else:
            if obj_file:
                print(f"OBJ not found: {obj_file}")
            # Try workspace t-shirt
            tshirt = _find_tshirt_obj()
            if tshirt:
                print(f"Loading cloth from OBJ: {tshirt}")
                vertices, indices = load_cloth_from_obj(tshirt)
            else:
                # Fall back to Newton's bundled shirt
                usd_path = newton.examples.get_asset("unisex_shirt.usd")
                print(f"Loading cloth from bundled USD: {usd_path}")
                vertices, indices = load_cloth_from_usd(usd_path)

        self.scene.add_cloth_mesh(
            vertices=vertices,
            indices=indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 70.0, 30.0),  # drop position (cm)
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

    # ── Control ──────────────────────────────────────────────────────────

    def _setup_control(self):
        """Prepare IK Jacobian computation for end-effector velocity control."""
        self.control = self.model.control()
        out_dim = 6
        in_dim = self.model.joint_dof_count

        self.Jacobian_one_hots = [
            wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            for i in range(out_dim)
        ]

        endeffector_id = self.endeffector_id
        endeffector_offset = self.endeffector_offset

        @wp.kernel
        def compute_body_out(body_qd: wp.array[wp.spatial_vector], body_out: wp.array[float]):
            mv = transform_twist(wp.static(endeffector_offset), body_qd[wp.static(endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def _compute_body_jacobian(self, joint_q, joint_qd):
        """Compute 6×n end-effector velocity Jacobian w.r.t. joint velocities."""
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

    def _generate_control(self, state_in: State):
        """Compute target joint velocities to follow the key-pose trajectory."""
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self._compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        # Null-space projection to bias toward initial pose
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J
        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        delta_q = J_inv @ delta_target + N @ (1.0 * (q_des - q))

        # Gripper finger control
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    # ── Simulation ───────────────────────────────────────────────────────

    def _capture(self):
        """CUDA graph capture for faster simulation."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate(self):
        """One frame of coupled robot + cloth simulation."""
        self.cloth_solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Apply viewer forces (e.g. interactive dragging)
            self.viewer.apply_forces(self.state_0)

            # Robot step (Featherstone) — disable particles temporarily
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0

            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # Cloth step (VBD)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        """Advance one frame."""
        self._generate_control(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render current state (particle + body positions scaled to metres)."""
        if self.viewer is None:
            return

        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap to metre-scale shape data for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()

        # Restore sim-scale shape data
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    parser.add_argument(
        "--obj-file",
        type=str,
        default=None,
        help="Path to t-shirt OBJ file. Uses bundled Newton shirt if omitted.",
    )
    viewer, args = newton.examples.init(parser)

    example = FrankaClothNewton(viewer, args)
    newton.examples.run(example, args)
