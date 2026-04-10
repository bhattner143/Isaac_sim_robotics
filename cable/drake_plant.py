"""
cable/drake_plant.py

Headless Drake FK wrapper for cable tangent computation.

Used by Isaac Sim scripts and RL scripts to compute cable waypoints
without rendering.
"""

import numpy as np

from robots.cup_manipulator_cable import create_cable_manipulator_config, build_plant


class DrakeCablePlant:
    """Headless Drake plant used solely for cable FK / tangent computation.

    No rendering — just MultibodyPlant + CableRig.  Joint angles are
    synchronised from Isaac Sim (or any caller), then compute_tangents()
    produces world-frame cable waypoints that can be drawn as USD prims.

    Usage::

        from cable import DrakeCablePlant
        dc = DrakeCablePlant(drake_urdf, q1=q1_rad, q2=q2_rad)
        dc.update(new_q1, new_q2)
        for route, pts in dc.get_cable_world_points():
            ...
    """

    def __init__(self, drake_urdf: str, q1: float = 0.0, q2: float = 0.0,
                 springs_enabled: bool = True):
        config = create_cable_manipulator_config(
            urdf_path=drake_urdf,
            joint_angles={"link1_base": q1, "link2_link1": q2},
        )
        self.builder, self.plant, self._sg, self.manipulator = build_plant(config)
        self.manipulator.init_cable_rig(urdf_path=drake_urdf,
                                        springs_enabled=springs_enabled)
        self.rig = self.manipulator.rig
        self.diagram = self.builder.Build()
        self._root_ctx = self.diagram.CreateDefaultContext()
        self.plant_ctx = self.plant.GetMyMutableContextFromRoot(self._root_ctx)
        self._set_angles(q1, q2)
        self.manipulator.compute_tangents(self.plant, self.plant_ctx)

    def _set_angles(self, q1: float, q2: float):
        self.manipulator.set_positions_user_order(
            self.plant, self.plant_ctx,
            {"link1_base": q1, "link2_link1": q2},
        )
        self.plant.SetVelocities(self.plant_ctx,
                                 np.zeros(self.plant.num_velocities()))

    def update(self, q1: float, q2: float):
        """Sync joint angles and recompute cable tangents."""
        self._set_angles(q1, q2)
        self.manipulator.compute_tangents(self.plant, self.plant_ctx)

    def get_cable_world_points(self):
        """Return list of (route, world_pts) for each cable."""
        return [
            (route, route.world_points(self.plant, self.plant_ctx, self.manipulator))
            for route in self.rig.routes
        ]

    def get_wrap_arcs(self, n_arc_pts: int = 24):
        """Return wrap-arc world-frame points for each pulley wrap segment.

        Yields ``(label, color_rgb, arc_pts_Nx3)`` tuples.
        """
        dp = self.rig.drive_pulley
        ir = self.rig.idler_r
        il = self.rig.idler_l
        pb = self.rig.pulley_big
        _G = (0.1, 0.85, 0.1)   # green
        _R = (0.9, 0.1,  0.1)   # red
        wraps = [
            (dp, dp.A_R, dp.B_R, "drive_green",  _G),
            (ir, ir.A_R, ir.B_R, "idlerR_green", _G),
            (pb, pb.A_L, pb.B_L, "big_green",    _G),
            (dp, dp.A_L, dp.B_L, "drive_red",    _R),
            (il, il.A_L, il.B_L, "idlerL_red",   _R),
            (pb, pb.A_R, pb.B_R, "big_red",      _R),
        ]

        def _Xw(body_name):
            body = self.plant.GetBodyByName(body_name, self.manipulator.model_instance)
            X = self.plant.CalcRelativeTransform(
                self.plant_ctx, self.plant.world_frame(), body.body_frame())
            return X.rotation().matrix(), X.translation()

        results = []
        for pulley, A_body, B_body, label, color in wraps:
            if A_body is None or B_body is None:
                continue
            R_wb, t_wb = _Xw(pulley.body_name)
            center_w = R_wb @ np.asarray(pulley.centroid) + t_wb
            shaft_w  = R_wb @ pulley.shaft_axis_body
            A_w = R_wb @ np.asarray(A_body) + t_wb
            B_w = R_wb @ np.asarray(B_body) + t_wb
            ax = shaft_w / np.linalg.norm(shaft_w)
            dA = A_w - center_w
            dA -= np.dot(dA, ax) * ax
            dA /= np.linalg.norm(dA)
            dB = B_w - center_w
            dB -= np.dot(dB, ax) * ax
            dB /= np.linalg.norm(dB)
            cos_ab = float(np.clip(np.dot(dA, dB), -1.0, 1.0))
            angle = np.sign(np.dot(np.cross(dA, dB), ax)) * np.arccos(cos_ab)
            ax_cross_dA = np.cross(ax, dA)
            arc_pts = np.array([
                center_w + pulley.radius * (dA * np.cos(th) + ax_cross_dA * np.sin(th))
                for th in np.linspace(0.0, angle, n_arc_pts)
            ])
            results.append((label, color, arc_pts))
        return results


