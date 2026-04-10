"""
robots/cup_manipulator_tendon_isaac.py
--------------------------------------
Isaac Sim counterpart of cup_manipulator_tendon.py (PyDrake).

Follows the proven URDF → USD → Articulation pattern from test_combined_urdf.py:
  1. Convert URDF to USD file via URDFParseAndImportFile
  2. Add USD to stage via add_reference_to_stage
  3. Wrap as Articulation (isaacsim.core.experimental.prims)

Method names mirror the PyDrake CupManipulatorTendon class so that
scene-building code is engine-agnostic.

IMPORTANT: SimulationApp MUST be created before importing this module.
"""

import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from termcolor import colored

# Isaac Sim imports — safe only after SimulationApp() has been created.
import omni.usd
import omni.kit.commands
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.experimental.prims import Articulation
from pxr import UsdGeom, UsdPhysics, Gf, Usd, Vt

# ArticulationView provides batched dynamics queries (M, C, g) needed for
# computed-torque control and future GPU-parallel RL.
from omni.isaac.core.articulations import ArticulationView

# Re-use the same config types from PyDrake side
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.robot.robot_types import ManipulatorConfig, JointConfig, Pose


# ============================================================================
# URDF → USD CONVERSION (same pattern as test_combined_urdf.py)
# ============================================================================

def import_urdf_to_usd(urdf_path: str, usd_path: str) -> bool:
    """Convert URDF to USD format.

    Follows the proven pattern from test_combined_urdf.py.
    """
    _urdf.acquire_urdf_interface()

    config = _urdf.ImportConfig()
    config.convex_decomp = False
    config.fix_base = True
    config.make_default_prim = True
    config.self_collision = False
    config.distance_scale = 1.0
    config.merge_fixed_joints = False

    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=config,
        dest_path=usd_path,
    )

    if result:
        print(colored(f"✓ Converted URDF → USD: {usd_path}", 'green'))
        return True
    else:
        print(colored(f"✗ Failed to parse URDF: {urdf_path}", 'red'))
        return False


# ============================================================================
# ISAAC SIM COUNTERPART OF CupManipulatorTendon
# ============================================================================

class CupManipulatorTendonIsaac:
    """Isaac Sim counterpart of CupManipulatorTendon (PyDrake).

    Loads the same URDF (manipulator_cable_obj.urdf) into Isaac Sim via
    the URDF → USD → Articulation pipeline and provides the same method
    interface as the PyDrake class.

    Joint names (same as PyDrake):
        JT1_NAME = "link1_base"   (q1: base → link1/pulley)
        JT2_NAME = "link2_link1"  (q2: link1/pulley → link2)
    """

    # ── Constants matching PyDrake CupManipulatorTendon ──────────────────────
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

    # Pulley geometry — HTD 5M 60T belt: 60 teeth × 5 mm pitch / 2π
    PULLEY_RADIUS = 60 * 0.005 / (2 * np.pi)  # ≈ 0.047746 m

    @property
    def r_p(self) -> float:
        """Pulley pitch radius [m]."""
        return self.PULLEY_RADIUS

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        self.config = config
        self.name = config.name
        self.enable_visualization = enable_visualization

        self.joint_names: List[str]    = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.dof_names: List[str]      = []

        # Isaac Sim handles
        self.robot: Optional[Articulation] = None
        self.prim_path: str = "/World/CupManipulatorTendon"
        self._usd_path: Optional[str] = None
        self._stage = None

        # Joint index map: joint_name → index in robot.dof_names
        self._joint_index: Dict[str, int] = {}

        # EE prim for world-frame queries
        self._ee_xformable = None
        self._ee_prim_path: Optional[str] = None

        # ArticulationView for dynamics queries (M, C, g) — set by initialize_dynamics_view()
        self._art_view = None
        self._av_jt1_idx: Optional[int] = None
        self._av_jt2_idx: Optional[int] = None

    # ── URDF → USD → Articulation ───────────────────────────────────────────

    @staticmethod
    def _resolve_package_paths(urdf_path: str) -> str:
        """Resolve ``package://assets/`` → absolute paths and fix USD-invalid names.

        Isaac Sim's URDF importer sometimes fails to resolve package://
        relative to the URDF file directory.  Also, USD prim names must
        not start with a digit (SdfPath restriction), so we prefix any
        mesh filenames/material names that start with a digit with ``m_``.
        """
        import re
        urdf_abs = str(Path(urdf_path).resolve())
        assets_dir = str(Path(urdf_abs).parent / "assets")
        with open(urdf_abs) as f:
            xml_text = f.read()
        # Resolve package:// to absolute paths
        resolved = xml_text.replace("package://assets/", assets_dir + "/")
        # Prefix digit-starting filenames in mesh tags: 623zz.obj → m_623zz.obj
        # We only rename the filename portion, and create a symlink for the asset
        digit_meshes = set(re.findall(r'filename="[^"]*?/(\d[^"/]*)"', resolved))
        for mesh_name in digit_meshes:
            new_name = "m_" + mesh_name
            # Create symlink so the file can be found
            src = os.path.join(assets_dir, mesh_name)
            dst = os.path.join(assets_dir, new_name)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
            resolved = resolved.replace(mesh_name, new_name)
        # Prefix digit-starting material/name attributes: name="623zz..." → name="m_623zz..."
        resolved = re.sub(
            r'name="(\d)',
            r'name="m_\1',
            resolved,
        )
        # Write resolved URDF
        resolved_path = str(Path(urdf_abs).with_name(
            Path(urdf_abs).stem + "_resolved.urdf"
        ))
        with open(resolved_path, "w") as f:
            f.write(resolved)
        return resolved_path

    def prepare_usd(self) -> None:
        """Convert URDF → USD file on disk.

        MUST be called BEFORE World() is created — this matches the
        test_combined_urdf.py pattern where import_urdf_to_usd() runs
        before World() so the stage context is clean.
        """
        urdf_path = str(Path(self.config.urdf_path).absolute())
        print(f"\n[Isaac Sim] Converting URDF → USD: {urdf_path}")

        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        # Resolve package:// paths to absolute paths for the URDF importer
        resolved_urdf = self._resolve_package_paths(urdf_path)

        usd_path = str(Path(urdf_path).with_suffix('.usd'))
        if os.path.exists(usd_path):
            os.remove(usd_path)  # fresh conversion

        if not import_urdf_to_usd(resolved_urdf, usd_path):
            raise RuntimeError(f"URDF→USD conversion failed for {urdf_path}")
        self._usd_path = usd_path
        # Bake URDF colors into the USD sublayer files on disk before
        # add_reference_to_stage reads them into the live stage.
        self.apply_urdf_colors()

    def load_urdf(self, world=None) -> None:
        """Add the pre-converted USD to the stage.

        Counterpart of ``load_urdf_to_plant(plant, parser)`` in Drake.
        Call prepare_usd() BEFORE World() creation, then call this method
        AFTER World() is created (but before world.reset()).
        Follows the pattern from test_combined_urdf.py.
        """
        if self._usd_path is None:
            raise RuntimeError(
                "prepare_usd() must be called before load_urdf(). "
                "prepare_usd() must run before World() is created."
            )

        self._stage = omni.usd.get_context().get_stage()

        # Add USD to stage (same as test_combined_urdf.py)
        add_reference_to_stage(usd_path=self._usd_path, prim_path=self.prim_path)
        print(colored(
            f"✓ Added robot to scene at {self.prim_path}",
            'green'
        ))

    def apply_urdf_colors(self) -> None:
        """Write URDF <material><color> values into the USB sublayer files.

        The URDFParseAndImportFile importer creates raw Mesh geometry but
        drops all material/color definitions. The visual meshes are stored
        in USD sublayer files (configuration/manipulator_cable_obj_base.usd
        etc.) — NOT in the live Isaac Sim stage — so we must open those USD
        files directly with pxr, set primvars:displayColor on each Mesh
        prim, and save them before the stage loads the reference.
        """
        if self._usd_path is None:
            return

        urdf_path = str(Path(self.config.urdf_path).absolute())
        try:
            tree = ET.parse(urdf_path)
        except Exception as e:
            print(colored(f"[apply_urdf_colors] Could not parse URDF: {e}", 'yellow'))
            return

        # Build map: part_name (OBJ filename stem) → RGB
        # Duplicate part names in URDF (used on multiple links) are fine —
        # last occurrence wins, but colors are identical for same-named parts.
        color_map: Dict[str, Gf.Vec3f] = {}
        for visual in tree.iter('visual'):
            mesh_elem = visual.find('geometry/mesh')
            color_elem = visual.find('material/color')
            if mesh_elem is None or color_elem is None:
                continue
            filename = mesh_elem.get('filename', '')
            part_name = Path(filename).stem          # e.g. "base", "tutup_base_1"
            rgba_str = color_elem.get('rgba', '')
            try:
                r, g, b, _ = [float(v) for v in rgba_str.split()]
                color_map[part_name] = Gf.Vec3f(r, g, b)
            except ValueError:
                continue

        if not color_map:
            print(colored("[apply_urdf_colors] No colors found in URDF", 'yellow'))
            return

        # Find all USD sublayer files: the main USD + everything in configuration/.
        usd_dir = Path(self._usd_path).parent
        sublayer_files: list[Path] = list(usd_dir.glob("*.usd"))
        config_dir = usd_dir / "configuration"
        if config_dir.is_dir():
            sublayer_files += list(config_dir.glob("*.usd"))

        total_applied = 0
        for usd_file in sublayer_files:
            sub_stage = Usd.Stage.Open(str(usd_file))
            if sub_stage is None:
                continue
            applied = 0
            for prim in sub_stage.Traverse():
                if prim.GetTypeName() != 'Mesh':
                    continue
                # Path: /visuals/base_mate/tutup_base_1/World/mesh
                # part name is the 3rd-from-end component (index -3 relative to /mesh)
                path_parts = prim.GetPath().pathString.split('/')
                matched_color = None
                for component in reversed(path_parts):
                    if not component:
                        continue
                    if component in color_map:
                        matched_color = color_map[component]
                        break
                    # The URDF importer prefixes invalid prim names with "part_"
                    # (e.g. "623zz" → "part_623zz"), so strip it and try again.
                    if component.startswith('part_') and component[5:] in color_map:
                        matched_color = color_map[component[5:]]
                        break
                    # Isaac Sim appends _0, _1 etc. for duplicate prims; try base name
                    if '_' in component:
                        prefix, suffix = component.rsplit('_', 1)
                        if suffix.isdigit() and prefix in color_map:
                            matched_color = color_map[prefix]
                            break
                if matched_color is None:
                    continue
                gprim = UsdGeom.Gprim(prim)
                gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([matched_color]))
                applied += 1
            if applied > 0:
                sub_stage.Save()
                total_applied += applied

        print(colored(f"✓ Applied URDF colors to {total_applied} mesh prims across {len(sublayer_files)} USD files", 'green'))

    # ── Weld base ───────────────────────────────────────────────────────────

    def weld_base_to_world(
        self,
        position:    np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        """Set the base transform in world frame.

        The URDF importer with fix_base=True already creates a fixed joint.
        This adjusts the root prim's transform — same pattern as
        test_import_plate_dips_to_isaac_franka.py apply_transform().
        """
        prim = self._stage.GetPrimAtPath(self.prim_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        xformable.AddTranslateOp().Set(Gf.Vec3d(
            float(position[0]), float(position[1]), float(position[2])
        ))
        roll, pitch, yaw = (float(np.rad2deg(a)) for a in orientation)
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(roll, pitch, yaw))

        print(colored(
            f"✓ Welded '{self.BASE_LINK_NAME}' to world at "
            f"pos={position}, rpy_deg=({roll:.1f}, {pitch:.1f}, {yaw:.1f})",
            'green'
        ))

    # ── End-effector frame ──────────────────────────────────────────────────

    def add_end_effector_frame(self):
        """Create an Xform prim at the EE location on link2.

        Isaac Sim doesn't have Drake-style named frames, so we create a child
        Xform prim under the link2 prim at the known EE offset.
        """
        link2_prim_path = self._find_link_prim_path(self.LINK2_NAME)
        if link2_prim_path is None:
            print(colored(
                f"⚠️  Could not find link2 prim '{self.LINK2_NAME}' in stage",
                'yellow'
            ))
            return

        self._ee_prim_path = f"{link2_prim_path}/{self.EE_FRAME_NAME}"
        ee_prim = self._stage.GetPrimAtPath(self._ee_prim_path)
        if not ee_prim.IsValid():
            ee_prim = UsdGeom.Xform.Define(self._stage, self._ee_prim_path)
            xformable = UsdGeom.Xformable(ee_prim.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(
                float(self.EE_XYZ_LINK2[0]),
                float(self.EE_XYZ_LINK2[1]),
                float(self.EE_XYZ_LINK2[2]),
            ))
            print(colored(
                f"✓ EE frame '{self.EE_FRAME_NAME}' created at "
                f"offset={self.EE_XYZ_LINK2} under {link2_prim_path}",
                'green'
            ))
        else:
            print(colored(
                f"✓ EE frame '{self.EE_FRAME_NAME}' already exists",
                'green'
            ))

        self._ee_xformable = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(self._ee_prim_path)
        )

    def get_end_effector_frame(self):
        """Return the EE Xformable (counterpart of Drake's GetFrameByName)."""
        return self._ee_xformable

    # ── Joint properties ────────────────────────────────────────────────────

    def set_joint_properties(self):
        """Configure joint damping from config (counterpart of set_joint_properties)."""
        print(colored(f"\nSetting joint properties for '{self.name}':", 'yellow'))
        for jt_name in self.joint_names:
            if jt_name in self.config.joint_configs:
                cfg = self.config.joint_configs[jt_name]
                damping = cfg.damping
                jt_prim_path = self._find_joint_prim_path(jt_name)
                if jt_prim_path:
                    jt_prim = self._stage.GetPrimAtPath(jt_prim_path)
                    drive = UsdPhysics.DriveAPI.Get(jt_prim, "angular")
                    if drive:
                        drive.GetDampingAttr().Set(float(damping))
                print(colored(f"  ✓ {jt_name}: damping={damping}", 'cyan'))
        print(colored(f"✓ Joint properties configured", 'green'))

    def add_joint_actuators(self):
        """Ensure joints have DriveAPI applied (counterpart of add_joint_actuators).

        For effort-based (computed-torque) control:
        - drive type = "force"
        - stiffness = 0  (no position spring fighting our efforts)
        - maxForce = large  (URDF effort limit is often too low)
        """
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        for jt_name in self.joint_names:
            jt_prim_path = self._find_joint_prim_path(jt_name)
            if jt_prim_path:
                jt_prim = self._stage.GetPrimAtPath(jt_prim_path)
                if not UsdPhysics.DriveAPI.Get(jt_prim, "angular"):
                    UsdPhysics.DriveAPI.Apply(jt_prim, "angular")
                drive = UsdPhysics.DriveAPI.Get(jt_prim, "angular")
                drive.GetTypeAttr().Set("force")
                # Zero stiffness — no position spring for effort control
                drive.GetStiffnessAttr().Set(0.0)
                # Large maxForce — don't let URDF effort limit cap our torques
                drive.GetMaxForceAttr().Set(1e4)
        print(colored(
            f"✓ Actuators configured (stiffness=0, maxForce=1e4): "
            f"{self.ACT1_NAME}, {self.ACT2_NAME}",
            'green'
        ))

    # ── Initialization (post world.reset) ───────────────────────────────────

    def initialize_state(self):
        """Initialize after world.reset() — create Articulation and build DOF map.

        Counterpart of ``initialize_state(plant)`` in Drake.
        Must be called AFTER world.reset() — same as test_combined_urdf.py.
        """
        # Create Articulation (only valid after world.reset)
        self.robot = Articulation(self.prim_path)

        # Build joint name → index map
        num_dofs = self.robot.num_dofs
        dof_names = list(self.robot.dof_names) if self.robot.dof_names is not None else []
        self.dof_names = dof_names
        self._joint_index = {name: i for i, name in enumerate(dof_names)}

        print(colored(
            f"✓ Articulation '{self.name}' created with {num_dofs} DOFs",
            'green', attrs=['bold']
        ))
        for i, name in enumerate(dof_names):
            print(colored(f"  [{i}] {name}", 'cyan'))

    def initialize_dynamics_view(self, world, *, reset: bool = True) -> None:
        """Initialize ArticulationView for dynamics queries (M, C, g).

        Must be called AFTER world.reset() and initialize_state().
        The ArticulationView wraps the same prim but provides batched
        dynamics queries that the experimental Articulation API lacks.

        Parameters
        ----------
        world : omni.isaac.core.World
            The simulation world instance.
        reset : bool
            If True (default), call world.reset() and finalize immediately.
            Set False for multi-robot batching: call once per robot, then
            do a single world.reset() followed by finalize_dynamics_view()
            on each robot.
        """
        _view_name = self.prim_path.lstrip("/").replace("/", "_") + "_dyn_view"
        self._art_view = ArticulationView(
            prim_paths_expr=self.prim_path,
            name=_view_name,
        )
        world.scene.add(self._art_view)
        if reset:
            world.reset()  # re-initialize physics to register the view
            self.finalize_dynamics_view(world)

    def finalize_dynamics_view(self, world) -> None:
        """Finalize ArticulationView after a world.reset().

        Called automatically by initialize_dynamics_view(reset=True).
        For multi-robot batching, call manually after a single world.reset().
        """
        self._art_view.initialize(world.physics_sim_view)

        # Resolve DOF indices in the ArticulationView
        av_dof_names = list(self._art_view.dof_names)
        # Flatten if nested
        if av_dof_names and isinstance(av_dof_names[0], (list, tuple)):
            av_dof_names = list(av_dof_names[0])
        self._av_jt1_idx = None
        self._av_jt2_idx = None
        for i, name in enumerate(av_dof_names):
            if self.JT1_NAME in name:
                self._av_jt1_idx = i
            if self.JT2_NAME in name:
                self._av_jt2_idx = i

        if self._av_jt1_idx is None or self._av_jt2_idx is None:
            raise RuntimeError(
                f"Dynamics view: joints [{self.JT1_NAME}, {self.JT2_NAME}] "
                f"not found in ArticulationView DOFs: {av_dof_names}"
            )
        print(colored(
            f"✓ Dynamics ArticulationView initialized: "
            f"jt1_idx={self._av_jt1_idx}, jt2_idx={self._av_jt2_idx}",
            'green',
        ))

    # ── Dynamics Queries (for Computed Torque) ────────────────────────────

    def get_mass_matrix(self) -> np.ndarray:
        """Return 2×2 mass matrix M(q) for the two actuated joints.

        Uses ArticulationView.get_mass_matrices() which returns the full
        (num_envs, ndof, ndof) tensor, then extracts the 2×2 block.
        """
        M_full = self._art_view.get_mass_matrices()  # (num_envs, ndof, ndof)
        M = M_full[0]  # first environment
        idx = [self._av_jt1_idx, self._av_jt2_idx]
        return np.array(M[np.ix_(idx, idx)])

    def get_coriolis_and_centrifugal(self) -> np.ndarray:
        """Return C(q,q̇)·q̇ vector (2,) for the two actuated joints."""
        h_c_full = self._art_view.get_coriolis_and_centrifugal_forces()  # (num_envs, ndof)
        h_c = h_c_full[0]
        return np.array([h_c[self._av_jt1_idx], h_c[self._av_jt2_idx]])

    def get_generalized_gravity(self) -> np.ndarray:
        """Return g(q) gravity vector (2,) for the two actuated joints."""
        g_full = self._art_view.get_generalized_gravity_forces()  # (num_envs, ndof)
        g = g_full[0]
        return np.array([g[self._av_jt1_idx], g[self._av_jt2_idx]])

    def get_bias_forces(self) -> np.ndarray:
        """Return h(q, q̇) = C(q,q̇)·q̇ + g(q) — bias term for CT control."""
        return self.get_coriolis_and_centrifugal() + self.get_generalized_gravity()

    def set_initial_positions(self):
        """Set initial joint angles from config.

        Counterpart of ``set_initial_positions(plant, context)`` in Drake.
        Uses Articulation.set_dof_positions() (same as test_combined_urdf.py).
        """
        print(colored(f"\nSetting initial positions for '{self.name}':", 'yellow'))
        positions = np.zeros(len(self.dof_names))

        for jt_name in self.joint_names:
            if jt_name in self.config.joint_configs and jt_name in self._joint_index:
                angle = self.config.joint_configs[jt_name].position
                idx = self._joint_index[jt_name]
                positions[idx] = angle
                print(colored(
                    f"  ✓ {jt_name}: {np.rad2deg(angle):.2f}° ({angle:.4f} rad)",
                    'cyan'
                ))

        self.robot.set_dof_positions(positions)
        print(colored(f"✓ Initial positions set", 'green'))

    # ── Helper: read positions/velocities from Articulation ─────────────────

    def _get_dof_positions_np(self) -> np.ndarray:
        """Get DOF positions as a flat numpy array."""
        pos = self.robot.get_dof_positions()
        if pos is None:
            return np.zeros(len(self.dof_names))
        # Articulation returns warp array — convert to numpy
        return pos.numpy().flatten()

    def _get_dof_velocities_np(self) -> np.ndarray:
        """Get DOF velocities as a flat numpy array."""
        vel = self.robot.get_dof_velocities()
        if vel is None:
            return np.zeros(len(self.dof_names))
        return vel.numpy().flatten()

    # ── EE kinematics ───────────────────────────────────────────────────────

    def get_end_effector_position(self) -> np.ndarray:
        """World-frame EE position [x, y, z].

        Counterpart of ``get_end_effector_position(plant, context)`` in Drake.
        """
        if self._ee_xformable is not None:
            xform = self._ee_xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            t = xform.ExtractTranslation()
            return np.array([t[0], t[1], t[2]])
        return self._fk_2r()

    def CalcPosition(self) -> np.ndarray:
        """Alias for get_end_effector_position() — matches Drake API."""
        return self.get_end_effector_position()

    # ── State helpers ───────────────────────────────────────────────────────

    def get_state_from_plant(self) -> np.ndarray:
        """Return [q1, q2, q1_dot, q2_dot] in user order."""
        q = self.get_positions_user_order()
        v = self.get_velocities_user_order()
        return np.concatenate([q, v])

    def set_state_in_plant(self, user_state: np.ndarray):
        """Set [q1, q2, q1_dot, q2_dot] from user order."""
        q1, q2, q1_dot, q2_dot = user_state
        self.set_jt([self.JT1_NAME, self.JT2_NAME], [q1, q2])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], [q1_dot, q2_dot])

    def get_positions_user_order(self) -> np.ndarray:
        """Return [q1, q2] in user (logical) order."""
        return np.array(self.get_jt([self.JT1_NAME, self.JT2_NAME]))

    def set_positions_user_order(self, user_positions):
        """Set joint positions. Accepts dict {name: angle} or array [q1, q2]."""
        if isinstance(user_positions, dict):
            for jt_name, angle in user_positions.items():
                self.set_jt([jt_name], [angle])
        else:
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], [q1, q2])

    def get_velocities_user_order(self) -> np.ndarray:
        """Return [q1_dot, q2_dot] in user order."""
        return np.array(self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME]))

    def set_velocities_user_order(self, user_velocities):
        """Set joint velocities. Accepts dict {name: vel} or array [v1, v2]."""
        if isinstance(user_velocities, dict):
            for jt_name, vel in user_velocities.items():
                self.set_jt_velocity([jt_name], [vel])
        else:
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity(
                [self.JT1_NAME, self.JT2_NAME], [q1_dot, q2_dot]
            )

    def get_joint_positions(self) -> dict:
        """Return {joint_name: angle} for all DOFs."""
        positions = self._get_dof_positions_np()
        return {name: float(positions[i]) for name, i in self._joint_index.items()}

    def get_joint_velocities(self) -> dict:
        """Return {joint_name: velocity} for all DOFs."""
        velocities = self._get_dof_velocities_np()
        return {name: float(velocities[i]) for name, i in self._joint_index.items()}

    # ── Joint helpers ───────────────────────────────────────────────────────

    def get_joint_by_name(self, joint_name: str) -> int:
        """Return the DOF index for a joint name."""
        if joint_name not in self._joint_index:
            raise KeyError(
                f"Joint '{joint_name}' not found. "
                f"Available: {list(self._joint_index.keys())}"
            )
        return self._joint_index[joint_name]

    def get_jt(self, joint_name, default=0.0):
        """Get joint angle(s). Accepts a single name or list of names."""
        positions = self._get_dof_positions_np()
        if isinstance(joint_name, list):
            return np.array([
                float(positions[self._joint_index[n]])
                if n in self._joint_index else default
                for n in joint_name
            ])
        if joint_name in self._joint_index:
            return float(positions[self._joint_index[joint_name]])
        return default

    def set_jt(self, joint_name, angle):
        """Set joint angle(s). Accepts a single name or list of names."""
        positions = self._get_dof_positions_np()
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            for name, ang in zip(joint_name, angles):
                if name in self._joint_index:
                    positions[self._joint_index[name]] = float(ang)
        else:
            if joint_name in self._joint_index:
                positions[self._joint_index[joint_name]] = float(angle)
        self.robot.set_dof_positions(positions)

    def get_jt_velocity(self, joint_name, default=0.0):
        """Get joint velocity(s). Accepts a single name or list of names."""
        velocities = self._get_dof_velocities_np()
        if isinstance(joint_name, list):
            return np.array([
                float(velocities[self._joint_index[n]])
                if n in self._joint_index else default
                for n in joint_name
            ])
        if joint_name in self._joint_index:
            return float(velocities[self._joint_index[joint_name]])
        return default

    def set_jt_velocity(self, joint_name, velocity):
        """Set joint velocity(s). Accepts a single name or list of names."""
        velocities = self._get_dof_velocities_np()
        if isinstance(joint_name, list):
            vels = np.atleast_1d(velocity)
            for name, vel in zip(joint_name, vels):
                if name in self._joint_index:
                    velocities[self._joint_index[name]] = float(vel)
        else:
            if joint_name in self._joint_index:
                velocities[self._joint_index[joint_name]] = float(velocity)
        self.robot.set_dof_velocities(velocities)

    # ── Torque application ────────────────────────────────────────────────

    def set_joint_torques(self, torques: np.ndarray) -> None:
        """Apply joint torques [tau1, tau2] in user order (q1, q2).

        Uses the ArticulationView effort API for direct torque control.
        """
        if self._art_view is None:
            raise RuntimeError("Call initialize_dynamics_view() first.")
        # Build full-DOF effort vector (zeros for fixed/passive DOFs)
        n_dof = self._art_view.num_dof
        efforts = np.zeros((1, n_dof))
        efforts[0, self._av_jt1_idx] = torques[0]
        efforts[0, self._av_jt2_idx] = torques[1]
        self._art_view.set_joint_efforts(efforts)

    # ── Inverse kinematics (analytical 2R) ──────────────────────────────────

    def compute_ik_analytical(
        self,
        target_xy: np.ndarray,
        q_seed: np.ndarray,
        pos_tol: float = 1e-3,
    ) -> tuple:
        """Closed-form 2R planar IK.

        Matches ``compute_ik_analytical(plant, target_xy, q_seed)`` in Drake.
        Returns (q_solution, success).
        """
        L1, L2 = self._get_link_lengths()
        x, y = float(target_xy[0]), float(target_xy[1])
        d2 = x * x + y * y
        d  = math.sqrt(d2)

        if d > L1 + L2 - 1e-6 or d < abs(L1 - L2) + 1e-6:
            return q_seed.copy(), False

        cos_q2 = (d2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        sin_q2 = math.sqrt(1.0 - cos_q2**2)

        q2_a = math.atan2(sin_q2, cos_q2)
        q2_b = math.atan2(-sin_q2, cos_q2)
        q2 = q2_a if abs(q2_a - q_seed[1]) < abs(q2_b - q_seed[1]) else q2_b

        q1 = math.atan2(y, x) - math.atan2(
            L2 * math.sin(q2), L1 + L2 * math.cos(q2)
        )

        ee_x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
        ee_y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
        err = math.sqrt((ee_x - x) ** 2 + (ee_y - y) ** 2)

        return np.array([q1, q2]), err < pos_tol

    # ── Private helpers ─────────────────────────────────────────────────────

    def _get_link_lengths(self) -> tuple:
        """Return (L1, L2) in metres."""
        L1 = 0.335
        L2 = float(self.EE_XYZ_LINK2[0])  # 0.19
        return L1, L2

    def _fk_2r(self) -> np.ndarray:
        """Simple 2R forward kinematics for XY position (fallback)."""
        L1, L2 = self._get_link_lengths()
        q1 = self.get_jt(self.JT1_NAME)
        q2 = self.get_jt(self.JT2_NAME)
        x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
        y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
        z = float(self.EE_XYZ_LINK2[2])
        return np.array([x, y, z])

    def _find_link_prim_path(self, link_name: str) -> Optional[str]:
        """Walk the stage to find a prim whose name matches *link_name*."""
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(self.prim_path)):
            if prim.GetName() == link_name:
                return str(prim.GetPath())
        return None

    def _find_joint_prim_path(self, joint_name: str) -> Optional[str]:
        """Walk the stage to find a physics joint prim matching *joint_name*."""
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(self.prim_path)):
            if prim.GetName() == joint_name:
                return str(prim.GetPath())
        return None


# ============================================================================
# MODULE-LEVEL HELPER FUNCTIONS (pure NumPy, no class needed)
# ============================================================================

def solve_2r_ik(
    L1: float, L2: float,
    target_xy: np.ndarray,
    q_seed: np.ndarray,
) -> tuple:
    """Analytical 2R planar IK. Returns (q_solution, success)."""
    x, y = float(target_xy[0]), float(target_xy[1])
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 - 1e-6 or d < abs(L1 - L2) + 1e-6:
        return q_seed.copy(), False
    cos_q2 = (d2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = math.sqrt(1.0 - cos_q2**2)
    q2_a = math.atan2(sin_q2, cos_q2)
    q2_b = math.atan2(-sin_q2, cos_q2)
    q2 = q2_a if abs(q2_a - q_seed[1]) < abs(q2_b - q_seed[1]) else q2_b
    q1 = math.atan2(y, x) - math.atan2(
        L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2]), True


def forward_kinematics_2r(
    L1: float, L2: float, q1: float, q2: float,
) -> np.ndarray:
    """2R planar FK → [x, y]."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    return np.array([x, y])


def analytical_jacobian_2r(
    L1: float, L2: float, q1: float, q2: float,
) -> np.ndarray:
    """2×2 analytical Jacobian for 2R planar arm."""
    s1 = math.sin(q1)
    c1 = math.cos(q1)
    s12 = math.sin(q1 + q2)
    c12 = math.cos(q1 + q2)
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12],
    ])


# ============================================================================
# FACTORY — mirrors create_cable_manipulator_config() from PyDrake side
# ============================================================================

def create_cable_manipulator_config(
    urdf_path: str = "model_using_onshape_to_robot/manipulator_cable_isaac/manipulator_cable_obj.urdf",
    joint_angles: Optional[dict] = None,
    damping:       tuple = (0.1, 0.1),
    stiffness:     tuple = (0.0, 0.0),
    friction:      tuple = (0.0, 0.0),
    tilt_roll_deg:  float = 0.0,
    tilt_pitch_deg: float = 0.0,
) -> ManipulatorConfig:
    """Factory for the cable manipulator config (same signature as PyDrake)."""
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
        tilt_roll_deg=tilt_roll_deg,
        tilt_pitch_deg=tilt_pitch_deg,
        package_map={"assets": urdf_dir + "/assets/"},
    )
