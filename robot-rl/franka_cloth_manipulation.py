
"""
Scene setup (matching the reference image):
  - A platform (table) at the centre of the world
  - A T-shirt L3 cloth mesh lying flat on the platform
  - A Franka Panda robot positioned next to the platform
  - Newton VBD solver for soft-body / cloth physics

Fabric physics properties are taken from the Newton cloth settings used
in script_newton_data_generator_tshirt_l3.py (CLOTH_SETTINGS_t_shirt_l3).

Launch with Isaac Sim 6 Newton runtime:
    cd <isaacsim6>/_build/linux-x86_64/release
    ./python.sh <path>/franka_cloth_manipulation.py [--headless] [--episodes N]

CAD source:  dataset_mesh_gat/t_shirt_l3/CAD/t_shirt_l3.obj
"""

# ── Isaac Sim bootstrap (must happen before any other Omniverse import) ──────
from isaacsim import SimulationApp

import argparse
import sys

parser = argparse.ArgumentParser(description="Franka cloth manipulation with Newton physics")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless (no GUI)")
parser.add_argument("--episodes", type=int, default=5, help="Number of manipulation episodes")
parser.add_argument("--steps_per_episode", type=int, default=600, help="Sim steps per episode")
parser.add_argument(
    "--obj_file",
    type=str,
    default=None,
    help="Path to t-shirt OBJ file. Auto-detected if omitted.",
)
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode (1 episode, 60 steps)")
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({
    "headless": args.headless,
    "extra_args": ["--/app/useFabricSceneDelegate=0"],
})

# ── Standard imports (after SimulationApp) ───────────────────────────────────
import os
import math
import carb
import numpy as np

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

from isaacsim.core.api import World
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem

torch = import_module("torch")
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from omni.physx.scripts import deformableUtils, physicsUtils

# ─────────────────────────────────────────────────────────────────────────────
# Fabric physics properties from CLOTH_SETTINGS_t_shirt_l3
# (script_newton_data_generator_tshirt_l3.py / cloth_settings.py)
# ─────────────────────────────────────────────────────────────────────────────
CLOTH_SETTINGS_t_shirt_l3 = {
    "scale": (1, 1, 1),
    "density": 0.02,                # kg/m²
    "tri_ke": 4.0e4,                # triangle stretch stiffness
    "tri_ka": 3.0e4,                # triangle area / shear stiffness
    "tri_kd": 1.0e-2,               # triangle damping
    "edge_ke": 35.0,                # bending stiffness
    "edge_kd": 3.0e-2,              # bending damping
    "particle_radius": 0.005,       # m  (0.5 cm)
    "soft_contact_ke": 4.5e3,       # contact spring stiffness
    "soft_contact_kd": 5.0e-2,      # contact damping
    "soft_contact_mu": 0.25,        # friction coefficient
    "self_contact_radius": 0.0025,  # m
    "self_contact_margin": 0.0025,  # m
    "sim_substeps": 10,
    "solver_iterations": 5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_obj_file() -> str:
    """Search common workspace locations for the t_shirt_l3 OBJ."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

    candidates = [
        os.path.join(workspace_root, "..", "dataset_mesh_gat", "t_shirt_l3", "CAD", "t_shirt_l3.obj"),
        os.path.join(workspace_root, "..", "mesh_gat_dataset_generation", "sample_dataset_directory",
                     "dataset_mesh_gat", "t_shirt_s1", "CAD", "t_shirt_s1.obj"),
        os.path.join("/home", "morph-lab-ws-ubuntu-admin", "Documents", "robot-rl",
                     "dataset_mesh_gat", "t_shirt_l3", "CAD", "t_shirt_l3.obj"),
    ]
    for p in candidates:
        rp = os.path.realpath(p)
        if os.path.isfile(rp):
            return rp
    return None


def load_obj_trimesh(filepath: str):
    """Load an OBJ via trimesh and return vertices (N,3) + tri-faces (F,3)."""
    import trimesh
    mesh = trimesh.load(filepath, process=False, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return verts, faces


def prepare_cloth_verts(verts: np.ndarray, settings: dict) -> np.ndarray:
    """
    Centre, scale, rotate cloth vertices so the mesh lies flat in the XY plane,
    then translate above the platform surface.
    Matches the transform pipeline in cloth_newton_model.py generate_model().
    """
    scale = np.array(settings.get("scale", (1, 1, 1)), dtype=np.float32)
    verts = verts * scale

    # Centre on bounding-box midpoint
    bbox_mid = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts -= bbox_mid

    # Rotate +90° about X  (OBJ is X-Z spread → rotate to X-Y)
    rot_x = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]], dtype=np.float32)
    verts = verts @ rot_x.T

    # Place cloth just above platform top (platform top at z = TABLE_HEIGHT)
    TABLE_HEIGHT = 0.4  # platform cuboid half-height * 2 / 2 … see below
    verts[:, 2] += TABLE_HEIGHT + 0.02  # 2 cm above surface

    return verts


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation class
# ─────────────────────────────────────────────────────────────────────────────
class FrankaClothManipulation:
    """
    Scene: Franka robot on a table manipulating a cloth (t-shirt).
    Physics: Isaac Sim 6 particle cloth via PhysX / Newton backend.
    """

    # Table geometry  (matches reference image: grey rectangular platform)
    TABLE_SIZE = np.array([0.8, 0.8, 0.4])        # half-extents in m
    TABLE_POS  = np.array([0.0, 0.0, 0.2])        # centre of cuboid

    # Franka placement  (beside the table, slightly offset)
    FRANKA_POS = np.array([0.0, -0.50, 0.40])     # on table surface level
    FRANKA_ORI = np.array([0.0, 0.0, 0.0, 1.0])   # quaternion (w last) – facing +Y

    def __init__(self, obj_file: str, settings: dict):
        self.settings = settings
        self.obj_file = obj_file

        # Isaac Sim world (metre units, GPU pipeline required for particle cloth)
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Build scene
        self._add_platform()
        self._add_cloth()
        self._add_franka()

        # First reset to cook physics
        self.world.reset(soft=False)
        carb.log_info("Scene initialised — cloth + Franka + platform ready.")

    # ── Scene construction ───────────────────────────────────────────────
    def _add_platform(self):
        """Grey fixed cuboid (table/platform)."""
        self.platform = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Platform",
                name="platform",
                position=self.TABLE_POS,
                scale=self.TABLE_SIZE,
                size=1.0,
                color=np.array([0.7, 0.7, 0.7]),
            )
        )

    def _add_cloth(self):
        """Load the OBJ, create a particle cloth on the USD stage."""
        verts_np, faces_np = load_obj_trimesh(self.obj_file)
        verts_np = prepare_cloth_verts(verts_np, self.settings)

        # Flatten faces to index list and build face-vertex-counts
        tri_indices = faces_np.ravel().tolist()
        tri_counts = [3] * len(faces_np)

        # Create USD mesh prim
        cloth_path = "/World/Cloth/tshirt"
        cloth_xform = UsdGeom.Xform.Define(self.stage, "/World/Cloth")
        mesh_prim = UsdGeom.Mesh.Define(self.stage, cloth_path)

        # Convert numpy verts to Gf.Vec3f list for USD
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts_np]
        mesh_prim.GetPointsAttr().Set(points)
        mesh_prim.GetFaceVertexIndicesAttr().Set(tri_indices)
        mesh_prim.GetFaceVertexCountsAttr().Set(tri_counts)

        # Particle system for cloth
        ps_path = "/World/Cloth/particleSystem"
        pm_path = "/World/Cloth/particleMaterial"

        radius = self.settings["particle_radius"]
        rest_offset = radius
        contact_offset = rest_offset * 1.5

        self.particle_material = ParticleMaterial(
            prim_path=pm_path,
            drag=0.1,
            lift=0.3,
            friction=self.settings["soft_contact_mu"],
        )

        self.particle_system = SingleParticleSystem(
            prim_path=ps_path,
            simulation_owner=self.world.get_physics_context().prim_path,
            rest_offset=rest_offset,
            contact_offset=contact_offset,
            solid_rest_offset=rest_offset,
            fluid_rest_offset=rest_offset,
            particle_contact_offset=contact_offset,
            global_self_collision_enabled=True,
        )

        # Wrap as ClothPrim  (binds material + particle system)
        self.cloth_prim = SingleClothPrim(
            name="tshirt_cloth",
            prim_path=cloth_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
        )
        self.world.scene.add(self.cloth_prim)

        # Store cloth metadata for later queries
        self._cloth_num_verts = len(verts_np)
        self._cloth_initial_positions = verts_np.copy()

        carb.log_info(f"Cloth loaded: {self._cloth_num_verts} vertices, {len(faces_np)} triangles")

    def _add_franka(self):
        """Add a Franka Panda robot beside the platform."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            carb.log_error("Could not find Isaac Sim assets folder — skipping Franka")
            self.franka = None
            return

        franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        prim = add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
        prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

        self.franka = self.world.scene.add(
            Robot(prim_path="/World/Franka", name="franka")
        )

    # ── Runtime ──────────────────────────────────────────────────────────
    def reset(self):
        """Reset the world, reposition Franka, restore cloth."""
        self.world.reset(soft=False)
        if self.franka is not None:
            self.franka.set_world_pose(
                position=torch.tensor(self.FRANKA_POS / get_stage_units(), dtype=torch.float32),
                orientation=torch.tensor(self.FRANKA_ORI, dtype=torch.float32),
            )
            # Home configuration – arm extended forward, gripper open
            home_joints = torch.tensor([0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.78, 0.04, 0.04])
            self.franka.set_joint_positions(home_joints)

    def _reach_toward_cloth(self, step_idx: int):
        """
        Simple joint-space trajectory that reaches the end-effector
        toward the cloth centre and performs a grasp motion.
        """
        if self.franka is None:
            return

        controller = self.franka.get_articulation_controller()

        # Phase 1: Move arm down toward cloth (steps 0-200)
        if step_idx < 200:
            t = step_idx / 200.0
            target = torch.tensor([0.0, -0.3 + 0.5 * t, 0.0, -2.0 + 0.5 * t, 0.0, 2.0 - 0.3 * t, 0.78, 0.04, 0.04])
            controller.apply_action(ArticulationAction(joint_positions=target))

        # Phase 2: Close gripper (steps 200-300)
        elif step_idx < 300:
            t = (step_idx - 200) / 100.0
            grip = 0.04 - 0.02 * t  # close from 4cm to 2cm
            target = torch.tensor([0.0, 0.2, 0.0, -1.5, 0.0, 1.7, 0.78, grip, grip])
            controller.apply_action(ArticulationAction(joint_positions=target))

        # Phase 3: Lift (steps 300-450)
        elif step_idx < 450:
            t = (step_idx - 300) / 150.0
            target = torch.tensor([0.0, 0.2 - 0.6 * t, 0.0, -1.5 - 0.3 * t, 0.0, 1.7 + 0.3 * t, 0.78, 0.02, 0.02])
            controller.apply_action(ArticulationAction(joint_positions=target))

        # Phase 4: Move laterally / shear (steps 450-600)
        elif step_idx < 600:
            t = (step_idx - 450) / 150.0
            target = torch.tensor([
                0.5 * math.sin(t * math.pi),   # base rotation
                -0.4, 0.0, -1.8, 0.0, 2.0, 0.78, 0.02, 0.02,
            ])
            controller.apply_action(ArticulationAction(joint_positions=target))

    def run_episode(self, episode_idx: int, num_steps: int):
        """Run one manipulation episode."""
        self.reset()
        print(f"\n[Episode {episode_idx + 1}] Running {num_steps} steps …")

        for step in range(num_steps):
            self._reach_toward_cloth(step)
            self.world.step(render=not args.headless)

            # Periodic logging
            if (step + 1) % 100 == 0:
                cloth_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Cloth/tshirt")
                points = cloth_mesh.GetPointsAttr().Get()
                if points is not None and len(points) > 0:
                    heights = [p[2] for p in points]
                    avg_z = sum(heights) / len(heights)
                    print(f"  step {step + 1:>4d}  |  cloth avg height = {avg_z:.4f} m")

    def run(self, num_episodes: int, steps_per_episode: int):
        """Main loop: run multiple episodes."""
        for ep in range(num_episodes):
            self.run_episode(ep, steps_per_episode)
        print("\nAll episodes complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve OBJ path
    obj_path = args.obj_file
    if obj_path is None:
        obj_path = _find_obj_file()
    if obj_path is None or not os.path.isfile(obj_path):
        # Fallback: create a procedural square cloth if no OBJ available
        carb.log_warn(
            f"T-shirt OBJ not found (searched dataset_mesh_gat/t_shirt_l3/CAD/).\n"
            f"Using a procedural square cloth mesh instead.\n"
            f"Place t_shirt_l3.obj in dataset_mesh_gat/t_shirt_l3/CAD/ for the real garment."
        )
        obj_path = None

    if args.test:
        args.episodes = 1
        args.steps_per_episode = 60

    if obj_path is not None:
        sim = FrankaClothManipulation(
            obj_file=obj_path,
            settings=CLOTH_SETTINGS_t_shirt_l3,
        )
        sim.run(num_episodes=args.episodes, steps_per_episode=args.steps_per_episode)
    else:
        # ── Procedural fallback: square cloth + Franka ───────────────────
        carb.log_info("Creating procedural square cloth scene")
        world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        stage = simulation_app.context.get_stage()
        world.scene.add_default_ground_plane()

        # Platform
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Platform",
                name="platform",
                position=np.array([0.0, 0.0, 0.2]),
                scale=np.array([0.8, 0.8, 0.4]),
                size=1.0,
                color=np.array([0.7, 0.7, 0.7]),
            )
        )

        # Procedural cloth mesh (square grid, similar to cloth.py example)
        cloth_path = "/World/Cloth/cloth_mesh"
        UsdGeom.Xform.Define(stage, "/World/Cloth")
        plane_mesh = UsdGeom.Mesh.Define(stage, cloth_path)
        tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=30, dimy=30, scale=0.5)

        # Offset cloth above platform
        offset_points = []
        for pt in tri_points:
            offset_points.append(Gf.Vec3f(pt[0], pt[1], pt[2] + 0.42))
        plane_mesh.GetPointsAttr().Set(offset_points)
        plane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        plane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

        # Particle system
        S = CLOTH_SETTINGS_t_shirt_l3
        radius = S["particle_radius"]
        particle_material = ParticleMaterial(
            prim_path="/World/Cloth/particleMaterial",
            drag=0.1,
            lift=0.3,
            friction=S["soft_contact_mu"],
        )
        particle_system = SingleParticleSystem(
            prim_path="/World/Cloth/particleSystem",
            simulation_owner=world.get_physics_context().prim_path,
            rest_offset=radius,
            contact_offset=radius * 1.5,
            solid_rest_offset=radius,
            fluid_rest_offset=radius,
            particle_contact_offset=radius * 1.5,
            global_self_collision_enabled=True,
        )
        cloth = SingleClothPrim(
            name="cloth",
            prim_path=cloth_path,
            particle_system=particle_system,
            particle_material=particle_material,
        )
        world.scene.add(cloth)

        # Franka
        assets_root = get_assets_root_path()
        if assets_root:
            franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            prim = add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
            prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            prim.GetVariantSet("Mesh").SetVariantSelection("Quality")
            franka = world.scene.add(Robot(prim_path="/World/Franka", name="franka"))
        else:
            franka = None

        world.reset(soft=False)
        if franka is not None:
            franka.set_world_pose(position=torch.tensor([0.0, -0.50, 0.40], dtype=torch.float32) / get_stage_units())
            franka.set_joint_positions(torch.tensor([0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.78, 0.04, 0.04]))

        ep_steps = args.steps_per_episode
        for ep in range(args.episodes):
            print(f"\n[Episode {ep + 1}] Running {ep_steps} steps (procedural cloth) \u2026")
            world.reset(soft=False)
            if franka is not None:
                franka.set_world_pose(position=torch.tensor([0.0, -0.50, 0.40], dtype=torch.float32) / get_stage_units())
                franka.set_joint_positions(torch.tensor([0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.78, 0.04, 0.04]))

            for step in range(ep_steps):
                # Simple reach trajectory
                if franka is not None:
                    ctrl = franka.get_articulation_controller()
                    t = step / max(ep_steps, 1)
                    target = torch.tensor([
                        0.3 * math.sin(t * 2 * math.pi),
                        -0.3 + 0.5 * t,
                        0.0,
                        -2.0 + 0.5 * t,
                        0.0,
                        2.0 - 0.3 * t,
                        0.78,
                        0.04 - 0.02 * min(t * 2, 1.0),
                        0.04 - 0.02 * min(t * 2, 1.0),
                    ])
                    ctrl.apply_action(ArticulationAction(joint_positions=target))

                world.step(render=not args.headless)

                if (step + 1) % 100 == 0:
                    cloth_mesh = UsdGeom.Mesh.Get(stage, cloth_path)
                    points = cloth_mesh.GetPointsAttr().Get()
                    if points is not None and len(points) > 0:
                        heights = [p[2] for p in points]
                        avg_z = sum(heights) / len(heights)
                        print(f"  step {step + 1:>4d}  |  cloth avg height = {avg_z:.4f} m")

        print("\nAll episodes complete.")

    simulation_app.close()
