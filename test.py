from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.simulation_manager import SimulationManager
import omni.usd
from omni.physx.scripts import utils as physx_utils


# Toggle CPU vs GPU
USE_GPU = False  # True for GPU PhysX, False for CPU

if USE_GPU:
    backend = "torch"
    device = "cuda:0"
else:
    backend = "numpy"
    device = None  # CPU


def make_vel(x, y, z):
    if USE_GPU:
        return torch.tensor([x, y, z], device=device, dtype=torch.float32)
    else:
        return np.array([x, y, z], dtype=np.float32)


def dist_between(pos_a, pos_b):
    if USE_GPU:
        if not isinstance(pos_a, torch.Tensor):
            pos_a = torch.tensor(pos_a, device=device, dtype=torch.float32)
        if not isinstance(pos_b, torch.Tensor):
            pos_b = torch.tensor(pos_b, device=device, dtype=torch.float32)
        return torch.linalg.norm(pos_a - pos_b).item()
    else:
        return float(np.linalg.norm(pos_a - pos_b))


def weld_rigid_bodies(prim_path_a: str, prim_path_b: str, joint_type: str = "Fixed"):
    stage = omni.usd.get_context().get_stage()

    prim_a = stage.GetPrimAtPath(prim_path_a)
    prim_b = stage.GetPrimAtPath(prim_path_b)

    if not prim_a or not prim_a.IsValid():
        raise RuntimeError(f"weld_rigid_bodies: Prim not found or invalid: {prim_path_a}")
    if not prim_b or not prim_b.IsValid():
        raise RuntimeError(f"weld_rigid_bodies: Prim not found or invalid: {prim_path_b}")

    joint_prim = physx_utils.createJoint(stage, joint_type, prim_a, prim_b)

    print(
        f"weld_rigid_bodies: Created {joint_type} joint at {joint_prim.GetPath()} "
        f"between '{prim_path_a}' and '{prim_path_b}'"
    )

    return joint_prim


# Scene setup
world = World(backend=backend, device=device)
physics_ctx = world.get_physics_context()
if USE_GPU:
    physics_ctx.enable_gpu_dynamics(True)
    physics_ctx.enable_fabric(True)
    print("[config] Using GPU PhysX pipeline (torch/cuda, Fabric ON)")
else:
    physics_ctx.enable_gpu_dynamics(False)
    physics_ctx.enable_fabric(False)
    print("[config] Using CPU PhysX pipeline (numpy, Fabric OFF)")

world.scene.add_default_ground_plane()

big_cube_path = "/World/big_cube"
small_cube_path = "/World/small_cube"

# Big cube: moves toward the small cube
big_scale = np.array([0.4, 0.4, 0.4], dtype=np.float32)
small_scale = np.array([0.2, 0.2, 0.2], dtype=np.float32)

big_cube = world.scene.add(
    DynamicCuboid(
        prim_path=big_cube_path,
        name="big_cube",
        position=np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        scale=big_scale,
        size=1.0,
        color=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
)

# Small cube: initially stationary
small_cube = world.scene.add(
    DynamicCuboid(
        prim_path=small_cube_path,
        name="small_cube",
        position=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        scale=small_scale,
        size=1.0,
        # <<< same here
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
)

world.reset()

stage = omni.usd.get_context().get_stage()

# Params
half_big = 0.5 * 1.0 * float(big_scale[0])
half_small = 0.5 * 1.0 * float(small_scale[0])
contact_distance = half_big + half_small  # ≈ 0.3
weld_surface_epsilon = 0.1

print(f"Contact distance (center-to-center) ~ {contact_distance:.3f} m")
print(f"Weld when surface gap < {weld_surface_epsilon:.3f} m")

welded = False
weld_frame = -1
impulse_applied = False

num_steps = 1000
target_speed = 1.0
upward_speed = 15.0


# Simulation loop
for frame in range(num_steps):
    if not welded:
        big_cube.set_linear_velocity(make_vel(target_speed, 0.0, 0.0))

    pos_big, _ = big_cube.get_world_pose()
    pos_small, _ = small_cube.get_world_pose()
    distance = dist_between(pos_big, pos_small)
    surface_gap = distance - contact_distance

    if not welded:
        if frame % 30 == 0:
            print(f"[Frame {frame}] dist={distance:.3f}, surface_gap={surface_gap:.3f}")

        if surface_gap < weld_surface_epsilon:
            print(
                f"[Frame {frame}] WELD CONDITION MET! "
                f"dist={distance:.3f}, surface_gap={surface_gap:.3f}"
            )
            joint_prim = weld_rigid_bodies(big_cube_path, small_cube_path)
            welded = True
            weld_frame = frame
            print(f"[Frame {frame}] >>> CUBES ARE NOW WELDED RIGIDLY ATTACHED <<<")

    if welded and not impulse_applied and frame >= weld_frame + 30:
        print(
            f"[Frame {frame}] Applying upward velocity "
            f"({upward_speed} m/s) to small_cube."
        )
        small_cube.set_linear_velocity(make_vel(0.0, 0.0, upward_speed))
        impulse_applied = True

    world.step(render=True)

simulation_app.close()