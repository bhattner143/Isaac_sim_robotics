from pydrake.all import MultibodyPlant, Parser
import numpy as np

plant = MultibodyPlant(time_step=0.0)
parser = Parser(plant)
parser.AddModels("model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_mate"))
plant.Finalize()
ctx = plant.CreateDefaultContext()
world = plant.world_frame()

j1 = plant.GetJointByName("link1_base")
j2 = plant.GetJointByName("link2_link1")
print(f"j1 position_start={j1.position_start()}  velocity_start={j1.velocity_start()}")
print(f"j2 position_start={j2.position_start()}  velocity_start={j2.velocity_start()}")

# Joint axis frames in world
X_Wj1 = plant.CalcRelativeTransform(ctx, world, j1.frame_on_child())
X_Wj2 = plant.CalcRelativeTransform(ctx, world, j2.frame_on_child())
print(f"\njoint1 world pos (q=0): {X_Wj1.translation()*1e3} mm")
print(f"joint2 world pos (q=0): {X_Wj2.translation()*1e3} mm")

# All frames - find EE
from pydrake.multibody.tree import JacobianWrtVariable
for fi in range(plant.num_frames()):
    fr = plant.get_frame(plant.FrameIndex(plant.FrameIndex(fi)))
    nm = fr.name().lower()
    if any(k in nm for k in ['ee', 'simple_ball_5', 'tendon_ee', 'end_eff']):
        X = plant.CalcRelativeTransform(ctx, world, fr)
        print(f"  EE candidate '{fr.name()}': {X.translation()*1e3} mm")

# Try adding EE frame and computing Jacobian
try:
    ee_frame = plant.GetFrameByName("tendon_ee")
except Exception:
    # use link2_tendon body frame + offset
    ee_frame = plant.GetBodyByName("link2_tendon").body_frame()

from pydrake.multibody.tree import JacobianWrtVariable
p_BQ = np.zeros(3)
X_WEE = plant.CalcPointsPositions(ctx, ee_frame, p_BQ.reshape(3,1), world).ravel()
print(f"\nEE world pos (q=0): {X_WEE*1e3} mm")

J = plant.CalcJacobianTranslationalVelocity(
    ctx, JacobianWrtVariable.kV, ee_frame, p_BQ, world, world)
print(f"J shape: {J.shape}")
print(f"J (3 x nv):\n{J}")
print(f"\nJ columns for j1 (v={j1.velocity_start()}): {J[:, j1.velocity_start()]}")
print(f"J columns for j2 (v={j2.velocity_start()}): {J[:, j2.velocity_start()]}")
