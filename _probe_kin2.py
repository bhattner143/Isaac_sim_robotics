"""Probe exact kinematics: joint positions, EE position, Jacobian at q=0."""
import numpy as np
from pydrake.all import MultibodyPlant, Parser
from pydrake.multibody.tree import JacobianWrtVariable

plant = MultibodyPlant(time_step=0.0)
parser = Parser(plant)
parser.AddModels("model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_mate"))
plant.Finalize()
ctx = plant.CreateDefaultContext()
world = plant.world_frame()

j1 = plant.GetJointByName("link1_base")
j2 = plant.GetJointByName("link2_link1")
print(f"j1: position_start={j1.position_start()} velocity_start={j1.velocity_start()}")
print(f"j2: position_start={j2.position_start()} velocity_start={j2.velocity_start()}")

X_Wj1 = plant.CalcRelativeTransform(ctx, world, j1.frame_on_child())
X_Wj2 = plant.CalcRelativeTransform(ctx, world, j2.frame_on_child())
print(f"\nJoint1 world (q=0): {X_Wj1.translation()*1e3} mm")
print(f"Joint2 world (q=0): {X_Wj2.translation()*1e3} mm")

# Try all body frames to find EE candidate
bodies = ["link2_tendon", "simple_ball_5"]
for bn in bodies:
    try:
        body = plant.GetBodyByName(bn)
        X = plant.CalcRelativeTransform(ctx, world, body.body_frame())
        print(f"\n'{bn}' body origin (q=0): {X.translation()*1e3} mm")
    except Exception as e:
        print(f"  {bn}: {e}")

# EE frame
for fname in ["tendon_ee", "simple_ball_5"]:
    try:
        ee_frame = plant.GetFrameByName(fname)
        p_BQ = np.zeros(3)
        ee_pos = plant.CalcPointsPositions(ctx, ee_frame, p_BQ.reshape(3,1), world).ravel()
        print(f"\nEE frame '{fname}' (q=0): {ee_pos*1e3} mm")

        J = plant.CalcJacobianTranslationalVelocity(
            ctx, JacobianWrtVariable.kV, ee_frame, p_BQ, world, world)
        print(f"Jacobian (3 x {J.shape[1]}):")
        print(J)
        print(f"  col j1(v={j1.velocity_start()}): {J[:, j1.velocity_start()]}")
        print(f"  col j2(v={j2.velocity_start()}): {J[:, j2.velocity_start()]}")
        break
    except Exception as e:
        print(f"  {fname}: {e}")
