import numpy as np, time
from robots.cup_manipulator_tendon import CupManipulatorTendon, create_cable_manipulator_config
from pydrake.all import MultibodyPlant, Parser, DiagramBuilder, SceneGraph

URDF = 'model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf'
cfg = create_cable_manipulator_config(URDF)
builder = DiagramBuilder()
plant = MultibodyPlant(time_step=0.0)
plant.RegisterAsSourceForSceneGraph(builder.AddSystem(SceneGraph()))
m = CupManipulatorTendon(cfg)
m.load_urdf_to_plant(plant, Parser(plant))
m.weld_base_to_world(plant)
m.add_joint_actuators(plant)
m.add_end_effector_frame(plant)
plant.Finalize()

from pydrake.multibody.tree import JacobianWrtVariable

ctx = plant.CreateDefaultContext()
plant.SetPositions(ctx, m.model_instance, np.zeros(2))
plant.SetVelocities(ctx, m.model_instance, np.zeros(2))

try:
    ee_frame = plant.GetFrameByName(m.EE_FRAME_NAME, m.model_instance)
    p_BQ = np.zeros(3)
except Exception:
    link2_body = plant.GetBodyByName(m.LINK2_NAME, m.model_instance)
    ee_frame = link2_body.body_frame()
    p_BQ = np.array(m.EE_XYZ_LINK2)

ee_home = plant.CalcPointsPositions(ctx, ee_frame, p_BQ.reshape(3,1), plant.world_frame()).ravel()
print(f'EE at home (q1=q2=0): X={ee_home[0]:.4f}  Y={ee_home[1]:.4f}  Z={ee_home[2]:.4f}')

j1 = plant.GetJointByName(m.JT1_NAME, m.model_instance)
j2 = plant.GetJointByName(m.JT2_NAME, m.model_instance)
print(f'nv={plant.num_velocities(m.model_instance)}  v1_idx={j1.velocity_start()}  v2_idx={j2.velocity_start()}')

J_full = plant.CalcJacobianTranslationalVelocity(
    ctx, JacobianWrtVariable.kV, ee_frame, p_BQ, plant.world_frame(), plant.world_frame())
print(f'J_full shape={J_full.shape}')
print(f'J_full=\n{J_full}')

# Try target near home XY
home_xy = ee_home[:2]
test_targets = [home_xy + np.array([0.01, 0.0]),
                home_xy + np.array([0.0, 0.01]),
                home_xy + np.array([0.02, 0.01])]

t0 = time.time()
for tgt in test_targets:
    q, ok = m.compute_ik_analytical(
        plant, tgt, [0, 0], pos_tol=5e-3,
        q2_limit_rad=np.deg2rad(20), verbose=True, step=0.1)
    print(f'  target={tuple(tgt.round(4))}  ok={ok}  q1={np.rad2deg(q[0]):.2f}°  q2={np.rad2deg(q[1]):.2f}°')
print(f'Total time: {time.time()-t0:.2f}s for {len(test_targets)} points')
