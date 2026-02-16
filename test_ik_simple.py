#!/usr/bin/env python3
import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from robot_types import create_cup_manipulator_config
from script_cart_pendulam_2d_extended_ofc_v2 import CupManipulator

# Setup
config = create_cup_manipulator_config(
    urdf_path='model_using_onshape_to_robot/cup_manipulator2/cup_manipulator_obj_right_frame.urdf',
    joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),
    damping=(0.1, 0.1),
)

plant = MultibodyPlant(time_step=0.0)
parser = Parser(plant)
manip = CupManipulator(config, enable_visualization=False)
manip.load_urdf_to_plant(plant, parser)
manip.weld_base_to_world(plant, position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0]))
manip.add_joint_actuators(plant)
manip.add_end_effector_frame(plant)
plant.Finalize()

# Test FK at seed
q_seed = np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])
context = plant.CreateDefaultContext()
manip.set_positions_user_order(plant, context, q_seed)
ee_seed = manip.get_end_effector_position(plant, context)
print(f'Seed config: q1=-10°, q2=20°')
print(f'FK at seed: ({ee_seed[0]:.3f}, {ee_seed[1]:.3f}, {ee_seed[2]:.3f})')

# Test IK
target_x, target_y = 0.0, 1.0
print(f'\nTarget: ({target_x}, {target_y}, z_unchanged)')
q_sol, success = manip.solve_initial_pose_via_ik(plant, np.array([target_x, target_y]), q_seed, pos_tol=0.01, verbose=True, target_z=None)

if success:
    manip.set_positions_user_order(plant, context, q_sol)
    ee_sol = manip.get_end_effector_position(plant, context)
    print(f'IK solution: q1={np.rad2deg(q_sol[0]):.2f}°, q2={np.rad2deg(q_sol[1]):.2f}°')
    print(f'FK at solution: ({ee_sol[0]:.3f}, {ee_sol[1]:.3f}, {ee_sol[2]:.3f})')
    print(f'Error: {np.linalg.norm(ee_sol[:2] - [target_x, target_y]):.6f} m')
else:
    print('IK FAILED')
