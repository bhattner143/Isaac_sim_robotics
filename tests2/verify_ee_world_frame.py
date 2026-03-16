#!/usr/bin/env python3
"""Verify that EE position is computed correctly in world frame."""

import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.robot.robot_types import create_cup_manipulator_config
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
manip.weld_base_to_world(plant, position=np.array([0.0, 0.0, 0.0]), 
                         orientation=np.array([0.0, 0.0, 0.0]))
manip.add_joint_actuators(plant)
manip.add_end_effector_frame(plant)
plant.Finalize()

# Test at multiple configurations
test_configs = [
    ("Zero config", np.array([0.0, 0.0])),
    ("Default config", np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])),
    ("Negative q1", np.array([np.deg2rad(-45.0), np.deg2rad(45.0)])),
]

print("\n" + "="*70)
print("VERIFICATION: EE Position in World Frame")
print("="*70)
print("\nManipulator base welded at world origin [0, 0, 0] with no rotation")
print(f"EE offset from link2: {manip.EE_OFFSET}")
print("\n" + "-"*70)

context = plant.CreateDefaultContext()

for config_name, q_config in test_configs:
    print(f"\n{config_name}: q1={np.rad2deg(q_config[0]):.1f}°, q2={np.rad2deg(q_config[1]):.1f}°")
    
    # Set configuration
    manip.set_positions_user_order(plant, context, q_config)
    
    # Method 1: Using get_end_effector_position (uses cup_center frame)
    ee_pos_method1 = manip.get_end_effector_position(plant, context)
    print(f"  Method 1 (cup_center frame): ({ee_pos_method1[0]:.4f}, {ee_pos_method1[1]:.4f}, {ee_pos_method1[2]:.4f})")
    
    # Method 2: Using CalcPointsPositions with link2 + offset
    link2_frame = plant.GetBodyByName("link2", manip.model_instance).body_frame()
    ee_pos_method2 = plant.CalcPointsPositions(
        context, link2_frame, manip.EE_OFFSET.reshape(3, 1), plant.world_frame()
    ).flatten()
    print(f"  Method 2 (link2 + offset):    ({ee_pos_method2[0]:.4f}, {ee_pos_method2[1]:.4f}, {ee_pos_method2[2]:.4f})")
    
    # Check if they match
    diff = np.linalg.norm(ee_pos_method1 - ee_pos_method2)
    if diff < 1e-6:
        print(f"  ✓ Both methods agree (diff = {diff:.2e})")
    else:
        print(f"  ✗ Methods differ by {diff:.6f} m")
    
    # Also check link2 origin position for reference
    link2_pos = plant.CalcPointsPositions(
        context, link2_frame, np.zeros((3, 1)), plant.world_frame()
    ).flatten()
    print(f"  Reference - link2 origin:     ({link2_pos[0]:.4f}, {link2_pos[1]:.4f}, {link2_pos[2]:.4f})")

print("\n" + "="*70)
print("CONCLUSION: If both methods agree, EE position is correctly computed")
print("in world frame coordinates.")
print("="*70 + "\n")
