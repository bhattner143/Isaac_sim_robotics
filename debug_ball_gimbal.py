#!/usr/bin/env python3
"""
PyDrake Ball Gimbal Debugging Script
Analyzes why ball settles upward instead of downward
"""

import numpy as np
from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
)
import os

def analyze_ball_gimbal():
    """Analyze ball gimbal dynamics and verify COM positioning."""
    
    print("="*70)
    print("BALL GIMBAL DYNAMICS ANALYSIS")
    print("="*70)
    
    # Load URDF
    urdf_path = os.path.join(
        os.getcwd(),
        'model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf'
    )
    
    plant = MultibodyPlant(time_step=0.001)
    parser = Parser(plant)
    parser.AddModels(urdf_path)
    plant.Finalize()
    
    # Get gravity vector
    gravity = plant.gravity_field().gravity_vector()
    print(f"\n1. GRAVITY VECTOR (world frame):")
    print(f"   {gravity}")
    print(f"   Direction: {'DOWN' if gravity[2] < 0 else 'UP'} (Z-axis)")
    
    # Create context and set joints to zero
    context = plant.CreateDefaultContext()
    
    print(f"\n2. PLANT INFO:")
    print(f"   Total positions (DOFs): {plant.num_positions()}")
    print(f"   Total velocities: {plant.num_velocities()}")
    
    # Get joint positions (use default which should be zero)
    joint_positions = plant.GetPositions(context)
    print(f"   Default joint positions: {joint_positions}")
    
    # Get ball_with_rod body
    ball_body = plant.GetBodyByName("ball_with_rod")
    gimbal_body = plant.GetBodyByName("gimbal_link")
    
    # Get COM position in world frame
    ball_com_body = ball_body.CalcCenterOfMassInBodyFrame(context)
    
    # Transform to world
    X_WB_ball = plant.CalcRelativeTransform(context, plant.world_frame(), ball_body.body_frame())
    ball_com_world = X_WB_ball @ ball_com_body
    
    print(f"\n3. BALL BODY ANALYSIS:")
    print(f"   Mass: {ball_body.default_mass()} kg")
    print(f"   Spatial inertia:\n{ball_body.default_spatial_inertia()}")
    
    # Get ball_gimbal joint frame position
    ball_gimbal_joint = plant.GetJointByName("ball_gimbal")
    gimbal_cup_joint = plant.GetJointByName("gimbal_cup")
    
    # Get the frame positions
    gimbal_link_frame = gimbal_body.body_frame()
    ball_frame = ball_body.body_frame()
    
    # Transform to world
    X_WG = plant.CalcRelativeTransform(context, plant.world_frame(), gimbal_link_frame)
    X_WB = plant.CalcRelativeTransform(context, plant.world_frame(), ball_frame)
    
    gimbal_pivot_world = X_WG.translation()
    ball_origin_world = X_WB.translation()
    
    print(f"\n4. WORLD POSITIONS (joints at [0,0,0,0]):")
    print(f"   Gimbal link origin: {gimbal_pivot_world}")
    print(f"   Ball link origin:   {ball_origin_world}")
    print(f"   Ball COM (world):   {ball_com_world}")
    
    # Calculate offset from pivot to COM
    pivot_to_com = ball_com_world - gimbal_pivot_world
    
    print(f"\n5. PIVOT-TO-COM ANALYSIS:")
    print(f"   Vector (gimbal → ball COM): {pivot_to_com}")
    print(f"   Magnitude: {np.linalg.norm(pivot_to_com):.3f} m")
    print(f"   Z-component: {pivot_to_com[2]:.6f} m")
    
    # Check stability
    print(f"\n6. STABILITY CHECK:")
    if pivot_to_com[2] < -0.01:  # COM below pivot
        print(f"   ✓ COM is BELOW pivot (Z={pivot_to_com[2]:.4f}m)")
        print(f"   ✓ Ball should hang DOWN (stable at 0°)")
    elif pivot_to_com[2] > 0.01:  # COM above pivot
        print(f"   ✗ COM is ABOVE pivot (Z={pivot_to_com[2]:.4f}m)")
        print(f"   ✗ Ball will flip UP (stable at ±180°)")
    else:
        print(f"   ? COM near pivot (Z={pivot_to_com[2]:.4f}m)")
        print(f"   ? Marginally stable - sensitive to perturbations")
    
    # Check inertial frame rotation
    print(f"\n7. INERTIAL FRAME ANALYSIS:")
    
    # Get inertial pose in body frame
    M_BBcm_B = ball_body.default_spatial_inertia()
    print(f"   COM in ball body frame: {M_BBcm_B.get_com()}")
    
    # Joint axis analysis
    print(f"\n8. GIMBAL JOINT AXES:")
    print(f"   gimbal_cup axis:  {gimbal_cup_joint.revolute_axis()}")
    print(f"   ball_gimbal axis: {ball_gimbal_joint.revolute_axis()}")
    
    # Get joint origins
    print(f"\n9. JOINT ORIGIN TRANSFORMS:")
    # These are in parent frame
    print(f"   gimbal_cup:  origin xyz (in link2 frame)")
    print(f"   ball_gimbal: origin xyz (in gimbal_link frame)")
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    return {
        'ball_com_world': ball_com_world,
        'gimbal_pivot_world': gimbal_pivot_world,
        'pivot_to_com': pivot_to_com,
        'gravity': gravity,
        'com_z_offset': pivot_to_com[2]
    }

if __name__ == "__main__":
    results = analyze_ball_gimbal()
    
    print(f"\n{'='*70}")
    print("RECOMMENDED FIXES")
    print("="*70)
    
    if results['com_z_offset'] > 0:
        print("\n⚠️  CRITICAL: Ball COM is ABOVE the pivot!")
        print("\nROOT CAUSE ANALYSIS (ranked by likelihood):")
        print("1. ★★★ Inertial origin incorrectly rotated")
        print("   - The <inertial><origin rpy=\"...\"> uses same rotation as visual")
        print("   - This rotates the COM offset, placing it in wrong direction")
        print("   - FIX: Remove rotation from inertial origin, use xyz only")
        print("\n2. ★★☆ Joint origin has wrong sign")
        print("   - If ball rod points 'down' in mesh, joint may invert it")
        print("   - Check ball_gimbal joint rpy values")
        print("\n3. ★☆☆ COM xyz offset has wrong sign")
        print("   - Current: xyz=\"-0.2 0 0\"")
        print("   - If mesh has ball at +X, COM should be +0.2 (not -0.2)")
    else:
        print("\n✓ Ball COM appears to be below pivot (correct for pendulum)")
        print("\nPossible minor issues:")
        print("- Damping too high in gimbal joints")
        print("- Initial conditions not at stable equilibrium")
        print("- Numerical integration issues")
