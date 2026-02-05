#!/usr/bin/env python3
"""Test script to check ball orientation at different angles."""

import numpy as np
import os
import time
from pydrake.all import (
    StartMeshcat,
    MultibodyPlant,
    Parser,
    DiagramBuilder,
    SceneGraph,
    MeshcatVisualizer,
)

def main():
    meshcat = StartMeshcat()
    print(f"Meshcat visualization: http://localhost:7003")
    
    urdf_path = os.path.join(
        os.getcwd(),
        'model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf'
    )
    
    # First, check how many positions we have
    plant_check = MultibodyPlant(time_step=0.001)
    Parser(plant_check).AddModels(urdf_path)
    plant_check.Finalize()
    
    print(f"\nSystem info:")
    print(f"  Total positions: {plant_check.num_positions()}")
    print(f"  Total velocities: {plant_check.num_velocities()}")
    
    # Test different ball_gimbal angles
    test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/2]
    
    for angle in test_angles:
        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
        Parser(plant).AddModels(urdf_path)
        plant.Finalize()
        
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        
        # Get default positions
        positions = plant.GetPositions(plant_context).copy()
        print(f"\nDefault positions shape: {positions.shape}, values: {positions}")
        
        # Assuming welded base adds 7 DOF (quaternion + position), then 4 joints
        # Set ball_gimbal (last joint) to test angle
        if len(positions) > 4:
            positions[-1] = angle  # Last position is ball_gimbal
        else:
            positions[3] = angle  # 4th joint (index 3)
        
        plant.SetPositions(plant_context, positions)
        diagram.ForcedPublish(context)
        
        print(f"\nBall gimbal angle = {np.rad2deg(angle):+6.1f}°")
        print(f"  Check Meshcat - is the ball hanging DOWN?")
        print(f"  Waiting 4 seconds...")
        time.sleep(4)

if __name__ == "__main__":
    main()
