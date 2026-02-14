#!/usr/bin/env python3
"""
Test Manipulator Forward Kinematics

Pure kinematic test: prescribed joint angles → end effector positions
No dynamics simulation required.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from termcolor import colored
from pydrake.all import (
    MultibodyPlant, 
    Parser,
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Sphere,
    Rgba,
    SceneGraph,
    DiagramBuilder,
)
from pydrake.geometry import GeometryInstance, MakePhongIllustrationProperties

from robot_types import create_cup_manipulator_config
from script_cup_manipulator_controller_ofc import CupManipulator


def test_manipulator_kinematics(q_initial, q_final, num_points=30, visualize=True):
    """
    Test manipulator forward kinematics: joint trajectory → end effector positions.
    
    Args:
        q_initial: Initial joint angles [q1, q2] in radians
        q_final: Final joint angles [q1, q2] in radians
        num_points: Number of points along trajectory
        visualize: Enable Meshcat visualization
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR FORWARD KINEMATICS TEST", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Start Meshcat if visualization enabled
    meshcat = None
    scene_graph = None
    if visualize:
        meshcat = StartMeshcat()
        print(colored(f"\n🌐 Meshcat server started at: {meshcat.web_url()}", "green", attrs=["bold"]))
    
    # Create plant with scene graph for visualization
    builder = DiagramBuilder()
    plant, scene_graph = builder.AddNamedSystem(
        "plant",
        MultibodyPlant(time_step=0.001)
    ), builder.AddNamedSystem(
        "scene_graph", 
        SceneGraph()
    )
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    parser = Parser(plant)
    
    # Load manipulator
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(0.0, 0.0),
        damping=(0.0, 0.0),
        friction=(0.0, 0.0),
    )
    
    manipulator = CupManipulator(manipulator_config, enable_visualization=False)
    manipulator.load_urdf_to_plant(plant, parser)
    manipulator.weld_base_to_world(plant)
    
    plant.Finalize()
    
    # Connect plant to scene graph
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Add Meshcat visualizer
    if visualize:
        params = MeshcatVisualizerParams()
        params.role = Role.kIllustration
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
        meshcat.SetProperty("/Background", "visible", False)
    
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    print(colored(f"\n✓ Plant created", "green"))
    print(colored(f"  Model: Cup Manipulator", "cyan"))
    print(colored(f"  DOF: {plant.num_positions()}", "cyan"))
    
    # Create joint trajectory (linear interpolation)
    q1_traj = np.linspace(q_initial[0], q_final[0], num_points)
    q2_traj = np.linspace(q_initial[1], q_final[1], num_points)
    
    print(colored(f"\n📐 Computing forward kinematics...", "yellow"))
    
    # Compute end effector positions along trajectory
    ee_positions = []
    for i in range(num_points):
        q = np.array([q1_traj[i], q2_traj[i]])
        plant.SetPositions(plant_context, manipulator.model_instance, q)
        
        # Forward kinematics using custom method: joint angles → EE position (simple_ball center)
        # CalcPosition() automatically uses EE_OFFSET and world_frame
        ee_pos = manipulator.CalcPosition(plant, plant_context)
        
        ee_positions.append(ee_pos)
    
    ee_positions = np.array(ee_positions)
    
    # Print results
    print(colored(f"\n✓ Forward kinematics complete", "green"))
    print(colored(f"\n📊 Results:", "yellow", attrs=["bold"]))
    print(colored(f"  Initial joints: q1={np.rad2deg(q_initial[0]):7.2f}°, q2={np.rad2deg(q_initial[1]):7.2f}°", "cyan"))
    print(colored(f"  Final joints:   q1={np.rad2deg(q_final[0]):7.2f}°, q2={np.rad2deg(q_final[1]):7.2f}°", "cyan"))
    print(colored(f"\n  Initial EE: X={ee_positions[0, 0]:7.4f} m, Y={ee_positions[0, 1]:7.4f} m, Z={ee_positions[0, 2]:7.4f} m", "green"))
    print(colored(f"  Final EE:   X={ee_positions[-1, 0]:7.4f} m, Y={ee_positions[-1, 1]:7.4f} m, Z={ee_positions[-1, 2]:7.4f} m", "green"))
    
    ee_displacement = np.linalg.norm(ee_positions[-1] - ee_positions[0])
    ee_displacement_xy = np.linalg.norm(ee_positions[-1, :2] - ee_positions[0, :2])
    
    print(colored(f"\n  Total displacement: {ee_displacement:.4f} m (3D)", "yellow"))
    print(colored(f"  X-Y displacement:   {ee_displacement_xy:.4f} m (planar)", "yellow"))
    
    # Detailed table
    print(colored(f"\n📋 Detailed trajectory (every {max(1, num_points//10)} points):", "yellow"))
    print(colored("  " + "-"*76, "white"))
    print(colored(f"  {'Step':>5} | {'q1 [deg]':>9} | {'q2 [deg]':>9} | {'EE_X [m]':>10} | {'EE_Y [m]':>10} | {'EE_Z [m]':>10}", "white", attrs=["bold"]))
    print(colored("  " + "-"*76, "white"))
    
    step_size = max(1, num_points // 10)
    for i in range(0, num_points, step_size):
        print(colored(f"  {i:5d} | {np.rad2deg(q1_traj[i]):9.2f} | {np.rad2deg(q2_traj[i]):9.2f} | "
                     f"{ee_positions[i, 0]:10.4f} | {ee_positions[i, 1]:10.4f} | {ee_positions[i, 2]:10.4f}", "cyan"))
    
    # Always show last point
    if (num_points - 1) % step_size != 0:
        i = num_points - 1
        print(colored(f"  {i:5d} | {np.rad2deg(q1_traj[i]):9.2f} | {np.rad2deg(q2_traj[i]):9.2f} | "
                     f"{ee_positions[i, 0]:10.4f} | {ee_positions[i, 1]:10.4f} | {ee_positions[i, 2]:10.4f}", "cyan"))
    
    print(colored("  " + "-"*76, "white"))
    
    # Visualize trajectory in Meshcat
    if visualize and meshcat is not None:
        while True:  # Loop for repeated playback
            print(colored(f"\n🎬 Animating trajectory in Meshcat...", "yellow"))
            
            for i in range(num_points):
                q = np.array([q1_traj[i], q2_traj[i]])
                plant.SetPositions(plant_context, manipulator.model_instance, q)
                
                # Publish visualization
                diagram.ForcedPublish(context)
                
                # Draw sphere at end effector position
                ee_pos = ee_positions[i]
                meshcat.SetObject(
                    f"ee_marker",
                    Sphere(0.03),  # 3cm radius sphere
                    Rgba(1.0, 0.0, 0.0, 0.9)  # Red
                )
                from pydrake.math import RigidTransform
                meshcat.SetTransform(f"ee_marker", RigidTransform(ee_pos))
                
                # Draw trajectory path (line segments)
                if i > 0:
                    # Draw line from previous to current position
                    segment_name = f"trajectory/segment_{i}"
                    from pydrake.geometry import Cylinder
                    
                    # Compute line segment
                    p1 = ee_positions[i-1]
                    p2 = ee_positions[i]
                    midpoint = (p1 + p2) / 2
                    direction = p2 - p1
                    length = np.linalg.norm(direction)
                    
                    if length > 1e-6:
                        # Create cylinder for line segment
                        meshcat.SetObject(
                            segment_name,
                            Cylinder(0.004, length),  # 4mm radius
                            Rgba(0.2, 0.6, 1.0, 0.7)  # Blue
                        )
                        
                        # Orient cylinder along direction
                        from pydrake.math import RotationMatrix
                        z_axis = np.array([0, 0, 1])
                        if length > 0:
                            direction_normalized = direction / length
                            # Rotation from z-axis to direction
                            axis = np.cross(z_axis, direction_normalized)
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                axis = axis / axis_norm
                                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                                from scipy.spatial.transform import Rotation as R
                                rot = R.from_rotvec(angle * axis).as_matrix()
                                rotation = RotationMatrix(rot)
                            else:
                                rotation = RotationMatrix()
                            
                            transform = RigidTransform(rotation, midpoint)
                            meshcat.SetTransform(segment_name, transform)
                
                time.sleep(0.08)  # Animation speed
            
            print(colored("✓ Animation complete", "green"))
            print(colored(f"🌐 View at: {meshcat.web_url()}", "green", attrs=["bold"]))
            
            # Ask if user wants to replay
            replay = input(colored("\nReplay animation? (y/n): ", "yellow")).strip().lower()
            if replay != 'y' and replay != 'yes':
                break
    
    # Plot results
    print(colored(f"\n📈 Generating plots...", "yellow"))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Joint angles vs step
    ax1 = axes[0, 0]
    ax1.plot(np.rad2deg(q1_traj), 'b-o', linewidth=2, markersize=5, label='q₁', alpha=0.7)
    ax1.plot(np.rad2deg(q2_traj), 'r-s', linewidth=2, markersize=5, label='q₂', alpha=0.7)
    ax1.set_xlabel('Step Index', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Joint Angle [deg]', fontweight='bold', fontsize=11)
    ax1.set_title('Joint Space Trajectory', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: EE position components vs step
    ax2 = axes[0, 1]
    ax2.plot(ee_positions[:, 0], 'b-o', linewidth=2, markersize=4, label='X', alpha=0.7)
    ax2.plot(ee_positions[:, 1], 'r-s', linewidth=2, markersize=4, label='Y', alpha=0.7)
    ax2.plot(ee_positions[:, 2], 'g-^', linewidth=2, markersize=4, label='Z', alpha=0.7)
    ax2.set_xlabel('Step Index', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Position [m]', fontweight='bold', fontsize=11)
    ax2.set_title('Task Space Position Components', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: EE trajectory in X-Y plane
    ax3 = axes[1, 0]
    ax3.plot(ee_positions[:, 0], ee_positions[:, 1], 'purple', linewidth=2.5, alpha=0.7)
    ax3.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=14, label='Start', zorder=5)
    ax3.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=14, label='End', zorder=5)
    # Add arrows to show direction
    for i in range(0, num_points-1, max(1, num_points//8)):
        dx = ee_positions[i+1, 0] - ee_positions[i, 0]
        dy = ee_positions[i+1, 1] - ee_positions[i, 1]
        ax3.arrow(ee_positions[i, 0], ee_positions[i, 1], dx, dy, 
                 head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
    ax3.set_xlabel('X Position [m]', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Y Position [m]', fontweight='bold', fontsize=11)
    ax3.set_title('End Effector Path (X-Y Plane)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.legend(fontsize=10)
    
    # Plot 4: EE trajectory in X-Z plane
    ax4 = axes[1, 1]
    ax4.plot(ee_positions[:, 0], ee_positions[:, 2], 'purple', linewidth=2.5, alpha=0.7)
    ax4.plot(ee_positions[0, 0], ee_positions[0, 2], 'go', markersize=14, label='Start', zorder=5)
    ax4.plot(ee_positions[-1, 0], ee_positions[-1, 2], 'ro', markersize=14, label='End', zorder=5)
    # Add arrows
    for i in range(0, num_points-1, max(1, num_points//8)):
        dx = ee_positions[i+1, 0] - ee_positions[i, 0]
        dz = ee_positions[i+1, 2] - ee_positions[i, 2]
        ax4.arrow(ee_positions[i, 0], ee_positions[i, 2], dx, dz,
                 head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
    ax4.set_xlabel('X Position [m]', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Z Position [m]', fontweight='bold', fontsize=11)
    ax4.set_title('End Effector Path (X-Z Plane)', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.legend(fontsize=10)
    
    plt.suptitle('Forward Kinematics: Joint Space → Task Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print(colored("✓ Plots generated", "green"))
    
    return {
        'q1': q1_traj,
        'q2': q2_traj,
        'ee_positions': ee_positions,
    }


def main():
    parser = argparse.ArgumentParser(description='Test manipulator forward kinematics')
    parser.add_argument('--q1_init', type=float, default=-10.0, help='Initial q1 [deg]')
    parser.add_argument('--q2_init', type=float, default=20.0, help='Initial q2 [deg]')
    parser.add_argument('--q1_final', type=float, default=-30.0, help='Final q1 [deg]')
    parser.add_argument('--q2_final', type=float, default=60.0, help='Final q2 [deg]')
    parser.add_argument('--num_points', type=int, default=30, help='Number of trajectory points')
    parser.add_argument('--no-viz', action='store_true', help='Disable Meshcat visualization')
    args = parser.parse_args()
    
    # Convert to radians
    q_init = (np.deg2rad(args.q1_init), np.deg2rad(args.q2_init))
    q_final = (np.deg2rad(args.q1_final), np.deg2rad(args.q2_final))
    
    # Run kinematics test
    result = test_manipulator_kinematics(
        q_init, 
        q_final, 
        args.num_points,
        visualize=not args.no_viz
    )
    
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("KINEMATICS TEST COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    
    if not args.no_viz:
        input(colored("\nPress Enter to close Meshcat and exit...", "yellow"))


if __name__ == "__main__":
    main()
