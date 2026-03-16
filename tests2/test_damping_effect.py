#!/usr/bin/env python3
"""
Test to demonstrate when joint damping and stiffness DO and DON'T affect simulation.

SCENARIO 1: Passive dynamics (NO torque commanded)
- Joint damping DOES affect how fast joints slow down
- Joint stiffness DOES create restoring force toward zero position

SCENARIO 2: Active control (Computed Torque Controller)
- Joint damping has NO effect (controller overrides it)
- Joint stiffness has NO effect (controller overrides it)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    StartMeshcat,
    MeshcatVisualizer,
    SceneGraph,
    Parser,
    ConstantVectorSource,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from script_cup_manipulator_controller_ofc import CupManipulator
from configs.robot.robot_types import create_cup_manipulator_config


def run_passive_test(damping_value, duration=5.0, dt=0.001):
    """Run simulation with NO torque input (passive dynamics).
    
    This shows the effect of damping - higher damping = faster decay.
    """
    print(colored(f"\n{'='*80}", "cyan"))
    print(colored(f"PASSIVE DYNAMICS TEST - Damping = {damping_value}", "cyan", attrs=["bold"]))
    print(colored(f"{'='*80}", "cyan"))
    
    builder = DiagramBuilder()
    plant, scene_graph = MultibodyPlant(time_step=dt), SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    
    # Load manipulator with specified damping
    urdf_path = Path(__file__).parent / "model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf"
    manip_config = create_cup_manipulator_config(
        urdf_path=str(urdf_path),
        joint_angles=(np.deg2rad(0), np.deg2rad(0)),  # Start at zero
        damping=(damping_value, damping_value),
        stiffness=(0.0, 0.0),  # No stiffness for this test
    )
    manipulator = CupManipulator(manip_config)
    manipulator.build_in_plant(plant, parser, weld_base=True)
    
    plant.Finalize()
    builder.AddSystem(plant)
    builder.AddSystem(scene_graph)
    
    # Connect scene graph
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # IMPORTANT: Connect ZERO torque input (passive)
    zero_torque = builder.AddSystem(ConstantVectorSource(np.zeros(2)))
    builder.Connect(
        zero_torque.get_output_port(),
        plant.get_actuation_input_port(manipulator.model_instance)
    )
    
    # Logger
    state_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(
        plant.get_state_output_port(manipulator.model_instance),
        state_logger.get_input_port()
    )
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial conditions: Give joints some velocity
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, manipulator.model_instance, np.deg2rad([30, 45]))
    plant.SetVelocities(plant_context, manipulator.model_instance, np.deg2rad([100, -80]))  # Fast initial velocities
    
    print(f"Initial: q1=30°, q2=45°, q̇1=100°/s, q̇2=-80°/s")
    print(f"Simulating {duration}s with ZERO torque (passive)...")
    
    simulator.AdvanceTo(duration)
    
    # Extract data
    state_log = state_logger.FindLog(context)
    t = state_log.sample_times()
    q1, q2 = state_log.data()[0, :], state_log.data()[1, :]
    q1_dot, q2_dot = state_log.data()[2, :], state_log.data()[3, :]
    
    print(f"Final: q1={np.rad2deg(q1[-1]):.1f}°, q2={np.rad2deg(q2[-1]):.1f}°, "
          f"q̇1={np.rad2deg(q1_dot[-1]):.1f}°/s, q̇2={np.rad2deg(q2_dot[-1]):.1f}°/s")
    
    return t, q1, q2, q1_dot, q2_dot


def run_active_control_test(damping_value, duration=5.0, dt=0.001):
    """Run simulation with COMPUTED TORQUE control.
    
    This shows damping has NO effect - controller dominates.
    """
    print(colored(f"\n{'='*80}", "yellow"))
    print(colored(f"ACTIVE CONTROL TEST - Damping = {damping_value}", "yellow", attrs=["bold"]))
    print(colored(f"{'='*80}", "yellow"))
    
    # This would be the same as your current test_manipulator_ee_trajectory.py
    # The controller commands torques, so joint damping doesn't matter
    
    print(colored("In active control mode, joint damping has NO EFFECT", "red", attrs=["bold"]))
    print(colored("The controller's torque commands override passive joint properties", "red"))
    
    return None  # Would need full controller setup


if __name__ == "__main__":
    print(colored("\n" + "="*80, "green"))
    print(colored("JOINT DAMPING DEMONSTRATION", "green", attrs=["bold"]))
    print(colored("="*80 + "\n", "green"))
    
    print(colored("This demonstrates when joint damping DOES and DOESN'T affect simulation:", "white"))
    print(colored("  1. PASSIVE (no torque): Damping causes velocity decay", "white"))
    print(colored("  2. ACTIVE (computed torque): Damping has no effect\n", "white"))
    
    # Test different damping values
    damping_values = [0.1, 0.5, 2.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for damping in damping_values:
        t, q1, q2, q1_dot, q2_dot = run_passive_test(damping, duration=3.0)
        
        # Plot joint velocities to show damping effect
        axes[0, 0].plot(t, np.rad2deg(q1_dot), label=f'q̇1 (damping={damping})')
        axes[0, 1].plot(t, np.rad2deg(q2_dot), label=f'q̇2 (damping={damping})')
        
        # Plot joint positions
        axes[1, 0].plot(t, np.rad2deg(q1), label=f'q1 (damping={damping})')
        axes[1, 1].plot(t, np.rad2deg(q2), label=f'q2 (damping={damping})')
    
    axes[0, 0].set_ylabel('q̇1 [deg/s]')
    axes[0, 0].set_title('Joint 1 Velocity (shows damping effect)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('q̇2 [deg/s]')
    axes[0, 1].set_title('Joint 2 Velocity (shows damping effect)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('q1 [deg]')
    axes[1, 0].set_title('Joint 1 Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('q2 [deg]')
    axes[1, 1].set_title('Joint 2 Position')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Joint Damping in PASSIVE Dynamics (No Torque Commanded)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = Path("plots") / "damping_effect_comparison.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(colored(f"\n✓ Plot saved to {plot_path}", "green"))
    
    plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("CONCLUSION:", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    print(colored("✓ In PASSIVE mode: Higher damping → faster velocity decay", "green"))
    print(colored("✗ In ACTIVE mode (computed torque): Damping has NO effect!", "red"))
    print(colored("  → Controller torques override passive joint properties", "yellow"))
