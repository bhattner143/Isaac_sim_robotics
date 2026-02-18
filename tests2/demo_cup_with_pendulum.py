#!/usr/bin/env python3
"""
Demonstration: Build Cup Manipulator with 3D Pendulum and Meshcat Visualization

This script shows how to:
1. Build a Drake plant with the cup manipulator
2. Attach a 3D pendulum to the manipulator's link2
3. Setup Meshcat visualization
4. Run a simple simulation

The pendulum is attached programmatically (not in URDF) using Drake's
multibody API, demonstrating how to compose complex systems.
"""

import numpy as np
from script_cup_manipulator_controller_ofc import build_cup_manipulator_with_pendulum
from termcolor import colored


def demo_static_visualization():
    """Demo 1: Build and visualize the robot (no simulation)."""
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("DEMO 1: Static Visualization", "magenta", attrs=["bold"]))
    print(colored("=" * 80, "magenta"))
    
    # Build with default settings
    diagram, simulator, plant, scene_graph, meshcat, cup_manip, pendulum = \
        build_cup_manipulator_with_pendulum(
            enable_pendulum=True,
            enable_visualization=True,
            initial_joint_angles=(np.deg2rad(45), np.deg2rad(-90)),  # Interesting pose
            initial_pendulum_pitch=0.0,   # Hanging down
            initial_pendulum_roll=180.0,  # Default orientation
        )
    
    print(colored("\n✓ Robot built and visualized!", "green", attrs=["bold"]))
    print(colored("  The robot is now visible in Meshcat", "cyan"))
    print(colored("  Press Ctrl+C to exit", "yellow"))
    
    try:
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        pass


def demo_gravity_simulation():
    """Demo 2: Let the pendulum swing under gravity."""
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("DEMO 2: Gravity Simulation (No Control)", "magenta", attrs=["bold"]))
    print(colored("=" * 80, "magenta"))
    
    # Build robot with pendulum starting at an angle
    diagram, simulator, plant, scene_graph, meshcat, cup_manip, pendulum = \
        build_cup_manipulator_with_pendulum(
            enable_pendulum=True,
            enable_visualization=True,
            initial_joint_angles=(0.0, 0.0),  # Manipulator upright
            initial_pendulum_pitch=45.0,      # Pendulum tilted 45°
            initial_pendulum_roll=180.0,
        )
    
    print(colored("\n🎬 Starting gravity simulation...", "cyan"))
    print(colored("  Pendulum will swing under gravity (no control torques)", "yellow"))
    print(colored("  Manipulator joints are passive (zero torque)", "yellow"))
    
    # Simulate for 5 seconds
    simulation_time = 5.0
    dt = 0.01
    
    for t in np.arange(0, simulation_time, dt):
        simulator.AdvanceTo(t)
        
        # Print progress every second
        if int(t) > int(t - dt):
            print(colored(f"  Time: {t:.1f}s / {simulation_time}s", "cyan"))
    
    print(colored(f"\n✓ Simulation complete ({simulation_time}s)", "green", attrs=["bold"]))


def demo_custom_configuration():
    """Demo 3: Build with custom configuration."""
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("DEMO 3: Custom Configuration", "magenta", attrs=["bold"]))
    print(colored("=" * 80, "magenta"))
    
    # Build with interesting initial configuration
    diagram, simulator, plant, scene_graph, meshcat, cup_manip, pendulum = \
        build_cup_manipulator_with_pendulum(
            enable_pendulum=True,
            enable_visualization=True,
            initial_joint_angles=(np.deg2rad(80), np.deg2rad(-160)),
            initial_pendulum_pitch=30.0,
            initial_pendulum_roll=180.0,
        )
    
    print(colored("\n📊 System Information:", "yellow"))
    print(colored(f"  Total DOF: {plant.num_positions()}", "cyan"))
    print(colored(f"  Actuated DOF: {plant.num_actuators()}", "cyan"))
    print(colored(f"  Passive DOF: {plant.num_positions() - plant.num_actuators()}", "cyan"))
    
    # Get joint names
    print(colored("\n🔧 Joint Names:", "yellow"))
    for i in range(plant.num_joints()):
        joint = plant.get_joint(i)
        if joint.num_positions() > 0:
            print(colored(f"  - {joint.name()} ({joint.type_name()})", "cyan"))
    
    print(colored("\n✓ Robot configured and ready", "green", attrs=["bold"]))


def demo_without_pendulum():
    """Demo 4: Build without pendulum."""
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("DEMO 4: Cup Manipulator Only (No Pendulum)", "magenta", attrs=["bold"]))
    print(colored("=" * 80, "magenta"))
    
    diagram, simulator, plant, scene_graph, meshcat, cup_manip, pendulum = \
        build_cup_manipulator_with_pendulum(
            enable_pendulum=False,  # No pendulum
            enable_visualization=True,
            initial_joint_angles=(np.deg2rad(60), np.deg2rad(-120)),
        )
    
    print(colored("\n✓ Cup manipulator built (no pendulum)", "green", attrs=["bold"]))
    print(colored(f"  Total DOF: {plant.num_positions()} (only manipulator joints)", "cyan"))
    print(colored(f"  Pendulum: {pendulum}", "cyan"))


def main():
    """Run all demonstrations."""
    demos = [
        ("Demo 1: Static Visualization", demo_static_visualization),
        ("Demo 2: Gravity Simulation", demo_gravity_simulation),
        ("Demo 3: Custom Configuration", demo_custom_configuration),
        ("Demo 4: Without Pendulum", demo_without_pendulum),
    ]
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  CUP MANIPULATOR WITH 3D PENDULUM - DEMONSTRATIONS".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print(colored("\nAvailable Demonstrations:", "yellow", attrs=["bold"]))
    for i, (name, _) in enumerate(demos, 1):
        print(colored(f"  {i}. {name}", "cyan"))
    print(colored("  0. Run all demos", "cyan"))
    print(colored("  q. Quit", "cyan"))
    
    choice = input(colored("\nSelect demo (0-4, q): ", "yellow")).strip().lower()
    
    if choice == 'q':
        print(colored("Goodbye!", "green"))
        return
    elif choice == '0':
        for name, demo_func in demos:
            try:
                demo_func()
            except KeyboardInterrupt:
                print(colored("\n\nDemo interrupted by user", "yellow"))
                break
            except Exception as e:
                print(colored(f"\n✗ Demo failed: {e}", "red"))
                import traceback
                traceback.print_exc()
    elif choice in ['1', '2', '3', '4']:
        idx = int(choice) - 1
        name, demo_func = demos[idx]
        try:
            demo_func()
        except KeyboardInterrupt:
            print(colored("\n\nDemo interrupted by user", "yellow"))
        except Exception as e:
            print(colored(f"\n✗ Demo failed: {e}", "red"))
            import traceback
            traceback.print_exc()
    else:
        print(colored("Invalid choice", "red"))


if __name__ == "__main__":
    # Quick run - just demo 1 by default
    # Uncomment next line for interactive menu
    # main()
    
    # Or run a specific demo directly:
    demo_static_visualization()
    # demo_gravity_simulation()
    # demo_custom_configuration()
    # demo_without_pendulum()
