#!/usr/bin/env python3
"""
Test script for CupManipulatorLinearizedSystem class.

This script demonstrates:
1. Creating a linearized cup manipulator system
2. Using both Drake and numerical linearization methods
3. Verifying the linearization
4. Printing linearization matrices
"""

import numpy as np
from pathlib import Path
from pydrake.all import DiagramBuilder
from termcolor import colored

# Import from the main script
from script_cup_manipulator_controller_ofc import (
    CupManipulatorLinearizedSystem,
    create_cup_manipulator_config,
    create_muscle_dynamics_config,
)


def test_linearized_system(linearization_method='drake'):
    """
    Test the CupManipulatorLinearizedSystem with specified method.
    
    Args:
        linearization_method: 'drake' or 'numerical'
    """
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored(f"TESTING CupManipulatorLinearizedSystem - Method: {linearization_method}", "magenta", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "magenta"))
    
    # Create configuration
    urdf_path = str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute())
    config = create_cup_manipulator_config(
        urdf_path=urdf_path,
        joint_angles=(0.0, 0.0),
        damping=(0.1, 0.1),
        stiffness=(0.0, 0.0),
        friction=(0.05, 0.05),
    )
    
    # Create muscle dynamics config (optional)
    muscle_config = create_muscle_dynamics_config(
        muscle_tau=0.03,
        muscle_initial_force=0.0,
        command_limit=None,
    )
    
    # Create builder
    builder = DiagramBuilder()
    
    # Equilibrium configuration (upright position)
    equilibrium_state = np.array([0.0, 0.0, 0.0, 0.0])  # [q1, q2, q̇1, q̇2]
    equilibrium_input = np.array([0.0, 0.0])  # [τ1, τ2]
    
    # Create linearized system
    print(colored("\nStep 1: Creating linearized system...", "cyan"))
    linearized_system = CupManipulatorLinearizedSystem(
        config=config,
        builder=builder,
        linearization_method=linearization_method,
        muscle_config=None,  # Disable muscle dynamics for now
        equilibrium_state=equilibrium_state,
        equilibrium_input=equilibrium_input,
    )
    
    # Build the linearized system
    print(colored("\nStep 2: Building linearized plant...", "cyan"))
    linearized_system.build_linearized_system_with_muscle()
    
    # Add muscle dynamics (optional)
    print(colored("\nStep 3: Adding muscle dynamics...", "cyan"))
    linearized_system.add_muscle_dynamics_to_linearized_plant()
    
    # Print linearization summary
    print(colored("\nStep 4: Printing linearization matrices...", "cyan"))
    linearized_system.print_linearization_summary()
    
    # Verify linearization (if numerical method was used)
    if linearization_method == 'numerical':
        print(colored("\nStep 5: Verifying numerical linearization...", "cyan"))
        linearized_system.verify_linearization(epsilon=1e-5)
    
    # Print port information
    print(colored("\n" + "=" * 80, "green"))
    print(colored("PORT INFORMATION", "green", attrs=["bold"]))
    print(colored("=" * 80, "green"))
    output_port = linearized_system.get_output_port()
    input_port = linearized_system.get_input_port()
    print(colored(f"  Output port size: {output_port.size()}", "cyan"))
    print(colored(f"  Input port size: {input_port.size()}", "cyan"))
    
    print(colored("\n" + "=" * 80, "green"))
    print(colored("✓ TEST COMPLETED SUCCESSFULLY", "green", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "green"))
    
    return linearized_system


def main():
    """Run tests for both linearization methods."""
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("CupManipulatorLinearizedSystem Test Suite", "yellow", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "yellow"))
    
    # Test 1: Drake's built-in linearization
    print(colored("\n" + "▼" * 80, "blue"))
    print(colored("TEST 1: Drake's Built-in Linearize() Method", "blue", attrs=["bold"]))
    print(colored("▼" * 80 + "\n", "blue"))
    
    linearized_drake = test_linearized_system(linearization_method='drake')
    
    # Test 2: Numerical finite difference linearization
    print(colored("\n" + "▼" * 80, "blue"))
    print(colored("TEST 2: Numerical Finite Difference Method", "blue", attrs=["bold"]))
    print(colored("▼" * 80 + "\n", "blue"))
    
    linearized_numerical = test_linearized_system(linearization_method='numerical')
    
    # Compare the two methods
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("COMPARISON: Drake vs Numerical", "magenta", attrs=["bold"]))
    print(colored("=" * 80, "magenta"))
    
    A_drake = linearized_drake.linearized_matrices['A_plant']
    A_numerical = linearized_numerical.linearized_matrices['A_plant']
    B_drake = linearized_drake.linearized_matrices['B_plant']
    B_numerical = linearized_numerical.linearized_matrices['B_plant']
    
    A_diff = np.linalg.norm(A_drake - A_numerical)
    B_diff = np.linalg.norm(B_drake - B_numerical)
    
    print(colored(f"\n  ||A_drake - A_numerical|| = {A_diff:.2e}", "cyan"))
    print(colored(f"  ||B_drake - B_numerical|| = {B_diff:.2e}", "cyan"))
    
    if A_diff < 1e-4 and B_diff < 1e-4:
        print(colored("\n✓ Methods agree to high precision!", "green", attrs=["bold"]))
    else:
        print(colored("\n⚠ Methods differ - check implementation", "yellow", attrs=["bold"]))
    
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("ALL TESTS COMPLETED", "magenta", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "magenta"))


if __name__ == "__main__":
    main()
