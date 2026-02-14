"""
Example script demonstrating the DrakeSceneManager with muscle dynamics.

Shows how to use the unified DrakeSceneManager for both standard controllers
and muscle dynamics simulations using a clean manager pattern.
"""

import sys

# Ensure --initial-theta is set BEFORE importing the module
# (module imports happen at parse time, and the module does args parsing at import time)
if '--initial-theta' not in ' '.join(sys.argv):
    sys.argv.extend(['--initial-theta', '180'])

import numpy as np
from script_cart_pendulum_muscle_dynamics import DrakeSceneManager

def example_1_simple_constant_force():
    """Example 1: Apply constant force to the cart (muscle dynamics mode)."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Constant Force Actuation (Using DrakeSceneManager)")
    print("="*70)
    
    manager = DrakeSceneManager(
        controller_mode='muscle',  # Special mode for muscle dynamics
        plant_type='multibody',
        visualize=True,
        constant_force=5.0,         # Apply 5 N force
        muscle_tau=0.03,            # Muscle time constant
        simulation_time=5.0,
        initial_angle=np.deg2rad(180)  # Pendulum pointing down
    )
    
    manager.run_full_simulation()
    print(f"✓ Simulation completed")


def example_2_no_force():
    """Example 2: Free fall response."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Free Fall (No Applied Force)")
    print("="*70)
    
    manager = DrakeSceneManager(
        controller_mode='muscle',
        plant_type='multibody',
        visualize=False,            # Faster without visualization
        constant_force=0.0,         # No external force
        muscle_tau=0.05,            # Different muscle time constant
        simulation_time=3.0,
        initial_angle=np.deg2rad(45)  # Start at 45 degrees
    )
    
    manager.run_full_simulation()
    print(f"✓ Simulation completed")


def example_3_compare_muscle_tau():
    """Example 3: Compare different muscle time constants."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Different Muscle Dynamics (tau)")
    print("="*70)
    
    for tau in [0.01, 0.03, 0.1]:
        print(f"\n--- Running with muscle tau = {tau} s ---")
        
        manager = DrakeSceneManager(
            controller_mode='muscle',
            plant_type='multibody',
            visualize=False,
            constant_force=3.0,
            muscle_tau=tau,
            simulation_time=2.0,
            initial_angle=np.deg2rad(180)
        )
        
        manager.run_full_simulation()
        print(f"  Completed with tau={tau}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DrakeSceneManager Muscle Dynamics Examples")
    print("="*70)
    print("\nThis demonstrates the unified DrakeSceneManager pattern:")
    print("- Step 1: Setup Drake system (with muscle dynamics)")
    print("- Step 2: Create force input")
    print("- Step 3: Setup visualization")
    print("- Step 4: Build diagram")
    print("- Step 5: Create and run simulator")
    
    # Run examples
    example_1_simple_constant_force()
    # example_2_no_force()
    # example_3_compare_muscle_tau()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
