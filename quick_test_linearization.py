#!/usr/bin/env python3
"""Quick test of cup manipulator linearization."""

import numpy as np
from script_cup_manipulator_controller_ofc import linearize_cup_manipulator_and_print

# Test 1: Simple linearization at upright position
print("\nTest: Linearize cup manipulator at upright position")
print("="*80)

system = linearize_cup_manipulator_and_print(
    linearization_method='drake'
)

print("\n✓ Linearization test completed!")
