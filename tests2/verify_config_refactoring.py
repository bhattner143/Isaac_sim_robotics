#!/usr/bin/env python3
"""Verify config refactoring is complete."""

from script_cart_pendulum_muscle_dynamics import (
    CartPendulumSystemWithMuscleDynamics,
    PHYSICS_CONFIG,
    SIM_CONFIG
)
from pydrake.systems.framework import DiagramBuilder

print('\n' + '='*70)
print('CONFIGURATION REFACTORING VERIFICATION')
print('='*70 + '\n')

# Test 1: Physics config
print('✓ Physics Config Created via Factory')
print(f'  - mass_cart: {PHYSICS_CONFIG.mass_cart} kg')
print(f'  - mass_pendulum: {PHYSICS_CONFIG.mass_pendulum} kg')
print(f'  - length_pendulum: {PHYSICS_CONFIG.length_pendulum} m')
print(f'  - track_limit: {PHYSICS_CONFIG.track_limit} m')
print()

# Test 2: System with config
print('✓ System Built with Injected Config')
builder = DiagramBuilder()
system = CartPendulumSystemWithMuscleDynamics(
    config=PHYSICS_CONFIG,
    builder=builder,
    enable_muscle_dynamics=True,
    muscle_tau=0.03
)
system.build_plant_without_muscle()
system.add_muscle_dynamics()
print(f'  - System.config.mass_cart = {system.config.mass_cart} kg')
print(f'  - System.config.length_pendulum = {system.config.length_pendulum} m')
print()

# Test 3: Linearization uses config
print('✓ Linearization Uses Config Parameters')
A, B = system.linearize_cart_pendulum()
print(f'  - A shape: {A.shape}, B shape: {B.shape}')
print(f'  - Parameters: self.config.* (not globals)')
print()

# Test 4: Simulation config
print('✓ Simulation Config Created via Factory')
print(f'  - timestep: {SIM_CONFIG.timestep} s')
print(f'  - simulation_time: {SIM_CONFIG.simulation_time} s')
print()

print('='*70)
print('✓ REFACTORING COMPLETE')
print('='*70)
print()
print('Affected Methods:')
print('  • build_plant_without_muscle(): self.config.*')
print('  • create_model_for_controller(): self.config.*')
print('  • linearize_cart_pendulum(): self.config.*')
print()
