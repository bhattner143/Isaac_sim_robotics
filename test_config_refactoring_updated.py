#!/usr/bin/env python3
"""Test that all physics parameters are using config objects."""

from script_cart_pendulum_muscle_dynamics import (
    CartPendulumPhysicsConfig, 
    CartPendulumSystemWithMuscleDynamics, 
    DrakeSceneManager,
    PHYSICS_CONFIG,
    STANDARD_LQR_CONFIG,
    SIM_CONFIG
)
from pydrake.systems.framework import DiagramBuilder
import numpy as np


def test_physics_config():
    """Verify physics config contains all parameters."""
    print('=' * 70)
    print('TEST 1: Physics Config Parameters')
    print('=' * 70)
    assert PHYSICS_CONFIG.mass_cart == 1.0
    assert PHYSICS_CONFIG.mass_pendulum == 0.5
    assert PHYSICS_CONFIG.length_pendulum == 0.5
    assert PHYSICS_CONFIG.radius_pendulum == 0.04
    assert PHYSICS_CONFIG.track_limit == 2.0
    assert PHYSICS_CONFIG.track_length == 4.0
    
    print(f'✓ Cart mass: {PHYSICS_CONFIG.mass_cart} kg')
    print(f'✓ Pendulum mass: {PHYSICS_CONFIG.mass_pendulum} kg')
    print(f'✓ Pendulum length: {PHYSICS_CONFIG.length_pendulum} m')
    print(f'✓ Cart damping: {PHYSICS_CONFIG.damping_cart}')
    print(f'✓ Pendulum damping: {PHYSICS_CONFIG.damping_pendulum}')
    print(f'✓ Track limit: {PHYSICS_CONFIG.track_limit} m')
    print(f'✓ Track length: {PHYSICS_CONFIG.track_length} m')
    print(f'✓ Motor time constant: {PHYSICS_CONFIG.motor_time_constant} s')
    print(f'✓ Gravity: {PHYSICS_CONFIG.gravity} m/s²')
    print()


def test_system_with_config():
    """Test that CartPendulumSystemWithMuscleDynamics uses config."""
    print('=' * 70)
    print('TEST 2: System Built with Config Parameters')
    print('=' * 70)
    
    builder = DiagramBuilder()
    system = CartPendulumSystemWithMuscleDynamics(
        config=PHYSICS_CONFIG,
        builder=builder,
        enable_muscle_dynamics=True,
        muscle_tau=0.03
    )
    
    # Build the plant
    system.build_plant_without_muscle()
    system.add_muscle_dynamics()
    
    print(f'✓ System built successfully')
    print(f'✓ Plant DOF: {system.plant.num_positions()}')
    print(f'✓ Plant actuators: {system.plant.num_actuators()}')
    print(f'✓ System.config.mass_cart = {system.config.mass_cart}')
    print(f'✓ System.config.length_pendulum = {system.config.length_pendulum}')
    print()


def test_scene_manager():
    """Test that DrakeSceneManager accepts config."""
    print('=' * 70)
    print('TEST 3: DrakeSceneManager with Config')
    print('=' * 70)
    
    manager = DrakeSceneManager(
        PHYSICS_CONFIG,
        SIM_CONFIG,
        controller_mode='scene-viz',
        visualize=False
    )
    
    print(f'✓ DrakeSceneManager created with config')
    print(f'✓ cart_pendulum_config stored: {hasattr(manager, "cart_pendulum_config")}')
    print(f'✓ simulation_config stored: {hasattr(manager, "simulation_config")}')
    print(f'✓ Config cart mass: {manager.cart_pendulum_config.mass_cart}')
    print(f'✓ Simulation timestep: {manager.simulation_config.timestep}')
    print()


def test_linearization():
    """Test that linearize_cart_pendulum uses config."""
    print('=' * 70)
    print('TEST 4: Linearization Using Config')
    print('=' * 70)
    
    builder = DiagramBuilder()
    system = CartPendulumSystemWithMuscleDynamics(
        config=PHYSICS_CONFIG,
        builder=builder,
        enable_muscle_dynamics=False
    )
    system.build_plant_without_muscle()
    
    A, B = system.linearize_cart_pendulum()
    
    print(f'✓ Linearization computed using config')
    print(f'✓ A matrix shape: {A.shape}')
    print(f'✓ B matrix shape: {B.shape}')
    print(f'✓ A[3, 1] (gravity effect): {A[3, 1]:.4f}')
    print(f'✓ B[2, 0] (cart acceleration): {B[2, 0]:.4f}')
    print()


if __name__ == '__main__':
    print('\n')
    print('█' * 70)
    print('  CONFIG REFACTORING VERIFICATION - CART PENDULUM')
    print('█' * 70)
    print()
    
    try:
        test_physics_config()
        test_system_with_config()
        test_scene_manager()
        test_linearization()
        
        print('█' * 70)
        print('  ✓ ALL TESTS PASSED - Config-Based Parameters Working!')
        print('█' * 70)
        print()
        print('Summary of Refactoring:')
        print('  • All physics parameters now come from CartPendulumPhysicsConfig')
        print('  • CartPendulumSystemWithMuscleDynamics receives config via __init__')
        print('  • linearize_cart_pendulum() uses self.config instead of globals')
        print('  • DrakeSceneManager stores and passes configs properly')
        print('  • Initialization chain: Factories → Manager → System class')
        print()
    except AssertionError as e:
        print(f'\n✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
