#!/usr/bin/env python3
"""
Verification script for refactored script_cart_pendulum_diagram.py

Tests that the new architecture still works correctly.
"""

import sys
import os

# Add workspace to path
sys.path.insert(0, '/Volumes/Data/Isaac_sim_robotics')

def test_import():
    """Test that the module imports correctly."""
    print("=" * 70)
    print("TEST 1: Importing script_cart_pendulum_diagram...")
    print("=" * 70)
    try:
        import script_cart_pendulum_diagram as cart_pd
        print("✅ Import successful")
        return cart_pd
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_class_instantiation(cart_pd):
    """Test that DrakeSceneManager can be instantiated."""
    print("\n" + "=" * 70)
    print("TEST 2: Instantiating DrakeSceneManager...")
    print("=" * 70)
    try:
        manager = cart_pd.DrakeSceneManager(
            controller_mode='scene-viz',
            plant_type='multibody',
            visualize=False
        )
        print("✅ DrakeSceneManager instantiation successful")
        print(f"   - Controller mode: {manager.controller_mode}")
        print(f"   - Plant type: {manager.plant_type}")
        print(f"   - Builder: {type(manager.builder).__name__}")
        if hasattr(manager, 'system'):
            print(f"   - System: {type(manager.system).__name__}")
        else:
            print(f"   - System: <not initialized yet>")
        return manager
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_plant_building(manager):
    """Test that plant building works."""
    print("\n" + "=" * 70)
    print("TEST 3: Building Drake system (STEP 1)...")
    print("=" * 70)
    try:
        manager.setup_drake_system()
        print("✅ Plant building successful")
        print(f"   - Plant has {manager.plant.num_actuators()} actuators")
        print(f"   - Plant has {manager.plant.num_bodies()} bodies")
        print(f"   - Plant is finalized: {manager.plant.is_finalized()}")
        return manager
    except Exception as e:
        print(f"❌ Plant building failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dependency_injection(cart_pd):
    """Test that CartPendulumSystem properly uses injected builder."""
    print("\n" + "=" * 70)
    print("TEST 4: Verifying dependency injection...")
    print("=" * 70)
    try:
        from pydrake.systems.framework import DiagramBuilder
        builder = DiagramBuilder()
        system = cart_pd.CartPendulumSystem(builder)
        
        assert system.builder is builder, "Builder not properly injected"
        assert system.plant is None, "Plant should be None before build"
        
        system.build_plant()
        
        assert system.plant is not None, "Plant should exist after build"
        assert system.plant.is_finalized(), "Plant should be finalized"
        
        print("✅ Dependency injection working correctly")
        print(f"   - Builder injected: {system.builder is builder}")
        print(f"   - Plant built successfully: {system.plant is not None}")
        return True
    except Exception as e:
        print(f"❌ Dependency injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_separation_of_concerns():
    """Test that CartPendulumSystem ONLY has plant-related fields."""
    print("\n" + "=" * 70)
    print("TEST 5: Verifying separation of concerns...")
    print("=" * 70)
    try:
        from pydrake.systems.framework import DiagramBuilder
        import script_cart_pendulum_diagram as cart_pd
        
        builder = DiagramBuilder()
        system = cart_pd.CartPendulumSystem(builder)
        
        # Check that system DOESN'T have non-plant fields
        forbidden_fields = ['controller', 'meshcat', 'diagram', 'simulator', 
                           'time_log', 'state_log', 'force_log']
        
        bad_fields = [f for f in forbidden_fields if hasattr(system, f)]
        if bad_fields:
            print(f"❌ CartPendulumSystem has non-plant fields: {bad_fields}")
            return False
        
        # Check that system HAS required plant fields
        required_fields = ['builder', 'plant', 'scene_graph']
        good_fields = [f for f in required_fields if hasattr(system, f)]
        
        if len(good_fields) != len(required_fields):
            print(f"❌ CartPendulumSystem missing required fields")
            return False
        
        print("✅ Separation of concerns verified")
        print(f"   - Plant-builder only: ✓")
        print(f"   - No orchestration fields: ✓")
        print(f"   - Required fields present: {required_fields}")
        return True
        
    except Exception as e:
        print(f"❌ Separation of concerns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "REFACTORING VERIFICATION TESTS" + " " * 28 + "║")
    print("║" + " " * 8 + "script_cart_pendulum_diagram.py" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    results = []
    
    # Test 1: Import
    cart_pd = test_import()
    results.append(("Import", cart_pd is not None))
    
    if cart_pd is None:
        print("\n❌ TESTS FAILED: Cannot continue without successful import")
        return False
    
    # Test 2: Instantiation
    manager = test_class_instantiation(cart_pd)
    results.append(("Instantiation", manager is not None))
    
    # Test 3: Plant Building (needs manager)
    if manager is not None:
        manager = test_plant_building(manager)
        results.append(("Plant Building", manager is not None))
    else:
        results.append(("Plant Building", False))
    
    # Test 4: Dependency Injection
    di_ok = test_dependency_injection(cart_pd)
    results.append(("Dependency Injection", di_ok))
    
    # Test 5: Separation of Concerns
    soc_ok = test_separation_of_concerns()
    results.append(("Separation of Concerns", soc_ok))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY".center(70))
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Refactoring verified successfully.".center(70))
    else:
        print("⚠️  SOME TESTS FAILED. Review output above.".center(70))
    print("=" * 70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
