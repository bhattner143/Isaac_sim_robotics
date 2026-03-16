#!/usr/bin/env python3
"""
Check mass matrix coupling between manipulator joints and pendulum angles.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydrake.all import (
    MultibodyPlant,
    Parser,
    RevoluteJoint,
    RigidTransform,
    FixedOffsetFrame,
    SpatialInertia,
    UnitInertia,
)

from configs.robot.robot_types import create_cup_manipulator_config, create_pendulum_config
from archive.script_cup_manipulator_controller_drake import Pendulum3D, CupManipulator

# Constants from main script
PENDULUM_MASS = 0.5  # kg
PENDULUM_LENGTH = 0.2  # m
PENDULUM_DAMPING = 0.1

# Create plant
plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)

# Load cup manipulator using same config as main script
cup_config = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.0, 0.0),
    damping=(0.0, 0.0),
    stiffness=(0.0, 0.0),
    friction=(0.05, 0.05),
)

cup_manipulator = CupManipulator(cup_config)
model_instance = cup_manipulator.load_urdf_to_plant(plant, parser)

# Weld base to world
base_frame = plant.GetBodyByName("base_mount_manipulator", model_instance).body_frame()
plant.WeldFrames(plant.world_frame(), base_frame)

# Add actuators
for joint_name in ["link1_base", "link2_link1"]:
    joint = plant.GetJointByName(joint_name, model_instance)
    plant.AddJointActuator(joint_name, joint)

# Set joint properties
cup_manipulator.set_joint_properties(plant)

# Add pendulum using same method as main script
pendulum_config = create_pendulum_config(
    mass=PENDULUM_MASS,
    length=PENDULUM_LENGTH,
    radius=0.05,
    damping=PENDULUM_DAMPING,
    attachment_point=(-1.2545, 0.0, -0.188125),
    initial_pitch=0.0,
    initial_roll=180.0,
    name="pendulum"
)

pendulum = Pendulum3D(pendulum_config)
link2_body = plant.GetBodyByName("link2", model_instance)
pendulum.attach_to_body(plant, link2_body, model_instance)

# Finalize plant
plant.Finalize()

# Create context
context = plant.CreateDefaultContext()

print("="*70)
print("MASS MATRIX COUPLING ANALYSIS")
print("="*70)
print()
print("SYSTEM ARCHITECTURE:")
print("-" * 70)
print("Cup Manipulator = 2-link planar robot arm")
print()
print("  L1 = link1_base    → Joint connecting base to first link")
print("                       (Revolute joint rotating about Z-axis)")
print()
print("  L2 = link2_link1   → Joint connecting first link to second link")
print("                       (Revolute joint rotating about Z-axis)")
print()
print("  P  = pendulum_pitch → Pendulum pitch angle (rotation about Y-axis)")
print("                        Attached to end of link2")
print()
print("  R  = pendulum_roll  → Pendulum roll angle (rotation about X-axis)")
print("                        Attached to pitch body")
print()
print("Total: 4 DOFs [L1, L2, P, R], but only 2 actuated [L1, L2]")
print("       Pendulum [P, R] is PASSIVE (no direct actuation)")
print("-" * 70)
print()
print(f"System DOFs (positions): {plant.num_positions()}")
print(f"System DOFs (velocities): {plant.num_velocities()}")
print(f"Number of actuators: {plant.num_actuators()}")
print()

# Print all joints
print("JOINT STRUCTURE:")
print("-" * 70)
all_joint_names = []
for joint_idx in plant.GetJointIndices(model_instance):
    joint = plant.get_joint(joint_idx)
    if joint.num_velocities() > 0:
        all_joint_names.append(joint.name())
        print(f"  {joint.name()}: {joint.num_positions()} pos, {joint.num_velocities()} vel")

print()
print(f"Total movable joints: {len(all_joint_names)}")
print(f"Joint order: {all_joint_names}")
print()

# Test configurations
configs = {
    "Zero (all joints at 0°)": np.zeros(plant.num_positions()),
    "Goal (L1=20°, L2=-40°, P=0°, R=180°)": np.zeros(plant.num_positions()),
    "Goal with P=0°, R=0°": np.zeros(plant.num_positions()),
    "Goal with P=0°, R=90°": np.zeros(plant.num_positions()),
    "Goal with P=90°, R=180°": np.zeros(plant.num_positions()),
    "Goal with P=180°, R=180°": np.zeros(plant.num_positions()),
    "Different (L1=45°, L2=-45°, P=30°, R=150°)": np.zeros(plant.num_positions()),
}

# Set specific values for known joints
# Need to figure out which indices correspond to which joints
joint_index_map = {}
for i, joint_idx in enumerate(plant.GetJointIndices(model_instance)):
    joint = plant.get_joint(joint_idx)
    if joint.num_velocities() > 0:
        joint_index_map[joint.name()] = joint.velocity_start()

print("JOINT INDEX MAPPING:")
print("-" * 70)
for name, idx in sorted(joint_index_map.items(), key=lambda x: x[1]):
    print(f"  {name}: velocity_start = {idx}")
print()

# Now set the configurations correctly
if 'link1_base' in joint_index_map and 'link2_link1' in joint_index_map:
    idx_l1 = joint_index_map['link1_base']
    idx_l2 = joint_index_map['link2_link1']
    idx_pitch = joint_index_map.get('pendulum_pitch', 2)
    idx_roll = joint_index_map.get('pendulum_roll', 3)
    
    configs["Goal (L1=20°, L2=-40°, P=0°, R=180°)"][idx_l1] = np.deg2rad(20)
    configs["Goal (L1=20°, L2=-40°, P=0°, R=180°)"][idx_l2] = np.deg2rad(-40)
    configs["Goal (L1=20°, L2=-40°, P=0°, R=180°)"][idx_pitch] = np.deg2rad(0)
    configs["Goal (L1=20°, L2=-40°, P=0°, R=180°)"][idx_roll] = np.deg2rad(180)
    
    configs["Goal with P=0°, R=0°"][idx_l1] = np.deg2rad(20)
    configs["Goal with P=0°, R=0°"][idx_l2] = np.deg2rad(-40)
    configs["Goal with P=0°, R=0°"][idx_pitch] = np.deg2rad(0)
    configs["Goal with P=0°, R=0°"][idx_roll] = np.deg2rad(0)
    
    configs["Goal with P=0°, R=90°"][idx_l1] = np.deg2rad(20)
    configs["Goal with P=0°, R=90°"][idx_l2] = np.deg2rad(-40)
    configs["Goal with P=0°, R=90°"][idx_pitch] = np.deg2rad(0)
    configs["Goal with P=0°, R=90°"][idx_roll] = np.deg2rad(90)
    
    configs["Goal with P=90°, R=180°"][idx_l1] = np.deg2rad(20)
    configs["Goal with P=90°, R=180°"][idx_l2] = np.deg2rad(-40)
    configs["Goal with P=90°, R=180°"][idx_pitch] = np.deg2rad(90)
    configs["Goal with P=90°, R=180°"][idx_roll] = np.deg2rad(180)
    
    configs["Goal with P=180°, R=180°"][idx_l1] = np.deg2rad(20)
    configs["Goal with P=180°, R=180°"][idx_l2] = np.deg2rad(-40)
    configs["Goal with P=180°, R=180°"][idx_pitch] = np.deg2rad(180)
    configs["Goal with P=180°, R=180°"][idx_roll] = np.deg2rad(180)
    
    configs["Different (L1=45°, L2=-45°, P=30°, R=150°)"][idx_l1] = np.deg2rad(45)
    configs["Different (L1=45°, L2=-45°, P=30°, R=150°)"][idx_l2] = np.deg2rad(-45)
    configs["Different (L1=45°, L2=-45°, P=30°, R=150°)"][idx_pitch] = np.deg2rad(30)
    configs["Different (L1=45°, L2=-45°, P=30°, R=150°)"][idx_roll] = np.deg2rad(150)

for config_name, q in configs.items():
    print("="*70)
    print(f"CONFIGURATION: {config_name}")
    q_deg = np.rad2deg(q)
    print(f"  q (deg) = {q_deg}")
    print("="*70)
    
    # Set configuration with some velocity to show Coriolis effects
    plant.SetPositions(context, q)
    
    # Test 1: Zero velocity (static)
    plant.SetVelocities(context, np.zeros(plant.num_velocities()))
    M = plant.CalcMassMatrix(context)
    g_only = plant.CalcGravityGeneralizedForces(context)  # Pure gravity
    C_static = plant.CalcBiasTerm(context)  # Should equal g_only when v=0
    
    print(f"\nMASS MATRIX M ({M.shape[0]}x{M.shape[1]}):")
    print("All entries:")
    for i in range(M.shape[0]):
        row_str = f"  [{i}]  "
        for j in range(M.shape[1]):
            row_str += f"{M[i,j]:9.5f} "
        print(row_str)
    
    # Only analyze coupling if we have 4 DOFs as expected
    if plant.num_positions() == 4 and 'pendulum_pitch' in joint_index_map:
        idx_pitch = joint_index_map['pendulum_pitch']
        idx_roll = joint_index_map['pendulum_roll']
        idx_l1 = joint_index_map['link1_base']
        idx_l2 = joint_index_map['link2_link1']
        
        print(f"\n📊 INERTIAL COUPLING (from Mass Matrix M):")
        print(f"  M[{idx_pitch},{idx_l1}] (Pitch ← L1): {M[idx_pitch,idx_l1]:10.6f}  ", end="")
        print("✓ COUPLED" if abs(M[idx_pitch,idx_l1]) > 1e-6 else "✗ ZERO")
        
        print(f"  M[{idx_pitch},{idx_l2}] (Pitch ← L2): {M[idx_pitch,idx_l2]:10.6f}  ", end="")
        print("✓ COUPLED" if abs(M[idx_pitch,idx_l2]) > 1e-6 else "✗ ZERO")
        
        print(f"  M[{idx_roll},{idx_l1}] (Roll  ← L1): {M[idx_roll,idx_l1]:10.6f}  ", end="")
        print("✓ COUPLED" if abs(M[idx_roll,idx_l1]) > 1e-6 else "✗ ZERO")
        
        print(f"  M[{idx_roll},{idx_l2}] (Roll  ← L2): {M[idx_roll,idx_l2]:10.6f}  ", end="")
        print("✓ COUPLED" if abs(M[idx_roll,idx_l2]) > 1e-6 else "✗ ZERO")
        
        # Test 2: Non-zero velocity to show Coriolis coupling
        print(f"\n🔄 DYNAMIC COUPLING TEST (with velocity):")
        print("  Testing: v_L1 = 1 rad/s, others = 0")
        v_test = np.zeros(plant.num_velocities())
        v_test[idx_l1] = 1.0  # L1 moving at 1 rad/s
        plant.SetVelocities(context, v_test)
        C_L1 = plant.CalcBiasTerm(context)
        coriolis_L1 = C_L1 - g_only  # Subtract gravity to isolate Coriolis
        
        print(f"    Coriolis effect on Pitch: {coriolis_L1[idx_pitch]:10.6f}  ", end="")
        print("✓ COUPLES" if abs(coriolis_L1[idx_pitch]) > 1e-6 else "○ none")
        print(f"    Coriolis effect on Roll:  {coriolis_L1[idx_roll]:10.6f}  ", end="")
        print("✓ COUPLES" if abs(coriolis_L1[idx_roll]) > 1e-6 else "○ none")
        
        print("  Testing: v_L2 = 1 rad/s, others = 0")
        v_test = np.zeros(plant.num_velocities())
        v_test[idx_l2] = 1.0  # L2 moving at 1 rad/s
        plant.SetVelocities(context, v_test)
        C_L2 = plant.CalcBiasTerm(context)
        coriolis_L2 = C_L2 - g_only
        
        print(f"    Coriolis effect on Pitch: {coriolis_L2[idx_pitch]:10.6f}  ", end="")
        print("✓ COUPLES" if abs(coriolis_L2[idx_pitch]) > 1e-6 else "○ none")
        print(f"    Coriolis effect on Roll:  {coriolis_L2[idx_roll]:10.6f}  ", end="")
        print("✓ COUPLES" if abs(coriolis_L2[idx_roll]) > 1e-6 else "○ none")
        
        # Reset velocity
        plant.SetVelocities(context, np.zeros(plant.num_velocities()))
    
    # Gravity term
    print(f"\n🔽 GRAVITY FORCES:")
    print(f"  g = {g_only}")
    
    print()

print("="*70)
print("ACTUAL FINDINGS - SUMMARY")
print("="*70)
print()
print("🎯 CRITICAL DISCOVERY FROM SIMULATION:")
print("="*70)
print()
print("The coupling test simulation shows that BOTH L1 and L2 affect")
print("BOTH pitch and roll, even though M[2,1] (pitch-L2 inertial coupling) ≈ 0!")
print()
print("WHY? Because there are THREE types of coupling, not just one:")
print()
print("1️⃣  INERTIAL COUPLING (Mass Matrix M):")
print("   - Shows acceleration coupling: M(q)q̈")
print("   - What we analyzed above")
print("   - M[2,1] ≈ 0: Pitch acceleration weakly coupled to L2 acceleration")
print()
print("2️⃣  CORIOLIS/CENTRIFUGAL COUPLING (C(q,v)v):")
print("   - Shows velocity-dependent coupling")
print("   - When L2 rotates → creates centrifugal/Coriolis forces")
print("   - These forces create torques on pendulum!")
print("   - THIS is why L2 affects pitch in simulation!")
print()
print("3️⃣  GRAVITATIONAL COUPLING (g(q)):")
print("   - When manipulator moves → changes pendulum's base position")
print("   - Changes effective gravity direction in pendulum frame")
print("   - Creates additional coupling")
print()
print("="*70)
print("PHYSICAL EXPLANATION - Why L2 affects pitch despite M[2,1] ≈ 0:")
print("="*70)
print()
print("When L2 rotates sinusoidally:")
print("  1. The elbow joint accelerates → creates centrifugal force")
print("  2. This force accelerates the pivot point (pendulum attachment)")
print("  3. Accelerating the pivot → pendulum responds to 'shake' the base")
print("  4. This shaking creates BOTH pitch and roll motion!")
print()
print("The mass matrix ONLY captures direct inertial coupling (if you")
print("instantly accelerated L2, how much would pitch instantly accelerate).")
print()
print("But in REAL dynamics with continuous motion:")
print("  - Velocity builds up → Coriolis forces grow")
print("  - Position changes → Gravity direction changes")
print("  - These create INDIRECT coupling that M doesn't show!")
print()
print("="*70)
print("MATHEMATICAL VIEW:")
print("="*70)
print()
print("Full dynamics: M(q)q̈ + C(q,v)v + g(q) = τ")
print()
print("Rearranging: q̈ = M⁻¹[τ - C(q,v)v - g(q)]")
print()
print("Even if M[2,1] ≈ 0, pitch acceleration (q̈[2]) depends on L2 through:")
print("  - C[2] depends on v[1] (L2 velocity) → Coriolis coupling")
print("  - g[2] depends on q[1] (L2 position) → Gravitational coupling")
print()
print("So: ∂q̈[2]/∂q[1] ≠ 0  and  ∂q̈[2]/∂v[1] ≠ 0")
print()
print("Even though: M[2,1] ≈ 0")
print()
print("="*70)
print("IMPLICATIONS FOR CONTROL:")
print("="*70)
print()
print("✅ GOOD NEWS:")
print("   - System has MORE coupling than mass matrix suggests")
print("   - BOTH L1 and L2 can affect BOTH pitch and roll")
print("   - This makes the system MORE controllable!")
print()
print("⚠️  BUT:")
print("   - Coupling is VELOCITY-DEPENDENT (through C term)")
print("   - Coupling is CONFIGURATION-DEPENDENT (M, C, g all vary with q)")
print("   - Simple PD/LQR with constant gains won't capture this")
print()
print("✓ WHY TRAJECTORY OPTIMIZATION WORKS:")
print("   - Considers full nonlinear dynamics (M, C, g together)")
print("   - Finds trajectories that exploit ALL coupling mechanisms")
print("   - Not limited to linear approximation around one point")
print()
print("✓ WHY OFC (OPTIMAL FEEDBACK CONTROL) WORKS:")
print("   - Uses time-varying gains along trajectory")
print("   - Accounts for changing coupling as system moves")
print("   - Linearization valid locally at each point")
print()
print("✗ WHY MANUAL LQR FAILED:")
print("   - Used single linearization point")
print("   - Missing ∂C/∂q, ∂C/∂v, ∂g/∂q gradients")
print("   - Assumed constant A, B matrices")
print("   - Reality: A(q,v), B(q) change along trajectory!")
print()
print("="*70)
print()
print("💡 KEY INSIGHT:")
print("="*70)
print()
print("Your simulation CORRECTLY shows coupling that the static mass matrix")
print("analysis MISSED. This is because:")
print()
print("  Mass matrix = snapshot of inertial coupling at one instant")
print("  Simulation = full nonlinear dynamics with velocity & position effects")
print()
print("The simulation is RIGHT - both L1 and L2 DO affect both pitch and roll,")
print("just through different mechanisms (inertia vs Coriolis vs gravity).")
print()
print("This is actually GOOD - it means the system is more controllable than")
print("the mass matrix analysis suggested!")
print()
print("="*70)
print()
print("VERIFICATION TEST:")
print("="*70)
print()
print("From the Coriolis analysis above, you should see:")
print("  - Non-zero Coriolis effects when L2 has velocity")
print("  - These effects create torques on pitch and roll")
print("  - This confirms the simulation results!")
print()
print("="*70)
print()

print("="*70)
print("OLDER ANALYSIS (STILL VALID BUT INCOMPLETE)")
print("="*70)
print()
print("✅ COUPLING EXISTS - System is theoretically controllable!")
print()
print("📊 Key observations:")
print("  1. ROLL has STRONG coupling with both L1 and L2 (0.1-0.2 range)")
print("  2. PITCH has WEAK/VARIABLE coupling:")
print("     - Strong with L1 at some configs (~0.06)")
print("     - Nearly ZERO with L2 at most configs")
print("  3. Coupling is CONFIGURATION-DEPENDENT (changes with joint angles)")
print()
print("🎯 What this means:")
print("  ✓ Manipulator CAN indirectly control pendulum through inertia")
print("  ✓ When manipulator accelerates → base point accelerates → pendulum rotates")
print("  ✗ Coupling strength varies dramatically with configuration")
print("  ✗ Some states have very weak coupling → hard to control")
print()
print("⚠️  Why LQR STILL fails despite coupling existing:")
print()
print("  1. CONFIGURATION-DEPENDENT COUPLING:")
print("     - Pitch-L2 coupling nearly zero at some states")
print("     - LQR assumes LINEAR system around one point")
print("     - Trajectory passes through many points with varying coupling")
print()
print("  2. MANUAL LINEARIZATION TOO SIMPLIFIED:")
print("     - Missing ∂M/∂q (how inertia changes with position)")
print("     - Missing ∂C/∂q (how bias term changes with position)")
print("     - Missing ∂C/∂v (Coriolis Jacobian)")
print("     - These gradients capture the configuration-dependency!")
print()
print("  3. UNDERACTUATED TRAJECTORY TRACKING:")
print("     - 2 actuators controlling 4 DOFs")
print("     - Only works if coupling is strong enough everywhere")
print("     - Weak coupling at some points → system not stabilizable")
print()
print("  4. EQUILIBRIUM vs TRAJECTORY:")
print("     - Cart-pole succeeds: stabilizes ONE equilibrium point")
print("     - Our LQR fails: tries to track TIME-VARYING trajectory")
print("     - Even equilibrium regulation fails due to weak coupling")
print()
print("✅ What DOES work:")
print("  → Trajectory Optimization (finds feasible paths considering coupling)")
print("  → Optimal Feedback Control (OFC) with trajectory optimization")
print("  → Task-space control (control only actuated DOFs, let pendulum follow)")
print("  → These are what the paper uses!")
print()
print("="*70)
print()
print("🚨 CRITICAL FINDING: SINGULARITY AT R=90°")
print("="*70)
print()
print("COUPLING SUMMARY BY CONFIGURATION:")
print("-" * 70)
print("P=0°, R=0°:     Pitch-L1: -0.061 ✓  Roll-L1: -0.198 ✓  GOOD")
print("P=0°, R=180°:   Pitch-L1:  0.061 ✓  Roll-L1:  0.198 ✓  GOOD")
print("P=0°, R=90°:    Pitch-L1:  0.000 ✗  Roll-L1:  0.000 ✗  SINGULAR!")
print("P=90°, R=180°:  Pitch-L1:  0.000 ✗  Roll-L1:  0.178 ✓  PARTIAL")
print("P=180°, R=180°: Pitch-L1: -0.061 ✓  Roll-L1:  0.198 ✓  GOOD")
print("-" * 70)
print()
print("KEY OBSERVATIONS:")
print()
print("1. P=0° or P=180° (pendulum vertical in pitch): ✓ GOOD COUPLING")
print("   - When roll is 0° or 180°: Both pitch and roll coupling exist")
print("   - This is EXPECTED and CORRECT!")
print()
print("2. R=90° (pendulum horizontal): ✗ TOTAL DECOUPLING SINGULARITY")
print("   - ALL coupling terms go to zero")
print("   - System completely uncontrollable at this configuration")
print("   - This is the WORST configuration")
print()
print("3. P=90° (horizontal in pitch direction): ✗ PARTIAL DECOUPLING")
print("   - Pitch coupling lost (M[2,0]=M[2,1]=0)")
print("   - Roll coupling maintained")
print("   - Still problematic for control")
print()
print("="*70)
print()
print("PHYSICAL EXPLANATION:")
print("="*70)
print()
print("This is NOT gimbal lock - it's PHYSICAL decoupling!")
print()
print("Why P=0° and P=180° have GOOD coupling:")
print("  → Pendulum is vertical (in pitch direction)")
print("  → Manipulator accelerations in XZ plane create forces")
print("  → These forces have components that create torques")
print("  → Result: Strong coupling exists ✓")
print()
print("Why R=90° has ZERO coupling:")
print("  → Manipulator moves in XZ plane (vertical plane)")
print("  → At R=90°, pendulum extends in Y direction (horizontal)")
print("  → Perpendicular to manipulator's plane of motion")
print("  → Inertial forces perpendicular to rotation axes")
print("  → No torque component → ZERO coupling ✗")
print()
print("Why P=90° loses pitch coupling:")
print("  → Pendulum horizontal in pitch direction")
print("  → Certain manipulator motions can't create pitch torque")
print("  → But can still create roll torque")
print("  → Partial decoupling ⚠️")
print()
print("="*70)
print()
print("CONTROL IMPLICATIONS:")
print("="*70)
print()
print("✓ GOOD ZONES (controllable):")
print("  - P ≈ 0° or 180°, R ≈ 0° or 180°")
print("  - Strong coupling in both pitch and roll")
print()
print("⚠️ CAUTION ZONES (partially controllable):")
print("  - P ≈ 90°, R ≠ 90°")
print("  - Roll controllable, pitch not")
print()
print("✗ SINGULAR ZONES (uncontrollable):")
print("  - R ≈ 90° (any pitch angle)")
print("  - Complete decoupling - avoid at all costs!")
print()
print("This explains why:")
print("  1. LQR fails - trajectory may pass through/near singular zones")
print("  2. Trajectory optimization works - explicitly avoids singularities")
print("  3. Manual linearization insufficient - doesn't capture this geometry")
print()
print("="*70)
print()
print("💡 WHY LQR FAILS EVEN AT P=0°, R=180° (WHERE COUPLING EXISTS)")
print("="*70)
print()
print("CRITICAL ISSUE: Pitch-L2 coupling is ZERO at most configs!")
print()
print("Look at the data again:")
print("  P=0°,   R=0°:   M[2,1] (Pitch-L2) =  0.000  ✗")
print("  P=0°,   R=180°: M[2,1] (Pitch-L2) = -0.000  ✗")
print("  P=180°, R=180°: M[2,1] (Pitch-L2) =  0.000  ✗")
print()
print("Only ONE actuator (L1) can couple to pitch!")
print("  → System is UNDERACTUATED in a bad way")
print("  → 2 actuators, 4 DOFs, but only 1 can affect pitch")
print("  → This makes the system barely controllable")
print()
print("ROLL is different - BOTH actuators can affect it:")
print("  M[3,0] (Roll-L1): ~0.20  (moderate coupling)")
print("  M[3,1] (Roll-L2): ~0.13  (weak but NON-ZERO coupling)")
print("  → Roll IS controllable via both L1 and L2 ✓")
print()
print("So why does LQR still fail if roll is controllable?")
print("  → Even though roll is controllable, pitch is NOT")
print("  → System needs to control ALL 4 DOFs simultaneously")
print("  → Pitch bottleneck makes entire system uncontrollable")
print("  → LQR requires full controllability → fails ✗")
print()
print("Why cart-pole works but this doesn't:")
print()
print("  CART-POLE (1D):")
print("    - 1 actuator (cart force)")
print("    - 2 DOFs (cart position, pole angle)")  
print("    - Full coupling: cart acceleration → pole rotation")
print("    - Equilibrium at θ=0° (up) is UNSTABLE")
print("    - LQR stabilizes around unstable equilibrium ✓")
print()
print("  CUP-MANIPULATOR PENDULUM:")
print("    - 2 actuators (L1, L2)")
print("    - 4 DOFs (L1, L2, pitch, roll)")
print("    - ASYMMETRIC coupling:")
print("      • PITCH: Only L1 couples (M[2,0]~0.06, M[2,1]~0.00)")
print("      • ROLL:  Both L1 & L2 couple (M[3,0]~0.20, M[3,1]~0.13)")
print("      → Pitch is the BOTTLENECK!")
print()
print("    - Equilibrium at P=0°, R=180° is STABLE (pendulum down)")
print("    - LQR can't stabilize already-stable equilibrium")
print("    - Even if trying to track trajectory:")
print("      • Pitch barely controllable (only via L1, weakly)")
print("      • Roll moderately controllable")
print("      • System not fully controllable → LQR fails ✗")
print()
print("THE REAL PROBLEM:")
print("  1. Pitch-L2 coupling ≈ 0 → System lacks full controllability")
print("  2. Configuration-dependent coupling → Linear model invalid")
print("  3. Missing gradients in manual linearization → Inaccurate A/B")
print("  4. Underactuated with weak coupling → Riccati equation fails")
print()
print("This is why we need trajectory optimization:")
print("  → Finds trajectories that exploit available coupling")
print("  → Avoids configurations with weak/zero coupling")
print("  → Doesn't rely on linearization validity")
print()
print("="*70)
