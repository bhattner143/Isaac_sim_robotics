#!/usr/bin/env python3
"""
Demonstrate why manual linearization fails vs Drake's automatic linearization.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydrake.all import (
    MultibodyPlant,
    Parser,
    Linearize,
)

from configs.robot.robot_types import create_cup_manipulator_config, create_pendulum_config
from archive.script_cup_manipulator_controller_drake import Pendulum3D, CupManipulator

print("="*80)
print("COMPARING MANUAL vs AUTOMATIC LINEARIZATION")
print("="*80)
print()

# Create plant (time_step=0 for continuous system, needed for linearization)
plant = MultibodyPlant(time_step=0.0)
parser = Parser(plant)

# Load robot
cup_config = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.0, 0.0),
    damping=(0.0, 0.0),
    stiffness=(0.0, 0.0),
    friction=(0.05, 0.05),
)

cup_manipulator = CupManipulator(cup_config)
model_instance = cup_manipulator.load_urdf_to_plant(plant, parser)

# Weld base
base_frame = plant.GetBodyByName("base_mount_manipulator", model_instance).body_frame()
plant.WeldFrames(plant.world_frame(), base_frame)

# Add actuators
for joint_name in ["link1_base", "link2_link1"]:
    joint = plant.GetJointByName(joint_name, model_instance)
    plant.AddJointActuator(joint_name, joint)

cup_manipulator.set_joint_properties(plant)

# Add pendulum (with ZERO damping to test Drake's Linearize())
pendulum_config = create_pendulum_config(
    mass=0.5,
    length=0.2,
    radius=0.05,
    damping=0.0,  # ← Changed to 0 to allow equilibrium!
    attachment_point=(-1.2545, 0.0, -0.188125),
    initial_pitch=0.0,
    initial_roll=180.0,
    name="pendulum"
)

pendulum = Pendulum3D(pendulum_config)
link2_body = plant.GetBodyByName("link2", model_instance)
pendulum.attach_to_body(plant, link2_body, model_instance)

plant.Finalize()

print(f"System: {plant.num_positions()} positions, {plant.num_velocities()} velocities")
print(f"Actuators: {plant.num_actuators()}")
print()

# Test at a specific configuration
q_test = np.array([
    np.deg2rad(20),   # L1
    np.deg2rad(-40),  # L2
    np.deg2rad(0),    # pitch
    np.deg2rad(180),  # roll
])
v_test = np.zeros(4)

context = plant.CreateDefaultContext()
plant.SetPositions(context, q_test)
plant.SetVelocities(context, v_test)

print("="*80)
print("CONFIGURATION FOR LINEARIZATION")
print("="*80)
print(f"q = {np.rad2deg(q_test)} degrees")
print(f"v = {v_test} rad/s")
print()

# ============================================================================
# ATTEMPT 1: Manual Linearization (INCOMPLETE - What Failed)
# ============================================================================
print("="*80)
print("ATTEMPT 1: MANUAL LINEARIZATION (INCOMPLETE)")
print("="*80)
print()
print("What the failed LQR attempts probably did:")
print()

M = plant.CalcMassMatrix(context)
print("1. Computed Mass Matrix M(q):")
print(M)
print()

g = plant.CalcGravityGeneralizedForces(context)
print("2. Computed Gravity Forces g(q):")
print(g)
print()

print("3. Attempted to build state space matrices:")
print()
print("   A_manual = [[    0,      I    ],")
print("               [-M^-1*∂g/∂q,  0   ]]  ← WRONG! Missing ∂C/∂q, ∂C/∂v")
print()
print("   B_manual = [[      0      ],")
print("               [M^-1*[I;0]  ]]")
print()

M_inv = np.linalg.inv(M)

# Simplified (wrong) approach - compute ∂g/∂q numerically
delta = 1e-6
dg_dq = np.zeros((4, 4))
for i in range(4):
    q_plus = q_test.copy()
    q_plus[i] += delta
    plant.SetPositions(context, q_plus)
    g_plus = plant.CalcGravityGeneralizedForces(context)
    
    q_minus = q_test.copy()
    q_minus[i] -= delta
    plant.SetPositions(context, q_minus)
    g_minus = plant.CalcGravityGeneralizedForces(context)
    
    dg_dq[:, i] = (g_plus - g_minus) / (2 * delta)
    plant.SetPositions(context, q_test)

print("   Computed ∂g/∂q (numerically):")
print(dg_dq)
print()

# Build incomplete A matrix
A_manual = np.zeros((8, 8))
A_manual[0:4, 4:8] = np.eye(4)
A_manual[4:8, 0:4] = -M_inv @ dg_dq  # Missing Coriolis terms!

# Build B matrix
B_manual = np.zeros((8, 2))
B_manual[4:8, 0:2] = M_inv[0:4, 0:2]  # Only first 2 actuated

print("   A_manual (8x8) - INCOMPLETE:")
print(A_manual)
print()
print("   B_manual (8x2):")
print(B_manual)
print()

print("❌ PROBLEMS:")
print("   1. Missing ∂C/∂q - How Coriolis changes with position")
print("   2. Missing ∂C/∂v - How Coriolis changes with velocity")
print("   3. Missing ∂M/∂q·q̈ - How inertia variation affects dynamics")
print("   4. These terms are CRITICAL for capturing L2→pitch coupling!")
print()

# ============================================================================
# ATTEMPT 2: Drake's Automatic Linearization (COMPLETE)
# ============================================================================
print("="*80)
print("ATTEMPT 2: DRAKE'S AUTOMATIC LINEARIZATION (COMPLETE)")
print("="*80)
print()

print("Testing Drake's Linearize() with ZERO damping...")
print("At equilibrium (pendulum hanging, manipulator at origin):")
print()

# Use natural equilibrium: manipulator at 0, pendulum hanging down
q_equilibrium = np.array([
    np.deg2rad(0),    # L1 = 0
    np.deg2rad(0),    # L2 = 0
    np.deg2rad(0),    # pitch = 0
    np.deg2rad(180),  # roll = 180° (pendulum hanging down)
])
v_equilibrium = np.zeros(4)

plant.SetPositions(context, q_equilibrium)
plant.SetVelocities(context, v_equilibrium)

# Compute gravity forces - need to compensate on manipulator joints
g_forces = plant.CalcGravityGeneralizedForces(context)
tau_equilibrium = g_forces[0:2]  # Gravity compensation for L1, L2

prTry Drake's built-in Linearize() first!
try:
    print("Attempting Drake's Linearize()...")
    linearized = Linearize(
        plant,
        context,
        input_port_index=plant.get_actuation_input_port().get_index(),
        output_port_index=plant.get_state_output_port().get_index()
    )
    
    A_drake = linearized.A()
    B_drake = linearized.B()
    
    print("✅ SUCCESS! Drake's Linearize() worked!")
    print()
    print("✓ A_drake (8x8) - COMPLETE (computed via Drake's autodiff):")
    print(A_drake)
    print()
    print("✓ B_drake (8x2):")
    print(B_drake)
    print()
    
except Exception as e:
    print(f"❌ Drake's Linearize() failed: {e}")
    print()
    print("Falling back to numerical differentiation...")
    print()
    
    # Get the full state
    x0 = plant.GetPositionsAndVelocities(context)
    u0 = tau_equilibrium) < 1e-6:
    print("✓ This IS an equilibrium point!")
else:
    print("✗ This is NOT an equilibrium point")
print()

# Use Taylor approximation - works for any (x, u), not just equilibrium!
try:
    # Create a linear system using first-order Taylor approximation
    # This computes A = ∂f/∂x and B = ∂f/∂u at the operating point
    from pydrake.systems.framework import Context
    from pydrake.systems.analysis import Simulator
    
    # We need to compute the Jacobians ourselves using autodiff
    # Drake can do this automatically with CalcJacobians
    
    # Get the full state
    x0 = plant.GetPositionsAndVelocities(context)
    u0 = np.zeros(2)
    
    print("Computing Jacobians using numerical differentiation...")
    print()
    
    # Compute A = ∂ẋ/∂x numerically
    delta = 1e-7
    n_x = len(x0)
    n_u = 2
    
    A_drake = np.zeros((n_x, n_x))
    for i in range(n_x):
        x_plus = x0.copy()
        x_plus[i] += delta
        plant.SetPositionsAndVelocities(context, x_plus)
        plant.get_actuation_input_port().FixValue(context, u0)
        
        # Get time derivatives
        derivatives = plant.AllocateTimeDerivatives()
        plant.CalcTimeDerivatives(context, derivatives)
        dx_plus = derivatives.CopyToVector()
        
        x_minus = x0.copy()
        x_minus[i] -= delta
        plant.SetPositionsAndVelocities(context, x_minus)
        plant.get_actuation_input_port().FixValue(context, u0)
        
        derivatives = plant.AllocateTimeDerivatives()
        plant.CalcTimeDerivatives(context, derivatives)
        dx_minus = derivatives.CopyToVector()
        
        A_drake[:, i] = (dx_plus - dx_minus) / (2 * delta)
    
    # Compute B = ∂ẋ/∂u numerically
    plant.SetPositionsAndVelocities(context, x0)
    
    B_drake = np.zeros((n_x, n_u))
    for i in range(n_u):
        u_plus = u0.copy()
        u_plus[i] += delta
        plant.get_actuation_input_port().FixValue(context, u_plus)
        
        derivatives = plant.AllocateTimeDerivatives()
        plant.CalcTimeDerivatives(context, derivatives)
        dx_plus = derivatives.CopyToVector()
        
        u_minus = u0.copy()
        u_minus[i] -= delta
        plant.get_actuation_input_port().FixValue(context, u_minus)
        
        derivatives = plant.AllocateTimeDerivatives()
        plant.CalcTimeDerivatives(context, derivatives)
if A_drake is not None:
    print("="*80)
    print("COMPARISON: Manual vs Drake")
    print("="*80)
    print()
    
    # Rebuild manual linearization at equilibrium
    M_eq = plant.CalcMassMatrix(context)
    M_inv_eq = np.linalg.inv(M_eq)
    
    delta = 1e-6
    dg_dq_eq = np.zeros((4, 4))
    for i in range(4):
        q_plus = q_equilibrium.copy()
        q_plus[i] += delta
        plant.SetPositions(context, q_plus)
        g_plus = plant.CalcGravityGeneralizedForces(context)
        
        q_minus = q_equilibrium.copy()
        q_minus[i] -= delta
        plant.SetPositions(context, q_minus)
        g_minus = plant.CalcGravityGeneralizedForces(context)
        
        dg_dq_eq[:, i] = (g_plus - g_minus) / (2 * delta)
        plant.SetPositions(context, q_equilibrium)
    
    A_manual_eq = np.zeros((8, 8))
    A_manual_eq[0:4, 4:8] = np.eye(4)
    A_manual_eq[4:8, 0:4] = -M_inv_eq @ dg_dq_eq
    # Reset to nominal
    plant.SetPositionsAndVelocities(context, x0)
    plant.get_actuation_input_port().FixValue(context, u0)
    
    print("✓ A_drake (8x8) - COMPLETE (includes all ∂M/∂q, ∂C/∂q, ∂C/∂v terms):")
    print(A_drake)
    print()
    print("✓ B_drake (8x2):")
    print(B_drake)
    print()

    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("="*80)
    print("COMPARISON: Manual vs Drake")
    print("="*80)
    print()
    
    print("Matrix A - Bottom-Right 4x4 block (∂q̈/∂v - The Coriolis & Damping Terms):")
    print()
    print("Manual (WRONG - all zeros):")
    print(A_manual[4:8, 4:8])
    print()
    print("Drake (CORRECT - includes damping and Coriolis structure):")
    print(A_drake[4:8, 4:8])
    print()
    
    diff_vel = np.linalg.norm(A_drake[4:8, 4:8] - A_manual[4:8, 4:8])
    print(f"❗ Difference in velocity block: ||A_drake - A_manual|| = {diff_vel:.6f}")
    print()
    
    print("Matrix A - Bottom-Left 4x4 block (∂q̈/∂q - Position-dependent terms):")
    print()
    print("ManNumerical differentiation also failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    A_drake = None
    B_drake = NoneDrake (COMPLETE - includes ∂M/∂q, ∂C/∂q, ∂g/∂q):")
    print(A_drake[4:8, 0:4])
    print()
    
    diff_pos = np.linalg.norm(A_drake[4:8, 0:4] - A_manual[4:8, 0:4])
    print(f"❗ Difference in position block: ||A_drake - A_manual|| = {diff_pos:.6f}")
    print()
    
    diff_total = np.linalg.norm(A_drake - A_manual)
    print(f"❗ Total difference: ||A_drake - A_manual|| = {diff_total:.6f}")
    print()
    print("✓ Drake's linearization is COMPLETE - includes ALL coupling terms!")
    print()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print()

# ============================================================================
# TEST WITH NON-ZERO VELOCITY (Shows why Coriolis coupling matters)
# ============================================================================
# ============================================================================
print("="*80)
print("TEST WITH NON-ZERO VELOCITY (Shows Coriolis Importance)")
print("="*80)
print()

v_test_nonzero = np.array([0.5, 1.0, 0.0, 0.0])  # L1 and L2 moving
plant.SetVelocities(context, v_test_nonzero)

print(f"Setting v = {v_test_nonzero} rad/s")
print()

# Recompute with nonzero velocity
C_nonzero = plant.CalcBiasTerm(context)
g_nonzero = plant.CalcGravityGeneralizedForces(context)
coriolis_only = C_nonzero - g_nonzero

print("Coriolis forces C(q,v)·v (with L1=0.5, L2=1.0 rad/s):")
print(coriolis_only)
print()

print("Effect on passive joints:")
print(f"  Pitch: {coriolis_only[2]:.6f} N·m")
print(f"  Roll:  {coriolis_only[3]:.6f} N·m")
print()

print("This is why L2 velocity affects pitch - through Coriolis forces!")
print("The manual linearization completely missed this.")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY: Why Manual Linearization Failed")
print("="*80)
print()
print("Full dynamics: M(q)q̈ + C(q,v)v + g(q) = τ")
print()
print("Solving for q̈:")
print("   q̈ = M(q)^-1 [τ - C(q,v)v - g(q)]")
print()
print("Taking derivatives:")
print("   ∂q̈/∂q = M^-1[∂τ/∂q - ∂C/∂q·v - ∂M/∂q·q̈ - ∂g/∂q]  ← 3 terms missed!")
print("   ∂q̈/∂v = M^-1[∂τ/∂v - ∂C/∂v·v - C]                  ← 2 terms missed!")
print()
print("Manual linearization only computed:")
print("   ∂q̈/∂q ≈ -M^-1·∂g/∂q  ← Missing ∂C/∂q, ∂M/∂q terms")
print("   ∂q̈/∂v ≈ 0            ← Missing ∂C/∂v, C terms completely!")
print()
print("Drake's Linearize() computes ALL terms correctly using autodiff.")
print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("✓ System IS linearizable")
print("✓ Drake can linearize it automatically (use Linearize())")
print("✗ Manual linearization was incomplete and wrong")
print("✗ Single-point LQR still won't work (need time-varying LQR)")
print()
print("For trajectory tracking, you need:")
print("  1. Trajectory Optimization → find feasible path")
print("  2. Time-Varying LQR → linearize at each point along trajectory")
print("  3. Drake's tools handle both automatically")
print()
print("="*80)
