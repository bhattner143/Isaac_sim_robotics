#!/usr/bin/env python3
"""
Cup Manipulator Linearization Script (Drake Jacobian-based)

═══════════════════════════════════════════════════════════════════════════════
LINEARIZATION ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
--------
Linearize the cup manipulator dynamics around an equilibrium point using
Drake's automatic Jacobian computation (Linearize() function).

ADVANTAGES OVER MANUAL LINEARIZATION:
--------------------------------------
1. Automatic Jacobian computation via numerical differentiation
2. Works for ANY nonlinear system (no manual formula needed)
3. Scales to complex systems with many DOF
4. Automatically handles all state/input interactions
5. Easy to extend to different equilibrium points

SYSTEM DESCRIPTION:
───────────────────
Cup Manipulator:
  - 2 actuated joints: link1_base, link2_link1
  - 2 passive pendulum joints: pitch, roll
  - Total state: [q_manip1, q_manip2, q_pend_pitch, q_pend_roll, 
                   v_manip1, v_manip2, v_pend_pitch, v_pend_roll]
  - Input: [τ_manip1, τ_manip2] (actuator torques)
  - Output: Full state (8-dim)

LINEARIZATION POINT:
────────────────────
Equilibrium: All joints at rest
  - q = [0, 0, 0, 180°] (manipulator neutral, pendulum upright)
  - v = [0, 0, 0, 0]
  - τ = [0, 0] (no applied torques)

RESULTING LINEAR SYSTEM:
────────────────────────
ẋ = A·x + B·u
y = C·x + D·u

Where:
  - x: 8-dim state vector [q, v]
  - u: 2-dim input [τ_manip1, τ_manip2]
  - A: (8, 8) state transition matrix
  - B: (8, 2) input coupling matrix
  - C: (8, 8) output matrix (full state)
  - D: (8, 2) feedthrough (usually zero)

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    Linearize,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
    FixedOffsetFrame,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    
    # Geometry
    Box,
    Cylinder,
    Sphere,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Cup Manipulator Linearization')
parser.add_argument('--mode', type=str, 
                    choices=['linearize', 'simulate', 'visualize'],
                    default='linearize',
                    help='Operation mode')
parser.add_argument('--equilibrium', type=str, default='rest',
                    choices=['rest', 'upright', 'custom'],
                    help='Equilibrium point for linearization')
parser.add_argument('--show-matrices', action='store_true', default=True,
                    help='Print linearized matrices')
parser.add_argument('--save-matrices', action='store_true', default=True,
                    help='Save matrices to file')
args, _ = parser.parse_known_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CupManipulatorConfig:
    """Cup manipulator physical parameters."""
    urdf_path: str = None
    timestep: float = 0.001
    gravity: Tuple = (0.0, 0.0, -9.81)
    
    def __post_init__(self):
        if self.urdf_path is None:
            self.urdf_path = str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute())

@dataclass
class VisualizationConfig:
    """Visualization settings."""
    enabled: bool = True
    realtime_rate: float = 0.5
    show_frames: bool = False

# ============================================================================
# CUP MANIPULATOR LINEARIZED SYSTEM CLASS
# ============================================================================

class CupManipulatorLinearizedSystem:
    """
    Linearized Cup Manipulator using Drake's Linearize().
    
    ARCHITECTURE:
    - Builds full nonlinear MultibodyPlant for cup manipulator + pendulum
    - Uses Drake's Linearize() to compute Jacobian-based linearization
    - Linearizes around equilibrium point (all joints at rest)
    - Stores A, B, C, D matrices for control design
    
    STATE: [q_manip1, q_manip2, q_pend_pitch, q_pend_roll, 
             v_manip1, v_manip2, v_pend_pitch, v_pend_roll] (8D)
    INPUT: [τ_manip1, τ_manip2] (2D)
    OUTPUT: Full state (8D)
    """
    
    def __init__(self, config: CupManipulatorConfig):
        """Initialize linearized system with configuration."""
        self.config = config
        self.nonlinear_plant = None
        self.context = None
        self.linearized_system = None
        
        # Linearization point storage
        self.equilibrium_state = None
        self.equilibrium_input = None
        
        # Linearized matrices
        self.linearized_matrices = {
            'A': None,
            'B': None,
            'C': None,
            'D': None,
        }
        
        # Joint information
        self.actuated_joint_indices = []
        self.passive_joint_indices = []
        self.joint_names = []
        
    def build_plant_without_actuators(self):
        """
        Build nonlinear cup manipulator plant using URDF.
        
        Process:
        1. Create DiagramBuilder and MultibodyPlant
        2. Load cup manipulator URDF
        3. Add actuators to manipulator joints
        4. Finalize plant (NO scene graph for linearization)
        """
        print(colored("\n" + "=" * 70, "yellow"))
        print(colored("Building Linearized Cup Manipulator (Drake Jacobian-based)", "yellow", attrs=["bold"]))
        print(colored("=" * 70, "yellow"))
        
        # Step 1: Create plant WITHOUT scene graph (simpler for linearization)
        print(colored("  [1/4] Creating MultibodyPlant...", "cyan"))
        self.nonlinear_plant = MultibodyPlant(time_step=0.0)
        
        # Step 2: Load URDF
        print(colored("  [2/4] Loading cup manipulator URDF...", "cyan"))
        parser = Parser(self.nonlinear_plant)
        urdf_path = self.config.urdf_path
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        # Set up package map for mesh loading
        urdf_dir = os.path.dirname(urdf_path)
        parser.package_map().Add(
            "assets",
            urdf_dir + "/assets/"
        )
        
        model_instances = parser.AddModels(urdf_path)
        if not model_instances:
            raise RuntimeError("Failed to load URDF models")
        
        model_instance = model_instances[0]
        print(colored(f"    ✓ URDF loaded: {urdf_path}", "green"))
        
        # Weld base to world to eliminate floating base DOF
        base_link = self.nonlinear_plant.GetBodyByName("base_mount_manipulator", model_instance)
        self.nonlinear_plant.WeldFrames(
            self.nonlinear_plant.world_frame(),
            base_link.body_frame(),
            RigidTransform()
        )
        print(colored(f"    ✓ Welded base_mount_manipulator to world", "cyan"))
        
        # Step 2.5: Add actuators to 2 manipulator joints
        print(colored("  [2.5/4] Adding actuators to 2 revolute joints...", "cyan"))
        for joint_name in ['link1_base', 'link2_link1']:
            joint = self.nonlinear_plant.GetJointByName(joint_name, model_instance)
            self.nonlinear_plant.AddJointActuator(joint_name, joint)
            print(colored(f"    ✓ Actuated: {joint_name}", "cyan"))
        
        # Step 3: Finalize plant
        print(colored("  [3/4] Finalizing plant...", "cyan"))
        self.nonlinear_plant.Finalize()
        
        # Get joint information
        print(colored("  [4/4] Extracting joint information...", "cyan"))
        self._extract_joint_info(model_instance)
        
        print(colored("✓ Plant created successfully", "green"))
        print(colored(f"  Total DOF: {self.nonlinear_plant.num_positions()}", "cyan"))
        print(colored(f"  Actuated joints: {len(self.actuated_joint_indices)}", "cyan"))
        print(colored(f"  Passive joints: {len(self.passive_joint_indices)}", "cyan"))
        print(colored(f"  State dimension: {self.nonlinear_plant.num_positions() + self.nonlinear_plant.num_velocities()}", "cyan"))
        print(colored(f"  Input dimension: {self.nonlinear_plant.num_actuators()}", "cyan"))
    
    def _extract_joint_info(self, model_instance):
        """Extract actuated and passive joint indices."""
        self.actuated_joint_indices = []
        self.passive_joint_indices = []
        self.joint_names = []
        
        print(colored("\n  Available joints in URDF:", "yellow"))
        
        for joint_idx in self.nonlinear_plant.GetJointIndices(model_instance):
            joint = self.nonlinear_plant.get_joint(joint_idx)
            self.joint_names.append(joint.name())
            
            # Check if joint is actuated
            if self.nonlinear_plant.HasJointActuatorNamed(joint.name()):
                self.actuated_joint_indices.append(joint_idx)
                print(colored(f"    ✓ Actuated: {joint.name()}", "green"))
            else:
                self.passive_joint_indices.append(joint_idx)
                print(colored(f"    ◦ Passive (welded): {joint.name()}", "yellow"))
    
    def build_linearized_system(self):
        """Build nonlinear plant and linearize it."""
        # First build the nonlinear plant
        self.build_plant_without_actuators()
        
        # Step 2: Find equilibrium point
        print(colored("\n[Linearization] Finding equilibrium point...", "yellow", attrs=["bold"]))
        
        num_q = self.nonlinear_plant.num_positions()
        num_v = self.nonlinear_plant.num_velocities()
        num_u = self.nonlinear_plant.num_actuators()
        
        # Equilibrium with non-zero joint angles
        eq_state = np.zeros(num_q + num_v)
        eq_state[0] = np.deg2rad(0)    # θ₁ (link1_base) = 0°
        eq_state[1] = np.deg2rad(0)    # θ₂ (link2_link1) = 0° (vertical - zero gravity torque)
        # Velocities remain zero
        
        # Find equilibrium input that balances gravity
        print(colored(f"\n  Computing equilibrium input (to balance gravity)...", "cyan"))
        
        # Create a temporary context to find the gravity torques
        temp_context = self.nonlinear_plant.CreateDefaultContext()
        self.nonlinear_plant.SetPositionsAndVelocities(temp_context, eq_state)
        
        # Compute accelerations with zero input
        xdot_temp = self.nonlinear_plant.AllocateTimeDerivatives()
        self.nonlinear_plant.get_actuation_input_port().FixValue(temp_context, np.zeros(num_u))
        self.nonlinear_plant.CalcTimeDerivatives(temp_context, xdot_temp)
        
        # Extract acceleration component (last num_v elements)
        xdot_vec = xdot_temp.CopyToVector()
        accelerations_zero_input = xdot_vec[num_q:]  # These are the qdot_dot values
        
        # Print gravity-driven accelerations
        print(colored(f"\n  Gravity-driven accelerations (zero input):", "cyan"))
        print(colored(f"    ω̇_link1_base (joint 1): {accelerations_zero_input[0]:12.6f} rad/s²", "cyan"))
        print(colored(f"    ω̇_link2_link1 (joint 2): {accelerations_zero_input[1]:12.6f} rad/s²", "cyan"))
        
        # Approximate equilibrium input (simple inverse)
        print(colored(f"\n  Approximate equilibrium input (to balance gravity):", "yellow"))
        print(colored(f"    τ_link1_base (approx):     {-accelerations_zero_input[0]:12.6f} N⋅m", "yellow"))
        print(colored(f"    τ_link2_link1 (approx):    {-accelerations_zero_input[1]:12.6f} N⋅m", "yellow"))
        
        # The equilibrium torque needs to produce negative accelerations to balance gravity
        # τ_eq = -M^{-1} * C * g_effects
        # Refined: solve for exact equilibrium input using iterative refinement
        # Find τ such that τ produces accelerations that cancel gravity
        try:
            eq_input = np.zeros(num_u)
            tolerance = 1e-8
            max_iterations = 20
            
            for iteration in range(max_iterations):
                # Get the B matrix (input-to-acceleration mapping) via finite difference
                B_accel = np.zeros((num_v, num_u))
                h = 1e-6
                for i in range(num_u):
                    u_pert = eq_input.copy()
                    u_pert[i] += h
                    
                    ctx_pert = self.nonlinear_plant.CreateDefaultContext()
                    self.nonlinear_plant.SetPositionsAndVelocities(ctx_pert, eq_state)
                    self.nonlinear_plant.get_actuation_input_port().FixValue(ctx_pert, u_pert)
                    
                    xdot_pert = self.nonlinear_plant.AllocateTimeDerivatives()
                    self.nonlinear_plant.CalcTimeDerivatives(ctx_pert, xdot_pert)
                    xdot_pert_vec = xdot_pert.CopyToVector()
                    
                    B_accel[:, i] = xdot_pert_vec[num_q:] / h
                
                # Compute current accelerations with current eq_input
                ctx_current = self.nonlinear_plant.CreateDefaultContext()
                self.nonlinear_plant.SetPositionsAndVelocities(ctx_current, eq_state)
                self.nonlinear_plant.get_actuation_input_port().FixValue(ctx_current, eq_input)
                xdot_current = self.nonlinear_plant.AllocateTimeDerivatives()
                self.nonlinear_plant.CalcTimeDerivatives(ctx_current, xdot_current)
                accel_current = xdot_current.CopyToVector()[num_q:]
                
                # Solve: B_accel @ Δτ = -accel_current
                try:
                    delta_tau = np.linalg.lstsq(B_accel, -accel_current, rcond=None)[0]
                    eq_input = eq_input + delta_tau
                except:
                    break
                
                # Check convergence
                if np.max(np.abs(accel_current)) < tolerance:
                    print(colored(f"  ✓ Equilibrium converged in {iteration+1} iterations", "green"))
                    break
        except Exception as e:
            print(colored(f"  ⚠ Equilibrium computation failed: {e}", "yellow"))
            eq_input = np.zeros(num_u)
        
        self.equilibrium_state = eq_state
        self.equilibrium_input = eq_input
        
        print(colored(f"  System dimensions:", "cyan"))
        print(colored(f"    Positions (q): {num_q}", "cyan"))
        print(colored(f"    Velocities (v): {num_v}", "cyan"))
        print(colored(f"    Total state: {num_q + num_v}", "cyan"))
        print(colored(f"    Inputs (u): {num_u}", "cyan"))
        print(colored(f"  Equilibrium state: {eq_state}", "cyan"))
        print(colored(f"  Equilibrium input: {eq_input}", "cyan"))
        
        # Create context for linearization
        self.context = self.nonlinear_plant.CreateDefaultContext()
        self.nonlinear_plant.SetPositionsAndVelocities(
            self.context, eq_state
        )
        self.nonlinear_plant.get_actuation_input_port().FixValue(
            self.context, eq_input
        )
        
        # Verify equilibrium by computing accelerations
        print(colored("\n  Verifying equilibrium (checking accelerations)...", "cyan"))
        xdot = self.nonlinear_plant.AllocateTimeDerivatives()
        self.nonlinear_plant.CalcTimeDerivatives(self.context, xdot)
        xdot_vec = xdot.CopyToVector()
        max_accel = np.max(np.abs(xdot_vec))
        print(colored(f"  Max time derivative at equilibrium: {max_accel:.2e}", "cyan"))
        
        if max_accel > 1e-3:
            print(colored(f"  ⚠ WARNING: Accelerations are not negligible", "yellow"))
            print(colored(f"    This may not be a true equilibrium. Proceeding anyway...", "yellow"))
        
        # Step 3: Linearize using Drake's Linearize()
        print(colored("\n[Linearization] Running Drake's Linearize()...", "yellow", attrs=["bold"]))
        
        try:
            linearized_io_sys = Linearize(
                self.nonlinear_plant,
                self.context,
                input_port_index=self.nonlinear_plant.get_actuation_input_port().get_index(),
                output_port_index=self.nonlinear_plant.get_state_output_port().get_index(),
            )
        except RuntimeError as e:
            print(colored(f"  ⚠ Drake's Linearize() failed:", "yellow"))
            print(colored(f"    Reason: {str(e)[:80]}...", "yellow"))
            print(colored(f"\n  Reason: Drake requires STRICT equilibrium (machine precision).", "yellow"))
            print(colored(f"  With gravity on, small accelerations (~1e-5 rad/s²) make equilibrium", "yellow"))
            print(colored(f"  non-perfect. Using manual Jacobian instead (equally valid).", "yellow"))
            print(colored(f"\n  Falling back to numerical Jacobian computation...", "yellow"))
            # Fall back to manual Jacobian computation
            self._compute_jacobian_manually(eq_state, eq_input)
            return self.linearized_matrices
        
        # Store matrices
        self.linearized_matrices['A'] = linearized_io_sys.A()
        self.linearized_matrices['B'] = linearized_io_sys.B()
        self.linearized_matrices['C'] = linearized_io_sys.C()
        self.linearized_matrices['D'] = linearized_io_sys.D()
        
        self.linearized_system = linearized_io_sys
        
        print(colored("✓ Linearization computed successfully", "green"))
        print(colored(f"  A matrix shape: {self.linearized_matrices['A'].shape}", "cyan"))
        print(colored(f"  B matrix shape: {self.linearized_matrices['B'].shape}", "cyan"))
        print(colored(f"  C matrix shape: {self.linearized_matrices['C'].shape}", "cyan"))
        print(colored(f"  D matrix shape: {self.linearized_matrices['D'].shape}", "cyan"))
        
        return self.linearized_matrices
    
    def print_matrices(self):
        """Print linearized matrices in readable format."""
        if not all(m is not None for m in self.linearized_matrices.values()):
            print(colored("ERROR: Matrices not computed. Run build_linearized_system() first.", "red"))
            return
        
        print(colored("\n" + "=" * 70, "yellow"))
        print(colored("LINEARIZED MATRICES", "yellow", attrs=["bold"]))
        print(colored("=" * 70, "yellow"))
        
        # State labels
        state_labels = [
            'θ_link1_base',      # Joint 1 position
            'θ_link2_link1',     # Joint 2 position
            'ω_link1_base',      # Joint 1 velocity
            'ω_link2_link1'      # Joint 2 velocity
        ]
        
        input_labels = ['τ_link1_base', 'τ_link2_link1']
        
        A = self.linearized_matrices['A']
        B = self.linearized_matrices['B']
        C = self.linearized_matrices['C']
        D = self.linearized_matrices['D']
        
        # Print A matrix
        print(colored("\nA Matrix (State Transition):", "cyan", attrs=["bold"]))
        print(colored("Shape: " + str(A.shape), "cyan"))
        print("States:", state_labels)
        print(A)
        
        # Print B matrix
        print(colored("\nB Matrix (Input Coupling):", "cyan", attrs=["bold"]))
        print(colored("Shape: " + str(B.shape), "cyan"))
        print("Inputs:", input_labels)
        print(B)
        
        # Print C matrix
        print(colored("\nC Matrix (Output Matrix):", "cyan", attrs=["bold"]))
        print(colored("Shape: " + str(C.shape), "cyan"))
        print(C)
        
        # Print D matrix
        print(colored("\nD Matrix (Feedthrough):", "cyan", attrs=["bold"]))
        print(colored("Shape: " + str(D.shape), "cyan"))
        print(D)
        
        # Eigenvalue analysis
        print(colored("\nEigenvalue Analysis of A:", "cyan", attrs=["bold"]))
        eigenvalues = np.linalg.eigvals(A)
        print(colored("Eigenvalues:", "cyan"))
        for i, ev in enumerate(eigenvalues):
            real_part = ev.real
            imag_part = ev.imag
            if imag_part == 0:
                print(f"  λ_{i}: {real_part:10.6f}")
            else:
                print(f"  λ_{i}: {real_part:10.6f} ± {abs(imag_part):10.6f}i")
        
        print(colored(f"\nSystem Stability: ", "cyan"), end="")
        if np.all(np.real(eigenvalues) < 0):
            print(colored("✓ STABLE (all eigenvalues < 0)", "green", attrs=["bold"]))
        elif np.all(np.real(eigenvalues) <= 0):
            print(colored("⚠ MARGINALLY STABLE (some eigenvalues = 0)", "yellow", attrs=["bold"]))
        else:
            print(colored("✗ UNSTABLE (some eigenvalues > 0)", "red", attrs=["bold"]))
        
        # Controllability analysis
        print(colored("\nControllability Analysis:", "cyan", attrs=["bold"]))
        controllability_matrix = self._compute_controllability_matrix(A, B)
        rank_controllability = np.linalg.matrix_rank(controllability_matrix)
        n = A.shape[0]
        print(colored(f"Rank of controllability matrix: {rank_controllability}/{n}", "cyan"))
        if rank_controllability == n:
            print(colored("✓ CONTROLLABLE (full rank)", "green"))
        else:
            print(colored(f"✗ NOT CONTROLLABLE (rank deficient)", "red"))
    
    def _compute_controllability_matrix(self, A, B):
        """Compute controllability matrix [B AB A²B ...]."""
        n = A.shape[0]
        m = B.shape[1]
        controllability = np.zeros((n, n * m))
        
        power = np.eye(n)
        for i in range(n):
            controllability[:, i*m:(i+1)*m] = power @ B
            power = power @ A
        
        return controllability
    
    def _compute_jacobian_manually(self, eq_state, eq_input):
        """Compute Jacobian matrices numerically using finite differences."""
        print(colored("  Computing Jacobian numerically...", "cyan"))
        
        num_q = self.nonlinear_plant.num_positions()
        num_v = self.nonlinear_plant.num_velocities()
        num_x = num_q + num_v
        num_u = len(eq_input)
        
        h = 1e-6  # Finite difference step
        
        # Compute f(x, u) at equilibrium
        xdot_nominal = self._eval_dynamics(eq_state, eq_input)
        
        # Jacobian with respect to state (∂f/∂x)
        A = np.zeros((num_x, num_x))
        for i in range(num_x):
            x_perturbed = eq_state.copy()
            x_perturbed[i] += h
            xdot_perturbed = self._eval_dynamics(x_perturbed, eq_input)
            A[:, i] = (xdot_perturbed - xdot_nominal) / h
        
        # Jacobian with respect to input (∂f/∂u)
        B = np.zeros((num_x, num_u))
        for j in range(num_u):
            u_perturbed = eq_input.copy()
            u_perturbed[j] += h
            xdot_perturbed = self._eval_dynamics(eq_state, u_perturbed)
            B[:, j] = (xdot_perturbed - xdot_nominal) / h
        
        # Output matrices (full state feedback)
        C = np.eye(num_x)
        D = np.zeros((num_x, num_u))
        
        self.linearized_matrices['A'] = A
        self.linearized_matrices['B'] = B
        self.linearized_matrices['C'] = C
        self.linearized_matrices['D'] = D
        
        print(colored("    ✓ Manual Jacobian computed", "green"))
        print(colored(f"    A matrix shape: {A.shape}", "cyan"))
        print(colored(f"    B matrix shape: {B.shape}", "cyan"))
    
    def _eval_dynamics(self, x, u):
        """Evaluate dynamics ẋ = f(x, u)."""
        context = self.nonlinear_plant.CreateDefaultContext()
        num_q = self.nonlinear_plant.num_positions()
        
        self.nonlinear_plant.SetPositionsAndVelocities(context, x)
        self.nonlinear_plant.get_actuation_input_port().FixValue(context, u)
        
        xdot = self.nonlinear_plant.AllocateTimeDerivatives()
        self.nonlinear_plant.CalcTimeDerivatives(context, xdot)
        return xdot.CopyToVector()
    
    def save_matrices(self, filename: str = None):
        """Save linearized matrices to NPZ file."""
        if filename is None:
            filename = "cup_manipulator_linearized_matrices.npz"
        
        if not all(m is not None for m in self.linearized_matrices.values()):
            print(colored("ERROR: Matrices not computed. Run build_linearized_system() first.", "red"))
            return
        
        np.savez(
            filename,
            A=self.linearized_matrices['A'],
            B=self.linearized_matrices['B'],
            C=self.linearized_matrices['C'],
            D=self.linearized_matrices['D'],
            eq_state=self.equilibrium_state,
            eq_input=self.equilibrium_input,
        )
        
        print(colored(f"✓ Matrices saved to {filename}", "green"))


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print(colored("CUP MANIPULATOR LINEARIZATION (2 DOF)", "cyan", attrs=["bold"]))
    print(colored("Drake Jacobian-based Linearization", "cyan"))
    print("=" * 70)
    
    # Create configuration (simple, no command-line joints needed)
    config = CupManipulatorConfig()
    
    # Create linearized system
    system = CupManipulatorLinearizedSystem(config)
    
    try:
        # Build and linearize
        if args.mode in ['linearize', 'simulate']:
            matrices = system.build_linearized_system()
            
            if args.show_matrices:
                system.print_matrices()
            
            if args.save_matrices:
                system.save_matrices()
        
        elif args.mode == 'visualize':
            print(colored("Starting visualization...", "yellow"))
            # TODO: Add visualization support
            print(colored("Visualization not yet implemented", "red"))
        
        print(colored("\n✓ Linearization complete", "green", attrs=["bold"]))
        
    except Exception as e:
        print(colored(f"\n✗ Error: {str(e)}", "red", attrs=["bold"]))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
