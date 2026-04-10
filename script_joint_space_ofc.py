"""
Joint-Space Optimal Feedback Control - Standalone Simulation Script
Based on Section C.2 from Razavian et al. (2021)

This script demonstrates the simplified joint-space OFC implementation with:
- d = 0 (no disturbance)
- ε = 0 (no error dynamics)
- ω = 0 (no oscillator dynamics)

Usage:
    python script_joint_space_ofc.py --mode effort --duration 3.0
    python script_joint_space_ofc.py --mode smoothness --duration 4.0
"""

import sys
import numpy as np
import argparse
from pathlib import Path

# Import Drake components
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
)

# Import joint-space OFC controller
from joint_space_ofc_implementation import JointSpaceOFC

# Import configuration types from robot_types
from robot_types import (
    create_cup_manipulator_config,
    create_pendulum_config,
    SimulationConfig,
    VisualizationConfig,
)


# ============================================================================
# COMMAND-LINE ARGUMENTS (must be before imports from main script)
# ============================================================================

parser = argparse.ArgumentParser(description='Joint-Space OFC Simulation (Section C.2)')
parser.add_argument('--mode', type=str, choices=['effort', 'smoothness'],
                    default='smoothness',  # Note: effort mode has stabilizability issues with current formulation
                    help='OFC mode: effort (minimize forces) or smoothness (minimize jerk)')
parser.add_argument('--duration', type=float, default=3.0,
                    help='Motion duration in seconds')
parser.add_argument('--visualize', type=bool, default=True,
                    help='Enable visualization')
parser.add_argument('--Ma', type=float, default=1.0,
                    help='Virtual/impedance mass [kg]')
parser.add_argument('--kp', type=float, default=100.0,
                    help='Spring stiffness [N/m]')
parser.add_argument('--kd', type=float, default=20.0,
                    help='Damping coefficient [N·s/m]')
parser.add_argument('--q_start', type=float, nargs=2, 
                    default=[80.0, -160.0],
                    help='Start joint angles [deg]')
parser.add_argument('--q_goal', type=float, nargs=2,
                    default=[20.0, -40.0],
                    help='Goal joint angles [deg]')
parser.add_argument('--Q_position', type=float, nargs=2,
                    default=[100.0, 100.0],
                    help='LQR position cost weights')
parser.add_argument('--Q_velocity', type=float, nargs=2,
                    default=[10.0, 10.0],
                    help='LQR velocity cost weights')
parser.add_argument('--Q_pendulum', type=float, nargs=2,
                    default=[500.0, 500.0],
                    help='LQR pendulum position cost weights (only used if --include_pendulum)')
parser.add_argument('--Q_pendulum_vel', type=float, nargs=2,
                    default=[50.0, 50.0],
                    help='LQR pendulum velocity cost weights (only used if --include_pendulum)')
parser.add_argument('--R', type=float, nargs=2,
                    default=[0.1, 0.1],
                    help='LQR control cost weights')
parser.add_argument('--tau_filter', type=float, default=0.01,
                    help='F-dot filter time constant [s] (effort mode only)')
parser.add_argument('--include_pendulum', type=lambda x: x.lower() == 'true', default='true',
                    help='Include pendulum states in LQR (true=full-state, false=manip-only baseline)')
parser.add_argument('--sim_time', type=float, default=8.0,
                    help='Total simulation time [s]')

args = parser.parse_args()

# Now import Pendulum3D class from main script
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "controller_module", 
    Path(__file__).parent / "script_cup_manipulator_controller_drake.py"
)
controller_module = importlib.util.module_from_spec(spec)

# Temporarily redirect sys.argv to prevent argparse conflicts
old_argv = sys.argv
sys.argv = ['script_cup_manipulator_controller_drake.py']  # Fake argv to avoid conflicts
try:
    spec.loader.exec_module(controller_module)
finally:
    sys.argv = old_argv

# Extract the Pendulum3D class
Pendulum3D = controller_module.Pendulum3D


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cup Manipulator Configuration
CUP_MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.1, 0.1),
    damping=(0.5, 0.5),
    stiffness=(50.0, 50.0),
    friction=(0.05, 0.05),
)

# Pendulum Configuration
PENDULUM_CONFIG = create_pendulum_config(
    mass=0.5,
    length=0.2,
    radius=0.05,
    damping=0.1,
    attachment_point=(-1.2545, 0.0, -0.188125),
    initial_pitch=0.0,
    initial_roll=180.0,
    name="pendulum"
)

# Visualization Configuration
VISUALIZATION_CONFIG = VisualizationConfig(
    enabled=args.visualize,
    plot_frames=False,
    interactive=True,
    realtime_rate=0.5,
    update_every_step=True,
    print_interval=0.25,
    logging_interval=0.02,
    show_frames=False,
    show_contact_forces=True,
    show_hydroelastic=True,
)

# Simulation Configuration
SIMULATION_CONFIG = SimulationConfig(
    mode=f'ofc-{args.mode}',
    timestep=0.001,
    simulation_time=args.sim_time,
    gravity=(0.0, 0.0, -9.81),
    visualization=VISUALIZATION_CONFIG,
)

# Joint-Space OFC Parameters
OFC_PARAMS = {
    'q_start': np.hstack([np.deg2rad(args.q_start), [0.0, np.deg2rad(180.0)]]),  # [manip, pend]
    'q_goal': np.hstack([np.deg2rad(args.q_goal), [0.0, np.deg2rad(180.0)]]),   # Pendulum hanging down
    'duration': args.duration,
    'mode': args.mode,
    'Ma': args.Ma,
    'kp': args.kp,
    'kd': args.kd,
    'tau_filter': args.tau_filter,
    'include_pendulum': args.include_pendulum,
    'Q_position': np.array(args.Q_position),
    'Q_velocity': np.array(args.Q_velocity),
    'Q_pendulum': np.array(args.Q_pendulum),
    'Q_pendulum_vel': np.array(args.Q_pendulum_vel),
    'R': np.array(args.R),
}


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def run_joint_space_ofc_simulation():
    """
    Run simulation with joint-space OFC controller.
    
    Architecture:
    1. MultibodyPlant: Physics simulation
    2. JointSpaceOFC: Section C.2 controller (from joint_space_ofc_implementation.py)
    3. SceneGraph + Visualizer: 3D visualization
    
    Data flow:
        Plant.state → OFC.input
        OFC.output → Plant.actuation
    """
    print("="*80)
    print("JOINT-SPACE OPTIMAL FEEDBACK CONTROL SIMULATION")
    print("Section C.2 Implementation (d=ε=ω=0)")
    print("="*80)
    print(f"\nConfiguration: {'FULL-STATE (manip + pendulum)' if args.include_pendulum else 'MANIPULATOR-ONLY (baseline)'}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'effort':
        print(f"F-dot filter: τ={args.tau_filter} s")
    print(f"Duration: {args.duration} s")
    print(f"Start: {args.q_start} deg")
    print(f"Goal: {args.q_goal} deg")
    print(f"Impedance: Ma={args.Ma} kg, kp={args.kp} N/m, kd={args.kd} N·s/m")
    if args.include_pendulum:
        print(f"Q_pendulum: {args.Q_pendulum}")
        print(f"Q_pendulum_vel: {args.Q_pendulum_vel}")
    print("="*80)
    
    # Start visualization
    meshcat = StartMeshcat() if args.visualize else None
    
    # Create diagram builder
    builder = DiagramBuilder()
    
    # Create plant and scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Load manipulator URDF
    print("\n" + "="*80)
    print("LOADING ROBOT MODEL")
    print("="*80)
    
    parser_obj = Parser(plant)
    
    # Add package path for meshes
    urdf_path = Path(CUP_MANIPULATOR_CONFIG.urdf_path)
    package_dir = urdf_path.parent
    parser_obj.package_map().Add("assets", str(package_dir / "assets"))
    
    print(f"Loading URDF: {urdf_path}")
    
    model_instance = parser_obj.AddModels(str(urdf_path))[0]
    
    # *** FIX FOR 11-DOF ISSUE: Weld base to world ***
    # Without welding, the manipulator base floats (7 extra DOF: 3 translation + 4 quaternion)
    # This causes 11 positions, 10 velocities instead of expected 4 DOF (2 arm + 2 pendulum)
    print("\n⚠️  Welding manipulator base to world frame...")
    base_frame = plant.GetBodyByName("base_mount_manipulator", model_instance).body_frame()
    plant.WeldFrames(plant.world_frame(), base_frame)
    print("✓ Base welded - eliminates 7 floating-base DOFs")
    
    # Set joint properties (only damping is available for RevoluteJoint)
    for joint_name, joint_config in CUP_MANIPULATOR_CONFIG.joint_configs.items():
        joint = plant.GetJointByName(joint_name)
        joint.set_default_damping(joint_config.damping)
    
    # Add actuators
    for joint_name in ["link1_base", "link2_link1"]:
        joint = plant.GetJointByName(joint_name, model_instance)
        plant.AddJointActuator(joint_name, joint)
    
    # Add pendulum if enabled
    if PENDULUM_CONFIG:
        print(f"Adding pendulum: mass={PENDULUM_CONFIG.mass} kg, length={PENDULUM_CONFIG.length} m")
        pendulum = Pendulum3D(PENDULUM_CONFIG)
        link2_body = plant.GetBodyByName("link2", model_instance)
        pendulum.attach_to_body(plant, link2_body, model_instance)
    
    # Finalize plant
    plant.Finalize()
    print(f"✓ Plant finalized: {plant.num_positions()} positions, {plant.num_velocities()} velocities")
    
    # Add visualizer
    if meshcat:
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat,
            MeshcatVisualizerParams()
        )
    
    # Create joint-space OFC controller
    print("\n" + "="*80)
    print("INITIALIZING JOINT-SPACE OFC CONTROLLER")
    print("="*80)
    
    ofc_controller = JointSpaceOFC(
        plant=plant,
        **OFC_PARAMS
    )
    
    # Add controller to diagram
    builder.AddSystem(ofc_controller)
    
    # Connect ports: Plant state → OFC input
    builder.Connect(
        plant.get_state_output_port(),
        ofc_controller.get_input_port(0)
    )
    
    # Connect ports: OFC output → Plant actuation
    builder.Connect(
        ofc_controller.get_output_port(0),
        plant.get_actuation_input_port()
    )
    
    print("\n✓ Controller connected to plant")
    print("  Plant state → OFC input")
    print("  OFC output → Plant actuation")
    
    # Build diagram
    diagram = builder.Build()
    
    # Set initial configuration
    print("\n" + "="*80)
    print("SETTING INITIAL CONFIGURATION")
    print("="*80)
    
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Set manipulator to start position  
    n_pos = plant.num_positions()
    q_init = np.zeros(n_pos)
    q_init[0:2] = OFC_PARAMS['q_start']  # Manipulator joints
    # Leave rest as zeros (pendulum will default to its configuration)
    
    plant.SetPositions(plant_context, q_init)
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    print(f"Initial configuration:")
    print(f"  Manipulator: {np.rad2deg(OFC_PARAMS['q_start'])} deg")
    print(f"  Total DOF: {n_pos} positions, {plant.num_velocities()} velocities")
    
    # Create simulator
    print("\n" + "="*80)
    print("STARTING SIMULATION")
    print("="*80)
    
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(SIMULATION_CONFIG.visualization.realtime_rate)
    simulator.Initialize()
    
    # Run simulation
    print(f"\nSimulating for {SIMULATION_CONFIG.simulation_time} seconds...")
    print(f"Trajectory duration: {args.duration} s")
    print(f"Realtime rate: {SIMULATION_CONFIG.visualization.realtime_rate}x")
    
    try:
        simulator.AdvanceTo(SIMULATION_CONFIG.simulation_time)
        print("\n✓ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Simulation error: {e}")
        raise
    
    # Final state report
    print("\n" + "="*80)
    print("FINAL STATE")
    print("="*80)
    
    final_state = plant.get_state_output_port().Eval(plant_context)
    q_final = final_state[0:plant.num_positions()]
    v_final = final_state[plant.num_positions():]
    
    print(f"Manipulator position: {np.rad2deg(q_final[0:2])} deg")
    print(f"Manipulator velocity: {np.rad2deg(v_final[0:2])} deg/s")
    
    # Goal comparison
    q_error = np.rad2deg(q_final[0:2] - OFC_PARAMS['q_goal'])
    print(f"\nGoal tracking error: {q_error} deg")
    print(f"Position error norm: {np.linalg.norm(q_error):.3f} deg")
    
    if args.visualize:
        print("\n" + "="*80)
        print("Visualization running. Close browser tab to exit.")
        print("="*80)
        input("Press Enter to exit...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    try:
        run_joint_space_ofc_simulation()
    except KeyboardInterrupt:
        print("\n\n✗ Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
