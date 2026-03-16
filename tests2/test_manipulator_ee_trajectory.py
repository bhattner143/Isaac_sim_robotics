#!/usr/bin/env python3
"""
Test script for manipulator end-effector trajectory tracking.

Prescribed trajectory:
- X: -2.0m → 1.0m over 4 seconds (smooth), then hold at 1.0m until 10s
- Y:  0.0m → 3.0m over 3 seconds (smooth), then hold at 3.0m until 10s

Uses computed torque control for precise tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from termcolor import colored
from scipy.interpolate import interp1d
import csv
import argparse

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    LeafSystem,
    StartMeshcat,
    MeshcatVisualizer,
    Demultiplexer,
    Multiplexer,
    SceneGraph,
    JacobianWrtVariable,
    Parser,
    InverseKinematics,
    Solve,
)
from pydrake.multibody.tree import MultibodyForces

# Import manipulator
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from script_cup_manipulator_controller_ofc import CupManipulator
from configs.robot.robot_types import create_cup_manipulator_config


def smooth_trajectory(t, t_start, t_end, val_start, val_end):
    """
    Smooth trajectory using 5th-order polynomial (zero velocity/accel at endpoints).
    
    Args:
        t: Current time
        t_start: Start time
        t_end: End time
        val_start: Starting value
        val_end: Ending value
    
    Returns:
        value, velocity, acceleration at time t
    """
    if t <= t_start:
        return val_start, 0.0, 0.0
    elif t >= t_end:
        return val_end, 0.0, 0.0
    else:
        # Normalized time [0, 1]
        tau = (t - t_start) / (t_end - t_start)
        
        # 5th-order polynomial: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / (t_end - t_start)
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (t_end - t_start)**2
        
        val = val_start + s * (val_end - val_start)
        vel = s_dot * (val_end - val_start)
        acc = s_ddot * (val_end - val_start)
        
        return val, vel, acc


class PrescribedTrajectorySource(LeafSystem):
    """Outputs prescribed EE trajectory: [x, y, ẋ, ẏ]."""
    
    def __init__(self, x_start=-2.0, x_end=1.0, x_duration=4.0,
                 y_start=0.0, y_end=3.0, y_duration=3.0):
        LeafSystem.__init__(self)
        self.x_start = x_start
        self.x_end = x_end
        self.x_duration = x_duration
        self.y_start = y_start
        self.y_end = y_end
        self.y_duration = y_duration
        
        # Output: [x, y, ẋ, ẏ]
        self.DeclareVectorOutputPort("trajectory", 4, self.calc_trajectory)
    
    def calc_trajectory(self, context, output):
        t = context.get_time()
        
        # X trajectory: -2 → 1 over 4 seconds
        x, x_dot, x_ddot = smooth_trajectory(t, 0.0, self.x_duration, 
                                             self.x_start, self.x_end)
        
        # Y trajectory: 0 → 3 over 3 seconds
        y, y_dot, y_ddot = smooth_trajectory(t, 0.0, self.y_duration,
                                             self.y_start, self.y_end)
        
        output.SetFromVector([x, y, x_dot, y_dot])


class CSVTrajectorySource(LeafSystem):
    """Outputs trajectory from CSV file: [x, y, ẋ, ẏ, ẍ, ÿ]."""
    
    def __init__(self, csv_path):
        LeafSystem.__init__(self)
        self.csv_path = csv_path
        
        # Load CSV data
        print(colored(f"\n📂 Loading trajectory from {csv_path}...", "yellow"))
        times, x_data, y_data, vx_data, vy_data = [], [], [], [], []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row['time_s']))
                x_data.append(float(row['x_m']))
                y_data.append(float(row['y_m']))
                vx_data.append(float(row['vx_m_s']))
                vy_data.append(float(row['vy_m_s']))
        
        times = np.array(times)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        vx_data = np.array(vx_data)
        vy_data = np.array(vy_data)
        
        # Compute accelerations using finite differences
        ax_data = np.gradient(vx_data, times)
        ay_data = np.gradient(vy_data, times)
        
        # Replace any NaN values with 0.0 (happens at duplicate/identical time points)
        ax_data = np.nan_to_num(ax_data, nan=0.0)
        ay_data = np.nan_to_num(ay_data, nan=0.0)
        
        # Create interpolation functions (linear for stability)
        self.x_interp = interp1d(times, x_data, kind='linear', fill_value=(x_data[0], x_data[-1]), bounds_error=False)
        self.y_interp = interp1d(times, y_data, kind='linear', fill_value=(y_data[0], y_data[-1]), bounds_error=False)
        self.vx_interp = interp1d(times, vx_data, kind='linear', fill_value=(vx_data[0], vx_data[-1]), bounds_error=False)
        self.vy_interp = interp1d(times, vy_data, kind='linear', fill_value=(vy_data[0], vy_data[-1]), bounds_error=False)
        self.ax_interp = interp1d(times, ax_data, kind='linear', fill_value=(ax_data[0], ax_data[-1]), bounds_error=False)
        self.ay_interp = interp1d(times, ay_data, kind='linear', fill_value=(ay_data[0], ay_data[-1]), bounds_error=False)
        
        self.t_min = times[0]
        self.t_max = times[-1]
        
        print(colored(f"✓ Loaded {len(times)} trajectory points", "green"))
        print(colored(f"  Time range: [{self.t_min:.3f}, {self.t_max:.3f}] s", "cyan"))
        print(colored(f"  X range: [{np.min(x_data):.3f}, {np.max(x_data):.3f}] m", "cyan"))
        print(colored(f"  Y range: [{np.min(y_data):.3f}, {np.max(y_data):.3f}] m", "cyan"))
        
        # Output: [x, y, ẋ, ẏ]
        self.DeclareVectorOutputPort("trajectory", 4, self.calc_trajectory)
    
    def calc_trajectory(self, context, output):
        t = context.get_time()
        
        # Clamp time to valid range
        t_clamped = np.clip(t, self.t_min, self.t_max)
        
        x = float(self.x_interp(t_clamped))
        y = float(self.y_interp(t_clamped))
        x_dot = float(self.vx_interp(t_clamped))
        y_dot = float(self.vy_interp(t_clamped))
        # Note: Accelerations are computed but not used (controller sets them to 0)
        
        output.SetFromVector([x, y, x_dot, y_dot])


class ComputedTorqueEEController(LeafSystem):
    """
    Computed torque controller for end-effector trajectory tracking.
    
    Inputs:
      0: desired_trajectory (4) = [x_d, y_d, ẋ_d, ẏ_d]
      1: manipulator_state (4) = [q2, q1, q̇2, q̇1]  (Drake GetJointIndices order!)
    Output:
      0: joint_torques (2) = [τ2, τ1]  (Drake GetJointIndices order!)
    """
    
    def __init__(self, manipulator, plant, Kp=200.0, Kd=30.0, tau_max=100.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max
        self.call_count = 0
        
        # Inputs
        self.DeclareVectorInputPort("desired_trajectory", 4)
        self.DeclareVectorInputPort("manipulator_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("joint_torques", 2, self.calc_torques)
    
    def calc_torques(self, context, output):
        # Get inputs
        traj = self.get_input_port(0).Eval(context)
        manip_state = self.get_input_port(1).Eval(context)
        
        x_d, y_d, x_dot_d, y_dot_d = traj
        x_ddot_d, y_ddot_d = 0.0, 0.0  # Zero acceleration
        # CRITICAL: manip_state comes from plant.get_state_output_port(model_instance)
        # which outputs in GetJointIndices() order: [link2_link1, link1_base] = [q2, q1, q̇2, q̇1]
        q2, q1, q2_dot, q1_dot = manip_state
        q_manip = np.array([q2, q1])
        q_dot_manip = np.array([q2_dot, q1_dot])
        
        # Create plant context
        plant_context = self.plant.CreateDefaultContext()
        
        # Set manipulator joint states using GetJointIndices ordering
        manip_joint_indices = list(self.plant.GetJointIndices(self.manipulator.model_instance))
        for i, joint_index in enumerate(manip_joint_indices):
            joint = self.plant.get_joint(joint_index)
            if joint.num_velocities() > 0:
                # q_manip[i] matches GetJointIndices ordering: [0]=q2 (link2_link1), [1]=q1 (link1_base)
                joint.set_angle(plant_context, q_manip[i])
                joint.set_angular_rate(plant_context, q_dot_manip[i])
        
        # Get EE frame and compute current position
        ee_frame = self.plant.GetFrameByName("link2", self.manipulator.model_instance)
        world_frame = self.plant.world_frame()
        EE_OFFSET = self.manipulator.EE_OFFSET
        
        ee_pos = self.plant.CalcPointsPositions(
            plant_context, ee_frame, EE_OFFSET.reshape(3, 1), world_frame
        ).flatten()
        x_current, y_current = ee_pos[0], ee_pos[1]
        
        # Compute Jacobian
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            EE_OFFSET,
            world_frame,
            world_frame
        )
        
        # Extract manipulator velocity indices and Jacobian
        manip_velocity_indices = []
        for joint_index in manip_joint_indices:
            joint = self.plant.get_joint(joint_index)
            if joint.num_velocities() > 0:
                manip_velocity_indices.append(joint.velocity_start())
        
        J_xy = J_full[0:2, manip_velocity_indices]  # 2x2
        
        # Current EE velocity
        ee_vel = J_xy @ q_dot_manip
        
        # Task-space errors
        e_pos = np.array([x_d - x_current, y_d - y_current])
        e_vel = np.array([x_dot_d - ee_vel[0], y_dot_d - ee_vel[1]])
        
        # Desired task-space acceleration with PD feedback
        x_ddot_des_vec = np.array([x_ddot_d, y_ddot_d])
        x_ddot_control = x_ddot_des_vec + self.Kp * e_pos + self.Kd * e_vel
        
        # Map to joint space
        J_pinv = np.linalg.pinv(J_xy)
        q_ddot_desired = J_pinv @ x_ddot_control
        
        # Compute inverse dynamics
        vd = np.zeros(self.plant.num_velocities())
        for i, vel_idx in enumerate(manip_velocity_indices):
            vd[vel_idx] = q_ddot_desired[i]
        
        # Debug: Check if any values are NaN before CalcInverseDynamics
        if self.call_count == 0:
            print(colored(f"\n🔍 DEBUG ComputedTorqueEEController (first call):", "yellow"))
            print(f"  q_manip = {q_manip}")
            print(f"  q_dot_manip = {q_dot_manip}")
            print(f"  Desired trajectory: x={x_d:.3f}, y={y_d:.3f}, ẋ={x_dot_d:.3f}, ẏ={y_dot_d:.3f}")
            print(f"  Current EE: x={x_current:.3f}, y={y_current:.3f}")
            print(f"  Jacobian J_xy:\n{J_xy}")
            print(f"  Position error: {e_pos}")
            print(f"  Velocity error: {e_vel}")
            print(f"  x_ddot_control: {x_ddot_control}")
            print(f"  q_ddot_desired: {q_ddot_desired}")
            print(f"  vd (full): {vd}")
            print(f"  manip_velocity_indices: {manip_velocity_indices}")
        
        external_forces = MultibodyForces(self.plant)
        tau_all = self.plant.CalcInverseDynamics(
            plant_context,
            vd,
            external_forces
        ).flatten()
        
        tau = np.array([tau_all[idx] for idx in manip_velocity_indices])
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Debug output
        self.call_count += 1
        if self.call_count % 100 == 1:
            error_mm = np.linalg.norm(e_pos) * 1000
            sat = " SAT" if np.any(np.abs(tau) >= self.tau_max - 0.1) else ""
            print(f"[t={context.get_time():.2f}s] EE=[{x_current:+5.2f},{y_current:+5.2f}] "
                  f"Desired=[{x_d:+5.2f},{y_d:+5.2f}] Err={error_mm:4.0f}mm "
                  f"τ=[{tau[0]:+5.1f},{tau[1]:+5.1f}]{sat}")
        
        output.SetFromVector(tau)


def solve_initial_pose_via_ik(plant, manipulator, target_xy, q_seed, pos_tol=1e-3):
    """Solve for joint angles that place the EE at target (x, y).

    Keeps the EE z coordinate near the forward-kinematics value of the seed
    configuration to avoid over-constraining a planar arm.
    """
    ik = InverseKinematics(plant)
    ik_context = ik.context()
    # CRITICAL: Drake expects [q2, q1] but q_seed is [q1, q2]
    plant.SetPositions(ik_context, manipulator.model_instance, np.array([q_seed[1], q_seed[0]]))
    ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
    world = plant.world_frame()

    # Use seed pose to set the desired z target and keep a tight x/y box.
    ee_pos_seed = plant.CalcPointsPositions(
        ik_context,
        ee_frame,
        manipulator.EE_OFFSET.reshape(3, 1),
        world,
    ).flatten()
    z_target = ee_pos_seed[2]

    lower = np.array([target_xy[0], target_xy[1], z_target]) - pos_tol
    upper = np.array([target_xy[0], target_xy[1], z_target]) + pos_tol
    ik.AddPositionConstraint(
        frameB=ee_frame,
        p_BQ=manipulator.EE_OFFSET,
        frameA=world,
        p_AQ_lower=lower,
        p_AQ_upper=upper,
    )

    prog = ik.prog()
    q_vars = ik.q()
    prog.AddQuadraticErrorCost(np.eye(plant.num_positions()), q_seed, q_vars)

    result = Solve(prog)
    if result.is_success():
        return result.GetSolution(q_vars), True
    return q_seed, False


def run_test(duration=10.0, dt=0.001, traj_type='generated', csv_path=None):
    """Run the manipulator EE trajectory tracking test.
    
    Args:
        duration: Simulation duration [s]
        dt: Time step [s]
        traj_type: 'generated' or 'csv'
        csv_path: Path to CSV file (required if traj_type='csv')
    """
    
    print(colored("\n" + "="*80, "cyan"))
    print(colored("MANIPULATOR END-EFFECTOR TRAJECTORY TRACKING TEST", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    if traj_type == 'csv':
        if csv_path is None:
            raise ValueError("csv_path must be provided when traj_type='csv'")
        print(f"\nTrajectory: CSV file ({csv_path})")
    else:
        print(f"\nTrajectory: Generated (AGGRESSIVE - mimicking LQR cart motion)")
        print(colored(f"  X: -1.13m → -0.85m over 0.5s (560 mm/s avg)", "yellow"))
        print(colored(f"  Y:  1.74m →  1.25m over 0.5s (980 mm/s avg)", "yellow"))
        print(colored(f"  ⚠️  This is ~10x faster than the previous feasible trajectory!", "red", attrs=["bold"]))
    
    print(f"  Duration: {duration}s")
    print(f"  Control: Computed Torque (Kp=200, Kd=60, τ_max=100 Nm)\n")
    
    # Create meshcat
    meshcat = StartMeshcat()
    
    # Build diagram
    builder = DiagramBuilder()
    plant, scene_graph = MultibodyPlant(time_step=dt), SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    
    # Load manipulator
    print(colored("Loading manipulator...", "yellow"))
    urdf_path = Path(__file__).parent / "model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf"
    manip_config = create_cup_manipulator_config(
        urdf_path=str(urdf_path),
        joint_angles=(np.deg2rad(33.7), np.deg2rad(40.5)),  # IK solution for (-1.13, 1.74)
        damping = (0.5, 0.5),  # NOTE: Has NO effect with computed torque control!
        stiffness= (10.0, 10.0),  # NOTE: Has NO effect with computed torque control!
    )
    print(colored("  ⚠️ Note: Damping/stiffness have NO effect when using computed torque control!", "yellow"))
    print(colored("     These only affect passive dynamics (e.g., free fall, no torque commanded)", "yellow"))
    print(colored(f"  🎯 Starting configuration matches LQR script: q1=33.7°, q2=40.5° → EE≈(-1.13, 1.74)", "cyan"))
    manipulator = CupManipulator(manip_config)
    manipulator.build_in_plant(plant, parser, weld_base=True)
    
    plant.Finalize()
    builder.AddSystem(plant)
    builder.AddSystem(scene_graph)
    
    # Connect scene graph
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Add visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    # Add trajectory source based on type
    if traj_type == 'csv':
        traj_source = builder.AddSystem(CSVTrajectorySource(csv_path))
    else:
        # AGGRESSIVE trajectory mimicking LQR cart motion from main script
        # Starts near manipulator home position and moves fast like LQR does
        traj_source = builder.AddSystem(PrescribedTrajectorySource(
            x_start=-1.13, x_end=-0.85, x_duration=0.5,  # Fast x motion: 0.28m in 0.5s
            y_start=1.74, y_end=1.25, y_duration=0.5     # Fast y motion: 0.49m in 0.5s
        ))
    
    # Add controller
    controller = builder.AddSystem(
        ComputedTorqueEEController(manipulator, plant, Kp=200.0, Kd=60.0, tau_max=100.0)
    )
    
    # Connect: trajectory → controller
    builder.Connect(
        traj_source.get_output_port(), # [x, y, ẋ, ẏ, ẍ, ÿ]
        controller.get_input_port(0), # → controller desired trajectory
    )
    
    # Connect: plant state → controller
    builder.Connect(
        plant.get_state_output_port(manipulator.model_instance), # [q2, q1, q̇2, q̇1]
        controller.get_input_port(1)# → controller manipulator state (note the ordering)
    )
    
    # Connect: controller → plant
    builder.Connect(
        controller.get_output_port(),# [τ1, τ2]
        plant.get_actuation_input_port(manipulator.model_instance) #→ plant joint torques (note the ordering)
    )
    
    # Add loggers
    traj_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(traj_source.get_output_port(), traj_logger.get_input_port())
    
    state_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(
        plant.get_state_output_port(manipulator.model_instance),
        state_logger.get_input_port()
    )
    
    torque_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(controller.get_output_port(), torque_logger.get_input_port())
    
    # Build diagram
    diagram = builder.Build()
    
    # Check IK feasibility for entire trajectory BEFORE creating simulator
    print(colored("\n🔍 Testing IK feasibility along trajectory...", "cyan"))
    test_times = np.arange(0.0, duration + 0.5, 0.5)  # Sample every 0.5s
    feasible_count = 0
    ik_results = []
    
    for t_test in test_times:
        traj_ctx = traj_source.CreateDefaultContext()
        traj_ctx.SetTime(t_test)
        traj_output = traj_source.get_output_port().Eval(traj_ctx)
        x_test, y_test = traj_output[0], traj_output[1]
        
        # Use previous solution as seed (or initial config for first point)
        if ik_results:
            q_seed_ik = ik_results[-1][2]  # Use last successful q
        else:
            q_seed_ik = np.array([
                manip_config.get_joint_position("link1_base"),
                manip_config.get_joint_position("link2_link1"),
            ])
        
        # Solve IK for this point
        q_sol, ik_success = solve_initial_pose_via_ik(
            plant, manipulator, np.array([x_test, y_test]), q_seed_ik, pos_tol=0.01  # Increased tolerance to 10mm
        )
        
        # Verify the solution
        test_context = plant.CreateDefaultContext()
        # CRITICAL: Drake expects [q2, q1] but q_sol from IK is [q1, q2]
        plant.SetPositions(test_context, manipulator.model_instance, np.array([q_sol[1], q_sol[0]]))
        ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
        ee_pos_actual = plant.CalcPointsPositions(
            test_context, ee_frame, 
            manipulator.EE_OFFSET.reshape(3, 1), 
            plant.world_frame()
        ).flatten()
        
        error_mm = np.linalg.norm([x_test - ee_pos_actual[0], y_test - ee_pos_actual[1]]) * 1000
        
        if ik_success and error_mm < 20.0:  # Within 20mm (increased tolerance)
            feasible_count += 1
            status = "✓"
            color = "green"
        else:
            status = "✗"
            color = "red"
        
        ik_results.append((t_test, ik_success, q_sol, error_mm))
        
        q_deg = np.rad2deg(q_sol)
        print(colored(
            f"  t={t_test:4.1f}s: target=({x_test:+5.2f}, {y_test:+5.2f}) "
            f"q=[{q_deg[0]:+5.1f}°, {q_deg[1]:+5.1f}°] err={error_mm:4.1f}mm {status}",
            color
        ))
    
    # Summary
    print(f"\n  IK Feasibility: {feasible_count}/{len(test_times)} points solvable")
    if feasible_count < len(test_times):
        failed = len(test_times) - feasible_count
        print(colored(f"  ⚠️ Warning: {failed} point(s) failed IK!", "yellow"))
        print(colored(f"  → Consider reducing trajectory range or slowing motion", "yellow"))
    else:
        print(colored("  ✓ All trajectory points are IK-feasible!", "green"))
    
    # Create simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Use the IK solution from t=0 for initial configuration
    q_init = ik_results[0][2]  # This is [q1, q2] from IK
    # CRITICAL: Drake's joint ordering is [link2_link1, link1_base] = [q2, q1]
    plant.SetPositions(plant_context, manipulator.model_instance, np.array([q_init[1], q_init[0]]))
    plant.SetVelocities(plant_context, manipulator.model_instance, np.zeros(2))
    
    # Verify actual EE position after setting initial conditions
    ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
    ee_pos_actual = plant.CalcPointsPositions(
        plant_context, ee_frame,
        manipulator.EE_OFFSET.reshape(3, 1),
        plant.world_frame()
    ).flatten()
    
    # Get desired position at t=0
    traj_t0_ctx = traj_source.CreateDefaultContext()
    traj_t0_ctx.SetTime(0.0)
    traj_t0 = traj_source.get_output_port().Eval(traj_t0_ctx)
    x_desired_t0, y_desired_t0 = traj_t0[0], traj_t0[1]
    
    initial_error_mm = np.linalg.norm([x_desired_t0 - ee_pos_actual[0], 
                                        y_desired_t0 - ee_pos_actual[1]]) * 1000
    
    q_init_deg = np.rad2deg(q_init)
    print(colored(
        f"\n✓ Initial configuration set: q1={q_init_deg[0]:+.2f}°, q2={q_init_deg[1]:+.2f}°",
        "cyan",
    ))
    print(colored(
        f"  Desired EE at t=0: ({x_desired_t0:+.3f}, {y_desired_t0:+.3f})",
        "cyan"
    ))
    print(colored(
        f"  Actual  EE at t=0: ({ee_pos_actual[0]:+.3f}, {ee_pos_actual[1]:+.3f})",
        "cyan"
    ))
    print(colored(
        f"  Initial position error: {initial_error_mm:.1f} mm",
        "green" if initial_error_mm < 10 else "yellow"
    ))
    
    # Run simulation
    print(colored("\n🚀 Starting simulation...", "green"))
    print(
        f"   Initial config (IK): q1={q_init_deg[0]:+.2f}°, q2={q_init_deg[1]:+.2f}°"
    )
    print(f"   Meshcat: {meshcat.web_url()}\n")
    
    # Initialize and take one tiny step to propagate initial state through diagram
    simulator.Initialize()
    simulator.AdvanceTo(1e-6)  # Tiny step to ensure state propagation
    
    visualizer.StartRecording()
    simulator.AdvanceTo(duration)
    visualizer.StopRecording()
    visualizer.PublishRecording()
    
    print(colored(f"\n✓ Simulation complete!", "green"))
    print(f"   Animation: {meshcat.web_url()}")
    
    # Extract logged data
    traj_log = traj_logger.FindLog(context)
    state_log = state_logger.FindLog(context)
    torque_log = torque_logger.FindLog(context)
    
    t = traj_log.sample_times()
    x_d, y_d = traj_log.data()[0, :], traj_log.data()[1, :]
    x_dot_d, y_dot_d = traj_log.data()[2, :], traj_log.data()[3, :]
    
    # CRITICAL: Drake's state output is ordered by GetJointIndices: [q2, q1, q2_dot, q1_dot]
    # NOT the intuitive [q1, q2, q1_dot, q2_dot]
    q2, q1 = state_log.data()[0, :], state_log.data()[1, :]  # Swapped!
    q2_dot, q1_dot = state_log.data()[2, :], state_log.data()[3, :]  # Swapped!
    
    # Torques also follow Drake's joint ordering
    tau2, tau1 = torque_log.data()[0, :], torque_log.data()[1, :]  # Swapped!
    
    # Compute actual EE position
    x_actual = np.zeros_like(t)
    y_actual = np.zeros_like(t)
    
    # DEBUG: Check joint ordering
    manip_joint_indices = list(plant.GetJointIndices(manipulator.model_instance))
    print(colored("\n🔍 Debugging joint ordering:", "yellow"))
    print(f"  Manipulator model instance joints:")
    for j, joint_index in enumerate(manip_joint_indices):
        joint = plant.get_joint(joint_index)
        print(f"    [{j}] {joint.name()} (num_vel={joint.num_velocities()})")
    
    # FIX: Joint ordering is [link2_link1, link1_base], so we need [q2, q1]
    for i in range(len(t)):
        plant_context = plant.CreateDefaultContext()
        for j, joint_index in enumerate(manip_joint_indices):
            joint = plant.get_joint(joint_index)
            if joint.num_velocities() > 0 and j < 2:
                # Now q1 and q2 are correctly swapped above, so use them directly
                joint.set_angle(plant_context, [q2[i], q1[i]][j])
        
        ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
        ee_pos = plant.CalcPointsPositions(
            plant_context, ee_frame, 
            manipulator.EE_OFFSET.reshape(3, 1), 
            plant.world_frame()
        ).flatten()
        x_actual[i], y_actual[i] = ee_pos[0], ee_pos[1]
    
    # Compute desired joint angles via IK from desired EE trajectory
    print(colored("\n📐 Computing desired joint angles from desired EE trajectory via IK...", "yellow"))
    q1_desired = np.zeros_like(t)
    q2_desired = np.zeros_like(t)
    
    # Subsample for efficiency (IK is slow)
    subsample_indices = np.arange(0, len(t), max(1, len(t) // 100))  # ~100 points
    q_seed_for_desired = q_init.copy()
    
    for idx in subsample_indices:
        target_xy = np.array([x_d[idx], y_d[idx]])
        q_sol, ik_success = solve_initial_pose_via_ik(
            plant, manipulator, target_xy, q_seed_for_desired, pos_tol=0.01
        )
        q1_desired[idx] = q_sol[0]
        q2_desired[idx] = q_sol[1]
        q_seed_for_desired = q_sol  # Warm start for next
    
    # Interpolate for all time points
    interp_q1 = interp1d(subsample_indices, q1_desired[subsample_indices], 
                         kind='cubic', fill_value='extrapolate')
    interp_q2 = interp1d(subsample_indices, q2_desired[subsample_indices], 
                         kind='cubic', fill_value='extrapolate')
    all_indices = np.arange(len(t))
    q1_desired = interp_q1(all_indices)
    q2_desired = interp_q2(all_indices)
    
    print(colored(f"✓ Computed desired joint angles for {len(t)} timesteps " 
                  f"(via IK at {len(subsample_indices)} points + interpolation)", "green"))
    
    # Compute IK joint angles from ACTUAL EE trajectory
    print(colored("\n📐 Computing IK joint angles from actual EE trajectory...", "yellow"))
    q1_actual_ik = np.zeros_like(t)
    q2_actual_ik = np.zeros_like(t)
    
    q_seed_for_actual = q_init.copy()
    
    # DEBUG: Check first point
    print(colored(f"\n  🔍 IK at t=0 from actual EE ({x_actual[0]:.3f}, {y_actual[0]:.3f}):", "yellow"))
    
    for idx in subsample_indices:
        target_xy = np.array([x_actual[idx], y_actual[idx]])
        q_sol, ik_success = solve_initial_pose_via_ik(
            plant, manipulator, target_xy, q_seed_for_actual, pos_tol=0.01
        )
        q1_actual_ik[idx] = q_sol[0]
        q2_actual_ik[idx] = q_sol[1]
        q_seed_for_actual = q_sol  # Warm start for next
        
        # DEBUG: Check first point
        if idx == 0:
            print(f"    IK solution: q1 = {np.rad2deg(q_sol[0]):.2f}°, q2 = {np.rad2deg(q_sol[1]):.2f}°")
            print(f"    Actual logged: q1 = {np.rad2deg(q1[0]):.2f}°, q2 = {np.rad2deg(q2[0]):.2f}°")
            print(f"    Difference: Δq1 = {np.rad2deg(q_sol[0] - q1[0]):.2f}°, Δq2 = {np.rad2deg(q_sol[1] - q2[0]):.2f}°")
            
            # Verify FK of both solutions
            plant_check = plant.CreateDefaultContext()
            manip_joint_indices = list(plant.GetJointIndices(manipulator.model_instance))
            
            # FK from IK solution
            for j, joint_index in enumerate(manip_joint_indices):
                joint = plant.get_joint(joint_index)
                if joint.num_velocities() > 0 and j < 2:
                    joint.set_angle(plant_check, q_sol[j])
            ee_frame = plant.GetFrameByName("link2", manipulator.model_instance)
            ee_pos_ik = plant.CalcPointsPositions(
                plant_check, ee_frame, manipulator.EE_OFFSET.reshape(3, 1), plant.world_frame()
            ).flatten()
            print(f"    FK(IK solution) → ({ee_pos_ik[0]:.3f}, {ee_pos_ik[1]:.3f})")
            
            # FK from logged angles
            for j, joint_index in enumerate(manip_joint_indices):
                joint = plant.get_joint(joint_index)
                if joint.num_velocities() > 0 and j < 2:
                    joint.set_angle(plant_check, [q1[0], q2[0]][j])
            ee_pos_logged = plant.CalcPointsPositions(
                plant_check, ee_frame, manipulator.EE_OFFSET.reshape(3, 1), plant.world_frame()
            ).flatten()
            print(f"    FK(logged q) → ({ee_pos_logged[0]:.3f}, {ee_pos_logged[1]:.3f})")
    
    # Interpolate for all time points
    interp_q1_act = interp1d(subsample_indices, q1_actual_ik[subsample_indices], 
                             kind='cubic', fill_value='extrapolate')
    interp_q2_act = interp1d(subsample_indices, q2_actual_ik[subsample_indices], 
                             kind='cubic', fill_value='extrapolate')
    q1_actual_ik = interp_q1_act(all_indices)
    q2_actual_ik = interp_q2_act(all_indices)
    
    print(colored(f"✓ Computed IK joint angles from actual EE for {len(t)} timesteps " 
                  f"(via IK at {len(subsample_indices)} points + interpolation)", "green"))
    
    # Compute errors
    e_x = (x_d - x_actual) * 1000  # mm
    e_y = (y_d - y_actual) * 1000  # mm
    e_total = np.sqrt(e_x**2 + e_y**2)
    
    # Statistics
    print(colored("\n📊 Tracking Performance:", "cyan"))
    print(f"   Mean X error:     {np.mean(np.abs(e_x)):6.1f} mm")
    print(f"   Mean Y error:     {np.mean(np.abs(e_y)):6.1f} mm")
    print(f"   Mean total error: {np.mean(e_total):6.1f} mm")
    print(f"   Max total error:  {np.max(e_total):6.1f} mm")
    print(f"   RMS total error:  {np.sqrt(np.mean(e_total**2)):6.1f} mm")
    
    # Plot results
    print(colored("\n📈 Generating plots...", "yellow"))
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x_d, 'b--', label='X desired', linewidth=2)
    ax1.plot(t, x_actual, 'b-', label='X actual', alpha=0.7)
    ax1.set_ylabel('X Position [m]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('X Trajectory')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, y_d, 'r--', label='Y desired', linewidth=2)
    ax2.plot(t, y_actual, 'r-', label='Y actual', alpha=0.7)
    ax2.set_ylabel('Y Position [m]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Y Trajectory')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x_d, y_d, 'k--', label='Desired', linewidth=2)
    ax3.plot(x_actual, y_actual, 'g-', label='Actual', alpha=0.7)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('EE Path (X-Y)')
    ax3.axis('equal')
    
    # Row 2: Errors and Velocities
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, e_x, 'b-', label='X error')
    ax4.plot(t, e_y, 'r-', label='Y error')
    ax4.plot(t, e_total, 'k-', label='Total error', linewidth=2)
    ax4.set_ylabel('Error [mm]')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Tracking Errors')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, x_dot_d, 'b--', label='ẋ desired', linewidth=2)
    ax5.plot(t, y_dot_d, 'r--', label='ẏ desired', linewidth=2)
    ax5.set_ylabel('Velocity [m/s]')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Desired EE Velocities')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, np.rad2deg(q1_desired), 'b--', label='q1 desired (IK from desired EE)', linewidth=2, alpha=0.8)
    ax6.plot(t, np.rad2deg(q2_desired), 'r--', label='q2 desired (IK from desired EE)', linewidth=2, alpha=0.8)
    ax6.plot(t, np.rad2deg(q1), 'b-', label='q1 actual (from plant)', linewidth=2.5, alpha=0.7)
    ax6.plot(t, np.rad2deg(q2), 'r-', label='q2 actual (from plant)', linewidth=2.5, alpha=0.7)
    ax6.plot(t, np.rad2deg(q1_actual_ik), 'b:', label='q1 IK from actual EE', linewidth=2, alpha=0.6)
    ax6.plot(t, np.rad2deg(q2_actual_ik), 'r:', label='q2 IK from actual EE', linewidth=2, alpha=0.6)
    ax6.set_ylabel('Angle [deg]')
    ax6.legend(fontsize=7, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Joint Angles: Desired IK vs Actual vs Actual-EE IK')
    
    # Row 3: Joint velocities and torques
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, np.rad2deg(q1_dot), 'b-', label='q̇1')
    ax7.plot(t, np.rad2deg(q2_dot), 'r-', label='q̇2')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Velocity [deg/s]')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Joint Velocities')
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, tau1, 'b-', label='τ1')
    ax8.plot(t, tau2, 'r-', label='τ2')
    ax8.axhline(100, color='k', linestyle='--', alpha=0.3, label='Limits')
    ax8.axhline(-100, color='k', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Torque [Nm]')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Joint Torques')
    
    ax9 = fig.add_subplot(gs[2, 2])
    # Show trajectory phases
    ax9.axvspan(0, 3, alpha=0.2, color='red', label='Y motion (0-3s)')
    ax9.axvspan(0, 4, alpha=0.2, color='blue', label='X motion (0-4s)')
    ax9.axvspan(4, duration, alpha=0.2, color='gray', label='Hold phase')
    ax9.plot(t, e_total, 'k-', linewidth=2)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Total Error [mm]')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_title('Error vs Time (Phases)')
    
    plt.suptitle('Manipulator EE Trajectory Tracking - Computed Torque Control', 
                 fontsize=14, fontweight='bold')
    
    # Save plot
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / "manipulator_ee_trajectory_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved to {plot_path}", "green"))
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Manipulator EE trajectory tracking test with computed torque control'
    )
    parser.add_argument(
        '--traj-type',
        type=str,
        choices=['generated', 'csv'],
        default='generated',
        help='Trajectory type: generated (prescribed) or csv (from file)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/cart_position_velocity.csv',
        help='Path to CSV file (used when --traj-type=csv)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='Simulation duration [s]'
    )
    
    args = parser.parse_args()
    
    run_test(
        duration=args.duration,
        traj_type=args.traj_type,
        csv_path=args.csv_path if args.traj_type == 'csv' else None
    )
