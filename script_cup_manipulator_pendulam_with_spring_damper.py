#!/usr/bin/env python3
"""
2D Cart-Pendulum with Muscle Dynamics & Optimal Feedback Control
Extended from 1D CartPendulum architecture

This script extends the CartPendulum class from script_cart_pendulum_muscle_dynamics_ofc.py
to 2D motion (x, y) instead of just 1D (x).

STATE VECTOR (14D):
==================
1. x       - Cart X position [m]
2. y       - Cart Y position [m]
3. α       - Pendulum pitch angle [rad]
4. β       - Pendulum roll angle [rad]
5. ẋ       - Cart X velocity [m/s]
6. ẏ       - Cart Y velocity [m/s]
7. α̇       - Pendulum pitch velocity [rad/s]
8. β̇       - Pendulum roll velocity [rad/s]
9. F_x     - Muscle force state X [N]
10. F_y    - Muscle force state Y [N]
11. x_ref  - ZFT reference X position [m]
12. y_ref  - ZFT reference Y position [m]
13. ẋ_ref  - ZFT reference X velocity [m/s]
14. ẏ_ref  - ZFT reference Y velocity [m/s]

CONTROL: u = [u_x, u_y] (2D neural command)
"""

import numpy as np
import argparse
import sys
import os
import time

# Set interactive backend BEFORE importing pyplot.
# On macOS the default can fall back to the non-interactive 'Agg' backend
# (which renders to memory only), causing plt.show() to be a no-op.
# 'MacOSX' is the native macOS backend; 'TkAgg' is the cross-platform fallback.
import matplotlib
try:
    matplotlib.use('MacOSX')
except Exception:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass   # leave whatever default was set; savefig still works

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from termcolor import colored
from scipy.linalg import solve_discrete_are
from typing import Literal

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    LogVectorOutput,
    LeafSystem,
    BasicVector,
    MeshcatVisualizer,
    StartMeshcat,
    Multiplexer,
    Demultiplexer,
    Saturation,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    RigidTransform,
    RevoluteJoint,
    PrismaticJoint,
    Sphere,
    Cylinder,
    Parser,
    ZeroOrderHold,
    JacobianWrtVariable,
    InverseKinematics,
    Solve,
    ConstantVectorSource,
    TrajectorySource,
    PiecewisePolynomial,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint, PrismaticJoint, RevoluteSpring
from pydrake.math import RigidTransform, RollPitchYaw

from pydrake.multibody.tree import MultibodyForces

# Import from existing script
import sys
sys.path.append(str(Path(__file__).parent))
from configs.robot.robot_types import (
    create_cart_pendulum_config, 
    create_cup_manipulator_config,
    ManipulatorConfig
)


from scipy.linalg import solve_discrete_are

from utils.utils import (
    build_linearized_system_2d,
    build_linearized_for_complete_system_2d,
    check_trajectory_feasibility,
    test_and_visualize_ik_feasibility,
)

# ============================================================================
# VISUALIZATION (moved to viz.py)
# ============================================================================
from utils.viz import (
    visualize_plant_meshcat,
    add_frames_to_meshcat,
    plot_frames_top_view,
    plot_lqr_manip_ee_traj_track_results,
    set_meshcat_camera_view,
)

from script_cup_manipulator_pendulam_lqr_min_effort_2d import (
    SimulationConfig,
    MeshcatFrameUpdater,
    ManipulatorIKDesiredAngles,
    SystemBuilder,
    ControlSystemBuilder,
    LQRWithOFCOnlyCartPendulumBuilder,
    Simulation,
)

from robots.cup_manipulator import RobotBase, CupManipulator, CartPendulum2DExtended
from robots.cup_manipulator_tendon import CupManipulatorTendon, CupManipulatorIKSystem, ComputedTorqueController, create_cable_manipulator_config

# ── Cable routing ─────────────────────────────────────────────────────────────
from utils.viz_cables import (
    draw_cables,
    print_cable_routing_points,
    visualize_cable_routing_top_view,
    _Xw,
)

# Config classes from dedicated modules
from configs.robot.robot_configs import (
    CartPendulumPhysicsConfig,
    EndEffectorKinematics2DConfig,
    create_physics_config,
)
from configs.controller.controller_configs import (
    MuscleDynamicsConfig,
    ImpedanceForceConfig,
    ZFTReferenceMassConfig,
    FiniteHorizonLQRConfig,
    ZFTJointReferenceIKConfig,
    create_muscle_config,
    create_impedance_config,
    create_zft_config,
    create_lqr_config,
)

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

# ---------------------------------------------------------------------------
# Argument parser — grouped by which mode(s) each flag applies to.
#
# Dependency map:
#   --mode                   → required by all
#   --duration               → ik-diagram (lap duration)
#   Meshcat camera group     → scene-viz-q | ee-trajectory | ik-diagram
#   Trajectory group         → ee-trajectory | ik-diagram
#     --traj-shape rect      →   also needs --traj-x-range / --traj-y-range
#     --traj-shape circle    →   also needs --traj-cx, --traj-cy, --traj-radius
#     --traj-shape line      →   also needs --traj-cx, --traj-cy, --traj-radius
#   IK solver group          → ee-trajectory only
#     --ik-damping           →   only meaningful when --ik-method velocity|hybrid
#
# NOTE: argparse argument groups are cosmetic (they appear as labelled sections
# in --help) but do NOT enforce runtime exclusivity.  Cross-argument validation
# is done in _validate_args() below immediately after parse_known_args().
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Cup-manipulator tendon simulation (Drake / Meshcat)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# ── 1. Core ─────────────────────────────────────────────────────────────────
_core = parser.add_argument_group('core  [all modes]')
_core.add_argument(
    '--mode', type=str,
    choices=['scene-viz', 'scene-viz-q', 'ee-trajectory', 'pd-control',
             'simulate-cup-manipulator-with-spring-damper', 'computed-torque'],
    default='computed-torque',
    help='Simulation mode  (default: computed-torque)',
)
_core.add_argument(
    '--duration', type=float, default=10.0,
    help='Lap duration [s] — one full trajectory cycle  (default: 10.0)',
)

# ── Robot mount tilt ─────────────────────────────────────────────────────────
_mount = parser.add_argument_group('robot mount  [all simulation modes]')
_mount.add_argument(
    '--tilt-roll', type=float, default=0.0,
    help='Base roll tilt around X-axis [deg]  (default: 0 → horizontal SCARA)',
)
_mount.add_argument(
    '--tilt-pitch', type=float, default=0.0,
    help='Base pitch tilt around Y-axis [deg]  (default: 0 → horizontal SCARA)',
)

#── Joint damping and stiffness ─────────────────────────────────────────────────────────
_mount.add_argument(
    '--joint-damping', type=float, nargs=2, default=[0.05, 0.05],
    metavar=('D1', 'D2'),
    help='Viscous joint damping [Nm·s/rad] for [link1_base, link2_link1]  (default: 0 0)',
)
_mount.add_argument(
    '--joint-stiffness', type=float, nargs=2, default=[0.5, 0.5],
    metavar=('K1', 'K2'),
    help='Passive joint spring stiffness [Nm/rad] for [link1_base, link2_link1]  (default: 0 0)',
)

# ── 2. Meshcat camera ────────────────────────────────────────────────────────
_cam = parser.add_argument_group('Meshcat camera  [scene-viz-q | ee-trajectory | ik-diagram]')
_cam.add_argument('--meshcat-azimuth',   type=float, default=0.0,
                  help='Camera azimuth [deg]  0=+X  90=+Y  (default: 0)')
_cam.add_argument('--meshcat-elevation', type=float, default=75.0,
                  help='Camera elevation [deg]  90=top  0=side  (default: 75)')
_cam.add_argument('--meshcat-distance',  type=float, default=1.3,
                  help='Camera distance from target [m]  (default: 1.3)')

# ── 4. IK solver ─────────────────────────────────────────────────────────────
_ik = parser.add_argument_group('IK solver  [ee-trajectory only]')
_ik.add_argument(
    '--ik-method', type=str, default='hybrid',
    choices=['analytical', 'velocity', 'hybrid'],
    help='analytical=exact 2R  velocity=Jacobian Δq  hybrid=actuation-space  (default: hybrid)',
)
_ik.add_argument(
    '--ik-damping', type=float, default=1e-4,
    help='Damping λ for damped pseudo-inverse  [velocity|hybrid only]  (default: 1e-4)',
)

# ── 3. Trajectory ────────────────────────────────────────────────────────────
_traj = parser.add_argument_group(
    'trajectory  [ee-trajectory | ik-diagram]\n'
    '  --traj-shape rect   → also set --traj-x-range / --traj-y-range\n'
    '  --traj-shape circle → also set --traj-cx --traj-cy --traj-radius\n'
    '  --traj-shape line   → also set --traj-cx --traj-cy --traj-radius'
)
_traj.add_argument(
    '--traj-shape', type=str, default='rect',
    choices=['circle', 'line', 'rect'],
    help='Trajectory shape  (default: rect)',
)
_traj.add_argument(
    '--traj-n', type=int, default=60,
    help='Number of waypoints along the trajectory  (default: 60)',
)
# rect shape
_traj.add_argument(
    '--traj-x-range', type=float, nargs=2, default=[0.49, 0.51],
    metavar=('X_MIN', 'X_MAX'),
    help='X extents for rect [m]  (default: 0.49 0.51)',
)
_traj.add_argument(
    '--traj-y-range', type=float, nargs=2, default=[-0.08, 0.08],
    metavar=('Y_MIN', 'Y_MAX'),
    help='Y extents for rect [m]  (default: -0.08 0.08)',
)
# rect velocity profile
_traj.add_argument('--move-duration',     type=float, default=3.0,
                   help='Time [s] to move smoothly from q=0 to the first waypoint before '
                        'trajectory tracking begins. Set to 0 to skip.  (default: 3.0)')

# ── Home position ─────────────────────────────────────────────────────────────
# Exactly one of --home-ee or --home-joints may be given.  When neither is
# provided the home is auto-resolved to the IK of the first trajectory waypoint
# (best default — avoids the singular q=0 fully-extended pose).
_home = parser.add_argument_group(
    'home position  [ik-diagram | computed-torque]\n'
    '  Override the robot\'s starting pose before the move-to-start preamble.\n'
    '  Use --home-ee (Cartesian) OR --home-joints (joint-space); not both.\n'
    '  Default (neither flag): auto-resolved from the first trajectory waypoint.'
)
_home.add_argument(
    '--home-ee', type=float, nargs=2, default=[0.40, 0.00],
    metavar=('X_M', 'Y_M'),
    help='Home end-effector XY position [m].  IK is solved to reach this point.  '
         '(default: 0.50 0.00 — on the X axis at mid-reach)',
)
_home.add_argument(
    '--home-joints', type=float, nargs=2, default=None,
    metavar=('Q1_DEG', 'Q2_DEG'),
    help='Home joint angles [degrees] for [link1_base, link2_link1].  '
         'Overrides --home-ee when provided.  Example: --home-joints 10 -20',
)

_traj.add_argument('--traj-v-max',        type=float, default=0.9,
                   help='Rect: peak EE speed on straight sections [m/s]  (default: 0.9)')
_traj.add_argument('--traj-v-corner',     type=float, default=0.05,
                   help='Rect: slow EE speed at corners [m/s]  (default: 0.05)')
_traj.add_argument('--traj-corner-blend', type=float, default=0.35,
                   help='Rect: blend-zone width as fraction of shorter side  (default: 0.35)')
# circle / line shape
_traj.add_argument('--traj-cx',     type=float, default=0.51,
                   help='Circle/line centre X [m]  (default: 0.51)')
_traj.add_argument('--traj-cy',     type=float, default=0.10,
                   help='Circle/line centre Y [m]  (default: 0.10)')
_traj.add_argument('--traj-radius', type=float, default=0.02,
                   help='Circle radius or half-line length [m]  (default: 0.02)')

# ── 5. Computed-torque gains ─────────────────────────────────────────────────
_ct = parser.add_argument_group('computed-torque  [computed-torque mode only]')
_ct.add_argument(
    '--ct-kp', type=float, default=10000.0,
    help='Computed-torque position gain Kp [1/s²]  → ωn = sqrt(Kp) ≈ 20 rad/s  (default: 10000)',
)
_ct.add_argument(
    '--ct-kd', type=float, default=400.0,
    help='Computed-torque velocity gain Kd [1/s]   → ζ = Kd/(2√Kp)  (default: 40)',
)
_ct.add_argument(
    '--ct-tau-max', type=float, default=10.0,
    help='Computed-torque torque saturation [Nm]  (default: 10.0)',
)

# ── Unused / reserved ────────────────────────────────────────────────────────
# The arguments below existed in earlier versions but are not currently wired up.
# Kept as comments so we don't lose the intent; uncomment if a mode needs them.
#
# parser.add_argument('--target-x',      type=float, default=0.45,
#                     help='[UNUSED] Fixed EE target X — replaced by --traj-*')
# parser.add_argument('--target-y',      type=float, default=0.10,
#                     help='[UNUSED] Fixed EE target Y — replaced by --traj-*')
# parser.add_argument('--cart-x-init',   type=float, default=2.0,
#                     help='[UNUSED] Initial cart X — not wired in current modes')
# parser.add_argument('--cart-y-init',   type=float, default=0.0,
#                     help='[UNUSED] Initial cart Y — not wired in current modes')
# parser.add_argument('--horizon',       type=float, default=10.0,
#                     help='[UNUSED] LQR horizon — belongs to lqr_min_effort script')
# parser.add_argument('--speed-scale',   type=float, default=0.54,
#                     help='[UNUSED] Trajectory speed scaling — not wired in current modes')


def _validate_args(a: argparse.Namespace) -> None:
    """Enforce cross-argument dependencies after parsing."""
    traj_modes = {'ee-trajectory', 'pd-control', 'computed-torque'}
    if a.mode in traj_modes:
        if a.traj_shape in ('circle', 'line') and a.traj_radius <= 0:
            parser.error(f'--traj-radius must be > 0 for --traj-shape {a.traj_shape}')
        if a.traj_shape == 'rect':
            if a.traj_x_range[0] >= a.traj_x_range[1]:
                parser.error('--traj-x-range: X_MIN must be < X_MAX')
            if a.traj_y_range[0] >= a.traj_y_range[1]:
                parser.error('--traj-y-range: Y_MIN must be < Y_MAX')
    if a.mode == 'ee-trajectory' and a.ik_method != 'analytical' and a.ik_damping <= 0:
        parser.error('--ik-damping must be > 0 when --ik-method is velocity or hybrid')


# Parse and save our args FIRST, then validate cross-argument dependencies.
_parsed_args, _ = parser.parse_known_args()
_validate_args(_parsed_args)

# Temporarily clear sys.argv to prevent CupManipulator module from parsing our args
# (it has its own argparse that would fail on our mode choices)
import sys
_saved_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Keep only script name

# Now import CupManipulator safely
# from script_cup_manipulator_controller_ofc import CupManipulator

# Restore sys.argv and our parsed args
sys.argv = _saved_argv
args = _parsed_args

# Global configurations
PHYSICS_CONFIG = create_physics_config()


# Configuration for the cable (tendon) manipulator — used in scene-viz mode
CABLE_MANIPULATOR_CONFIG = create_cable_manipulator_config(
    urdf_path="model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
    joint_angles={
        'link1_base':  np.deg2rad(10.0),   # q1
        'link2_link1': np.deg2rad(-10.0),   # q2
    },
    damping=tuple(args.joint_damping),
    stiffness=tuple(args.joint_stiffness),
    tilt_roll_deg=args.tilt_roll,
    tilt_pitch_deg=args.tilt_pitch,
)
# SIMULATION_CONFIG = SimulationConfig.from_args(args, PHYSICS_CONFIG, MUSCLE_CONFIG, IMPEDANCE_CONFIG, ZFT_CONFIG, None)


# ============================================================================
# BASE SIMULATION CLASS
# ============================================================================

class Simulation(ABC):
    """Abstract base for all Drake simulation modes in this project.

    Builder pattern — call steps in order:
      build_plant()        → create DiagramBuilder, MultibodyPlant, SceneGraph,
                             load URDF, call plant.Finalize()
      build_trajectory()   → (optional) build reference trajectory from CLI args;
                             no-op by default, override in trajectory-following modes
      build_controller()   → create and wire controller LeafSystem(s)
      connect_and_build()  → add loggers, call builder.Build(), create Simulator
      initialize()         → patch zero-mass bodies, set initial state,
                             call simulator.Initialize()
      run(**kw)            → simulation loop (blocking until Ctrl-C or end time)
      plot(**kw)           → generate matplotlib figures from logged data

    Shared helpers (call from subclass methods as needed):
      _add_scene_graph_connections()  — plant ↔ scene_graph geometry ports
      _add_meshcat_visualizer()       — no-op when meshcat is None
      _patch_zero_mass_bodies()       — inject nominal inertia into bodies whose
                                        Onshape material was never set (mass = 0);
                                        prevents SAP solver NaN on first step

    Subclasses: IKPDSimulation, ComputedTorqueSimulation, [future modes here]

    DELETED (was here before): the old cart-pendulum-specific Simulation class
    that used SimulationConfig / SystemBuilder / ControlSystemBuilder.  That
    class was only used in lqr_min_effort_2d.py (imported at runtime); it was
    dead code in this file.
    """

    _DT = 0.002  # default plant discrete time step [s]

    # Nominal inertia injected into zero-mass bodies so SAP solver doesn't NaN.
    # (All URDF parts have no material assigned in Onshape → default_mass = 0.)
    _M_PATCH = SpatialInertia(
        mass=0.3,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
    )

    def __init__(self, manip_config, meshcat=None):
        self.manip_config = manip_config
        self.meshcat      = meshcat
        # Set by build_plant()
        self.builder     = None
        self.plant       = None
        self.scene_graph = None
        self.manipulator = None
        # Set by connect_and_build()
        self.diagram     = None
        self.simulator   = None
        self.loggers     = {}   # populated by subclass connect_and_build()

    # ── Shared plant helpers ──────────────────────────────────────────────────

    def _add_scene_graph_connections(self):
        """Wire plant → scene_graph geometry ports (call from build_plant)."""
        self.builder.Connect(
            self.plant.get_geometry_pose_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id()),
        )
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port(),
        )

    def _add_meshcat_visualizer(self):
        """Add MeshcatVisualizer to builder (no-op when meshcat is None)."""
        if self.meshcat is not None:
            MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, self.meshcat)

    def _patch_zero_mass_bodies(self):
        """Inject nominal inertia into zero-mass bodies to prevent SAP NaN.

        Must be called from initialize() *after* self.simulator is created.
        Requires self.plant and self.manipulator to be set by build_plant().
        """
        context   = self.simulator.get_mutable_context()
        plant_ctx = self.plant.GetMyMutableContextFromRoot(context)
        patched   = []
        for idx in self.plant.GetBodyIndices(self.manipulator.model_instance):
            body = self.plant.get_body(idx)
            if body.default_mass() < 1e-6:
                body.SetSpatialInertiaInBodyFrame(plant_ctx, self._M_PATCH)
                patched.append(body.name())
        if patched:
            print(colored(f"  ⚠ Patched zero-mass bodies (0.3 kg nominal): {patched}", "yellow"))

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_plant(self):
        """Create DiagramBuilder, MultibodyPlant, SceneGraph; load URDF; Finalize."""

    @abstractmethod
    def build_controller(self):
        """Create and add controller LeafSystem(s) to builder."""

    @abstractmethod
    def connect_and_build(self):
        """Wire all connections, add loggers, call builder.Build(), create Simulator."""

    @abstractmethod
    def initialize(self):
        """Set initial state, patch zero-mass bodies, call simulator.Initialize()."""

    @abstractmethod
    def run(self, **kwargs):
        """Run simulation (blocking until complete or Ctrl-C)."""

    @abstractmethod
    def plot(self, **kwargs):
        """Generate matplotlib figures from logged data."""

    # ── Optional override ─────────────────────────────────────────────────────

    def build_trajectory(self, args):
        """Build reference trajectory from CLI args.

        Override in trajectory-following modes (e.g. IKPDSimulation).
        Default implementation is a no-op for modes that have no reference path.
        """
        pass

# ============================================================================
# IK-PD SIMULATION CLASS
# ============================================================================

class IKPDSimulation(Simulation):
    """
    Closed-loop Drake diagram simulation for the cable (tendon) manipulator.

    Diagram topology
    ─────────────────
      LoopingTrajectorySource ──► CupManipulatorIKSystem ──► MultibodyPlant
                                          ▲ plant_state ────────────┘

    Usage (Builder pattern — call methods in order)
    ────────────────────────────────────────────────
      sim = IKPDSimulation(CABLE_MANIPULATOR_CONFIG, meshcat)
      sim.build_plant()           # cable manipulator plant only (no cart-pendulum)
      sim.build_trajectory(args)  # LoopingTrajectorySource from CLI args
      sim.build_controller()      # CupManipulatorIKSystem with PD + cable physics
      sim.connect_and_build()     # wire connections, add loggers, build Diagram
      sim.initialize()            # patch zero-mass bodies, set home config
      sim.run()                   # while-loop until Ctrl+C
      sim.plot(args.traj_shape)   # 4-panel time series + EE XY trajectory
    """

    _URDF_PATH = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
    # _DT and _M_PATCH inherited from Simulation base class

    # PD gains (see comments in ik-diagram block for bandwidth rationale)
    KP1      = 80.0    # Joint 1 stiffness  [Nm/rad]    → ωn ≈ 34 rad/s, overdamped
    KD1      = 16.0    # Joint 1 damping    [Nm·s/rad]
    KP_CABLE = 260.0   # Joint 2 cable spring  [N/m]    → ωn ≈ 15 rad/s, critically damped
    KD_CABLE = 35.0    # Joint 2 cable damping [N·s/m]
    TAU_MAX  = 10.0    # Saturation [Nm] — matches URDF <limit effort="10">

    def __init__(self, manip_config, meshcat=None):
        super().__init__(manip_config, meshcat)

        # Set by build_trajectory()
        self.traj_ref     = None
        self.lap_duration = None
        self.ee_x_tgt     = None
        self.ee_y_tgt     = None

        # Set by build_move_to_start()
        self.move_duration = 0.0
        self.move_traj     = None    # PiecewisePolynomial position preamble
        self.move_traj_vel = None    # derivative 1 — velocity
        self.move_traj_acc = None    # derivative 2 — acceleration

        # Set by build_controller()
        self.ik_system = None

        # Set by connect_and_build()
        self.ee_ref  = None
        # self.loggers: keys 'state', 'q_des', 'cable', 'ref'  (dict in Simulation base)

    # ── 1. Plant ──────────────────────────────────────────────────────────────
    def build_plant(self):
        """Build MultibodyPlant with the cable manipulator only (no cart-pendulum)."""
        self.builder     = DiagramBuilder()
        self.plant       = MultibodyPlant(time_step=self._DT)
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)

        self.manipulator = CupManipulatorTendon(self.manip_config, enable_visualization=True)
        parser_urdf = Parser(self.plant)
        self.manipulator.load_urdf_to_plant(self.plant, parser_urdf)
        _orientation = np.deg2rad([self.manip_config.tilt_roll_deg, self.manip_config.tilt_pitch_deg, 0.0])
        self.manipulator.weld_base_to_world(self.plant, position=np.zeros(3), orientation=_orientation)
        self.manipulator.add_joint_actuators(self.plant)
        self.manipulator.set_joint_properties(self.plant)   # applies --joint-damping
        # Passive joint springs — stiffness read from ManipulatorConfig.joint_configs
        for jt_name in [self.manipulator.JT1_NAME, self.manipulator.JT2_NAME]:
            joint_cfg = self.manip_config.joint_configs.get(jt_name)
            K = joint_cfg.stiffness if joint_cfg is not None else 0.0
            if K > 0.0:
                jt = self.manipulator.get_joint_by_name(self.plant, jt_name)
                self.plant.AddForceElement(RevoluteSpring(jt, nominal_angle=0.0, stiffness=K))
        self.manipulator.add_end_effector_frame(self.plant)
        self.plant.Finalize()
        # Initialise cable rig geometry (needed for draw_cables during simulation)
        self.manipulator.init_cable_rig(self._URDF_PATH)
        self.rig = self.manipulator.rig
        self.builder.AddSystem(self.plant)

        self._add_scene_graph_connections()
        self._add_meshcat_visualizer()

    # ── 2. Trajectory ─────────────────────────────────────────────────────────
    def build_trajectory(self, args):
        """Build the looping EE reference trajectory from parsed CLI args."""
        L1, L2 = self.manipulator.ik.get_link_lengths(self.plant)
        r_max  = L1 + L2
        r_min  = abs(L1 - L2)

        N  = args.traj_n
        cx, cy = args.traj_cx, args.traj_cy
        R  = args.traj_radius

        if args.traj_shape == 'circle':
            angles   = np.linspace(0, 2 * np.pi, N, endpoint=False)
            ee_x_tgt = cx + R * np.cos(angles)
            ee_y_tgt = cy + R * np.sin(angles)
        elif args.traj_shape == 'rect':
            x_min, x_max = args.traj_x_range
            y_min, y_max = args.traj_y_range
            W  = x_max - x_min
            H  = y_max - y_min
            P  = 2.0 * (W + H)                        # total perimeter

            # Arc-length positions of the 4 corners along the perimeter
            _cs = np.array([0.0, W, W + H, 2.0 * W + H])

            def _s_to_xy(s):
                """Arc-length s → (x, y) on rectangle perimeter."""
                s = s % P
                if   s <= W:            return x_min + s,            y_min
                elif s <= W + H:        return x_max,                 y_min + (s - W)
                elif s <= 2.0 * W + H:  return x_max - (s - W - H),  y_max
                else:                   return x_min,                 y_max - (s - 2.0 * W - H)

            def _dist_corner(s):
                """Minimum arc-length distance from s to the nearest corner."""
                s = s % P
                d = np.abs(s - _cs)
                return float(np.minimum(d, P - d).min())

            # Speed profile parameters
            _v_max    = args.traj_v_max                          # peak speed [m/s]
            _v_corner = args.traj_v_corner                       # corner speed [m/s]
            _d_blend  = args.traj_corner_blend * min(W, H)       # blend zone [m]

            def _speed(s):
                """Smoothstep: v_corner at corners, ramps to v_max on straights."""
                t = np.clip(_dist_corner(s) / _d_blend, 0.0, 1.0)
                return _v_corner + (_v_max - _v_corner) * t * t * (3.0 - 2.0 * t)

            # N+1 arc-length-uniform samples (index N == index 0, closure)
            _s_vals = np.linspace(0.0, P, N + 1, endpoint=True)
            _speeds  = np.array([_speed(s) for s in _s_vals])

            # Non-uniform timestamps: dt_i = ds / v_avg between samples i and i+1
            _ds = P / N
            _t_raw = np.zeros(N + 1)
            for _i in range(N):
                _t_raw[_i + 1] = _t_raw[_i] + _ds / (0.5 * (_speeds[_i] + _speeds[_i + 1]))
            _rect_t_wp_raw = _t_raw   # scaled to lap_duration after clamping

            # Waypoints (arc-length uniform along perimeter)
            _xy      = np.array([_s_to_xy(s) for s in _s_vals])
            ee_x_tgt = _xy[:N, 0]
            ee_y_tgt = _xy[:N, 1]

            print(colored(
                f"  ✓ Rect velocity profile: v_max={_v_max:.2f} m/s  "
                f"v_corner={_v_corner:.2f} m/s  "
                f"blend={args.traj_corner_blend*100:.0f}% of {min(W,H)*1e3:.1f} mm",
                "cyan",
            ))
        else:  # line
            ee_x_tgt = np.linspace(cx - R, cx + R, N)
            ee_y_tgt = np.full(N, cy)

        # Clamp all waypoints to the reachable workspace [r_min, r_max]
        _r = np.hypot(ee_x_tgt, ee_y_tgt)
        _too_far   = _r > r_max
        _too_close = _r < r_min + 0.01
        if _too_far.any():
            print(colored(f"  ⚠ {_too_far.sum()} waypoints outside max reach ({r_max*1e3:.0f} mm) — clamped.", "yellow"))
            ee_x_tgt[_too_far] = ee_x_tgt[_too_far] / _r[_too_far] * (r_max * 0.97)
            ee_y_tgt[_too_far] = ee_y_tgt[_too_far] / _r[_too_far] * (r_max * 0.97)
        if _too_close.any():
            _r_c = np.maximum(_r[_too_close], 1e-6)
            print(colored(f"  ⚠ {_too_close.sum()} waypoints inside min reach ({r_min*1e3:.0f} mm) — clamped.", "yellow"))
            ee_x_tgt[_too_close] = ee_x_tgt[_too_close] / _r_c * (r_min + 0.01)
            ee_y_tgt[_too_close] = ee_y_tgt[_too_close] / _r_c * (r_min + 0.01)

        self.ee_x_tgt     = ee_x_tgt
        self.ee_y_tgt     = ee_y_tgt
        self.lap_duration = args.duration

        # Build cubic spline: N+1 points (last wraps back to first waypoint).
        # CubicWithContinuousSecondDerivatives gives a C² trajectory through
        # each waypoint — non-zero velocity AND acceleration at every instant,
        # so the feedforward term q̈_ref in the CT law is meaningful.
        #
        # For rect: use non-uniform time stamps so the EE slows at corners and
        # speeds up on straights.  The absolute speed ratio v_max/v_corner sets
        # the time distribution; everything is then scaled to lap_duration.
        if args.traj_shape == 'rect':
            t_wp = _rect_t_wp_raw * (self.lap_duration / _rect_t_wp_raw[-1])
        else:
            t_wp = np.linspace(0.0, self.lap_duration, N + 1)
        wp   = np.column_stack([
            np.append(ee_x_tgt, ee_x_tgt[0]),
            np.append(ee_y_tgt, ee_y_tgt[0]),
        ]).T   # shape (2, N+1)
        self.traj_ref     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
        self.traj_vel_ref = self.traj_ref.derivative(1)   # EE velocity  [m/s]
        self.traj_acc_ref = self.traj_ref.derivative(2)   # EE acceleration [m/s²]

        print(colored(
            f"  ✓ EE trajectory: {args.traj_shape}  N={N} waypoints"
            f"  lap={self.lap_duration:.1f} s  workspace [{r_min*1e3:.0f}, {r_max*1e3:.0f}] mm",
            "green",
        ))

    # ── 2b. Move-to-start ────────────────────────────────────────────────────
    def build_move_to_start(self, move_duration: float):
        """Build a smooth approach trajectory from a pre-home pose to the first
        tracking waypoint.  Must be called *after* build_trajectory().

        Pre-home is computed by solving IK for the first waypoint and then
        applying a small joint-space offset (a few degrees) so the robot starts
        nearby but slightly off-trajectory — giving the preamble a meaningful
        motion while avoiding the singular q=(0,0) fully-extended pose.

        Sets self.move_duration, self.move_traj, self.move_traj_vel,
        self.move_traj_acc, and self._q_init (used by initialize()).
        If move_duration <= 0 simply disables the preamble.
        """
        self.move_duration = float(move_duration)

        # Always compute _q_init (IK of first waypoint) so initialize() can use it.
        L1, L2 = self.manipulator.ik.get_link_lengths(self.plant)
        p_end = np.array([self.ee_x_tgt[0], self.ee_y_tgt[0]])
        # Use a non-zero seed to stay away from the singular q=(0,0) branch
        _seed = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
        q_end, ok = self.manipulator.ik._solve_2r_core(L1, L2, p_end, q_seed=_seed)
        if not ok:
            q_end = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])  # safe fallback
            print(colored("  ⚠  IK for first waypoint failed in move-to-start — using fallback pose.", "yellow"))
        self._q_init = q_end   # robot will be placed here by initialize()

        if self.move_duration <= 0.0 or self.traj_ref is None:
            self.move_traj = self.move_traj_vel = self.move_traj_acc = None
            print(colored("  ℹ  Move-to-start disabled (--move-duration 0).", "white"))
            return

        # Pre-home: small q-space offset from q_end so the preamble has a
        # short meaningful motion (a few cm) without visiting y=0.
        # Offset: subtract ~5° from q1, add ~5° to q2 — stays in reachable workspace.
        q_pre = q_end + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
        tmp_ctx = self.plant.CreateDefaultContext()
        self.manipulator.set_positions_user_order(self.plant, tmp_ctx, q_pre)
        pre_pos = self.manipulator.get_end_effector_position(self.plant, tmp_ctx)
        p_start = np.array([pre_pos[0], pre_pos[1]])

        # First waypoint velocity for C¹ continuity at the hand-off
        v_end = self.traj_vel_ref.value(0.0).ravel()

        # Cubic Hermite: start at rest (v=0), arrive with traj initial velocity
        t_breaks    = np.array([0.0, self.move_duration])
        samples     = np.column_stack([p_start, p_end])       # (2, 2)
        samples_dot = np.column_stack([np.zeros(2), v_end])   # (2, 2)
        self.move_traj     = PiecewisePolynomial.CubicHermite(t_breaks, samples, samples_dot)
        self.move_traj_vel = self.move_traj.derivative(1)
        self.move_traj_acc = self.move_traj.derivative(2)

        print(colored(
            f"  ✓ Move-to-start: pre-home EE ({p_start[0]*1e3:.1f}, {p_start[1]*1e3:.1f}) mm"
            f" → first waypoint ({p_end[0]*1e3:.1f}, {p_end[1]*1e3:.1f}) mm"
            f"  in {move_duration:.1f} s"
            f"  [q_init=[{np.rad2deg(q_end[0]):.1f}°, {np.rad2deg(q_end[1]):.1f}°]]",
            "green",
        ))

    # ── 3. Controller ─────────────────────────────────────────────────────────
    def build_controller(self):
        """Create and add CupManipulatorIKSystem to the builder."""
        self.ik_system = self.builder.AddSystem(
            CupManipulatorIKSystem(
                self.plant, self.manipulator,
                Kp1=self.KP1, Kd1=self.KD1,
                Kp_cable=self.KP_CABLE, Kd_cable=self.KD_CABLE,
                tau_max=self.TAU_MAX,
            )
        )
        self.ik_system.set_name("IK_Controller")

    # ── 4. Connect + Build ────────────────────────────────────────────────────
    def connect_and_build(self):
        """Wire all connections, add loggers, and call builder.Build()."""

        # _PreambleSource: during the move-to-start phase (t < move_duration) it
        # outputs the smooth approach spline; afterwards it loops the main trajectory.
        # If move_traj is None (preamble disabled), loops main trajectory from t=0.
        class _PreambleSource(LeafSystem):
            def __init__(inner_self, move_traj, move_dur, main_traj, period):
                super().__init__()
                inner_self._move_traj = move_traj
                inner_self._move_dur  = float(move_dur)
                inner_self._main_traj = main_traj
                inner_self._period    = float(period)
                inner_self.DeclareVectorOutputPort("output", main_traj.rows(), inner_self._calc)

            def _calc(inner_self, context, output):
                t = context.get_time()
                if inner_self._move_traj is not None and t < inner_self._move_dur:
                    val = inner_self._move_traj.value(t).ravel()
                else:
                    t_wrap = max(0.0, t - inner_self._move_dur) % inner_self._period
                    val = inner_self._main_traj.value(t_wrap).ravel()
                output.SetFromVector(val)

        self.ee_ref = self.builder.AddSystem(
            _PreambleSource(self.move_traj, self.move_duration, self.traj_ref, self.lap_duration)
        )
        self.ee_ref.set_name("EE_Trajectory")

        # Signal connections
        self.builder.Connect(
            self.ee_ref.get_output_port(),
            self.ik_system.GetInputPort("desired_ee_pos"),
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.ik_system.GetInputPort("plant_state"),
        )
        self.builder.Connect(
            self.ik_system.GetOutputPort("actuation"),
            self.plant.get_actuation_input_port(),
        )

        # Loggers
        self.loggers['state']    = LogVectorOutput(self.plant.get_state_output_port(),                self.builder)
        self.loggers['q_des']    = LogVectorOutput(self.ik_system.GetOutputPort("joint_positions"),   self.builder)
        self.loggers['actuation']= LogVectorOutput(self.ik_system.GetOutputPort("actuation"),         self.builder)
        self.loggers['cable']    = LogVectorOutput(self.ik_system.GetOutputPort("cable_lengths"),     self.builder)
        self.loggers['tensions'] = LogVectorOutput(self.ik_system.GetOutputPort("cable_tensions"),    self.builder)
        self.loggers['ref']      = LogVectorOutput(self.ee_ref.get_output_port(),                     self.builder)

        self.diagram   = self.builder.Build()
        self.simulator = Simulator(self.diagram)

    # ── 5. Initialize ─────────────────────────────────────────────────────────
    def initialize(self, home_override=None):
        """Patch zero-mass bodies and set initial configuration.

        Priority for determining the starting joint angles q0:
          1. home_override dict from the caller:
               {'ee':     np.array([x, y])}       solve IK for this Cartesian target
               {'joints': np.array([q1, q2])}     use these joint angles directly [rad]
          2. self._q_init set by build_move_to_start() (IK of first waypoint)
          3. IK of first trajectory waypoint (if trajectory was built)
          4. Hard fallback: q=[5°, 15°]  (safe non-singular pose)
        """
        self._patch_zero_mass_bodies()

        plant_ctx = self.plant.GetMyMutableContextFromRoot(
            self.simulator.get_mutable_context()
        )

        q0 = None

        # ── Priority 1: explicit home override ────────────────────────────
        if home_override is not None:
            if 'joints' in home_override:
                q0 = np.asarray(home_override['joints'], dtype=float)
                print(colored(
                    f"  ✓ Home (--home-joints): "
                    f"q=[{np.rad2deg(q0[0]):.1f}°, {np.rad2deg(q0[1]):.1f}°]",
                    "cyan",
                ))
            elif 'ee' in home_override:
                L1, L2 = self.manipulator.ik.get_link_lengths(self.plant)
                p_home = np.asarray(home_override['ee'], dtype=float)
                # Warm-start from move-to-start IK if available
                _q_init = getattr(self, "_q_init", None)
                _seed = _q_init if _q_init is not None \
                        else np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
                q0, ok = self.manipulator.ik._solve_2r_core(
                    L1, L2, p_home, q_seed=_seed
                )
                if not ok:
                    q0 = None
                    print(colored(
                        f"  ⚠  IK failed for --home-ee "
                        f"({p_home[0]:.3f}, {p_home[1]:.3f}) m — falling back.",
                        "yellow",
                    ))
                else:
                    print(colored(
                        f"  ✓ Home (--home-ee): "
                        f"target=({p_home[0]*1e3:.1f}, {p_home[1]*1e3:.1f}) mm  "
                        f"→ q=[{np.rad2deg(q0[0]):.1f}°, {np.rad2deg(q0[1]):.1f}°]",
                        "cyan",
                    ))

        # ── Priority 2: pre-computed IK from build_move_to_start() ────────
        if q0 is None:
            q0 = getattr(self, "_q_init", None)

        # ── Priority 3: IK of first trajectory waypoint ───────────────────
        if q0 is None and self.ee_x_tgt is not None:
            L1, L2 = self.manipulator.ik.get_link_lengths(self.plant)
            p0 = np.array([self.ee_x_tgt[0], self.ee_y_tgt[0]])
            _seed = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
            q0, ok = self.manipulator.ik._solve_2r_core(L1, L2, p0, q_seed=_seed)
            if not ok:
                q0 = None
                print(colored("  ⚠  IK failed for first waypoint — using fallback.", "yellow"))

        # ── Priority 4: hard fallback ─────────────────────────────────────
        if q0 is None:
            q0 = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
            print(colored("  ⚠  Using hard fallback home pose: q=[5°, 15°].", "yellow"))

        self.manipulator.set_positions_user_order(self.plant, plant_ctx, q0)
        self.plant.SetVelocities(plant_ctx, np.zeros(self.plant.num_velocities()))

        # FK report
        ee_pos = self.manipulator.get_end_effector_position(self.plant, plant_ctx)
        print(colored(
            f"  ✓ Init: q=[{np.rad2deg(q0[0]):.1f}°, {np.rad2deg(q0[1]):.1f}°]  "
            f"EE=({ee_pos[0]*1e3:.1f}, {ee_pos[1]*1e3:.1f}) mm",
            "green",
        ))
        self.manipulator.ik.verify(self.plant, plant_ctx, label="init")

        self.simulator.set_target_realtime_rate(1.0)
        self.simulator.Initialize()
        self._cable_viz_tick()  # draw cables at the initial robot pose

    def _cable_viz_tick(self):
        """Redraw cable geometry in Meshcat at the current simulator state.

        No-op when meshcat is None or the cable rig was not initialised.
        Called once per outer simulation loop step (≈10 fps at dt_chunk=0.1 s).
        """
        if self.meshcat is None or not hasattr(self, 'rig') or self.rig is None:
            return
        ctx       = self.simulator.get_mutable_context()
        plant_ctx = self.plant.GetMyMutableContextFromRoot(ctx)
        self.manipulator.compute_tangents(self.plant, plant_ctx)
        draw_cables(self.meshcat, self.plant, plant_ctx, self.manipulator, self.rig)

    # ── 6. Run ────────────────────────────────────────────────────────────────
    def run(self, traj_shape: str = 'trajectory'):
        """Advance simulation in 0.1 s chunks until Ctrl+C. Prints lap counter."""
        context = self.simulator.get_mutable_context()
        _move_info = (
            f"  move-to-start: {self.move_duration:.1f} s  then  "
            if self.move_duration > 0.0 else ""
        )
        print(colored(
            f"\n▶  {_move_info}Looping {traj_shape} — lap={self.lap_duration:.1f} s (runs until Ctrl-C)"
            f"\n   J1: Kp1={self.KP1} Nm/rad  Kd1={self.KD1} Nm·s/rad"
            f"\n   J2: Kp_cable={self.KP_CABLE} N/m  Kd_cable={self.KD_CABLE} N·s/m"
            f"  tau_max={self.TAU_MAX} Nm"
            f"\n   dt={self._DT*1e3:.1f} ms  —  Press Ctrl-C to stop and show plots.",
            "cyan",
        ))
        _chunk         = 0.1
        _lap_prev      = 0
        _move_reported = self.move_duration <= 0.0
        try:
            while True:
                t_now = context.get_time()
                if not _move_reported and t_now >= self.move_duration:
                    _move_reported = True
                    print(colored(
                        f"  ✓ Move-to-start complete at t={t_now:.2f} s — trajectory tracking begins.",
                        "green",
                    ))
                _lap_now = int(max(0.0, t_now - self.move_duration) / self.lap_duration)
                if _lap_now > _lap_prev:
                    _lap_prev = _lap_now
                    print(colored(f"  Lap {_lap_now} complete  (t={t_now:.1f} s)", "cyan"))
                self.simulator.AdvanceTo(t_now + _chunk)
                self._cable_viz_tick()
        except KeyboardInterrupt:
            _elapsed_tracking = max(0.0, context.get_time() - self.move_duration)
            _laps = int(_elapsed_tracking / self.lap_duration)
            print(colored(
                f"\n  Simulation stopped at t={context.get_time():.2f} s  ({_laps} full laps).",
                "yellow",
            ))

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    def plot(self, traj_shape: str = ''):
        """Generate time-series and EE trajectory plots from logged data."""
        context   = self.simulator.get_mutable_context()
        t_log     = self.loggers['state'].FindLog(context).sample_times()
        state_log = self.loggers['state'].FindLog(context).data()      # (nstate, T)
        q_des_log = self.loggers['q_des'].FindLog(context).data()      # (2, T)
        act_log   = self.loggers['actuation'].FindLog(context).data()  # (2, T) torques [Nm]
        cable_log = self.loggers['cable'].FindLog(context).data()      # (2, T) displacements [m]
        tens_log  = self.loggers['tensions'].FindLog(context).data()   # (2, T) tensions [N]
        ref_log   = self.loggers['ref'].FindLog(context).data()        # (2, T)
        nq        = self.plant.num_positions()

        # FK loop to compute actual EE positions
        ee_x_log = np.full(len(t_log), np.nan)
        ee_y_log = np.full(len(t_log), np.nan)
        tmp_ctx  = self.plant.CreateDefaultContext()
        for k in range(len(t_log)):
            self.plant.SetPositionsAndVelocities(tmp_ctx, state_log[:, k])
            ee_pos       = self.manipulator.get_end_effector_position(self.plant, tmp_ctx)
            ee_x_log[k] = ee_pos[0]
            ee_y_log[k] = ee_pos[1]

        ee_x_ref = ref_log[0, :]
        ee_y_ref = ref_log[1, :]

        print(colored(f"\n  Final EE position : [{ee_x_log[-1]:.4f}, {ee_y_log[-1]:.4f}] m", "green"))
        print(colored(f"  Reference at t_end: [{ee_x_ref[-1]:.4f}, {ee_y_ref[-1]:.4f}] m", "green"))
        print(colored(f"  Tracking error    : {np.hypot(ee_x_log[-1]-ee_x_ref[-1], ee_y_log[-1]-ee_y_ref[-1])*1e3:.2f} mm", "green"))
        print(colored(f"  Mean tracking RMS : {np.sqrt(np.mean((ee_x_log-ee_x_ref)**2 + (ee_y_log-ee_y_ref)**2))*1e3:.2f} mm", "green"))

        # 4-panel time series
        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        fig.suptitle(
            f"IK-Diagram Closed-Loop — {traj_shape.capitalize()} Trajectory\n"
            f"J1: Kp1={self.KP1} Kd1={self.KD1}  |  J2 cable: Kp={self.KP_CABLE} Kd={self.KD_CABLE}"
            f"  dt={self._DT*1e3:.0f} ms  N={len(self.ee_x_tgt)} pts",
            fontsize=11, fontweight='bold',
        )

        ax0 = axes[0]
        ax0.plot(t_log, np.rad2deg(q_des_log[0]), 'b--', lw=1.2, label='q1 desired (IK)')
        ax0.plot(t_log, np.rad2deg(q_des_log[1]), 'r--', lw=1.2, label='q2 desired (IK)')
        ax0.plot(t_log, np.rad2deg(state_log[0]), 'b-',  lw=1.0, label='q1 actual')
        ax0.plot(t_log, np.rad2deg(state_log[1]), 'r-',  lw=1.0, label='q2 actual')
        ax0.set_ylabel('Joint angle [deg]')
        ax0.legend(fontsize=8, ncol=2)
        ax0.grid(True, alpha=0.4)
        ax0.set_title('Joint Angles — Desired (IK) vs Actual (Plant)')

        ax1 = axes[1]
        ax1.plot(t_log, (ee_x_log - ee_x_ref) * 1e3, 'b-', label='X error')
        ax1.plot(t_log, (ee_y_log - ee_y_ref) * 1e3, 'r-', label='Y error')
        ax1.axhline(0, color='k', lw=0.7)
        ax1.set_ylabel('EE tracking error [mm]')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)
        ax1.set_title('End-Effector Cartesian Tracking Error (actual − reference)')

        # Panel 2: Applied torques
        ax2 = axes[2]
        ax2.plot(t_log, act_log[0], 'b-', lw=1.2, label='τ1 (direct drive) [Nm]')
        ax2.plot(t_log, act_log[1], 'r-', lw=1.2, label='τ2 (cable joint) [Nm]')
        ax2.axhline( self.TAU_MAX, color='k', ls=':', lw=0.8, label=f'±{self.TAU_MAX} Nm limit')
        ax2.axhline(-self.TAU_MAX, color='k', ls=':', lw=0.8)
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_ylabel('Applied torque [Nm]')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.4)
        ax2.set_title('Applied Joint Torques (PD controller output, clipped at ±tau_max)')

        ax3 = axes[3]
        r_p = self.manipulator.PULLEY_RADIUS
        tau2_log = (tens_log[0] - tens_log[1]) * r_p   # net torque [Nm]
        ax3.plot(t_log, tens_log[0], 'g-',  lw=1.2, label='T_green (retracting side) [N]')
        ax3.plot(t_log, tens_log[1], 'r-',  lw=1.2, label='T_red   (extending side)  [N]')
        ax3.plot(t_log, tau2_log,    'k--', lw=0.9, label='τ2 = (T_green−T_red)·r_p  [Nm]')
        ax3.axhline(0, color='k', lw=0.5)
        ax3.set_ylabel('Tension [N] / Torque [Nm]')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.4)
        ax3.set_title('Cable Tensions  (τ = J_cᵀ · T,   J_c = r_p = {:.1f} mm)'.format(r_p * 1e3))

        ax4 = axes[4]
        ax4.plot(t_log, np.rad2deg(state_log[nq + 0]), 'b-', label='q1_dot')
        ax4.plot(t_log, np.rad2deg(state_log[nq + 1]), 'r-', label='q2_dot')
        ax4.axhline(0, color='k', lw=0.7)
        ax4.set_ylabel('Joint velocity [deg/s]')
        ax4.set_xlabel('Time [s]')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)
        ax4.set_title('Joint Velocities')
        fig.tight_layout()

        # EE XY trajectory
        fig2, ax_ee = plt.subplots(figsize=(6, 5))
        ax_ee.plot(self.ee_x_tgt, self.ee_y_tgt, 'k--', lw=1.0, label='Reference trajectory')
        ax_ee.plot(ee_x_log,      ee_y_log,       'b-',  lw=1.2, label='EE path (actual)')
        ax_ee.plot(ee_x_log[0],   ee_y_log[0],    'go',  ms=8,   label='Start')
        ax_ee.set_aspect('equal')
        ax_ee.set_xlabel('X [m]')
        ax_ee.set_ylabel('Y [m]')
        ax_ee.set_title(f'EE Trajectory — world XY plane ({traj_shape})')
        ax_ee.legend(fontsize=9)
        ax_ee.grid(True, alpha=0.4)
        fig2.tight_layout()

        # ── Save to plots/ directory (always works, regardless of display backend) ──
        import time as _time
        _stamp = _time.strftime('%Y%m%d_%H%M%S')
        _plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(_plots_dir, exist_ok=True)
        _f1 = os.path.join(_plots_dir, f'ik_diagram_{traj_shape}_{_stamp}.png')
        _f2 = os.path.join(_plots_dir, f'ik_diagram_{traj_shape}_ee_{_stamp}.png')
        fig.savefig(_f1, dpi=150, bbox_inches='tight')
        fig2.savefig(_f2, dpi=150, bbox_inches='tight')
        print(colored(f"\n  📊 Figures saved:\n     {_f1}\n     {_f2}", "green"))

        # ── Interactive display (requires a GUI backend) ──────────────────────────
        try:
            plt.show(block=True)
        except Exception as _e:
            print(colored(f"  ⚠ plt.show() failed ({_e}) — open the saved PNG files above.", "yellow"))


# ============================================================================
# COMPUTED-TORQUE SIMULATION
# ============================================================================

class ComputedTorqueSimulation(IKPDSimulation):
    """Closed-loop computed-torque Drake diagram simulation.

    Identical builder pattern to IKPDSimulation; overrides:
      build_controller()    — uses ComputedTorqueController instead of CupManipulatorIKSystem
      connect_and_build()   — wires ``torques_raw`` logger instead of ``cable_lengths``
      plot()                — replaces cable-length panel with raw joint torques
      run()                 — prints CT-specific gain info

    Diagram topology (same as IKPDSimulation):
      LoopingTrajectorySource ──► ComputedTorqueController ──► MultibodyPlant
                                           ▲ plant_state ──────────────┘
    """

    def __init__(self, manip_config, meshcat=None, Kp=400.0, Kd=40.0, tau_max=10.0):
        super().__init__(manip_config, meshcat)
        self._ct_Kp      = float(Kp)
        self._ct_Kd      = float(Kd)
        self._ct_tau_max = float(tau_max)

    # ── 3. Controller (override) ──────────────────────────────────────────────
    def build_controller(self):
        """Replace IK-PD controller with ComputedTorqueController."""
        self.ik_system = self.builder.AddSystem(
            ComputedTorqueController(
                self.plant, self.manipulator,
                Kp=self._ct_Kp, Kd=self._ct_Kd, tau_max=self._ct_tau_max,
            )
        )
        self.ik_system.set_name("ComputedTorque_Controller")

    # ── 4. Connect + Build (override) ─────────────────────────────────────────
    def connect_and_build(self):
        """Wire connections, add loggers (torques_raw replaces cable_lengths)."""

        # _PreambleSource: during move-to-start (t < move_duration) outputs the
        # approach spline; afterwards wraps into the main looping trajectory.
        class _PreambleSource(LeafSystem):
            def __init__(inner_self, move_traj, move_dur, main_traj, period):
                super().__init__()
                inner_self._move_traj = move_traj
                inner_self._move_dur  = float(move_dur)
                inner_self._main_traj = main_traj
                inner_self._period    = float(period)
                inner_self.DeclareVectorOutputPort("output", main_traj.rows(), inner_self._calc)

            def _calc(inner_self, context, output):
                t = context.get_time()
                if inner_self._move_traj is not None and t < inner_self._move_dur:
                    val = inner_self._move_traj.value(t).ravel()
                else:
                    t_wrap = max(0.0, t - inner_self._move_dur) % inner_self._period
                    val = inner_self._main_traj.value(t_wrap).ravel()
                output.SetFromVector(val)

        self.ee_ref     = self.builder.AddSystem(
            _PreambleSource(self.move_traj,     self.move_duration, self.traj_ref,     self.lap_duration)
        )
        self.ee_vel_src = self.builder.AddSystem(
            _PreambleSource(self.move_traj_vel, self.move_duration, self.traj_vel_ref, self.lap_duration)
        )
        self.ee_acc_src = self.builder.AddSystem(
            _PreambleSource(self.move_traj_acc, self.move_duration, self.traj_acc_ref, self.lap_duration)
        )
        self.ee_ref.set_name("EE_Position_Ref")
        self.ee_vel_src.set_name("EE_Velocity_Ref")
        self.ee_acc_src.set_name("EE_Accel_Ref")

        self.builder.Connect(
            self.ee_ref.get_output_port(),
            self.ik_system.GetInputPort("desired_ee_pos"),
        )
        self.builder.Connect(
            self.ee_vel_src.get_output_port(),
            self.ik_system.GetInputPort("ee_vel_ref"),
        )
        self.builder.Connect(
            self.ee_acc_src.get_output_port(),
            self.ik_system.GetInputPort("ee_acc_ref"),
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.ik_system.GetInputPort("plant_state"),
        )
        self.builder.Connect(
            self.ik_system.GetOutputPort("actuation"),
            self.plant.get_actuation_input_port(),
        )

        self.loggers['state']      = LogVectorOutput(self.plant.get_state_output_port(),              self.builder)
        self.loggers['q_des']      = LogVectorOutput(self.ik_system.GetOutputPort("joint_positions"), self.builder)
        self.loggers['tau_raw']    = LogVectorOutput(self.ik_system.GetOutputPort("torques_raw"),     self.builder)
        self.loggers['actuation']  = LogVectorOutput(self.ik_system.GetOutputPort("actuation"),       self.builder)
        self.loggers['tensions']   = LogVectorOutput(self.ik_system.GetOutputPort("cable_tensions"),  self.builder)
        self.loggers['ref']        = LogVectorOutput(self.ee_ref.get_output_port(),                   self.builder)
        self.loggers['vel_ref']    = LogVectorOutput(self.ee_vel_src.get_output_port(),               self.builder)
        self.loggers['acc_ref']    = LogVectorOutput(self.ee_acc_src.get_output_port(),               self.builder)

        self.diagram   = self.builder.Build()
        self.simulator = Simulator(self.diagram)

    # ── 6. Run (override) ────────────────────────────────────────────────────
    def run(self, traj_shape: str = 'trajectory'):
        context = self.simulator.get_mutable_context()
        wn   = np.sqrt(self._ct_Kp)
        zeta = self._ct_Kd / (2.0 * wn) if wn > 0 else 0.0
        _move_info = (
            f"  move-to-start: {self.move_duration:.1f} s  then  "
            if self.move_duration > 0.0 else ""
        )
        print(colored(
            f"\n▶  COMPUTED-TORQUE — {_move_info}Looping {traj_shape}"
            f" — lap={self.lap_duration:.1f} s  (runs until Ctrl-C)"
            f"\n   Gains: Kp={self._ct_Kp}  Kd={self._ct_Kd}"
            f"   →  ωn={wn:.1f} rad/s  ζ={zeta:.2f}"
            f"\n   tau_max={self._ct_tau_max} Nm   dt={self._DT*1e3:.1f} ms"
            f"\n   Press Ctrl-C to stop and show plots.",
            "cyan",
        ))
        _chunk         = 0.1
        _lap_prev      = 0
        _move_reported = self.move_duration <= 0.0
        try:
            while True:
                t_now = context.get_time()
                if not _move_reported and t_now >= self.move_duration:
                    _move_reported = True
                    print(colored(
                        f"  ✓ Move-to-start complete at t={t_now:.2f} s — trajectory tracking begins.",
                        "green",
                    ))
                _lap_now = int(max(0.0, t_now - self.move_duration) / self.lap_duration)
                if _lap_now > _lap_prev:
                    _lap_prev = _lap_now
                    print(colored(f"  Lap {_lap_now} complete  (t={t_now:.1f} s)", "cyan"))
                self.simulator.AdvanceTo(t_now + _chunk)
                self._cable_viz_tick()
        except KeyboardInterrupt:
            _elapsed_tracking = max(0.0, context.get_time() - self.move_duration)
            _laps = int(_elapsed_tracking / self.lap_duration)
            print(colored(
                f"\n  Simulation stopped at t={context.get_time():.2f} s  ({_laps} full laps).",
                "yellow",
            ))

    # ── 7. Plot (override) ────────────────────────────────────────────────────
    def plot(self, traj_shape: str = ''):
        """3-column time-series (position | velocity | acceleration) + EE XY path."""
        _logs = {k: self.loggers[k].FindLog(self.simulator.get_context())
                 for k in ('state', 'q_des', 'tau_raw', 'actuation',
                           'tensions', 'ref', 'vel_ref', 'acc_ref')}

        # Each logger may have a slightly different number of samples when
        # Ctrl+C fires mid-step.  Build a common uniform time grid from the
        # state logger, then resample every other logger onto it using its
        # own sample_times() as the source axis.
        t_log   = _logs['state'].sample_times()
        t_uni   = np.linspace(t_log[0], t_log[-1], len(t_log))

        def _resamp(log_obj):
            """Resample a logger onto t_uni using its own sample_times() as xp."""
            xp  = log_obj.sample_times()
            sig = log_obj.data()
            # Clamp t_uni to the logger's actual range to avoid extrapolation
            xp0, xp1 = xp[0], xp[-1]
            x = np.clip(t_uni, xp0, xp1)
            if sig.ndim == 1:
                return np.interp(x, xp, sig)
            return np.vstack([np.interp(x, xp, sig[i]) for i in range(sig.shape[0])])

        state_log = _resamp(_logs['state'])
        q_des_log = _resamp(_logs['q_des'])
        tau_log   = _resamp(_logs['tau_raw'])
        act_log   = _resamp(_logs['actuation'])
        tens_log  = _resamp(_logs['tensions'])
        ref_log   = _resamp(_logs['ref'])
        vel_log   = _resamp(_logs['vel_ref'])   # EE velocity reference  (2, T) [m/s]
        acc_log   = _resamp(_logs['acc_ref'])   # EE acceleration reference (2, T) [m/s²]
        t_log     = t_uni

        nq  = self.plant.num_positions()
        r_p = self.manipulator.PULLEY_RADIUS

        # ── EE actual position via FK ────────────────────────────────────────
        plant_ctx_plot = self.plant.CreateDefaultContext()
        ee_x_log = np.zeros(len(t_log))
        ee_y_log = np.zeros(len(t_log))
        for k, t in enumerate(t_log):
            self.plant.SetPositionsAndVelocities(plant_ctx_plot, state_log[:, k])
            p = self.manipulator.get_end_effector_position(self.plant, plant_ctx_plot)
            ee_x_log[k], ee_y_log[k] = p[0], p[1]

        # ── Derived signals ──────────────────────────────────────────────────
        # Joint velocity (actual) read directly from resampled state — no diff needed
        q1_dot_act  = state_log[nq]
        q2_dot_act  = state_log[nq + 1]

        # Analytical link lengths and Jacobians at ACTUAL joint positions
        L1, L2 = self.manipulator.ik.get_link_lengths(self.plant)

        q1_act  = state_log[0];  q2_act  = state_log[1]
        s1_act  = np.sin(q1_act);              c1_act  = np.cos(q1_act)
        s12_act = np.sin(q1_act + q2_act);     c12_act = np.cos(q1_act + q2_act)

        # EE velocity actual = J(q_act) @ q_dot_act  — exact, no finite diff
        ee_vx_act = (-L1*s1_act - L2*s12_act)*q1_dot_act + (-L2*s12_act)*q2_dot_act
        ee_vy_act = ( L1*c1_act + L2*c12_act)*q1_dot_act + ( L2*c12_act)*q2_dot_act

        # EE acceleration actual = d/dt(EE vel) via gradient on the clean J@qdot signal
        ee_ax_act = np.gradient(ee_vx_act, t_log)
        ee_ay_act = np.gradient(ee_vy_act, t_log)

        # Joint acceleration actual via finite diff of clean joint velocities
        q1_ddot_act = np.gradient(q1_dot_act, t_log)
        q2_ddot_act = np.gradient(q2_dot_act, t_log)

        # Joint velocity / acceleration reference via analytical J^{-1} at desired joints
        # J(q) = [[-L1 s1 - L2 s12, -L2 s12],
        #          [ L1 c1 + L2 c12,  L2 c12]]
        q1d = q_des_log[0];  q2d = q_des_log[1]
        s1  = np.sin(q1d);   c1  = np.cos(q1d)
        s12 = np.sin(q1d + q2d);  c12 = np.cos(q1d + q2d)
        J_all = np.stack([
            np.stack([-L1 * s1 - L2 * s12, -L2 * s12], axis=1),
            np.stack([ L1 * c1 + L2 * c12,  L2 * c12], axis=1),
        ], axis=1)   # shape (T, 2, 2)
        q_dot_ref  = np.array([np.linalg.pinv(J_all[k]) @ vel_log[:, k]
                                for k in range(len(t_log))])   # (T, 2)
        q_ddot_ref = np.array([np.linalg.pinv(J_all[k]) @ acc_log[:, k]
                                for k in range(len(t_log))])   # (T, 2)

        wn   = np.sqrt(self._ct_Kp)
        zeta = self._ct_Kd / (2.0 * wn) if wn > 0 else 0.0

        def _pct_ylim(*arrays, pct=99.0, margin=0.15):
            """Y-limits based on percentile — ignores startup spikes."""
            all_vals = np.concatenate([a.ravel() for a in arrays])
            lo = np.percentile(all_vals, 100 - pct)
            hi = np.percentile(all_vals, pct)
            span = max(hi - lo, 1e-9)
            return lo - margin * span, hi + margin * span

        # ── 3 × 3 figure ─────────────────────────────────────────────────────
        # Col 0 = Position | Col 1 = Velocity | Col 2 = Acceleration
        # Row 0 = EE       | Row 1 = Joints   | Row 2 = Torques/Tensions/XY
        fig, axes = plt.subplots(3, 3, figsize=(18, 11))
        fig.suptitle(
            f'Computed Torque — {traj_shape}   '
            f'Kp={self._ct_Kp}  Kd={self._ct_Kd}  '
            f'ωn={wn:.1f} rad/s  ζ={zeta:.2f}',
            fontsize=12, fontweight='bold',
        )

        # ── Row 0: End-Effector ───────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(t_log, ee_x_log,   'b-',  lw=1.8, label='x actual')
        ax.plot(t_log, ee_y_log,   'r-',  lw=1.8, label='y actual')
        ax.plot(t_log, ref_log[0], 'b--', lw=1.5, label='x ref')
        ax.plot(t_log, ref_log[1], 'r--', lw=1.5, label='y ref')
        ax.set_title('EE Position'); ax.set_ylabel('[m]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        ax = axes[0, 1]
        ax.plot(t_log, ee_vx_act,   'b-',  lw=1.8, label='ẋ actual')
        ax.plot(t_log, ee_vy_act,   'r-',  lw=1.8, label='ẏ actual')
        ax.plot(t_log, vel_log[0],  'b--', lw=1.5, label='ẋ ref')
        ax.plot(t_log, vel_log[1],  'r--', lw=1.5, label='ẏ ref')
        ax.set_ylim(*_pct_ylim(ee_vx_act, ee_vy_act, vel_log[0], vel_log[1]))
        ax.set_title('EE Velocity'); ax.set_ylabel('[m/s]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        ax = axes[0, 2]
        ax.plot(t_log, ee_ax_act,   'b-',  lw=1.8, label='ẍ actual')
        ax.plot(t_log, ee_ay_act,   'r-',  lw=1.8, label='ÿ actual')
        ax.plot(t_log, acc_log[0],  'b--', lw=1.5, label='ẍ ref')
        ax.plot(t_log, acc_log[1],  'r--', lw=1.5, label='ÿ ref')
        ax.set_ylim(*_pct_ylim(ee_ax_act, ee_ay_act, acc_log[0], acc_log[1]))
        ax.set_title('EE Acceleration'); ax.set_ylabel('[m/s²]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        # ── Row 1: Joints ─────────────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(t_log, np.rad2deg(state_log[0]), 'b-',  lw=1.8, label='q1 act')
        ax.plot(t_log, np.rad2deg(state_log[1]), 'r-',  lw=1.8, label='q2 act')
        ax.plot(t_log, np.rad2deg(q_des_log[0]), 'b--', lw=1.5, label='q1 des')
        ax.plot(t_log, np.rad2deg(q_des_log[1]), 'r--', lw=1.5, label='q2 des')
        ax.set_title('Joint Position'); ax.set_ylabel('[deg]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        ax = axes[1, 1]
        ax.plot(t_log, np.rad2deg(q1_dot_act),      'b-',  lw=1.8, label='q̇1 act')
        ax.plot(t_log, np.rad2deg(q2_dot_act),      'r-',  lw=1.8, label='q̇2 act')
        ax.plot(t_log, np.rad2deg(q_dot_ref[:, 0]), 'b--', lw=1.5, label='q̇1 ref')
        ax.plot(t_log, np.rad2deg(q_dot_ref[:, 1]), 'r--', lw=1.5, label='q̇2 ref')
        ax.set_ylim(*_pct_ylim(np.rad2deg(q1_dot_act), np.rad2deg(q2_dot_act),
                               np.rad2deg(q_dot_ref[:, 0]), np.rad2deg(q_dot_ref[:, 1])))
        ax.set_title('Joint Velocity'); ax.set_ylabel('[deg/s]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        ax = axes[1, 2]
        ax.plot(t_log, np.rad2deg(q1_ddot_act),      'b-',  lw=1.8, label='q̈1 act')
        ax.plot(t_log, np.rad2deg(q2_ddot_act),      'r-',  lw=1.8, label='q̈2 act')
        ax.plot(t_log, np.rad2deg(q_ddot_ref[:, 0]), 'b--', lw=1.5, label='q̈1 ref')
        ax.plot(t_log, np.rad2deg(q_ddot_ref[:, 1]), 'r--', lw=1.5, label='q̈2 ref')
        ax.set_ylim(*_pct_ylim(np.rad2deg(q1_ddot_act), np.rad2deg(q2_ddot_act),
                               np.rad2deg(q_ddot_ref[:, 0]), np.rad2deg(q_ddot_ref[:, 1])))
        ax.set_title('Joint Acceleration'); ax.set_ylabel('[deg/s²]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        # ── Row 2: Torques | Tensions | EE XY path ───────────────────────────
        ax = axes[2, 0]
        ax.plot(t_log, tau_log[0], 'b-',  lw=1.8, label='τ1 required')
        ax.plot(t_log, tau_log[1], 'r-',  lw=1.8, label='τ2 required')
        ax.plot(t_log, act_log[0], 'b--', lw=1.5, label='τ1 applied')
        ax.plot(t_log, act_log[1], 'r--', lw=1.5, label='τ2 applied')
        ax.axhline( self._ct_tau_max, color='k', ls=':', lw=0.8, label=f'±{self._ct_tau_max} Nm')
        ax.axhline(-self._ct_tau_max, color='k', ls=':', lw=0.8)
        ax.axhline(0, color='k', lw=0.5)
        _tau_peak = max(np.abs(tau_log).max(), self._ct_tau_max) * 1.15
        ax.set_ylim(-_tau_peak, _tau_peak)
        ax.set_title('Torque: required (solid) vs applied (dashed)')
        ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

        ax = axes[2, 1]
        ax.plot(t_log, tens_log[0],      'g-',  lw=1.2, label='T_green')
        ax.plot(t_log, tens_log[1],      'r-',  lw=1.2, label='T_red')
        ax.plot(t_log, tau_log[1] / r_p, 'k--', lw=0.8,
                label=f'F_net=τ2/r_p  (r_p={r_p*1e3:.1f} mm)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title('Cable Tensions'); ax.set_ylabel('[N]'); ax.set_xlabel('Time [s]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

        ax = axes[2, 2]
        ax.plot(self.ee_x_tgt, self.ee_y_tgt, 'k--', lw=1.0, label='Reference')
        ax.plot(ee_x_log,      ee_y_log,       'b-',  lw=1.3, label='Actual')
        ax.plot(ee_x_log[0],   ee_y_log[0],    'go',  ms=8,   label='Start')
        ax.set_aspect('equal')
        ax.set_title(f'EE Path ({traj_shape})')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

        fig.tight_layout()

        import time as _time
        _stamp = _time.strftime('%Y%m%d_%H%M%S')
        _plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(_plots_dir, exist_ok=True)
        _f1 = os.path.join(_plots_dir, f'computed_torque_{traj_shape}_{_stamp}.png')
        fig.savefig(_f1, dpi=150, bbox_inches='tight')
        print(colored(f"\n  📊 Figure saved:\n     {_f1}", "green"))

        try:
            plt.show(block=True)
        except Exception as _e:
            print(colored(f"  ⚠ plt.show() failed ({_e}) — open the saved PNG files above.", "yellow"))


# ============================================================================
# HELPERS
# ============================================================================

def _parse_home_args(args):
    """Convert --home-ee / --home-joints into a home_override dict for initialize().

    Returns:
        {'ee':     np.array([x, y])}      if --home-ee was given
        {'joints': np.array([q1, q2])}    if --home-joints was given (radians)
        None                              if neither flag was given (auto-resolve)
    """
    if getattr(args, 'home_ee', None) is not None and \
       getattr(args, 'home_joints', None) is not None:
        parser.error('--home-ee and --home-joints are mutually exclusive; specify only one.')
    if getattr(args, 'home_ee', None) is not None:
        return {'ee': np.array(args.home_ee, dtype=float)}
    if getattr(args, 'home_joints', None) is not None:
        return {'joints': np.deg2rad(np.array(args.home_joints, dtype=float))}
    return None


# ============================================================================
# SIMULATION REGISTRY
# Maps CLI --mode strings to (SimulationClass, extra_kwargs_fn, banner_label).
# All entries share the same call sequence:
#   sim = Cls(config, meshcat=meshcat, **kwargs_fn(args))
#   sim.build_plant() → build_trajectory() → build_move_to_start()
#     → build_controller() → connect_and_build() → initialize() → run() → plot()
# ============================================================================

_SIM_REGISTRY: dict = {
    'pd-control': (
        IKPDSimulation,
        lambda args: {},
        "PD CONTROL — IK + joint-space PD simulation",
    ),
    'computed-torque': (
        ComputedTorqueSimulation,
        lambda args: dict(Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max),
        "COMPUTED TORQUE — Inverse-dynamics closed-loop simulation",
    ),
}


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    
    # ========================================================================
    # BUILD CONTROL DIAGRAM
    # ========================================================================
    # Build the diagram for the selected control mode
    # Plant and scene_graph are added fresh to each mode's builder
    # For control modes, we pass them to the mode functions which handle the setup
    
    

    if args.mode == 'scene-viz-q':
        # ====================================================================
        # SCENE-VIZ-Q: Interactive joint-angle control of the cable manipulator
        # ====================================================================
        # Lets the user type q1 q2 (degrees) at the prompt; the manipulator
        # pose in Meshcat updates immediately on each entry.
        # No cart-pendulum is added — the plant contains only the manipulator.
        # ====================================================================
        print("\n" + "="*80)
        print(colored("CABLE MANIPULATOR — JOINT-ANGLE INTERACTIVE VIEWER", "cyan", attrs=["bold"]))
        print("="*80)

        meshcat = StartMeshcat()
        print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3)
        )

        # ── Cable rig setup ───────────────────────────────────────────────────
        _URDF_PATH = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"

        # ------------------------------------------------------------------
        # Build plant with manipulator only
        # ------------------------------------------------------------------
        builder = DiagramBuilder()
        plant   = MultibodyPlant(time_step=0.0)
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)

        manipulator = CupManipulatorTendon(CABLE_MANIPULATOR_CONFIG, enable_visualization=True)
        parser_urdf = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser_urdf)
        manipulator.weld_base_to_world(plant, position=np.zeros(3),
                                        orientation=np.deg2rad([args.tilt_roll, args.tilt_pitch, 0.0]))
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        plant.Finalize()

        manipulator.init_cable_rig(_URDF_PATH)
        rig = manipulator.rig  # local alias for draw_cables / viz helpers

        builder.AddSystem(plant)
        builder.Connect(
            plant.get_geometry_pose_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id())
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port()
        )

        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        diagram     = builder.Build()
        simulator   = Simulator(diagram)
        context     = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        # Start from home position
        current_q = np.array([0.0, 0.0])   # [q1, q2] in radians
        manipulator.set_positions_user_order(plant, plant_context, {
            "link1_base":  current_q[0],
            "link2_link1": current_q[1],
        })
        plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
        diagram.ForcedPublish(context)

        # ── Draw cables at home pose ──────────────────────────────────────────
        manipulator.compute_tangents(plant, plant_context)
        draw_cables(meshcat, plant, plant_context, manipulator, rig)
        print_cable_routing_points(plant, plant_context, manipulator, rig)
        _top_fig, _ = visualize_cable_routing_top_view(
            plant, plant_context, manipulator, 0.0, 0.0, rig
        )
        plt.show(block=False)
        plt.pause(0.05)

        # Draw joint and EE coordinate-frame triads (X=red, Y=green, Z=blue)
        # add_frames_to_meshcat creates one triad per plant frame (body frames +
        # the custom tendon_ee frame).  Returns frame_list for cheap re-updates.
        frame_list = add_frames_to_meshcat(meshcat, plant, plant_context, manipulator)

        # Lightweight helper: only update triad positions, no redundant SetObject calls
        def _update_frame_triads():
            for frame_name, frame, _ in frame_list:
                X_WF = plant.CalcRelativeTransform(
                    plant_context, plant.world_frame(), frame
                )
                meshcat.SetTransform(f"/Frames/{frame_name}", X_WF)

        ee_pos = manipulator.get_end_effector_position(plant, plant_context)
        print(colored(f"\n📄 Home position:", "cyan"))
        print(colored(f"  q1 = {np.rad2deg(current_q[0]):.1f}°   q2 = {np.rad2deg(current_q[1]):.1f}°", "cyan"))
        print(colored(f"  EE = ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m", "yellow"))
        print(colored("  Frame triads: X=Red  Y=Green  Z=Blue", "yellow"))

        print(colored("\n   Enter joint angles in degrees.", "yellow"))
        print(colored("   Format:  q1  q2    (e.g.  30  -15)", "yellow"))
        print(colored("   Ctrl+C to exit.\n", "yellow"))

        try:
            while True:
                user_input = input(colored("q1  q2 [deg]: ", "cyan")).strip()
                if not user_input:
                    continue
                try:
                    parts = user_input.split()
                    if len(parts) != 2:
                        print(colored("  ✗ Expected exactly 2 values: q1 q2", "red"))
                        continue
                    q1_deg = float(parts[0])
                    q2_deg = float(parts[1])
                    current_q = np.array([np.deg2rad(q1_deg), np.deg2rad(q2_deg)])

                    manipulator.set_positions_user_order(plant, plant_context, {
                        "link1_base":  current_q[0],
                        "link2_link1": current_q[1],
                    })
                    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
                    diagram.ForcedPublish(context)
                    _update_frame_triads()  # move frame triads to new pose

                    # Recompute cable tangents and redraw
                    manipulator.compute_tangents(plant, plant_context)
                    draw_cables(meshcat, plant, plant_context, manipulator, rig)
                    plt.close(_top_fig)
                    _top_fig, _ = visualize_cable_routing_top_view(
                        plant, plant_context, manipulator, q1_deg, q2_deg, rig
                    )
                    plt.show(block=False)
                    plt.pause(0.05)

                    ee_pos = manipulator.get_end_effector_position(plant, plant_context)
                    print(colored(
                        f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                        f"→  EE=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m",
                        "green"
                    ))
                except ValueError:
                    print(colored("  ✗ Invalid numbers. Enter two floats: q1 q2", "red"))
        except KeyboardInterrupt:
            print(colored("\n✓ scene-viz-q stopped.", "green"))

        return

    elif args.mode == 'ee-trajectory':
        # ====================================================================
        # EE-TRAJECTORY: IK along a Cartesian path → joint angles + cable lengths
        # ====================================================================
        # Sweeps the end-effector (simple_ball_5, xyz=[0.19, 0, 0.0515] on
        # link2_tendon) along a circle or line in the world XZ-plane.
        # At each point: IK → (q1, q2) → cable tangents → cable lengths.
        #
        # Cable lengths are the world-frame path lengths:
        #   Green: drive_pulley.B_R → idlerR → bigPulley → cable_end_l
        #   Red  : drive_pulley.B_L → idlerL → bigPulley → cable_end_r
        # ====================================================================
        print("\n" + "="*80)
        print(colored("EE TRAJECTORY — IK + CABLE LENGTHS", "cyan", attrs=["bold"]))
        print("="*80)

        # ── Build plant ───────────────────────────────────────────────────────
        _URDF_PATH = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
        builder     = DiagramBuilder()
        plant       = MultibodyPlant(time_step=0.0)
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)

        manipulator = CupManipulatorTendon(CABLE_MANIPULATOR_CONFIG, enable_visualization=True)
        parser_urdf = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser_urdf)
        manipulator.weld_base_to_world(plant, position=np.zeros(3),
                                        orientation=np.deg2rad([args.tilt_roll, args.tilt_pitch, 0.0]))
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        plant.Finalize()

        manipulator.init_cable_rig(_URDF_PATH)
        rig = manipulator.rig

        # Meshcat for 3-D animation
        meshcat = StartMeshcat()
        print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3)
        )
        builder.AddSystem(plant)
        builder.Connect(
            plant.get_geometry_pose_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id())
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port()
        )
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        diagram   = builder.Build()
        simulator = Simulator(diagram)
        context   = simulator.get_mutable_context()
        plant_ctx = plant.GetMyMutableContextFromRoot(context)

        # ── Home pose → get EE world Z so the trajectory stays at that height ─
        manipulator.set_positions_user_order(plant, plant_ctx, {"link1_base": 0.0, "link2_link1": 0.0})
        plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
        diagram.ForcedPublish(context)
        ee_home = manipulator.get_end_effector_position(plant, plant_ctx)
        z_traj  = ee_home[2]   # keep Z constant along the trajectory

        # ── Build EE target list (world XY-plane, Z fixed at home height) ──────
        # The robot moves in the horizontal XY plane; Z (world vertical) is held
        # constant at the home EE height (z_traj).
        cx, cy = args.traj_cx, args.traj_cy
        R, N   = args.traj_radius, args.traj_n

        if args.traj_shape == 'circle':
            angles   = np.linspace(0, 2 * np.pi, N, endpoint=False)
            ee_x_tgt = cx + R * np.cos(angles)
            ee_y_tgt = cy + R * np.sin(angles)
            traj_param = np.rad2deg(angles)
            traj_label = 'Angle [deg]'
        elif args.traj_shape == 'rect':
            # Walk the perimeter of the [x_min,x_max] × [y_min,y_max] rectangle
            x_min, x_max = args.traj_x_range
            y_min, y_max = args.traj_y_range
            n4 = N // 4   # points per side
            # Bottom: left→right, Right: bottom→top, Top: right→left, Left: top→bottom
            sides_x = np.concatenate([
                np.linspace(x_min, x_max, n4, endpoint=False),
                np.full(n4, x_max),
                np.linspace(x_max, x_min, n4, endpoint=False),
                np.full(N - 3 * n4, x_min),
            ])
            sides_y = np.concatenate([
                np.full(n4, y_min),
                np.linspace(y_min, y_max, n4, endpoint=False),
                np.full(n4, y_max),
                np.linspace(y_max, y_min, N - 3 * n4, endpoint=False),
            ])
            ee_x_tgt   = sides_x
            ee_y_tgt   = sides_y
            traj_param = np.arange(N)
            traj_label = 'Step'
        else:  # line
            ee_x_tgt   = np.linspace(cx - R, cx + R, N)
            ee_y_tgt   = np.full(N, cy)
            traj_param = ee_x_tgt
            traj_label = 'EE X [m]'

        # ── IK sweep ─────────────────────────────────────────────────────────
        q1_arr      = np.full(N, np.nan)
        q2_arr      = np.full(N, np.nan)
        len_green   = np.full(N, np.nan)
        len_red     = np.full(N, np.nan)
        ee_x_actual = np.full(N, np.nan)
        ee_y_actual = np.full(N, np.nan)

        # ── Reference cable length at home (q1=q2=0) — computed FIRST ─────────
        manipulator.compute_tangents(plant, plant_ctx)  # already at home from ForcedPublish above
        L0_green = manipulator.length_cable_route(rig.cable_green, plant, plant_ctx, "Drive exit B_R")
        L0_red   = manipulator.length_cable_route(rig.cable_red,   plant, plant_ctx, "Drive exit B_L")
        print(colored(f"\n  Home cable lengths (q1=q2=0):", "cyan"))
        print(colored(f"    Green: {L0_green*1e3:.2f} mm", "green"))
        print(colored(f"    Red  : {L0_red*1e3:.2f} mm", "red"))

        q_seed = np.array([0.0, 0.0])   # warm-start seed for both IK methods

        Q2_DIAG_LIMIT = np.deg2rad(20)   # for diagnostic reporting only (not blocking)
        q2_warn_count = 0

        use_vel_ik    = (args.ik_method == 'velocity')
        use_hybrid_ik = (args.ik_method == 'hybrid')
        # For velocity / hybrid IK: dt between consecutive trajectory points
        dt_traj = 1.0 / N   # treat one full revolution as 1 s

        if use_vel_ik or use_hybrid_ik:
            # Both velocity modes need a valid starting configuration — bootstrap with analytical IK
            q_seed_init, ok_init = manipulator.compute_ik_analytical(
                plant, np.array([ee_x_tgt[0], ee_y_tgt[0]]), q_seed, pos_tol=5e-3
            )
            if ok_init:
                q_seed = q_seed_init
            ik_label = (
                f"velocity IK, λ={args.ik_damping:.0e}" if use_vel_ik
                else f"hybrid IK (actuation space), λ={args.ik_damping:.0e}"
            )
            print(colored(
                f"\nSolving IK for {N} trajectory points ({ik_label},"
                f" dt={dt_traj*1e3:.1f} ms/step) ...", "cyan"
            ))
        else:
            print(colored(f"\nSolving IK for {N} trajectory points (analytical 2R IK) ...", "cyan"))

        for i, (ex, ey) in enumerate(zip(ee_x_tgt, ee_y_tgt)):
            target_xy = np.array([ex, ey])   # world [X, Y] — Z locked to z_traj

            if use_vel_ik:
                # ── Velocity IK: q_dot = J⁺ x_dot, integrate to get q ──────────
                manipulator.set_positions_user_order(plant, plant_ctx, q_seed)
                ee_now  = manipulator.get_end_effector_position(plant, plant_ctx)
                x_dot   = (target_xy - ee_now[:2]) / dt_traj   # position error → desired vel
                q_dot   = manipulator.compute_velocity_ik(
                    plant, plant_ctx, x_dot, damping=args.ik_damping
                )
                q_sol   = q_seed + q_dot * dt_traj
                # Re-apply and check residual
                manipulator.set_positions_user_order(plant, plant_ctx, q_sol)
                ee_check = manipulator.get_end_effector_position(plant, plant_ctx)
                ok = np.linalg.norm(target_xy - ee_check[:2]) < 0.05   # 50 mm loose check
            elif use_hybrid_ik:
                # ── Hybrid IK: u_dot = J_h⁺ x_dot, recover q_dot via A⁻¹ ──────
                manipulator.set_positions_user_order(plant, plant_ctx, q_seed)
                ee_now  = manipulator.get_end_effector_position(plant, plant_ctx)
                x_dot   = (target_xy - ee_now[:2]) / dt_traj
                u_dot   = manipulator.compute_velocity_ik_hybrid(
                    plant, plant_ctx, x_dot, damping=args.ik_damping
                )
                # u_dot = [q1_dot, l_G_dot]; recover q2_dot = l_G_dot / r_p
                r_p     = manipulator.PULLEY_RADIUS
                q_dot   = np.array([u_dot[0], u_dot[1] / r_p])
                q_sol   = q_seed + q_dot * dt_traj
                manipulator.set_positions_user_order(plant, plant_ctx, q_sol)
                ee_check = manipulator.get_end_effector_position(plant, plant_ctx)
                ok = np.linalg.norm(target_xy - ee_check[:2]) < 0.05
            else:
                # ── Analytical 2R IK ─────────────────────────────────────────
                q_sol, ok = manipulator.compute_ik_analytical(
                    plant, target_xy, q_seed,
                    pos_tol=5e-3,
                    target_z=z_traj,
                )

            if not ok:
                print(colored(f"  ⚠  IK failed at point {i} target=({ex:.3f}, {ey:.3f})", "yellow"))
                continue

            if abs(q_sol[1]) > Q2_DIAG_LIMIT:
                q2_warn_count += 1

            q1_arr[i], q2_arr[i] = q_sol[0], q_sol[1]
            q_seed = q_sol  # warm-start next point

            # Apply solution to plant
            manipulator.set_positions_user_order(plant, plant_ctx, {
                "link1_base":  q_sol[0],
                "link2_link1": q_sol[1],
            })
            plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
            diagram.ForcedPublish(context)

            # FK-based cable tangents
            manipulator.compute_tangents(plant, plant_ctx)

            # Cable lengths from drive-pulley exit → anchor on link2
            # Green: drive_pulley.B_R ("Drive exit B_R") → cable_end_l
            len_green[i] = manipulator.length_cable_route(
                rig.cable_green, plant, plant_ctx, "Drive exit B_R")
            # Red:   drive_pulley.B_L ("Drive exit B_L") → cable_end_r
            len_red[i]   = manipulator.length_cable_route(
                rig.cable_red,   plant, plant_ctx, "Drive exit B_L")

            ee_pos = manipulator.get_end_effector_position(plant, plant_ctx)
            ee_x_actual[i] = ee_pos[0]
            ee_y_actual[i] = ee_pos[1]   # world Y (horizontal)

        valid = ~np.isnan(q1_arr)
        n_valid_ik = int(np.sum(valid))
        print(colored(f"\n  {n_valid_ik}/{N} IK solutions found", "cyan"))
        if q2_warn_count > 0:
            print(colored(
                f"  ⚠  {q2_warn_count}/{n_valid_ik} points have |q2| > 20°"
                f"  (trajectory at r≈{np.hypot(cx, cy):.3f} m from base;"
                f" |q2|≤20° workspace requires r≥0.525 m)",
                "yellow"
            ))

        # ── Meshcat animation replay (loop until Ctrl-C or window closes) ────
        dt_anim   = 0.08  # seconds between frames
        n_anim    = int(np.sum(valid))
        print(colored(
            f"\n▶  Animating {n_anim} poses in Meshcat  (dt={dt_anim*1e3:.0f} ms/frame)."
            "  Close the Meshcat tab or press Ctrl-C to proceed to plots.",
            "cyan"
        ))
        q1_valid = q1_arr[valid]
        q2_valid = q2_arr[valid]
        try:
            while True:
                for j in range(n_anim):
                    manipulator.set_positions_user_order(plant, plant_ctx, {
                        "link1_base":  q1_valid[j],
                        "link2_link1": q2_valid[j],
                    })
                    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
                    diagram.ForcedPublish(context)
                    manipulator.compute_tangents(plant, plant_ctx)
                    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
                    time.sleep(dt_anim)
        except KeyboardInterrupt:
            pass
        # Leave robot at home after animation
        manipulator.set_positions_user_order(plant, plant_ctx, {"link1_base": 0.0, "link2_link1": 0.0})
        plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
        diagram.ForcedPublish(context)

        # ── Plots ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(
            f"EE Trajectory IK — {args.traj_shape.capitalize()}"
            f"  (cx={cx:.3f}, cy={cy:.3f}, R={R:.3f} m)",
            fontsize=12, fontweight='bold'
        )

        tp = traj_param[valid]

        # Joint angles
        ax0 = axes[0]
        ax0.plot(tp, np.rad2deg(q1_arr[valid]), 'b-o', ms=3, label='q1 (link1_base)')
        ax0.plot(tp, np.rad2deg(q2_arr[valid]), 'r-o', ms=3, label='q2 (link2_link1)')
        ax0.set_ylabel('Joint angle [deg]')
        ax0.legend(fontsize=9)
        ax0.grid(True, alpha=0.4)
        ax0.set_title('IK Joint Angles')

        # Absolute cable lengths
        ax1 = axes[1]
        ax1.plot(tp, len_green[valid] * 1e3, 'g-o', ms=3, label='Green cable  (Drive B_R → End_L)')
        ax1.plot(tp, len_red[valid]   * 1e3, 'r-o', ms=3, label='Red cable    (Drive B_L → End_R)')
        ax1.axhline(L0_green * 1e3, color='g', ls='--', alpha=0.5, label=f'Home green {L0_green*1e3:.1f} mm')
        ax1.axhline(L0_red   * 1e3, color='r', ls='--', alpha=0.5, label=f'Home red   {L0_red*1e3:.1f} mm')
        ax1.set_ylabel('Cable length [mm]')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4)
        ax1.set_title('Cable Lengths (sum of straight segments, drive-pulley exit → link2 anchor)')

        # Delta cable lengths (change from home)
        ax2 = axes[2]
        ax2.plot(tp, (len_green[valid] - L0_green) * 1e3, 'g-o', ms=3, label='ΔGreen')
        ax2.plot(tp, (len_red[valid]   - L0_red)   * 1e3, 'r-o', ms=3, label='ΔRed')
        ax2.axhline(0, color='k', lw=0.7)
        ax2.set_xlabel(traj_label)
        ax2.set_ylabel('ΔCable length [mm]')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4)
        ax2.set_title('Cable Length Change from Home')

        fig.tight_layout()

        # EE path figure
        fig2, ax_ee = plt.subplots(figsize=(6, 5))
        ax_ee.plot(ee_x_tgt, ee_y_tgt, 'k--', lw=0.8, label='Target path')
        ax_ee.plot(ee_x_actual[valid], ee_y_actual[valid], 'b-o', ms=3, label='IK solution')
        ax_ee.set_aspect('equal')
        ax_ee.set_xlabel('X [m]')
        ax_ee.set_ylabel('Y [m]  (world horizontal)')
        ax_ee.set_title(f'EE Trajectory (world XY plane,  Z = {z_traj*1e3:.1f} mm fixed)')
        ax_ee.legend(fontsize=9)
        ax_ee.grid(True, alpha=0.4)
        fig2.tight_layout()

        plt.show(block=True)
        return

    elif args.mode in _SIM_REGISTRY:
        # ====================================================================
        # SIMULATION-CLASS MODES  (ik-diagram, computed-torque, …)
        # Dispatch through _SIM_REGISTRY — all share the same call sequence.
        # ====================================================================
        cls, kwargs_fn, label = _SIM_REGISTRY[args.mode]

        print("\n" + "=" * 80)
        print(colored(label, "cyan", attrs=["bold"]))
        print("=" * 80)

        meshcat = StartMeshcat()
        print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        from utils.viz import set_meshcat_camera_spherical
        set_meshcat_camera_spherical(
            meshcat,
            azimuth_deg=args.meshcat_azimuth,
            elevation_deg=args.meshcat_elevation,
            distance=args.meshcat_distance,
            target=np.zeros(3),
        )

        sim = cls(CABLE_MANIPULATOR_CONFIG, meshcat=meshcat, **kwargs_fn(args))
        sim.build_plant()
        sim.build_trajectory(args)
        sim.build_move_to_start(args.move_duration)
        sim.build_controller()
        sim.connect_and_build()
        sim.initialize(home_override=_parse_home_args(args))
        sim.run(traj_shape=args.traj_shape)
        sim.plot(traj_shape=args.traj_shape)
        return

    else:
        print(colored(f"Unknown mode '{args.mode}' - no simulation run", "red"))
if __name__ == "__main__":
    main()
