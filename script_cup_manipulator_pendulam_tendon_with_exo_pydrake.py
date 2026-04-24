#!/usr/bin/env python3
"""
script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py
===========================================================
Series Elastic Actuator (SEA) cable simulation with exosuit co-contraction.

Extends the base SEA tendon simulation with two exosuit cable actuators
(Method B — centred elbow pulley).  The exosuit can be activated at a
configurable time to inject co-contraction stiffness at the elbow joint.

Diagram wiring::

    Trajectory ── CT ──→ SEA (drive) ──╮
                                       ├─ ActuationSum ──→ Plant
    ExoCmd ──→ SEA (exo) ─────────────╯
                                       └─→ plant_state (feedback)

USAGE
─────
  # Exo deactivated (default — motors track encoder, zero added stiffness)
  python script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py

  # Exo activated after 5 s with Δθ=0.1 rad (co-contraction)
  python script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py --exo-activate --exo-activate-time 5

  # Exo always active from t=0
  python script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py --exo-activate --exo-activate-time 0

  # Higher exo stiffness
  python script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py --exo-activate --exo-ks 500 --exo-delta-theta 0.15
"""

import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np

import matplotlib
import platform as _platform
if _platform.system() == 'Darwin':
    try:
        matplotlib.use('MacOSX')
    except Exception:
        matplotlib.use('TkAgg')
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    LogVectorOutput,
    LeafSystem,
    MeshcatVisualizer,
    StartMeshcat,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    Parser,
    BasicVector,
)
from pydrake.geometry import Rgba
from pydrake.multibody.tree import RevoluteSpring

sys.path.insert(0, str(Path(__file__).parent))
from robots.cup_manipulator_tendon_with_exo import (
    CupManipulatorTendonWithExo,
)
from robots.cup_manipulator_tendon import create_cable_manipulator_config
from controller.controller import ComputedTorqueController
from actuators.sea import SEACableActuator
from actuators.sea_exo import SEAExoActuator
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES
from project_utils.viz_cables import draw_cables
from controller.trajectory_drake import (
    build_trajectory,
    build_move_to_start,
    PreambleSrc,
)

# ─── Constants ────────────────────────────────────────────────────────────────
_DT   = 0.01   # plant & controller timestep [s]
_URDF = ("model_using_onshape_to_robot/"
         "manipulator_cable_exo_springs_elbow_follow/"
         "manipulator_cable_exo_springs_elbow_follow_obj.urdf")
_M_PATCH = SpatialInertia(
    mass=0.3, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)


# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="SEA cable simulation with exosuit co-contraction — "
                "joint 1 rigid, joint 2 drive cable + exo cables",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Drive motor
_mot = parser.add_argument_group("drive motor (elbow / joint 2)")
_mot.add_argument("--motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config",
                  help="CubeMars motor for the drive cable.")

# Drive SEA parameters
_sea = parser.add_argument_group("SEA drive cable (joint 2)")
_sea.add_argument("--sea-mode", choices=["torque", "position"], default="torque",
                  help="Motor dynamics mode for drive cable.")
_sea.add_argument("--spring-stiffness", type=float, default=10, metavar="K_S",
                  help="Drive cable spring stiffness k_s [N/m].")
_sea.add_argument("--cable-damping", type=float, default=2.0, metavar="B_C",
                  help="Drive cable dashpot damping b_c [N·s/m].")
_sea.add_argument("--motor-bandwidth", type=float, default=None, metavar="W_M",
                  help="Motor position servo bandwidth ω_m [rad/s] (position mode).")

# Exosuit parameters
_exo = parser.add_argument_group("exosuit co-contraction (Method B)")
_exo.add_argument("--exo-motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config",
                  help="CubeMars motor for both exo cables.")
_exo.add_argument("--exo-ks", type=float, default=500.0, metavar="K_EXO",
                  help="Exo cable spring stiffness [N/m].")
_exo.add_argument("--exo-bc", type=float, default=2.0, metavar="B_EXO",
                  help="Exo cable dashpot damping [N·s/m].")
_exo.add_argument("--exo-r", type=float, default=0.04775, metavar="R_EXO",
                  help="Exo elbow pulley radius [m].")
_exo.add_argument("--exo-delta-theta", type=float, default=0.1, metavar="DTHETA",
                  help="Co-contraction offset Δθ [rad] when activated.")
_exo.add_argument("--exo-activate-time", type=float, default=5.0, metavar="T_ACT",
                  help="Time [s] at which exo activates (for timed mode).")
_exo.add_argument("--no-exo-activate", action="store_true", default=True,
                  help="Keep exo deactivated for the entire simulation (passive). [default: True]")
_exo.add_argument("--exo-activate", dest="no_exo_activate", action="store_false",
                  help="Enable exo activation at --exo-activate-time (timed mode).")
_exo.add_argument("--exo-reactive", action="store_true", default=False,
                  help="Reactive mode: exo activates on tracking error, deactivates on recovery.")
_exo.add_argument("--exo-e-on", type=float, default=5.0, metavar="DEG",
                  help="Reactive mode: elbow error threshold to activate [deg]. Default: 5°")
_exo.add_argument("--exo-e-off", type=float, default=2.0, metavar="DEG",
                  help="Reactive mode: elbow error threshold to deactivate [deg]. Default: 2°")
_exo.add_argument("--exo-t-hold", type=float, default=0.5, metavar="T_HOLD",
                  help="Reactive mode: seconds error must stay below e_off before deactivating. Default: 0.5")

# CT gains
_ct = parser.add_argument_group("computed-torque gains")
_ct.add_argument("--ct-kp", type=float, nargs="+", default=[100.0],
                 metavar="KP",
                 help="CT position gain Kp [1/s²]. One value → both joints, "
                      "two values → [shoulder, elbow] separately.")
_ct.add_argument("--ct-kd", type=float, nargs="+", default=[40.0],
                 metavar="KD",
                 help="CT velocity gain Kd [1/s]. One value → both joints, "
                      "two values → [shoulder, elbow] separately.")
_ct.add_argument("--ct-tau-max", type=float, default=None,
                 help="Torque saturation [Nm]. Default: motor peak_torque_joint.")

# Simulation
_sim = parser.add_argument_group("simulation")
_sim.add_argument("--duration", type=float, default=10.0, help="Lap duration [s]")
_sim.add_argument("--num-laps", type=int, default=3, metavar="N",
                  help="Number of laps before auto-stopping (0 = infinite, Ctrl-C to stop).")
_sim.add_argument("--move-duration", type=float, default=3.0,
                  help="Move-to-start preamble [s]. 0 to disable.")
_sim.add_argument("--no-meshcat", action="store_true", help="Disable Meshcat 3-D visualisation")
_sim.add_argument("--no-show", action="store_true",
                  help="Do not block on plt.show() (for sweeps / headless runs).")

# Robot
_rob = parser.add_argument_group("robot mount")
_rob.add_argument("--tilt-roll", type=float, default=0.0)
_rob.add_argument("--tilt-pitch", type=float, default=0.0)
_rob.add_argument("--joint-damping", type=float, nargs=2, default=[0.05, 0.05],
                  metavar=("D1", "D2"))
_rob.add_argument("--joint-stiffness", type=float, nargs=2, default=[0.0, 0.0],
                  metavar=("K1", "K2"))

# Disturbance (collision simulation)
_dist = parser.add_argument_group("disturbance (collision)")
_dist.add_argument("--disturbance", action="store_true", default=False,
                   help="Inject a disturbance to simulate collision.")
_dist.add_argument("--disturbance-time", type=float, default=7.0, metavar="T_DIST",
                   help="Time [s] at which the disturbance is applied.")
_dist.add_argument("--disturbance-mode", choices=["vel", "pos", "torque", "sine"], default="vel",
                   help="Disturbance type: 'vel'=velocity impulse Δq̇₂ (brief kick), "
                        "'pos'=position jump Δq₂, "
                        "'torque'=sustained external torque (push-and-hold, best for showing exo stiffness), "
                        "'sine'=sinusoidal external torque (high-frequency, shows passive bandwidth).")
_dist.add_argument("--disturbance-dqdot", type=float, default=60.0, metavar="DQDOT_DEGS",
                   help="Velocity impulse Δq̇₂ [deg/s]. Used when --disturbance-mode=vel.")
_dist.add_argument("--disturbance-dq", type=float, default=15.0, metavar="DQ_DEG",
                   help="Position jump Δq₂ [deg]. Used when --disturbance-mode=pos.")
_dist.add_argument("--disturbance-tau", type=float, default=2.0, metavar="TAU_EXT",
                   help="External torque amplitude [Nm] on joint 2. Used for --disturbance-mode=torque|sine.")
_dist.add_argument("--disturbance-dur", type=float, default=1.5, metavar="T_DUR",
                   help="Duration [s] the external torque is applied (torque/sine modes). "
                        "Overridden by --disturbance-cycles when both are supplied.")
_dist.add_argument("--disturbance-freq", type=float, default=3.0, metavar="F_HZ",
                   help="Sinusoid frequency [Hz] for --disturbance-mode=sine.")
_dist.add_argument("--disturbance-cycles", type=float, default=1, metavar="N_CYCLES",
                   help="Number of sine cycles to apply (sine mode). "
                        "Sets disturbance-dur = N_CYCLES / disturbance-freq, "
                        "overriding --disturbance-dur.")

# Trajectory
_traj = parser.add_argument_group("trajectory")
_traj.add_argument("--traj-type", choices=["rect", "circle", "figure8", "line"],
                   default="rect",
                   help="EE trajectory shape. rect=rectangle (default), "
                        "circle=circular path, figure8=lemniscate, "
                        "line=back-and-forth along y.")
_traj.add_argument("--traj-x-range", type=float, nargs=2, default=[0.49, 0.51],
                   metavar=("X_MIN", "X_MAX"))
_traj.add_argument("--traj-y-range", type=float, nargs=2, default=[-0.08, 0.08],
                   metavar=("Y_MIN", "Y_MAX"))
_traj.add_argument("--traj-radius", type=float, default=None,
                   help="Radius [m] for circle/figure8 (overrides default from ranges).")
_traj.add_argument("--traj-n", type=int, default=60)
_traj.add_argument("--traj-v-max", type=float, default=0.9)
_traj.add_argument("--traj-v-corner", type=float, default=0.05)
_traj.add_argument("--traj-corner-blend", type=float, default=0.35)

args = parser.parse_args()

# --disturbance-cycles overrides --disturbance-dur for sine mode
if args.disturbance_cycles is not None:
    args.disturbance_dur = args.disturbance_cycles / args.disturbance_freq

# ─── Expand per-joint CT gains ──────────────────────────────────────
# Accept 1 value (broadcast) or 2 values [shoulder, elbow]
if len(args.ct_kp) == 1:
    args.ct_kp = np.array([args.ct_kp[0], args.ct_kp[0]])
elif len(args.ct_kp) == 2:
    args.ct_kp = np.array(args.ct_kp)
else:
    parser.error("--ct-kp expects 1 or 2 values")

if len(args.ct_kd) == 1:
    args.ct_kd = np.array([args.ct_kd[0], args.ct_kd[0]])
elif len(args.ct_kd) == 2:
    args.ct_kd = np.array(args.ct_kd)
else:
    parser.error("--ct-kd expects 1 or 2 values")

# ─── Motor-derived defaults ─────────────────────────────────────────
_drive_motor = get_motor(args.motor)
_exo_motor   = get_motor(args.exo_motor)
_motor_mode  = MotorMode(args.sea_mode)
if args.motor_bandwidth is None:
    args.motor_bandwidth = 100.0  # conservative closed-loop estimate
if args.ct_tau_max is None:
    args.ct_tau_max = _drive_motor.peak_torque_joint

_mode_label = "torque (2nd-order rotor)" if _motor_mode == MotorMode.TORQUE else "position (1st-order servo)"
print(colored(
    f"\n  Drive motor: {args.motor}  —  SEA mode: {_mode_label}"
    f"\n    gear ratio      = {_drive_motor.gear_ratio}"
    f"\n    peak torque     = {_drive_motor.peak_torque_joint} Nm"
    f"\n  Exo motor: {args.exo_motor}"
    f"\n    gear ratio      = {_exo_motor.gear_ratio}"
    f"\n    peak torque     = {_exo_motor.peak_torque_joint} Nm"
    f"\n  Exo: k_exo={args.exo_ks} N/m  r_exo={args.exo_r:.5f} m"
    f"  Δθ={args.exo_delta_theta:.3f} rad"
    f"\n  k_eff = 2·k_exo·r² = {2*args.exo_ks*args.exo_r**2:.4f} Nm/rad",
    "yellow",
))


# ════════════════════════════════════════════════════════════════════════════
# Helper LeafSystems
# ════════════════════════════════════════════════════════════════════════════

# ─── _ExoCommandSource ──────────────────────────────────────────────────────
# PURPOSE:
# Turns ON/OFF based on: Fixed time (--exo-activate-time)
# Ignores: Trajectory tracking performance
# Behavior: At t=4s → ON, stays ON regardless of tracking quality
# Use case: Simple A/B baseline comparisons
#   Time-triggered exosuit command source.  Outputs a 2-element vector
#   [activated, Δθ] that controls when the exosuit switches from passive
#   (motors track encoder) to active co-contraction.
#
# USED IN CURRENT IMPLEMENTATION: YES — when --exo-reactive is NOT set
#   (i.e. the default timed-activation mode).  Instantiated as 'ExoCmd'.
#   NOT used when --exo-reactive is active (replaced by _ExoReactiveSource).
#
# SIGNAL FLOW:
#   _ExoCommandSource → SEAExoActuator.activate_cmd
#
# BEHAVIOUR:
#   • t < t_activate  (or never_activate=True):
#       output = [0.0, 0.0]   → exo motors track encoder, zero added torque.
#   • t ≥ t_activate:
#       output = [1.0, Δθ]    → co-contraction active with the given offset.
#   Setting never_activate=True (default when --no-exo-activate is used)
#   keeps the exo permanently off, which is the baseline comparison run.
#
# OUTPUT PORT:
#   'cmd'  [2]  [activated (0/1), Δθ (rad)]
class _ExoCommandSource(LeafSystem):
    """Outputs [activated, Δθ] based on simulation time.

    Before ``t_activate``: outputs [0.0, 0.0] (deactivated, motors track encoder).
    After  ``t_activate``: outputs [1.0, delta_theta] (co-contraction active).
    """
    def __init__(self, t_activate: float, delta_theta: float, never_activate: bool = False):
        super().__init__()
        self._t_act     = float(t_activate)
        self._dtheta    = float(delta_theta)
        self._never     = never_activate
        self.DeclareVectorOutputPort("cmd", 2, self._calc)

    def _calc(self, ctx, out):
        if self._never or ctx.get_time() < self._t_act:
            out.SetFromVector(np.array([0.0, 0.0]))
        else:
            out.SetFromVector(np.array([1.0, self._dtheta]))


# ─── _ExoReactiveSource ─────────────────────────────────────────────────────
# PURPOSE:
# Turns ON/OFF based on: Trajectory tracking error on joint 2 (elbow)
# Thresholds:
# Activate when error ≥ e_on (5° default)
# Deactivate when error < e_off (2° default) for ≥ t_hold seconds
# Monitors: |q₂_actual − q₂_desired| in real time
# Use case: Realistic "smart assist"—only helps when the robot struggles to track
#   Error-triggered exosuit command source with hysteresis and hold-time
#   de-bounce.  Monitors the elbow (joint 2) tracking error in real time and
#   automatically switches the exosuit between passive and active co-contraction
#   depending on whether the error exceeds configurable thresholds.
#
# USED IN CURRENT IMPLEMENTATION: CONDITIONAL — only instantiated when
#   --exo-reactive is passed on the command line.  In the default run
#   (timed mode) this class is NOT used; _ExoCommandSource is used instead.
#   When active, named 'ExoCmd_reactive' in the diagram.
#
# SIGNAL FLOW:
#   plant.state_output  ──→ _ExoReactiveSource.plant_state
#   CT.joint_positions  ──→ _ExoReactiveSource.q_des
#   _ExoReactiveSource  ──→ SEAExoActuator.activate_cmd
#
# BEHAVIOUR:
#   Uses a periodic discrete update (every 2 ms) to check elbow error:
#   • Inactive  → activates  when |q₂ − q₂_des| ≥ e_on.
#   • Active    → deactivates when |q₂ − q₂_des| < e_off for ≥ t_hold s.
#   This hysteresis prevents rapid chattering around the threshold.
#
# DISCRETE STATE:
#   [is_active (0.0 or 1.0),  t_below_threshold (s)]
#
# INPUT PORTS:
#   'plant_state'  [n_multibody_states]  full plant state from MultibodyPlant
#   'q_des'        [2]                   desired joint angles [q₁, q₂]
# OUTPUT PORT:
#   'cmd'          [2]                   [activated (0/1), Δθ (rad)]
class _ExoReactiveSource(LeafSystem):
    """Error-triggered exo activation with hysteresis.

    Normally transparent (motors track encoder, τ_exo = 0).  When the elbow
    tracking error |q₂ − q₂_des| exceeds ``e_on``, activates co-contraction
    with the specified Δθ.  Deactivates when error falls below ``e_off`` for
    at least ``t_hold`` seconds.

    Input ports
    -----------
    ``plant_state``     [n]     full plant state vector
    ``q_des``           [2]     desired joint positions [q₁, q₂] (user order)

    Output port
    -----------
    ``cmd``             [2]     [activated, Δθ]
    """
    def __init__(
        self,
        plant: "MultibodyPlant",
        manipulator,
        delta_theta: float,
        e_on:   float = np.deg2rad(5.0),
        e_off:  float = np.deg2rad(2.0),
        t_hold: float = 0.5,
    ):
        super().__init__()
        self._dtheta = float(delta_theta)
        self._e_on   = float(e_on)
        self._e_off  = float(e_off)
        self._t_hold = float(t_hold)

        # q₂ index in Drake's position vector
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._q2_idx = j2.velocity_start()   # same for 1-DOF revolute
        self._nq     = plant.num_positions()

        # Discrete state: [is_active (0/1), t_below_threshold]
        self._state_idx = self.DeclareDiscreteState(2)
        self.DeclarePeriodicDiscreteUpdateEvent(0.002, 0.0, self._update)

        nstate = plant.num_multibody_states()
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)
        self._qdes_port  = self.DeclareVectorInputPort("q_des", 2)
        self.DeclareVectorOutputPort("cmd", 2, self._calc)

    def _update(self, context, discrete_state):
        ds = context.get_discrete_state(self._state_idx).value()
        is_active   = ds[0] > 0.5
        t_below     = ds[1]

        state = self._state_port.Eval(context)
        q_des = self._qdes_port.Eval(context)
        q2_actual = state[self._q2_idx]
        q2_des    = q_des[1]     # user order: [q₁, q₂]
        err = abs(q2_actual - q2_des)

        dt = 0.002
        if not is_active:
            if err >= self._e_on:
                is_active = True
                t_below   = 0.0
        else:
            if err < self._e_off:
                t_below += dt
                if t_below >= self._t_hold:
                    is_active = False
                    t_below   = 0.0
            else:
                t_below = 0.0

        discrete_state.get_mutable_vector(self._state_idx).SetFromVector(
            np.array([1.0 if is_active else 0.0, t_below])
        )

    def _calc(self, ctx, out):
        ds = ctx.get_discrete_state(self._state_idx).value()
        if ds[0] > 0.5:
            out.SetFromVector(np.array([1.0, self._dtheta]))
        else:
            out.SetFromVector(np.array([0.0, 0.0]))


# ─── _ActuationSum ───────────────────────────────────────────────────────────
# PURPOSE:
#   Final torque mixer before the plant.  Combines three independent torque
#   contributions on joint 2 (the elbow) into a single 2-DOF actuation vector
#   that is fed to MultibodyPlant.get_actuation_input_port().
#
# USED IN CURRENT IMPLEMENTATION: YES — always present in the diagram.
#   Named 'ActuationSum'.
#
# SIGNAL FLOW:
#   SEACableActuator.actuation  ──→ ActuationSum.drive_actuation  [2]
#   SEAExoActuator.exo_torque   ──→ ActuationSum.exo_torque       [1]
#   _ExternalTorqueSource.tau   ──→ ActuationSum.ext_torque        [1]
#                                          ↓
#                               ActuationSum.actuation  [2]  ──→ plant
#
# OUTPUT FORMULA:
#   τ₁_out = τ₁_drive          (shoulder — only drive cable, no exo on jt 1)
#   τ₂_out = τ₂_drive + τ_exo + τ_ext
#
# INPUT PORTS:
#   'drive_actuation'  [2]  from SEACableActuator (shoulder + elbow drive)
#   'exo_torque'       [1]  from SEAExoActuator  (co-contraction on elbow)
#   'ext_torque'       [1]  from _ExternalTorqueSource (disturbance on elbow)
# OUTPUT PORT:
#   'actuation'        [2]  combined torque for both joints → plant
class _ActuationSum(LeafSystem):
    """Sum drive SEA output + exo output + external disturbance torque.

    Input ports:
        ``drive_actuation``  [2]  from SEACableActuator
        ``exo_torque``       [1]  from SEAExoActuator
        ``ext_torque``       [1]  external disturbance torque on joint 2

    Output ports:
        ``actuation``        [2]  to plant actuation input
        → [τ₁, τ₂_drive + τ_exo + τ_ext]
    """
    def __init__(self):
        super().__init__()
        self._drive_port = self.DeclareVectorInputPort("drive_actuation", 2)
        self._exo_port   = self.DeclareVectorInputPort("exo_torque", 1)
        self._ext_port   = self.DeclareVectorInputPort("ext_torque", 1)
        self.DeclareVectorOutputPort("actuation", 2, self._calc)

    def _calc(self, ctx, out):
        drive = self._drive_port.Eval(ctx)
        exo   = self._exo_port.Eval(ctx)
        ext   = self._ext_port.Eval(ctx)
        out.SetFromVector(np.array([drive[0], drive[1] + exo[0] + ext[0]]))


# ─── _ExternalTorqueSource ───────────────────────────────────────────────────
# PURPOSE:
#   Simulates an external physical disturbance applied to joint 2 (the elbow)
#   during a defined time window.  Models scenarios such as someone pushing
#   the forearm, a payload collision, or a sustained external load.  This is
#   the primary mechanism for comparing exo co-contraction stiffness against
#   the baseline CT-only controller.
#
# USED IN CURRENT IMPLEMENTATION: ALWAYS instantiated, but only ACTIVE when:
#   --disturbance is set  AND  --disturbance-mode is 'torque' or 'sine'.
#   Named 'ExtTorque' in the diagram.
#
#   For 'vel' and 'pos' disturbance modes the output is permanently 0 because
#   those disturbances are applied by directly mutating the plant state in
#   the simulation run-loop (not through a torque port).  The system is still
#   wired into ActuationSum but contributes nothing in those modes.
#
# SIGNAL FLOW:
#   _ExternalTorqueSource.tau  ──→  _ActuationSum.ext_torque  [1]
#
# SUPPORTED SHAPES:
#   'torque':  constant τ_ext for t ∈ [t_start, t_start + duration].
#              Best for measuring steady-state deflection: Δq = τ / k_eff.
#   'sine':    τ_ext · sin(2π f t) for the same window.
#              Best for measuring passive bandwidth above CT cutoff.
#   (If enabled=False or shape is neither, output is always 0.)
#
# OUTPUT PORT:
#   'tau'  [1]  external torque on joint 2 [Nm]
class _ExternalTorqueSource(LeafSystem):
    """Time-windowed external torque applied to joint 2 (the elbow).

    Represents an external disturbance (e.g. someone pushing the forearm,
    a payload bumping the arm).  Supports two shapes:

    * ``torque``: constant amplitude τ_ext for t ∈ [t_start, t_start+dur].
      Best for showing co-contraction stiffness — steady-state deflection
      under load is τ/(k_CT_eff + k_exo_eff).
    * ``sine``:   τ_ext · sin(2π f t) for t ∈ [t_start, t_start+dur].
      Best for showing passive bandwidth advantage above CT cutoff.

    If ``enabled=False`` the port outputs 0 at all times.
    """
    def __init__(
        self,
        enabled:   bool,
        shape:     str,         # 'torque' or 'sine'
        amplitude: float,       # [Nm]
        t_start:   float,       # [s]
        duration:  float,       # [s]
        freq_hz:   float = 0.0, # used for 'sine'
    ):
        super().__init__()
        self._on     = bool(enabled) and shape in ("torque", "sine")
        self._shape  = shape
        self._amp    = float(amplitude)
        self._t0     = float(t_start)
        self._t1     = float(t_start) + float(duration)
        self._omega  = 2.0 * np.pi * float(freq_hz)
        self.DeclareVectorOutputPort("tau", 1, self._calc)

    def _calc(self, ctx, out):
        t = ctx.get_time()
        if (not self._on) or (t < self._t0) or (t >= self._t1):
            out.SetFromVector(np.array([0.0]))
            return
        if self._shape == "torque":
            out.SetFromVector(np.array([self._amp]))
        else:  # sine
            out.SetFromVector(np.array([
                self._amp * np.sin(self._omega * (t - self._t0))
            ]))


# ════════════════════════════════════════════════════════════════════════════
# Single simulation run
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(meshcat) -> dict:
    """Build, run, and collect logs for SEA + exo simulation."""
    print(colored(f"\n{'=' * 68}", "cyan"))
    print(colored(f"  SEA + Exo Simulation", "cyan"))
    print(colored(f"  Drive: {args.motor}  k_s={args.spring_stiffness} N/m  mode={args.sea_mode}", "cyan"))
    print(colored(f"  Exo:   {args.exo_motor}  k_exo={args.exo_ks} N/m  Δθ={args.exo_delta_theta} rad"
                  f"  activate@{args.exo_activate_time}s", "cyan"))
    print(colored(f"{'=' * 68}", "cyan"))

    # ── 1. Config ────────────────────────────────────────────────────────────
    manip_config = create_cable_manipulator_config(
        urdf_path=_URDF,
        joint_angles={
            "link1_base":   np.deg2rad(5.0),
            "link2_link1":  np.deg2rad(15.0),
        },
        damping=tuple(args.joint_damping),
        stiffness=tuple(args.joint_stiffness),
        tilt_roll_deg=args.tilt_roll,
        tilt_pitch_deg=args.tilt_pitch,
    )

    # ── 2. Plant ─────────────────────────────────────────────────────────────
    builder     = DiagramBuilder()
    plant       = MultibodyPlant(time_step=_DT)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulatorTendonWithExo(manip_config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    orientation = np.deg2rad([
        manip_config.tilt_roll_deg, manip_config.tilt_pitch_deg, 0.0,
    ])
    manipulator.weld_base_to_world(plant, position=np.zeros(3), orientation=orientation)
    manipulator.add_joint_actuators(plant)
    manipulator.set_joint_properties(plant)

    for jt_name in ["link1_base", "link2_link1"]:
        cfg = manip_config.joint_configs.get(jt_name)
        if cfg and cfg.stiffness and cfg.stiffness > 0.0:
            jt = manipulator.get_joint_by_name(plant, jt_name)
            plant.AddForceElement(
                RevoluteSpring(jt, nominal_angle=0.0, stiffness=cfg.stiffness),
            )

    manipulator.add_end_effector_frame(plant)
    plant.Finalize()
    builder.AddSystem(plant)

    # Geometry wiring
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )
    _visualizer = None
    if meshcat is not None:
        _visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # ── 3. Trajectory ────────────────────────────────────────────────────────
    traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt = \
        build_trajectory(manipulator, plant, args)

    if args.move_duration > 0.0:
        move_traj, move_vel, move_acc, q_init = \
            build_move_to_start(manipulator, plant, traj, traj_vel, args.move_duration)
    else:
        move_traj = move_vel = move_acc = None
        L1, L2 = manipulator.ik.get_link_lengths(plant)
        p0 = traj.value(0.0).ravel()
        q_init, _ = manipulator.ik._solve_2r_core(
            L1, L2, p0, np.array([np.deg2rad(5.0), np.deg2rad(15.0)]),
        )

    # ── 4. Controller + Actuators  (CT → SEA_drive + SEA_exo → Plant) ────────
    ct = builder.AddSystem(
        ComputedTorqueController(
            plant, manipulator,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        ),
    )
    ct.set_name("CT_ctrl")

    sea = builder.AddSystem(
        SEACableActuator(
            plant, manipulator,
            k_s=args.spring_stiffness, b_c=args.cable_damping,
            tau_max=args.ct_tau_max, dt=_DT,
            motor_mode=_motor_mode,
            motor_cfg=_drive_motor,
            omega_m=args.motor_bandwidth,
        ),
    )
    sea.set_name("SEA_drive")

    exo = builder.AddSystem(
        SEAExoActuator(
            plant, manipulator,
            k_exo=args.exo_ks, b_exo=args.exo_bc, r_exo=args.exo_r,
            tau_max=_exo_motor.peak_torque_joint, dt=_DT,
            motor_cfg=_exo_motor,
        ),
    )
    exo.set_name("SEA_exo")

    act_sum = builder.AddSystem(_ActuationSum())
    act_sum.set_name("ActuationSum")

    # External disturbance torque source (push-and-hold or sinusoid on joint 2).
    # For 'torque'/'sine' modes, this system applies the load itself.
    # For 'vel'/'pos' modes, state is kicked in the run loop and this source
    # is silent (outputs 0).
    _ext_on    = args.disturbance and args.disturbance_mode in ("torque", "sine")
    ext_tau_src = builder.AddSystem(
        _ExternalTorqueSource(
            enabled=_ext_on,
            shape=args.disturbance_mode if _ext_on else "torque",
            amplitude=args.disturbance_tau,
            t_start=args.disturbance_time,
            duration=args.disturbance_dur,
            freq_hz=args.disturbance_freq,
        ),
    )
    ext_tau_src.set_name("ExtTorque")

    # Exo command source — reactive (error-triggered) or timed
    if args.exo_reactive:
        exo_cmd = builder.AddSystem(
            _ExoReactiveSource(
                plant, manipulator,
                delta_theta=args.exo_delta_theta,
                e_on=np.deg2rad(args.exo_e_on),
                e_off=np.deg2rad(args.exo_e_off),
                t_hold=args.exo_t_hold,
            ),
        )
        exo_cmd.set_name("ExoCmd_reactive")
    else:
        exo_cmd = builder.AddSystem(
            _ExoCommandSource(
                t_activate=args.exo_activate_time,
                delta_theta=args.exo_delta_theta,
                never_activate=args.no_exo_activate,
            ),
        )
        exo_cmd.set_name("ExoCmd")

    # Trajectory sources
    ee_src  = builder.AddSystem(PreambleSrc(move_traj, args.move_duration, traj,     args.duration))
    vel_src = builder.AddSystem(PreambleSrc(move_vel,  args.move_duration, traj_vel, args.duration))
    acc_src = builder.AddSystem(PreambleSrc(move_acc,  args.move_duration, traj_acc, args.duration))
    ee_src.set_name("EE_ref"); vel_src.set_name("Vel_ref"); acc_src.set_name("Acc_ref")

    # ── 5. Wire ──────────────────────────────────────────────────────────────
    #  Trajectory → CT
    builder.Connect(ee_src.get_output_port(),       ct.GetInputPort("desired_ee_pos"))
    builder.Connect(vel_src.get_output_port(),      ct.GetInputPort("ee_vel_ref"))
    builder.Connect(acc_src.get_output_port(),      ct.GetInputPort("ee_acc_ref"))
    builder.Connect(plant.get_state_output_port(),  ct.GetInputPort("plant_state"))

    #  CT → Drive SEA
    builder.Connect(ct.GetOutputPort("actuation"),      sea.GetInputPort("tau_desired"))
    builder.Connect(plant.get_state_output_port(),      sea.GetInputPort("plant_state"))

    #  ExoCmd → Exo SEA
    builder.Connect(exo_cmd.get_output_port(),          exo.GetInputPort("activate_cmd"))
    builder.Connect(plant.get_state_output_port(),      exo.GetInputPort("plant_state"))
    #  CT reference trajectory → Exo (so exo provides stiffness ABOUT q_des,
    #  not a frozen anchor — otherwise exo fights intentional trajectory motion).
    builder.Connect(ct.GetOutputPort("joint_positions"),exo.GetInputPort("q_des"))
    #  Reactive exo needs plant state + desired q from CT
    if args.exo_reactive:
        builder.Connect(plant.get_state_output_port(),      exo_cmd.GetInputPort("plant_state"))
        builder.Connect(ct.GetOutputPort("joint_positions"),exo_cmd.GetInputPort("q_des"))

    #  Drive SEA + Exo + External disturbance → ActuationSum → Plant
    builder.Connect(sea.GetOutputPort("actuation"),     act_sum.GetInputPort("drive_actuation"))
    builder.Connect(exo.GetOutputPort("exo_torque"),    act_sum.GetInputPort("exo_torque"))
    builder.Connect(ext_tau_src.get_output_port(),      act_sum.GetInputPort("ext_torque"))
    builder.Connect(act_sum.GetOutputPort("actuation"), plant.get_actuation_input_port())

    # ── 6. Loggers ───────────────────────────────────────────────────────────
    log_state    = LogVectorOutput(plant.get_state_output_port(),      builder)
    log_act      = LogVectorOutput(act_sum.GetOutputPort("actuation"), builder)
    log_sea_diag = LogVectorOutput(sea.GetOutputPort("diagnostics"),   builder)
    log_exo_diag = LogVectorOutput(exo.GetOutputPort("diagnostics"),   builder)
    log_exo_tau  = LogVectorOutput(exo.GetOutputPort("exo_torque"),    builder)
    log_qdes     = LogVectorOutput(ct.GetOutputPort("joint_positions"),builder)
    log_ref      = LogVectorOutput(ee_src.get_output_port(),           builder)

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()

    # ── 7. Initialize ────────────────────────────────────────────────────────
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Patch zero-mass Onshape bodies
    patched = []
    for idx in plant.GetBodyIndices(manipulator.model_instance):
        body = plant.get_body(idx)
        if body.default_mass() < 1e-6:
            body.SetSpatialInertiaInBodyFrame(plant_ctx, _M_PATCH)
            patched.append(body.name())
    if patched:
        print(colored(f"  ✓ Patched zero-mass bodies: {patched}", "yellow"))

    manipulator.set_positions_user_order(plant, plant_ctx, q_init)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    # Initialize drive SEA spring at rest
    sea_ctx = sea.GetMyMutableContextFromRoot(sim_ctx)
    sea.initialize_spring_at_rest(sea_ctx, q_init[1])

    # Initialize exo motors at rest (δ_R = δ_L = 0)
    exo_ctx = exo.GetMyMutableContextFromRoot(sim_ctx)
    exo.initialize_at_rest(exo_ctx, q_init[1])

    ee0 = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored(
        f"  ✓ Init: q=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE=({ee0[0]*1e3:.1f}, {ee0[1]*1e3:.1f}) mm",
        "green",
    ))
    print(colored(
        f"  ✓ k_eff (exo co-contraction) = {exo.k_eff:.4f} Nm/rad",
        "green",
    ))

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    if _visualizer is not None:
        _visualizer.StartRecording()

    # ── 8. Cable visualisation ───────────────────────────────────────────────
    try:
        manipulator.init_cable_rig(_URDF, springs_enabled=True)
        _rig = manipulator.rig
    except Exception:
        _rig = None

    try:
        manipulator.init_exo_cable_rig(_URDF, springs_enabled=True)
        _exo_rig = manipulator.exo_rig
    except Exception:
        _exo_rig = None

    _ee_trail: list = []   # accumulates world-frame EE positions (3,)
    _TRAIL_PATH  = "/ee_trail"
    _TRAIL_COLOR = Rgba(0.0, 0.9, 1.0, 0.85)   # cyan
    _TRAIL_WIDTH = 2.5   # px

    def _viz_tick():
        if meshcat is None:
            return
        _ctx = simulator.get_mutable_context()
        _pc  = plant.GetMyMutableContextFromRoot(_ctx)
        # Drive cables
        if _rig is not None:
            _sea_ctx = sea.GetMyMutableContextFromRoot(_ctx)
            _diag = sea.GetOutputPort("diagnostics").Eval(_sea_ctx)
            _delta = _diag[2]
            manipulator.compute_tangents(plant, _pc)
            draw_cables(meshcat, plant, _pc, manipulator, _rig,
                        spring_extension=_delta)
        # Exo cables (reuse draw_cables with exo_rig)
        if _exo_rig is not None:
            manipulator.compute_exo_tangents(plant, _pc)
            draw_cables(meshcat, plant, _pc, manipulator, _exo_rig,
                        spring_extension=0.0)
        # EE trail — append current EE world position and redraw polyline
        _ee_pos = manipulator.get_end_effector_position(plant, _pc)
        _ee_trail.append(_ee_pos.copy())
        if len(_ee_trail) >= 2:
            _pts = np.column_stack(_ee_trail)   # (3, N)
            meshcat.SetLine(_TRAIL_PATH, _pts, _TRAIL_WIDTH, _TRAIL_COLOR)

    # ── 9. Run ───────────────────────────────────────────────────────────────
    wn   = np.sqrt(args.ct_kp)        # per-joint [shoulder, elbow]
    zeta = np.where(wn > 0, args.ct_kd / (2.0 * wn), 0.0)
    _kp_str = (f"{args.ct_kp[0]:.1f}" if args.ct_kp[0] == args.ct_kp[1]
               else f"[{args.ct_kp[0]:.1f}, {args.ct_kp[1]:.1f}]")
    _kd_str = (f"{args.ct_kd[0]:.1f}" if args.ct_kd[0] == args.ct_kd[1]
               else f"[{args.ct_kd[0]:.1f}, {args.ct_kd[1]:.1f}]")
    if args.exo_reactive:
        _exo_mode_str = (f"  REACTIVE (e_on={args.exo_e_on:.1f}° e_off={args.exo_e_off:.1f}°"
                         f" t_hold={args.exo_t_hold:.1f}s)")
    elif args.no_exo_activate:
        _exo_mode_str = "  OFF (transparent)"
    else:
        _exo_mode_str = f"  activate@{args.exo_activate_time:.1f}s"
    print(colored(
        f"\n▶  SEA + Exo Cable  "
        f"\n   Drive: k_s={args.spring_stiffness} N/m  b_c={args.cable_damping} N·s/m"
        f"\n   Exo:   k_exo={args.exo_ks} N/m  Δθ={args.exo_delta_theta:.3f} rad"
        f"{_exo_mode_str}"
        f"\n   CT:    Kp={_kp_str}  Kd={_kd_str}  ωn=[{wn[0]:.1f},{wn[1]:.1f}]  ζ=[{zeta[0]:.2f},{zeta[1]:.2f}]"
        f"\n   Looping — lap={args.duration:.1f} s  (Ctrl-C to stop & plot)",
        "cyan",
    ))

    _chunk         = 0.1
    _lap_prev      = 0
    _move_reported = args.move_duration <= 0.0
    _exo_reported  = args.no_exo_activate or args.exo_reactive  # reactive prints its own msgs
    _exo_was_active = False  # for reactive mode logging
    _dist_applied  = not args.disturbance       # skip if not requested
    # torque/sine disturbances are applied by _ExternalTorqueSource; don't
    # inject state impulses in the run loop for those modes.
    if args.disturbance and args.disturbance_mode in ("torque", "sine"):
        _dist_applied = True
        _ext_label = (f"τ_ext = {args.disturbance_tau:+.2f} Nm (const)"
                      if args.disturbance_mode == "torque"
                      else f"τ_ext = {args.disturbance_tau:.2f}·sin(2π·{args.disturbance_freq:.1f}Hz·t) Nm")
        print(colored(
            f"  💥 External-torque disturbance armed: {_ext_label}  "
            f"window=[{args.disturbance_time:.2f}, {args.disturbance_time + args.disturbance_dur:.2f}] s",
            "red",
        ))
    _max_laps      = args.num_laps          # 0 = infinite
    try:
        while True:
            t_now = sim_ctx.get_time()
            if not _move_reported and t_now >= args.move_duration:
                _move_reported = True
                print(colored(
                    f"  ✓ Move-to-start complete at t={t_now:.2f} s",
                    "green",
                ))
            if not _exo_reported and t_now >= args.exo_activate_time:
                _exo_reported = True
                print(colored(
                    f"  ⚡ Exo ACTIVATED at t={t_now:.2f} s  "
                    f"(Δθ={args.exo_delta_theta:.3f} rad, "
                    f"k_eff={exo.k_eff:.4f} Nm/rad)",
                    "magenta",
                ))
            # Reactive mode: log activation/deactivation transitions
            if args.exo_reactive:
                _exo_ctx = exo_cmd.GetMyContextFromRoot(sim_ctx)
                _ds = _exo_ctx.get_discrete_state(exo_cmd._state_idx).value()
                _active_now = _ds[0] > 0.5
                if _active_now and not _exo_was_active:
                    print(colored(
                        f"  ⚡ Exo ACTIVATED (reactive) at t={t_now:.2f} s  "
                        f"(Δθ={args.exo_delta_theta:.3f} rad, "
                        f"k_eff={exo.k_eff:.4f} Nm/rad)",
                        "magenta",
                    ))
                elif not _active_now and _exo_was_active:
                    print(colored(
                        f"  💤 Exo DEACTIVATED (reactive) at t={t_now:.2f} s  "
                        f"— back to transparent",
                        "magenta",
                    ))
                _exo_was_active = _active_now
            # ── Disturbance injection ──────────────────────────────────
            if not _dist_applied and t_now >= args.disturbance_time:
                _dist_applied = True
                _pc = plant.GetMyMutableContextFromRoot(sim_ctx)
                if args.disturbance_mode == "vel":
                    _dqdot_rad = np.deg2rad(args.disturbance_dqdot)
                    _v_before = manipulator.get_velocities_user_order(plant, _pc)
                    _v_after = _v_before.copy()
                    _v_after[1] += _dqdot_rad
                    manipulator.set_velocities_user_order(plant, _pc, _v_after)
                    print(colored(
                        f"  💥 DISTURBANCE at t={t_now:.2f} s  "
                        f"q̇₂: {np.rad2deg(_v_before[1]):.1f} → "
                        f"{np.rad2deg(_v_after[1]):.1f} °/s "
                        f"(Δq̇₂ = +{args.disturbance_dqdot:.1f}°/s) [vel impulse]",
                        "red",
                    ))
                else:
                    _dq_rad = np.deg2rad(args.disturbance_dq)
                    _q_before = manipulator.get_positions_user_order(plant, _pc)
                    _q_after = _q_before.copy()
                    _q_after[1] += _dq_rad        # q₂ jump
                    manipulator.set_positions_user_order(plant, _pc, _q_after)
                    print(colored(
                        f"  💥 DISTURBANCE at t={t_now:.2f} s  "
                        f"q₂: {np.rad2deg(_q_before[1]):.1f}° → "
                        f"{np.rad2deg(_q_after[1]):.1f}° "
                        f"(Δq₂ = +{args.disturbance_dq:.1f}°) [pos jump]",
                        "red",
                    ))
            _lap_now = int(max(0.0, t_now - args.move_duration) / args.duration)
            if _lap_now > _lap_prev:
                _lap_prev = _lap_now
                print(colored(f"  Lap {_lap_now} complete  (t={t_now:.1f} s)", "cyan"))
                if _max_laps > 0 and _lap_now >= _max_laps:
                    print(colored(
                        f"\n  ✓ {_max_laps} lap(s) done — auto-stopping."
                        f"\n  Generating plots...",
                        "yellow",
                    ))
                    break
            simulator.AdvanceTo(t_now + _chunk)
            _viz_tick()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(colored(
            f"\n  Simulation stopped at t={sim_ctx.get_time():.2f} s."
            f"\n  Generating plots...",
            "yellow",
        ))
        signal.signal(signal.SIGINT, signal.default_int_handler)

    if _visualizer is not None:
        _visualizer.StopRecording()
        _visualizer.PublishRecording()

    # ── 10. Collect logs ─────────────────────────────────────────────────────
    def _get(log):
        obj = log.FindLog(sim_ctx)
        return obj.sample_times(), obj.data()

    t_log, state_data    = _get(log_state)
    _,     act_data      = _get(log_act)
    _,     sea_diag_data = _get(log_sea_diag)
    _,     exo_diag_data = _get(log_exo_diag)
    _,     exo_tau_data  = _get(log_exo_tau)
    _,     qdes_data     = _get(log_qdes)
    _,     ref_data      = _get(log_ref)

    N = min(len(t_log), state_data.shape[1], act_data.shape[1],
            sea_diag_data.shape[1], exo_diag_data.shape[1],
            exo_tau_data.shape[1], qdes_data.shape[1], ref_data.shape[1])
    t_log        = t_log[:N]
    state_data   = state_data[:, :N]
    act_data     = act_data[:, :N]
    sea_diag_data= sea_diag_data[:, :N]
    exo_diag_data= exo_diag_data[:, :N]
    exo_tau_data = exo_tau_data[:, :N]
    qdes_data    = qdes_data[:, :N]
    ref_data     = ref_data[:, :N]

    # FK for actual EE position
    nq       = plant.num_positions()
    ee_x_act = np.zeros(N)
    ee_y_act = np.zeros(N)
    tmp_ctx  = plant.CreateDefaultContext()
    for k in range(N):
        plant.SetPositionsAndVelocities(tmp_ctx, state_data[:, k])
        p = manipulator.get_end_effector_position(plant, tmp_ctx)
        ee_x_act[k] = p[0]
        ee_y_act[k] = p[1]

    return dict(
        t=t_log, state=state_data, actuation=act_data,
        sea_diag=sea_diag_data, exo_diag=exo_diag_data,
        exo_tau=exo_tau_data, q_des=qdes_data, ref=ref_data,
        ee_x=ee_x_act, ee_y=ee_y_act,
        ee_x_tgt=ee_x_tgt, ee_y_tgt=ee_y_tgt,
        nq=nq, r_p=manipulator.PULLEY_RADIUS,
        r_exo=manipulator.EXO_PULLEY_RADIUS,
        t_disturbance=args.disturbance_time if args.disturbance else None,
    )


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_results(data: dict):
    """Generate two diagnostic figures for SEA + exo simulation.

    **Figure 1 — Manipulator & Drive SEA** (5 × 2):
        Row 0: EE position X / EE position Y
        Row 1: Joint 1 (q₁) / Joint 2 (q₂)
        Row 2: Joint torques / τ₂ desired vs applied
        Row 3: Drive SEA: spring δ & cable force (twin) / T_green & T_red
        Row 4: Drive cable length & velocity / Motor torque & EE error

    **Figure 2 — Exosuit Co-Contraction** (4 × 2):
        Row 0: Exo cable lengths R/L / Exo cable velocities R/L
        Row 1: Exo spring δ_R,L & forces F_R,L (twin) / Exo motor positions
        Row 2: τ_exo contribution / EE XY path
        Row 3: τ₂ total breakdown (drive + exo) / EE tracking error
    """
    import time as _time

    t        = data["t"]
    nq       = data["nq"]
    state    = data["state"]
    act      = data["actuation"]
    sea_diag = data["sea_diag"]
    exo_diag = data["exo_diag"]
    exo_tau  = data["exo_tau"]
    q_des    = data["q_des"]
    ref      = data["ref"]
    r_p      = data["r_p"]
    r_exo    = data["r_exo"]

    t_act  = args.exo_activate_time
    t_dist = data.get("t_disturbance", None)

    # ── Derived signals ──────────────────────────────────────────────────
    q2     = state[1]
    q2_dot = state[nq + 1]

    # Drive SEA diagnostics
    l_m       = sea_diag[0]   # motor cable displacement [m]
    l_m_des   = sea_diag[1]   # desired motor cable [m]
    sea_delta = sea_diag[2]   # spring extension δ [m]
    sea_F     = sea_diag[3]   # net cable force [N]
    tau1_des  = sea_diag[4]   # desired τ₁ [Nm]
    tau2_des  = sea_diag[5]   # desired τ₂ [Nm]
    T_green   = sea_diag[6]   # green (retract) cable tension [N]
    T_red     = sea_diag[7]   # red (extend) cable tension [N]
    tau_motor = sea_diag[8] if sea_diag.shape[0] > 8 else np.zeros_like(t)

    # Drive cable kinematics (joint-side)
    l_drive     = r_p * q2
    l_dot_drive = r_p * q2_dot

    # Exo diagnostics
    exo_dR = exo_diag[0]      # δ_R [m]
    exo_dL = exo_diag[1]      # δ_L [m]
    exo_FR = exo_diag[2]      # F_R [N]
    exo_FL = exo_diag[3]      # F_L [N]
    exo_mR = exo_diag[4]      # θ_mR/N [rad]
    exo_mL = exo_diag[5]      # θ_mL/N [rad]

    # Exo cable kinematics
    l_exo_R     = r_exo * q2
    l_dot_exo_R = r_exo * q2_dot
    l_exo_L     = r_exo * (-q2)
    l_dot_exo_L = r_exo * (-q2_dot)

    # Actuation
    tau1_act = act[0]
    tau2_act = act[1]

    # EE tracking error
    ee_err = np.sqrt((data["ee_x"] - ref[0])**2 + (data["ee_y"] - ref[1])**2)

    # Exo state label used in title, filename, and figure badge
    if args.exo_reactive:
        _exo_label = "EXO: REACTIVE"
        _exo_tag   = "exo_reactive"
    elif not args.no_exo_activate:
        _exo_label = f"EXO: ON  (t_act={args.exo_activate_time:.1f} s)"
        _exo_tag   = "exo_on"
    else:
        _exo_label = "EXO: OFF  (deactivated)"
        _exo_tag   = "exo_off"

    _suptitle = (
        f"SEA + Exo Co-Contraction  —  k_s={args.spring_stiffness}  "
        f"k_exo={args.exo_ks}  Δθ={args.exo_delta_theta:.3f}  "
        f"k_eff={2*args.exo_ks*args.exo_r**2:.4f} Nm/rad  |  {_exo_label}"
    )

    def _style(ax, ylabel="", title="", xlabel=""):
        ax.axvline(t_act, color="m", ls=":", lw=1, alpha=0.6)
        if t_dist is not None:
            ax.axvline(t_dist, color="r", ls="-", lw=1.5, alpha=0.7)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1 — Manipulator & Drive SEA  (5 × 2)
    # ════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(5, 2, figsize=(15, 18),
                              num="Fig 1: Manipulator & Drive SEA")
    fig1.suptitle(_suptitle + "\n(Manipulator & Drive SEA)", fontsize=11)
    fig1.text(0.99, 0.99, _exo_label, ha="right", va="top", fontsize=9,
             fontweight="bold",
             color="white" if _exo_tag != "exo_off" else "dimgray",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="green" if _exo_tag != "exo_off" else "lightgray",
                       alpha=0.85))

    # ── Row 0: EE position X / Y ────────────────────────────────────────
    # x_EE = FK(q₁, q₂),  reference from trajectory planner
    ax1[0][0].plot(t, ref[0], "r--", lw=1, label="ref X")
    ax1[0][0].plot(t, data["ee_x"], "b-", lw=1.2, label="actual X")
    _style(ax1[0][0], "[m]", "EE Position X  —  x_EE = FK(q₁, q₂)")

    ax1[0][1].plot(t, ref[1], "r--", lw=1, label="ref Y")
    ax1[0][1].plot(t, data["ee_y"], "b-", lw=1.2, label="actual Y")
    _style(ax1[0][1], "[m]", "EE Position Y  —  y_EE = FK(q₁, q₂)")

    # ── Row 1: Joint 1 / Joint 2 ────────────────────────────────────────
    # q₁ = shoulder angle (link1_base), directly actuated by τ₁
    ax1[1][0].plot(t, np.rad2deg(state[0]), "b-", lw=1.2, label="q₁ actual")
    ax1[1][0].plot(t, np.rad2deg(q_des[0]), "r--", lw=1, label="q₁ desired")
    _style(ax1[1][0], "[deg]",
           "Joint 1 (Shoulder)  —  q₁: link1_base angle, τ₁ = CT direct")

    # q₂ = elbow angle (link2_link1), actuated via drive cable SEA
    ax1[1][1].plot(t, np.rad2deg(q2), "b-", lw=1.2, label="q₂ actual")
    ax1[1][1].plot(t, np.rad2deg(q_des[1]), "r--", lw=1, label="q₂ desired")
    _style(ax1[1][1], "[deg]",
           "Joint 2 (Elbow)  —  q₂: link2_link1, actuated via cable SEA")

    # ── Row 2: Joint torques / τ₂ desired vs applied ────────────────────
    # τ₁ = CT direct, τ₂ = r_p·F_cable + τ_exo,  τ_exo = r_exo·(F_R − F_L)
    ax1[2][0].plot(t, tau1_act, "g-", lw=1.2, label="τ₁ (CT direct)")
    ax1[2][0].plot(t, tau2_act, "b-", lw=1.2, label="τ₂ total (drive+exo)")
    ax1[2][0].plot(t, exo_tau[0], "m-", lw=1, label="τ_exo")
    ax1[2][0].axhline(0, color="k", lw=0.5)
    _style(ax1[2][0], "[Nm]",
           "Joint Torques  —  τ₂ = r_p·F_cable + τ_exo")

    # τ₂_des from CT,  τ₂_applied = r_p·(T_green − T_red) + τ_exo
    ax1[2][1].plot(t, tau2_des, "r--", lw=1, label="τ₂_des (CT)")
    ax1[2][1].plot(t, tau2_act, "b-", lw=1.2, label="τ₂_applied")
    ax1[2][1].plot(t, r_p * sea_F, "g-", lw=0.9, alpha=0.7,
                   label="r_p·F_cable (drive only)")
    ax1[2][1].axhline(0, color="k", lw=0.5)
    _style(ax1[2][1], "[Nm]",
           "τ₂ Desired vs Applied  —  τ₂_app = r_p·(T_g−T_r) + τ_exo")

    # ── Row 3: Drive SEA spring+force (twin axis) / T_green & T_red ────
    # δ = l_m − r_p·q₂,  F = k_s·δ + b_c·δ̇  (linearly related)
    ax1[3][0].plot(t, sea_delta * 1e3, "b-", lw=1.2, label="δ drive [mm]")
    ax1[3][0].axhline(0, color="k", lw=0.5)
    ax_3r = ax1[3][0].twinx()
    ax_3r.plot(t, sea_F, "r-", lw=1, alpha=0.7, label="F_cable [N]")
    ax_3r.set_ylabel("F [N]", color="r"); ax_3r.tick_params(axis="y", labelcolor="r")
    ax_3r.legend(loc="upper right", fontsize=7)
    _style(ax1[3][0], "δ [mm]",
           "Drive SEA: δ = l_m − r_p·q₂,  F = k_s·δ + b_c·δ̇")

    # T_green = max(F,0) when δ>0 (retract),  T_red = max(−F,0) when δ<0 (extend)
    ax1[3][1].plot(t, T_green, "g-", lw=1.2, label="T_green (retract)")
    ax1[3][1].plot(t, T_red, "r-", lw=1.2, label="T_red (extend)")
    ax1[3][1].axhline(0, color="k", lw=0.5)
    _style(ax1[3][1], "[N]",
           "Cable Tensions  —  T_g = max(F,0),  T_r = max(−F,0)")

    # ── Row 4: Drive cable length+velocity / Motor torque + EE error ────
    # l_drive = r_p·q₂ (cable arc at BigPulley),  l̇ = r_p·q̇₂
    ax1[4][0].plot(t, l_drive * 1e3, "b-", lw=1.2, label="l = r_p·q₂ [mm]")
    ax_4r = ax1[4][0].twinx()
    ax_4r.plot(t, l_dot_drive * 1e3, "c-", lw=0.9, alpha=0.7,
               label="l̇ = r_p·q̇₂ [mm/s]")
    ax_4r.set_ylabel("l̇ [mm/s]", color="c")
    ax_4r.tick_params(axis="y", labelcolor="c")
    ax_4r.legend(loc="upper right", fontsize=7)
    ax_4r.axhline(0, color="k", lw=0.3)
    _style(ax1[4][0], "l [mm]",
           "Drive Cable  —  l = r_p·q₂ (BigPulley arc)", "Time [s]")

    # Motor electromagnetic torque + EE position error
    ax1[4][1].plot(t, tau_motor, "g-", lw=1.2, label="τ_motor [Nm]")
    ax1[4][1].axhline(0, color="k", lw=0.5)
    ax_4r2 = ax1[4][1].twinx()
    ax_4r2.plot(t, ee_err * 1e3, "b-", lw=0.9, alpha=0.6,
                label="‖e_EE‖ [mm]")
    rms = np.sqrt(np.mean(ee_err**2))
    ax_4r2.axhline(rms * 1e3, color="r", ls="--", lw=0.8, alpha=0.6,
                   label=f"RMS = {rms*1e3:.2f} mm")
    ax_4r2.set_ylabel("EE err [mm]", color="b")
    ax_4r2.tick_params(axis="y", labelcolor="b")
    ax_4r2.legend(loc="upper right", fontsize=7)
    _style(ax1[4][1], "τ [Nm]",
           "Motor Torque (τ_m = τ₂_des/N)  &  EE Error", "Time [s]")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2 — Exosuit Co-Contraction  (4 × 2)
    # ════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(4, 2, figsize=(15, 14),
                              num="Fig 2: Exosuit Co-Contraction")
    fig2.suptitle(_suptitle + "\n(Exosuit Co-Contraction Detail)", fontsize=11)
    fig2.text(0.99, 0.99, _exo_label, ha="right", va="top", fontsize=9,
             fontweight="bold",
             color="white" if _exo_tag != "exo_off" else "dimgray",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="green" if _exo_tag != "exo_off" else "lightgray",
                       alpha=0.85))

    # ── Row 0: Exo cable lengths / velocities ───────────────────────────
    # l_R = r_exo·q₂,  l_L = r_exo·(−q₂)  — antagonistic pair
    ax2[0][0].plot(t, l_exo_R * 1e3, "tab:orange", lw=1.2,
                   label="l_R = r_exo·q₂ [mm]")
    ax2[0][0].plot(t, l_exo_L * 1e3, "tab:purple", lw=1.2,
                   label="l_L = −r_exo·q₂ [mm]")
    ax2[0][0].axhline(0, color="k", lw=0.5)
    _style(ax2[0][0], "[mm]",
           "Exo Cable Lengths  —  l_R = r_exo·q₂,  l_L = −r_exo·q₂")

    # l̇_R = r_exo·q̇₂,  l̇_L = −r_exo·q̇₂
    ax2[0][1].plot(t, l_dot_exo_R * 1e3, "tab:orange", lw=1.2,
                   label="l̇_R [mm/s]")
    ax2[0][1].plot(t, l_dot_exo_L * 1e3, "tab:purple", lw=1.2,
                   label="l̇_L [mm/s]")
    ax2[0][1].axhline(0, color="k", lw=0.5)
    _style(ax2[0][1], "[mm/s]",
           "Exo Cable Velocities  —  l̇ = ±r_exo·q̇₂")

    # ── Row 1: Exo spring δ + force (twin axis) / Exo motor positions ──
    # δ_R = l_mR − r_exo·q₂,   F_R = k_exo·δ_R + b_exo·δ̇_R
    ax2[1][0].plot(t, exo_dR * 1e3, "tab:orange", lw=1.2, label="δ_R [mm]")
    ax2[1][0].plot(t, exo_dL * 1e3, "tab:purple", lw=1.2, label="δ_L [mm]")
    ax2[1][0].axhline(0, color="k", lw=0.5)
    ax_e1r = ax2[1][0].twinx()
    ax_e1r.plot(t, exo_FR, "tab:orange", ls="--", lw=0.9, alpha=0.6,
                label="F_R [N]")
    ax_e1r.plot(t, exo_FL, "tab:purple", ls="--", lw=0.9, alpha=0.6,
                label="F_L [N]")
    ax_e1r.set_ylabel("F [N]"); ax_e1r.legend(loc="lower right", fontsize=7)
    _style(ax2[1][0], "δ [mm]",
           "Exo Spring  —  δ = l_m − r_exo·q₂,  F = k_exo·δ + b_exo·δ̇")
    ax2[1][0].legend(loc="upper left", fontsize=7)

    # θ_mR/N, θ_mL/N — exo motor-side joint-referred angles [deg]
    ax2[1][1].plot(t, np.rad2deg(exo_mR), "tab:orange", lw=1.2,
                   label="θ_mR/N (motor R)")
    ax2[1][1].plot(t, np.rad2deg(exo_mL), "tab:purple", lw=1.2,
                   label="θ_mL/N (motor L)")
    ax2[1][1].plot(t, np.rad2deg(q2), "b-", lw=0.9, alpha=0.5,
                   label="q₂ actual")
    ax2[1][1].plot(t, np.rad2deg(q_des[1]), "r--", lw=0.8, alpha=0.5,
                   label="q₂ desired")
    _style(ax2[1][1], "[deg]",
           "Exo Motor Positions  —  θ_m/N: motor-side at joint [rad→deg]")

    # ── Row 2: τ_exo contribution / EE XY path ─────────────────────────
    # τ_exo = r_exo·(F_R − F_L) — net exo torque at elbow
    ax2[2][0].plot(t, exo_tau[0], "m-", lw=1.2, label="τ_exo [Nm]")
    ax2[2][0].axhline(0, color="k", lw=0.5)
    ax_e2r = ax2[2][0].twinx()
    ax_e2r.plot(t, exo_FR - exo_FL, "k-", lw=0.8, alpha=0.5,
                label="F_R − F_L [N]")
    ax_e2r.set_ylabel("ΔF [N]"); ax_e2r.legend(loc="lower right", fontsize=7)
    _style(ax2[2][0], "[Nm]",
           "Exo Torque  —  τ_exo = r_exo·(F_R − F_L)", "Time [s]")
    ax2[2][0].legend(loc="upper left", fontsize=7)

    ax2[2][1].plot(ref[0], ref[1], "r--", lw=1, label="reference")
    ax2[2][1].plot(data["ee_x"], data["ee_y"], "b-", lw=1.2, label="actual")
    ax2[2][1].set_aspect("equal", adjustable="datalim")
    ax2[2][1].set_ylabel("[m]"); ax2[2][1].set_xlabel("[m]")
    ax2[2][1].set_title("EE XY Path", fontsize=9)
    ax2[2][1].legend(fontsize=7); ax2[2][1].grid(True, alpha=0.4)

    # ── Row 3: τ₂ total breakdown / EE tracking error ──────────────────
    # τ₂_total = r_p·F_cable + τ_exo — drive + exo contributions
    ax2[3][0].plot(t, r_p * sea_F, "g-", lw=1.2,
                   label="r_p·F_cable (drive)")
    ax2[3][0].plot(t, exo_tau[0], "m-", lw=1.2, label="τ_exo")
    ax2[3][0].plot(t, tau2_act, "b--", lw=1, label="τ₂_total")
    ax2[3][0].plot(t, tau2_des, "r:", lw=1, alpha=0.6, label="τ₂_des (CT)")
    ax2[3][0].axhline(0, color="k", lw=0.5)
    _style(ax2[3][0], "[Nm]",
           "τ₂ Breakdown  —  τ₂ = r_p·F_cable + τ_exo", "Time [s]")

    # ‖e_EE‖ = √((x−x_ref)² + (y−y_ref)²)
    ax2[3][1].plot(t, ee_err * 1e3, "b-", lw=1.2, label="‖e_EE‖ [mm]")
    ax2[3][1].axhline(rms * 1e3, color="r", ls="--", lw=1,
                      label=f"RMS = {rms*1e3:.2f} mm")
    _style(ax2[3][1], "[mm]",
           "EE Tracking Error  —  ‖e‖ = √((x−x_ref)² + (y−y_ref)²)",
           "Time [s]")

    fig2.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Save both figures ────────────────────────────────────────────────
    _stamp    = _time.strftime("%Y%m%d_%H%M%S")
    _plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(_plot_dir, exist_ok=True)
    _base = f"sea_exo_kexo{int(args.exo_ks)}_dth{args.exo_delta_theta:.2f}_{_exo_tag}_{_stamp}"
    _f1 = os.path.join(_plot_dir, f"{_base}_manip.png")
    _f2 = os.path.join(_plot_dir, f"{_base}_exo.png")
    fig1.savefig(_f1, dpi=150, bbox_inches="tight")
    fig2.savefig(_f2, dpi=150, bbox_inches="tight")
    print(colored(f"\n  📊 Fig 1 saved: {_f1}", "green"))
    print(colored(f"  📊 Fig 2 saved: {_f2}", "green"))

    # ── Structured metrics (for sweeps / A-B comparisons) ────────────────
    ee_err_mm = ee_err * 1e3
    q2_err_deg = np.rad2deg(q2 - q_des[1])
    mask_all = np.ones_like(t, dtype=bool)
    if t_dist is not None:
        _t0 = float(t_dist)
        _t1 = _t0 + float(getattr(args, "disturbance_dur", 1.5))
        mask_dist = (t >= _t0) & (t <= _t1)
        mask_post = (t >= _t0) & (t <= _t1 + 1.5)
    else:
        mask_dist = mask_all
        mask_post = mask_all
    def _safe(a, op):
        return float(op(a)) if a.size else float("nan")
    peak_ee_dist   = _safe(ee_err_mm[mask_dist], np.max)
    rms_ee_dist    = _safe(ee_err_mm[mask_dist], lambda x: np.sqrt(np.mean(x**2)))
    peak_q2_err    = _safe(np.abs(q2_err_deg[mask_dist]), np.max)
    rms_q2_err     = _safe(q2_err_deg[mask_dist], lambda x: np.sqrt(np.mean(x**2)))
    peak_tau_exo   = _safe(np.abs(exo_tau[0]), np.max)
    peak_tau_drive = _safe(np.abs(tau2_act), np.max)
    print(colored(
        "\n  METRICS  |  "
        f"peak_ee_dist={peak_ee_dist:7.2f} mm  "
        f"rms_ee_dist={rms_ee_dist:7.2f} mm  "
        f"peak_q2_err={peak_q2_err:6.2f} deg  "
        f"rms_q2_err={rms_q2_err:6.2f} deg  "
        f"peak|tau_exo|={peak_tau_exo:5.2f} Nm  "
        f"peak|tau_drive|={peak_tau_drive:5.2f} Nm",
        "cyan",
    ))
    print(
        f"METRICS_CSV,{_exo_tag},{args.exo_ks:.1f},{args.exo_delta_theta:.3f},"
        f"{args.disturbance_tau:.3f},{args.disturbance_dur:.3f},"
        f"{peak_ee_dist:.3f},{rms_ee_dist:.3f},{peak_q2_err:.3f},"
        f"{rms_q2_err:.3f},{peak_tau_exo:.3f},{peak_tau_drive:.3f}"
    )
    if getattr(args, "no_show", False):
        plt.close("all")
        return
    try:
        plt.show(block=True)
    except Exception as _e:
        print(colored(f"  ⚠ plt.show() failed ({_e}) — open the saved PNGs above.", "yellow"))


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    meshcat = None if args.no_meshcat else StartMeshcat()
    if meshcat is not None:
        print(colored(f"  Meshcat: {meshcat.web_url()}", "green"))

    data = run_simulation(meshcat)
    plot_results(data)


if __name__ == "__main__":
    main()
