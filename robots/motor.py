"""Motor model definitions for CubeMars (and future) actuators.

All motor/gearbox parameters are referred to the **joint side** (after the
gearbox) unless the attribute name ends in ``_motor`` (rotor / motor side).

Usage::

    from robots.motor import get_motor, MOTOR_CHOICES, AK80_8_KV60_Config

    motor = get_motor("AK80_8_KV60_Config")   # default-instantiated
    print(motor.peak_torque_joint)             # 18.0 Nm

    # argparse integration:
    parser.add_argument("--motor", choices=MOTOR_CHOICES, default="AK80_8_KV60_Config")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

# ── Registry ───────────────────────────────────────────────────────────────────
# Populated automatically by @motor_choice.
_MOTOR_REGISTRY: Dict[str, type] = {}


def motor_choice(cls):
    """Class decorator that registers a motor dataclass by name.

    Example::

        @motor_choice
        @dataclass(frozen=True)
        class AK80_8_KV60_Config(MotorModelConfig):
            ...

    Registered motors are accessible via :func:`get_motor` or
    :data:`MOTOR_CHOICES` (a list of valid names for argparse).
    """
    _MOTOR_REGISTRY[cls.__name__] = cls
    return cls


def get_motor(name: str) -> "MotorModelConfig":
    """Return a default-instantiated motor for *name*.

    Args:
        name: Key registered via ``@motor_choice``,
              e.g. ``'AK80_8_KV60_Config'``.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _MOTOR_REGISTRY:
        raise KeyError(
            f"Unknown motor '{name}'. Valid choices: {list(_MOTOR_REGISTRY)}"
        )
    return _MOTOR_REGISTRY[name]()


# ── Abstract base ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MotorModelConfig(ABC):
    """Abstract base for actuator (motor + gearbox) models.

    All parameters are referred to the **joint side** (after the gearbox)
    unless the attribute name ends in ``_motor`` (rotor / motor side).

    Concrete subclasses should be decorated with ``@motor_choice`` and
    ``@dataclass(frozen=True)`` so they are immutable and auto-registered.
    """

    # ── Joint-side limits ──────────────────────────────────────────────────
    @property
    @abstractmethod
    def gear_ratio(self) -> float:
        """Gearbox reduction ratio N  (motor turns per joint turn)."""

    @property
    @abstractmethod
    def peak_torque_joint(self) -> float:
        """Peak (short-duration) output torque  [Nm]."""

    @property
    @abstractmethod
    def continuous_torque_joint(self) -> float:
        """Continuous (thermally limited) output torque  [Nm]."""

    @property
    @abstractmethod
    def no_load_rpm_motor(self) -> float:
        """No-load motor-shaft speed at rated voltage  [rpm].
        This is the raw datasheet value (motor side, before gearbox)."""

    @property
    def max_velocity_joint(self) -> float:
        """No-load maximum joint velocity  [rad/s].
        Derived as  no_load_rpm_motor / N × (2π/60)."""
        import math
        return self.no_load_rpm_motor / self.gear_ratio * (2 * math.pi / 60)

    # ── Inertia ────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def rotor_inertia_motor(self) -> float:
        """Rotor moment of inertia on the motor shaft  [kg·m²]."""

    @property
    def rotor_inertia_joint(self) -> float:
        """Reflected rotor inertia at the joint  [kg·m²].
        Computed as  I_r = N² · I_motor."""
        return self.gear_ratio ** 2 * self.rotor_inertia_motor

    # ── Friction / damping ────────────────────────────────────────────────
    @property
    @abstractmethod
    def viscous_damping_joint(self) -> float:
        """Viscous friction coefficient referred to joint  [Nm·s/rad]."""

    @property
    @abstractmethod
    def back_drive_torque(self) -> float:
        """Static torque required to back-drive the joint  [Nm].
        Reflects cogging + gearbox friction.  Important for impedance
        control transparency and safety assessments."""

    # ── Electrical ────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def torque_constant(self) -> float:
        """Motor-side torque constant KT  [Nm/A].
        Maps phase current to motor-shaft torque."""

    @property
    @abstractmethod
    def rated_voltage(self) -> float:
        """Nominal bus voltage  [V].
        Determines voltage-limited max speed and driver/battery selection."""

    @property
    @abstractmethod
    def pole_pairs(self) -> int:
        """Number of magnetic pole pairs.
        Required for FOC commutation angle on real hardware."""

    @property
    @abstractmethod
    def winding_resistance(self) -> float:
        """Phase-to-phase winding resistance  [Ω]."""

    @property
    @abstractmethod
    def winding_inductance(self) -> float:
        """Phase-to-phase winding inductance  [H]."""

    @property
    def electrical_time_constant(self) -> float:
        """Electrical time constant  τ_e = L / R  [s]."""
        return self.winding_inductance / self.winding_resistance

    # ── Physical ──────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def mass(self) -> float:
        """Total actuator mass (motor + gearbox)  [kg]."""

    # ── Derived (joint-side) ──────────────────────────────────────────────
    @property
    def torque_constant_joint(self) -> float:
        """Joint-side torque constant  KT_joint = KT × N  [Nm/A]."""
        return self.torque_constant * self.gear_ratio

    @property
    def peak_current(self) -> float:
        """Peak phase current at peak joint torque  [A].
        Computed as  τ_peak / (N × KT)."""
        return self.peak_torque_joint / self.torque_constant_joint


# ── Concrete motor models ──────────────────────────────────────────────────────

@motor_choice
@dataclass(frozen=True)
class AK80_8_KV60_Config(MotorModelConfig):
    """CubeMars AK80-8  KV60 quasi-direct-drive actuator  (48 V variant).

    Specs (joint side, 8:1 planetary gearbox):

    ========================  =========  =====
    Parameter                 Value      Unit
    ========================  =========  =====
    Gear ratio                8          —
    Peak torque               25         Nm
    Continuous torque (rated) 10         Nm
    Rated speed               243        rpm  (joint)
    No-load speed (motor)     297.5      rpm  (interp: ω_0 = 243/(1−10/25))
    Rotor inertia             608.6      gcm²  → 6.086e-5 kg·m²
    Reflected inertia         0.0390     kg·m²  (N²×6.086e-5)
    Viscous damping (joint)   0.30       Nm·s/rad  (est. from back-drive/rated ω)
    Back-drive torque         0.75       Nm
    KT (motor)                0.199      Nm/A
    KV                        60         rpm/V
    Rated voltage             48         V
    Pole pairs                21         —
    Phase resistance          430        mΩ  → τ_e = 0.50 ms
    Phase inductance          214        μH
    Mass                      570        g
    ========================  =========  =====
    """

    @property
    def gear_ratio(self) -> float:              return 8.0
    @property
    def peak_torque_joint(self) -> float:       return 25.0           # Nm
    @property
    def continuous_torque_joint(self) -> float: return 10.0           # Nm  (rated torque)
    @property
    def no_load_rpm_motor(self) -> float:       return 297.5          # rpm motor side
    @property
    def rotor_inertia_motor(self) -> float:     return 608.6e-7       # 608.6 gcm² → kg·m²
    @property
    def viscous_damping_joint(self) -> float:   return 0.30           # Nm·s/rad  (est.)
    @property
    def back_drive_torque(self) -> float:       return 0.75           # Nm
    @property
    def torque_constant(self) -> float:         return 0.199          # Nm/A  (motor side)
    @property
    def rated_voltage(self) -> float:           return 48.0           # V
    @property
    def pole_pairs(self) -> int:                return 21
    @property
    def winding_resistance(self) -> float:      return 0.430          # Ω  (phase-to-phase)
    @property
    def winding_inductance(self) -> float:      return 214e-6         # H  (phase-to-phase)
    @property
    def mass(self) -> float:                    return 0.570          # kg


@motor_choice
@dataclass(frozen=True)
class AK60_6_KV80_Config(MotorModelConfig):
    """CubeMars AK60-6 V3.0  KV80 quasi-direct-drive actuator.

    Specs (joint side, 6:1 planetary gearbox):

    =====================  =========  =====
    Parameter              Value      Unit
    =====================  =========  =====
    Gear ratio             6          —
    Peak torque            9          Nm
    Continuous torque      3          Nm
    No-load speed (48 V)   640        rpm motor → 106.7 rpm joint → 11.17 rad/s
    Rotor inertia          243.5      gcm²  → 2.435e-5 kg·m²
    Reflected inertia      8.766e-4   kg·m²  (N²×2.435e-5)
    Viscous damping        0.12       Nm·s/rad  (est.)
    Back-drive torque      0.2        Nm
    KT (motor)             0.135      Nm/A
    KV                     80         rpm/V
    Rated voltage          48         V
    Pole pairs             14         —
    Phase resistance       595        mΩ
    Phase inductance       676        μH  → τ_e ≈ 1.14 ms
    Mass                   0.380      kg
    =====================  =========  =====
    """

    @property
    def gear_ratio(self) -> float:              return 6.0
    @property
    def peak_torque_joint(self) -> float:       return 9.0            # Nm
    @property
    def continuous_torque_joint(self) -> float: return 3.0            # Nm
    @property
    def no_load_rpm_motor(self) -> float:       return 640.0          # rpm (motor side, from datasheet at 48 V)
    @property
    def rotor_inertia_motor(self) -> float:     return 243.5e-7       # 243.5 gcm² → kg·m²
    @property
    def viscous_damping_joint(self) -> float:   return 0.12           # Nm·s/rad (est.)
    @property
    def back_drive_torque(self) -> float:       return 0.2            # Nm (from datasheet)
    @property
    def torque_constant(self) -> float:         return 0.135          # Nm/A (motor side, from datasheet)
    @property
    def rated_voltage(self) -> float:           return 48.0           # V
    @property
    def pole_pairs(self) -> int:                return 14
    @property
    def winding_resistance(self) -> float:      return 0.595          # Ω (phase-to-phase)
    @property
    def winding_inductance(self) -> float:      return 676e-6         # H (phase-to-phase)
    @property
    def mass(self) -> float:                    return 0.380          # kg


# Convenience list of all registered motor class names.
# Use as:  parser.add_argument("--motor", choices=MOTOR_CHOICES, default="AK80_8_KV60_Config")
MOTOR_CHOICES: List[str] = list(_MOTOR_REGISTRY)


if __name__ == "__main__":
    import math
    import argparse
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── CLI arguments ──────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="Motor registry smoke-test + single-link manipulator performance test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python robots/motor.py                              # defaults\n"
            "  python robots/motor.py --theta-start 10 --theta-end 170 --duration 1.0\n"
            "  python robots/motor.py --mass 3.0 --link-length 1.0 --com-percent 75 --no-smoke\n"
        ),
    )
    # manipulator geometry
    ap.add_argument("--mass",        type=float, default=2.0,
                    help="Link mass [kg]  (default: 2.0)")
    ap.add_argument("--link-length", type=float, default=1.0,
                    help="Total link length [m]  (default: 1.0)")
    ap.add_argument("--com-percent", type=float, default=75.0,
                    help="CoM position as %% of link length from joint  (default: 75.0)")
    # motion parameters
    ap.add_argument("--theta-start", type=float, default=10.0,
                    help="Start angle [deg]  (default: 10)")
    ap.add_argument("--theta-end",   type=float, default=85.0,
                    help="End angle [deg]  (default: 85)")
    ap.add_argument("--duration",    type=float, default=1.0,
                    help="Move duration [s]  (default: 1.0)")
    ap.add_argument("--gravity",     type=float, default=9.81,
                    help="Gravitational acceleration [m/s²]  (default: 9.81)")
    ap.add_argument("--n-points",    type=int,   default=500,
                    help="Number of time-points for simulation  (default: 500)")
    # misc
    ap.add_argument("--no-smoke",    action="store_true",
                    help="Skip the registry smoke test")
    args = ap.parse_args()

    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"

    # ── Smoke test ─────────────────────────────────────────────────────────────
    if not args.no_smoke:
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  Motor Registry — smoke test{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"\n{CYAN}Registered motors:{RESET}  {MOTOR_CHOICES}\n")

        for name in MOTOR_CHOICES:
            m = get_motor(name)
            print(f"{BOLD}{YELLOW}── {name} ──{RESET}")
            print(f"  {'Gear ratio':<30} {m.gear_ratio}")
            print(f"  {'Peak torque (joint)':<30} {m.peak_torque_joint} Nm")
            print(f"  {'Continuous torque (joint)':<30} {m.continuous_torque_joint} Nm")
            print(f"  {'No-load speed (motor)':<30} {m.no_load_rpm_motor:.0f} rpm")
            print(f"  {'No-load speed (joint)':<30} {m.no_load_rpm_motor / m.gear_ratio:.1f} rpm")
            print(f"  {'Max velocity (joint)':<30} {m.max_velocity_joint:.4f} rad/s"
                  f"  ({m.max_velocity_joint * 60 / (2*math.pi):.1f} rpm)  [derived]")
            print(f"  {'Rotor inertia (motor side)':<30} {m.rotor_inertia_motor*1e6:.1f} g·cm²"
                  f"  ({m.rotor_inertia_motor:.6f} kg·m²)")
            print(f"  {'Reflected inertia (joint)':<30} {m.rotor_inertia_joint:.5f} kg·m²"
                  f"  (= N²·I_motor = {m.gear_ratio}²·{m.rotor_inertia_motor})")
            print(f"  {'Viscous damping (joint)':<30} {m.viscous_damping_joint} Nm·s/rad")
            print(f"  {'Back-drive torque':<30} {m.back_drive_torque} Nm")
            print(f"  {'Torque constant KT (motor)':<30} {m.torque_constant} Nm/A")
            print(f"  {'Torque constant KT (joint)':<30} {m.torque_constant_joint:.3f} Nm/A"
                  f"  (= KT×N = {m.torque_constant}×{m.gear_ratio})")
            print(f"  {'Peak current':<30} {m.peak_current:.2f} A"
                  f"  (= τ_peak / KT_joint)")
            print(f"  {'Rated voltage':<30} {m.rated_voltage} V")
            print(f"  {'Pole pairs':<30} {m.pole_pairs}")
            print(f"  {'Winding resistance':<30} {m.winding_resistance} Ω")
            print(f"  {'Winding inductance':<30} {m.winding_inductance*1e6:.1f} μH")
            print(f"  {'Electrical time constant':<30} {m.electrical_time_constant*1e3:.3f} ms"
                  f"  (τ_e = L/R)")
            print(f"  {'Mass':<30} {m.mass} kg")
            print()

        print(f"{CYAN}get_motor() round-trip test:{RESET}")
        for name in MOTOR_CHOICES:
            m = get_motor(name)
            print(f"  get_motor('{name}') → {type(m).__name__}  ✓")

        print(f"\n{CYAN}KeyError on unknown motor:{RESET}")
        try:
            get_motor("NonExistentMotor")
        except KeyError as e:
            print(f"  Caught expected KeyError: {e}")

        print(f"\n{GREEN}All checks passed.{RESET}\n")

    # ── Single-link performance test ───────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Single-link manipulator — motor performance test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    m_link  = args.mass
    L_link  = args.link_length
    com_pct = args.com_percent

    if not (0.0 < com_pct < 100.0):
        raise ValueError(f"--com-percent must be between 0 and 100 (got {com_pct}).")

    l_com = L_link * com_pct / 100.0   # CoM distance from joint [m]
    g       = args.gravity
    T       = args.duration
    th0_deg = args.theta_start
    thf_deg = args.theta_end
    N_pts   = args.n_points

    # Link inertia about joint via parallel-axis theorem (uniform rod assumed)
    # I = (1/12)·m·L² + m·l_com²
    I_link = (1.0 / 12.0) * m_link * L_link**2 + m_link * l_com**2

    th0 = math.radians(th0_deg)
    thf = math.radians(thf_deg)
    dth = thf - th0

    print(f"  Planar manipulator in XY plane — gravity along Z (no joint torque from gravity)")
    print(f"  Link:  mass = {m_link} kg   L = {L_link:.4f} m   l_com = {l_com:.4f} m  ({com_pct:.1f}% of length from joint)")
    print(f"  I_link (about Z-axis, via parallel-axis) = {I_link:.5f} kg·m²")
    print(f"  Move:  {th0_deg}° → {thf_deg}°  (Δ = {math.degrees(dth):.1f}°)  in T = {T} s")
    print(f"  g = {g} m/s²  (acts along Z — contributes 0 Nm torque about Z rotation axis)\n")

    # ── Minimum-jerk trajectory (5th-order polynomial) ─────────────────────────
    # Guarantees zero velocity and acceleration at start and end.
    #   θ(τ)  = θ₀ + Δθ·(10τ³ − 15τ⁴ + 6τ⁵)
    #   θ̇(τ)  = Δθ/T·(30τ² − 60τ³ + 30τ⁴)
    #   θ̈(τ)  = Δθ/T²·(60τ − 180τ² + 120τ³)     τ = t/T ∈ [0,1]
    t_arr = np.linspace(0.0, T, N_pts)
    tau   = t_arr / T

    theta      = th0 + dth * (10*tau**3 - 15*tau**4 + 6*tau**5)
    theta_dot  = (dth / T)    * (30*tau**2 - 60*tau**3 + 30*tau**4)
    theta_ddot = (dth / T**2) * (60*tau    - 180*tau**2 + 120*tau**3)

    omega_peak = float(np.max(np.abs(theta_dot)))
    alpha_peak = float(np.max(np.abs(theta_ddot)))
    print(f"  Trajectory peak velocity:     {omega_peak:.4f} rad/s  ({math.degrees(omega_peak):.2f} °/s)")
    print(f"  Trajectory peak acceleration: {alpha_peak:.4f} rad/s²\n")

    # ── Gravity torque ─────────────────────────────────────────────────────────
    # Manipulator is planar in the XY plane; joint rotates about Z.
    # Gravity acts along −Z  →  τ_z = (r × F_g)·ẑ = 0  (no gravity torque about Z).
    tau_gravity = np.zeros_like(theta)

    # ── Per-motor analysis ─────────────────────────────────────────────────────
    motor_results: dict = {}
    for motor_name in MOTOR_CHOICES:
        motor   = get_motor(motor_name)
        I_total = I_link + motor.rotor_inertia_joint          # link + reflected rotor

        tau_inertia = I_total * theta_ddot                    # inertial torque (XY plane)
        tau_damping = motor.viscous_damping_joint * theta_dot # viscous friction
        tau_total   = tau_inertia + tau_damping               # gravity = 0 (planar XY, g along Z)

        peak_tau  = float(np.max(np.abs(tau_total)))
        torque_ok = peak_tau  <= motor.peak_torque_joint
        vel_ok    = omega_peak <= motor.max_velocity_joint
        feasible  = torque_ok and vel_ok

        motor_results[motor_name] = dict(
            motor=motor,
            I_total=I_total,
            tau_gravity=tau_gravity,
            tau_inertia=tau_inertia,
            tau_damping=tau_damping,
            tau_total=tau_total,
            peak_tau=peak_tau,
            torque_ok=torque_ok,
            vel_ok=vel_ok,
            feasible=feasible,
        )

        sym = f"{GREEN}✅ PASS{RESET}" if feasible else f"{RED}❌ FAIL{RESET}"
        print(f"  {BOLD}{motor_name}{RESET}  →  {sym}")
        print(f"    I_total (link + rotor reflected) = {I_total:.5f} kg·m²")
        print(f"    Peak torque required  = {peak_tau:.2f} Nm"
              f"   limit = {motor.peak_torque_joint} Nm"
              f"   {'✅' if torque_ok else '❌'}")
        print(f"    Peak velocity         = {omega_peak:.4f} rad/s"
              f"  limit = {motor.max_velocity_joint:.4f} rad/s"
              f"  {'✅' if vel_ok else '❌'}")
        vel_margin    = (motor.max_velocity_joint - omega_peak) / motor.max_velocity_joint * 100
        torque_margin = (motor.peak_torque_joint  - peak_tau)   / motor.peak_torque_joint  * 100
        print(f"    Velocity margin  = {vel_margin:+.1f}%   Torque margin = {torque_margin:+.1f}%")
        print()

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        matplotlib.use('MacOSX')
    except Exception:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass

    n_motors = len(MOTOR_CHOICES)
    t_ms = t_arr * 1e3   # ms for x-axis

    fig = plt.figure(figsize=(7 * n_motors, 17))
    fig.suptitle(
        f"Single-link manipulator — motor performance comparison\n"
        f"Planar XY plane  |  gravity along Z (0 Nm joint torque)  |  rotation about Z-axis\n"
        f"m = {m_link} kg   L = {L_link:.3f} m   l_com = {l_com:.3f} m ({com_pct:.0f}% of L)   "
        f"{th0_deg}° → {thf_deg}° in {T} s   (min-jerk trajectory)",
        fontsize=10, fontweight='bold',
    )
    gs = gridspec.GridSpec(5, n_motors, figure=fig, hspace=0.58, wspace=0.35)

    mk = max(1, N_pts // 15)   # marker spacing — ~15 markers across the plot

    # ── Row 0: joint kinematics (shared, spans all columns) ────────────────────
    ax_kin = fig.add_subplot(gs[0, :])
    ax_kin.set_title("Joint kinematics — minimum-jerk profile", fontsize=10)
    ax_kin.plot(t_ms, np.degrees(theta),
                color='royalblue',   lw=2,   label="θ  [deg]")
    ax_kin.plot(t_ms, np.degrees(theta_dot),
                color='darkorange',  lw=1.5, ls='--',
                marker='o', markersize=4, markevery=mk, label="ω  [deg/s]")
    ax_kin.plot(t_ms, np.degrees(theta_ddot),
                color='forestgreen', lw=1.5, ls='-.',
                marker='s', markersize=4, markevery=mk, label="α  [deg/s²]")
    ax_kin.axhline(0, color='k', lw=0.5)
    ax_kin.set_xlabel("Time [ms]")
    ax_kin.legend(fontsize=9, loc='upper right')
    ax_kin.grid(True, alpha=0.3)

    # ── Row 1: RPM (shared, spans all columns) ─────────────────────────────────────
    ax_rpm = fig.add_subplot(gs[1, :])
    ax_rpm.set_title("Joint & motor shaft velocity [RPM]", fontsize=10)

    joint_rpm = theta_dot * 60.0 / (2.0 * math.pi)   # joint shaft RPM over time

    ax_rpm.plot(t_ms, joint_rpm, color='royalblue', lw=2.5, label="Joint shaft RPM")

    _motor_colors  = ['darkorange', 'forestgreen', 'orchid', 'saddlebrown']
    _motor_markers = ['o', 's', '^', 'D']
    _motor_ls      = ['--', '-.', (0,(5,2)), (0,(3,1,1,1))]
    for _ci, _mn in enumerate(MOTOR_CHOICES):
        _m   = get_motor(_mn)
        _clr = _motor_colors[_ci % len(_motor_colors)]
        _mk  = _motor_markers[_ci % len(_motor_markers)]
        _ls  = _motor_ls[_ci % len(_motor_ls)]
        _shaft_rpm = joint_rpm * _m.gear_ratio
        ax_rpm.plot(t_ms, _shaft_rpm,
                    color=_clr, lw=1.8, ls=_ls,
                    marker=_mk, markersize=4, markevery=mk,
                    label=f"Motor shaft RPM  {_mn}  (x{_m.gear_ratio:.0f})")
        # Motor no-load RPM limit (motor shaft)
        ax_rpm.axhline(_m.no_load_rpm_motor,
                        color=_clr, ls=(0, (8, 3)), lw=1.5,
                        label=f"No-load RPM limit  {_mn} = {_m.no_load_rpm_motor:.0f} RPM")
        # Joint RPM limit
        _jlim_rpm = _m.max_velocity_joint * 60.0 / (2.0 * math.pi)
        ax_rpm.axhline(_jlim_rpm,
                        color=_clr, ls=(0, (2, 2)), lw=1.2,
                        label=f"Joint RPM limit  {_mn} = {_jlim_rpm:.1f} RPM")

    ax_rpm.axhline(0, color='k', lw=0.5)
    ax_rpm.set_xlabel("Time [ms]")
    ax_rpm.set_ylabel("Velocity [RPM]")
    ax_rpm.legend(fontsize=8, loc='upper right')
    ax_rpm.grid(True, alpha=0.3)

    # ── Rows 2–4: per-motor columns ─────────────────────────────────────────────
    for col, motor_name in enumerate(MOTOR_CHOICES):
        res   = motor_results[motor_name]
        motor = res["motor"]
        color_pass = 'seagreen' if res["feasible"] else 'crimson'

        # ── Row 2: torque components breakdown ──────────────────────────────────
        ax_comp = fig.add_subplot(gs[2, col])
        ax_comp.set_title(f"{motor_name}\nTorque components", fontsize=9)
        # Each curve: unique (colour, linestyle, marker) triple
        ax_comp.plot(t_ms, res["tau_gravity"],
                     color='steelblue',  lw=1.5, ls=':',
                     marker='^', markersize=4, markevery=mk,
                     label="Gravity τ_g = 0 (XY plane)")
        ax_comp.plot(t_ms, res["tau_inertia"],
                     color='darkorange', lw=1.5, ls='--',
                     marker='o', markersize=4, markevery=mk,
                     label="Inertial τ_i [Nm]")
        ax_comp.plot(t_ms, res["tau_damping"],
                     color='orchid',     lw=1.5, ls='-.',
                     marker='s', markersize=4, markevery=mk,
                     label="Damping τ_d [Nm]")
        ax_comp.plot(t_ms, res["tau_total"],
                     color='crimson',    lw=2.5,
                     label="Total τ [Nm]")
        # Horizontal limit lines — distinct dash patterns + labels
        ax_comp.axhline( motor.peak_torque_joint,
                          color='crimson', ls=(0, (8, 3)), lw=1.5,
                          label=f"τ_peak = {motor.peak_torque_joint} Nm")
        ax_comp.axhline( motor.continuous_torque_joint,
                          color='tomato',  ls=(0, (3, 3)), lw=1.5,
                          label=f"τ_cont = {motor.continuous_torque_joint} Nm")
        ax_comp.axhline(-motor.peak_torque_joint,
                          color='crimson', ls=(0, (8, 3)), lw=1.5)
        ax_comp.axhline(0, color='k', lw=0.5)
        ax_comp.set_xlabel("Time [ms]")
        ax_comp.set_ylabel("Torque [Nm]")
        ax_comp.legend(fontsize=7, loc='upper right')
        ax_comp.grid(True, alpha=0.3)

        # ── Row 3: total required torque vs limits + verdict ─────────────────────
        ax_tot = fig.add_subplot(gs[3, col])
        verdict = "[FEASIBLE]" if res["feasible"] else "[NOT FEASIBLE]"
        ax_tot.set_title(f"Required torque vs limits   {verdict}", fontsize=9)

        ax_tot.fill_between(t_ms, res["tau_total"], alpha=0.18, color=color_pass)
        ax_tot.plot(t_ms, res["tau_total"],
                    color=color_pass, lw=2.5,
                    marker='D', markersize=4, markevery=mk,
                    label="τ_required")
        ax_tot.axhline( motor.peak_torque_joint,
                         color='crimson', ls=(0, (8, 3)), lw=1.8,
                         label=f"τ_peak = {motor.peak_torque_joint} Nm")
        ax_tot.axhline( motor.continuous_torque_joint,
                         color='tomato',  ls=(0, (3, 3)), lw=1.5,
                         label=f"τ_cont = {motor.continuous_torque_joint} Nm")
        ax_tot.axhline(-motor.peak_torque_joint,
                         color='crimson', ls=(0, (8, 3)), lw=1.8)
        ax_tot.axhline(0, color='k', lw=0.5)

        vel_margin    = (motor.max_velocity_joint - omega_peak) / motor.max_velocity_joint * 100
        torque_margin = (motor.peak_torque_joint  - res["peak_tau"]) / motor.peak_torque_joint * 100
        info = (
            f"ω_peak   = {omega_peak:.3f} rad/s\n"
            f"ω_limit  = {motor.max_velocity_joint:.3f} rad/s\n"
            f"vel margin = {vel_margin:+.1f}%\n\n"
            f"τ_peak   = {res['peak_tau']:.2f} Nm\n"
            f"τ_limit  = {motor.peak_torque_joint} Nm\n"
            f"τ margin  = {torque_margin:+.1f}%"
        )
        ax_tot.text(0.02, 0.97, info, transform=ax_tot.transAxes,
                    fontsize=7.5, va='top', family='monospace',
                    color='seagreen' if res["feasible"] else 'crimson',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax_tot.set_xlabel("Time [ms]")
        ax_tot.set_ylabel("Torque [Nm]")
        ax_tot.legend(fontsize=7.5, loc='upper right')
        ax_tot.grid(True, alpha=0.3)

        # ── Row 4: joint velocity vs torque (operating-point trajectory) ───────
        ax_vt = fig.add_subplot(gs[4, col])
        ax_vt.set_title(f"Operating trajectory — velocity vs torque", fontsize=9)

        # Colour-map the trajectory by normalised time (y-axis = joint RPM)
        tau_req   = res["tau_total"]
        omega_rpm = theta_dot * 60.0 / (2.0 * math.pi)   # joint shaft RPM
        norm = plt.Normalize(t_arr[0], t_arr[-1])
        pts  = np.array([tau_req, omega_rpm]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segs, cmap='plasma', norm=norm, lw=2)
        lc.set_array(t_arr[:-1])
        ax_vt.add_collection(lc)
        fig.colorbar(lc, ax=ax_vt, label="Time [s]", pad=0.02)

        # Motor limits: feasibility box — different dash patterns per axis
        tau_pk       = motor.peak_torque_joint
        omega_lim_rpm = motor.max_velocity_joint * 60.0 / (2.0 * math.pi)  # joint RPM limit
        ax_vt.axvline( tau_pk,  color='crimson',   ls=(0, (8, 3)), lw=1.8,
                        label=f"τ_peak = {tau_pk} Nm")
        ax_vt.axvline(-tau_pk,  color='crimson',   ls=(0, (8, 3)), lw=1.8)
        ax_vt.axhline( omega_lim_rpm, color='steelblue', ls=(0, (4, 2, 1, 2)), lw=1.8,
                        label=f"ω_max = {omega_lim_rpm:.1f} RPM")
        ax_vt.axhline(-omega_lim_rpm, color='steelblue', ls=(0, (4, 2, 1, 2)), lw=1.8)
        # Shade the feasible region
        ax_vt.fill_betweenx(
            [-omega_lim_rpm, omega_lim_rpm],
            -tau_pk, tau_pk,
            alpha=0.08, color='seagreen', label="Feasible region"
        )

        # Mark start and end operating points
        ax_vt.scatter([tau_req[0]],   [omega_rpm[0]],  s=60, zorder=5,
                       color='lime',   edgecolors='k', lw=0.8, label="Start")
        ax_vt.scatter([tau_req[-1]],  [omega_rpm[-1]], s=60, zorder=5,
                       color='yellow', edgecolors='k', lw=0.8, label="End")

        # Auto-scale view with a small margin
        tau_margin   = tau_pk        * 0.15
        rpm_margin   = omega_lim_rpm * 0.15
        ax_vt.set_xlim(-tau_pk        - tau_margin, tau_pk        + tau_margin)
        ax_vt.set_ylim(-omega_lim_rpm - rpm_margin, omega_lim_rpm + rpm_margin)

        ax_vt.set_xlabel("Torque [Nm]")
        ax_vt.set_ylabel("Joint velocity [RPM]")
        ax_vt.legend(fontsize=7, loc='upper right')
        ax_vt.grid(True, alpha=0.3)
        ax_vt.axhline(0, color='k', lw=0.5)
        ax_vt.axvline(0, color='k', lw=0.5)

    plt.show()
