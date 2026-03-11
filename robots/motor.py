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
    def max_velocity_joint(self) -> float:
        """No-load maximum joint velocity  [rad/s]."""

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

    # ── Electrical ────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def winding_resistance(self) -> float:
        """Phase winding resistance  [Ω]."""

    @property
    @abstractmethod
    def winding_inductance(self) -> float:
        """Phase winding inductance  [H]."""

    @property
    def electrical_time_constant(self) -> float:
        """Electrical time constant  τ_e = L / R  [s]."""
        return self.winding_inductance / self.winding_resistance

    # ── Physical ──────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def mass(self) -> float:
        """Total actuator mass (motor + gearbox)  [kg]."""


# ── Concrete motor models ──────────────────────────────────────────────────────

@motor_choice
@dataclass(frozen=True)
class AK80_8_KV60_Config(MotorModelConfig):
    """CubeMars AK80-8 V3.0  KV60 quasi-direct-drive actuator.

    Specs (joint side, 8:1 planetary gearbox):

    =====================  ========  =====
    Parameter              Value     Unit
    =====================  ========  =====
    Gear ratio             8         —
    Peak torque            18        Nm
    Continuous torque       9        Nm
    Max velocity           ~4.19     rad/s  (40 rpm no-load at joint)
    Reflected inertia      0.008     kg·m²  (N²×0.000125)
    Viscous damping        0.18      Nm·s/rad
    Winding resistance     0.186     Ω
    Winding inductance     57        μH  → τ_e ≈ 0.31 ms
    Mass                   0.485     kg
    =====================  ========  =====
    """

    @property
    def gear_ratio(self) -> float:              return 8.0
    @property
    def peak_torque_joint(self) -> float:       return 18.0           # Nm
    @property
    def continuous_torque_joint(self) -> float: return 9.0            # Nm
    @property
    def max_velocity_joint(self) -> float:      return 40.0 * (2 * 3.14159265 / 60)  # 40 rpm → rad/s
    @property
    def rotor_inertia_motor(self) -> float:     return 0.000125       # kg·m²
    @property
    def viscous_damping_joint(self) -> float:   return 0.18           # Nm·s/rad
    @property
    def winding_resistance(self) -> float:      return 0.186          # Ω
    @property
    def winding_inductance(self) -> float:      return 57e-6          # H
    @property
    def mass(self) -> float:                    return 0.485          # kg


# Convenience list of all registered motor class names.
# Use as:  parser.add_argument("--motor", choices=MOTOR_CHOICES, default="AK80_8_KV60_Config")
MOTOR_CHOICES: List[str] = list(_MOTOR_REGISTRY)


if __name__ == "__main__":
    import math

    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"

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
        print(f"  {'Max velocity (joint)':<30} {m.max_velocity_joint:.4f} rad/s"
              f"  ({m.max_velocity_joint * 60 / (2*math.pi):.1f} rpm)")
        print(f"  {'Rotor inertia (motor side)':<30} {m.rotor_inertia_motor*1e6:.1f} g·cm²"
              f"  ({m.rotor_inertia_motor:.6f} kg·m²)")
        print(f"  {'Reflected inertia (joint)':<30} {m.rotor_inertia_joint:.5f} kg·m²"
              f"  (= N²·I_motor = {m.gear_ratio}²·{m.rotor_inertia_motor})")
        print(f"  {'Viscous damping (joint)':<30} {m.viscous_damping_joint} Nm·s/rad")
        print(f"  {'Winding resistance':<30} {m.winding_resistance} Ω")
        print(f"  {'Winding inductance':<30} {m.winding_inductance*1e6:.1f} μH")
        print(f"  {'Electrical time constant':<30} {m.electrical_time_constant*1e3:.3f} ms"
              f"  (τ_e = L/R)")
        print(f"  {'Mass':<30} {m.mass} kg")
        print()

    # ── Registry lookup by string (simulates CLI --motor flag) ────────────────
    print(f"{CYAN}get_motor() round-trip test:{RESET}")
    for name in MOTOR_CHOICES:
        m = get_motor(name)
        print(f"  get_motor('{name}') → {type(m).__name__}  ✓")

    # ── Bad key raises KeyError ────────────────────────────────────────────────
    print(f"\n{CYAN}KeyError on unknown motor:{RESET}")
    try:
        get_motor("NonExistentMotor")
    except KeyError as e:
        print(f"  Caught expected KeyError: {e}")

    print(f"\n{GREEN}All checks passed.{RESET}\n")
