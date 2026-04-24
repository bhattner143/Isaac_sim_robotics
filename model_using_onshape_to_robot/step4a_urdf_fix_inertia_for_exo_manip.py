#!/usr/bin/env python3
"""step4a_urdf_fix_inertia_for_exo_manip.py — Patch exosuit cable-manipulator URDF with realistic inertial values.

Onshape-to-robot assigns masses from solid CAD bodies (default material density),
leading to large overestimates.  The exosuit variant is *worse* than the original
because Link 1 now carries extra exo motors, gearboxes, a motor mount, an elbow
pulley, spring-cable pulleys and a shoulder bracket:

  Onshape (raw)                          Realistic estimate
  ─────────────────────────────────────  ────────────────────
  pulley_htd_5m_60t   →  6.49 kg  ✗     ~2.90 kg  (drive motor + 2 exo motors
                                                     + 4 gearboxes + Al arm + misc)
  link2_tendon         →  0.57 kg  ~OK   ~0.55 kg  (Al link + cup + cable balls)
  base_mate            →  0.90 kg  OK    ~0.90 kg  (fixed base — no dynamics effect)

Extra parts on Link 1 compared to the base manipulator_cable URDF:
  • 2 × exo motor  (~0.20 kg each)
  • 3 × extra gearbox_15_ratio  (~0.15 kg each)
  • mount_two_motors bracket  (~0.15 kg)
  • shoulder_bracker  (~0.10 kg)
  • exo_elbow_pulley_big  (~0.08 kg)
  • 2 × link1_spring_cable_pulley  (~0.05 kg each)
  • link1_base_pulley  (~0.05 kg)
  • misc (balls, tutup_dgn_mounting, second HTD-60t)  (~0.15 kg)
  ≈ +1.40 kg over the base manipulator_cable Link 1  (1.50 + 1.40 = 2.90 kg)

This script replaces each <inertial> block with values derived from:
  • Realistic mass estimates (itemised above)
  • COM placed accounting for the heavy proximal cluster (motors/gearboxes near shoulder)
  • Diagonal inertia tensor computed from uniform-rod + concentrated-mass model
  • Off-diagonal elements set to zero (cross-products ≈ 0 for symmetric arm)

Geometry reference (from URDF joint origins — identical to base manipulator):
  L1 = 0.3348 m   link1_base origin → link2_link1 origin  (x-component = arm reach)
  L2 = 0.1900 m   link2_link1 origin → cup EE visual  (x-component)
  W  = 0.050  m   typical arm cross-section width
  H  = 0.040  m   typical arm cross-section height

Usage
-----
  # Preview changes without writing anything:
  python step4a_urdf_fix_inertia_for_exo_manip.py --action dry-run

  # Patch in-place (auto-backup to .back unless one already exists):
  python step4a_urdf_fix_inertia_for_exo_manip.py

  # Patch to a different output file:
  python step4a_urdf_fix_inertia_for_exo_manip.py --out patched.urdf

  # Restore original from auto-backup:
  python step4a_urdf_fix_inertia_for_exo_manip.py --action restore

  # Custom URDF path:
  python step4a_urdf_fix_inertia_for_exo_manip.py --urdf path/to/robot.urdf

  # Use a different mass preset:
  python step4a_urdf_fix_inertia_for_exo_manip.py --preset heavy
"""

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Default URDF path (relative to workspace root — run from Isaac_sim_robotics/)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_URDF = Path(
    "model_using_onshape_to_robot/"
    "manipulator_cable_exo_springs_elbow_follow/"
    "manipulator_cable_exo_springs_elbow_follow_obj.urdf"
)

# ─────────────────────────────────────────────────────────────────────────────
# Arm geometry  (from URDF joint origin x-components — same kinematic chain
# as the base manipulator_cable; the exo hardware does not change link lengths)
# ─────────────────────────────────────────────────────────────────────────────
L1 = 0.3348   # Joint-1 → Joint-2 reach  [m]
L2 = 0.1900   # Joint-2 → EE (cup) reach [m]
W  = 0.050    # arm cross-section width   [m]  (Y direction)
H  = 0.040    # arm cross-section height  [m]  (Z direction)


# ─────────────────────────────────────────────────────────────────────────────
# Inertia helper functions  (uniform rectangular rod along local X)
# ─────────────────────────────────────────────────────────────────────────────

def _ixx(m, W, H):
    """Ixx — rotation about X (axial): (1/12)*m*(W² + H²)"""
    return m / 12.0 * (W**2 + H**2)

def _iyy(m, L, H):
    """Iyy — rotation about Y:  (1/12)*m*(L² + H²)"""
    return m / 12.0 * (L**2 + H**2)

def _izz(m, L, W):
    """Izz — rotation about Z (in-plane, key for SCARA): (1/12)*m*(L² + W²)"""
    return m / 12.0 * (L**2 + W**2)


# ─────────────────────────────────────────────────────────────────────────────
# Mass presets  (three scenarios selectable via --preset)
# ─────────────────────────────────────────────────────────────────────────────
#
# 'realistic'  — best estimate for the actual exo robot
# 'light'      — lightweight build (carbon-fibre arm, smaller exo motors)
# 'heavy'      — conservative upper-bound (steel arm, larger motors)
#
# All values in SI: kg, m, kg·m²
#
# Link 1 mass breakdown (realistic):
#   drive motor         0.40 kg
#   drive gearbox       0.45 kg
#   Al arm (link1)      0.35 kg
#   drive pulleys/misc  0.30 kg
#   ── subtotal (base manipulator) ──  1.50 kg
#   2 × exo motor      0.40 kg   (0.20 each)
#   3 × exo gearbox    0.45 kg   (0.15 each)
#   mount_two_motors    0.15 kg
#   shoulder_bracker    0.10 kg
#   exo_elbow_pulley    0.08 kg
#   2 × spring pulleys  0.10 kg  (0.05 each)
#   link1_base_pulley   0.05 kg
#   misc (balls, tutup) 0.07 kg
#   ── subtotal (exo additions) ──    1.40 kg
#   ── TOTAL ────────────────────────  2.90 kg
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {

    # ── realistic (default) ──────────────────────────────────────────────────
    "realistic": {
        "link1_mass":  2.90,   # kg  (vs. Onshape 6.49 — 2.2× over-estimate)
        "link2_mass":  0.55,   # kg  (vs. Onshape 0.57 — already close)
        "link1_com_x": 0.13,   # m   (shifted toward shoulder vs L1/2=0.167
                                #      because exo motors/gearboxes cluster there)
        "link2_com_x": L2 / 2.0,
        # proximal cluster: drive motor+gearbox + 2 exo motors + 3 exo gearboxes
        # + motor mount + shoulder bracket + spring pulleys near Joint-1
        "m1_proximal": 2.15,   # kg — everything near shoulder
        "r1_proximal": abs(0.13 - 0.04),  # COM to cluster centre [m]
        "m2_tip":      0.15,   # kg — cup + retained liquid at EE tip
    },

    # ── light (carbon-fibre arm / smaller exo motors) ────────────────────────
    "light": {
        "link1_mass":  1.80,
        "link2_mass":  0.35,
        "link1_com_x": 0.12,
        "link2_com_x": L2 / 2.0,
        "m1_proximal": 1.30,
        "r1_proximal": abs(0.12 - 0.04),
        "m2_tip":      0.10,
    },

    # ── heavy (steel arm / larger exo servos) ────────────────────────────────
    "heavy": {
        "link1_mass":  4.20,
        "link2_mass":  0.80,
        "link1_com_x": 0.14,
        "link2_com_x": L2 / 2.0,
        "m1_proximal": 3.20,
        "r1_proximal": abs(0.14 - 0.05),
        "m2_tip":      0.20,
    },
}


def _build_patches(preset_name: str) -> dict[str, dict]:
    """Compute the PATCHES dict from the given preset name."""
    p = PRESETS[preset_name]

    # ── base_mate  (fixed to world — does not appear in any joint's tau) ─────
    BASE = dict(
        mass = 0.90,
        com  = (0.0, -0.055, 0.025),
        Ixx  = 0.0025,
        Iyy  = 0.0015,
        Izz  = 0.0030,
    )

    # ── Link 1  (pulley_htd_5m_60t) ─────────────────────────────────────────
    m1       = p["link1_mass"]
    cx1      = p["link1_com_x"]
    m1_prox  = p["m1_proximal"]
    r1_prox  = p["r1_proximal"]
    m1_arm   = m1 - m1_prox
    Izz1_rod     = _izz(max(m1_arm, 0.0), L1, W)
    Izz1_cluster = m1_prox * r1_prox ** 2
    LNK1 = dict(
        mass = m1,
        com  = (cx1, 0.0, 0.0),
        Ixx  = _ixx(m1, W, H),
        Iyy  = _iyy(m1, L1, H) + Izz1_cluster,
        Izz  = Izz1_rod + Izz1_cluster,
    )

    # ── Link 2  (link2_tendon) ───────────────────────────────────────────────
    m2       = p["link2_mass"]
    cx2      = p["link2_com_x"]
    W2, H2   = W * 0.7, H * 0.7
    m2_tip   = p["m2_tip"]
    r2_tip   = L2 / 2.0
    Izz2_cup = m2_tip * r2_tip ** 2
    LNK2 = dict(
        mass = m2,
        com  = (cx2, 0.0, 0.0),
        Ixx  = _ixx(m2, W2, H2),
        Iyy  = _iyy(m2, L2, H2),
        Izz  = _izz(m2, L2, W2) + Izz2_cup,
    )

    return {
        "base_mate":         BASE,
        "pulley_htd_5m_60t": LNK1,
        "link2_tendon":      LNK2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# URDF text-level patching  (preserves visual/collision/comments exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format float compactly for URDF (7 significant digits)."""
    return f"{v:.7g}"


def _build_inertial_xml(p: dict, indent: str = "    ") -> str:
    """Return a clean <inertial>...</inertial> XML snippet."""
    i2 = indent + "  "
    cx, cy, cz = p["com"]
    return (
        f"{indent}<inertial>\n"
        f"{i2}<origin xyz=\"{_fmt(cx)} {_fmt(cy)} {_fmt(cz)}\" rpy=\"0 0 0\"/>\n"
        f"{i2}<mass value=\"{_fmt(p['mass'])}\"/>\n"
        f"{i2}<inertia"
        f" ixx=\"{_fmt(p['Ixx'])}\" ixy=\"0\" ixz=\"0\""
        f" iyy=\"{_fmt(p['Iyy'])}\" iyz=\"0\""
        f" izz=\"{_fmt(p['Izz'])}\"/>\n"
        f"{indent}</inertial>"
    )


_INERTIAL_RE = re.compile(r"<inertial>.*?</inertial>", re.DOTALL)


def patch_text(text: str, patches: dict, dry_run: bool = False) -> tuple[str, list[dict]]:
    """Replace <inertial> block for each target link in the URDF text.

    Returns (patched_text, report_rows).
    """
    report: list[dict] = []

    for link_name, p in patches.items():
        link_open = f'<link name="{link_name}">'
        link_start = text.find(link_open)
        if link_start == -1:
            print(f"  Warning: Link '{link_name}' not found — skipped.")
            continue

        link_end_tag = "</link>"
        link_end = text.find(link_end_tag, link_start)
        if link_end == -1:
            print(f"  Warning: Missing </link> for '{link_name}' — skipped.")
            continue
        link_end += len(link_end_tag)
        link_block = text[link_start:link_end]

        # Collect 'before' stats via ET
        try:
            link_el    = ET.fromstring(link_block)
            ine_el     = link_el.find("inertial")
            m_before   = float(ine_el.find("mass").get("value")) if ine_el is not None else 0.0
            izz_before = float(ine_el.find("inertia").get("izz")) if ine_el is not None else 0.0
            com_before = ine_el.find("origin").get("xyz", "?") if ine_el is not None else "?"
        except Exception:
            m_before = izz_before = 0.0
            com_before = "?"

        report.append(dict(
            link       = link_name,
            m_before   = m_before,   m_after  = p["mass"],
            izz_before = izz_before, izz_after= p["Izz"],
            com_before = com_before,
            com_after  = " ".join(_fmt(v) for v in p["com"]),
        ))

        if dry_run:
            continue

        # Detect indentation
        m = _INERTIAL_RE.search(link_block)
        indent = "    "
        if m:
            prefix = link_block[: m.start()]
            nl = prefix.rfind("\n")
            candidate = prefix[nl + 1:] if nl != -1 else prefix
            if candidate and not candidate.strip():
                indent = candidate

        new_inertial  = _build_inertial_xml(p, indent=indent)
        new_block     = _INERTIAL_RE.sub(new_inertial, link_block, count=1)
        text          = text[:link_start] + new_block + text[link_end:]

    return text, report


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(rows: list[dict], kp_values: list[float], tau_max: float) -> None:
    W80 = "─" * 80
    print(f"\n{W80}")
    print(f"  {'Link':<26}  {'Mass  before → after':<26}  Izz  before → after")
    print(W80)
    for r in rows:
        ms = f"{r['m_before']:.3f} → {r['m_after']:.3f} kg"
        izs = f"{r['izz_before']:.5f} → {r['izz_after']:.5f} kg·m²"
        print(f"  {r['link']:<26}  {ms:<26}  {izs}")
        print(f"  {'':26}  COM: {r['com_before']}  →  {r['com_after']}")
    print(W80)

    # ── Approximate effective inertia at each joint (horizontal SCARA, q=0)
    def _row(name, key):
        for r in rows:
            if r["link"] == name:
                return r[key]
        return 0.0

    # After-patch values
    m1n   = _row("pulley_htd_5m_60t", "m_after");   Iz1n = _row("pulley_htd_5m_60t", "izz_after")
    m2n   = _row("link2_tendon",      "m_after");   Iz2n = _row("link2_tendon",      "izz_after")
    cx1n  = float(_row("pulley_htd_5m_60t", "com_after").split()[0])
    cx2n  = float(_row("link2_tendon",      "com_after").split()[0])
    M11 = (Iz1n + m1n * cx1n ** 2) + (Iz2n + m2n * (L1 + cx2n) ** 2)
    M22 =  Iz2n + m2n * cx2n ** 2

    # Before-patch values (original Onshape COMs from the exo URDF)
    m1o, Iz1o = _row("pulley_htd_5m_60t", "m_before"), _row("pulley_htd_5m_60t", "izz_before")
    m2o, Iz2o = _row("link2_tendon",      "m_before"), _row("link2_tendon",      "izz_before")
    cx1o, cx2o = 0.0425, 0.1079   # x-component of Onshape COM for this URDF
    M11o = (Iz1o + m1o * cx1o ** 2) + (Iz2o + m2o * (L1 + cx2o) ** 2)

    print(f"\n  Effective inertia @ Joint-1 (SCARA, q=0):  {M11o:.4f} → {M11:.4f} kg·m²  "
          f"({M11o/M11:.2f}× reduction)")
    print(f"  Effective inertia @ Joint-2 (q=0):          — → {M22:.4f} kg·m²")
    print()
    for Kp in kp_values:
        dq = 0.0873  # 5 degrees in radians
        t_old = M11o * Kp * dq
        t_new = M11 * Kp * dq
        status = "OK" if t_new < tau_max else "SATURATES"
        print(f"  tau_1 for 5 deg error (Kp={Kp:.0f}):  {t_old:.2f} -> {t_new:.2f} Nm"
              f"  (limit={tau_max} Nm, {status})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--action",
        choices=["patch", "dry-run", "restore"],
        default="patch",
        metavar="ACTION",
        help=(
            "What to do (default: patch):\n"
            "  patch    — apply inertia fixes in-place (auto-backup as .back)\n"
            "  dry-run  — print what would change, write nothing\n"
            "  restore  — restore original from the .back backup file"
        ),
    )
    ap.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="realistic",
        metavar="PRESET",
        help=(
            "Mass/inertia scenario (default: realistic):\n"
            "  realistic — drive motor + 2 exo motors + 4 gearboxes + Al arm  L1=2.90 kg  L2=0.55 kg\n"
            "  light     — CF arm + smaller exo motors  L1=1.80 kg  L2=0.35 kg\n"
            "  heavy     — steel arm + larger exo servos L1=4.20 kg  L2=0.80 kg"
        ),
    )
    ap.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF),
        metavar="PATH",
        help=f"Path to the URDF file to patch  (default: {DEFAULT_URDF})",
    )
    ap.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path for the patched URDF  (default: overwrite --urdf in-place)",
    )
    ap.add_argument(
        "--kp",
        type=float,
        nargs="+",
        default=[100.0, 400.0],
        metavar="KP",
        help="Kp gain(s) for the torque-saturation report  (default: 100 400)",
    )
    ap.add_argument(
        "--tau-max",
        type=float,
        default=10.0,
        metavar="NM",
        help="Joint torque saturation limit used in report [Nm]  (default: 10.0)",
    )

    args = ap.parse_args()

    urdf_path = Path(args.urdf)
    back_path = urdf_path.with_suffix(urdf_path.suffix + ".back")
    patches   = _build_patches(args.preset)

    # ── Restore ───────────────────────────────────────────────────────────────
    if args.action == "restore":
        if not back_path.exists():
            print(f"  No backup found at {back_path}")
            return 1
        shutil.copy2(back_path, urdf_path)
        print(f"  Restored {urdf_path.name} from {back_path.name}")
        return 0

    # ── Check input ───────────────────────────────────────────────────────────
    if not urdf_path.exists():
        print(f"  URDF not found: {urdf_path}")
        return 1

    dry_run = args.action == "dry-run"
    text    = urdf_path.read_text()
    patched, report = patch_text(text, patches=patches, dry_run=dry_run)

    print(f"\n  Preset: {args.preset}")
    _print_report(report, kp_values=args.kp, tau_max=args.tau_max)

    if dry_run:
        print("  [dry-run] No files written.\n")
        return 0

    # ── Backup ────────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else urdf_path
    if out_path == urdf_path and not back_path.exists():
        shutil.copy2(urdf_path, back_path)
        print(f"  Backup saved -> {back_path.name}")
    elif out_path == urdf_path and back_path.exists():
        print(f"  Backup already exists at {back_path.name} — not overwriting.")

    out_path.write_text(patched)
    print(f"  Patched URDF written -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
