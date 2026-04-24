"""A/B sweep that isolates the co-contraction effect.

Strategy:
  * Use a nearly-stationary trajectory so CT tracking is tight.
  * Apply a sustained external torque at t=4 s for 2 s.
  * Rank configs by % reduction in peak elbow error.
"""
from __future__ import annotations
import subprocess, os
from pathlib import Path

PY = "/opt/anaconda3/envs/pydrake/bin/python"
SCRIPT = Path(__file__).parent / "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py"

COMMON = [
    "--spring-stiffness", "30", "--cable-damping", "2.0", "--sea-mode", "torque",
    "--motor", "AK60_6_KV80_Config",
    "--ct-kp", "100", "--ct-kd", "40",
    "--no-meshcat", "--no-show",
    "--traj-x-range", "0.4995", "0.5005",
    "--traj-y-range", "-0.0005", "0.0005",
    "--traj-v-max", "0.02", "--traj-v-corner", "0.01",
    "--traj-n", "8",
    "--duration", "6", "--num-laps", "1",
    "--move-duration", "2.0",
    "--disturbance", "--disturbance-mode", "torque",
    "--disturbance-time", "4.0", "--disturbance-dur", "2.0",
]

HEADER = "mode k_exo dth tau_ext dur peak_ee rms_ee peak_q2 rms_q2 peak_texo peak_tdrive".split()


def run(extra):
    cmd = [PY, str(SCRIPT), *COMMON, *extra]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    for line in res.stdout.splitlines():
        if line.startswith("METRICS_CSV,"):
            vals = line.split(",")[1:]
            return dict(zip(HEADER, vals))
    print("FAILED:", " ".join(cmd))
    print(res.stdout[-1500:])
    print(res.stderr[-500:])
    return None


def trial(k_exo, dth, tau_ext, exo_on):
    extra = [
        "--exo-ks", str(k_exo),
        "--exo-delta-theta", f"{dth}",
        "--disturbance-tau", f"{tau_ext}",
    ]
    if exo_on:
        extra += ["--exo-activate", "--exo-activate-time", "3.5"]
    print(f"  run  k_exo={k_exo:5d}  dth={dth:.2f}  tau_ext={tau_ext:.1f}Nm  "
          f"exo={'ON ' if exo_on else 'OFF'} ...", end=" ", flush=True)
    m = run(extra)
    if m is None:
        print("FAILED")
        return None
    print(f"peak_ee={m['peak_ee']:>7s} mm | peak_q2={m['peak_q2']:>6s} deg | "
          f"|tauexo|={m['peak_texo']:>5s} | |taudrv|={m['peak_tdrive']:>5s}")
    return m


def main():
    grid = [
        (2000,  0.05,  0.5),
        (3500,  0.05,  0.5),
        (5000,  0.05,  0.5),
        (2000,  0.05,  1.0),
        (3500,  0.05,  1.0),
        (5000,  0.05,  1.0),
        (3500,  0.05,  1.5),
        (5000,  0.05,  1.5),
        (5000,  0.03,  1.5),
        (7000,  0.03,  1.5),
    ]
    rows = []
    for k_exo, dth, tau_ext in grid:
        print(f"\n-- k_exo={k_exo}  dth={dth}  tau_ext={tau_ext} Nm --")
        off = trial(k_exo, dth, tau_ext, exo_on=False)
        on  = trial(k_exo, dth, tau_ext, exo_on=True)
        if off is None or on is None:
            continue
        try:
            d_peak_q2 = float(off["peak_q2"]) - float(on["peak_q2"])
            d_peak_ee = float(off["peak_ee"]) - float(on["peak_ee"])
            d_rms_ee  = float(off["rms_ee"])  - float(on["rms_ee"])
            reduction_q2 = (d_peak_q2 / float(off["peak_q2"]) * 100.0
                            if float(off["peak_q2"]) > 1e-6 else 0.0)
            reduction_ee = (d_peak_ee / float(off["peak_ee"]) * 100.0
                            if float(off["peak_ee"]) > 1e-6 else 0.0)
        except ValueError:
            continue
        rows.append(dict(
            k_exo=k_exo, dth=dth, tau_ext=tau_ext,
            off_peak_q2=float(off["peak_q2"]), on_peak_q2=float(on["peak_q2"]),
            off_peak_ee=float(off["peak_ee"]), on_peak_ee=float(on["peak_ee"]),
            off_rms_ee=float(off["rms_ee"]),   on_rms_ee=float(on["rms_ee"]),
            d_peak_q2=d_peak_q2, d_peak_ee=d_peak_ee, d_rms_ee=d_rms_ee,
            reduction_q2=reduction_q2, reduction_ee=reduction_ee,
            peak_texo=float(on["peak_texo"]),
            peak_tdrive_on=float(on["peak_tdrive"]),
            peak_tdrive_off=float(off["peak_tdrive"]),
        ))

    if not rows:
        print("No successful runs.")
        return

    print("\n\n" + "=" * 110)
    print("RESULTS SORTED BY %-REDUCTION IN peak elbow error (bigger = exo helps more)")
    print("=" * 110)
    rows.sort(key=lambda r: -r["reduction_q2"])
    hdr = (f"{'k_exo':>6} {'dth':>5} {'tau':>5} | "
           f"{'q2 OFF':>8} {'q2 ON':>8} {'d_q2':>7} {'%red':>6} | "
           f"{'ee OFF':>8} {'ee ON':>8} {'d_ee':>7} {'%red':>6} | "
           f"{'|texo|':>6} {'|tdrv|':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sat = ""
        if r["peak_texo"] >= 8.9:  sat += " EXO_SAT"
        if max(r["peak_tdrive_on"], r["peak_tdrive_off"]) >= 8.9: sat += " DRV_SAT"
        print(
            f"{r['k_exo']:>6d} {r['dth']:>5.2f} {r['tau_ext']:>5.2f} | "
            f"{r['off_peak_q2']:>8.3f} {r['on_peak_q2']:>8.3f} {r['d_peak_q2']:>+7.3f} {r['reduction_q2']:>5.1f}% | "
            f"{r['off_peak_ee']:>8.2f} {r['on_peak_ee']:>8.2f} {r['d_peak_ee']:>+7.2f} {r['reduction_ee']:>5.1f}% | "
            f"{r['peak_texo']:>6.2f} {r['peak_tdrive_on']:>6.2f}{sat}"
        )

    best = rows[0]
    print("\n>>> BEST CONFIG (by %red_q2):")
    print(f"  --exo-ks {best['k_exo']}  --exo-delta-theta {best['dth']}  "
          f"--disturbance-tau {best['tau_ext']}")
    print(f"  elbow error reduced by {best['reduction_q2']:.1f}% "
          f"({best['off_peak_q2']:.2f} -> {best['on_peak_q2']:.2f} deg)")
    print(f"  EE error reduced by {best['reduction_ee']:.1f}% "
          f"({best['off_peak_ee']:.2f} -> {best['on_peak_ee']:.2f} mm)")


if __name__ == "__main__":
    main()
