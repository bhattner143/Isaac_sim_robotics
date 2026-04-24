"""A/B sweep (v2) — isolate co-contraction with a folded pose + bigger motor."""
from __future__ import annotations
import subprocess, os
from pathlib import Path

PY = "/opt/anaconda3/envs/pydrake/bin/python"
SCRIPT = Path(__file__).parent / "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py"

# Folded pose (EE ~0.27 m out), AK80-8 drive (25 Nm peak), tighter CT.
COMMON = [
    "--spring-stiffness", "60", "--cable-damping", "3.0", "--sea-mode", "torque",
    "--motor", "AK80_8_KV60_Config",         # drive: 25 Nm peak
    "--exo-motor", "AK80_8_KV60_Config",     # exo:   25 Nm peak
    "--ct-kp", "400", "--ct-kd", "60",
    "--no-meshcat", "--no-show",
    # Near-stationary trajectory at a well-conditioned pose
    "--traj-x-range", "0.269", "0.271",
    "--traj-y-range", "-0.001", "0.001",
    "--traj-v-max", "0.02", "--traj-v-corner", "0.01",
    "--traj-n", "8",
    "--duration", "6", "--num-laps", "1",
    "--move-duration", "2.0",
    "--disturbance", "--disturbance-mode", "torque",
    "--disturbance-time", "4.0", "--disturbance-dur", "2.0",
]

HEADER = ("mode k_exo dth tau_ext dur peak_ee rms_ee peak_q2 rms_q2 "
          "peak_texo peak_tdrive").split()


def run(extra):
    cmd = [PY, str(SCRIPT), *COMMON, *extra]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    for line in res.stdout.splitlines():
        if line.startswith("METRICS_CSV,"):
            return dict(zip(HEADER, line.split(",")[1:]))
    print("FAILED:", " ".join(cmd))
    print(res.stdout[-1200:]); print(res.stderr[-500:])
    return None


def trial(k_exo, dth, tau_ext, exo_on):
    extra = [
        "--exo-ks", str(k_exo),
        "--exo-delta-theta", f"{dth}",
        "--disturbance-tau", f"{tau_ext}",
    ]
    if exo_on:
        extra += ["--exo-activate", "--exo-activate-time", "3.5"]
    tag = "ON " if exo_on else "OFF"
    print(f"  run k_exo={k_exo:5d} dth={dth:.2f} tau={tau_ext:.1f}Nm exo={tag} ...",
          end=" ", flush=True)
    m = run(extra)
    if m is None:
        print("FAILED"); return None
    print(f"peak_ee={m['peak_ee']:>7s}mm  peak_q2={m['peak_q2']:>6s}deg  "
          f"|texo|={m['peak_texo']:>5s}  |tdrv|={m['peak_tdrive']:>5s}")
    return m


def main():
    grid = [
        # (k_exo,  dth,  tau_ext [Nm])
        (2000,  0.05,  2.0),
        (5000,  0.05,  2.0),
        (8000,  0.05,  2.0),
        (5000,  0.05,  3.0),
        (8000,  0.05,  3.0),
        (12000, 0.04,  3.0),
        (8000,  0.05,  4.0),
        (12000, 0.04,  4.0),
        (15000, 0.03,  4.0),
        (20000, 0.03,  5.0),
    ]
    rows = []
    for k_exo, dth, tau_ext in grid:
        print(f"\n-- k_exo={k_exo}  dth={dth}  tau_ext={tau_ext} Nm --")
        off = trial(k_exo, dth, tau_ext, exo_on=False)
        on  = trial(k_exo, dth, tau_ext, exo_on=True)
        if off is None or on is None:
            continue
        off_pq2, on_pq2 = float(off["peak_q2"]), float(on["peak_q2"])
        off_pee, on_pee = float(off["peak_ee"]), float(on["peak_ee"])
        red_q2 = (off_pq2 - on_pq2) / off_pq2 * 100.0 if off_pq2 > 1e-6 else 0.0
        red_ee = (off_pee - on_pee) / off_pee * 100.0 if off_pee > 1e-6 else 0.0
        rows.append(dict(
            k_exo=k_exo, dth=dth, tau=tau_ext,
            off_pq2=off_pq2, on_pq2=on_pq2, red_q2=red_q2,
            off_pee=off_pee, on_pee=on_pee, red_ee=red_ee,
            peak_texo=float(on["peak_texo"]),
            peak_tdrv_on=float(on["peak_tdrive"]),
            peak_tdrv_off=float(off["peak_tdrive"]),
        ))

    if not rows:
        print("No successful runs."); return

    print("\n" + "=" * 108)
    print("SORTED BY %-reduction in peak elbow error (exo ON vs OFF)")
    print("=" * 108)
    rows.sort(key=lambda r: -r["red_q2"])
    hdr = (f"{'k_exo':>6} {'dth':>5} {'tau':>5} | "
           f"{'q2 OFF':>7} {'q2 ON':>7} {'%red':>6} | "
           f"{'ee OFF':>7} {'ee ON':>7} {'%red':>6} | "
           f"{'|texo|':>6} {'|tdrv|':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sat = ""
        if r["peak_texo"]       >= 24.5: sat += " EXO_SAT"
        if max(r["peak_tdrv_on"], r["peak_tdrv_off"]) >= 24.5: sat += " DRV_SAT"
        print(f"{r['k_exo']:>6d} {r['dth']:>5.2f} {r['tau']:>5.2f} | "
              f"{r['off_pq2']:>7.2f} {r['on_pq2']:>7.2f} {r['red_q2']:>5.1f}% | "
              f"{r['off_pee']:>7.2f} {r['on_pee']:>7.2f} {r['red_ee']:>5.1f}% | "
              f"{r['peak_texo']:>6.2f} {r['peak_tdrv_on']:>6.2f}{sat}")

    best = rows[0]
    print(f"\n>>> BEST: --exo-ks {best['k_exo']} --exo-delta-theta {best['dth']} "
          f"--disturbance-tau {best['tau']}")
    print(f"    elbow error reduced {best['red_q2']:.1f}%  "
          f"({best['off_pq2']:.2f} -> {best['on_pq2']:.2f} deg)")
    print(f"    EE    error reduced {best['red_ee']:.1f}%  "
          f"({best['off_pee']:.2f} -> {best['on_pee']:.2f} mm)")


if __name__ == "__main__":
    main()
