"""A/B sweep (v3) — uses known-good trajectory, small disturbance, narrow window.

Key changes vs v2:
  * Known-good EE box near (0.5, 0) so arm actually tracks.
  * Small sustained torque (so q2 stays far from joint stops).
  * Longer duration & lap so we start the disturbance during steady tracking.
  * AK80-8 on drive for torque headroom.
"""
from __future__ import annotations
import subprocess, os
from pathlib import Path

PY = "/opt/anaconda3/envs/pydrake/bin/python"
SCRIPT = Path(__file__).parent / "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py"

COMMON = [
    "--spring-stiffness", "60", "--cable-damping", "3.0", "--sea-mode", "torque",
    "--motor", "AK80_8_KV60_Config",          # drive: 25 Nm peak
    "--exo-motor", "AK80_8_KV60_Config",      # exo:   25 Nm peak
    "--ct-kp", "200", "--ct-kd", "50",
    "--no-meshcat", "--no-show",
    # Default known-good trajectory (rect -0.08 to 0.08 m at x=0.50)
    "--traj-x-range", "0.49", "0.51",
    "--traj-y-range", "-0.08", "0.08",
    "--traj-v-max", "0.9", "--traj-v-corner", "0.05",
    "--traj-n", "60",
    "--duration", "10", "--num-laps", "1",
    "--move-duration", "3.0",
    "--disturbance", "--disturbance-mode", "torque",
    # Push at t=7s for 1.5s; tracking is settled well before.
    "--disturbance-time", "7.0", "--disturbance-dur", "1.5",
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
        # Activate exo VERY early so symmetric pretension settles well before
        # the disturbance and doesn't add its own transient to the window.
        extra += ["--exo-activate", "--exo-activate-time", "0.5"]
    tag = "ON " if exo_on else "OFF"
    print(f"  k_exo={k_exo:5d} dth={dth:.2f} tau={tau_ext:.2f} exo={tag} ...",
          end=" ", flush=True)
    m = run(extra)
    if m is None:
        print("FAILED"); return None
    print(f"peak_ee={m['peak_ee']:>7s}mm  peak_q2={m['peak_q2']:>6s}deg  "
          f"|texo|={m['peak_texo']:>5s}  |tdrv|={m['peak_tdrive']:>5s}")
    return m


def main():
    # Small tau_ext so joint doesn't saturate.  Sweep k_exo widely.
    grid = [
        # (k_exo,  dth,  tau_ext [Nm])
        (1000,  0.05, 0.30),
        (3000,  0.05, 0.30),
        (6000,  0.05, 0.30),
        (10000, 0.04, 0.30),
        (1000,  0.05, 0.50),
        (3000,  0.05, 0.50),
        (6000,  0.05, 0.50),
        (10000, 0.04, 0.50),
        (15000, 0.03, 0.50),
        (6000,  0.05, 0.80),
        (10000, 0.04, 0.80),
    ]
    rows = []
    for k_exo, dth, tau_ext in grid:
        print(f"\n-- k_exo={k_exo}  dth={dth}  tau={tau_ext} Nm --")
        off = trial(k_exo, dth, tau_ext, exo_on=False)
        on  = trial(k_exo, dth, tau_ext, exo_on=True)
        if off is None or on is None: continue
        off_pq2, on_pq2 = float(off["peak_q2"]), float(on["peak_q2"])
        off_rq2, on_rq2 = float(off["rms_q2"]), float(on["rms_q2"])
        off_pee, on_pee = float(off["peak_ee"]), float(on["peak_ee"])
        off_ree, on_ree = float(off["rms_ee"]),  float(on["rms_ee"])
        red_pq2 = (off_pq2 - on_pq2) / off_pq2 * 100.0 if off_pq2 > 1e-6 else 0.0
        red_pee = (off_pee - on_pee) / off_pee * 100.0 if off_pee > 1e-6 else 0.0
        red_ree = (off_ree - on_ree) / off_ree * 100.0 if off_ree > 1e-6 else 0.0
        rows.append(dict(
            k_exo=k_exo, dth=dth, tau=tau_ext,
            off_pq2=off_pq2, on_pq2=on_pq2, red_pq2=red_pq2,
            off_pee=off_pee, on_pee=on_pee, red_pee=red_pee,
            off_ree=off_ree, on_ree=on_ree, red_ree=red_ree,
            peak_texo=float(on["peak_texo"]),
            peak_tdrv_on=float(on["peak_tdrive"]),
            peak_tdrv_off=float(off["peak_tdrive"]),
        ))

    if not rows:
        print("No successful runs."); return

    print("\n" + "=" * 120)
    print("SORTED BY %-reduction in peak elbow error during disturbance window (larger = exo helps more)")
    print("=" * 120)
    rows.sort(key=lambda r: -r["red_pq2"])
    hdr = (f"{'k_exo':>6} {'dth':>5} {'tau':>5} | "
           f"{'q2 OFF':>7} {'q2 ON':>7} {'%pq2':>6} {'%rq2':>6} | "
           f"{'ee OFF':>7} {'ee ON':>7} {'%pee':>6} {'%ree':>6} | "
           f"{'|texo|':>6} {'|tdrv|':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sat = ""
        if r["peak_texo"]       >= 24.5: sat += " EXO_SAT"
        if max(r["peak_tdrv_on"], r["peak_tdrv_off"]) >= 24.5: sat += " DRV_SAT"
        _rq2 = (r["off_pq2"] - r["on_pq2"]) * 0  # placeholder
        red_ree = r["red_ree"]
        # second %-col = rms_q2 change
        rq2_off, rq2_on = None, None  # not stored, omit
        print(f"{r['k_exo']:>6d} {r['dth']:>5.2f} {r['tau']:>5.2f} | "
              f"{r['off_pq2']:>7.3f} {r['on_pq2']:>7.3f} {r['red_pq2']:>5.1f}% {'    - ':>6} | "
              f"{r['off_pee']:>7.2f} {r['on_pee']:>7.2f} {r['red_pee']:>5.1f}% {red_ree:>5.1f}% | "
              f"{r['peak_texo']:>6.2f} {r['peak_tdrv_on']:>6.2f}{sat}")

    best = rows[0]
    print(f"\n>>> BEST: --exo-ks {best['k_exo']} --exo-delta-theta {best['dth']} "
          f"--disturbance-tau {best['tau']}")
    print(f"    peak elbow error : {best['off_pq2']:.3f} -> {best['on_pq2']:.3f} deg  ({best['red_pq2']:+.1f}%)")
    print(f"    peak EE error    : {best['off_pee']:.2f} -> {best['on_pee']:.2f} mm   ({best['red_pee']:+.1f}%)")
    print(f"    RMS  EE error    : {best['off_ree']:.2f} -> {best['on_ree']:.2f} mm   ({best['red_ree']:+.1f}%)")


if __name__ == "__main__":
    main()
