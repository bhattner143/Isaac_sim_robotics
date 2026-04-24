"""Final focused A/B: quasi-static pose, push-and-hold, measure deflection."""
from __future__ import annotations
import subprocess, os
from pathlib import Path

PY = "/opt/anaconda3/envs/pydrake/bin/python"
SCRIPT = Path(__file__).parent / "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py"

# Very slow, tiny trajectory (quasi-static) + tight CT on AK80-8 drive.
COMMON = [
    "--spring-stiffness", "100", "--cable-damping", "4.0", "--sea-mode", "torque",
    "--motor", "AK80_8_KV60_Config",
    "--exo-motor", "AK80_8_KV60_Config",
    "--ct-kp", "400", "--ct-kd", "80",
    "--joint-damping", "0.2", "0.2",
    "--no-meshcat", "--no-show",
    # Tiny 2 mm rect at 2 cm/s — essentially stationary
    "--traj-x-range", "0.499", "0.501",
    "--traj-y-range", "-0.001", "0.001",
    "--traj-v-max", "0.02", "--traj-v-corner", "0.005",
    "--traj-n", "8",
    "--duration", "12", "--num-laps", "1",
    "--move-duration", "3.0",
    "--disturbance", "--disturbance-mode", "torque",
    "--disturbance-time", "8.0", "--disturbance-dur", "2.0",
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
        # Activate during preamble (well before disturbance at t=8)
        extra += ["--exo-activate", "--exo-activate-time", "4.0"]
    tag = "ON " if exo_on else "OFF"
    print(f"  k_exo={k_exo:5d} dth={dth:.2f} tau={tau_ext:.2f} exo={tag} ...",
          end=" ", flush=True)
    m = run(extra)
    if m is None:
        print("FAILED"); return None
    print(f"rms_q2={m['rms_q2']:>6s}°  peak_q2={m['peak_q2']:>6s}°  "
          f"rms_ee={m['rms_ee']:>6s}mm  |texo|={m['peak_texo']:>5s}Nm")
    return m


def main():
    grid = [
        # (k_exo,  dth,  tau_ext)
        (2000,  0.05, 1.0),
        (5000,  0.05, 1.0),
        (8000,  0.04, 1.0),
        (12000, 0.03, 1.0),
        (2000,  0.05, 2.0),
        (5000,  0.05, 2.0),
        (8000,  0.04, 2.0),
        (12000, 0.03, 2.0),
        (5000,  0.05, 3.0),
        (8000,  0.04, 3.0),
        (12000, 0.03, 3.0),
    ]
    rows = []
    for k_exo, dth, tau_ext in grid:
        print(f"\n-- k_exo={k_exo}  dth={dth}  tau={tau_ext} Nm --")
        off = trial(k_exo, dth, tau_ext, exo_on=False)
        on  = trial(k_exo, dth, tau_ext, exo_on=True)
        if off is None or on is None: continue
        # Use RMS over disturbance window — more robust than peak
        off_rq2, on_rq2 = float(off["rms_q2"]), float(on["rms_q2"])
        off_pq2, on_pq2 = float(off["peak_q2"]), float(on["peak_q2"])
        off_ree, on_ree = float(off["rms_ee"]),  float(on["rms_ee"])
        off_pee, on_pee = float(off["peak_ee"]), float(on["peak_ee"])
        red_rq2 = ((abs(off_rq2) - abs(on_rq2)) / abs(off_rq2) * 100.0
                   if abs(off_rq2) > 1e-6 else 0.0)
        red_pq2 = ((off_pq2 - on_pq2) / off_pq2 * 100.0 if off_pq2 > 1e-6 else 0.0)
        red_ree = ((off_ree - on_ree) / off_ree * 100.0 if off_ree > 1e-6 else 0.0)
        red_pee = ((off_pee - on_pee) / off_pee * 100.0 if off_pee > 1e-6 else 0.0)
        rows.append(dict(
            k_exo=k_exo, dth=dth, tau=tau_ext,
            off_rq2=off_rq2, on_rq2=on_rq2, red_rq2=red_rq2,
            off_pq2=off_pq2, on_pq2=on_pq2, red_pq2=red_pq2,
            off_ree=off_ree, on_ree=on_ree, red_ree=red_ree,
            off_pee=off_pee, on_pee=on_pee, red_pee=red_pee,
            peak_texo=float(on["peak_texo"]),
            peak_tdrv_on=float(on["peak_tdrive"]),
            peak_tdrv_off=float(off["peak_tdrive"]),
        ))

    if not rows: print("No successful runs."); return

    print("\n" + "=" * 120)
    print("SORTED BY %-reduction in RMS elbow error during disturbance window (bigger = exo helps more)")
    print("=" * 120)
    rows.sort(key=lambda r: -r["red_rq2"])
    hdr = (f"{'k_exo':>6} {'dth':>5} {'tau':>5} | "
           f"{'rms_q2 OFF':>10} {'rms_q2 ON':>10} {'%red':>5} | "
           f"{'peak_q2 OFF':>11} {'peak_q2 ON':>10} {'%red':>5} | "
           f"{'rms_ee OFF':>10} {'rms_ee ON':>10} {'%red':>5} | "
           f"{'|texo|':>6} {'|tdrv|':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sat = ""
        if r["peak_texo"]       >= 24.5: sat += " EXO_SAT"
        if max(r["peak_tdrv_on"], r["peak_tdrv_off"]) >= 24.5: sat += " DRV_SAT"
        print(f"{r['k_exo']:>6d} {r['dth']:>5.2f} {r['tau']:>5.2f} | "
              f"{r['off_rq2']:>10.3f} {r['on_rq2']:>10.3f} {r['red_rq2']:>4.1f}% | "
              f"{r['off_pq2']:>11.3f} {r['on_pq2']:>10.3f} {r['red_pq2']:>4.1f}% | "
              f"{r['off_ree']:>10.2f} {r['on_ree']:>10.2f} {r['red_ree']:>4.1f}% | "
              f"{r['peak_texo']:>6.2f} {r['peak_tdrv_on']:>6.2f}{sat}")

    best = rows[0]
    print(f"\n>>> BEST: --exo-ks {best['k_exo']} --exo-delta-theta {best['dth']} "
          f"--disturbance-tau {best['tau']}")
    print(f"    rms elbow error : {abs(best['off_rq2']):.3f} -> {abs(best['on_rq2']):.3f} deg "
          f"(reduced by {best['red_rq2']:.1f}%)")
    print(f"    peak elbow err  : {best['off_pq2']:.3f} -> {best['on_pq2']:.3f} deg "
          f"(reduced by {best['red_pq2']:.1f}%)")
    print(f"    rms EE error    : {best['off_ree']:.2f} -> {best['on_ree']:.2f} mm "
          f"(reduced by {best['red_ree']:.1f}%)")


if __name__ == "__main__":
    main()
