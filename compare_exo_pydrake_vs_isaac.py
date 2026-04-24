#!/usr/bin/env python3
"""
compare_exo_pydrake_vs_isaac.py
───────────────────────────────
Side-by-side overlay comparison of

  * ``script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py``
  * ``script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py``

Given two ``--log-npz`` dumps (one from PyDrake, one from Isaac Sim, both
produced with matching CLI parameters), this tool renders a 2×2 overlay
plot of:

  * EE XY path (reference, PyDrake actual, Isaac actual)
  * Elbow angle q₂ vs time with desired reference
  * Elbow tracking error q₂ − q₂_des vs time
  * Exo torque τ_exo vs time

Usage
~~~~~
::

    # 1.  Run both simulations with --log-npz pointing at parallel paths
    python script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py \
        --traj-type circle --traj-radius 0.05 --duration 5 --num-laps 1 \
        --spring-stiffness 200 --cable-damping 8 \
        --ct-kp 400 --ct-kd 80 --joint-damping 0.3 0.3 \
        --exo-ks 8000 --exo-delta-theta 0.1 \
        --exo-activate --exo-activate-time 2 \
        --no-meshcat --no-show \
        --log-npz plots/exo_pydrake_circle_on.npz

    python script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py \
        --render headless \
        --traj-shape circle --traj-cx 0.5 --traj-cy 0.0 --traj-radius 0.05 \
        --duration 5 --num-laps 1 \
        --spring-stiffness 200 --cable-damping 8 \
        --ct-kp 400 --ct-kd 80 --joint-damping 0.3 0.3 \
        --exo-ks 8000 --exo-delta-theta 0.1 \
        --exo-activate --exo-activate-time 2 \
        --no-show \
        --log-npz plots/exo_isaac_circle_on.npz

    # 2.  Generate overlay plot
    python compare_exo_pydrake_vs_isaac.py \
        --pydrake plots/exo_pydrake_circle_on.npz \
        --isaac   plots/exo_isaac_circle_on.npz \
        --out     plots/exo_pydrake_vs_isaac_circle.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str):
    return np.load(path, allow_pickle=True)


def _pydrake_extract(npz):
    """Canonicalise the PyDrake log dict to a minimal comparison schema.

    PyDrake log layout (``run_simulation`` return value):
      * ``t``          (N,)
      * ``state``      (nq+nv, N)   — Drake joint order: q = [q2, q1]
      * ``q_des``      (2, N)       — user order [q1_des, q2_des]
      * ``ref``        (6, N)       — [x, y, ẋ, ẏ, ẍ, ÿ]
      * ``ee_x``/``ee_y`` (N,)
      * ``ee_x_tgt``/``ee_y_tgt`` (M,) — target EE path
      * ``exo_tau``    (1, N)
      * ``exo_diag``   (10, N)
      * ``sea_diag``   (8, N)
      * ``r_p`` / ``r_exo``  scalars wrapped in (1,) arrays
    """
    t = np.asarray(npz["t"]).ravel()
    state = np.asarray(npz["state"])
    # Drake joint ordering for this URDF is [q2, q1] — recover q1, q2.
    q2_drake = state[0]
    q1_drake = state[1]
    q_des = np.asarray(npz["q_des"])          # (2, N) in Drake joint order [q2_des, q1_des]
    q2_des = q_des[0]
    q1_des = q_des[1]
    ref = np.asarray(npz["ref"])              # (6, N)
    ee_x_ref = ref[0]
    ee_y_ref = ref[1]
    ee_x = np.asarray(npz["ee_x"]).ravel()
    ee_y = np.asarray(npz["ee_y"]).ravel()
    exo_tau = np.asarray(npz["exo_tau"]).ravel()
    return dict(
        t=t, q1=q1_drake, q2=q2_drake,
        q1_des=q1_des, q2_des=q2_des,
        ee_x=ee_x, ee_y=ee_y,
        ee_x_ref=ee_x_ref, ee_y_ref=ee_y_ref,
        tau_exo=exo_tau,
        label="PyDrake",
    )


def _isaac_extract(npz):
    """Canonicalise the Isaac Sim log dict (written by the exo Isaac script).

    Isaac log layout (from ``run_sea_exo`` savez):
      * ``t``              (N,)
      * ``q``              (N, 2)  — [q1, q2] user order
      * ``q_des``          (N, 2)
      * ``ee_ref``         (N, 2)  — [x, y]
      * ``tau_exo``        (N,)
      * ``tens``           (N, 2)
      * ``L1``, ``L2``, ``r_p``, ``r_exo`` scalars
    Actual EE path is reconstructed from (q, L1, L2) via 2R FK.
    """
    t = np.asarray(npz["t"]).ravel()
    q = np.asarray(npz["q"])
    q_des = np.asarray(npz["q_des"])
    ee_ref = np.asarray(npz["ee_ref"])
    tau_exo = np.asarray(npz["tau_exo"]).ravel()

    L1 = float(npz["L1"]) if "L1" in npz.files else 0.335
    L2 = float(npz["L2"]) if "L2" in npz.files else 0.190
    # 2R forward kinematics (matches forward_kinematics_2r):
    #   x = L1·cos(q1) + L2·cos(q1+q2)
    #   y = L1·sin(q1) + L2·sin(q1+q2)
    ee_x = L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1])
    ee_y = L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])

    return dict(
        t=t, q1=q[:, 0], q2=q[:, 1],
        q1_des=q_des[:, 0], q2_des=q_des[:, 1],
        ee_x=ee_x, ee_y=ee_y,
        ee_x_ref=ee_ref[:, 0], ee_y_ref=ee_ref[:, 1],
        tau_exo=tau_exo,
        label="Isaac Sim",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pydrake", required=True, help="Path to PyDrake .npz log")
    p.add_argument("--isaac", required=True, help="Path to Isaac Sim .npz log")
    p.add_argument("--out", default="plots/exo_pydrake_vs_isaac.png",
                   help="Output PNG path.")
    p.add_argument("--title", default=None,
                   help="Optional super-title (default: derived from filenames).")
    args = p.parse_args()

    pyd = _pydrake_extract(_load(args.pydrake))
    isc = _isaac_extract(_load(args.isaac))

    title = args.title or (
        f"Exosuit co-contraction — PyDrake vs Isaac Sim\n"
        f"(pydrake={Path(args.pydrake).name}, "
        f"isaac={Path(args.isaac).name})"
    )

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title, fontsize=11)

    # ── EE XY path overlay ────────────────────────────────────────────────
    ax[0, 0].plot(pyd["ee_x_ref"], pyd["ee_y_ref"], "k--", lw=1,
                  label="reference")
    ax[0, 0].plot(pyd["ee_x"], pyd["ee_y"], "b-", lw=1.4, alpha=0.9,
                  label=f"{pyd['label']}")
    ax[0, 0].plot(isc["ee_x"], isc["ee_y"], "r-", lw=1.2, alpha=0.7,
                  label=f"{isc['label']}")
    ax[0, 0].set_aspect("equal", adjustable="datalim")
    ax[0, 0].set_xlabel("EE x [m]")
    ax[0, 0].set_ylabel("EE y [m]")
    ax[0, 0].set_title("End-effector XY path")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, alpha=0.4)

    # ── q2 vs time overlay ────────────────────────────────────────────────
    ax[0, 1].plot(pyd["t"], np.rad2deg(pyd["q2_des"]), "k--", lw=1,
                  label="q₂ desired")
    ax[0, 1].plot(pyd["t"], np.rad2deg(pyd["q2"]), "b-", lw=1.2,
                  label=f"q₂ — {pyd['label']}")
    ax[0, 1].plot(isc["t"], np.rad2deg(isc["q2"]), "r-", lw=1.2, alpha=0.7,
                  label=f"q₂ — {isc['label']}")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("q₂ [deg]")
    ax[0, 1].set_title("Elbow angle q₂")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, alpha=0.4)

    # ── q2 tracking error overlay ─────────────────────────────────────────
    ax[1, 0].plot(pyd["t"], np.rad2deg(pyd["q2"] - pyd["q2_des"]), "b-",
                  lw=1.2, label=f"{pyd['label']} — err q₂")
    ax[1, 0].plot(isc["t"], np.rad2deg(isc["q2"] - isc["q2_des"]), "r-",
                  lw=1.2, alpha=0.7, label=f"{isc['label']} — err q₂")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("q₂ − q₂_des [deg]")
    ax[1, 0].set_title("Elbow tracking error")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.4)

    # ── Exo torque overlay ────────────────────────────────────────────────
    ax[1, 1].plot(pyd["t"], pyd["tau_exo"], "b-", lw=1.2,
                  label=f"τ_exo — {pyd['label']}")
    ax[1, 1].plot(isc["t"], isc["tau_exo"], "r-", lw=1.2, alpha=0.7,
                  label=f"τ_exo — {isc['label']}")
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("τ_exo [Nm]")
    ax[1, 1].set_title("Exo torque τ_exo = r_exo·(F_R − F_L)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved: {out_path}")

    # ── Simple quantitative metrics ───────────────────────────────────────
    def _rms(x):
        x = np.asarray(x)
        return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")

    pyd_rms_ee = _rms(np.hypot(pyd["ee_x"] - pyd["ee_x_ref"],
                               pyd["ee_y"] - pyd["ee_y_ref"]))
    isc_rms_ee = _rms(np.hypot(isc["ee_x"] - isc["ee_x_ref"],
                               isc["ee_y"] - isc["ee_y_ref"]))
    pyd_rms_q2 = _rms(np.rad2deg(pyd["q2"] - pyd["q2_des"]))
    isc_rms_q2 = _rms(np.rad2deg(isc["q2"] - isc["q2_des"]))

    print(f"\n  Tracking RMS (EE Euclid):   "
          f"PyDrake = {pyd_rms_ee*1e3:6.2f} mm    "
          f"Isaac Sim = {isc_rms_ee*1e3:6.2f} mm")
    print(f"  Tracking RMS (q₂ error):    "
          f"PyDrake = {pyd_rms_q2:6.2f} deg   "
          f"Isaac Sim = {isc_rms_q2:6.2f} deg")


if __name__ == "__main__":
    main()
