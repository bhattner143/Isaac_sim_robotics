"""Unit tests for elbow MIT feed-forward from cable command."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from controller.cable_wrench_mhp import (
    CableWrenchConfig,
    distribute_mhp_actuation,
    elbow_torque_from_cable_command,
)


def test_elbow_torque_from_cable_command() -> None:
    r_p = 0.0394
    F_cmd = 12.5
    assert elbow_torque_from_cable_command(F_cmd, r_p) == r_p * F_cmd


def test_default_elbow_ff_matches_ct() -> None:
    cfg = CableWrenchConfig()
    tau_req = np.array([1.2, -3.4])
    dist = distribute_mhp_actuation(tau_req, cfg)
    assert dist["tau_elbow_ff"] == tau_req[1]
    assert dist["F_net"] == tau_req[1] / cfg.r_elbow


def test_cable_ff_differs_when_inner_loop_corrects() -> None:
    cfg = CableWrenchConfig()
    tau_req = np.array([0.0, 2.0])
    dist = distribute_mhp_actuation(tau_req, cfg)
    F_net = dist["F_net"]
    F_meas = F_net * 0.8
    F_cmd = F_net + 0.5 * (F_net - F_meas)
    tau_from_cable = elbow_torque_from_cable_command(F_cmd, cfg.r_elbow)
    assert tau_from_cable != dist["tau_elbow_ff"]
    assert tau_from_cable == cfg.r_elbow * F_cmd


if __name__ == "__main__":
    test_elbow_torque_from_cable_command()
    test_default_elbow_ff_matches_ct()
    test_cable_ff_differs_when_inner_loop_corrects()
    print("ok")
