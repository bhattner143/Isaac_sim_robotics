"""Debug: run once, extract and plot q2, q2_des, ee_err during disturbance window."""
import subprocess, os, pickle, sys
from pathlib import Path

# Monkeypatch the script to dump state pickle before plotting
import importlib.util
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Just import and run via the script's machinery. Easier: subprocess + embedded code.
import runpy
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Minimal: patch sys.argv and run the module to run_simulation+plot_results; then
# intercept the data dict by monkey-patching plot_results.

sys.argv = [
    "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py",
    "--spring-stiffness", "30", "--cable-damping", "2.0", "--sea-mode", "torque",
    "--motor", "AK60_6_KV80_Config",
    "--ct-kp", "100", "--ct-kd", "40",
    "--no-meshcat", "--no-show",
    "--duration", "6", "--num-laps", "1",
    "--move-duration", "2.0",
    "--disturbance", "--disturbance-mode", "torque",
    "--disturbance-time", "4.0", "--disturbance-dur", "2.0",
    "--disturbance-tau", "1.0",
    "--exo-ks", "3500", "--exo-delta-theta", "0.05",
    "--exo-activate", "--exo-activate-time", "3.5",
]

# Load module, replace plot_results
spec = importlib.util.spec_from_file_location(
    "expt", str(root / "script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py")
)
m = importlib.util.module_from_spec(spec)
captured = {}
def _cap(data):
    captured.update(data)
m.plot_results = _cap
spec.loader.exec_module(m)
# main() is behind if __name__ == "__main__": — call run_simulation directly
data = m.run_simulation(None)

import numpy as np
t = data["t"]
nq = data["nq"]
q2 = data["state"][1]
q1 = data["state"][0]
q2des = data["q_des"][1]
q1des = data["q_des"][0]
ee_x = data["ee_x"]; ee_y = data["ee_y"]
ref = data["ref"]

# Find indices near t=4, 4.5, 5, 5.5, 6
for ts in [0.1, 1.5, 2.5, 3.5, 3.9, 4.1, 4.5, 5.0, 5.5, 5.9]:
    i = int(np.argmin(np.abs(t - ts)))
    print(f"t={t[i]:5.2f}  q1={np.rad2deg(q1[i]):7.2f}°  q1d={np.rad2deg(q1des[i]):7.2f}°   "
          f"q2={np.rad2deg(q2[i]):7.2f}°  q2d={np.rad2deg(q2des[i]):7.2f}°   "
          f"ee=({ee_x[i]*1e3:6.1f},{ee_y[i]*1e3:6.1f})mm  ref=({ref[0,i]*1e3:6.1f},{ref[1,i]*1e3:6.1f})mm")
print(f"\npeak |q2-q2d| over [4,6]s = {np.rad2deg(np.max(np.abs(q2[(t>=4)&(t<=6)]-q2des[(t>=4)&(t<=6)]))):.3f} deg")
print(f"peak |q1-q1d| over [4,6]s = {np.rad2deg(np.max(np.abs(q1[(t>=4)&(t<=6)]-q1des[(t>=4)&(t<=6)]))):.3f} deg")
print(f"peak ee_err over [4,6]s   = {1e3*np.max(np.sqrt((ee_x[(t>=4)&(t<=6)]-ref[0,(t>=4)&(t<=6)])**2 + (ee_y[(t>=4)&(t<=6)]-ref[1,(t>=4)&(t<=6)])**2)):.2f} mm")
