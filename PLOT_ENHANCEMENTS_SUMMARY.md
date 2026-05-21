# Plot Enhancements for `script_cup_manipulator_pendulam_tendon_with_exo_param_learning_pydrake.py`

## Overview
The plotting function has been significantly enhanced to provide comprehensive visualization of the probe-based exosuit impedance learning workflow. The figure now spans **5 rows × 2 columns** (from previous 3 × 2) and includes detailed exo dynamics, CT gain adaptation visualization, and expanded identification diagnostics.

## Enhanced Figure Layout

### **Row 0: Probe Phase Overview**
- **[0,0] Elbow deflection during probing**: Time-series of joint 2 angle deviation (q₂ - q₂_anchor) in degrees, showing the sinusoidal probe-induced oscillation within the probe window (highlighted in orange).
- **[0,1] LS identification fit quality**: Overlaid plots of measured τ_exo vs. the least-squares fitted model output, with correlation coefficient and residual RMS shown in title.

### **Row 1: Exo Impedance Identification & Cable Dynamics**
- **[1,0] Impedance comparison (bar chart)**: Side-by-side bars comparing ground-truth K_e, B_e vs. estimated K̂_e, B̂_e with numerical values labeled. Instantly shows identification accuracy.
- **[1,1] Exo spring extensions**: Time-series of right (δ_R) and left (δ_L) cable spring extensions [mm] from the SEA exo, showing how the bilateral cables respond to joint motion and probe torque.

### **Row 2: Exo Cable Forces & Torque**
- **[2,0] Exo cable forces**: Right (F_R) and left (F_L) tension forces [N] from the co-contraction cable pair, overlaid with probe window. Shows cable pre-tension and dynamic stiffness response.
- **[2,1] Exo torque vs position error**: τ_exo [Nm] on primary axis + position error regressor [rad] on secondary axis, illustrating the correlation between the dominant regressor signal and the measured torque output.

### **Row 3: CT Adaptation & Trajectory Tracking**
- **[3,0] CT gain comparison (bar chart)**: Before/after visualization of proportional (K_p) and derivative (K_d) gains, highlighting the magnitude of adaptation triggered by identified impedance. Shows weak initial values and updated values after learning.
- **[3,1] End-effector trajectory**: XY plot comparing target trajectory (dashed), baseline weak-CT tracking (if enabled), and updated-CT tracking, all overlaid with axis-equal aspect for geometric interpretation.

### **Row 4: Tracking Performance & Regressor Signals**
- **[4,0] Tracking error vs time**: Magnitude of EE position error [mm] for both weak-CT (baseline) and updated-CT phases, with RMS summaries in legend. Shows immediate improvement from learning.
- **[4,1] LS regressor excitation**: Position error regressor [scaled crad] and velocity regressor [scaled crad/s] signals, demonstrating that both regressors are sufficiently excited for stable identification.

## Key Enhancements

### 1. **Exo Spring Dynamics** ✓
   - Added plots for δ_R, δ_L (spring extensions) extracted from exo diagnostics port
   - Shows cable geometry and co-contraction pre-tension behavior
   - Data source: `exo_diag[0:2]` from SEAExoActuator.GetOutputPort("diagnostics")

### 2. **Exo Cable Forces** ✓
   - Added plots for F_R, F_L (cable tensions)
   - Illustrates the nonlinear spring-damper response
   - Data source: `exo_diag[2:4]`

### 3. **Exo Torque Analysis** ✓
   - Exo τ_exo [Nm] overlaid with excitation signals
   - Validates the linear impedance model: τ_exo ≈ -K_e·(q₂-q₂_a) - B_e·q̇₂
   - Data source: `exo_diag[8]`

### 4. **CT Gain Visualization** ✓
   - New row showing initial weak gains (Kp=10, Kd=2) vs. updated values
   - Demonstrates the adaptation law: `Kp_new = Kp_weak + α_K·K̂_e`, `Kd_new = Kd_weak + α_D·B̂_e`
   - Unit conversion factors α_K, α_D are CLI parameters (default: 8.0, 80.0)

### 5. **Improved Layout**
   - Figure size increased to 16×14 inches (from 14×10)
   - Gridspec with hspace=0.40, wspace=0.30 for readable spacing
   - Comprehensive docstring describing all 10 subplots
   - Fixed tight_layout warning using explicit subplots_adjust()

## Data Sources

All plots extract data from three main log dictionaries (`probe_log`, `base_log`, `track_log`), which collect:

| Data | Source | Shape |
|------|--------|-------|
| State (q, q̇) | Plant state output | [12, N] for 6D exo system |
| Exo diagnostics | SEAExoActuator port | [10, N]: δ_R,L; F_R,L; θ_m; τ_exo; active |
| Exo torque | SEAExoActuator port | [1, N] |
| CT desired positions | ComputedTorqueController port | [2, N] |
| EE position (FK) | Numerical FK evaluation | [2, N] |
| Probe torque | _ProbeTorqueSrc port | [1, N] |

## Example Output

Running with default parameters:
```bash
python script_cup_manipulator_pendulam_tendon_with_exo_param_learning_pydrake.py \
  --no-meshcat --no-show --track-laps 1 --track-duration 4.0 --probe-duration 3.0
```

Produces output showing:
- **Identification**: K̂_e ≈ 33.4 Nm/rad (true 36.5, -8.3% error), correlation = 1.0
- **CT Gain Update**: Kp: 10 → 277.6, Kd: 2.0 → 4.3
- **Tracking RMS**: 11.2 mm with updated CT (vs. ~50+ mm with weak CT alone)

The 10-panel figure is automatically displayed (or saved to file) and provides complete insight into:
1. Identification quality and regressor excitation
2. Exo cable dynamics during probing
3. CT gain adaptation magnitude
4. Tracking performance improvement

## Usage

The enhanced plots are generated automatically by the `plot_results()` function called at the end of `main()`. To view/save:

```python
# Display live (requires Matplotlib backend configured)
fig = plot_results(args, probe_log, ident, base_log, track_log, K_true, B_true)
plt.show()

# Or save to file
fig.savefig("exo_learning_results.pdf", dpi=150, bbox_inches="tight")
```

## Future Extensions

- Add phase-plane trajectories (q₂ vs q̇₂) to show convergence paths
- Include motor rotor dynamics (θ_m, τ_motor) to debug SEA lag during learning
- Add frequency-domain analysis (FFT of probe window) for regressor conditioning
- Animate the 2D trajectory in real-time during tracking phase
