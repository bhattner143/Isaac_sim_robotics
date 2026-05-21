# Quick Reference: Plot Data Sources

## Exo Diagnostics Vector (10 elements from SEAExoActuator)
```
exo_diag = [δ_R, δ_L, F_R, F_L, θ_mR/N, θ_mL/N, θ̇_mR/N, θ̇_mL/N, τ_exo, activated]
           [0    1    2    3    4        5        6        7        8       9]
```

| Index | Signal | Units | Plot Used In |
|-------|--------|-------|--------------|
| 0 | δ_R (right spring extension) | m | Row 1, Col 1 (exo spring extensions) |
| 1 | δ_L (left spring extension) | m | Row 1, Col 1 (exo spring extensions) |
| 2 | F_R (right cable force) | N | Row 2, Col 0 (exo cable forces) |
| 3 | F_L (left cable force) | N | Row 2, Col 0 (exo cable forces) |
| 4 | θ_mR/N (right motor joint-side pos) | rad | Available for future use |
| 5 | θ_mL/N (left motor joint-side pos) | rad | Available for future use |
| 6 | θ̇_mR/N (right motor joint-side vel) | rad/s | Available for future use |
| 7 | θ̇_mL/N (left motor joint-side vel) | rad/s | Available for future use |
| 8 | τ_exo (net exo torque) | Nm | Row 0 Col 1, Row 2 Col 1, Row 4 Col 0 |
| 9 | activated (1.0 if ON, 0.0 if OFF) | binary | Available for future use |

## State Vector (12 elements for 6D exo system)
```
state = [q1, q2, θ_m, θ̇_m, q̇1, q̇2, θ̇_m, θ̈_m, ...]  
         [0  1  2    3    4   5   6    7]   (approximate, see comments)
```

| Index | Signal | Units | Plot Used In |
|-------|--------|-------|--------------|
| q2_idx | q₂ (elbow angle) | rad | Row 0 Col 0 (elbow deflection), LS regressors |
| nq + q2_idx | q̇₂ (elbow velocity) | rad/s | Row 4 Col 1 (regressor signal) |

## Identification Results Dictionary
```python
ident = {
    "K_e_hat": float,      # Estimated stiffness [Nm/rad]
    "B_e_hat": float,      # Estimated damping [Nm·s/rad]
    "q2_a": float,         # Anchor angle [rad]
    "residual_rms": float, # LS fit RMS error [Nm]
    "fit_corr": float,     # Correlation between measured & fit τ_exo
    "Phi": ndarray,        # [N×2] regressor matrix
    "y": ndarray,          # [N] measured τ_exo vector
    "y_fit": ndarray,      # [N] LS-fitted τ_exo
    "t_probe": ndarray,    # [N] time samples in probe window
}
```

## Plot-to-Data Mapping

### **Row 0, Col 0** — Probe Phase Elbow Deflection
- **Data**: `state[q2_idx, :] - ident['q2_a']`, converted to degrees
- **Extracted From**: `probe_log["state"]`, `probe_log["q2_idx"]`, `ident["q2_a"]`
- **Time Range**: Full probe phase (0 to ~7s), highlighted probe window overlaid

### **Row 0, Col 1** — LS Fit Quality
- **Data**: `ident["y"]` vs `ident["y_fit"]`
- **Correlation**: `ident["fit_corr"]` (should be ~1.0 for linear exo model)
- **Residual RMS**: `ident["residual_rms"] * 1000` (converted to mNm)

### **Row 1, Col 0** — Impedance Bar Chart
- **Data**: 
  - Truth: `K_true = 2·exo_ks·r_exo²`, `B_true = 2·exo_bc·r_exo²`
  - Estimated: `ident["K_e_hat"]`, `ident["B_e_hat"]`

### **Row 1, Col 1** — Exo Spring Extensions
- **Data**: `probe_log["exo_diag"][0, :]` (δ_R), `probe_log["exo_diag"][1, :]` (δ_L)
- **Units**: Converted to mm for plot

### **Row 2, Col 0** — Exo Cable Forces
- **Data**: `probe_log["exo_diag"][2, :]` (F_R), `probe_log["exo_diag"][3, :]` (F_L)
- **Units**: N (direct from diagnostics)

### **Row 2, Col 1** — Exo Torque vs Regressor
- **Data (Primary Y-axis)**: `probe_log["exo_diag"][8, :]` (τ_exo) [Nm]
- **Data (Secondary Y-axis)**: `ident["Phi"][:, 0]` (position error regressor) [rad]
- **Demonstrates**: Linear relationship τ_exo ≈ -K_e·(q₂-q₂_a)

### **Row 3, Col 0** — CT Gain Comparison
- **Initial Gains**: `args.ct_kp_weak`, `args.ct_kd_weak`
- **Updated Gains**: `Kp_new = args.ct_kp_weak + args.alpha_K · ident["K_e_hat"]`
- **Updated Gains**: `Kd_new = args.ct_kd_weak + args.alpha_D · ident["B_e_hat"]`

### **Row 3, Col 1** — EE Trajectory
- **Data**: 
  - Target: `track_log["ee_x_tgt"]`, `track_log["ee_y_tgt"]`
  - Baseline (if available): `base_log["ee_x"]`, `base_log["ee_y"]`
  - Updated: `track_log["ee_x"]`, `track_log["ee_y"]`
- **Source**: FK computed numerically for each state sample

### **Row 4, Col 0** — Tracking Error vs Time
- **Data**: Distance from each EE position to nearest point on target trajectory
- **Function**: `_track_err(log)` computes per-sample error magnitude [m]
- **Converted to** [mm] for display

### **Row 4, Col 1** — LS Regressor Signals
- **Data**: `ident["Phi"][:, 0]` (position error), `ident["Phi"][:, 1]` (velocity)
- **Scaled** by 100× and 10× respectively for visibility
- **Purpose**: Shows both regressors are excited for stable 2-parameter fit

## Notes on Exo Model Extraction

The exosuit model equations used in identification are:

$$\tau_{exo} = -K_e (q_2 - q_{2,a}) - B_e \dot{q}_2$$

Where the true impedance is:
$$K_e = 2 k_{exo} r_{exo}^2$$
$$B_e = 2 b_{exo} r_{exo}^2$$

With default parameters:
- k_exo = 8000 N/m, r_exo = 0.04775 m → K_e_true ≈ 36.5 Nm/rad
- b_exo = 2.0 N·s/m, r_exo = 0.04775 m → B_e_true ≈ 0.009 Nm·s/rad

The bilateral cable pre-tension (Δθ = 0.1 rad) ensures co-contraction throughout tracking.
