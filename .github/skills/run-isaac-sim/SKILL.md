---
name: run-isaac-sim
description: "Launch, debug, and troubleshoot Isaac Sim simulation scripts. Use when: running test_*.py scripts, debugging SimulationApp crashes, PhysX errors, USD loading failures, WebRTC streaming issues, or ArticulationView initialization problems."
argument-hint: "Describe what you're trying to run or the error you're seeing"
---

# Run & Debug Isaac Sim Scripts

## When to Use
- Running any `test_*.py` or `*isaac_sim*.py` script
- Debugging SimulationApp startup failures
- Fixing PhysX / USD / ArticulationView errors
- Setting up WebRTC streaming for remote visualization

## Prerequisites
```bash
conda activate env_isaacsim
# Verify:
python -c "from isaacsim import SimulationApp; print('OK')"
```

## Running Scripts

### Single robot (local window)
```bash
python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py --render native
```

### Multi-instance (headless benchmarking)
```bash
python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \
    --num-envs 9 --render headless --duration 30.0
```

### WebRTC streaming (remote via Tailscale)
```bash
python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \
    --num-envs 4 --render websocket
# Connect to <tailscale-ip>:49100 in browser
```

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: isaacsim` | Wrong conda env — run `conda activate env_isaacsim` |
| `RuntimeError: Failed to create SimulationApp` | GPU not available or display not set. Use `--render headless` |
| `PhysX: exceeded max GPU pairs` | Reduce `--num-envs` or increase `gpu_max_rigid_contact_count` |
| `ArticulationView not initialized` | Missing `world.reset()` between `initialize_dynamics_view` and `finalize_dynamics_view` |
| `Segfault on Ctrl+C` | Use `signal.signal(SIGINT, handler)` pattern, NOT try/except |
| Plot not showing | Use `matplotlib.use('Agg')` + save PNG + open with `eog` after `simulation_app.close()` |
| `omni.kit.livestream.webrtc` not found | Extension not installed — check Isaac Sim version |

## Render Mode Decision
```
Need real-time visualization?
  ├─ Yes, local machine → --render native
  ├─ Yes, remote machine → --render websocket
  └─ No, just data/plots → --render headless (fastest)
```

## Performance Tips
- Headless mode is 2-5x faster than native rendering
- Batch `world.step(render=False)` for headless
- Use `ArticulationView` for batched state/torque queries across N robots
- Profile with `step % 500 == 0` RTF (real-time factor) printouts

---

## Exo Co-Contraction Scripts

### Main run script
`script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py`

Key argument groups:
```bash
# Control
--ct-kp 400 --ct-kd 80

# Drive SEA
--spring-stiffness 200 --cable-damping 8.0 --motor-bandwidth 100

# Exo co-contraction
--exo-ks 8000 --exo-delta-theta 0.1
--exo-activate --exo-activate-time 4.0   # enable timed co-contraction

# Disturbance (torque/sine/vel/pos)
--disturbance --disturbance-mode sine
--disturbance-tau 2.0 --disturbance-freq 2.0
--disturbance-cycles 3 --disturbance-time 8.0

# Trajectory
--traj-shape circle --traj-cx 0.5 --traj-radius 0.05
--traj-n 60 --traj-v-max 0.1 --num-laps 2
```

### VS Code tasks (run via `Tasks: Run Task`)
| Task | Purpose |
|------|---------|
| `🩹 Exo Isaac Sim: Circle ON (co-contraction @ t=4s)` | Standard circle, exo ON |
| `🩹 Exo Isaac Sim: Circle (passive)` | Standard circle, exo OFF for baseline |
| `🩹 Exo Isaac Sim: Rect ON (co-contraction @ t=4s)` | Rect track, exo ON |
| `🧪 Exo Isaac Sim: Sine Disturbance ON (co-contraction...)` | Narrow-line quasi-static + sine dist |
| `🧪 Exo Isaac Sim: Sine Disturbance OFF (baseline...)` | Same without exo |
| `⭕ Exo Isaac Sim: Circle + Disturbance ON (co-contraction...)` | Circle + sine disturbance + exo |
| `⭕ Exo Isaac Sim: Circle + Disturbance OFF (baseline...)` | Circle + sine disturbance without exo |
| `🩹 Exo Isaac Sim: Headless Circle ON (for plots)` | Headless, saves `.npz` and PNGs |
| `🎨 Exo Isaac Sim: Scene Viz (no control)` | Scene visualisation only |

### Expected output
```
✓ Move-to-start complete at t=3.01 s — tracking begins.
⚡ Exo ACTIVATED at t=4.01 s  (Δθ=0.100 rad, k_eff=36.4810 Nm/rad)
💥 External-torque disturbance armed: τ_ext = +2.00·sin(2π·2.0Hz·t) Nm  window=[8.00, 9.50] s
...
Mean tracking RMS: ~1.8 mm
```

### Plot files saved to `plots/`
- `sea_exo_isaac_*_manip.png` — joint angles, EE path, tracking error
- `sea_exo_isaac_*_exo.png` — δ_R/δ_L, τ_exo, cable forces, motor positions

### Troubleshooting oscillations
If EE RMS > 10 mm:
1. Check IK failure diagnostic at sim end — "⚠ IK failed X% of trajectory steps"
2. Reduce `--traj-radius` or shift `--traj-cx` so circle fits within `[|L1-L2|, L1+L2]`
3. Verify `L1+L2 ≈ 0.525 m` printed after `world.reset()`
4. `_clamp_to_reach` is active when `L1,L2` forwarded through `build_trajectory(L1=L1, L2=L2)`
