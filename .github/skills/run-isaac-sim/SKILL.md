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
