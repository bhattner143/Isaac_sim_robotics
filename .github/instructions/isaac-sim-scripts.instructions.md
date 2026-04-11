---
description: "Use when writing or editing Isaac Sim scripts (test_*.py, *isaac_sim*.py). Covers SimulationApp boot order, SIGINT handling, matplotlib Agg backend, conda env_isaacsim, ArticulationView patterns."
applyTo: ["test_*.py", "*isaac_sim*.py", "*isaac*.py"]
---

# Isaac Sim Script Conventions

## Import Order (CRITICAL)
`SimulationApp()` **must** be the first Isaac Sim import — before any `omni.*`, `pxr.*`, or project modules that touch USD:

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

# NOW safe to import omni / pxr / project modules
from omni.isaac.core import World
```

## Pre-parse CLI before SimulationApp
argparse cannot run before SimulationApp (it blocks). Pre-parse `--render` and `--num-envs` with a simple `sys.argv` loop, then build the full parser after SimulationApp.

## SIGINT Handling
Isaac Sim's C++ runtime swallows `KeyboardInterrupt`. Use `signal.signal` instead:

```python
import signal
_stop_requested = False
def _sigint_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
signal.signal(signal.SIGINT, _sigint_handler)
```

Check `_stop_requested` in the sim loop, then restore with `signal.signal(signal.SIGINT, _orig_sigint)`.

## Matplotlib
Isaac Sim owns the OpenGL context. Always use the Agg backend:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

Save to PNG, open with `eog` or `xdg-open` after `simulation_app.close()`.

## Conda Environment
Scripts run under `conda activate env_isaacsim` (Python 3.11, Isaac Sim 5.1.0).

## ArticulationView Pattern
```python
m.initialize_dynamics_view(world, reset=False)
world.reset()
m.initialize_state()
m.finalize_dynamics_view(world)
```
Always call `world.reset()` between `initialize_dynamics_view` and `finalize_dynamics_view`.

## Render Modes
Support three modes via `--render`: `native` (local window), `websocket` (WebRTC stream), `headless` (no display). Set `headless=(_render_mode != "native")` in SimulationApp config.
