# Isaac Sim WebRTC Streaming — Setup & Usage Guide

This folder provides scripts to run **Isaac Sim in headless streaming mode** and view it remotely using the bundled WebRTC streaming client. This is analogous to how PyDrake uses Meshcat — the simulation runs headlessly and you view it in a separate client — except here you get full interactive 3D rendering instead of a browser widget.

---

## Folder Contents

```
isaac-sim-webRTC/
├── example_webrtc_scene.py                          # Minimal self-contained example
├── run_server.sh                                    # Stream Isaac Sim GUI editor (optional)
├── launch_client.sh                                 # Launch the viewer client
├── isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage  # NVIDIA's viewer app
└── README.md                                        # This file
```

---

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Terminal 1: python example_webrtc_scene.py         │
│  ┌──────────────────────────────────────────────┐   │
│  │  Isaac Sim (headless, no display)            │   │
│  │  SimulationApp + omni.kit.livestream.webrtc  │   │
│  │                                              │   │
│  │  hide_ui=False → full editor UI streamed     │   │
│  │  hide_ui=True  → viewport only streamed      │   │
│  │                                              │   │
│  │  Ports exposed:                              │   │
│  │    49100 → WebRTC signaling (video stream)   │   │
│  │    8011  → REST API                          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
              ↕ WebRTC (GPU-accelerated video)
              ↕ localhost or Tailscale VPN
┌─────────────────────────────────────────────────────┐
│  Viewer: NVIDIA Streaming Client (Mac/Linux/Win)    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Server field: localhost:49100               │   │
│  │            or: 100.x.x.x:49100 (Tailscale)  │   │
│  │  → Streams live 3D view, forwards mouse/kbd  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

The **server** (Isaac Sim) runs the physics simulation and renders the 3D scene using the GPU. It encodes the video and streams it over WebRTC. The **client** receives the video stream and forwards mouse and keyboard back — so you can orbit the camera, click, etc., just like the native Isaac Sim window.

### Full GUI vs Viewport-Only

The `hide_ui` setting in `SimulationApp` controls what gets streamed:

```python
# Full Isaac Sim editor UI (menus, stage panel, timeline, viewport)
simulation_app = SimulationApp({"headless": True, "hide_ui": False})

# Viewport only — just the 3D render, no editor chrome
simulation_app = SimulationApp({"headless": True, "hide_ui": True})
```

Both scripts in this folder use `hide_ui: False` (full GUI streamed).

### Why Streaming Instead of Native Window?

| Mode | When to Use |
|------|-------------|
| Native window | Local development, direct interaction |
| WebRTC streaming | Remote machines, WSL, headless servers, multi-monitor setups |
| Headless (no display) | Automated pipelines, data collection, no visual needed |

The streaming approach via `./isaac-sim.streaming.sh` uses a specialised kit config (`isaacsim.exp.full.streaming.kit`) that loads the WebRTC backend instead of a display window.

---

## Quick Start

### Step 1 — Run your Python script (self-contained)

Python simulation scripts **start Isaac Sim and the WebRTC server automatically**. No `conda activate` needed — the script re-execs itself under `env_isaacsim` if required:

```bash
# From anywhere — the script handles the environment switch
python example_webrtc_scene.py

# Or any other Isaac Sim script in the project
python test_cup_manipulator_tendon_scene_viz.py
```

Wait until you see:
```
✓ Scene ready — streaming on port 49100
```

Isaac Sim can take **60–90 seconds** on first launch (shader compilation).

**Ports opened automatically:**
- `49100` — WebRTC signaling (the video stream)
- `8011` — HTTP REST API

### Step 2 — Launch the Viewer Client

Open a **second terminal** and run:

```bash
cd /home/user/Documents/isaac_sim_robotics/isaac-sim-webRTC
./launch_client.sh
```

This script waits for port 49100 to be ready, then launches the bundled AppImage client.

### Step 3 — Connect

When the Isaac Sim WebRTC Streaming Client window opens, enter in the **Server** field:

```
localhost:49100          ← if viewer is on the same machine
100.x.x.x:49100         ← if viewer is on another machine (Tailscale IP)
```

Click **Connect**. The full Isaac Sim editor UI streams into the client window. You can orbit the camera (middle mouse drag), zoom (scroll wheel), and continue typing commands in Terminal 1's interactive prompt (`g`, `d`, `r`, `p`, `q`).

---

## Script Details

### `launch_client.sh`

Waits for port 49100 to be ready, then launches the best available streaming client.

```bash
# Default — wait for the Python script to start Isaac Sim first (recommended)
./launch_client.sh

# Skip the port-readiness wait (if you know Isaac Sim is already up)
./launch_client.sh --nowait
```

**Client priority order:**
1. **Bundled AppImage** in this folder (always used if present)
2. **Omniverse Streaming Client** installed in `~/.local/share/ov/pkg/` or `/opt/ov/pkg/`
3. **Browser fallback** — opens `http://localhost:8011/streaming/client/`
4. **Manual instructions** printed to terminal

### `run_server.sh`

Only needed if you want to stream the **Isaac Sim GUI editor** itself — for opening `.usd` scenes manually, using the editor UI, etc. Python-driven scripts do not need this.

```bash
./run_server.sh
```

For all Python simulation scripts, run them directly (`python myscript.py`). The script re-execs itself under the correct interpreter.

---

## Remote Streaming over Tailscale

To view the simulation from another machine (e.g. Mac), use [Tailscale](https://tailscale.com) — a zero-config VPN that works through NAT/firewalls without port forwarding.

### Setup (one-time)

**Linux:**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up       # opens a browser URL — log in with your Tailscale account
tailscale ip -4         # note the 100.x.x.x IP
```

**Mac/Windows:** Install from [tailscale.com](https://tailscale.com), sign in with the same account.

### Streaming

```bash
# Linux — start the scene
python example_webrtc_scene.py

# Mac — open the NVIDIA streaming client, enter in Server field:
#   100.x.x.x:49100   ← Linux Tailscale IP from above
```

No firewall rules needed. Tailscale encrypts the connection end-to-end over WireGuard.

---

## Running Isaac Sim Python Scripts

All Isaac Sim scripts in this project are **self-contained**:
- They detect if they're running under the wrong Python and re-exec under `env_isaacsim` automatically.
- They default to `--render websocket` so WebRTC streaming starts without any extra flags.

```bash
# Run from any shell, any directory — no conda activate needed
python example_webrtc_scene.py
python test_cup_manipulator_tendon_scene_viz.py

# Override the render mode explicitly
python myscript.py --render native    # local OS window (requires display)
python myscript.py --render headless  # no display, no stream (data collection)

# Combine with scene-specific arguments
python test_cup_manipulator_tendon_scene_viz.py --q1 30 --q2 -15
```

### Pattern used in each script

```python
_ISAACSIM_PYTHON = "/home/user/anaconda3/envs/env_isaacsim/bin/python"
if os.path.exists(_ISAACSIM_PYTHON) and sys.executable != _ISAACSIM_PYTHON:
    os.execv(_ISAACSIM_PYTHON, [_ISAACSIM_PYTHON] + sys.argv)
    # process is replaced — nothing below runs in the wrong env
```

This works from any shell (base conda, system Python, venv) without needing to activate a specific environment first.

---

## Troubleshooting

### "Server did not respond on port 49100 after 60s"

Isaac Sim can take a long time to start. Options:
1. Wait longer and use `--nowait` then try connecting manually once you see server-ready output
2. Check the server terminal for error messages
3. Ensure no other Isaac Sim instance is using port 49100: `ss -tlnp | grep 49100`

### Client shows blank/black screen after connecting

- Wait 10–15 seconds — Isaac Sim may still be compiling shaders after the port opens.
- Try reconnecting (disconnect and connect again in the client).
- Verify the port is actually bound: `ss -tlnp | grep 49100` — should show `0.0.0.0:49100`.
- For **remote connections**: the pip-installed `omni.kit.livestream.webrtc` is designed for localhost. If you see blank remotely but fine locally, use Tailscale (see above) — it routes the WebRTC traffic correctly through the loopback path.

### "Exit Code 126" when running `run_server.sh`

The script isn't executable or the Isaac Sim path is wrong:
```bash
chmod +x run_server.sh launch_client.sh
# Then verify the Isaac Sim installation exists:
ls /home/user/Documents/isaac-sim/isaac-sim.streaming.sh
```

### AppImage won't open (sandbox error)

Run it directly with the `--no-sandbox` flag:
```bash
./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage --no-sandbox
```

Or add this to `launch_client.sh` after the `exec "$APPIMAGE"` line.

### Port 8011 web client shows 404

The HTTP web streaming client endpoint (`/streaming/client/`) is only served by some versions of Isaac Sim's HTTP extension. If it returns 404, use the AppImage client (port 49100) instead.

---

## Network Ports Reference

| Port | Protocol | Purpose |
|------|----------|---------|
| 49100 | WebSocket / WebRTC | Video stream + input forwarding (connect client here) |
| 8011 | HTTP | Isaac Sim REST API + optional web streaming client |
| 8011–8100 | HTTP | Port auto-range if 8011 is busy |

---

## System Requirements

- **NVIDIA GPU** with driver supporting hardware video encoding (H.264/HEVC)
- **Isaac Sim** installed at `/home/user/Documents/isaac-sim/`
- **FUSE** support for AppImage: `sudo apt install libfuse2` if the AppImage fails to mount
- Network: loopback (`localhost`) is sufficient for local streaming

---

## File Locations

| Path | Description |
|------|-------------|
| `/home/user/Documents/isaac-sim/` | Standalone Isaac Sim installation |
| `/home/user/Documents/isaac-sim/isaac-sim.streaming.sh` | The streaming launch script |
| `/home/user/Documents/isaac-sim/apps/isaacsim.exp.full.streaming.kit` | Kit config for streaming mode |
| `/home/user/anaconda3/envs/env_isaacsim/` | Conda env with Isaac Sim Python package (for scripts) |
