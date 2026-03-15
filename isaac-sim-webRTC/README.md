# Isaac Sim WebRTC Streaming — Setup & Usage Guide

This folder provides scripts to run **Isaac Sim in headless streaming mode** and view it remotely using the bundled WebRTC streaming client. This is analogous to how PyDrake uses Meshcat — the simulation runs headlessly and you view it in a separate client — except here you get full interactive 3D rendering instead of a browser widget.

---

## Folder Contents

```
isaac-sim-webRTC/
├── run_server.sh                                    # Start Isaac Sim streaming server
├── launch_client.sh                                 # Launch the viewer client
├── isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage  # NVIDIA's viewer app
└── README.md                                        # This file
```

---

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Terminal 1: run_server.sh                          │
│  ┌──────────────────────────────────────────────┐   │
│  │  Isaac Sim (headless, no display)            │   │
│  │  /home/user/Documents/isaac-sim/             │   │
│  │  └── apps/isaacsim.exp.full.streaming.kit    │   │
│  │                                              │   │
│  │  Ports exposed:                              │   │
│  │    49100 → WebRTC signaling (video stream)   │   │
│  │    8011  → REST API + web client URL         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
              ↕ WebRTC (GPU-accelerated video)
┌─────────────────────────────────────────────────────┐
│  Terminal 2: launch_client.sh                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  Isaac Sim WebRTC Streaming Client (Electron)│   │
│  │  → Enter server: localhost  port: 49100      │   │
│  │  → Streams live 3D view, handles mouse/kbd   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

The **server** (Isaac Sim) runs the physics simulation and renders the 3D scene using the GPU. It then encodes the video and streams it over WebRTC. The **client** (the AppImage) receives the video stream and forwards your mouse and keyboard back to the server — so you can orbit the camera, click, etc., just like the native Isaac Sim window.

### Why Streaming Instead of Native Window?

| Mode | When to Use |
|------|-------------|
| Native window | Local development, direct interaction |
| WebRTC streaming | Remote machines, WSL, headless servers, multi-monitor setups |
| Headless (no display) | Automated pipelines, data collection, no visual needed |

The streaming approach via `./isaac-sim.streaming.sh` uses a specialised kit config (`isaacsim.exp.full.streaming.kit`) that loads the WebRTC backend instead of a display window.

---

## Quick Start

### Step 1 — Start the Streaming Server

Open a terminal in this folder and run:

```bash
./run_server.sh
```

Wait until you see output like:
```
Isaac Sim streaming server started.
[Kit] App is running
```

This can take **60–90 seconds** on first launch (shader compilation). The server is ready when you see no more loading messages and output has stabilised.

**Ports used:**
- `49100` — WebRTC websocket signaling (the stream itself)
- `8011` — HTTP REST API (also serves a web-based client at `http://localhost:8011/streaming/client/`)

### Step 2 — Launch the Viewer Client

Open a **second terminal** and run:

```bash
./launch_client.sh
```

This script:
1. Waits up to 60 seconds for the streaming server to come up (checks port 49100)
2. Finds and launches the bundled AppImage client

### Step 3 — Connect

When the Isaac Sim WebRTC Streaming Client window opens, you will see a connection form. Enter:

```
Server:  localhost
Port:    49100
```

Then click **Connect**. The Isaac Sim 3D viewport will appear in the client window. You can now orbit the camera (middle mouse button drag), zoom (scroll wheel), and interact normally.

---

## Script Details

### `run_server.sh`

Wraps `~/Documents/isaac-sim/isaac-sim.streaming.sh` with user-friendly output. Passes through any extra arguments to the underlying Isaac Sim kit binary.

```bash
# Run with default settings
./run_server.sh

# Pass extra kit arguments (e.g., disable RTX)
./run_server.sh --no-window --/rtx/rendermode=native
```

**What it runs internally:**
```
/home/user/Documents/isaac-sim/kit/kit \
    apps/isaacsim.exp.full.streaming.kit \
    --no-window
```

Extensions loaded by the streaming kit:
- `omni.kit.livestream.webrtc` — WebRTC video encoder + signaling server
- `omni.services.livestream.nvcf` — streaming session manager
- `omni.services.transport.server.http` — REST API on port 8011

### `launch_client.sh`

Intelligently finds and launches the best available streaming client.

```bash
# Default — wait for server first (recommended)
./launch_client.sh

# Skip the port-readiness wait
./launch_client.sh --nowait
```

**Client priority order:**
1. **Bundled AppImage** in this folder (highest priority — always used if present)  
2. **Omniverse Streaming Client** installed in `~/.local/share/ov/pkg/` or `/opt/ov/pkg/`
3. **Browser fallback** — opens `http://localhost:8011/streaming/client/` (if Isaac Sim HTTP API serves a web client)
4. **Manual instructions** printed to terminal

---

## Running Your Isaac Sim Python Scripts via Streaming

Your existing Python scripts (e.g., `test_cup_manipulator_tendon_scene_viz.py`) can stream using the `--render websocket` flag implemented in the script:

```bash
# In one terminal — start the streaming server for your Python script
conda activate env_isaacsim
python test_cup_manipulator_tendon_scene_viz.py --render websocket

# In another terminal — launch the viewer
cd isaac-sim-webRTC && ./launch_client.sh
```

Note: This is different from `run_server.sh`. The difference:

| Method | Use Case |
|--------|----------|
| `run_server.sh` | Starts the **full Isaac Sim app** (opens the GUI editor headlessly). Load your scene through the Isaac Sim menus/console. |
| `python script.py --render websocket` | Starts Isaac Sim **inside your Python script** in streaming mode. Your scene is built programmatically. |

The `--render` options in your Python scripts:
```bash
--render native     # (default) Opens a real OS window
--render websocket  # Headless + starts WebRTC server on port 49100
--render headless   # Headless + no streaming (for batch runs)
```

---

## Troubleshooting

### "Server did not respond on port 49100 after 60s"

Isaac Sim can take a long time to start. Options:
1. Wait longer and use `--nowait` then try connecting manually once you see server-ready output
2. Check the server terminal for error messages
3. Ensure no other Isaac Sim instance is using port 49100: `ss -tlnp | grep 49100`

### Client shows blank/black screen after connecting

- The server may still be loading. Wait 10–15 seconds and try reconnecting.
- Check that the server port number matches what you entered in the client (must be `49100`).

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
