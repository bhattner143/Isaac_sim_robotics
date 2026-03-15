#!/usr/bin/env python3
"""
example_webrtc_scene.py
=======================
Minimal self-contained Isaac Sim scene that streams over WebRTC.

Run it directly — no run_server.sh needed:
    python example_webrtc_scene.py

The script automatically re-execs itself under env_isaacsim if needed,
enables WebRTC streaming on port 49100, then builds and runs the scene.

Then in a second terminal connect the viewer:
    ./launch_client.sh         →  Server=localhost  Port=49100

Interactive commands (type in Terminal 1 while streaming):
  g             — toggle gravity on/off
  d <x> <y> <z> — drop a new cube at (x, y, z)
  r             — reset simulation
  p             — print object positions
  q             — quit
"""

import sys
import os

# =============================================================================
# STEP 0: Auto re-exec under env_isaacsim Python if not already there.
# This means you can run: python example_webrtc_scene.py
# from any environment and it will switch to the correct one automatically.
# =============================================================================
_ISAACSIM_PYTHON = "/home/user/anaconda3/envs/env_isaacsim/bin/python"

if os.path.exists(_ISAACSIM_PYTHON) and sys.executable != _ISAACSIM_PYTHON:
    print(f"[re-exec] Switching to env_isaacsim Python: {_ISAACSIM_PYTHON}")
    os.execv(_ISAACSIM_PYTHON, [_ISAACSIM_PYTHON] + sys.argv)
    # execv replaces the current process — nothing below runs in the old env

# =============================================================================
# STEP 1: Pre-parse --render BEFORE SimulationApp (MUST be absolutely first).
# SimulationApp must be the very first Isaac Sim call — no argparse yet.
# =============================================================================

_RENDER_CHOICES = ("websocket", "native", "headless")
_render_mode = "websocket"          # default: always stream
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}")
            sys.exit(1)
        break

# =============================================================================
# STEP 2: Start SimulationApp — MUST be the very first Isaac Sim call
# =============================================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1280,
    "height": 720,
    "hide_ui": False,   # stream full editor UI (set True for viewport-only)
})

# =============================================================================
# STEP 3: Enable WebRTC extension (after SimulationApp, before scene build)
# =============================================================================
if _render_mode == "websocket":
    import subprocess
    from isaacsim.core.utils.extensions import enable_extension

    # Detect Tailscale IP so remote clients (Mac) can connect
    _tailscale_ip = ""
    try:
        _tailscale_ip = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True, timeout=3
        ).strip()
    except Exception:
        pass

    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/livestream/proto", "websocket")
    if _tailscale_ip:
        simulation_app.set_setting("/app/livestream/publicEndpointAddress", _tailscale_ip)
    enable_extension("omni.kit.livestream.webrtc")

    _connect_ip = _tailscale_ip if _tailscale_ip else "localhost"
    print()
    print("=" * 60)
    print("  Isaac Sim WebRTC Example Scene")
    print(f"  Streaming on port : 49100")
    if _tailscale_ip:
        print(f"  Tailscale IP      : {_tailscale_ip}")
    print()
    print(f"  ▶  Mac client: connect to  {_connect_ip} : 49100")
    print("=" * 60)
    print()

# =============================================================================
# STEP 4: Import everything AFTER SimulationApp
# =============================================================================
import threading
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from pxr import UsdLux, UsdGeom, Gf

# =============================================================================
# SCENE BUILDER
# =============================================================================

def build_scene(world: World):
    """Add ground, lights, and three coloured physics cubes."""
    stage = omni.usd.get_context().get_stage()

    # Ground plane
    world.scene.add_default_ground_plane()

    # Distant light (sun)
    distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
    distant.CreateIntensityAttr(1000.0)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sun"))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(315, 0, 45))

    # Dome light (ambient)
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(400.0)

    # Three cubes: red / green / blue
    cube_specs = [
        ("/World/CubeRed",   [0.0,  0.0, 0.5], (0.85, 0.15, 0.15)),
        ("/World/CubeGreen", [0.5,  0.0, 0.5], (0.15, 0.75, 0.15)),
        ("/World/CubeBlue",  [-0.5, 0.0, 0.5], (0.15, 0.35, 0.90)),
    ]
    cubes = []
    for path, pos, color in cube_specs:
        name = path.split("/")[-1].lower()
        cube = DynamicCuboid(
            prim_path=path,
            name=name,
            position=np.array(pos),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array(color),
            mass=0.5,
        )
        world.scene.add(cube)
        cubes.append(cube)
        print(f"  ✓ Added cube at {pos}  [{path.split('/')[-1]}]")

    return cubes


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def interactive_loop(world: World, cubes: list):
    """Background thread: accept commands while the render loop runs."""
    print()
    print("=" * 50)
    print("  Commands:")
    print("    g           — toggle gravity")
    print("    d <x> <y> <z> — drop a cube at position")
    print("    r           — reset simulation")
    print("    p           — print object positions")
    print("    q           — quit")
    print("=" * 50)
    print()

    gravity_on = True
    extra_cubes = []

    while simulation_app.is_running():
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "q":
            print("Quitting...")
            simulation_app.close()
            break

        elif cmd == "r":
            world.reset()
            print("  ✓ Simulation reset")

        elif cmd == "g":
            gravity_on = not gravity_on
            world.get_physics_context().set_gravity(
                -9.81 if gravity_on else 0.0
            )
            print(f"  ✓ Gravity {'ON  (-9.81 m/s²)' if gravity_on else 'OFF (0.0 m/s²)'}")

        elif cmd == "p":
            stage = omni.usd.get_context().get_stage()
            for cube in cubes + extra_cubes:
                pos = cube.get_world_pose()[0]
                name = cube.prim_path.split("/")[-1]
                print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        elif cmd == "d" and len(parts) >= 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                idx = len(extra_cubes)
                path = f"/World/DroppedCube_{idx}"
                cube = DynamicCuboid(
                    prim_path=path,
                    name=f"dropped_cube_{idx}",
                    position=np.array([x, y, z]),
                    scale=np.array([0.08, 0.08, 0.08]),
                    color=np.array([0.9, 0.6, 0.1]),
                    mass=0.3,
                )
                world.scene.add(cube)
                extra_cubes.append(cube)
                print(f"  ✓ Dropped cube at ({x}, {y}, {z})  [{path}]")
            except ValueError:
                print("  Usage: d <x> <y> <z>")

        else:
            print("  Unknown command. Try: g, d, r, p, q")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Building scene...")
    world = World(stage_units_in_meters=1.0)

    cubes = build_scene(world)

    # Reset world → initialises physics
    world.reset()
    print("  ✓ Physics initialised")
    print("  ✓ Scene ready — watching via WebRTC")
    print()

    # Start CLI in background thread
    cli = threading.Thread(
        target=interactive_loop,
        args=(world, cubes),
        daemon=True,
    )
    cli.start()

    # Main render loop
    try:
        while simulation_app.is_running():
            world.step(render=True)
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()
    print("Done.")


if __name__ == "__main__":
    main()
