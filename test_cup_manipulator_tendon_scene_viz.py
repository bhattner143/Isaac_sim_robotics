#!/usr/bin/env python3
"""
test_cup_manipulator_tendon_scene_viz.py
========================================
Isaac Sim scene-viz counterpart of the PyDrake ``--mode scene-viz``.

Follows the proven pattern from test_combined_urdf.py:
  1. Convert URDF → USD via URDFParseAndImportFile
  2. Add USD to stage via add_reference_to_stage
  3. world.reset()
  4. Create Articulation for joint control

Interactive CLI commands (same as PyDrake scene-viz):
  e <x> <y>     — move EE to (x, y) via analytical IK
  j <q1> <q2>   — set joint angles [degrees]
  p             — print current state
  q / Ctrl+C    — quit

Render modes (--render):
  native     — opens a local Isaac Sim window (default)
  websocket  — headless + WebRTC streaming via omni.kit.livestream.webrtc
               port 49100 (websocket signaling). View with the NVIDIA
               Omniverse Streaming Client app (connect to localhost:49100).
               Download: https://docs.omniverse.nvidia.com/streaming-client
  headless   — fully headless, no display (CI / testing)

NOTE: Isaac Sim streaming is NOT browser-based like Meshcat/PyDrake.
      It requires the Omniverse Streaming Client desktop app.

Usage:
    conda activate env_isaacsim
    python test_cup_manipulator_tendon_scene_viz.py
    python test_cup_manipulator_tendon_scene_viz.py --render websocket
    python test_cup_manipulator_tendon_scene_viz.py --render headless
    python test_cup_manipulator_tendon_scene_viz.py --q1 10 --q2 -20
    python test_cup_manipulator_tendon_scene_viz.py --tilt-roll 0 --tilt-pitch 45
"""

# ============================================================================
# PRE-PARSE --render BEFORE SimulationApp (must be absolutely first)
# SimulationApp() must be the very first Isaac Sim call, so we cannot use
# argparse here — manually scan sys.argv instead.
# ============================================================================
import sys

_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "websocket"  # default
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}, got '{_render_mode}'")
            sys.exit(1)
        break

# ============================================================================
# MUST BE FIRST — Isaac Sim requirement
# ============================================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",   # native → local window
    "width":    1280,
    "height":   720,
    "hide_ui":  False,
})

# Enable WebRTC streaming extension when mode is 'websocket'
if _render_mode == "websocket":
    from isaacsim.core.utils.extensions import enable_extension
    # omni.kit.livestream.webrtc: local WebRTC streaming, port 49100
    # View with NVIDIA Omniverse Streaming Client: connect to localhost:49100
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/livestream/proto", "websocket")
    enable_extension("omni.kit.livestream.webrtc")
    print("\n" + "=" * 60)
    print("  WebRTC streaming enabled (omni.kit.livestream.webrtc)")
    print("  Port: 49100")
    print("  Open the NVIDIA Omniverse Streaming Client app and")
    print("  connect to: localhost:49100")
    print("  https://docs.omniverse.nvidia.com/streaming-client")
    print("=" * 60 + "\n")

# ============================================================================
# IMPORTS (after SimulationApp)
# ============================================================================
import argparse
import threading
import numpy as np
from pathlib import Path

import omni.usd
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Gf
from termcolor import colored

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
)


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cup-manipulator tendon scene-viz (Isaac Sim)',
    )
    parser.add_argument(
        '--render',
        choices=_RENDER_CHOICES,
        default=_render_mode,
        help='Render mode: native (local window) | websocket (Omniverse Streaming Client, port 49100) | headless (no display)',
    )
    parser.add_argument(
        '--q1', type=float, default=10.0,
        help='Initial q1 angle [deg] for link1_base  (default: 10)',
    )
    parser.add_argument(
        '--q2', type=float, default=-10.0,
        help='Initial q2 angle [deg] for link2_link1  (default: -10)',
    )
    parser.add_argument(
        '--tilt-roll', type=float, default=0.0,
        help='Base roll tilt [deg]  (default: 0)',
    )
    parser.add_argument(
        '--tilt-pitch', type=float, default=0.0,
        help='Base pitch tilt [deg]  (default: 0)',
    )
    parser.add_argument(
        '--damping', type=float, nargs=2, default=[0.05, 0.05],
        metavar=('D1', 'D2'),
        help='Joint damping [Nm·s/rad]  (default: 0.05 0.05)',
    )
    return parser.parse_args()


# ============================================================================
# SCENE SETUP
# ============================================================================

def add_lighting(stage):
    """Add distant + dome lights (same pattern as test_import_plate_dips_to_isaac_franka.py)."""
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(1000.0)
    distant_prim = stage.GetPrimAtPath("/World/DistantLight")
    xformable = UsdGeom.Xformable(distant_prim)
    xformable.AddRotateXYZOp().Set(Gf.Vec3d(315.0, 0, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(300.0)
    print(colored("✓ Lights added", 'green'))


def add_ee_marker(stage, manipulator: CupManipulatorTendonIsaac):
    """Add a small visible sphere at the EE location for visual reference."""
    ee_pos = manipulator.get_end_effector_position()
    marker_path = "/World/EE_Marker"
    sphere = UsdGeom.Sphere.Define(stage, marker_path)
    sphere.GetRadiusAttr().Set(0.008)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])  # red

    xformable = UsdGeom.Xformable(sphere.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(
        float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
    ))
    print(colored(
        f"✓ EE marker at ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})",
        'magenta'
    ))


def update_ee_marker(stage, position: np.ndarray):
    """Move the EE marker sphere to a new position."""
    marker_prim = stage.GetPrimAtPath("/World/EE_Marker")
    if marker_prim.IsValid():
        xformable = UsdGeom.Xformable(marker_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(
            float(position[0]), float(position[1]), float(position[2])
        ))


# ============================================================================
# INTERACTIVE COMMAND LOOP
# ============================================================================

def interactive_loop(manipulator: CupManipulatorTendonIsaac, world: World, stage):
    """CLI command loop — runs in a background thread.

    Commands:
      e <x> <y>     — IK: move EE to world XY
      j <q1> <q2>   — set joints [degrees]
      p             — print current state
      q             — quit
    """
    print("\n" + "=" * 60)
    print("  Interactive Scene-Viz Commands:")
    print("    e <x> <y>     — move EE (IK)")
    print("    j <q1> <q2>   — set joints [deg]")
    print("    p             — print state")
    print("    q             — quit")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd == 'q':
            print("[Isaac Sim] Quitting...")
            simulation_app.close()
            break

        elif cmd == 'p':
            q = manipulator.get_positions_user_order()
            v = manipulator.get_velocities_user_order()
            ee = manipulator.get_end_effector_position()
            print(f"  q1={np.rad2deg(q[0]):+7.2f}°  q2={np.rad2deg(q[1]):+7.2f}°")
            print(f"  q1_dot={v[0]:+.4f}  q2_dot={v[1]:+.4f} rad/s")
            print(f"  EE=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) m")

        elif cmd == 'e' and len(parts) >= 3:
            try:
                tx, ty = float(parts[1]), float(parts[2])
                q_now = manipulator.get_positions_user_order()
                q_sol, ok = manipulator.compute_ik_analytical(
                    np.array([tx, ty]), q_now
                )
                if ok:
                    manipulator.set_positions_user_order(
                        {manipulator.JT1_NAME: q_sol[0],
                         manipulator.JT2_NAME: q_sol[1]}
                    )
                    world.step(render=True)
                    simulation_app.update()
                    ee = manipulator.get_end_effector_position()
                    update_ee_marker(stage, ee)
                    print(colored(
                        f"  ✓ IK → q1={np.rad2deg(q_sol[0]):+.2f}°  "
                        f"q2={np.rad2deg(q_sol[1]):+.2f}°  "
                        f"EE=({ee[0]:.4f}, {ee[1]:.4f})",
                        'green'
                    ))
                else:
                    print(colored(
                        f"  ✗ IK failed for target ({tx}, {ty})", 'red'
                    ))
            except ValueError:
                print("  Usage: e <x_m> <y_m>")

        elif cmd == 'j' and len(parts) >= 3:
            try:
                q1_deg, q2_deg = float(parts[1]), float(parts[2])
                manipulator.set_positions_user_order(
                    {manipulator.JT1_NAME: np.deg2rad(q1_deg),
                     manipulator.JT2_NAME: np.deg2rad(q2_deg)}
                )
                world.step(render=True)
                simulation_app.update()
                ee = manipulator.get_end_effector_position()
                update_ee_marker(stage, ee)
                print(colored(
                    f"  ✓ Set q1={q1_deg:+.2f}°  q2={q2_deg:+.2f}°  "
                    f"EE=({ee[0]:.4f}, {ee[1]:.4f})",
                    'green'
                ))
            except ValueError:
                print("  Usage: j <q1_deg> <q2_deg>")

        else:
            print("  Unknown command. Try: e, j, p, q")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("  Cup Manipulator Tendon — Scene Viz (Isaac Sim)")
    print("=" * 60)

    # ── Config (same factory as PyDrake) ─────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path="model_using_onshape_to_robot/manipulator_cable_isaac/manipulator_cable_obj.urdf",
        joint_angles={
            'link1_base':   np.deg2rad(args.q1),
            'link2_link1':  np.deg2rad(args.q2),
        },
        damping=tuple(args.damping),
        tilt_roll_deg=args.tilt_roll,
        tilt_pitch_deg=args.tilt_pitch,
    )

    # ── Manipulator config (created before World) ──────────────────────────
    manipulator = CupManipulatorTendonIsaac(config, enable_visualization=True)

    # Step 1: Convert URDF → USD — MUST happen BEFORE World() creation
    # (matches test_combined_urdf.py order)
    manipulator.prepare_usd()

    # ── World (same pattern as test_combined_urdf.py) ──────────────────────
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()

    # ── Lighting ─────────────────────────────────────────────────────────────
    add_lighting(stage)

    # ── Manipulator: add USD reference to stage ──────────────────────────────
    # Step 2: Add pre-converted USD to stage (after World creation)
    manipulator.load_urdf()

    # Step 2: Apply base transform (before world.reset)
    base_orientation = np.array([
        np.deg2rad(args.tilt_roll),
        np.deg2rad(args.tilt_pitch),
        0.0,
    ])
    manipulator.weld_base_to_world(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=base_orientation,
    )

    manipulator.set_joint_properties()
    manipulator.add_joint_actuators()
    manipulator.add_end_effector_frame()

    # ── Reset world to initialize physics (MUST be before Articulation) ──────
    world.reset()

    # ── Create Articulation and set initial state (after world.reset) ────────
    manipulator.initialize_state()
    manipulator.set_initial_positions()

    # Warm up — propagate joint positions to renderer (same as test_combined_urdf.py)
    for _ in range(10):
        world.step(render=True)

    # ── EE marker ────────────────────────────────────────────────────────────
    add_ee_marker(stage, manipulator)

    # ── Print initial state ──────────────────────────────────────────────────
    q = manipulator.get_positions_user_order()
    ee = manipulator.get_end_effector_position()
    print(colored(
        f"\n✓ Scene ready:\n"
        f"  q1 = {np.rad2deg(q[0]):+.2f}°  q2 = {np.rad2deg(q[1]):+.2f}°\n"
        f"  EE = ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) m",
        'green', attrs=['bold']
    ))

    # ── Interactive CLI in background thread ─────────────────────────────────
    cli_thread = threading.Thread(
        target=interactive_loop,
        args=(manipulator, world, stage),
        daemon=True,
    )
    cli_thread.start()

    # ── Render loop (main thread, with simulation_app.update) ────────────────
    print("\nPress Ctrl+C or type 'q' to exit...")
    try:
        while simulation_app.is_running():
            world.step(render=True)
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()
    print("[Isaac Sim] Simulation closed.")


if __name__ == '__main__':
    main()
