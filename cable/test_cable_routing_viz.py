#!/usr/bin/env python3
"""
test_cable_routing_viz.py
─────────────────────────
Visualizes the tendon/cable routing of the cable manipulator in Meshcat.

Cable route (from URDF visual origin xyz values):
  ① link1_base_pulley  — drive pulley, on pulley_htd_5m_60t body  [0.0142, 0, 0.2660]
  ② 623zz  (side A)   — idler bearing top,    on pulley_htd_5m_60t [0.2531,+0.0165, 0.1982]
  ③ 623zz_2 (side B)  — idler bearing bottom, on pulley_htd_5m_60t [0.2569,-0.0150, 0.2018]
  ④ pulley_big         — driven pulley, on link2_tendon body        [0, 0, 0.0045]

The cable exits ① on one side, wraps around ② then ③ (opposite Y — "the other
side"), and drives ④ on link2.  Because ④ is on link2_tendon (q2 body) while
①②③ are on pulley_htd_5m_60t (q1 body), CalcPointsPositions must be called on
the correct body each time.

Interactive: type  q1 q2 [deg]  at the prompt → manipulator moves + cable redraws.

Usage:
    python cable/test_cable_routing_viz.py [--no-springs]
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so this file can be run directly
# via the VS Code play button (cwd = workspaceFolder).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    MeshcatVisualizer,
    StartMeshcat,
    Simulator,
    Parser,
)
from termcolor import colored

from cable.pulley import PulleyBase
from robots.cup_manipulator_tendon import CupManipulatorTendon, create_cable_manipulator_config
from project_utils.viz_cables import (
    print_cable_routing_points,
    draw_cables,
    visualize_cable_routing_top_view,
    visualize_cable_routing_3d,
)


def main():
    ap = argparse.ArgumentParser(description="Cable routing visualization.")
    ap.add_argument("--no-springs", action="store_true",
                    help="Disable endpoint springs (default: springs enabled)")
    args = ap.parse_args()
    springs_enabled = not args.no_springs

    # ── Configuration ─────────────────────────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path="model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=(0.1, 0.1),
    )

    # ── Meshcat ───────────────────────────────────────────────────────────────
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    # ── Plant ─────────────────────────────────────────────────────────────────
    builder     = DiagramBuilder()
    plant       = MultibodyPlant(time_step=0.0)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulatorTendon(config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(plant, position=np.zeros(3), orientation=np.zeros(3))
    manipulator.add_joint_actuators(plant)
    manipulator.add_end_effector_frame(plant)
    plant.Finalize()

    builder.AddSystem(plant)
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    # ── Cable rig — owned by manipulator, mirrors physical assembly ───────────
    manipulator.init_cable_rig(springs_enabled=springs_enabled)
    rig = manipulator.rig

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)

    # ── Home pose ─────────────────────────────────────────────────────────────
    manipulator.set_positions_user_order(plant, plant_ctx, {
        "link1_base":  0.0,
        "link2_link1": 0.0,
    })
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    diagram.ForcedPublish(context)
    manipulator.compute_tangents(plant, plant_ctx)
    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
    print_cable_routing_points(plant, plant_ctx, manipulator, rig)

    # Figure 1 — top view (XY)
    _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, 0.0, 0.0, rig)
    plt.show(block=False)
    plt.pause(0.05)
    _viz_fig = None  # created on first interactive update

    ee = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored("Cable route: drive_pulley → 623zz (A) → 623zz_2 (B, other side) → pulley_big", "yellow"))
    print(colored(f"Home:  q1=0°  q2=0°  →  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m\n", "cyan"))
    print(colored("Enter joint angles in degrees  (e.g.  30  -15)  or Ctrl+C to exit.\n", "yellow"))

    # ── Interactive loop ───────────────────────────────────────────────────────
    try:
        while True:
            raw = input(colored("q1  q2 [deg]: ", "cyan")).strip()
            if not raw:
                continue
            try:
                parts = raw.split()
                if len(parts) != 2:
                    print(colored("  ✗ Expected exactly two values: q1 q2", "red"))
                    continue
                q1_deg, q2_deg = float(parts[0]), float(parts[1])

                manipulator.set_positions_user_order(plant, plant_ctx, {
                    "link1_base":  np.deg2rad(q1_deg),
                    "link2_link1": np.deg2rad(q2_deg),
                })
                plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
                diagram.ForcedPublish(context)
                manipulator.compute_tangents(plant, plant_ctx)
                draw_cables(meshcat, plant, plant_ctx, manipulator, rig)

                # Update plots
                plt.close(_top_fig)
                _top_fig, _ = visualize_cable_routing_top_view(
                    plant, plant_ctx, manipulator, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                if _viz_fig is not None:
                    plt.close(_viz_fig)
                _viz_fig, _ = visualize_cable_routing_3d(
                    plant, plant_ctx, manipulator, PulleyBase.assets_dir,
                    q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                ee = manipulator.get_end_effector_position(plant, plant_ctx)
                print(colored(
                    f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                    f"→  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m",
                    "green",
                ))
            except ValueError:
                print(colored("  ✗ Invalid numbers. Enter two floats: q1 q2", "red"))
    except KeyboardInterrupt:
        print(colored("\n✓ Stopped.", "green"))


if __name__ == "__main__":
    main()
