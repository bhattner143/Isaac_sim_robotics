#!/usr/bin/env python3
"""
Test script: Load manipulator_hybrid_planar_fusion URDF in Meshcat.
Press Enter to start joint actuation.
"""

import time
import numpy as np
from pydrake.all import (
    StartMeshcat,
    MeshcatVisualizer,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RevoluteJoint,
    PackageMap,
)
from pathlib import Path


def main():
    urdf_path = (
        Path(__file__).parent
        # / "model_using_onshape_to_robot"
        # / "manipulator_hybrid_planar_fusion"
        / "manipulator_hybrid_planar_fusion_obj.urdf"
    )

    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}")
        return 1

    assets_path = urdf_path.parent / "assets"
    if not assets_path.exists():
        print(f"Error: Assets directory not found: {assets_path}")
        return 1

    print(f"Loading URDF: {urdf_path}")
    print(f"Assets:      {assets_path}")

    meshcat = StartMeshcat()
    print(f"✓ Meshcat: {meshcat.web_url()}")

    builder = DiagramBuilder()

    # AddMultibodyPlantSceneGraph handles all plant↔scene_graph wiring
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    
    # Register the assets package directory with the parser
    package_map = parser.package_map()
    package_map.Add("assets", str(assets_path))
    
    model = parser.AddModels(str(urdf_path))[0]

    # Fix base_link to the world so the robot doesn't float
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_aka_shoulder_transmission", model))

    # Add actuators to every revolute joint BEFORE Finalize()
    joint_indices = plant.GetJointIndices(model)
    revolute_joints = []
    for idx in joint_indices:
        joint = plant.get_joint(idx)
        if isinstance(joint, RevoluteJoint):
            revolute_joints.append(joint)

    plant.Finalize()

    # Report joint ordering
    print(f"\n✓ Model: {len(joint_indices)} joints, {len(revolute_joints)} revolute:")
    for i, j in enumerate(revolute_joints):
        print(f"  [{i}] {j.name()}")

    # Add Meshcat visualizer
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    print("\n" + "=" * 60)
    print("  Open browser at:", meshcat.web_url())
    print("  Starting joint actuation in 2 seconds...")
    print("=" * 60)
    time.sleep(2)  # Give user time to open browser

    print(f"\n▶ Animating {len(revolute_joints)} joints (sinusoidal, 10 s)...")
    print("  Ctrl+C to stop early")

    n = len(revolute_joints)
    nq = plant.num_positions(model)   # authoritative DOF count from Drake
    print(f"  num_positions(model) = {nq}")

    dt = 0.02  # 50 Hz update rate
    t = 0.0
    try:
        while t < 10.0:
            q = np.array([
                1.0 * np.sin(2 * np.pi * 0.5 * t + i * np.pi / max(nq, 1))
                for i in range(nq)
            ])
            plant.SetPositions(plant_context, model, q)
            diagram.ForcedPublish(context)
            time.sleep(dt)
            t += dt
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")

    print(f"\n✓ Done  (t = {t:.2f} s) — replay available in Meshcat")
    return 0


if __name__ == "__main__":
    exit(main())
