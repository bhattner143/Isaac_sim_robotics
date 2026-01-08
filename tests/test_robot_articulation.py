"""
Franka Panda Articulation Test with Target Sphere Visualization

Loads the Franka Panda robot, applies smooth joint test patterns to
its articulation, and moves a visual target sphere around the workspace
for easy observation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from isaacsim import SimulationApp

# -----------------------------------------------------------------------------#
# Argument parsing (must happen before SimulationApp init)
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cpu",
    help="Device used for generating joint test trajectories.",
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# Isaac Sim imports that rely on SimulationApp
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.timeline
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Sphere
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path


# -----------------------------------------------------------------------------#
# Dataclasses
# -----------------------------------------------------------------------------#
@dataclass
class RobotParams:
    usd_path: str
    prim_path: str
    variants: List[tuple[str, str]]
    default_dof_positions: List[float]
    ee_link_name: str
    num_arm_dof: int = 7


@dataclass
class TargetParams:
    prim_path: str
    radius: float
    color: List[float]
    region_center: List[float]
    region_size: float


@dataclass
class SimulationParams:
    num_trials: int
    steps_per_trial: int
    device: str
    test_amplitude: float


# -----------------------------------------------------------------------------#
# Franka articulation controller (no IK, just patterned joint targets)
# -----------------------------------------------------------------------------#
class FrankaController:
    def __init__(self, robot_params: RobotParams, device: str):
        self.robot_params = robot_params
        self.device = torch.device(device)
        self.robot: Optional[Articulation] = None
        self.total_dofs = len(robot_params.default_dof_positions)
        self.default_dof_tensor = torch.tensor(
            robot_params.default_dof_positions, device=self.device
        )
        self.arm_frequencies = torch.linspace(
            0.6, 1.4, robot_params.num_arm_dof, device=self.device
        )

    def initialize(self):
        print("\nInitializing Franka articulation …")
        self.robot = Articulation(self.robot_params.prim_path)
        self.robot.initialize()
        self.robot.set_default_state(dof_positions=self.robot_params.default_dof_positions)
        print("✓ Franka ready for articulation testing.")

    def reset_to_default(self):
        if self.robot:
            self.robot.reset_to_default_state()

    def build_test_targets(
        self,
        step_index: int,
        steps_per_trial: int,
        trial_index: int,
        amplitude: float,
    ) -> torch.Tensor:
        """Create smooth sinusoidal joint offsets for the arm DOFs."""
        if steps_per_trial <= 1:
            phase_progress = 0.0
        else:
            phase_progress = step_index / (steps_per_trial - 1)

        time_scalar = 2.0 * torch.pi * phase_progress
        phase_shift = trial_index * 0.4

        offsets = amplitude * torch.sin(self.arm_frequencies * time_scalar + phase_shift)

        targets = self.default_dof_tensor.clone()
        targets[: self.robot_params.num_arm_dof] += offsets
        return targets

    def apply_joint_targets(self, targets: torch.Tensor):
        if not self.robot:
            return
        self.robot.set_dof_position_targets(targets.detach().cpu().numpy())


# -----------------------------------------------------------------------------#
# Scene manager
# -----------------------------------------------------------------------------#
class SceneManager:
    def __init__(
        self,
        robot_params: RobotParams,
        target_params: TargetParams,
        sim_params: SimulationParams,
    ):
        self.robot_params = robot_params
        self.target_params = target_params
        self.sim_params = sim_params
        self.robot_controller: Optional[FrankaController] = None
        self.target_sphere: Optional[Sphere] = None

    def initialize_stage(self):
        print("Setting up stage …")
        SimulationManager.set_physics_sim_device(self.sim_params.device)
        simulation_app.update()

        stage_utils.create_new_stage(template="sunlight")

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + self.robot_params.usd_path,
            path=self.robot_params.prim_path,
            variants=self.robot_params.variants,
        )

        material = PreviewSurfaceMaterial("/World/Looks/TargetSphere")
        material.set_input_values("diffuseColor", self.target_params.color)

        self.target_sphere = Sphere(
            self.target_params.prim_path,
            radii=[self.target_params.radius],
            reset_xform_op_properties=True,
        )
        self.target_sphere.apply_visual_materials(material)
        print("✓ Stage initialized with robot and visual target.")

    def initialize_robot(self):
        self.robot_controller = FrankaController(self.robot_params, self.sim_params.device)
        self.robot_controller.initialize()

    def _sample_target_position(self) -> torch.Tensor:
        center = torch.tensor(self.target_params.region_center, device=self.sim_params.device)
        random_offset = torch.rand(3, device=self.sim_params.device) - 0.5
        return center + self.target_params.region_size * random_offset

    def _update_target_visual(self, position: torch.Tensor):
        if not self.target_sphere:
            return
        pos_np = position.detach().cpu().numpy().reshape(1, 3)
        quat_np = np.array([[0.0, 0.0, 0.0, 1.0]])
        self.target_sphere.set_world_poses(positions=pos_np, orientations=quat_np)

    def run_simulation(self):
        print("\nStarting articulation tests …")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for trial in range(self.sim_params.num_trials):
            target_position = self._sample_target_position()
            self._update_target_visual(target_position)

            self.robot_controller.reset_to_default()
            simulation_app.update()

            print(
                f"Trial {trial + 1}/{self.sim_params.num_trials} | "
                f"sphere @ ({target_position[0]:.3f}, "
                f"{target_position[1]:.3f}, {target_position[2]:.3f})"
            )

            for step in range(self.sim_params.steps_per_trial):
                targets = self.robot_controller.build_test_targets(
                    step_index=step,
                    steps_per_trial=self.sim_params.steps_per_trial,
                    trial_index=trial,
                    amplitude=self.sim_params.test_amplitude,
                )
                self.robot_controller.apply_joint_targets(targets)
                simulation_app.update()

        print("\n✓ Test sequence complete! Keeping simulation running...")
        print("  Press Ctrl+C or close the window to exit.\n")
        
        # Keep simulation running to show visualization
        while simulation_app.is_running():
            simulation_app.update()


# -----------------------------------------------------------------------------#
# Main entry point
# -----------------------------------------------------------------------------#
def main():
    print("=" * 72)
    print("Franka Panda Articulation Test (Visualization Only)")
    print("=" * 72)

    robot_params = RobotParams(
        usd_path="/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        prim_path="/World/Franka",
        variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
        default_dof_positions=[0.0, -0.3, 0.0, -2.0, 0.0, 2.5, 0.8, 0.02, 0.02],
        ee_link_name="panda_hand",
        num_arm_dof=7,
    )

    target_params = TargetParams(
        prim_path="/World/TargetSphere",
        radius=0.05,
        color=[1.0, 0.2, 0.2],
        region_center=[0.5, 0.0, 0.5],
        region_size=0.25,
    )

    sim_params = SimulationParams(
        num_trials=10,
        steps_per_trial=120,
        device=args.device,
        test_amplitude=0.25,
    )

    scene = SceneManager(robot_params, target_params, sim_params)

    try:
        scene.initialize_stage()
        scene.initialize_robot()
        scene.run_simulation()
        print("\nArticulation tests completed!")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()