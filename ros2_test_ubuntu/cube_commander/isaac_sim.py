"""
Isaac Sim Cube Commander — conda env_isaacsim
=============================================
Native ROS 2 bridge architecture (second pipe eliminated):
    Drake → [OS pipe] → ros2_publisher.py → [DDS] → this script

The isaacsim.ros2.bridge extension is enabled BEFORE rclpy is imported,
allowing Isaac Sim to participate in ROS 2 DDS natively.
A daemon thread runs rclpy.spin() while the main thread runs world.step().

Usage:
    bash ros2_test_ubuntu/cube_commander/run_isaac.sh
"""

# ============================================================================
# MUST BE FIRST — Isaac Sim requirement
# ============================================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width":    1280,
    "height":   720,
    "window_title": "Drake → ROS 2 → Isaac Sim: Cube Commander (Native Bridge)",
})

# ============================================================================
# ENABLE ROS 2 BRIDGE — must happen before importing rclpy
# ============================================================================
import omni.kit.app
_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# ============================================================================
# IMPORTS (after SimulationApp + extension enable)
# ============================================================================
import threading
import queue
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point

import omni
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from pxr import UsdGeom

# BEST_EFFORT + depth=1: always deliver the latest command, never queue stale ones.
# Must match the QoS declared in ros2_publisher.py or DDS silently drops the connection.
_CONTROL_QOS = QoSProfile(
    reliability = ReliabilityPolicy.BEST_EFFORT,
    history     = HistoryPolicy.KEEP_LAST,
    depth       = 1,
)


# ============================================================================
# ROS 2 SUBSCRIBER NODE — runs inside the Isaac Sim process
# ============================================================================
class CubeSubscriberNode(Node):
    """
    Subscribes to /cube_target_pos and pushes received positions into a
    thread-safe queue for the main simulation loop to consume.
    """

    def __init__(self, position_queue: queue.Queue):
        super().__init__('isaac_sim_cube_node')
        self._queue = position_queue
        self.create_subscription(
            Point,
            '/cube_target_pos',
            self._callback,
            _CONTROL_QOS,
        )
        self.get_logger().info("Subscribed to /cube_target_pos — waiting for Drake commands.")

    def _callback(self, msg: Point):
        self._queue.put(np.array([msg.x, msg.y, msg.z]))


# ============================================================================
# ISAAC SIM SCENE SETUP
# ============================================================================
def setup_scene(world: World):
    """Create ground plane and the Drake-controlled blue cube."""
    # Set Z-up so Isaac Sim matches Drake / ROS2 convention:
    #   X = forward, Y = left, Z = up
    # Without this Isaac Sim defaults to Y-up (OpenGL/USD convention)
    # where Drake's X maps to Isaac Sim's Z.
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world.scene.add_default_ground_plane()

    # VisualCuboid: no physics/gravity — set_world_pose() is always respected.
    # z=0.1 = half of 0.2m scale → cube sits flush on the ground plane.
    cube = world.scene.add(
        VisualCuboid(
            prim_path = "/World/DrakeCube",
            name      = "drake_cube",
            position  = np.array([0.0, 0.0, 0.1]),
            scale     = np.array([0.2, 0.2, 0.2]),
            color     = np.array([0.2, 0.6, 1.0]),
        )
    )
    print("[Isaac Sim] Blue cube created at x=0.0, y=0.0, z=0.1 (on ground)")
    return cube


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  Drake → ROS 2 → Isaac Sim: Cube Commander (Native Bridge)")
    print("=" * 60)
    print("  Subscribing to /cube_target_pos via isaacsim.ros2.bridge")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # Initialise ROS 2 and start subscriber node
    rclpy.init()
    position_queue = queue.Queue()
    ros_node = CubeSubscriberNode(position_queue)

    # Spin rclpy in a daemon thread — main thread is owned by Isaac Sim
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    # Create world
    # physics_dt=1/500: 500 Hz physics substeps resolve contact/joint forces
    #   accurately at 200 Hz impedance control command rates.
    # rendering_dt=1/60: GPU renders at 60 Hz independently of physics.
    world = World(
        stage_units_in_meters = 1.0,
        physics_dt            = 1 / 500,
        rendering_dt          = 1 / 60,
    )
    cube = setup_scene(world)
    world.reset()

    GROUND_Z = 0.1           # half of 0.2m cube scale — keeps cube on ground
    current_position = np.array([0.0, 0.0, GROUND_Z])
    step_count = 0

    # ── Simulation loop ───────────────────────────────────────────────────────
    while simulation_app.is_running():
        world.step(render=True)

        # Drain the queue — keep only the most recent position
        latest = None
        while not position_queue.empty():
            try:
                latest = position_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            # Lock z to ground level regardless of what Drake sends
            candidate = np.array([latest[0], latest[1], GROUND_Z])
            if not np.allclose(candidate, current_position):
                current_position = candidate
                step_count += 1
                cube.set_world_pose(
                    position    = current_position,
                    orientation = np.array([1.0, 0.0, 0.0, 0.0]),
                )
                print(
                    f"[Isaac Sim] step #{step_count} | "
                    f"x={current_position[0]*100:.1f}cm  "
                    f"y={current_position[1]*100:.1f}cm  "
                    f"z={current_position[2]*100:.1f}cm",
                    flush=True,
                )

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
    print("[Isaac Sim] Simulation closed.")


if __name__ == '__main__':
    main()
