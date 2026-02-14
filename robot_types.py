"""
Dataclass definitions for robot configuration and scene management.
Supports multi-manipulator setups with type-safe configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

Vec3 = Tuple[float, float, float]
Vec6 = Tuple[float, float, float, float, float, float]  # For 6-DOF poses

@dataclass(frozen=True)
class Pose:
    """Represents a 6-DOF pose in 3D space."""
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)  # Roll, Pitch, Yaw in radians
    
    def to_tuple(self) -> Vec6:
        """Convert to 6-element tuple (x, y, z, roll, pitch, yaw)."""
        return (*self.xyz, *self.rpy)


@dataclass(frozen=True)
class JointConfig:
    """Configuration for a single joint."""
    position: float = 0.0
    damping: float = 0.0
    stiffness: float = 0.0
    friction: float = 0.0
    
    # Optional limits (can override URDF values)
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    effort_limit: Optional[float] = None


@dataclass
class ManipulatorConfig:
    """Configuration for a single manipulator/robot."""
    name: str
    urdf_path: str
    joint_configs: Dict[str, JointConfig]
    
    # Base pose in world frame (or parent frame if mounted)
    base_pose: Pose = Pose()
    
    # For stacking manipulators (e.g., one mounted on another)
    parent_manipulator: Optional[str] = None
    parent_link: Optional[str] = None  # Which link of parent to attach to
    
    # Package mapping for mesh loading (package_name -> directory_path)
    package_map: Dict[str, str] = field(default_factory=dict)
    
    def get_urdf_path(self) -> Path:
        """Get the absolute path to the URDF file."""
        return Path(self.urdf_path).resolve()
    
    def get_joint_position(self, joint_name: str) -> float:
        """Get initial position for a joint."""
        return self.joint_configs.get(joint_name, JointConfig()).position
    
    def get_joint_positions_dict(self) -> Dict[str, float]:
        """Get all joint positions as a dictionary."""
        return {name: config.position for name, config in self.joint_configs.items()}


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    enabled: bool = True  # Enable Meshcat visualization
    plot_frames: bool = True  # Plot coordinate frames in Meshcat
    interactive: bool = True  # Enable interactive play/pause/repeat controls
    realtime_rate: float = 0.5  # 1.0 = real-time, 0.5 = half speed
    update_every_step: bool = True  # Update Meshcat every simulation step
    print_interval: float = 0.25  # Print status every N seconds (terminal output)
    logging_interval: float = 0.02  # Data logging frequency for smooth plots (50 Hz)
    
    # Meshcat settings
    meshcat_host: str = "localhost"
    meshcat_port: int = 7000
    show_frames: bool = False  # Show frames in scene graph
    show_contact_forces: bool = False  # Show contact forces
    show_hydroelastic: bool = True  # Show hydroelastic contact visualization


@dataclass
class SimulationConfig:
    """Configuration for the simulation environment."""
    mode: str = "simulation"  # Simulation mode: 'scene-viz', 'simulation', 'joint-motion', 'run-all-jts'
    timestep: float = 0.001
    simulation_time: float = 10.0
    gravity: Vec3 = (0.0, 0.0, -9.81)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class PendulumConfig:
    """Configuration for a 3D pendulum with 2-DOF gimbal joints."""
    mass: float = 0.5  # kg
    length: float = 0.2  # meters (from pivot to COM)
    radius: float = 0.05  # meters (ball radius)
    damping: float = 0.1  # Joint damping coefficient
    attachment_point: Vec3 = (-1.2545, 0.0, -0.188125)  # Attachment point on parent body
    initial_pitch: float = 0.0  # degrees - initial swing angle from vertical
    initial_roll: float = 0.0  # degrees - initial roll angle
    name: str = "pendulum"


@dataclass
class CartPendulumConfig:
    """Configuration for cart-pendulum 3D system with 2D cart motion."""
    cart_mass: float = 1.0
    cart_size: float = 0.1
    cart_damping: float = 0.0
    pendulum_mass: float = 0.5
    pendulum_length: float = 0.2
    pendulum_radius: float = 0.05
    pendulum_damping: float = 0.0
    attachment_offset: Vec3 = (0.0, 0.0, 0.0)
    initial_cart_x: float = 0.0
    initial_cart_y: float = 0.0
    initial_pitch: float = 0.0
    initial_roll: float = 0.0
    name: str = "cart_pendulum"


@dataclass
class SceneConfig:
    """Configuration for a complete multi-robot scene."""
    name: str
    manipulators: List[ManipulatorConfig]
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    def get_manipulator(self, name: str) -> Optional[ManipulatorConfig]:
        """Get manipulator config by name."""
        for m in self.manipulators:
            if m.name == name:
                return m
        return None
    
    def get_manipulator_names(self) -> List[str]:
        """Get list of all manipulator names."""
        return [m.name for m in self.manipulators]

def create_simulation_config() -> SimulationConfig:
    """Factory function to create a default SimulationConfig."""
    return SimulationConfig(
        mode="simulation",
        timestep=0.001,
        simulation_time=10.0,
        gravity=(0.0, 0.0, -9.81)
    )
# Factory functions for common configurations

def create_cup_manipulator_config(
    urdf_path: str,
    joint_angles: Tuple[float, float] = (0.0, 0.0),
    damping: Tuple[float, float] = (0.1, 0.1),
    stiffness: Tuple[float, float] = (0.0, 0.0),
    friction: Tuple[float, float] = (0.0, 0.0),
) -> ManipulatorConfig:
    """
    Create a ManipulatorConfig for the cup manipulator.
    
    Args:
        urdf_path: Path to URDF file
        joint_angles: Initial angles [link1_base, link2_link1] in radians
        damping: Damping coefficients [link1_base, link2_link1]
        stiffness: Stiffness coefficients [link1_base, link2_link1]
        friction: Friction coefficients [link1_base, link2_link1]
    
    Note:
        This is a 2-DOF manipulator. Pendulum will be added programmatically.
    """
    
    # Get the directory containing the URDF for package mapping
    urdf_dir = str(Path(urdf_path).parent)
    
    # Joint names in order - MUST match the actual URDF joint names!
    joint_names = ["link1_base", "link2_link1"]
    
    joint_configs = {}
    for i, joint_name in enumerate(joint_names):
        joint_configs[joint_name] = JointConfig(
            position=joint_angles[i],
            damping=damping[i],
            stiffness=stiffness[i],
            friction=friction[i]
        )
    
    return ManipulatorConfig(
        name="cup_manipulator",
        urdf_path=urdf_path,
        joint_configs=joint_configs,
        base_pose=Pose(),
        package_map={"assets": urdf_dir + "/assets/"}
    )


def create_ball_config(
    urdf_path: str,
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> ManipulatorConfig:
    """
    Create a ManipulatorConfig for a free-floating ball.
    
    Args:
        urdf_path: Path to ball URDF file
        initial_position: Initial XYZ position of the ball
    """
    
    # Get the directory containing the URDF for package mapping
    urdf_dir = str(Path(urdf_path).parent)
    
    return ManipulatorConfig(
        name="ball",
        urdf_path=urdf_path,
        joint_configs={},  # No actuated joints - free floating
        base_pose=Pose(xyz=initial_position),
        package_map={"ball_assets": urdf_dir + "/assets/"}
    )


def create_cart_pendulum_config(
    cart_mass: float = 1.0,
    cart_size: float = 0.1,
    cart_damping: float = 0.0,
    pendulum_mass: float = 0.5,
    pendulum_length: float = 0.2,
    pendulum_radius: float = 0.05,
    pendulum_damping: float = 0.0,
    attachment_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_cart_x: float = 0.0,
    initial_cart_y: float = 0.0,
    initial_pitch: float = 0.0,
    initial_roll: float = 0.0,
    name: str = "cart_pendulum",
) -> CartPendulumConfig:
    """Factory function to create a CartPendulumConfig."""
    return CartPendulumConfig(
        cart_mass=cart_mass,
        cart_size=cart_size,
        cart_damping=cart_damping,
        pendulum_mass=pendulum_mass,
        pendulum_length=pendulum_length,
        pendulum_radius=pendulum_radius,
        pendulum_damping=pendulum_damping,
        attachment_offset=attachment_offset,
        initial_cart_x=initial_cart_x,
        initial_cart_y=initial_cart_y,
        initial_pitch=initial_pitch,
        initial_roll=initial_roll,
        name=name,
    )


def create_pendulum_config(
    mass: float = 0.5,
    length: float = 0.2,
    radius: float = 0.05,
    damping: float = 0.1,
    attachment_point: Tuple[float, float, float] = (-1.2545, 0.0, -0.188125),
    initial_pitch: float = 0.0,
    initial_roll: float = 0.0,
    name: str = "pendulum"
) -> PendulumConfig:
    """
    Create a PendulumConfig for a 3D pendulum with 2-DOF gimbal joints.
    
    Args:
        mass: Mass of pendulum ball (kg)
        length: Length from pivot to COM (m)
        radius: Radius of pendulum ball (m)
        damping: Joint damping coefficient
        attachment_point: Attachment point on parent body (x, y, z)
        initial_pitch: Initial pitch angle in degrees (equilibrium at 180°)
        initial_roll: Initial roll angle in degrees (equilibrium at -180°)
        name: Name prefix for pendulum bodies/joints
    
    Returns:
        PendulumConfig with specified parameters
    """
    return PendulumConfig(
        mass=mass,
        length=length,
        radius=radius,
        damping=damping,
        attachment_point=attachment_point,
        initial_pitch=initial_pitch,
        initial_roll=initial_roll,
        name=name
    )
