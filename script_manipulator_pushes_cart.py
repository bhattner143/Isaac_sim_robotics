#!/usr/bin/env python3
"""
Manipulator Pushes Cart - Drake OFC Controller Architecture

═══════════════════════════════════════════════════════════════════════════════
SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Manipulator → Impedance Control → Cart-Pendulum System

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DRAKE DIAGRAM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  state   ┌────────────┐  F_imp   ┌─────────────┐        │
│  │ Manipulator  │──────────>│ Impedance  │─────────>│ Cart-       │        │
│  │ (2-DOF)      │           │ Controller │          │ Pendulum    │        │
│  │              │<──────────│            │          │ (4-DOF)     │        │
│  └──────────────┘  τ        └────────────┘          └─────────────┘        │
│         │                         ▲                                         │
│         │                         │                                         │
│         │                  ┌──────────────┐                                │
│         └─────────────────>│ ZFT          │                                │
│                            │ Reference    │                                │
│                            │ Mass         │                                │
│                            └──────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

SYSTEM 1: Manipulator (Active)
──────────────────────────────
• 2-DOF planar arm
• Impedance control via Jacobian transpose
• ZFT reference mass for trajectory generation

SYSTEM 2: Cart-Pendulum (Passive)
─────────────────────────────────
• 4-DOF: cart (x,y) + pendulum (pitch, roll)
• Receives force from manipulator via impedance
• No active control

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    AbstractValue,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    
    # Visualization
    MeshcatVisualizer,
    StartMeshcat,
    
    # Systems
    VectorLogSink,
    ConstantVectorSource,
    Multiplexer,
    Demultiplexer,
    
    # Geometry/Forces
    SpatialForce,
    ExternallyAppliedSpatialForce,
)

# Local imports
import sys
sys.path.append(str(Path(__file__).parent))

from robot_types import (
    create_cup_manipulator_config,
    create_cart_pendulum_config,
    ManipulatorConfig,
    CartPendulumConfig,
)

from script_cup_manipulator_controller_ofc import (
    CupManipulator,
    CartPendulum3D,
    MuscleDynamics,
    ImpedanceForce,
    ZFTReferenceMass,
    EndEffectorKinematics,
    ManipulatorJacobianTransposeController,
    ImpedanceToCartForce,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Manipulator Pushes Cart System')
parser.add_argument('--duration', type=float, default=5.0, help='Simulation duration (s)')
parser.add_argument('--distance', type=float, default=0.5, help='Desired cart travel distance (m)')
parser.add_argument('--q1_init', type=float, default=-10.0, help='Initial joint 1 angle (deg)')
parser.add_argument('--q2_init', type=float, default=20.0, help='Initial joint 2 angle (deg)')
parser.add_argument('--visualize', action='store_true', default=True, help='Enable visualization')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ManipulatorPushesCartConfig:
    """Configuration for manipulator-pushes-cart simulation.
    
    Architecture: Manipulator → Impedance Force → Cart-Pendulum
    - Manipulator: 2-DOF (active with impedance control)
    - Cart-Pendulum: 4-DOF (passive, receives F_imp force)
    - Control: ZFT reference mass → impedance force → cart force
    """
    # Cart-Pendulum parameters
    cart_mass: float = 5.0  # kg
    cart_damping: float = 0.1  # N·s/m
    pendulum_mass: float = 0.5  # kg
    pendulum_length: float = 0.2  # m
    pendulum_damping: float = 0.1  # N·m·s/rad
    
    # Impedance control parameters (2D, but only X-axis active)
    K_imp: float = 100.0  # N/m (stiffness)
    D_imp: float = 20.0  # N·s/m (damping)
    M_ref: float = 2.0  # kg (reference mass for ZFT)
    
    # Desired motion
    distance: float = 0.5  # m (desired cart travel in X direction)
    duration: float = 5.0  # s (simulation time)
    
    # Initial configuration
    q1_init: float = -10.0  # deg (manipulator joint 1)
    q2_init: float = 20.0  # deg (manipulator joint 2)
    initial_pitch: float = 0.0  # rad (pendulum pitch angle α)
    initial_roll: float = 0.0  # rad (pendulum roll angle β)


@dataclass
class MuscleDynamicsConfig:
    """Configuration for muscle/actuator dynamics.
    
    First-order dynamics: τ_m Ḟ = u - F
    """
    muscle_tau: float = 0.03  # s (muscle time constant)
    muscle_initial_force: np.ndarray = field(default_factory=lambda: np.zeros(2))  # N (2D)


@dataclass
class ImpedanceForceConfig:
    """Configuration for 2D impedance force controller.
    
    F_imp = K_p·(y_ref - y) + K_d·(ẏ_ref - ẏ)
    """
    kp: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0]))  # N/m (2×2 stiffness)
    kd: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0]))  # N·s/m (2×2 damping)
    force_limit: Optional[float] = None  # N (optional saturation)


@dataclass
class ZFTReferenceMassConfig:
    """Configuration for 2D ZFT reference mass.
    
    Dynamics:
        ẏ_ref = v_ref
        M_ref·v̇_ref = K_p·(y - y_ref) + K_d·(v - v_ref) + F
    """
    Mh: np.ndarray = field(default_factory=lambda: np.diag([2.0, 2.0]))  # kg (2×2 reference mass)
    kp: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0]))  # N/m (coupling stiffness)
    kd: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0]))  # N·s/m (coupling damping)
    yref0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # m (initial ref position)
    vref0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # m/s (initial ref velocity)


@dataclass
class SimulationConfig:
    """Global simulation parameters."""
    timestep: float = 0.001  # s (1 kHz)
    realtime_rate: float = 1.0  # 1.0 = real-time
    print_interval: float = 0.5  # s
    logging_interval: float = 0.02  # s (50 Hz)


# ============================================================================
# CONFIG CREATION FUNCTIONS
# ============================================================================

def create_manipulator_pushes_cart_config(
    cart_mass: float = 5.0,
    cart_damping: float = 0.1,
    pendulum_mass: float = 0.5,
    pendulum_length: float = 0.2,
    pendulum_damping: float = 0.1,
    K_imp: float = 100.0,
    D_imp: float = 20.0,
    M_ref: float = 2.0,
    distance: float = 0.5,
    duration: float = 5.0,
    q1_init: float = -10.0,
    q2_init: float = 20.0,
    initial_pitch: float = 0.0,
    initial_roll: float = 0.0,
) -> ManipulatorPushesCartConfig:
    """Create ManipulatorPushesCartConfig with custom parameters."""
    return ManipulatorPushesCartConfig(
        cart_mass=cart_mass,
        cart_damping=cart_damping,
        pendulum_mass=pendulum_mass,
        pendulum_length=pendulum_length,
        pendulum_damping=pendulum_damping,
        K_imp=K_imp,
        D_imp=D_imp,
        M_ref=M_ref,
        distance=distance,
        duration=duration,
        q1_init=q1_init,
        q2_init=q2_init,
        initial_pitch=initial_pitch,
        initial_roll=initial_roll,
    )


def create_muscle_dynamics_config_2d(
    muscle_tau: float = 0.03,
    muscle_initial_force: np.ndarray = None,
) -> MuscleDynamicsConfig:
    """Create MuscleDynamicsConfig for 2D system."""
    if muscle_initial_force is None:
        muscle_initial_force = np.zeros(2)
    return MuscleDynamicsConfig(
        muscle_tau=muscle_tau,
        muscle_initial_force=muscle_initial_force,
    )


def create_impedance_force_config_2d(
    kp: np.ndarray = None,
    kd: np.ndarray = None,
    force_limit: Optional[float] = None,
) -> ImpedanceForceConfig:
    """Create ImpedanceForceConfig for 2D system."""
    if kp is None:
        kp = np.diag([100.0, 100.0])
    if kd is None:
        kd = np.diag([20.0, 20.0])
    return ImpedanceForceConfig(
        kp=kp,
        kd=kd,
        force_limit=force_limit,
    )


def create_zft_reference_mass_config_2d(
    Mh: np.ndarray = None,
    kp: np.ndarray = None,
    kd: np.ndarray = None,
    yref0: np.ndarray = None,
    vref0: np.ndarray = None,
) -> ZFTReferenceMassConfig:
    """Create ZFTReferenceMassConfig for 2D system."""
    if Mh is None:
        Mh = np.diag([2.0, 2.0])
    if kp is None:
        kp = np.diag([100.0, 100.0])
    if kd is None:
        kd = np.diag([20.0, 20.0])
    if yref0 is None:
        yref0 = np.array([0.0, 0.0])
    if vref0 is None:
        vref0 = np.array([0.0, 0.0])
    return ZFTReferenceMassConfig(
        Mh=Mh,
        kp=kp,
        kd=kd,
        yref0=yref0,
        vref0=vref0,
    )


def create_simulation_config(
    timestep: float = 0.001,
    realtime_rate: float = 1.0,
    print_interval: float = 0.5,
    logging_interval: float = 0.02,
) -> SimulationConfig:
    """Create SimulationConfig with custom parameters."""
    return SimulationConfig(
        timestep=timestep,
        realtime_rate=realtime_rate,
        print_interval=print_interval,
        logging_interval=logging_interval,
    )


# ============================================================================
# CREATE GLOBAL CONFIG INSTANCES
# ============================================================================

# Main configuration
MANIPULATOR_PUSHES_CART_CONFIG = create_manipulator_pushes_cart_config(
    distance=args.distance,
    duration=args.duration,
    q1_init=args.q1_init,
    q2_init=args.q2_init,
)

# Component configurations
MUSCLE_DYNAMICS_CONFIG = create_muscle_dynamics_config_2d()
IMPEDANCE_FORCE_CONFIG = create_impedance_force_config_2d()
ZFT_REFERENCE_MASS_CONFIG = create_zft_reference_mass_config_2d()

# Global simulation config
SIM_CONFIG = create_simulation_config()


# ============================================================================
# CART FORCE APPLICATOR (Applies impedance force to cart)
# ============================================================================

class CartForceApplicator(LeafSystem):
    """Apply 2D force to cart via spatial forces.
    
    INPUT: cart_force [F_x, F_y]^T (2D)
    OUTPUT: spatial_forces (ExternallyAppliedSpatialForce)
    """
    
    def __init__(self, cart_body_index):
        LeafSystem.__init__(self)
        self.cart_body_index = cart_body_index
        
        # Input: 2D cart force
        self.DeclareVectorInputPort("cart_force", BasicVector(2))
        
        # Output: spatial forces
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_output
        )
    
    def _calc_output(self, context, output):
        """Convert 2D force to spatial force on cart."""
        cart_force_2d = self.GetInputPort("cart_force").Eval(context)
        
        spatial_force = ExternallyAppliedSpatialForce()
        spatial_force.body_index = self.cart_body_index
        spatial_force.F_Bq_W = SpatialForce(
            tau=np.zeros(3),  # No torque
            f=np.array([cart_force_2d[0], cart_force_2d[1], 0.0])  # Force in XY plane
        )
        spatial_force.p_BoBq_B = np.zeros(3)  # Applied at body origin
        
        output.set_value([spatial_force])


# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class ManipulatorPushesCartSceneManager:
    """
    Scene Manager for manipulator-pushes-cart simulation.
    
    Manages:
    - Diagram construction (plant + controller wiring)
    - System creation and configuration
    - Simulator setup and execution
    - Visualization and data logging
    
    Pattern: Following cart_pendulum architecture
    """
    
    def __init__(
        self,
        config: ManipulatorPushesCartConfig,
        visualize: bool = True,
    ):
        """Initialize scene manager.
        
        Args:
            config: Main configuration for simulation
            visualize: Enable/disable visualization
        """
        self.config = config
        self.visualize = visualize
        
        # Drake components (initialized in setup)
        self.builder = None
        self.meshcat = None
        self.plant = None
        self.scene_graph = None
        self.manipulator = None
        self.cart_pendulum = None
        
        # Systems
        self.ee_kinematics = None
        self.zft_ref_mass = None
        self.impedance_force = None
        self.jacobian_controller = None
        self.imp_to_cart = None
        self.cart_force_applicator = None
        
        # Simulation
        self.diagram = None
        self.simulator = None
        self.context = None
        
        # Logging
        self.state_logger = None
        
        print(colored("\n" + "=" * 70, "cyan"))
        print(colored("Manipulator Pushes Cart - Scene Manager", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        print(colored(f"Duration: {config.duration} s", "yellow"))
        print(colored(f"Distance: {config.distance} m", "yellow"))
        print(colored(f"Visualization: {'Enabled' if visualize else 'Disabled'}", "yellow"))
        print(colored("=" * 70 + "\n", "cyan"))
    
    def setup_drake_system(self):
        """Build the complete Drake diagram."""
        print(colored("Building Drake diagram...", "yellow"))
        
        # Initialize builder and meshcat
        self.builder = DiagramBuilder()
        if self.visualize:
            self.meshcat = StartMeshcat()
            print(colored(f"🌐 Meshcat: {self.meshcat.web_url()}\n", "green", attrs=["bold"]))
        
        # Add plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=SIM_CONFIG.timestep
        )
        
        # Load manipulator
        self._load_manipulator()
        
        # Load cart-pendulum
        self._load_cart_pendulum()
        
        # Finalize plant
        self.plant.Finalize()
        
        # Get initial EE position
        self._compute_initial_state()
        
        # Build control system
        self._build_control_system()
        
        # Add visualization
        if self.visualize:
            self._setup_visualization()
        
        # Add logging
        self._setup_logging()
        
        print(colored("✓ Drake diagram built successfully\n", "green"))
    
    def _load_manipulator(self):
        """Load manipulator into plant."""
        print(colored("Loading manipulator...", "yellow"))
        
        urdf_path = Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()
        cup_config = create_cup_manipulator_config(
            urdf_path=str(urdf_path),
            joint_angles=(np.deg2rad(self.config.q1_init), np.deg2rad(self.config.q2_init)),
            damping=(0.1, 0.1),
            stiffness=(0.0, 0.0),
            friction=(0.05, 0.05),
        )
        
        self.manipulator = CupManipulator(cup_config)
        parser = Parser(self.plant)
        self.manipulator.build_in_plant(self.plant, parser, weld_base=True)
        
        print(colored("✓ Manipulator loaded\n", "green"))
    
    def _load_cart_pendulum(self):
        """Load cart-pendulum into plant."""
        print(colored("Creating cart-pendulum system...", "yellow"))
        
        cart_config = create_cart_pendulum_config(
            cart_mass=self.config.cart_mass,
            cart_damping=self.config.cart_damping,
            pendulum_mass=self.config.pendulum_mass,
            pendulum_length=self.config.pendulum_length,
            pendulum_damping=self.config.pendulum_damping,
            initial_cart_x=0.0,  # Will be set to EE position
            initial_cart_y=0.0,
            initial_pitch=self.config.initial_pitch,
            initial_roll=self.config.initial_roll,
        )
        
        self.cart_pendulum = CartPendulum3D(
            cart_config,
            visualize_cart=True,
            add_cart_actuators=True
        )
        
        cart_model_instance = self.plant.AddModelInstance("cart_pendulum")
        self.cart_pendulum.attach_to_plant(
            self.plant,
            cart_model_instance,
            register_visuals=True
        )
        
        print(colored("✓ Cart-Pendulum created\n", "green"))
    
    def _compute_initial_state(self):
        """Compute initial end-effector position."""
        temp_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(
            temp_context,
            self.manipulator.model_instance,
            np.array([np.deg2rad(self.config.q1_init), np.deg2rad(self.config.q2_init)])
        )
        
        ee_body = self.plant.GetBodyByName("link2", self.manipulator.model_instance)
        ee_pose = self.plant.EvalBodyPoseInWorld(temp_context, ee_body)
        self.ee_pos_init = ee_pose.translation() + ee_pose.rotation() @ self.manipulator.EE_OFFSET
        
        print(colored(f"Initial EE position: x={self.ee_pos_init[0]:.3f} m, "
                     f"y={self.ee_pos_init[1]:.3f} m\n", "cyan"))
    
    def _build_control_system(self):
        """Build the control system with 2D classes."""
        print(colored("Building control system...", "yellow"))
        
        # Create systems
        self.ee_kinematics = self.builder.AddSystem(
            EndEffectorKinematics(self.plant, self.manipulator)
        )
        
        # ZFT Reference Mass (2D, but Y locked in place for 1D X-only motion)
        # Use very high stiffness in Y to prevent y_ref drift (effectively 1D in X)
        zft_config = create_zft_reference_mass_config_2d(
            Mh=np.diag([self.config.M_ref, self.config.M_ref]),
            kp=np.diag([self.config.K_imp, self.config.K_imp * 1000.0]),  # Y: 1000x stiffer to lock
            kd=np.diag([self.config.D_imp, self.config.D_imp * 100.0]),   # Y: 100x more damped
            yref0=np.array([self.ee_pos_init[0], self.ee_pos_init[1]]),
            vref0=np.array([0.0, 0.0])
        )
        self.zft_ref_mass = self.builder.AddSystem(ZFTReferenceMass(zft_config))
        
        # Impedance Force (2D, but Y very stiff to match locked y_ref)
        imp_config = create_impedance_force_config_2d(
            kp=np.diag([self.config.K_imp, self.config.K_imp * 1000.0]),  # Y: 1000x stiffer
            kd=np.diag([self.config.D_imp, self.config.D_imp * 100.0]),   # Y: 100x more damped
            force_limit=None
        )
        self.impedance_force = self.builder.AddSystem(ImpedanceForce(imp_config))
        
        # Controllers
        self.jacobian_controller = self.builder.AddSystem(
            ManipulatorJacobianTransposeController(self.plant, self.manipulator)
        )
        self.imp_to_cart = self.builder.AddSystem(ImpedanceToCartForce())
        
        # Zero muscle force (no LQR for now) - 2D to match ZFT input
        zero_muscle_force = self.builder.AddSystem(ConstantVectorSource(np.zeros(2)))
        
        # Demultiplexers and multiplexers
        state_demux = self.builder.AddSystem(Demultiplexer([4, 8]))  # [manip(4), cart(8)]
        imp_force_demux = self.builder.AddSystem(Demultiplexer([1, 1]))  # [F_x, F_y]
        ee_state_mux = self.builder.AddSystem(Multiplexer([2, 2]))  # [pos(2), vel(2)] -> [y, v](4)
        
        # Cart force applicator
        self.cart_force_applicator = self.builder.AddSystem(
            CartForceApplicator(self.cart_pendulum.cart_body.index())
        )
        
        # Wire connections
        self._wire_control_system(
            state_demux,
            imp_force_demux,
            ee_state_mux,
            zero_muscle_force
        )
        
        print(colored("✓ Control system built\n", "green"))
    
    def _wire_control_system(self, state_demux, imp_force_demux, ee_state_mux, zero_muscle_force):
        """Wire all control system connections."""
        builder = self.builder
        
        # Plant state demux
        builder.Connect(self.plant.get_state_output_port(), state_demux.get_input_port())
        
        # Manipulator state
        builder.Connect(
            state_demux.get_output_port(0),
            self.ee_kinematics.GetInputPort("manipulator_state")
        )
        builder.Connect(
            state_demux.get_output_port(0),
            self.jacobian_controller.GetInputPort("manipulator_state")
        )
        
        # Create 4D EE state [x, y, vx, vy] for 2D systems
        builder.Connect(self.ee_kinematics.GetOutputPort("ee_position"), ee_state_mux.get_input_port(0))
        builder.Connect(self.ee_kinematics.GetOutputPort("ee_velocity"), ee_state_mux.get_input_port(1))
        
        # ZFT connections
        builder.Connect(ee_state_mux.get_output_port(0), self.zft_ref_mass.GetInputPort("y_v"))
        builder.Connect(zero_muscle_force.get_output_port(0), self.zft_ref_mass.GetInputPort("F"))
        
        # Impedance connections
        builder.Connect(ee_state_mux.get_output_port(0), self.impedance_force.GetInputPort("y_v"))
        builder.Connect(
            self.zft_ref_mass.GetOutputPort("yref_vref"),
            self.impedance_force.GetInputPort("yref_vref")
        )
        
        # Extract X-component of impedance force (1D control)
        builder.Connect(self.impedance_force.GetOutputPort("F_imp"), imp_force_demux.get_input_port())
        
        # Connect to controllers (X-component only)
        builder.Connect(imp_force_demux.get_output_port(0), self.jacobian_controller.GetInputPort("F_imp"))
        builder.Connect(imp_force_demux.get_output_port(0), self.imp_to_cart.GetInputPort("F_imp"))
        
        # Apply forces
        builder.Connect(
            self.imp_to_cart.GetOutputPort("cart_force"),
            self.cart_force_applicator.GetInputPort("cart_force")
        )
        builder.Connect(
            self.cart_force_applicator.GetOutputPort("spatial_forces"),
            self.plant.get_applied_spatial_force_input_port()
        )
        
        # Joint torques to manipulator
        builder.Connect(
            self.jacobian_controller.get_output_port(),
            self.plant.get_actuation_input_port(self.manipulator.model_instance)
        )
    
    def _setup_visualization(self):
        """Add visualization."""
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat
        )
        self.meshcat.SetProperty("/Background", "visible", False)
    
    def _setup_logging(self):
        """Add data logging."""
        self.state_logger = self.builder.AddSystem(
            VectorLogSink(self.plant.num_multibody_states())
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.state_logger.get_input_port()
        )
    
    def build_diagram(self):
        """Build the complete diagram."""
        self.diagram = self.builder.Build()
        print(colored("✓ Diagram finalized\n", "green"))
    
    def create_simulator(self):
        """Create and configure simulator."""
        print(colored("Creating simulator...", "yellow"))
        
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        
        # Set initial state
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.context)
        
        # Manipulator initial state
        self.plant.SetPositions(
            plant_context,
            self.manipulator.model_instance,
            np.array([np.deg2rad(self.config.q1_init), np.deg2rad(self.config.q2_init)])
        )
        
        # Cart initial state (at EE position)
        self.cart_pendulum.set_cart_state(
            plant_context,
            x=self.ee_pos_init[0],
            y=self.ee_pos_init[1],
            x_dot=0.0,
            y_dot=0.0
        )
        
        # Pendulum initial state
        self.cart_pendulum.set_pendulum_state(
            plant_context,
            pitch=self.config.initial_pitch,
            roll=self.config.initial_roll,
            pitch_dot=0.0,
            roll_dot=0.0
        )
        
        # Configure simulator
        self.simulator.set_target_realtime_rate(SIM_CONFIG.realtime_rate)
        self.simulator.Initialize()
        
        print(colored("✓ Simulator ready\n", "green"))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored(f"Running simulation for {self.config.duration} s...", "yellow", attrs=["bold"]))
        print(colored("-" * 70, "yellow"))
        
        self.simulator.AdvanceTo(self.config.duration)
        
        print(colored("\n✓ Simulation complete\n", "green", attrs=["bold"]))
    
    def plot_results(self):
        """Plot simulation results."""
        print(colored("Generating plots...", "yellow"))
        
        # Extract data
        log = self.state_logger.FindLog(self.context)
        t = log.sample_times()
        x = log.data()
        
        # State indices: [q_manip(2), q_cart(4), q_pend(2), v_manip(2), v_cart(4), v_pend(2)]
        # Manipulator: [q1, q2, q̇1, q̇2] - indices [0,1,6,7]
        # Cart: [x, y, ẋ, ẏ] - indices [2,3,8,9]
        # Pendulum: [pitch, roll, pitcḣ, roll̇] - indices [4,5,10,11]
        
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Plot 1: Joint angles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, np.rad2deg(x[0, :]), 'b-', linewidth=2, label='q₁')
        ax1.plot(t, np.rad2deg(x[1, :]), 'r-', linewidth=2, label='q₂')
        ax1.set_xlabel('Time [s]', fontweight='bold')
        ax1.set_ylabel('Joint Angle [deg]', fontweight='bold')
        ax1.set_title('Manipulator Joint Configuration', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Cart X position
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, x[2, :], 'g-', linewidth=2.5, label='x_cart')
        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('X Position [m]', fontweight='bold')
        ax2.set_title('Cart X Motion', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Cart Y position
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, x[3, :], 'r-', linewidth=2.5, label='y_cart')
        ax3.set_xlabel('Time [s]', fontweight='bold')
        ax3.set_ylabel('Y Position [m]', fontweight='bold')
        ax3.set_title('Cart Y Position', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Cart velocities
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, x[8, :], 'b-', linewidth=2, label='vₓ')
        ax4.plot(t, x[9, :], 'r-', linewidth=2, label='vᵧ')
        ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time [s]', fontweight='bold')
        ax4.set_ylabel('Velocity [m/s]', fontweight='bold')
        ax4.set_title('Cart Velocities', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: 2D Cart trajectory
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(x[2, :], x[3, :], 'g-', linewidth=2.5, alpha=0.8, label='Cart Path')
        ax5.plot(x[2, 0], x[3, 0], 'go', markersize=12, label='Start')
        ax5.plot(x[2, -1], x[3, -1], 'ro', markersize=12, label='End')
        ax5.set_xlabel('X Position [m]', fontweight='bold')
        ax5.set_ylabel('Y Position [m]', fontweight='bold')
        ax5.set_title('2D Cart Trajectory', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
        ax5.legend()
        
        # Plot 6: Joint velocities
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(t, np.rad2deg(x[6, :]), 'b-', linewidth=2, label='q̇₁')
        ax6.plot(t, np.rad2deg(x[7, :]), 'r-', linewidth=2, label='q̇₂')
        ax6.set_xlabel('Time [s]', fontweight='bold')
        ax6.set_ylabel('Angular Velocity [deg/s]', fontweight='bold')
        ax6.set_title('Joint Velocities', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7: Joint space trajectory
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(np.rad2deg(x[0, :]), np.rad2deg(x[1, :]), 'purple', linewidth=2)
        ax7.plot(np.rad2deg(x[0, 0]), np.rad2deg(x[1, 0]), 'go', markersize=12, label='Initial')
        ax7.plot(np.rad2deg(x[0, -1]), np.rad2deg(x[1, -1]), 'ro', markersize=12, label='Final')
        ax7.set_xlabel('q₁ [deg]', fontweight='bold')
        ax7.set_ylabel('q₂ [deg]', fontweight='bold')
        ax7.set_title('Joint Space Trajectory', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Plot 8: Cart position components
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(t, x[2, :], 'b-', linewidth=2, label='x')
        ax8.plot(t, x[3, :], 'r-', linewidth=2, label='y')
        ax8.set_xlabel('Time [s]', fontweight='bold')
        ax8.set_ylabel('Position [m]', fontweight='bold')
        ax8.set_title('Cart Position Components', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # Plot 9: Cart acceleration
        ax9 = fig.add_subplot(gs[2, 2])
        ax_cart = np.gradient(x[8, :], t)
        ay_cart = np.gradient(x[9, :], t)
        ax9.plot(t, ax_cart, 'b-', linewidth=2, label='aₓ')
        ax9.plot(t, ay_cart, 'r-', linewidth=2, label='aᵧ')
        ax9.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax9.set_xlabel('Time [s]', fontweight='bold')
        ax9.set_ylabel('Acceleration [m/s²]', fontweight='bold')
        ax9.set_title('Cart Acceleration', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        
        # Plot 10: Pendulum angles
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.plot(t, np.rad2deg(x[4, :]), 'b-', linewidth=2, label='α (Pitch)')
        ax10.plot(t, np.rad2deg(x[5, :]), 'r-', linewidth=2, label='β (Roll)')
        ax10.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax10.set_xlabel('Time [s]', fontweight='bold')
        ax10.set_ylabel('Angle [deg]', fontweight='bold')
        ax10.set_title('Pendulum Angles', fontweight='bold')
        ax10.grid(True, alpha=0.3)
        ax10.legend()
        
        # Plot 11: Pendulum angular velocities
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.plot(t, np.rad2deg(x[10, :]), 'b-', linewidth=2, label='α̇')
        ax11.plot(t, np.rad2deg(x[11, :]), 'r-', linewidth=2, label='β̇')
        ax11.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax11.set_xlabel('Time [s]', fontweight='bold')
        ax11.set_ylabel('Angular Velocity [deg/s]', fontweight='bold')
        ax11.set_title('Pendulum Angular Rates', fontweight='bold')
        ax11.grid(True, alpha=0.3)
        ax11.legend()
        
        # Plot 12: Summary statistics
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')
        summary_text = f"""
SYSTEM STATE SUMMARY

Initial Configuration:
  q₁ = {np.rad2deg(x[0, 0]):6.2f}°
  q₂ = {np.rad2deg(x[1, 0]):6.2f}°

Final Configuration (t = {t[-1]:.1f}s):
  q₁ = {np.rad2deg(x[0, -1]):6.2f}°
  q₂ = {np.rad2deg(x[1, -1]):6.2f}°

Joint Changes:
  Δq₁ = {np.rad2deg(x[0, -1] - x[0, 0]):6.2f}°
  Δq₂ = {np.rad2deg(x[1, -1] - x[1, 0]):6.2f}°

Cart Displacement:
  ΔX = {x[2, -1] - x[2, 0]:5.3f} m
  ΔY = {x[3, -1] - x[3, 0]:5.3f} m

Cart Final Velocity:
  vₓ = {x[8, -1]:5.3f} m/s
  vᵧ = {x[9, -1]:5.3f} m/s

Pendulum Final Angles:
  α = {np.rad2deg(x[4, -1]):5.2f}° (pitch)
  β = {np.rad2deg(x[5, -1]):5.2f}° (roll)

Control Parameters:
  K_imp = {self.config.K_imp:5.1f} N/m
  D_imp = {self.config.D_imp:5.1f} N·s/m
  M_ref = {self.config.M_ref:5.1f} kg
        """
        ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Manipulator Pushes Cart - Impedance Control with ZFT\n(M_ref → x_ref, F_imp = K(x_ee - x_ref) + D·v_err → Cart)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(colored("✓ Plots generated\n", "green"))
    
    def print_summary(self):
        """Print simulation summary."""
        log = self.state_logger.FindLog(self.context)
        x_final = log.data()[:, -1]
        
        print(colored("\n" + "=" * 70, "cyan"))
        print(colored("SIMULATION SUMMARY", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        print(colored(f"Duration: {self.config.duration} s", "yellow"))
        print(colored(f"Final cart position: x={x_final[2]:.3f} m, y={x_final[3]:.3f} m", "yellow"))
        print(colored(f"Cart displacement: Δx={x_final[2] - self.ee_pos_init[0]:.3f} m", "yellow"))
        print(colored(f"Final pendulum angles: α={np.rad2deg(x_final[4]):.2f}°, β={np.rad2deg(x_final[5]):.2f}°", "yellow"))
        print(colored("=" * 70 + "\n", "cyan"))
    
    def run_full_simulation(self):
        """Execute complete simulation workflow."""
        self.setup_drake_system()
        self.build_diagram()
        self.create_simulator()
        self.run_simulation()
        self.print_summary()
        
        if self.visualize:
            print(colored("Animation playing in Meshcat. Close browser to continue.", "yellow"))
            input(colored("Press Enter to generate plots...", "yellow"))
        
        self.plot_results()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(colored("MANIPULATOR PUSHES CART - Drake OFC Architecture", 'cyan', attrs=['bold']))
    print(colored("Impedance Control with ZFT Reference Mass", 'cyan'))
    print("=" * 70)
    print(colored(f"Duration: {MANIPULATOR_PUSHES_CART_CONFIG.duration} s", 'yellow'))
    print(colored(f"Distance: {MANIPULATOR_PUSHES_CART_CONFIG.distance} m", 'yellow'))
    print(colored(f"Impedance: K={MANIPULATOR_PUSHES_CART_CONFIG.K_imp} N/m, "
                 f"D={MANIPULATOR_PUSHES_CART_CONFIG.D_imp} N·s/m", 'yellow'))
    print(colored(f"Reference Mass: M={MANIPULATOR_PUSHES_CART_CONFIG.M_ref} kg", 'yellow'))
    print("=" * 70 + "\n")
    
    # Create and run scene manager
    manager = ManipulatorPushesCartSceneManager(
        config=MANIPULATOR_PUSHES_CART_CONFIG,
        visualize=args.visualize,
    )
    
    manager.run_full_simulation()
    
    print(colored("\n" + "="*70, 'green'))
    print(colored("Execution Complete!", 'green', attrs=['bold']))
    print(colored("="*70 + "\n", 'green'))


if __name__ == "__main__":
    main()
