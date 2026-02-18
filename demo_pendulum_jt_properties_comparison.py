"""
=============================================================================
PENDULUM JOINT PROPERTIES COMPARISON
=============================================================================
Four pendulum systems to understand joint stiffness and damping effects:
1. Simple Pendulum - No joint stiffness, no damping (gravity only)
2. Pendulum with Joint Stiffness - Spring torque at joint, no damping
3. Pendulum with Joint Damping - Viscous damping at joint, no stiffness
4. Pendulum with Stiffness + Damping - Both spring and damper at joint

All built from scratch using LeafSystem with simultaneous visualization and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Simulator,
    VectorLogSink,
    RigidTransform,
    StartMeshcat,
    Rgba,
)
from pydrake.geometry import Sphere, Cylinder
from pydrake.math import RotationMatrix
from pydrake.systems.primitives import ConstantVectorSource


# =============================================================================
# PARAMETERS
# =============================================================================
class PendulumParams:
    """Physical parameters for pendulum systems."""
    def __init__(self) -> None:
        self.m: float = 1.0      # mass (kg)
        self.L: float = 1.0      # length (m)
        self.g: float = 9.81     # gravity (m/s^2)
        self.b: float = 0.1      # joint damping coefficient (N·m·s)
        self.k: float = 5.0      # joint stiffness (N·m/rad)
        self.mu: float = 0.5     # joint friction torque (N·m)


class VisualizationParams:
    def __init__(self) -> None:
        self.pivot_radius: float = 0.05
        self.rod_radius: float = 0.02
        self.bob_radius: float = 0.08
        self.update_rate: float = 30.0  # Hz


# =============================================================================
# SYSTEM 1: SIMPLE PENDULUM (NO STIFFNESS, NO DAMPING)
# =============================================================================
class SimplePendulumSystem(LeafSystem):
    """
    Simple pendulum with gravity only.
    
    State x = [theta, theta_dot]
    Dynamics: I*theta_ddot = -m*g*L*sin(theta)
    
    No joint stiffness, no damping - pure gravitational pendulum.
    """
    def __init__(self, params: Optional[PendulumParams] = None) -> None:
        super().__init__()
        self.set_name("SimplePendulum")
        
        self.params = params if params is not None else PendulumParams()
        
        # Continuous state: [theta, theta_dot]
        self.DeclareContinuousState(2)
        
        # Input: external torque
        self.DeclareVectorInputPort("torque", 1)
        
        # Output: state vector
        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()},
        )
    
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot = context.get_continuous_state_vector().CopyToVector()
        tau = self.get_input_port(0).Eval(context)[0]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        
        I = m * L**2
        theta_ddot = (tau - m * g * L * np.sin(theta)) / I
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def CalcBobPosition(self, theta: float) -> np.ndarray:
        return np.array([self.params.L * np.sin(theta), 0.0, -self.params.L * np.cos(theta)])


# =============================================================================
# SYSTEM 2: PENDULUM WITH JOINT STIFFNESS ONLY
# =============================================================================
class PendulumWithStiffness(LeafSystem):
    """
    Pendulum with torsional spring at joint (no damping).
    
    State x = [theta, theta_dot]
    Dynamics: I*theta_ddot = -m*g*L*sin(theta) - k*theta
    
    Joint stiffness creates restoring torque proportional to angle.
    """
    def __init__(self, params: Optional[PendulumParams] = None) -> None:
        super().__init__()
        self.set_name("PendulumStiffness")
        
        self.params = params if params is not None else PendulumParams()
        
        self.DeclareContinuousState(2)
        self.DeclareVectorInputPort("torque", 1)
        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()},
        )
    
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot = context.get_continuous_state_vector().CopyToVector()
        tau = self.get_input_port(0).Eval(context)[0]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        k = self.params.k
        
        I = m * L**2
        # Spring torque opposes angular displacement
        theta_ddot = (tau - m * g * L * np.sin(theta) - k * theta) / I
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def CalcBobPosition(self, theta: float) -> np.ndarray:
        return np.array([self.params.L * np.sin(theta), 0.0, -self.params.L * np.cos(theta)])


# =============================================================================
# SYSTEM 3: PENDULUM WITH JOINT DAMPING ONLY
# =============================================================================
class PendulumWithDamping(LeafSystem):
    """
    Pendulum with viscous damping at joint (no stiffness).
    
    State x = [theta, theta_dot]
    Dynamics: I*theta_ddot = -m*g*L*sin(theta) - b*theta_dot
    
    Damping creates torque opposing angular velocity.
    """
    def __init__(self, params: Optional[PendulumParams] = None) -> None:
        super().__init__()
        self.set_name("PendulumDamping")
        
        self.params = params if params is not None else PendulumParams()
        
        self.DeclareContinuousState(2)
        self.DeclareVectorInputPort("torque", 1)
        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()},
        )
    
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot = context.get_continuous_state_vector().CopyToVector()
        tau = self.get_input_port(0).Eval(context)[0]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        b = self.params.b
        
        I = m * L**2
        # Damping torque opposes angular velocity
        theta_ddot = (tau - m * g * L * np.sin(theta) - b * theta_dot) / I
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def CalcBobPosition(self, theta: float) -> np.ndarray:
        return np.array([self.params.L * np.sin(theta), 0.0, -self.params.L * np.cos(theta)])


# =============================================================================
# SYSTEM 4: PENDULUM WITH STIFFNESS AND DAMPING
# =============================================================================
class PendulumWithStiffnessAndDamping(LeafSystem):
    """
    Pendulum with both torsional spring and damper at joint.
    
    State x = [theta, theta_dot]
    Dynamics: I*theta_ddot = -m*g*L*sin(theta) - k*theta - b*theta_dot
    
    Complete joint model with stiffness and damping.
    """
    def __init__(self, params: Optional[PendulumParams] = None) -> None:
        super().__init__()
        self.set_name("PendulumStiffnessDamping")
        
        self.params = params if params is not None else PendulumParams()
        
        self.DeclareContinuousState(2)
        self.DeclareVectorInputPort("torque", 1)
        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()},
        )
    
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot = context.get_continuous_state_vector().CopyToVector()
        tau = self.get_input_port(0).Eval(context)[0]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        k = self.params.k
        b = self.params.b
        
        I = m * L**2
        # Both spring and damping torques
        theta_ddot = (tau - m * g * L * np.sin(theta) - k * theta - b * theta_dot) / I
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def CalcBobPosition(self, theta: float) -> np.ndarray:
        return np.array([self.params.L * np.sin(theta), 0.0, -self.params.L * np.cos(theta)])


# =============================================================================
# SYSTEM 5: PENDULUM WITH JOINT FRICTION
# =============================================================================
class PendulumWithFriction(LeafSystem):
    """
    Pendulum with Coulomb friction at joint.
    
    State x = [theta, theta_dot]
    Dynamics: I*theta_ddot = -m*g*L*sin(theta) - mu*sign(theta_dot)
    
    Friction creates constant torque opposing velocity (independent of speed).
    """
    def __init__(self, params: Optional[PendulumParams] = None) -> None:
        super().__init__()
        self.set_name("PendulumFriction")
        
        self.params = params if params is not None else PendulumParams()
        
        self.DeclareContinuousState(2)
        self.DeclareVectorInputPort("torque", 1)
        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()},
        )
    
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        theta, theta_dot = context.get_continuous_state_vector().CopyToVector()
        tau = self.get_input_port(0).Eval(context)[0]
        
        m = self.params.m
        L = self.params.L
        g = self.params.g
        mu = self.params.mu
        
        I = m * L**2
        # Coulomb friction: constant torque opposing motion
        # Use tanh for smooth approximation of sign function
        friction_torque = mu * np.tanh(100 * theta_dot)  # Smooth sign function
        theta_ddot = (tau - m * g * L * np.sin(theta) - friction_torque) / I
        
        derivatives.get_mutable_vector().SetFromVector([theta_dot, theta_ddot])
    
    def CalcBobPosition(self, theta: float) -> np.ndarray:
        return np.array([self.params.L * np.sin(theta), 0.0, -self.params.L * np.cos(theta)])


# =============================================================================
# VISUALIZER: PENDULUM IN MESHCAT
# =============================================================================
class PendulumVisualizer(LeafSystem):
    """Visualizes a single pendulum in Meshcat."""
    
    def __init__(self, meshcat, plant: LeafSystem, system_name: str, x_offset: float,
                 viz_params: Optional[VisualizationParams] = None,
                 color: Rgba = Rgba(0.2, 0.5, 0.9, 1.0)) -> None:
        super().__init__()
        self.set_name(f"{system_name}_Visualizer")
        
        self.meshcat = meshcat
        self.plant = plant
        self.system_name = system_name
        self.x_offset = x_offset
        self.viz = viz_params if viz_params is not None else VisualizationParams()
        self.color = color
        
        # Input: state [theta, theta_dot]
        self.DeclareVectorInputPort("state", 2)
        
        # Periodic publish
        self.DeclarePeriodicPublishEvent(
            period_sec=1.0 / self.viz.update_rate,
            offset_sec=0.0,
            publish=self.UpdateVisualization
        )
        
        self._BuildScene()
    
    def _BuildScene(self):
        """Create visual elements."""
        base = f"{self.system_name}"
        
        # Pivot
        self.meshcat.SetObject(
            f"{base}/pivot",
            Sphere(self.viz.pivot_radius),
            Rgba(0.3, 0.3, 0.3, 1.0)
        )
        pivot_pos = RigidTransform([self.x_offset, 0.0, 0.0])
        self.meshcat.SetTransform(f"{base}/pivot", pivot_pos)
        
        # Rod
        self.meshcat.SetObject(
            f"{base}/rod",
            Cylinder(self.viz.rod_radius, self.plant.params.L),
            Rgba(0.5, 0.5, 0.5, 1.0)
        )
        
        # Bob
        self.meshcat.SetObject(
            f"{base}/bob",
            Sphere(self.viz.bob_radius),
            self.color
        )
    
    def UpdateVisualization(self, context):
        """Update pendulum visualization."""
        state = self.get_input_port(0).Eval(context)
        theta = float(state[0])
        
        base = f"{self.system_name}"
        
        # Bob position
        p_bob = self.plant.CalcBobPosition(theta)
        p_bob[0] += self.x_offset  # Offset horizontally
        T_bob = RigidTransform(p_bob)
        self.meshcat.SetTransform(f"{base}/bob", T_bob)
        
        # Rod orientation and position
        rod_dir = p_bob.copy()
        rod_dir[0] -= self.x_offset
        norm = np.linalg.norm(rod_dir)
        if norm > 1e-9:
            rod_dir = rod_dir / norm
        else:
            rod_dir = np.array([0.0, 0.0, -1.0])
        
        R_rod = RotationMatrix.MakeFromOneVector(rod_dir, axis_index=2)
        p_rod = np.array([self.x_offset, 0.0, 0.0]) + 0.5 * (p_bob - np.array([self.x_offset, 0.0, 0.0]))
        T_rod = RigidTransform(R_rod, p_rod)
        self.meshcat.SetTransform(f"{base}/rod", T_rod)


# =============================================================================
# SIMULATION HELPER
# =============================================================================
def simulate_system(system: LeafSystem, x0: np.ndarray, sim_time: float,
                   meshcat=None, visualizer=None):
    """Simulate a single pendulum system."""
    builder = DiagramBuilder()
    
    plant = builder.AddSystem(system)
    
    # Zero external torque
    zero_torque = builder.AddSystem(ConstantVectorSource([0.0]))
    builder.Connect(zero_torque.get_output_port(0), plant.get_input_port(0))
    
    # Logger
    logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(plant.get_output_port(0), logger.get_input_port(0))
    
    # Add visualizer if provided
    if visualizer is not None:
        viz = builder.AddSystem(visualizer)
        builder.Connect(plant.get_output_port(0), viz.get_input_port(0))
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial condition
    plant_context = plant.GetMyContextFromRoot(context)
    plant_context.SetContinuousState(x0)
    
    # Run simulation
    if meshcat is not None:
        simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(sim_time)
    
    # Extract log data
    log = logger.FindLog(context)
    time_log = log.sample_times()
    state_log = log.data().T
    
    return time_log, state_log


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_comparison(t1, z1, t2, z2, t3, z3, t4, z4, t5=None, z5=None,
                   labels=["Simple", "Stiffness", "Damping", "Stiff+Damp", "Friction"],
                   save_name: str = None):
    """
    Compare four or five simulation results.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    
    colors = ['b-', 'r--', 'g-.', 'm:', 'c-']
    
    # Angle plot
    axes[0].plot(t1, z1[:, 0], colors[0], linewidth=2, label=labels[0], alpha=0.8)
    axes[0].plot(t2, z2[:, 0], colors[1], linewidth=2, label=labels[1], alpha=0.8)
    axes[0].plot(t3, z3[:, 0], colors[2], linewidth=2, label=labels[2], alpha=0.8)
    axes[0].plot(t4, z4[:, 0], colors[3], linewidth=2, label=labels[3], alpha=0.8)
    if t5 is not None and z5 is not None:
        axes[0].plot(t5, z5[:, 0], colors[4], linewidth=2, label=labels[4], alpha=0.8)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Angle θ (rad)', fontsize=12)
    axes[0].set_title('Pendulum Joint Properties Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Angular velocity plot
    axes[1].plot(t1, z1[:, 1], colors[0], linewidth=2, label=labels[0], alpha=0.8)
    axes[1].plot(t2, z2[:, 1], colors[1], linewidth=2, label=labels[1], alpha=0.8)
    axes[1].plot(t3, z3[:, 1], colors[2], linewidth=2, label=labels[2], alpha=0.8)
    axes[1].plot(t4, z4[:, 1], colors[3], linewidth=2, label=labels[3], alpha=0.8)
    if t5 is not None and z5 is not None:
        axes[1].plot(t5, z5[:, 1], colors[4], linewidth=2, label=labels[4], alpha=0.8)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Angular Velocity θ̇ (rad/s)', fontsize=12)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Phase portrait
    axes[2].plot(z1[:, 0], z1[:, 1], colors[0], linewidth=2, label=labels[0], alpha=0.8)
    axes[2].plot(z2[:, 0], z2[:, 1], colors[1], linewidth=2, label=labels[1], alpha=0.8)
    axes[2].plot(z3[:, 0], z3[:, 1], colors[2], linewidth=2, label=labels[2], alpha=0.8)
    axes[2].plot(z4[:, 0], z4[:, 1], colors[3], linewidth=2, label=labels[3], alpha=0.8)
    if t5 is not None and z5 is not None:
        axes[2].plot(z5[:, 0], z5[:, 1], colors[4], linewidth=2, label=labels[4], alpha=0.8)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Angle θ (rad)', fontsize=12)
    axes[2].set_ylabel('Angular Velocity θ̇ (rad/s)', fontsize=12)
    axes[2].set_title('Phase Portrait', fontsize=12, fontweight='bold')
    axes[2].legend(loc='best', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_name:
        import os
        output_dir = '/Volumes/Data/pydrake_analysis/figures'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{save_name}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    
    plt.show()


# =============================================================================
# MAIN COMPARISON DEMO
# =============================================================================
def comparison_demo() -> None:
    print("\n" + "="*80)
    print("PENDULUM JOINT PROPERTIES COMPARISON")
    print("="*80)
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(f"\nMeshcat viewer available at: {meshcat.web_url()}")
    
    # Add interactive controls
    meshcat.AddButton("Reset Simulation", "KeyR")
    meshcat.AddSlider("Playback Speed", min=0.1, max=3.0, step=0.1, value=1.0)
    
    # Shared parameters
    params = PendulumParams()
    viz_params = VisualizationParams()
    
    # Initial condition: released from angle
    theta0 = np.pi / 4  # 45 degrees
    x0 = np.array([theta0, 0.0])  # [theta, theta_dot]
    
    sim_time = 15.0  # seconds
    
    print(f"\nPhysical Parameters:")
    print(f"  Mass (m):            {params.m:.2f} kg")
    print(f"  Length (L):          {params.L:.2f} m")
    print(f"  Gravity (g):         {params.g:.2f} m/s^2")
    print(f"  Joint stiffness (k): {params.k:.2f} N·m/rad")
    print(f"  Joint damping (b):   {params.b:.2f} N·m·s")
    print(f"  Joint friction (μ):  {params.mu:.2f} N·m")
    print(f"\nInitial angle: {theta0*180/np.pi:.1f}°")
    print(f"Simulation time: {sim_time:.1f} s")
    
    # Create visualizers (positioned side-by-side)
    systems_info = [
        ("SimplePendulum", -1.5, Rgba(0.2, 0.5, 0.9, 1.0), "blue"),
        ("Stiffness", -0.5, Rgba(0.9, 0.2, 0.2, 1.0), "red"),
        ("Damping", 0.5, Rgba(0.2, 0.9, 0.2, 1.0), "green"),
        ("Stiff+Damp", 1.5, Rgba(0.9, 0.5, 0.2, 1.0), "orange")
    ]
    
    # System 1: Simple Pendulum
    print("\n[1/5] Simulating Simple Pendulum (no stiffness, no damping, no friction)...")
    system1 = SimplePendulumSystem(params)
    viz1 = PendulumVisualizer(meshcat, system1, "Pendulum1", -4.0, viz_params, Rgba(0.2, 0.5, 0.9, 1.0))
    t1, z1 = simulate_system(system1, x0, sim_time, meshcat, viz1)
    print(f"      Completed: {len(t1)} samples")
    
    # System 2: Pendulum with Stiffness
    print("[2/5] Simulating Pendulum with Joint Stiffness...")
    system2 = PendulumWithStiffness(params)
    viz2 = PendulumVisualizer(meshcat, system2, "Pendulum2", -2.0, viz_params, Rgba(0.9, 0.2, 0.2, 1.0))
    t2, z2 = simulate_system(system2, x0, sim_time, meshcat, viz2)
    print(f"      Completed: {len(t2)} samples")
    
    # System 3: Pendulum with Damping
    print("[3/5] Simulating Pendulum with Joint Damping...")
    system3 = PendulumWithDamping(params)
    viz3 = PendulumVisualizer(meshcat, system3, "Pendulum3", 0.0, viz_params, Rgba(0.2, 0.9, 0.2, 1.0))
    t3, z3 = simulate_system(system3, x0, sim_time, meshcat, viz3)
    print(f"      Completed: {len(t3)} samples")
    
    # System 4: Pendulum with Stiffness and Damping
    print("[4/5] Simulating Pendulum with Stiffness + Damping...")
    system4 = PendulumWithStiffnessAndDamping(params)
    viz4 = PendulumVisualizer(meshcat, system4, "Pendulum4", 2.0, viz_params, Rgba(0.9, 0.5, 0.2, 1.0))
    t4, z4 = simulate_system(system4, x0, sim_time, meshcat, viz4)
    print(f"      Completed: {len(t4)} samples")
    
    # System 5: Pendulum with Friction
    print("[5/5] Simulating Pendulum with Joint Friction...")
    system5 = PendulumWithFriction(params)
    viz5 = PendulumVisualizer(meshcat, system5, "Pendulum5", 4.0, viz_params, Rgba(0.0, 0.8, 0.8, 1.0))
    t5, z5 = simulate_system(system5, x0, sim_time, meshcat, viz5)
    print(f"      Completed: {len(t5)} samples")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(t1, z1, t2, z2, t3, z3, t4, z4, t5, z5,
                   labels=["Simple (no k, b, μ)", "Stiffness (k only)", 
                          "Damping (b only)", "Stiff+Damp (k+b)", "Friction (μ only)"],
                   save_name="pendulum_joint_comparison")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("1. SIMPLE PENDULUM (blue):")
    print("   - Oscillates indefinitely (conservative system)")
    print("   - No energy dissipation, no restoring torque to vertical")
    print("   - Period depends only on length and gravity")
    print()
    print("2. STIFFNESS ONLY (red):")
    print("   - Joint spring creates restoring torque to vertical")
    print("   - Higher frequency oscillation (stiffer = faster)")
    print("   - Still conservative (no energy loss)")
    print(f"   - Effective frequency increased by joint stiffness")
    print()
    print("3. DAMPING ONLY (green):")
    print("   - Energy dissipates, oscillations decay")
    print("   - Eventually hangs straight down")
    print("   - Same frequency as simple pendulum (no stiffness)")
    print()
    print("4. STIFFNESS + DAMPING (orange):")
    print("   - Combines both effects")
    print("   - Higher frequency from stiffness")
    print("   - Decaying amplitude from damping")
    print("   - Settles to vertical position (stable equilibrium)")
    print()
    print("5. FRICTION ONLY (cyan):")
    print("   - Coulomb friction: constant torque opposing motion")
    print("   - Unlike damping, friction is independent of velocity")
    print("   - Energy loss causes amplitude decay")
    print("   - Eventually stops at some angle (not necessarily vertical)")
    print("="*80)
    print("\nMeshcat visualization shows all five systems side-by-side:")
    print("  Position 1 (Blue):   Simple Pendulum")
    print("  Position 2 (Red):    With Joint Stiffness")
    print("  Position 3 (Green):  With Joint Damping")
    print("  Position 4 (Orange): With Stiffness & Damping")
    print("  Position 5 (Cyan):   With Joint Friction")
    print("\nPress Ctrl+C to exit (Meshcat will stay up)...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    comparison_demo()
