"""
Simple Cart-Pendulum Simulation with Basic U Control

Minimal implementation:
- CartPendulumPlant: Physics-based plant only
- DrakeSceneManager: Build diagram and run simulation with control input u
- Simple visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from termcolor import colored
import argparse

# Drake imports
from pydrake.all import (
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    Role,
    RigidTransform,
    RotationMatrix,
)
from pydrake.systems.primitives import ConstantVectorSource


# ============================================================================
# SIMPLE CONTROL INPUT SYSTEM
# ============================================================================

class ControlInputSource(LeafSystem):
    """
    Simple system that outputs a constant or time-varying control input u.
    Can be used for:
    - Constant force: u = constant
    - Time-varying: u = sin(t), u = step, etc.
    """
    
    def __init__(self, control_mode='constant', constant_value=0.0):
        super().__init__()
        self.control_mode = control_mode
        self.constant_value = constant_value
        self.DeclareVectorOutputPort("u", BasicVector(1), self._calc_output)
    
    def _calc_output(self, context, output):
        t = context.get_time()
        
        if self.control_mode == 'constant':
            u = np.array([self.constant_value])
        elif self.control_mode == 'sine':
            u = np.array([self.constant_value * np.sin(2 * np.pi * t)])
        elif self.control_mode == 'step':
            u = np.array([self.constant_value if t > 2.0 else 0.0])
        elif self.control_mode == 'ramp':
            u = np.array([min(self.constant_value * t / 2.0, self.constant_value)])
        else:
            u = np.array([self.constant_value])
        
        output.SetFromVector(u)


# ============================================================================
# CART-PENDULUM PLANT
# ============================================================================

class CartPendulumPlant:
    """
    Minimal cart-pendulum plant using Drake's MultibodyPlant.
    
    System:
    - Cart on 1D track (prismatic joint)
    - Inverted pendulum (revolute joint)
    - 2 DOF: [x, θ]
    - 1 actuator: horizontal force on cart
    """
    
    def __init__(self, 
                 mass_cart=1.0,
                 mass_pendulum=0.5,
                 length_pendulum=0.5,
                 gravity=9.81):
        """Initialize plant parameters."""
        self.mass_cart = mass_cart
        self.mass_pendulum = mass_pendulum
        self.length_pendulum = length_pendulum
        self.gravity = gravity
        
        self.plant = None
        self.scene_graph = None
    
    def build(self, builder):
        """Build the plant in the diagram."""
        print(colored("Building Cart-Pendulum Plant", 'yellow', attrs=['bold']))
        
        # Create plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        
        # Create and add simple URDF
        self._create_and_load_plant()
        
        print(colored(f"  ✓ Cart mass: {self.mass_cart} kg", 'cyan'))
        print(colored(f"  ✓ Pendulum mass: {self.mass_pendulum} kg", 'cyan'))
        print(colored(f"  ✓ Pendulum length: {self.length_pendulum} m", 'cyan'))
        print(colored(f"  ✓ Gravity: {self.gravity} m/s²", 'cyan'))
        
        return self.plant, self.scene_graph
    
    def _create_and_load_plant(self):
        """Create a simple URDF and load it."""
        import tempfile
        from pydrake.all import Parser
        
        # Create minimal URDF
        urdf_content = f'''<?xml version="1.0"?>
<robot name="cart_pendulum">
  <!-- World frame -->
  
  <!-- Cart body -->
  <link name="cart">
    <inertial>
      <mass value="{self.mass_cart}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <!-- Pendulum body -->
  <link name="pendulum">
    <inertial>
      <mass value="{self.mass_pendulum}"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <!-- Cart slider (world to cart) -->
  <joint name="cart_slider" type="prismatic">
    <parent link="world"/>
    <child link="cart"/>
    <axis xyz="1 0 0"/>
    <limit lower="-5" upper="5" effort="100" velocity="10"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <dynamics damping="0.1"/>
  </joint>
  
  <!-- Pendulum pivot (cart to pendulum) -->
  <joint name="pend_pin" type="revolute">
    <parent link="cart"/>
    <child link="pendulum"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="10"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <dynamics damping="0.01"/>
  </joint>
  
  <!-- Actuator on cart -->
  <transmission name="cart_actuator">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="cart_slider">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="cart_force">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>'''
        
        # Write URDF to temp file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            temp_urdf = f.name
        
        try:
            parser = Parser(self.plant)
            parser.AddModels(temp_urdf)
            self.plant.Finalize()
            print(colored(f"  ✓ Plant finalized: 2 DOF, 1 actuator", 'green'))
        finally:
            import os
            os.unlink(temp_urdf)
    
    def _create_plant_geometry(self):
        """Create cart and pendulum bodies."""
        from pydrake.all import (
            SpatialInertia, UnitInertia,
            PrismaticJoint, RevoluteJoint
        )
        
        # World frame
        world_body = self.plant.world_body()
        
        # Cart body - use simple point mass
        cart_body = self.plant.AddRigidBody(
            name="cart",
            model_instance=self.plant.default_model_instance(),
            M_BBo_B=SpatialInertia(
                mass=self.mass_cart,
                p_PScm_E=np.array([0.0, 0.0, 0.0]),
                G_SP_E=UnitInertia.Zero()
            )
        )
        
        # Prismatic joint (cart slides horizontally)
        cart_joint = PrismaticJoint(
            name="cart_slider",
            frame_on_parent=world_body.body_frame(),
            frame_on_child=cart_body.body_frame(),
            axis=np.array([1.0, 0.0, 0.0]),
            limits=np.array([-5.0, 5.0]),
            damping=0.1
        )
        self.plant.AddJoint(cart_joint)
        
        # Pendulum body - simple point mass at center
        pend_body = self.plant.AddRigidBody(
            name="pendulum",
            model_instance=self.plant.default_model_instance(),
            M_BBo_B=SpatialInertia(
                mass=self.mass_pendulum,
                p_PScm_E=np.array([0.0, 0.0, -self.length_pendulum/2]),
                G_SP_E=UnitInertia.Zero()
            )
        )
        
        # Revolute joint (pendulum rotates about cart)
        pend_joint = RevoluteJoint(
            name="pend_pin",
            frame_on_parent=cart_body.body_frame(),
            frame_on_child=pend_body.body_frame(),
            axis=np.array([0.0, 1.0, 0.0]),
            damping=0.01
        )
        self.plant.AddJoint(pend_joint)
        
        # Actuator on cart joint
        self.plant.AddJointActuator(name="cart_force", joint=cart_joint)
        
        # Finalize plant
        self.plant.Finalize()
        
        print(colored(f"  ✓ Plant finalized: 2 DOF, 1 actuator", 'green'))


# ============================================================================
# DRAKE SCENE MANAGER
# ============================================================================

class DrakeSceneManager:
    """
    Manages Drake simulation with simple control input.
    
    Responsibilities:
    - Build diagram (plant + control input source)
    - Create simulator
    - Run simulation and log data
    """
    
    def __init__(self,
                 plant_config=None,
                 initial_theta=45.0,
                 simulation_time=10.0,
                 timestep=0.001,
                 control_mode='constant',
                 control_value=0.0,
                 visualize=True):
        """Initialize scene manager."""
        self.plant_config = plant_config or {}
        self.initial_theta = np.deg2rad(initial_theta)
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.control_mode = control_mode
        self.control_value = control_value
        self.visualize = visualize
        
        # Drake objects
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.control_input = None
        self.diagram = None
        self.simulator = None
        self.meshcat = None
        
        # Data logging
        self.time_log = []
        self.state_log = []
        self.control_log = []
        
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("DrakeSceneManager Initialization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"  Control Mode: {control_mode}", 'yellow'))
        print(colored(f"  Control Value: {control_value}", 'yellow'))
        print(colored(f"  Initial Angle: {np.rad2deg(self.initial_theta):.1f}°", 'yellow'))
        print(colored(f"  Simulation Time: {simulation_time}s", 'yellow'))
        print(colored(f"  Visualization: {'Enabled' if visualize else 'Disabled'}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def setup_diagram(self):
        """Build the diagram with plant and control input."""
        print(colored("Building Diagram", 'yellow', attrs=['bold']))
        
        self.builder = DiagramBuilder()
        
        # Create and add plant
        cart_pend = CartPendulumPlant(**self.plant_config)
        self.plant, self.scene_graph = cart_pend.build(self.builder)
        
        # Create and add control input source
        self.control_input = self.builder.AddSystem(
            ControlInputSource(
                control_mode=self.control_mode,
                constant_value=self.control_value
            )
        )
        self.control_input.set_name("control_input")
        
        # Wire control input to plant actuator
        self.builder.Connect(
            self.control_input.get_output_port(0),
            self.plant.get_actuation_input_port()
        )
        
        print(colored(f"  ✓ Control input wired to plant", 'green'))
        
        # Add visualization if enabled
        if self.visualize and self.scene_graph is not None:
            self.meshcat = StartMeshcat()
            visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder, self.scene_graph, self.meshcat
            )
            print(colored(f"  ✓ Meshcat visualization enabled", 'green'))
            print(colored(f"  ✓ URL: {self.meshcat.web_url()}", 'cyan'))
        
        # Build diagram
        self.diagram = self.builder.Build()
        print(colored(f"  ✓ Diagram built successfully", 'green'))
    
    def create_simulator(self):
        """Create simulator."""
        print(colored("\nCreating Simulator", 'yellow', attrs=['bold']))
        
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)
        
        # Set initial conditions
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        x0 = np.array([0.0, self.initial_theta])
        v0 = np.array([0.0, 0.0])
        
        self.plant.SetPositions(plant_context, x0)
        self.plant.SetVelocities(plant_context, v0)
        
        self.diagram.ForcedPublish(context)
        
        print(colored(f"  ✓ Initial state set:", 'green'))
        print(colored(f"    x = {x0[0]:.3f} m", 'cyan'))
        print(colored(f"    θ = {np.rad2deg(x0[1]):.1f}°", 'cyan'))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("Running Simulation", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        t_next_print = 0.0
        t_next_log = 0.0
        print_interval = 0.5
        log_interval = 0.01
        
        while context.get_time() < self.simulation_time:
            self.simulator.AdvanceTo(context.get_time() + self.timestep)
            
            t = context.get_time()
            
            # Update visualization
            if self.meshcat is not None:
                self.diagram.ForcedPublish(context)
            
            # Print progress
            if t >= t_next_print:
                state = self.plant.GetPositionsAndVelocities(plant_context)
                x, theta, x_dot, theta_dot = state
                theta_deg = np.rad2deg(theta)
                
                pct = int(100 * t / self.simulation_time)
                print(f"[{t:6.2f}s / {self.simulation_time:6.2f}s | {pct:3d}%] "
                      f"x={x:+7.3f}m  θ={theta_deg:+7.1f}°  |  "
                      f"ẋ={x_dot:+7.3f}m/s  θ̇={np.rad2deg(theta_dot):+7.1f}°/s")
                t_next_print += print_interval
            
            # Log data
            if t >= t_next_log:
                state = self.plant.GetPositionsAndVelocities(plant_context)
                self.state_log.append(state.copy())
                self.time_log.append(t)
                
                control_context = self.control_input.GetMyContextFromRoot(context)
                control_val = self.control_input.get_output_port(0).Eval(control_context)
                self.control_log.append(control_val[0])
                
                t_next_log += log_interval
        
        print(colored(f"\n✓ Simulation completed!", 'green', attrs=['bold']))
    
    def plot_results(self):
        """Plot simulation results."""
        if len(self.time_log) == 0:
            print(colored("No data to plot", 'yellow'))
            return
        
        times = np.array(self.time_log)
        states = np.array(self.state_log)
        controls = np.array(self.control_log)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cart position
        axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Position (m)', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Cart Position', fontsize=12, fontweight='bold')
        
        # Pendulum angle
        pend_angle_deg = np.rad2deg(states[:, 1])
        axes[0, 1].plot(times, pend_angle_deg, 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Angle (°)', fontsize=11, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Pendulum Angle', fontsize=12, fontweight='bold')
        
        # Cart velocity
        axes[1, 0].plot(times, states[:, 2], 'b--', linewidth=2, label='ẋ')
        axes[1, 0].set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Cart Velocity', fontsize=12, fontweight='bold')
        
        # Control input
        axes[1, 1].plot(times, controls, 'g-', linewidth=2, label='u (N)')
        axes[1, 1].set_ylabel('Control Force (N)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Control Input', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Cart-Pendulum System - {self.control_mode.capitalize()} Control (u={self.control_value})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/simple_cart_pendulum_{self.control_mode}_{timestamp}.png"
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(colored(f"✓ Plot saved: {filename}", 'green'))
        
        plt.show()
    
    def run_full_simulation(self):
        """Execute complete simulation pipeline."""
        try:
            self.setup_diagram()
            self.create_simulator()
            self.run_simulation()
            self.plot_results()
        except KeyboardInterrupt:
            print(colored("\n\nSimulation interrupted by user", 'yellow'))
        except Exception as e:
            print(colored(f"\n\nError: {e}", 'red'))
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple cart-pendulum plant simulation")
    parser.add_argument('--theta', type=float, default=45.0, help='Initial pendulum angle (degrees)')
    parser.add_argument('--time', type=float, default=10.0, help='Simulation duration (seconds)')
    parser.add_argument('--control-mode', type=str, default='constant', 
                       choices=['constant', 'sine', 'step', 'ramp'],
                       help='Control input mode')
    parser.add_argument('--control-value', type=float, default=0.0, help='Control input value')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    print("\n" + colored("="*70, 'cyan'))
    print(colored("SIMPLE CART-PENDULUM PLANT SIMULATION", 'cyan', attrs=['bold']))
    print(colored("="*70, 'cyan'))
    
    manager = DrakeSceneManager(
        initial_theta=args.theta,
        simulation_time=args.time,
        control_mode=args.control_mode,
        control_value=args.control_value,
        visualize=not args.no_viz
    )
    
    manager.run_full_simulation()
    
    print(colored("\n" + "="*70, 'green'))
    print(colored("Simulation Complete!", 'green', attrs=['bold']))
    print(colored("="*70 + "\n", 'green'))


if __name__ == "__main__":
    main()
