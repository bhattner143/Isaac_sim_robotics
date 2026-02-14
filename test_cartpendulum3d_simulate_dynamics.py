#!/usr/bin/env python3
"""
Cart-Pendulum Dynamics Simulation with Force Actuation

This script simulates a 2D cart with a 3D gimbal-mounted pendulum,
applies forces to the cart, and plots the resulting dynamics.

System:
- Cart: 2 DOF (x, y position), actuated by forces [F_x, F_y]
- Pendulum: 2 DOF (pitch, roll), passive
- Total: 8D state [x, y, pitch, roll, ẋ, ẏ, pitch_dot, roll_dot]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
)
from termcolor import colored

# Import from main script
from robot_types import create_cart_pendulum_config
from test_cartpendulum3d_linearization import CartPendulum3D


class ForceController:
    """Simple force controller that applies time-varying forces to the cart."""
    
    def __init__(self, force_profile='step'):
        """
        Initialize force controller.
        
        Args:
            force_profile: Type of force profile
                - 'step': Step forces at different times
                - 'sinusoid': Sinusoidal forces
                - 'impulse': Impulse forces
                - 'ramp': Ramping forces
        """
        self.force_profile = force_profile
    
    def compute_force(self, t):
        """
        Compute force at time t.
        
        Args:
            t: Current time [s]
            
        Returns:
            forces: [F_x, F_y] force vector [N]
        """
        if self.force_profile == 'step':
            # Step forces at different times
            if t < 1.0:
                return np.array([0.0, 0.0])
            elif t < 3.0:
                return np.array([5.0, 0.0])  # Push in +X
            elif t < 5.0:
                return np.array([0.0, 5.0])  # Push in +Y
            elif t < 7.0:
                return np.array([-3.0, -3.0])  # Pull back
            else:
                return np.array([0.0, 0.0])
        
        elif self.force_profile == 'sinusoid':
            # Sinusoidal forces with different frequencies (reduced amplitude)
            Fx = 0.5 * np.sin(2.0 * np.pi * 0.5 * t)  # 0.5 Hz, 0.5N amplitude
            Fy = 0.3 * np.sin(2.0 * np.pi * 0.3 * t)  # 0.3 Hz, 0.3N amplitude
            return np.array([Fx, Fy])
        
        elif self.force_profile == 'impulse':
            # Short impulse forces
            Fx = 10.0 if 1.0 < t < 1.2 else 0.0
            Fy = 10.0 if 3.0 < t < 3.2 else 0.0
            return np.array([Fx, Fy])
        
        elif self.force_profile == 'ramp':
            # Ramping forces
            if t < 2.0:
                Fx = 2.5 * t
                Fy = 0.0
            elif t < 4.0:
                Fx = 5.0
                Fy = 2.5 * (t - 2.0)
            else:
                Fx = 5.0 - 2.5 * (t - 4.0)
                Fy = 5.0 - 2.5 * (t - 4.0)
            return np.array([max(Fx, 0.0), max(Fy, 0.0)])
        
        else:
            return np.array([0.0, 0.0])


def simulate_cart_pendulum(duration=10.0, force_profile='step', visualize=False):
    """
    Simulate cart-pendulum system with force actuation.
    
    Args:
        duration: Simulation duration [s]
        force_profile: Type of force profile to apply
        visualize: If True, enable Meshcat visualization
        
    Returns:
        log_data: Dictionary with logged data
    """
    print(colored("\n" + "="*80, "cyan"))
    print(colored("CART-PENDULUM DYNAMICS SIMULATION", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Create configuration
    config = create_cart_pendulum_config(
        cart_mass=0.3,  # Increased from 0.1 to reduce acceleration
        cart_size=0.1,
        cart_damping=0.5,  # Increased from 0.1 for better energy dissipation
        pendulum_mass=0.5,
        pendulum_length=0.2,
        pendulum_radius=0.05,
        pendulum_damping=0.05,  # Slightly increased
        attachment_offset=(0.0, 0.0, 0.0),
        initial_cart_x=0.0,
        initial_cart_y=0.0,
        initial_pitch=0.0,  # Start hanging down
        initial_roll=0.0,
        name="cart_pendulum"
    )
    
    # Build plant
    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(time_step=0.001))  # 1 kHz
    model_instance = plant.AddModelInstance("cart_pendulum_model")
    
    # Create cart-pendulum system
    cart_pendulum = CartPendulum3D(config, visualize_cart=True)
    cart_pendulum.attach_to_plant(plant, model_instance, register_visuals=False)
    
    # Finalize plant
    plant.Finalize()
    
    # Create context and set initial state
    context = plant.CreateDefaultContext()
    cart_pendulum.set_cart_state(context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0)
    cart_pendulum.set_pendulum_state(context, pitch=0.0, roll=0.0, pitch_dot=0.0, roll_dot=0.0)
    
    # Create force controller
    force_controller = ForceController(force_profile=force_profile)
    
    # Create logger for state
    state_logger = VectorLogSink(plant.num_multibody_states())
    state_log = builder.AddSystem(state_logger)
    builder.Connect(
        plant.get_state_output_port(),
        state_log.get_input_port()
    )
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
    
    # Set initial state
    cart_pendulum.set_cart_state(plant_context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0)
    cart_pendulum.set_pendulum_state(plant_context, pitch=0.0, roll=0.0, pitch_dot=0.0, roll_dot=0.0)
    
    print(colored(f"\n✓ System initialized", "green"))
    print(colored(f"  Force profile: {force_profile}", "cyan"))
    print(colored(f"  Simulation duration: {duration} s", "cyan"))
    print(colored(f"  Cart mass: {config.cart_mass} kg", "cyan"))
    print(colored(f"  Pendulum mass: {config.pendulum_mass} kg", "cyan"))
    print(colored(f"  Pendulum length: {config.pendulum_length} m", "cyan"))
    
    # Simulate with force actuation
    print(colored(f"\nSimulating...", "yellow"))
    
    dt = 0.01  # 100 Hz logging
    num_steps = int(duration / dt)
    
    # Pre-allocate arrays for logging
    time_log = np.zeros(num_steps)
    force_log = np.zeros((num_steps, 2))
    
    for i in range(num_steps):
        t = i * dt
        time_log[i] = t
        
        # Compute force at current time
        force = force_controller.compute_force(t)
        force_log[i] = force
        
        # Apply force to plant
        plant.get_actuation_input_port().FixValue(plant_context, force)
        
        # Advance simulation
        simulator.AdvanceTo(t + dt)
        
        # Progress indicator
        if i % 100 == 0:
            progress = 100 * i / num_steps
            print(f"\r  Progress: {progress:.1f}% (t = {t:.2f} s)", end='', flush=True)
    
    print(f"\r  Progress: 100.0% (t = {duration:.2f} s)")
    print(colored("✓ Simulation complete", "green"))
    
    # Extract logged data
    log = state_logger.FindLog(simulator_context)
    state_data = log.data()
    time_data = log.sample_times()
    
    # Parse state data
    # State order: [x, y, pitch, roll, x_dot, y_dot, pitch_dot, roll_dot]
    x = state_data[0, :]
    y = state_data[1, :]
    pitch = state_data[2, :]
    roll = state_data[3, :]
    x_dot = state_data[4, :]
    y_dot = state_data[5, :]
    pitch_dot = state_data[6, :]
    roll_dot = state_data[7, :]
    
    # Interpolate forces to match state log times
    force_x_interp = np.interp(time_data, time_log, force_log[:, 0])
    force_y_interp = np.interp(time_data, time_log, force_log[:, 1])
    
    return {
        'time': time_data,
        'x': x,
        'y': y,
        'pitch': pitch,
        'roll': roll,
        'x_dot': x_dot,
        'y_dot': y_dot,
        'pitch_dot': pitch_dot,
        'roll_dot': roll_dot,
        'force_x': force_x_interp,
        'force_y': force_y_interp,
    }


def plot_dynamics(log_data, force_profile='step'):
    """
    Plot cart-pendulum dynamics.
    
    Args:
        log_data: Dictionary with logged data
        force_profile: Force profile type (for title)
    """
    print(colored("\nGenerating plots...", "cyan"))
    
    t = log_data['time']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Forces
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, log_data['force_x'], 'b-', linewidth=2, label='$F_x$ (X-force)')
    ax1.plot(t, log_data['force_y'], 'r-', linewidth=2, label='$F_y$ (Y-force)')
    ax1.set_ylabel('Force [N]', fontsize=12, fontweight='bold')
    ax1.set_title(f'Applied Forces (Profile: {force_profile})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim([t[0], t[-1]])
    
    # Plot 2: Cart X position
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, log_data['x'], 'b-', linewidth=2)
    ax2.set_ylabel('X Position [m]', fontsize=11, fontweight='bold')
    ax2.set_title('Cart X Position', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([t[0], t[-1]])
    
    # Plot 3: Cart Y position
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, log_data['y'], 'r-', linewidth=2)
    ax3.set_ylabel('Y Position [m]', fontsize=11, fontweight='bold')
    ax3.set_title('Cart Y Position', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([t[0], t[-1]])
    
    # Plot 4: Pendulum pitch angle
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, np.rad2deg(log_data['pitch']), 'g-', linewidth=2)
    ax4.set_ylabel('Pitch Angle [deg]', fontsize=11, fontweight='bold')
    ax4.set_title('Pendulum Pitch (Y-axis rotation)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlim([t[0], t[-1]])
    
    # Plot 5: Pendulum roll angle
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, np.rad2deg(log_data['roll']), 'm-', linewidth=2)
    ax5.set_ylabel('Roll Angle [deg]', fontsize=11, fontweight='bold')
    ax5.set_title('Pendulum Roll (X-axis rotation)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlim([t[0], t[-1]])
    
    # Plot 6: Cart X-Y trajectory
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(log_data['x'], log_data['y'], 'b-', linewidth=2, alpha=0.7)
    ax6.plot(log_data['x'][0], log_data['y'][0], 'go', markersize=10, label='Start')
    ax6.plot(log_data['x'][-1], log_data['y'][-1], 'ro', markersize=10, label='End')
    ax6.set_xlabel('X Position [m]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Y Position [m]', fontsize=11, fontweight='bold')
    ax6.set_title('Cart Trajectory (X-Y Plane)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.legend(fontsize=10)
    
    # Plot 7: Cart velocities
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(t, log_data['x_dot'], 'b-', linewidth=2, label=r'$\dot{x}$')
    ax7.plot(t, log_data['y_dot'], 'r-', linewidth=2, label=r'$\dot{y}$')
    ax7.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Velocity [m/s]', fontsize=11, fontweight='bold')
    ax7.set_title('Cart Velocities', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=10)
    ax7.set_xlim([t[0], t[-1]])
    
    # Add X-axis labels to bottom row
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    
    plt.suptitle('Cart-Pendulum Dynamics Simulation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    print(colored("✓ Plots generated", "green"))
    
    return fig


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate cart-pendulum dynamics')
    parser.add_argument('--duration', type=float, default=10.0, 
                       help='Simulation duration [s] (default: 10.0)')
    parser.add_argument('--force-profile', type=str, default='sinusoid',
                       choices=['step', 'sinusoid', 'impulse', 'ramp'],
                       help='Force profile type (default: step)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (for testing)')
    args = parser.parse_args()
    
    # Run simulation
    log_data = simulate_cart_pendulum(
        duration=args.duration,
        force_profile=args.force_profile,
        visualize=False
    )
    
    # Plot results
    if not args.no_plot:
        fig = plot_dynamics(log_data, force_profile=args.force_profile)
        plt.show()
    
    print(colored("\n" + "="*80, "green"))
    print(colored("COMPLETE", "green", attrs=["bold"]))
    print(colored("="*80, "green"))


if __name__ == "__main__":
    main()
