#!/usr/bin/env python3
"""
Analyze the force profile and diagnose the drift issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

def analyze_force_profile():
    """Analyze the sinusoid force profile characteristics."""
    
    print(colored("\n" + "="*80, "cyan"))
    print(colored("FORCE PROFILE ANALYSIS", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # System parameters (from the simulation)
    cart_mass = 0.1  # kg (PROBLEM: Too light!)
    cart_damping = 0.1  # N·s/m
    
    # Force profile parameters
    Fx_amplitude = 3.0  # N
    Fy_amplitude = 2.0  # N
    fx_freq = 0.5  # Hz
    fy_freq = 0.3  # Hz
    duration = 10.0  # s
    
    print(colored("\nSystem Parameters:", "yellow", attrs=["bold"]))
    print(colored(f"  Cart mass: {cart_mass} kg", "white"))
    print(colored(f"  Cart damping: {cart_damping} N·s/m", "white"))
    
    print(colored("\nForce Parameters:", "yellow", attrs=["bold"]))
    print(colored(f"  Fx amplitude: {Fx_amplitude} N", "white"))
    print(colored(f"  Fy amplitude: {Fy_amplitude} N", "white"))
    print(colored(f"  Fx frequency: {fx_freq} Hz (period = {1/fx_freq} s)", "white"))
    print(colored(f"  Fy frequency: {fy_freq} Hz (period = {1/fy_freq} s)", "white"))
    
    # Calculate peak accelerations
    ax_peak = Fx_amplitude / cart_mass
    ay_peak = Fy_amplitude / cart_mass
    
    print(colored("\nPeak Accelerations:", "yellow", attrs=["bold"]))
    print(colored(f"  X-direction: {ax_peak:.1f} m/s² ({ax_peak/9.81:.1f}g)", "red" if ax_peak > 10 else "white"))
    print(colored(f"  Y-direction: {ay_peak:.1f} m/s² ({ay_peak/9.81:.1f}g)", "red" if ay_peak > 10 else "white"))
    
    if ax_peak > 10:
        print(colored("\n⚠️  WARNING: X-acceleration exceeds 10 m/s²!", "red", attrs=["bold"]))
    if ay_peak > 10:
        print(colored("⚠️  WARNING: Y-acceleration exceeds 10 m/s²!", "red", attrs=["bold"]))
    
    # Estimate velocity accumulation (without damping)
    # For sinusoidal force F = A*sin(2πft), velocity accumulates as:
    # v(t) = ∫(F/m)dt = -(A/m)·cos(2πft)/(2πf) + C
    # Peak velocity amplitude (without damping): v_peak = A/(2πfm)
    
    vx_peak_undamped = Fx_amplitude / (2 * np.pi * fx_freq * cart_mass)
    vy_peak_undamped = Fy_amplitude / (2 * np.pi * fy_freq * cart_mass)
    
    print(colored("\nPeak Velocity (without damping):", "yellow", attrs=["bold"]))
    print(colored(f"  X-direction: {vx_peak_undamped:.2f} m/s", "white"))
    print(colored(f"  Y-direction: {vy_peak_undamped:.2f} m/s", "white"))
    
    # Estimate position drift (rough approximation)
    # With weak damping, position can accumulate significantly
    # Rough estimate: x ≈ v_peak * t (worst case)
    
    x_drift_estimate = vx_peak_undamped * duration / 2  # Divide by 2 for average
    y_drift_estimate = vy_peak_undamped * duration / 2
    
    print(colored("\nEstimated Position Drift (10s):", "yellow", attrs=["bold"]))
    print(colored(f"  X-direction: ~{x_drift_estimate:.1f} m", "red"))
    print(colored(f"  Y-direction: ~{y_drift_estimate:.1f} m", "red"))
    
    # Compare with observed drift from plot
    observed_x_drift = 8.0  # From the plot
    observed_y_drift = 8.0  # From the plot
    
    print(colored(f"\n  Observed X drift: {observed_x_drift} m (from plot)", "cyan"))
    print(colored(f"  Observed Y drift: {observed_y_drift} m (from plot)", "cyan"))
    
    # Recommendations
    print(colored("\n" + "="*80, "green"))
    print(colored("RECOMMENDATIONS", "green", attrs=["bold"]))
    print(colored("="*80, "green"))
    
    print(colored("\n1. Increase cart mass:", "yellow"))
    print(colored(f"   Current: {cart_mass} kg → Recommended: 1.0 kg", "white"))
    print(colored(f"   This reduces peak acceleration from {ax_peak:.1f} to {Fx_amplitude/1.0:.1f} m/s²", "white"))
    
    print(colored("\n2. Reduce force amplitude:", "yellow"))
    print(colored(f"   Current: Fx={Fx_amplitude}N, Fy={Fy_amplitude}N", "white"))
    print(colored(f"   Recommended: Fx=0.5N, Fy=0.3N (10x smaller)", "white"))
    
    print(colored("\n3. Increase damping:", "yellow"))
    print(colored(f"   Current: {cart_damping} N·s/m", "white"))
    print(colored(f"   Recommended: 0.5-1.0 N·s/m for better energy dissipation", "white"))
    
    print(colored("\n4. Alternative: Use bounded force profile:", "yellow"))
    print(colored("   Instead of continuous sinusoid, use:", "white"))
    print(colored("   - Time-limited pulses", "white"))
    print(colored("   - Damped sinusoids: F(t) = A·sin(ωt)·exp(-βt)", "white"))
    print(colored("   - Step-return-to-zero pattern", "white"))
    
    # Calculate improved parameters
    cart_mass_new = 1.0
    Fx_new = 0.5
    Fy_new = 0.3
    
    ax_new = Fx_new / cart_mass_new
    ay_new = Fy_new / cart_mass_new
    vx_new = Fx_new / (2 * np.pi * fx_freq * cart_mass_new)
    vy_new = Fy_new / (2 * np.pi * fy_freq * cart_mass_new)
    x_drift_new = vx_new * duration / 2
    y_drift_new = vy_new * duration / 2
    
    print(colored("\n" + "="*80, "magenta"))
    print(colored("IMPROVED CONFIGURATION ESTIMATES", "magenta", attrs=["bold"]))
    print(colored("="*80, "magenta"))
    
    print(colored(f"\nCart mass: {cart_mass_new} kg", "cyan"))
    print(colored(f"Forces: Fx={Fx_new}N, Fy={Fy_new}N", "cyan"))
    
    print(colored(f"\nPeak accelerations:", "yellow"))
    print(colored(f"  X: {ax_new:.2f} m/s² ({ax_new/9.81:.2f}g)", "green"))
    print(colored(f"  Y: {ay_new:.2f} m/s² ({ay_new/9.81:.2f}g)", "green"))
    
    print(colored(f"\nPeak velocities:", "yellow"))
    print(colored(f"  X: {vx_new:.2f} m/s", "green"))
    print(colored(f"  Y: {vy_new:.2f} m/s", "green"))
    
    print(colored(f"\nEstimated drift (10s):", "yellow"))
    print(colored(f"  X: ~{x_drift_new:.2f} m", "green"))
    print(colored(f"  Y: ~{y_drift_new:.2f} m", "green"))
    
    print(colored("\n✓ Much more reasonable!", "green", attrs=["bold"]))

if __name__ == "__main__":
    analyze_force_profile()
    print()
