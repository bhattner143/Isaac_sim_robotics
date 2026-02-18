#!/usr/bin/env python3
"""
Simple test of Finite-Horizon LQR for 2D Force Control

This demonstrates the LQR architecture without the full Drake simulation,
showing how LQR computes optimal neural commands u to generate muscle forces F.

SYSTEM:
    State: [x, y, θ, ẋ, ẏ, θ̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref] (12D)
    Control: [u_x, u_y] (2D neural commands)
    
CONTROL LAW:
    u(t) = -K(t) · (x(t) - x_goal)
    where K(t) is the time-varying LQR gain
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from termcolor import colored


class FiniteHorizonLQRController:
    """
    Finite-horizon LQR for 2D system.
    
    Solves: min ∫[x'Qx + u'Ru]dt + x(T)'QN·x(T)
    Subject to: ẋ = Ax + Bu
    """
    
    def __init__(self, A, B, Q, QN, R, x_goal, horizon=10.0, timestep=0.01):
        self.A = A
        self.B = B
        self.Q = Q
        self.QN = QN
        self.R = R
        self.x_goal = x_goal
        self.horizon = horizon
        self.timestep = timestep
        
        # Discretize
        self.Ad, self.Bd = self._discretize(A, B, timestep)
        
        # Solve Riccati recursion
        self.K_history, self.time_points = self._solve_lqr()
        
        print(colored(f"✓ LQR Controller initialized", "green"))
        print(colored(f"  State dim: {A.shape[0]}, Control dim: {B.shape[1]}", "cyan"))
        print(colored(f"  Horizon: {horizon:.1f} s, Steps: {len(self.time_points)}", "cyan"))
    
    def compute_control(self, t, x):
        """Compute u(t) = -K(t)·(x - x_goal)"""
        idx = int(np.clip(t / self.timestep, 0, len(self.time_points) - 1))
        K_t = self.K_history[idx]
        u = -K_t @ (x - self.x_goal)
        return u
    
    @staticmethod
    def _discretize(A, B, dt):
        """Zero-order hold discretization"""
        n = A.shape[0]
        Ad = np.eye(n) + A * dt
        Bd = B * dt
        return Ad, Bd
    
    def _solve_lqr(self):
        """Backward Riccati recursion"""
        N = int(self.horizon / self.timestep)
        n = self.Ad.shape[0]
        m = self.Bd.shape[1]
        
        # Storage
        P = [None] * (N + 1)
        K_history = [None] * (N + 1)
        time_points = np.arange(N + 1) * self.timestep
        
        # Terminal condition
        P[N] = self.QN
        K_history[N] = np.zeros((m, n))
        
        # Backward recursion
        for k in range(N - 1, -1, -1):
            P_next = P[k + 1]
            
            # Gain: K = (R + B'PB)^-1 · B'PA
            BtPB = self.Bd.T @ P_next @ self.Bd
            K_k = np.linalg.solve(self.R + BtPB, self.Bd.T @ P_next @ self.Ad)
            
            # Cost-to-go: P = Q + A'PA - A'PBK
            P_k = self.Q + self.Ad.T @ P_next @ self.Ad - self.Ad.T @ P_next @ self.Bd @ K_k
            
            K_history[k] = K_k
            P[k] = P_k
        
        return K_history, time_points


def build_2d_system_matrices(K_imp=100.0, D_imp=20.0, M_ref=2.0, 
                             M_cart=5.0, muscle_tau=0.03):
    """
    Build linearized 12D system for manipulator-cart.
    
    State: [x, y, θ, ẋ, ẏ, θ̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    """
    A = np.zeros((12, 12))
    
    # Position derivatives
    A[0, 3] = 1.0   # x_dot = ẋ
    A[1, 4] = 1.0   # y_dot = ẏ
    A[2, 5] = 1.0   # θ_dot = θ̇
    
    # Cart dynamics: ẍ = (K/M)*(x_ref - x) + (D/M)*(ẋ_ref - ẋ)
    A[3, 0] = -K_imp / M_cart
    A[3, 3] = -D_imp / M_cart
    A[3, 8] = K_imp / M_cart
    A[3, 10] = D_imp / M_cart
    
    A[4, 1] = -K_imp / M_cart
    A[4, 4] = -D_imp / M_cart
    A[4, 9] = K_imp / M_cart
    A[4, 11] = D_imp / M_cart
    
    # Pendulum: θ̈ = (g/L)*θ
    A[5, 2] = 9.81 / 0.2
    
    # Muscle: Ḟ = (-F + u) / τ
    A[6, 6] = -1.0 / muscle_tau
    A[7, 7] = -1.0 / muscle_tau
    
    # ZFT reference: ẍ_ref = (K*(x-x_ref) + D*(ẋ-ẋ_ref) + F) / M_ref
    A[8, 10] = 1.0   # x_ref_dot = ẋ_ref
    A[9, 11] = 1.0   # y_ref_dot = ẏ_ref
    
    A[10, 0] = K_imp / M_ref
    A[10, 3] = D_imp / M_ref
    A[10, 8] = -K_imp / M_ref
    A[10, 10] = -D_imp / M_ref
    A[10, 6] = 1.0 / M_ref  # F_x effect
    
    A[11, 1] = K_imp / M_ref
    A[11, 4] = D_imp / M_ref
    A[11, 9] = -K_imp / M_ref
    A[11, 11] = -D_imp / M_ref
    A[11, 7] = 1.0 / M_ref  # F_y effect
    
    # Input matrix B (control affects muscle dynamics)
    B = np.zeros((12, 2))
    B[6, 0] = 1.0 / muscle_tau  # u_x → Ḟ_x
    B[7, 1] = 1.0 / muscle_tau  # u_y → Ḟ_y
    
    return A, B


def simulate_lqr_control(duration=5.0, dx=0.3, dy=0.2):
    """Simulate LQR control in simplified dynamics"""
    
    print(colored("\n" + "="*80, "cyan"))
    print(colored("FINITE-HORIZON LQR TEST - 2D FORCE CONTROL", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    # Build system
    K_imp = 100.0
    D_imp = 20.0
    M_ref = 2.0
    M_cart = 5.0
    muscle_tau = 0.03
    
    A, B = build_2d_system_matrices(K_imp, D_imp, M_ref, M_cart, muscle_tau)
    print(colored(f"✓ System matrices: A({A.shape}), B({B.shape})", "green"))
    
    # LQR cost matrices
    Q = np.diag([100, 100, 10, 10, 10, 10, 1, 1, 50, 50, 10, 10])
    QN = Q * 10.0
    R = np.diag([0.01, 0.01])
    
    # Goal: move cart to (dx, dy)
    x_goal = np.array([dx, dy, 0, 0, 0, 0, 0, 0, dx, dy, 0, 0])
    
    print(colored(f"✓ Goal: cart at ({dx:.2f}, {dy:.2f}) m", "green"))
    
    # Create LQR controller
    lqr = FiniteHorizonLQRController(A, B, Q, QN, R, x_goal, horizon=10.0, timestep=0.01)
    
    # Simulate
    dt = 0.01
    t_span = np.arange(0, duration, dt)
    N = len(t_span)
    
    # State history
    x_hist = np.zeros((12, N))
    u_hist = np.zeros((2, N))
    
    # Initial state (all zeros)
    x = np.zeros(12)
    
    print(colored(f"\n🚀 Simulating {duration} s...", "yellow"))
    
    for i, t in enumerate(t_span):
        # Compute control
        u = lqr.compute_control(t, x)
        
        # Clip control (optional)
        u = np.clip(u, -100, 100)
        
        # State derivative
        x_dot = A @ x + B @ u
        
        # Euler integration
        x = x + x_dot * dt
        
        # Store
        x_hist[:, i] = x
        u_hist[:, i] = u
    
    print(colored("✓ Simulation complete\n", "green"))
    
    return t_span, x_hist, u_hist, x_goal


def plot_results(t, x_hist, u_hist, x_goal, dx_target, dy_target):
    """Plot LQR control results"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Cart position
    plt.subplot(3, 3, 1)
    plt.plot(t, x_hist[0, :], 'b-', linewidth=2, label='x')
    plt.plot(t, x_hist[1, :], 'r-', linewidth=2, label='y')
    plt.axhline(dx_target, color='b', linestyle='--', alpha=0.5, label=f'x_target={dx_target:.2f}')
    plt.axhline(dy_target, color='r', linestyle='--', alpha=0.5, label=f'y_target={dy_target:.2f}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Cart Position (LQR Control)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Cart velocity
    plt.subplot(3, 3, 2)
    plt.plot(t, x_hist[3, :], 'b-', linewidth=2, label='ẋ')
    plt.plot(t, x_hist[4, :], 'r-', linewidth=2, label='ẏ')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Cart Velocity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Pendulum angle
    plt.subplot(3, 3, 3)
    plt.plot(t, np.rad2deg(x_hist[2, :]), 'g-', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.title('Pendulum Angle')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Muscle forces
    plt.subplot(3, 3, 4)
    plt.plot(t, x_hist[6, :], 'b-', linewidth=2, label='F_x')
    plt.plot(t, x_hist[7, :], 'r-', linewidth=2, label='F_y')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.title('Muscle Forces')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Reference position
    plt.subplot(3, 3, 5)
    plt.plot(t, x_hist[8, :], 'b-', linewidth=2, label='x_ref')
    plt.plot(t, x_hist[9, :], 'r-', linewidth=2, label='y_ref')
    plt.axhline(dx_target, color='b', linestyle='--', alpha=0.5)
    plt.axhline(dy_target, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('ZFT Reference Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Reference velocity
    plt.subplot(3, 3, 6)
    plt.plot(t, x_hist[10, :], 'b-', linewidth=2, label='ẋ_ref')
    plt.plot(t, x_hist[11, :], 'r-', linewidth=2, label='ẏ_ref')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('ZFT Reference Velocity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 7: Neural commands
    plt.subplot(3, 3, 7)
    plt.plot(t, u_hist[0, :], 'b-', linewidth=2, label='u_x')
    plt.plot(t, u_hist[1, :], 'r-', linewidth=2, label='u_y')
    plt.xlabel('Time [s]')
    plt.ylabel('Command [N]')
    plt.title('LQR Neural Commands u(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 8: 2D Trajectory
    plt.subplot(3, 3, 8)
    plt.plot(x_hist[0, :], x_hist[1, :], 'b-', linewidth=2, label='Cart path')
    plt.plot(x_hist[8, :], x_hist[9, :], 'r--', linewidth=2, label='Ref path')
    plt.plot(0, 0, 'go', markersize=10, label='Start')
    plt.plot(dx_target, dy_target, 'r*', markersize=15, label='Target')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('2D Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    # Plot 9: Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    final_x = x_hist[0, -1]
    final_y = x_hist[1, -1]
    error_x = final_x - dx_target
    error_y = final_y - dy_target
    
    summary = f"""
LQR CONTROL SUMMARY

Target:
  x = {dx_target:.3f} m
  y = {dy_target:.3f} m

Final Position:
  x = {final_x:.3f} m
  y = {final_y:.3f} m

Error:
  Δx = {error_x:.3f} m
  Δy = {error_y:.3f} m

Final Forces:
  F_x = {x_hist[6,-1]:.2f} N
  F_y = {x_hist[7,-1]:.2f} N

Final Commands:
  u_x = {u_hist[0,-1]:.2f} N
  u_y = {u_hist[1,-1]:.2f} N
"""
    plt.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Finite-Horizon LQR for 2D Force Control\nOptimal Neural Commands u → Muscle Forces F',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Run LQR test"""
    t, x_hist, u_hist, x_goal = simulate_lqr_control(
        duration=5.0,
        dx=0.3,
        dy=0.2
    )
    
    plot_results(t, x_hist, u_hist, x_goal, 0.3, 0.2)
    
    print(colored("\n" + "="*80, "cyan"))
    print(colored("✓ LQR Test Complete!", "green", attrs=["bold"]))
    print(colored("="*80 + "\n", "cyan"))


if __name__ == "__main__":
    main()
