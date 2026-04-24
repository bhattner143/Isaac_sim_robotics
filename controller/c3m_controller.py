"""
controller/c3m_controller.py

C3M (Control Contraction Metrics) tracking controller for Drake.

Loads a pre-trained C3M checkpoint (PyTorch) and deploys it as a
Drake LeafSystem.  The controller takes the full 6D state + reference
trajectory and outputs joint torque commands [τ₁, τ₂_des].

Architecture
────────────
::

    Trajectory ──→ C3MController ──→ SEACableActuator ──→ Plant
     x_ref(t)      [τ₁, τ₂_des]     [τ₁, r_p·F_cable]

The C3M controller replaces the Computed Torque controller.  The SEA
actuator block remains unchanged — it models physical reality.

The controller network computes:
    xe = x - x_ref
    u  = w2([x_eff, x_ref_eff]) · tanh(w1([x_eff, x_ref_eff]) · xe) + u_ref

where w1, w2 are learned neural networks and u_ref is the feedforward
control at the reference trajectory.

Input ports
───────────
    ``plant_state``    [n]   plant multibody state
    ``x_ref``          [6]   reference state [q1, q2, q̇1, q̇2, θ_m, θ̇_m]
    ``u_ref``          [2]   feedforward control [τ₁_ref, τ₂_ref]

Output ports
────────────
    ``actuation``      [2]   joint torque commands [τ₁, τ₂_des]
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector,
)

if TYPE_CHECKING:
    from robots.cup_manipulator_tendon import CupManipulatorTendon


class C3MController(LeafSystem):
    """C3M tracking controller deployed as a Drake LeafSystem.

    Parameters
    ----------
    plant          Finalized MultibodyPlant.
    manipulator    CupManipulatorTendon instance.
    checkpoint_path  Path to ``controller_best.pth.tar`` from C3M training.
    tau_max        Torque saturation [Nm].
    effective_dim_start, effective_dim_end
                   Slice of the state vector used by the neural network.
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        checkpoint_path: str,
        tau_max: float = 9.0,
        effective_dim_start: int = 0,
        effective_dim_end: int = 6,
    ) -> None:
        super().__init__()
        self._plant = plant
        self._manip = manipulator
        self._tau_max = float(tau_max)
        self._eff_start = effective_dim_start
        self._eff_end = effective_dim_end

        # Velocity-vector indices for user-order joints [q1, q2]
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # ── Load trained controller ──────────────────────────────────────────
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"C3M checkpoint not found: {checkpoint_path}\n"
                "Train with: python contraction-theory/C3M/main.py "
                "--task CUPMANIP_SEA --log checkpoints/cupmanip_sea/lambda_1.5"
            )
        # torch.load unpickles the U_FUNC class, which requires model_CUPMANIP_SEA
        # to be importable. Add the C3M models/ directory to sys.path temporarily.
        # Path: .../C3M/checkpoints/cupmanip_sea/lambda_X/controller_best.pth.tar
        #       → go up 3 levels to C3M/, then into models/
        _c3m_models = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)),
            '..', '..', '..', 'models'
        )
        _c3m_models = os.path.normpath(_c3m_models)
        if _c3m_models not in sys.path:
            sys.path.insert(0, _c3m_models)
        self._controller = torch.load(
            checkpoint_path, map_location=torch.device('cpu'), weights_only=False,
        )
        self._controller.cpu()
        self._controller.eval()

        # ── Ports ────────────────────────────────────────────────────────────
        nstate = plant.num_multibody_states()
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)
        self._xref_port = self.DeclareVectorInputPort("x_ref", 6)
        self._uref_port = self.DeclareVectorInputPort("u_ref", 2)

        # Declare output with explicit prerequisites to avoid algebraic loops.
        # The sea_diagnostics port (added by subclass) reads *discrete* motor
        # state from the previous timestep, so it is NOT a true feedthrough.
        self.DeclareVectorOutputPort(
            "actuation", 2, self._calc_actuation,
            prerequisites_of_calc={
                self.input_port_ticket(self._state_port.get_index()),
                self.input_port_ticket(self._xref_port.get_index()),
                self.input_port_ticket(self._uref_port.get_index()),
            },
        )

    def _get_full_state(self, context) -> np.ndarray:
        """Extract 6D state [q1, q2, q̇1, q̇2, θ_m, θ̇_m] from plant state.

        Note: θ_m and θ̇_m are NOT in the plant state — they live in the SEA
        actuator's discrete state.  For the C3M controller, these are read
        from the reference state (x_ref) as an approximation, or from a
        separate diagnostic port.

        For a cleaner integration, the SEA actuator should export its motor
        state.  For now, we read the plant's 4 states and fill motor states
        from a separate input.
        """
        state = self._state_port.Eval(context)
        nq = self._plant.num_positions()
        q_all = state[:nq]
        v_all = state[nq:]

        q1 = q_all[self._v_idx[0]]
        q2 = q_all[self._v_idx[1]]
        q1_dot = v_all[self._v_idx[0]]
        q2_dot = v_all[self._v_idx[1]]

        return np.array([q1, q2, q1_dot, q2_dot])

    def _calc_actuation(self, context, output):
        """Compute C3M control action."""
        plant_state_4d = self._get_full_state(context)
        x_ref = self._xref_port.Eval(context)   # 6D reference
        u_ref = self._uref_port.Eval(context)   # 2D feedforward

        # Build full 6D state — motor states from x_ref (at-rest approximation)
        # In a proper integration, θ_m and θ̇_m would come from SEA diagnostics.
        # For tracking tasks near the reference, this is a reasonable approximation.
        x_full = np.array([
            plant_state_4d[0],  # q1
            plant_state_4d[1],  # q2
            plant_state_4d[2],  # q̇1
            plant_state_4d[3],  # q̇2
            x_ref[4],           # θ_m (from reference — best effort without SEA port)
            x_ref[5],           # θ̇_m
        ])

        # Tracking error
        xe = x_full - x_ref

        # Convert to torch tensors (batch dim = 1)
        x_t = torch.from_numpy(x_full).float().view(1, -1, 1)
        xe_t = torch.from_numpy(xe).float().view(1, -1, 1)
        uref_t = torch.from_numpy(u_ref).float().view(1, -1, 1)

        with torch.no_grad():
            u_t = self._controller(x_t, xe_t, uref_t)

        u = u_t.squeeze(0).numpy().ravel()

        # Saturate
        u_clipped = np.clip(u, -self._tau_max, self._tau_max)
        output.SetFromVector(u_clipped)


class C3MControllerWithMotorState(C3MController):
    """Extended C3M controller that reads motor state from SEA diagnostics.

    This variant reads the actual motor state (θ_m, θ̇_m) from the SEA
    actuator's diagnostics port instead of approximating from the reference.

    Additional input port
    ─────────────────────
    ``sea_diagnostics``  [9]   from SEACableActuator.diagnostics output
        slot [0] = θ_m / N  (joint-side motor position)  [rad]
        slot [1] = θ̇_m / N  (joint-side motor velocity)  [rad/s]
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        checkpoint_path: str,
        gear_ratio: float = 6.0,
        tau_max: float = 9.0,
        effective_dim_start: int = 0,
        effective_dim_end: int = 6,
    ) -> None:
        super().__init__(
            plant, manipulator, checkpoint_path,
            tau_max=tau_max,
            effective_dim_start=effective_dim_start,
            effective_dim_end=effective_dim_end,
        )
        self._N = float(gear_ratio)
        self._diag_port = self.DeclareVectorInputPort("sea_diagnostics", 9)

    def _calc_actuation(self, context, output):
        """Compute C3M control using actual motor state from SEA diagnostics."""
        plant_state_4d = self._get_full_state(context)
        x_ref = self._xref_port.Eval(context)
        u_ref = self._uref_port.Eval(context)

        # Read actual motor state from SEA diagnostics
        diag = self._diag_port.Eval(context)
        # diag[0] = θ_m / N (joint-side), diag[1] = θ̇_m / N (joint-side)
        theta_m = diag[0] * self._N       # convert back to motor-side
        theta_m_dot = diag[1] * self._N

        x_full = np.array([
            plant_state_4d[0],  # q1
            plant_state_4d[1],  # q2
            plant_state_4d[2],  # q̇1
            plant_state_4d[3],  # q̇2
            theta_m,            # θ_m (actual from SEA)
            theta_m_dot,        # θ̇_m (actual from SEA)
        ])

        xe = x_full - x_ref
        x_t = torch.from_numpy(x_full).float().view(1, -1, 1)
        xe_t = torch.from_numpy(xe).float().view(1, -1, 1)
        uref_t = torch.from_numpy(u_ref).float().view(1, -1, 1)

        with torch.no_grad():
            u_t = self._controller(x_t, xe_t, uref_t)

        u = u_t.squeeze(0).numpy().ravel()
        u_clipped = np.clip(u, -self._tau_max, self._tau_max)
        output.SetFromVector(u_clipped)
