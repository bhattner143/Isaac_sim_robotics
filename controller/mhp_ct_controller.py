"""Computed-torque controller alias for the MHP manipulator.

The core inverse-dynamics controller lives in :mod:`controller.controller`.
``MHPManipulator`` exposes the same joint/IK interface as
``CupManipulatorTendon``, so the existing implementation is reused unchanged.
"""
from controller.controller import ComputedTorqueController as MHPComputedTorqueController

__all__ = ["MHPComputedTorqueController"]
