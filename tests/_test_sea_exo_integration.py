#!/usr/bin/env python3
"""Quick integration test for SEA + Exo diagram build & advance."""
import sys, numpy as np
sys.path.insert(0, '.')

from robots.cup_manipulator_tendon_with_exo import CupManipulatorTendonWithExo
from robots.cup_manipulator_tendon import create_cable_manipulator_config
from pydrake.all import (
    DiagramBuilder, MultibodyPlant, SceneGraph, Parser, Simulator,
    SpatialInertia, UnitInertia, PiecewisePolynomial, LeafSystem,
)
from actuators.sea import SEACableActuator
from actuators.sea_exo import SEAExoActuator
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor
from controller.controller import ComputedTorqueController

# Import helper LeafSystems from the simulation script
from script_cup_manipulator_pendulam_tendon_with_exo_pydrake import (
    _ActuationSum, _ExoCommandSource,
)

_URDF = ('model_using_onshape_to_robot/'
         'manipulator_cable_exo_springs_elbow_follow/'
         'manipulator_cable_exo_springs_elbow_follow_obj.urdf')
_DT = 0.01

print("Building diagram...")
config = create_cable_manipulator_config(urdf_path=_URDF, damping=(0.05, 0.05))
builder = DiagramBuilder()
plant = MultibodyPlant(time_step=_DT)
sg = builder.AddSystem(SceneGraph())
plant.RegisterAsSourceForSceneGraph(sg)

manip = CupManipulatorTendonWithExo(config)
p = Parser(plant)
manip.load_urdf_to_plant(plant, p)
manip.weld_base_to_world(plant)
manip.add_joint_actuators(plant)
manip.set_joint_properties(plant)
manip.add_end_effector_frame(plant)
plant.Finalize()
builder.AddSystem(plant)

builder.Connect(plant.get_geometry_pose_output_port(),
                sg.get_source_pose_port(plant.get_source_id()))
builder.Connect(sg.get_query_output_port(),
                plant.get_geometry_query_input_port())

motor = get_motor('AK60_6_KV80_Config')

ct = builder.AddSystem(
    ComputedTorqueController(plant, manip, Kp=100, Kd=40, tau_max=9))
sea = builder.AddSystem(
    SEACableActuator(plant, manip, k_s=30, b_c=2, tau_max=9, dt=_DT,
                     motor_mode=MotorMode.TORQUE, motor_cfg=motor))
exo = builder.AddSystem(
    SEAExoActuator(plant, manip, k_exo=200, b_exo=2, r_exo=0.04775,
                   tau_max=9, dt=_DT, motor_cfg=motor))
act_sum = builder.AddSystem(_ActuationSum())
exo_cmd = builder.AddSystem(_ExoCommandSource(t_activate=1.0, delta_theta=0.1))

# Simple constant trajectory source
class _ConstSrc(LeafSystem):
    def __init__(self, val):
        super().__init__()
        self._v = np.array(val, float)
        self.DeclareVectorOutputPort('out', len(self._v), self._c)
    def _c(self, ctx, out):
        out.SetFromVector(self._v)

ee_src  = builder.AddSystem(_ConstSrc([0.5, 0.0]))
vel_src = builder.AddSystem(_ConstSrc([0.0, 0.0]))
acc_src = builder.AddSystem(_ConstSrc([0.0, 0.0]))

# Wire everything
builder.Connect(ee_src.get_output_port(),  ct.GetInputPort('desired_ee_pos'))
builder.Connect(vel_src.get_output_port(), ct.GetInputPort('ee_vel_ref'))
builder.Connect(acc_src.get_output_port(), ct.GetInputPort('ee_acc_ref'))
builder.Connect(plant.get_state_output_port(), ct.GetInputPort('plant_state'))
builder.Connect(ct.GetOutputPort('actuation'),     sea.GetInputPort('tau_desired'))
builder.Connect(plant.get_state_output_port(),     sea.GetInputPort('plant_state'))
builder.Connect(exo_cmd.get_output_port(),         exo.GetInputPort('activate_cmd'))
builder.Connect(plant.get_state_output_port(),     exo.GetInputPort('plant_state'))
builder.Connect(sea.GetOutputPort('actuation'),    act_sum.GetInputPort('drive_actuation'))
builder.Connect(exo.GetOutputPort('exo_torque'),   act_sum.GetInputPort('exo_torque'))
builder.Connect(act_sum.GetOutputPort('actuation'),plant.get_actuation_input_port())

print('✓ Diagram wired')

diagram = builder.Build()
sim = Simulator(diagram)
ctx = sim.get_mutable_context()

# Patch zero-mass + init
pc = plant.GetMyMutableContextFromRoot(ctx)
_M = SpatialInertia(mass=0.3, p_PScm_E=np.zeros(3),
                    G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2))
patched = []
for idx in plant.GetBodyIndices(manip.model_instance):
    body = plant.get_body(idx)
    if body.default_mass() < 1e-6:
        body.SetSpatialInertiaInBodyFrame(pc, _M)
        patched.append(body.name())
if patched:
    print(f"  Patched: {patched}")

q_init = np.array([np.deg2rad(5), np.deg2rad(15)])
manip.set_positions_user_order(plant, pc, q_init)

sea_ctx = sea.GetMyMutableContextFromRoot(ctx)
sea.initialize_spring_at_rest(sea_ctx, q_init[1])
exo_ctx = exo.GetMyMutableContextFromRoot(ctx)
exo.initialize_at_rest(exo_ctx, q_init[1])

sim.Initialize()
print('✓ Simulator initialised')

# t=0.5: exo should be OFF
sim.AdvanceTo(0.5)
ec = exo.GetMyMutableContextFromRoot(sim.get_mutable_context())
d = exo.GetOutputPort('diagnostics').Eval(ec)
print(f'  t=0.5: activated={d[9]:.0f}  δ_R={d[0]*1e3:.3f}mm  '
      f'δ_L={d[1]*1e3:.3f}mm  τ_exo={d[8]:.4f}Nm')
assert d[9] == 0.0, "Exo should be OFF at t=0.5"
assert abs(d[0]) < 1e-3, f"δ_R should be ≈0 when deactivated, got {d[0]*1e3:.3f}mm"
assert abs(d[1]) < 1e-3, f"δ_L should be ≈0 when deactivated, got {d[1]*1e3:.3f}mm"

# t=1.5: exo should be ON
sim.AdvanceTo(1.5)
ec = exo.GetMyMutableContextFromRoot(sim.get_mutable_context())
d = exo.GetOutputPort('diagnostics').Eval(ec)
print(f'  t=1.5: activated={d[9]:.0f}  δ_R={d[0]*1e3:.3f}mm  '
      f'δ_L={d[1]*1e3:.3f}mm  τ_exo={d[8]:.4f}Nm  '
      f'F_R={d[2]:.2f}N  F_L={d[3]:.2f}N')
assert d[9] == 1.0, "Exo should be ON at t=1.5"
assert d[0] > 0 and d[1] > 0, "Both springs should be extended in co-contraction"
assert d[2] > 0 and d[3] > 0, "Both cables should be under tension"

print('\n✅ ALL TESTS PASSED')
