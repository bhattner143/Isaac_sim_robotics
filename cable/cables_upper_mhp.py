"""Upper (elbow -Y) cable route configuration for MHP."""
from __future__ import annotations

from cable.cable_config_mhp import CableComponent, CableRouteConfig

def build_upper_cable_config() -> CableRouteConfig:
    """Config for the upper (−Y) half of the elbow antagonistic cable pair.

    Routed from the shoulder spool groove (−Y side) to the elbow roller.
    When this side is taut, the lower (+Y) cable is slack.
    """
    _C_LINE  = "#8B34C4"   # cable line — purple
    _C_GUIDE = "#333333"   # guide pulleys
    _C_ROLLER= "#888888"   # elbow roller
    _C_BALL  = "#87CEEB"   # sky-blue ball markers
    _C_LO_ARM= "#FFA500"   # lower-arm entry/exit

    physical = [
        CableComponent(
            name="Cable anchor (clamp on spool rim)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 0.020, 0.0645],  # on spool rim at +Y, r=20mm from centre
            diameter_mm=0.0,
            color=_C_LINE,
            role="cable_anchor",
            cable="upper",
            note="Physical clamp point where upper cable is fixed to spool drum, +Y side",
        ),
        CableComponent(
            name="Shoulder drive spool",
            obj_filename="mhp_arm_00_elbow_spool_v2.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 1.76602e-13, 0.0645],
            diameter_mm=40.0,
            color=_C_LINE,
            role="spool",
            cable="upper",
            note="Drive spool drum — upper cable groove, Z=64.5 mm",
            visual_pos_in_link=[-0.0795, 1.76602e-13, 0.0645],
            visual_rpy=[0.0, 0.0, 3.14159],
        ),
        CableComponent(
            name="Guide pulley 1 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[-0.0409243, -0.03445, 0.0475],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="First guide pulley, -Y side",
        ),
        CableComponent(
            name="Guide pulley 2 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.349567, -0.03445, 0.0476],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="Mid-span guide pulley, -Y side",
        ),
        CableComponent(
            name="Guide pulley 3 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.360376, -0.0226536, 0.0476],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="Pre-elbow guide pulley",
        ),
        CableComponent(
            name="Elbow roller groove (-Y)",
            obj_filename="mhp_arm_00_elbow_roller_v1.obj",
            link="lower_arm",
            pos_in_link=[6.10623e-16, 6.93889e-18, 0.0334],
            diameter_mm=78.8,
            color=_C_ROLLER,
            role="elbow_roller",
            cable="shared",
            note="Driven elbow roller, OD=85.44 mm (grove dia 78.8 mm)",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.037261, -0.0333478, 0.033401],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="upper",
            note="End point where cable leaves elbow roller and enters the lower arm mount Z=33.4 mm (upper groove)",
        )
    ]

    path = [
        CableComponent(
            name="Cable start point (spool anchor)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 0.020, 0.0645],  # on spool rim at +Y (anchor clamp, opp. to -Y pulleys)
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
            note="Cable clamp on spool rim — +Y side, angle=90°, r=20mm",
        ),
        CableComponent(
            name="Spool exit",
            obj_filename="ball_cable_spool_upper_arm_exit.obj",
            link="upper_arm",
            pos_in_link=[-0.0718787, -0.0159976, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B3",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0438212, -0.0387601, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B4",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0403245, -0.0400001, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B5",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.350166, -0.0400001, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B6",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.354258, -0.0381996, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B7",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.365068, -0.0264032, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B8",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.366505, -0.0221699, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Elbow roller enter",
            obj_filename="ball_cable_elbow_roller_enter.obj",
            link="upper_arm",
            pos_in_link=[0.360752, 0.0435847, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.037261, -0.0333478, 0.033401],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="upper",
            note="End point where cable leaves elbow roller and enters the lower arm mount Z=33.4 mm (upper groove)",
        ),
        # CableComponent(
        #     name="Lower arm exit",
        #     obj_filename="ball_cable_mount_lower_arm_exit.obj",
        #     link="lower_arm",
        #     pos_in_link=[0.0655164, -0.0660097, 0.0334],
        #     diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        #     note="Z=33.4 mm (upper groove)",
        # ),
    ]

    return CableRouteConfig(
        name="Upper Cable — elbow (-Y)",
        color=_C_LINE,
        physical=physical,
        path=path,
        branch_sign_seq = [-1, -1, -1, -1, +1],
        kind_seq        = ['external', 'external', 'external', 'internal', 'external'],
        elbow_roller_arc_dir   = 'cw',
        n_spool_turns          = 3,
    )

