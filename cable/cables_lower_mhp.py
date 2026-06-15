"""Lower (shoulder +Y) cable route configuration for MHP."""
from __future__ import annotations

from cable.cable_config_mhp import CableComponent, CableRouteConfig

def build_lower_cable_config() -> CableRouteConfig:
    """Config for the lower (+Y) half of the elbow antagonistic cable pair.

    Routed from the shoulder spool groove (+Y side) to the elbow roller.
    When this side is taut, the upper (−Y) cable is slack.
    """
    _C_LINE  = "#E37629"   # cable line — orange
    _C_SPOOL = "#D45F00"   # drive spool
    _C_GUIDE = "#333333"   # guide pulleys
    _C_ROLLER= "#888888"   # elbow roller
    _C_BALL  = "#87CEEB"   # sky-blue ball markers
    _C_LO_ARM= "#FFA500"   # lower-arm entry/exit

    # ─────────────────────────────────────────────────────────────────────────────
    # SPOOL POSITION CALCULATION (in upper_arm frame)
    #
    # Two different spool positions exist in the system:
    #
    # (1) URDF PHYSICAL POSITION [0.225, 0, 0.1268] (m)
    #     → From: manipulator_hybrid_planar_fusion_obj.urdf line ~391
    #     → Origin: Onshape CAD model geometry (mhp_arm_00_elbow_spool_v2.obj location)
    #     → Purpose: 3D visualization in Meshcat/Drake
    #     → Represents: Actual spool drum location inside the shoulder transmission housing
    #     → Note: High Z (0.1268 m) because spool sits near top of housing
    #
    # (2) CABLE ROUTING POSITION [-0.0795, 0, 0.0155] (m) ← USED HERE
    #
    #     EXTRACTION PROCESS (NOT CALCULATED):
    #     ─────────────────────────────────────
    #     Source: URDF file line 664, ball_cable_spool_upper_arm_start marker
    #
    #     URDF snippet:
    #       <!-- Part ball_cable_spool_upper_arm_start_2 -->
    #       <visual>
    #         <origin xyz="-0.0795 1.76602e-13 0.0155" rpy="..."/>
    #         <geometry>
    #           <mesh filename="package://assets/ball_cable_spool_upper_arm_start.obj"/>
    #         </geometry>
    #       </visual>
    #
    #     Value breakdown:
    #       X = -0.0795 m  ← backward from shoulder joint (negative direction)
    #       Y = 1.76602e-13 m ≈ 0  ← numerical precision artifact (Onshape export)
    #       Z = 0.0155 m   ← at lower arm channel height (18.4 mm groove level)
    #
    #     This position marks where the cable EXITS the housing into the arm channel.
    #     It was placed by the Onshape CAD designer as a reference point for cable routing.
    #
    # Why different positions?
    #   The physical spool (0.225, 0, 0.1268) is INSIDE the transmission housing.
    #   For cable routing visualization, we show the CABLE EXIT POINT (cable emerges from
    #   housing at -0.0795, 0, 0.0155) where the cable actually starts routing through
    #   pulleys. This makes the 2D visualization show the cable path from its actual
    #   routing starting point, not from the deep spool drum location.
    #
    # Coordinate system (upper_arm frame):
    #   Origin: jt_upper_base joint center
    #   X: along upper arm length (+ toward J2)
    #   Y: perpendicular (+ in original arm orientation)
    #   Z: vertical (+ upward)
    # ─────────────────────────────────────────────────────────────────────────────

    physical = [
        CableComponent(
            name="Cable anchor (clamp on spool rim)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, -0.020, 0.0155],  # on spool rim at -Y, r=20mm from centre
            diameter_mm=0.0,
            color=_C_SPOOL,
            role="cable_anchor",
            cable="lower",
            note="Physical clamp point where lower cable is fixed to spool drum, -Y side",
        ),
        CableComponent(
            name="Shoulder drive spool",
            obj_filename="mhp_arm_00_elbow_spool_v2__2.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 1.76602e-13, 0.0155],
            diameter_mm=40.0,
            color=_C_SPOOL,
            role="spool",
            cable="lower",
            note="Drive spool drum — lower cable groove, Z=15.5 mm",
            visual_pos_in_link=[0.225, 0.0, 0.1268],
            visual_rpy=[0.0, 0.0, 3.14159],
        ),
        CableComponent(
            name="Guide pulley 1 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[-0.0409243, 0.03445, 0.0325],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="First guide pulley, +Y side",
        ),
        CableComponent(
            name="Guide pulley 2 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.33, 0.03445, 0.0324],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="Mid-span guide pulley, +Y side",
        ),
        CableComponent(
            name="Guide pulley 3 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.353129, 0.0165902, 0.0324],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="Pre-elbow guide pulley",
        ),
        CableComponent(
            name="Elbow roller groove (+Y)",
            obj_filename="mhp_arm_00_elbow_roller_v1.obj",
            link="lower_arm",
            pos_in_link=[6.10623e-16, 6.93889e-18, 0.0184],#[6.10623e-16, 6.93889e-18, 0.0259],
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
            pos_in_link=[0.0333444, -0.0372498, 0.0183897],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="lower",
            note="End point where cable enters the lower arm mount Z=18.4 mm (lower groove)",
        ),
    ]

    path = [
        CableComponent(
            name="Cable start point (spool start)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, -0.020, 0.0155],  # on spool rim at -Y (anchor clamp, opp. to +Y pulleys)
            diameter_mm=5.0, color=_C_SPOOL, role="ball_marker", cable="lower",
            note="Cable clamp on spool rim — -Y side, angle=270°, r=20mm",
        ),
        CableComponent(
            name="Spool exit",
            obj_filename="ball_cable_spool_upper_arm_exit.obj",
            link="upper_arm",
            pos_in_link=[-0.0718787, 0.0159976, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A3",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0438212, 0.0387601, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A4",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0403245, 0.0400001, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A5",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.3306, 0.0400001, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A6",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.335406, 0.0372251, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A7",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.348922, 0.0138153, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A8",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.353245, 0.0110614, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Elbow roller enter",
            obj_filename="ball_cable_elbow_roller_enter.obj",
            link="upper_arm",
            pos_in_link=[0.397114, 0.00722337, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.0333444, -0.0372498, 0.0183897],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="lower",
            note="Z=18.4 mm (lower groove)",
        ),
        # CableComponent(
        #     name="Lower arm exit",
        #     obj_filename="ball_cable_mount_lower_arm_exit.obj",
        #     link="lower_arm",
        #     pos_in_link=[0.0613398, -0.012071, 0.0184],
        #     diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        #     note="Z=18.4 mm (lower groove)",
        # ),
    ]

    return CableRouteConfig(
        name="Lower Cable — shoulder (+Y)",
        color=_C_LINE,
        physical=physical,
        path=path,
        branch_sign_seq=[+1, +1, +1, -1, -1],
        kind_seq=['external', 'external', 'internal', 'external', 'external'],
        elbow_roller_arc_dir='ccw',
        n_spool_turns=3,
    )

    

