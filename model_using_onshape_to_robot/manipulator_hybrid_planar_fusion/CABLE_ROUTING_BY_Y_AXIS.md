# Cable Routing

## Joint Frames

| Joint | Link | Location (xyz) | Axis | Purpose |
|---|---|---|---|---|
| jt_upper_base | base_link → upper_arm | 0.045 -1.90933e-09 0.1249 | Z-axis (revolute) | Shoulder joint (cable lower actuation) |
| jt_lower_upper | upper_arm → lower_arm | 0.4 0.0470711 0.0141 | Z-axis (revolute) | Elbow joint (cable upper actuation) |

---

## Link Properties — Mass & Inertia

| Link | Mass (kg) | CoM (xyz) | Ixx | Iyy | Izz | Note |
|---|---|---|---|---|---|---|
| base_link_aka_shoulder_transmission | 2.0 | (0.08, 0, 0.13) | 0.0170 | 0.0170 | 0.0050 | Motor + transmission assembly |
| upper_arm | 0.8 | (0.325, 0.015, 0.065) | 0.0006 | 0.0137 | 0.0137 | Arm structure with pulleys & bearings |
| lower_arm | 0.4 | (0.125, 0, 0.01) | 0.0004 | 0.0021 | 0.0023 | End-effector arm with roller |

**Total system mass**: 3.2 kg

---

## Base link (World frame)

| Obj Filename | Location (xyz) | Note |
|---|---|---|
| ball_cable_pulleys_upper_arm.obj | -0.045 1.90933e-09 -0.1249 | World frame, rigid joint with ground, Z = -124.9 mm |


## Cable Routing, Lower cable complete, Mostly on +Y

### Lower Cable Path — Physical Components (Pulleys & Rollers)

**Cable routing sequence through mechanical components:**

| # | Component | Location (xyz) | Diameter | Note |
|---|---|---|---|---|
| 0 | mhp_arm_00_elbow_spool_v2 | 0.225 5.79026e-24 0.1268 | 40 mm | Cable spool anchor (source) |
| 1 | steel_v_groove_guide_pulley | -0.0409243 0.03445 0.0325 | 10 mm | First guide pulley |
| 2 | steel_v_groove_guide_pulley | 0.33 0.03445 0.0324 | 10 mm | Mid-span guide pulley |
| 3 | steel_v_groove_guide_pulley | 0.353129 0.0165902 0.0324 | 10 mm | Pre-elbow guide pulley |
| 4 | mhp_arm_00_elbow_roller_v1 | 6.10623e-16 6.93889e-18 0.0259 | 78.8 mm OD / 85.44 mm | Elbow roller (end) |

**Cable path**: Spool → Pulley 1 → Pulley 2 → Pulley 3 → Elbow Roller

---

### Upper Arm +Y (Y ≥ 0) — Cable routing sequence

| # | Obj Filename | Location (xyz) | Color |
|---|---|---|---|
| 1 | ball_cable_spool_upper_arm_start.obj | -0.0795 1.76602e-13 0.0155 | Sky Blue |
| 2 | ball_cable_spool_upper_arm_exit.obj | -0.0718787 0.0159976 0.0324999 | Sky Blue |
| 3 | ball_cable_pulleys_upper_arm.obj | -0.0438212 0.0387601 0.0324999 | Sky Blue |
| 4 | ball_cable_pulleys_upper_arm.obj | -0.0403245 0.0400001 0.0324999 | Sky Blue |
| 5 | ball_cable_pulleys_upper_arm.obj | 0.3306 0.0400001 0.0324999 | Sky Blue |
| 6 | ball_cable_pulleys_upper_arm.obj | 0.335406 0.0372251 0.0324999 | Sky Blue |
| 7 | ball_cable_pulleys_upper_arm.obj | 0.348922 0.0138153 0.0324999 | Sky Blue |
| 8 | ball_cable_pulleys_upper_arm.obj | 0.353245 0.0110614 0.0324999 | Sky Blue |
| 9 | ball_cable_elbow_roller_enter.obj | 0.397114 0.00722337 0.0324999 | Sky Blue |

**Total in sequence: 9 objects**



### Lower Arm, Lower Z Level (Z ≈ 0.0184 mm) — Cable routing by increasing X

| # | Obj Filename | Location (xyz) | Color | Note |
|---|---|---|---|---|
| 10 | ball_cable_mount_lower_arm_enter.obj | 0.0333444 -0.0372498 0.0183897 | Orange | Entry point, lower Z level |
| 11 | ball_cable_mount_lower_arm_exit.obj | 0.0613398 -0.012071 0.0184 | Sky Blue | Exit point, lower Z level |

**Total at lower Z: 2 objects** (Z = 18.4 mm, fixed)

---


## Cable Routing, Upper cable complete, Mostly on -Y

### Upper Cable Path — Physical Components (Pulleys & Rollers)

**Cable routing sequence through mechanical components:**

| # | Component | Location (xyz) | Diameter | Note |
|---|---|---|---|---|
| 0 | small_ball.obj (spool anchor) | -0.0795 1.76602e-13 0.0645 | 10 mm | Cable spool anchor (source) |
| 1 | steel_v_groove_guide_pulley | -0.0409243 -0.03445 0.0475 | 10 mm | First guide pulley |
| 2 | steel_v_groove_guide_pulley | 0.349567 -0.03445 0.0476 | 10 mm | Mid-span guide pulley |
| 3 | steel_v_groove_guide_pulley | 0.360376 -0.0226536 0.0476 | 10 mm | Pre-elbow guide pulley |
| 4 | mhp_arm_00_elbow_roller_v1 | 6.10623e-16 6.93889e-18 0.0259 | 78.8 mm OD / 85.44 mm | Elbow roller (end) |

**Cable path**: Spool → Pulley 1 → Pulley 2 → Pulley 3 → Elbow Roller

---

### Upper Arm -Y (Y < 0) — Cable routing sequence

| # | Obj Filename | Location (xyz) | Color | Note |
|---|---|---|---|---|
| 1 | ball_cable_spool_upper_arm_start.obj | -0.0795 1.76602e-13 0.0645 | Sky Blue | Top spool anchor |
| 2 | ball_cable_spool_upper_arm_exit.obj | -0.0718787 -0.0159976 0.0475001 | Sky Blue | Spool exit |
| 3 | ball_cable_pulleys_upper_arm.obj | -0.0438212 -0.0387601 0.0475001 | Sky Blue | |
| 4 | ball_cable_pulleys_upper_arm.obj | -0.0403245 -0.0400001 0.0475001 | Sky Blue | |
| 5 | ball_cable_pulleys_upper_arm.obj | 0.350166 -0.0400001 0.0475001 | Sky Blue | |
| 6 | ball_cable_pulleys_upper_arm.obj | 0.354258 -0.0381996 0.0475001 | Sky Blue | |
| 7 | ball_cable_pulleys_upper_arm.obj | 0.365068 -0.0264032 0.0475001 | Sky Blue | |
| 8 | ball_cable_pulleys_upper_arm.obj | 0.366505 -0.0221699 0.0475001 | Sky Blue | |
| 9 | ball_cable_elbow_roller_enter.obj| 0.360752 0.0435847 0.0475001  | 

**Total in sequence: 9 objects**


### Lower Arm, Higher Z Level (Z ≈ 0.0334 mm) — Cable routing by increasing X

| # | Obj Filename | Location (xyz) | Color | Note |
|---|---|---|---|---|
| 10 | ball_cable_mount_lower_arm_enter.obj | 0.037261 -0.0333478 0.033401 | Orange | Entry point, higher Z level |
| 11 | ball_cable_mount_lower_arm_exit.obj | 0.0655164 -0.0660097 0.0334 | Sky Blue | Exit point, higher Z level |

**Total at higher Z: 2 objects** (Z = 33.4 mm, fixed)


---