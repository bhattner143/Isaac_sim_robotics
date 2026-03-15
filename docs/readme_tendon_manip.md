# Cup Manipulator Tendon — Kinematics & IK Notes

## Kinematics at q1 = q2 = 0°

| Frame | Position |
|---|---|
| Joint 1 (`link1_base`) | (0, 0, 48 mm) — rotates about Z |
| Joint 2 (`link2_link1`) | (342.47, 0, 270.05 mm) — rotates about Z |
| EE (`tendon_ee`) | (532.47, 0, 321.55 mm) |

**Z is constant.** EE Z = 321.55 mm across **all** joint angles in the FK sweep.
This confirms the robot is a horizontal 2-DOF arm with both joint axes effectively vertical in world frame.

---

## Why Z Stays Constant

The individual link offsets in Z are non-zero (Joint1 at 48 mm, Joint2 at 270 mm)
but the URDF joint axes are oriented so their combined effect keeps Z fixed:

- At q1 = 20°: EE = (500.4, 182.1, **321.5**) mm
- At q2 = 20°: EE = (521.0, 65.0, **321.5**) mm

Z never changes. This means the problem reduces to pure **2R planar IK in the XY plane**.

### Effective link lengths (XY plane only)

- L1 = 342.47 mm (Joint1 → Joint2)
- L2 = 190.00 mm (Joint2 → EE)
- Z_EE = 321.5 mm (constant)

---

## Jacobian Singularity at q = 0

At the home configuration the XY Jacobian is:

```
J_XY = [[0,       0   ],
         [0.5325,  0.19]]
```

**Why the zero first row?** At q = 0°, the EE sits directly on the X-axis.
Rotations about the vertical joint axes are purely tangential — perpendicular to X —
so the instantaneous rate of change of X is zero. The Jacobian correctly captures this;
it is not a bug.

**Consequence:** The damped pseudo-inverse `dq = J^T (J J^T + λI)^{-1} Δe` can only
correct **Y errors** at this configuration. X-direction errors get heavily damped and
the solver stalls.

---

## Iterative IK Failure — Root Cause

The Jacobian IK fails when seeded at q = (0, 0) because:

1. The Jacobian is rank-deficient at q = 0. Adding damping λ makes it formally invertible
   but the X-direction correction is negligible.
2. The solver needs to first rotate enough to make the Jacobian non-singular before it can
   steer the EE toward an X-offset target.
3. Typical targets (e.g. (0.40, 0.10) m) require q1 ≈ 20–30°, so starting at q = 0
   guarantees poor convergence.

---

## Fix: Analytical IK

Replace the iterative solver with the **closed-form 2R solution**:

```
q2 = ± arccos( (tx² + ty² − L1² − L2²) / (2 L1 L2) )

q1 = atan2(ty, tx) − atan2(L2·sin(q2), L1 + L2·cos(q2))
```

where (tx, ty) is the target in world XY.

This gives an exact solution in one shot — no linearisation or convergence issues.
Two solutions exist (elbow-up / elbow-down); pick based on joint limits or continuity
with the previous configuration.

---

## Workspace Summary

The arm operates in the horizontal plane at Z = 321.5 mm.

| Property | Value |
|---|---|
| Outer reach | L1 + L2 = 532.47 mm |
| Inner dead zone | \|L1 − L2\| = 152.47 mm |
| Typical test region | centred ~(0.40, 0.10) m |

At q1 = 20°, q2 = 0°: EE ≈ (0.500, 0.182) m — confirms the target region is accessible.
