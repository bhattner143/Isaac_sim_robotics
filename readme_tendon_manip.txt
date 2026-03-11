Now I have the full picture:

Kinematics at q1=q2=0:

Joint1 (link1_base) at world [0, 0, 48mm] — it rotates about Z axis (world Z in this frame)
Joint2 (link2_link1) at world [342.47, 0, 270.05mm] — it also rotates about Z axis
EE (tendon_ee) at world [532.47, 0, 321.55mm]
Key observations:

Z is constant! EE Z = 321.55mm at ALL joint angles in the FK sweep. That confirms Z doesn't change with q1 or q2. The robot is a horizontal 2-DOF arm with both joint axes vertical.

The Jacobian's XY rows reveal something strange: X shows zero sensitivity to both joints while Y responds to both, but when I trace through the FK sweep, X actually does change with q1 — it drops from 532.5mm at the home position to 500.4mm when q1 moves to 20°. This contradicts what the Jacobian is telling me.

Then it clicks: at q=0, the end effector sits along the X axis, and since the joint rotations are about the vertical Z axis, any motion from q1 is purely tangential — perpendicular to the X direction. So the instantaneous rate of change in X is zero at that configuration, which is what the Jacobian correctly captures. The Y direction gets the full tangential velocity contribution from the arm length.

The real problem is that this singular Jacobian — with zeros in the first row — means the inverse kinematics can't correct X-direction errors at the home position. The robot can only move in Y at that point, making the system rank-deficient and unable to invert the Jacobian for a full solution.

The solution is to seed the IK solver away from q=0, somewhere like q1=30° where the Jacobian becomes non-singular and can actually converge. I should also consider that since the end effector operates in the XY plane with Z fixed, I need a strategy that accounts for the workspace geometry and uses better initial guesses or a more robust IK method.

Now I'm checking the arm geometry more carefully — the joint positions seem unusual, with joint2 at [342.47, 0, 270mm] rather than where I'd expect it in a standard 2R configuration. Let me verify whether both joints rotate about the Z-axis and trace through where the end effector actually lands given the link lengths and joint positions. distance L2 from Joint2 in XY plane = (L1+L2, 0, H1) (at q=0)

But our data shows:

Joint1 at (0, 0, 48mm)
Joint2 at (342.47, 0, 270mm)
EE at (532.47, 0, 321.5mm)
So Z changes between joints! Joint1 is at Z=48mm, Joint2 at Z=270mm, EE at Z=321.5mm. But Z stays constant with q1,q2 changes in the FK sweep...

This must mean the joint axes are NOT aligned with world Z. They're aligned with some other axis that, when rotated, keeps Z constant in world frame.

Looking at the URDF joint data would tell me exactly how the rotations work. The joints might rotate about world Y, or about an axis that naturally preserves Z. But the FK sweep is clear—Z=321.5mm stays constant across all q1,q2 values, which means the joints effectively rotate about world Z even though the link structure itself is angled. When I look at specific joint angles like q1=20° giving EE=(500.4, 182.1, 321.5)mm and q2=20° giving EE=(521.0, 65.0, 321.5)mm, I can see the XY motion while Z remains locked.

The Jacobian singularity at q=0 makes sense now—at that exact configuration, the joints are aligned such that infinitesimal rotations only affect Y, not X. The real solution isn't to fix the Jacobian math but to avoid starting from that singular point entirely, either by choosing a better initial seed or deriving the analytical inverse kinematics for this 2R arm structure. For the horizontal 2R arm projected into XY plane:

At any (q1, q2): EE_X = L1cos(q1) + L2cos(q1+q2) (approximately) EE_Y = L1sin(q1) + L2sin(q1+q2)

But the actual arm is more complex with the Z-offset geometry. Let me not assume analytical formulas and instead fix the Jacobian approach.

The actual problem with the Jacobian IK:

The Jacobian at q=0 is:

J_XY = [[0,     0   ],
         [0.5325, 0.19]]
The damped pseudo-inverse J^T(JJ^T + λI)^{-1} tries to invert this. With det=0, the matrix is singular, but adding damping λ to the diagonal makes it invertible. The resulting correction dq = J^T (JJ^T + λI)^{-1} err will scale the error differently depending on λ—large errors in the first row get heavily damped while the second row gets less damping, which explains why the algorithm struggles to move the first joint effectively.

The fundamental issue is that at q=0, the Jacobian can only correct Y errors, not X errors. If the target is at a different X position than home, the algorithm needs to first rotate enough to get a non-singular Jacobian before it can properly correct the error.

I could fix this by starting with a small initial perturbation to q1, or better yet, compute an analytical seed using the geometric 2R arm formula. Let me check the forward kinematics data more carefully—at home the end effector is at (532.47, 0, 321.55) mm, and rotating q1 by 20° shifts it to (500.4, 182.1, 321.5), so the arm can definitely reach different X and Y positions. The trajectory I'm working with is centered around (0.40m, 0.10m), so I need to make sure the initial configuration can actually reach that workspace.

Looking at the current defaults and the error messages showing targets like (0.420, 0.100) and (0.380, 0.100), the arm should be able to reach this region—at q1=20°, q2=0 it gets to (0.5004, 0.1821), which is close. The issue is that the Jacobian-based IK is converging to a valid solution but then failing partway through.

The real problem is the singularity near q=0. I should seed the solver with something like 20° for q1 instead of starting at zero, since the target clearly requires significant rotation. Even better would be to compute an analytical initial guess—with L1 around 342mm horizontally and the EE needing to reach (400, 100), I can work out a better starting point geometrically. dy=0. ✓

Now I need to set up the inverse kinematics equations. For a target end-effector position (tx, ty) in the world frame, I'm solving the standard 2-link planar arm equations where the base is at the origin in the XY plane.

I can use the analytical solution for 2R inverse kinematics: first computing the elbow angle q2 from the law of cosines, then finding q1 from the target angle and the geometry. This gives me a direct analytical solution rather than relying on iterative Jacobian methods, which should be much faster and more stable.

Let me verify that the arm actually rotates purely in the horizontal plane—the Z coordinate stays constant at 48mm while only the XY coordinates change with joint angles.

Looking at the joint positions from the URDF, Joint1 is at the origin in XY but elevated at 48mm in Z, while Joint2 is offset significantly in both XY and Z. The fact that the forward kinematics sweep shows Z absolutely constant at 321.5mm suggests the joint axes are oriented such that their combined rotation keeps the end effector at a fixed height, even though the individual links aren't aligned with standard axes—the URDF likely specifies non-standard joint orientations through RPY values.

Since the end effector is constrained to move only in the horizontal plane at Z=321.5mm, I can treat this as a pure 2R planar inverse kinematics problem in the XY plane with link lengths of 342.47mm and 190mm. Now I'm implementing the analytical solution using the standard 2R IK equations.

Implementing the analytical solver... I'm replacing the iterative inverse kinematics with a direct analytical solution for this 2-link arm. From the probe data, I can confirm L1 = 342.47mm and L2 = 190mm, and I'm verifying that the end effector sits exactly at distance L2 from joint2 in the XY plane.

Evaluating


L1 = 342.47 mm (joint1→joint2, XY plane)
L2 = 190.00 mm (joint2→EE, XY plane)
EE Z = 321.5 mm constant (never changes with joints)
Jacobian at q=0 is singular (EE aligned with X-axis, so infinitesimal rotation only moves Y)