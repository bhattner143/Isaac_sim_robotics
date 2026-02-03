# IK Methods Quick Reference

## Three IK Methods Available

### 1️⃣ Analytical IK (Original - Closed-Form)
```python
result = manipulator.inverse_kinematics(target_x, target_y, target_z)
```
- **Speed:** ⚡⚡⚡ Instant
- **Accuracy:** Perfect (if reachable)
- **Iterations:** None (0)
- **Works on:** Only XZ plane for 2-DOF arm
- **Returns:** Solution or `None` if unreachable

---

### 2️⃣ Jacobian Transpose IK
```python
result = manipulator.inverse_kinematics_jacobian(
    target_x, target_y, target_z,
    method="transpose",
    alpha=0.1
)
```
- **Speed:** ⚡⚡ ~10-30 iterations
- **Accuracy:** Good (but not exact)
- **Singularities:** Can diverge ❌
- **Works on:** Any workspace
- **Use when:** Quick approximate solution needed

---

### 3️⃣ Jacobian Damped Least-Squares IK ⭐ RECOMMENDED
```python
result = manipulator.inverse_kinematics_jacobian(
    target_x, target_y, target_z,
    method="damped_ls"
)
```
- **Speed:** ⚡⚡ ~5-20 iterations
- **Accuracy:** Good
- **Singularities:** Handles well ✅
- **Works on:** Any workspace
- **Use when:** Robust, production code

---

### 4️⃣ Hybrid IK (Smart Fallback)
```python
result = manipulator.inverse_kinematics_hybrid(
    target_x, target_y, target_z
)
```
- **Speed:** ⚡⚡⚡ (when analytical works) / ⚡⚡ (fallback)
- **Accuracy:** Best available
- **Reliability:** Highest ✅
- **Works on:** Everywhere
- **Use when:** You want best performance automatically

---

## Decision Tree

```
Want IK solution for target position?
│
├─ Know the arm is reachable & well-positioned?
│  └─ YES → Try ANALYTICAL (fastest)
│          Result good? → USE IT
│          Result null? → Fall back to JACOBIAN
│
├─ Need something that always works?
│  └─ YES → Use HYBRID (smart combination)
│
├─ Near singularities (fully extended/folded)?
│  └─ YES → Use DAMPED_LS (robust to singularities)
│          Or use HYBRID
│
└─ Want simple & fast approximation?
   └─ YES → Use TRANSPOSE (simple)
           or HYBRID (if you want fallback safety)
```

---

## Code Examples

### Quick Reach
```python
# Fast: try analytical first
pos = manipulator.inverse_kinematics(-2.5, 0.0, 1.0)
if pos:
    theta1, theta2 = pos
    arm.set_angles(theta1, theta2)
```

### Safe Reach (Production)
```python
# Robust: automatically best method
pos = manipulator.inverse_kinematics_hybrid(-2.5, 0.0, 1.0)
if pos:
    theta1, theta2 = pos
    arm.set_angles(theta1, theta2)
else:
    print("Target unreachable")
```

### Debug Near Singularity
```python
# Use damped LS for singularity-prone position
pos = manipulator.inverse_kinematics_jacobian(
    -2.5, 0.0, 2.0,  # Extended position
    method="damped_ls",
    max_iterations=100  # More iterations for convergence
)
```

### Compare All Methods
```python
target = (-2.5, 0.0, 1.0)

# Method 1: Analytical
r1 = manipulator.inverse_kinematics(*target)
print(f"1. Analytical: {r1}")

# Method 2: Jacobian Transpose  
r2 = manipulator.inverse_kinematics_jacobian(*target, method="transpose")
print(f"2. Transpose:  {r2}")

# Method 3: Jacobian Damped LS
r3 = manipulator.inverse_kinematics_jacobian(*target, method="damped_ls")
print(f"3. Damped LS:  {r3}")

# Method 4: Hybrid
r4 = manipulator.inverse_kinematics_hybrid(*target)
print(f"4. Hybrid:     {r4}")
```

---

## Performance Comparison

| Scenario | Best Method | Why |
|----------|------------|-----|
| Normal position | Analytical | Fastest, exact |
| Large workspace coverage | Hybrid | Covers all cases |
| Near singularity | Damped LS | Won't diverge |
| Trajectory tracking | Hybrid | Switches methods as needed |
| Mobile manipulator | Jacobian | Covers variable configurations |
| Real-time control | Hybrid | Fast when possible |
| Educational/learning | Transpose | Understand the math |

---

## Parameter Tuning

### Jacobian Transpose Method
```python
result = manipulator.inverse_kinematics_jacobian(
    *target,
    method="transpose",
    alpha=0.1,           # ← Adjust step size
    max_iterations=50,
    tolerance=1e-6
)
```

**Tuning alpha (step size):**
- `alpha=0.01`: Small steps, slower, more stable
- `alpha=0.1`: Medium steps, good balance (default)
- `alpha=0.5`: Large steps, faster but can diverge
- `alpha=1.0`: Maximum, risky near singularities

### Jacobian Damped LS Method
```python
result = manipulator.inverse_kinematics_jacobian(
    *target,
    method="damped_ls",
    max_iterations=50,    # ← More iterations helps
    tolerance=1e-6        # ← Tighter tolerance
)
```

**Tuning damping (in code, default λ=0.01):**
- `lambda=0.001`: Less damping, faster convergence
- `lambda=0.01`: Balanced (default)
- `lambda=0.1`: More damping, more robust

---

## Troubleshooting

### "IK returns None"
```python
# Check if target is reachable
target_dist = np.sqrt(x**2 + y**2)  # Distance from base
max_reach = L1 + L2  # Sum of link lengths
min_reach = abs(L1 - L2)  # Difference of link lengths

if min_reach <= target_dist <= max_reach:
    print("Target should be reachable")
else:
    print(f"Target out of workspace: {target_dist:.2f}m vs [{min_reach:.2f}, {max_reach:.2f}]m")
```

### "Iterative IK not converging"
```python
# Increase iterations and relax tolerance
result = manipulator.inverse_kinematics_jacobian(
    *target,
    method="damped_ls",
    max_iterations=100,    # Was 50
    tolerance=1e-4         # Was 1e-6 (more lenient)
)

# Or try with better initial guess
result = manipulator.inverse_kinematics_jacobian(
    *target,
    theta1_init=previous_theta1,  # Use last solution
    theta2_init=previous_theta2
)
```

### "Diverging near extended position"
```python
# Use damped LS instead of transpose
result = manipulator.inverse_kinematics_jacobian(
    *target,
    method="damped_ls"  # More robust
)
```

---

## Summary

| Method | Use Case | Key Point |
|--------|----------|-----------|
| **Analytical** | Reachable positions in XZ plane | Instant, exact |
| **Transpose** | Learning/prototyping | Simple math |
| **Damped LS** | Production, robustness needed | Handles singularities |
| **Hybrid** | Everything (recommended!) | Smart fallback |

**RECOMMENDATION:** Use `inverse_kinematics_hybrid()` for best overall performance! ⭐
