# Cart-Pole Implementation in ISAAC Sim

## Overview

Successfully implemented a cart-pole (inverted pendulum) system in ISAAC Sim that matches the PyDrake implementation.

## Files Created

1. **cart_pole_isaac_simple.py** - Simplified version using dynamic rigid bodies
   - Uses `DynamicCuboid` for cart
   - Uses `DynamicCylinder` for pole  
   - Implements the same hybrid controller as PyDrake version

2. **cart_pole_isaac_sim.py** - Advanced version with articulated joints (WIP)
   - Creates proper USD articulation with prismatic and revolute joints
   - More realistic joint-based physics

## Implementation Details

### Physical Parameters (matching PyDrake)
- Cart mass: 5.0 kg
- Pole mass: 1.0 kg
- Pole length: 2.0 m
- Cart friction: 1.0 N/(m/s)
- Gravity: 9.81 m/s²

### Controller
- **Swing-up mode**: Energy-based control to bring pole from hanging down to upright
- **LQR mode**: Linear-quadratic regulator to stabilize pole at upright position
- Automatic switching between modes based on angle and velocity thresholds

### LQR Gains
Computed using continuous-time algebraic Riccati equation (CARE):
- State weights: Q = I₄ (identity matrix)
- Control weight: R = 0.0001
- Linearization around upright equilibrium (θ = π)

## Running the Simulation

```bash
conda activate env_isaacsim
python cart_pole_isaac_simple.py
```

## Key Differences from PyDrake Version

| Feature | PyDrake | ISAAC Sim |
|---------|---------|-----------|
| Physics Engine | Custom dynamics | PhysX (NVIDIA) |
| Visualization | MeshCat | Real-time 3D viewport |
| Joint Type | Articulation with DOF | Rigid bodies with forces |
| State Access | Direct from system | From rigid body poses |
| Control Application | Joint efforts | External forces |

## Next Steps for Full Implementation

To create a fully articulated cart-pole:

1. Use USD Physics `PrismaticJoint` for cart sliding motion
2. Use USD Physics `RevoluteJoint` for pole rotation
3. Apply forces through joint drives (`DriveAPI`)
4. Read joint positions and velocities directly

Reference: `/home/dipankar/isaacsim/standalone_examples/api/isaacsim.core.api/`

## Performance

- Simulation runs at real-time or faster
- GPU-accelerated physics (PhysX on CUDA)
- Can visualize multiple camera angles simultaneously
- Compatible with ROS2 for robot integration

## Notes

The simplified version demonstrates the control algorithm working in ISAAC Sim. 
For production use, implement proper articulated joints for more accurate physics 
and easier integration with robot control frameworks.
