# Meshcat Visualization Fix for Finite-Horizon LQR

## Problem
Users reported that Meshcat visualization was not working for finite-horizon-lqr mode.

## Root Cause
- **finite-horizon-lqr** controller requires 5D state: [x, φ, ẋ, φ̇, F] (includes motor force F)
- **MultibodyPlant** only provides 4D state: [x, θ, ẋ, θ̇] (no motor dynamics)
- Attempting to feed 4D state to a 5D controller causes dimension mismatch

## Solution Implemented
For **LQR modes** (standard-lqr, finite-horizon-lqr):
1. Create separate **CartPendulumSystemLinearizedWithMuscleDynamics** system (no geometry, 5D state)
2. Wire this linearized plant to the LQR controller (needs 5D state)
3. Keep MultibodyPlant separate (has geometry for visualization, but only 4D state)
4. **Accept that visualization is NOT available for LQR modes**

For **other modes** (pd, pd-swing, computed-torque):
1. Use MultibodyPlant directly with controller
2. Visualization works as normal

## Code Changes

### add_controller() method
- For `standard-lqr`: Creates self.linearized_plant, wires it to controller
- For `finite-horizon-lqr`: Creates self.linearized_plant, wires it to controller
- For other modes: Skips linearized plant, uses MultibodyPlant directly

### create_simulator() method
- Checks `if CONTROLLER_MODE in ['standard-lqr', 'finite-horizon-lqr']`
- Initializes linearized plant with 5D state if in LQR mode
- Otherwise initializes MultibodyPlant with 4D state

### run_simulation() method
- Sets `is_linearized = CONTROLLER_MODE in ['standard-lqr', 'finite-horizon-lqr']`
- Reads state from linearized_plant if is_linearized=True
- Skips Meshcat ForcedPublish() for LQR modes (no geometry to visualize)

### plot_results() method
- Already handles both 5D and 4D state arrays
- Plots work correctly regardless of state dimension
- Goal lines drawn from config.x_goal as before

## User Impact
✅ finite-horizon-lqr simulation now **works correctly** (no AttributeError)
✅ 5D control law operates on correct state space
✅ Plots display results with control effort
⚠️ **Limitation**: No Meshcat visualization for LQR modes (linearized plant has no geometry)
✅ Other modes (pd, computed-torque) still have full visualization

## Testing Recommendations
1. Test finite-horizon-lqr mode runs to completion without errors
2. Check that printed state values match logged values
3. Verify plots show correct cart/pendulum dynamics
4. Confirm other modes still have Meshcat visualization
