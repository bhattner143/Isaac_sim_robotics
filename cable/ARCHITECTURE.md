# Cable Module Architecture

## Module Overview

The `cable/` module provides cable routing infrastructure for tendon-driven manipulators. It handles:
- **URDF parsing** for pulley positions from CAD exports
- **Geometry classes** for pulleys, idlers, and cable routing
- **Spring physics** for exosuit co-contraction
- **Headless Drake FK** for cable tangent computation without rendering

---

## File Dependencies & Roles

| File | Purpose | Depends On | Used By |
|------|---------|-----------|---------|
| **pulley.py** | URDF parser + pulley geometry classes | `numpy`, `pathlib`, `re` | routing.py, cable_with_exo_springs.py |
| **routing.py** | Cable route assembly & spring physics | `pulley.py`, `numpy` | test_cable_routing_viz.py, cable_with_exo_springs.py |
| **drake_plant.py** | Headless Drake FK (no rendering) | `robots.cup_manipulator_cable`, `routing.py` | Isaac Sim scripts |
| **__init__.py** | Public API re-exports | All three modules above | External scripts |
| **test_cable_routing_viz.py** | 📊 Interactive visualization tool | `routing.py`, `pulley.py` | Standalone (run directly) |
| **cable_with_exo_springs.py** | 🦾 Exosuit springs routing (base variant) | `routing.py`, `pulley.py` | Research experiments |
| **cable_with_exo_springs_elbow_follow.py** | 🦾 Exosuit springs (elbow-follow variant) | `cable_with_exo_springs.py` | Alternative routing mode |
| **scene.json** | 3D scene configuration | N/A | Isaac Sim rendering |

---

## Module Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    cable/ MODULE ARCHITECTURE                     │
└──────────────────────────────────────────────────────────────────┘

                            __init__.py
                                 │
                    (re-exports all public APIs)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
    pulley.py            routing.py               drake_plant.py
    (Pulleys)         (Cable Routes)           (Headless Drake FK)
         │                       │                       │
    PulleyBase           CableRoute                DrakeCablePlant
    ├─ CableStartPointR/L    ├─ FixedBodyPoint
    ├─ DrivePulley           ├─ CableSpring
    ├─ IdlerL/R              ├─ CableRig  ◄────────────────┐
    ├─ BigPulley             └─ spring_zigzag_points()    │
    └─ CableEndPointL/R                                    │
         │                                                 │
         └────────────────────────┬──────────────────────┘
                                  │
                          (used in applications)
                                  │
         ┌────────────────────────┼────────────────────┐
         │                        │                    │
   test_cable_routing_viz.py  cable_with_exo_springs.py  camera/
   (Interactive Meshcat viz)  (Exosuit springs ext.)    (Isaac Sim)
         │                        │
         └────────────────────────┼─────────────────┐
                                  │                 │
                      cable_with_exo_springs_    scene.json
                      elbow_follow.py            (3D scene config)
                      (Variant: elbow follow)
```

---

## Key Data Flow

```
URDF ──┐
       └─→ _parse_urdf_part_origins()  (pulley.py)
               │
               ├─→ PulleyBase subclasses ──┐
               │                           │
               │                      CableRig
               │   (routing.py)            │
               │                           │
               └─→ CableRoute ◄────────────┘
                       │
                       ├─→ test_cable_routing_viz.py (visualization)
                       │
                       ├─→ cable_with_exo_springs.py (exo routing)
                       │
                       └─→ drake_plant.py (FK computation for Isaac Sim)
```

---

## Class Hierarchy

### PulleyBase (pulley.py)
Base class for all cable-routing geometric primitives:

```python
PulleyBase (abstract)
├── CableStartPointR      # Right cable entry point (on link1)
├── CableStartPointL      # Left cable entry point (on link1)
├── DrivePulley           # Drive pulley (on link1)
├── IdlerL                # Left idler bearing (on link1)
├── IdlerR                # Right idler bearing (on link1)
├── BigPulley             # Driven pulley (on link2)
├── CableEndPointL        # Left cable exit point (on link2)
└── CableEndPointR        # Right cable exit point (on link2)
```

### CableRig (routing.py)
Manages the complete cable routing assembly:
- Stores `CableRoute` objects (one per cable)
- Computes tangent points via `compute_tangents()`
- Applies spring forces and co-contraction dynamics

---

## Usage Examples

### Basic Cable Visualization
```python
from cable import DrakeCablePlant

# Headless Drake plant for cable FK
dc = DrakeCablePlant("path/to/urdf.urdf", q1=0.0, q2=0.0, springs_enabled=True)
dc.update(q1_new, q2_new)  # Update joint angles
for route, world_pts in dc.get_cable_world_points():
    print(f"Route: {route.label}")
    print(f"Points: {world_pts}")  # (N, 3) array
```

### Interactive Testing
```bash
# Run interactive Meshcat visualization
python cable/test_cable_routing_viz.py

# Run exosuit springs variant
python cable/cable_with_exo_springs.py --no-springs
```

### Integration with Isaac Sim
```python
from cable import DrakeCablePlant

# Compute cable waypoints while Isaac Sim runs physics
dc = DrakeCablePlant(drake_urdf, springs_enabled=True)

# In Isaac Sim loop:
for t in range(num_steps):
    q1, q2 = isaac_sim_read_joint_angles()
    dc.update(q1, q2)
    cable_pts = dc.get_cable_world_points()
    # Render cable_pts as USD prims in Isaac Sim
```

---

## Common Tasks

### Add a New Pulley Type
1. Subclass `PulleyBase` in `pulley.py`
2. Implement `calc_position()` method
3. Register in `CableRig` constructor

### Modify Cable Routing
1. Edit waypoint list in `routing.py`'s `CableRoute` definition
2. Adjust spring parameters in `CableSpring` dataclass
3. Test with `test_cable_routing_viz.py`

### Debug Cable Collisions
1. Run `test_cable_routing_viz.py`
2. Inspect 3D plot at http://localhost:7000
3. Adjust pulley positions in URDF or `CableRig`

---

## Related Skills & Instructions

- **Skill: exosuit-cable-routing** — Detailed cable routing debugging and spring co-contraction tuning
- **Instruction: exosuit-cables.instructions.md** — Method A (offset pulleys) vs Method B (centred elbow pulley)
- **Instruction: sea-actuator.instructions.md** — Spring-damper models and cable actuation physics
