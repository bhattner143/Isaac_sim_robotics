# Cable Module Architecture Reference

This document provides a quick reference for the cable routing module architecture and file dependencies.

> **See Also**: [cable/ARCHITECTURE.md](../../cable/ARCHITECTURE.md) in the cable folder for full documentation.

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
         ┌────────────────────────┼────────────────────┐
         │                        │                    │
   test_cable_routing_viz.py  cable_with_exo_springs.py  camera/
         │                        │
         └────────────────────────┼─────────────────┐
                                  │                 │
                      cable_with_exo_springs_    scene.json
                      elbow_follow.py
```

---

## Data Flow

```
URDF → _parse_urdf_part_origins() → PulleyBase subclasses → CableRoute → CableRig
                                                                ↓
                    ┌───────────────────┬────────────────┐
                    │                   │                │
            test_cable_routing_viz.py  cable_with_exo_springs.py  drake_plant.py
            (Meshcat visualization)    (Exosuit springs)    (Isaac Sim FK)
```

---

## Quick Start

### View Cable Routing Interactively
```bash
cd cable/
python test_cable_routing_viz.py
# Open Meshcat at http://localhost:7000
# Type: q1 q2 [deg] to update joint angles
```

### Run Exosuit Springs Simulation
```bash
python cable_with_exo_springs.py
python cable_with_exo_springs.py --no-springs  # Disable springs
```

### Use in Isaac Sim
```python
from cable import DrakeCablePlant

dc = DrakeCablePlant(urdf_path, q1=0.0, q2=0.0, springs_enabled=True)
dc.update(q1_new, q2_new)
cable_world_points = dc.get_cable_world_points()
```

---

## Key Concepts

**PulleyBase**: Abstract geometry class for cable contact points  
**CableRoute**: Ordered sequence of waypoints forming one cable path  
**CableSpring**: Spring-damper model for exosuit co-contraction  
**CableRig**: Complete assembly managing all routes and physics  
**DrakeCablePlant**: Headless Drake wrapper for FK without rendering  

---

## Related Resources

- **Skill**: `exosuit-cable-routing` — Cable routing debugging and spring tuning
- **Instruction**: `exosuit-cables.instructions.md` — Method A vs Method B geometries
- **Instruction**: `sea-actuator.instructions.md` — Spring-damper and cable actuation
