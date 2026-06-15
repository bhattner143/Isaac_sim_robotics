# MHP Cable Routing Visualization Guide

## 📊 What's New

The cable routing visualization now includes:

1. **3D Mesh Rendering** — Actual robot component geometries (shoulder spool, guide pulleys, elbow roller) rendered as semi-transparent 3D objects
2. **Triple-Panel Layout** — Three complementary views:
   - **3D perspective view** (left) — Shows cables and meshes in full 3D space with configurable viewing angles
   - **Top-down XY view** (center) — Plan view of the robot workspace (2D)
   - **3D XZ side view** (right) — Side elevation showing vertical cable routing (3D perspective)
3. **Interactive 3D Rotation** — When viewing interactively (not saving to file), all 3D plots respond to mouse input
4. **Customizable Viewing Angles** — Command-line control over main 3D view elevation and azimuth

## 🖱️ Interactive Viewing (How to Rotate)

### To View Interactively with Mouse Rotation:
```bash
# Default view (no file saving)
python test_mhp_cable_routing_viz.py

# With custom joint angles
python test_mhp_cable_routing_viz.py --q1 30 --q2 -20

# Specify custom view angle (then enable interactive)
python test_mhp_cable_routing_viz.py --elev 45 --azim 0
```

**Once the plot window opens:**
- **Left-click + drag** — Rotate 3D view
- **Right-click + drag (or scroll wheel)** — Zoom in/out
- **Middle-click + drag** — Pan

### To Save Static PNG and Then View Interactively:
```bash
python test_mhp_cable_routing_viz.py --q1 30 --q2 -20 --save output.png --show
```

## 📸 Saving with Specific View Angles

### Change the 3D viewing angle before saving:
```bash
# Higher elevation angle (more top-down view)
python test_mhp_cable_routing_viz.py --elev 60 --azim -30 --save fig_topdown.png

# Lower elevation angle (more side view)
python test_mhp_cable_routing_viz.py --elev 10 --azim -30 --save fig_sideview.png

# Different azimuth (rotate around Z axis)
python test_mhp_cable_routing_viz.py --elev 30 --azim 30 --save fig_rotated.png
```

## 🔧 Command-Line Arguments

```
--q1 ANGLE       Shoulder joint angle [degrees] (default: 0)
--q2 ANGLE       Elbow joint angle [degrees] (default: 0)
--elev ANGLE     3D view elevation angle [degrees] (default: 28)
--azim ANGLE     3D view azimuth angle [degrees] (default: -55)
--save PATH      Save figure to PNG/PDF file (suppresses interactive window)
--show           Show interactive window even when saving (creates both)
--help           Display all options
```

## 📝 Common Use Cases

### 1️⃣ Quick View at Neutral Pose
```bash
python test_mhp_cable_routing_viz.py
```
*Opens interactive window — drag to rotate*

### 2️⃣ Check Cable at Specific Joint Angle
```bash
python test_mhp_cable_routing_viz.py --q1 45 --q2 -30 --elev 35 --azim -45
```

### 3️⃣ Generate Publication-Quality Figure (Front View)
```bash
python test_mhp_cable_routing_viz.py --elev 0 --azim 0 --save cable_front.png
```

### 4️⃣ Generate Publication-Quality Figure (Top View)
```bash
python test_mhp_cable_routing_viz.py --elev 80 --azim 0 --save cable_top.png
```

### 5️⃣ Compare Multiple Poses Side-by-Side
```bash
# Generate several views
python test_mhp_cable_routing_viz.py --q1 0 --q2 0 --save pose1.png
python test_mhp_cable_routing_viz.py --q1 45 --q2 0 --save pose2.png
python test_mhp_cable_routing_viz.py --q1 -45 --q2 0 --save pose3.png
```

## 🎨 Visualization Features

### Mesh Transparency
- Meshes rendered with **α = 0.20** (semi-transparent)
- Allows cables to be clearly visible through component geometry
- Prevents meshes from obscuring cable paths

### Cable Colors
- **Orange** — Lower cable (shoulder actuation, +Y side)
- **Purple** — Upper cable (elbow actuation, −Y side)

### Reference Frames
- **Red** — X-axis
- **Green** — Y-axis  
- **Blue** — Z-axis
- Origin marked with black "×" at (0,0,0)

### Info Box
- Lower-right corner of side view shows:
  - Current joint angles (q1, q2)
  - Joint positions (J1, J2)
  - Elbow end-effector position

## 💡 Tips & Tricks

### Suggested View Angles for Specific Tasks
| Task | --elev | --azim | Purpose |
|------|--------|--------|---------|
| Default | 28 | -55 | Overall 3D perspective |
| Front view | 0 | 0 | Direct X-Y plane view |
| Top view | 85 | 0 | Bird's eye view |
| Side view (Y-Z) | 0 | -90 | Side elevation |
| Isometric | 45 | 45 | Balanced 3D view |
| Cable path detail | 20 | -45 | Close inspection |

### Batch Generation Script
Save this as `generate_views.sh`:
```bash
#!/bin/bash
SCRIPT="python test_mhp_cable_routing_viz.py"
OUTPUT_DIR="./cable_views"
mkdir -p $OUTPUT_DIR

$SCRIPT --q1 0 --q2 0 --elev 28 --azim -55 --save $OUTPUT_DIR/default.png
$SCRIPT --q1 0 --q2 0 --elev 0 --azim 0 --save $OUTPUT_DIR/front.png
$SCRIPT --q1 0 --q2 0 --elev 85 --azim 0 --save $OUTPUT_DIR/top.png
$SCRIPT --q1 30 --q2 -20 --elev 30 --azim -30 --save $OUTPUT_DIR/pose_manip.png

echo "✅ Generated $OUTPUT_DIR/*.png"
```

## 🐛 Troubleshooting

### Q: "Why can't I rotate the 3D plot in the saved PNG?"
**A:** PNG files are static images. Rotation is only possible in the **interactive matplotlib window**. Use one of these approaches:
- Run without `--save` to get interactive window
- Use `--save file.png --show` to save AND open interactive window afterward
- Use `--elev` and `--azim` to pre-set the view angle before saving

### Q: "The meshes aren't showing in the plot"
**A:** The script looks for OBJ files in the `assets/` subdirectory relative to the script location. Ensure:
- OBJ files exist in `model_using_onshape_to_robot/manipulator_hybrid_planar_fusion/assets/`
- File paths in `CableComponent` dataclass match actual filenames
- Check the terminal output for OBJ loading messages

### Q: "How do I see the cable from different angles?"
**A:** The recommended workflow is:
1. Run interactively: `python test_mhp_cable_routing_viz.py`
2. Use your mouse to rotate the 3D view in real-time
3. When satisfied with the angle, take a screenshot or use the `--elev` and `--azim` values you see in the window title

### Q: "The figure is too small/large"
**A:** Edit the figure size in the script. Look for:
```python
figsize=(28, 8.5)  # Change these dimensions (width, height in inches)
```

## 📐 Technical Details

### Three-Panel Layout

| Panel | Type | Default View | Use Case |
|-------|------|--------------|----------|
| **Left** | 3D | Configurable (elev=28°, azim=-55°) | Full spatial visualization; rotatable with mouse |
| **Center** | 2D XY | Top-down (bird's eye) | Horizontal workspace layout |
| **Right** | 3D XZ | Side elevation (elev=0°, azim=90°) | Vertical cable path; rotatable with mouse |

### Coordinate Frame
- **World origin** at the shoulder joint (J1)
- **X-axis** — Horizontal, pointing right
- **Y-axis** — Horizontal, pointing out of page (+Y is lower cable side, −Y is upper)
- **Z-axis** — Vertical, pointing up

### Cable Components
Each cable consists of:
- **Shoulder spool** (40 mm diameter) — Rotates with q1
- **Guide pulley 1 (GP1)** (10 mm diameter) — Fixed to upper_arm
- **Guide pulley 2 (GP2)** (10 mm diameter) — Fixed to upper_arm  
- **Guide pulley 3 (GP3)** (10 mm diameter) — Fixed to upper_arm
- **Elbow roller** (85.44 mm diameter) — Rotates with q2, shared by both cables

### Cable Path
- **11 waypoints** per cable marking the cable path through components
- Waypoints are rendered as ball markers at contact points
- Cable rendering connects waypoints through geometric arcs and straight segments

## 🔗 References

- Script location: `model_using_onshape_to_robot/manipulator_hybrid_planar_fusion/test_mhp_cable_routing_viz.py`
- Mesh assets: `model_using_onshape_to_robot/manipulator_hybrid_planar_fusion/assets/`
- MHP robot config: `manipulator_hybrid_planar_fusion.urdf`

---

**Last Updated:** 2024-12-19  
**Visualization Version:** 3.1 (Mesh rendering + interactive rotation)
