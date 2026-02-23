import numpy as np

# =====================
# Meshcat Camera Utilities
# =====================
def set_meshcat_camera_view(meshcat, distance=3.0, target=np.zeros(3)):
    """
    Set Meshcat camera to top view (looking down +Z at the target).
    Args:
        meshcat: Meshcat instance
        distance: Distance from target (float)
        target: 3D numpy array (default: [0,0,0])
    """
    camera_pos = target + np.array([0, 0, distance])
    meshcat.SetCameraPose(camera_pos, target)

def set_meshcat_camera_spherical(meshcat, azimuth_deg, elevation_deg, distance=3.0, target=np.zeros(3)):
    """
    Set Meshcat camera using spherical coordinates (azimuth, elevation in degrees).
    Args:
        meshcat: Meshcat instance
        azimuth_deg: Azimuth angle in degrees (0 = +X, 90 = +Y)
        elevation_deg: Elevation angle in degrees (0 = XY plane, 90 = +Z)
        distance: Distance from target
        target: 3D numpy array (default: [0,0,0])
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = distance * np.cos(el) * np.cos(az)
    y = distance * np.cos(el) * np.sin(az)
    z = distance * np.sin(el)
    camera_pos = target + np.array([x, y, z])
    meshcat.SetCameraPose(camera_pos, target)
"""
Visualization utilities for the 2D Cart-Pendulum + Manipulator simulation.

Provides:
  - visualize_plant_meshcat   : Quick Meshcat preview of a plant state
  - add_frames_to_meshcat     : RGB coordinate-frame triads in Meshcat
  - plot_frames_top_view      : Matplotlib top/side-view frame plots
  - plot_lqr_manip_ee_traj_track_results : Full post-simulation results plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    MeshcatVisualizer,
    RigidTransform,
)


# ============================================================================
# PREVIEW VISUALIZATION
# ============================================================================

def visualize_plant_meshcat(
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    meshcat,
    positions_dict: dict = None,
    message: str = "Visualizing plant state in Meshcat..."
):
    """
    Visualize the plant configuration in Meshcat.
    
    Args:
        plant: Finalized MultibodyPlant
        scene_graph: SceneGraph for visualization
        meshcat: Meshcat instance for visualization
        positions_dict: Optional dict mapping ModelInstance to position arrays
                       e.g., {manipulator.model_instance: initial_q, cart_model: cart_pos}
        message: Custom message to display (default: "Visualizing plant state in Meshcat...")
    
    Returns:
        tuple: (preview_diagram, preview_context) for reuse
    
    Example:
        # Preview with default positions (zeros)
        diagram, ctx = visualize_plant_meshcat(plant, scene_graph, meshcat)
        
        # Preview with specific positions
        diagram, ctx = visualize_plant_meshcat(plant, scene_graph, meshcat, 
                     positions_dict={
                         manipulator.model_instance: initial_q,
                         cart_model: cart_init_pos
                     },
                     message="Configured initial state")
    """
    print(colored(f"\n📸 {message}", "cyan"))
    
    # Create a simple diagram just for visualization
    preview_builder = DiagramBuilder()
    preview_builder.AddSystem(plant)
    preview_builder.AddSystem(scene_graph)
    
    # Connect geometry ports
    preview_builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    preview_builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Add visualizer and build
    preview_visualizer = MeshcatVisualizer.AddToBuilder(preview_builder, scene_graph, meshcat)
    preview_diagram = preview_builder.Build()
    
    # Create default context
    preview_context = preview_diagram.CreateDefaultContext()
    preview_plant_context = plant.GetMyContextFromRoot(preview_context)
    
    # Set positions if provided
    if positions_dict:
        for model_instance, positions in positions_dict.items():
            plant.SetPositions(preview_plant_context, model_instance, positions)
    
    # Force visualization update
    preview_diagram.ForcedPublish(preview_context)
    
    print(colored(f"✓ State visualized at: {meshcat.web_url()}", "green"))
    
    return preview_diagram, preview_context


def add_frames_to_meshcat(meshcat, plant, context, manipulator=None, cart_model=None):
    """
    Add coordinate frame visualizations to Meshcat with visible XYZ triads.
    Creates RGB cylinders showing X (red), Y (green), Z (blue) axes.
    
    Args:
        meshcat: Meshcat instance
        plant: MultibodyPlant
        context: Plant context with current state
        manipulator: CupManipulator instance (optional)
        cart_model: Cart model instance (optional)
    
    Returns:
        frame_list: List of (frame_name, frame, length) tuples for updating
    """
    from pydrake.all import RigidTransform, RotationMatrix, Cylinder, Rgba
    from pydrake.multibody.tree import FrameIndex
    
    # Helper function to create a coordinate frame triad
    def add_frame_triad(meshcat, path, length=0.1, opacity=1.0):
        """Add XYZ coordinate frame to Meshcat with RGB colors.
        
        Args:
            meshcat: Meshcat instance
            path: Path for the frame (e.g., "/Frames/World")
            length: Length of axes
            opacity: Transparency (0=transparent, 1=opaque)
        """
        # Standard RGB colors
        x_color = Rgba(1.0, 0.0, 0.0, opacity)  # Red
        y_color = Rgba(0.0, 1.0, 0.0, opacity)  # Green
        z_color = Rgba(0.0, 0.0, 1.0, opacity)  # Blue
        
        radius = length * 0.015  # Cylinder radius proportional to length
        
        # X-axis (red) - rotate 90° around Y to align with +X
        meshcat.SetObject(f"{path}/X", Cylinder(radius=radius, length=length),
                        rgba=x_color)
        meshcat.SetTransform(f"{path}/X", 
                           RigidTransform(RotationMatrix.MakeYRotation(np.pi/2), 
                                        [length/2, 0, 0]))
        
        # Y-axis (green) - rotate -90° around X to align with +Y
        meshcat.SetObject(f"{path}/Y", Cylinder(radius=radius, length=length),
                        rgba=y_color)
        meshcat.SetTransform(f"{path}/Y", 
                           RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2), 
                                        [0, length/2, 0]))
        
        # Z-axis (blue) - already aligned with +Z
        meshcat.SetObject(f"{path}/Z", Cylinder(radius=radius, length=length),
                        rgba=z_color)
        meshcat.SetTransform(f"{path}/Z", 
                           RigidTransform([0, 0, length/2]))
    
    # Add world frame at origin
    add_frame_triad(meshcat, "/Frames/World", length=0.20)
    meshcat.SetTransform("/Frames/World", RigidTransform())
    
    # Frame list to return for updates
    frame_list = []
    
    # Add all frames from the plant
    for i in range(plant.num_frames()):
        frame = plant.get_frame(FrameIndex(i))
        frame_name = frame.name()
        
        # Skip world frame (already added)
        if frame_name == "world":
            continue
        
        # Determine frame length based on frame type
        if "link" in frame_name.lower() or "cup_center" in frame_name.lower():
            length = 0.15  # Manipulator links and EE
        elif "cart" in frame_name.lower():
            length = 0.12  # Cart frame
        elif "pendulum" in frame_name.lower() or "gimbal" in frame_name.lower():
            length = 0.10  # Pendulum frames
        else:
            length = 0.08  # Other frames
        
        # Add frame triad
        path = f"/Frames/{frame_name}"
        add_frame_triad(meshcat, path, length=length)
        
        # Update frame position
        X_WF = plant.CalcRelativeTransform(context, plant.world_frame(), frame)
        meshcat.SetTransform(path, X_WF)
        
        # Store for updates
        frame_list.append((frame_name, frame, length))
    
    print(colored("✓ Coordinate frame triads added to Meshcat", "green"))
    print(colored("  Legend: X=Red, Y=Green, Z=Blue", "yellow"))
    
    return frame_list


def plot_frames_top_view(plant, context, manipulator, cart_model, title="Frame Orientation (Top View)"):
    """
    Plot coordinate frames of manipulator and cart from top view (looking down Z-axis).
    
    Args:
        plant: MultibodyPlant
        context: Plant context with current state
        manipulator: CupManipulator instance
        cart_model: Cart model instance
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Helper function to draw frame axes
    def draw_frame_2d(ax, origin, rotation_matrix, scale=0.3, colors=['r', 'g'], labels=['X', 'Z'], 
                     axis_indices=[0, 2], alpha=1.0):
        """
        Draw coordinate frame in 2D projection.
        
        Args:
            ax: Matplotlib axis
            origin: 2D origin position [x, y] where frame is located in plot
            rotation_matrix: 3D rotation matrix (3x3)
            scale: Arrow length
            colors: Colors for each axis to draw
            labels: Labels for each axis
            axis_indices: Which columns of rotation matrix to draw (e.g., [0,2] for X-Z plane)
            alpha: Transparency
        """
        # Extract which components to use for 2D plotting
        # e.g., for X-Z plane: use components [0,2] of each 3D vector
        for idx, (axis_idx, color, label) in enumerate(zip(axis_indices, colors, labels)):
            # Get the 3D axis vector from rotation matrix
            axis_3d = rotation_matrix[:, axis_idx] * scale
            # Project onto 2D using axis_indices (e.g., [X, Z] components)
            axis_2d = np.array([axis_3d[axis_indices[0]], axis_3d[axis_indices[1]]])
            
            ax.arrow(origin[0], origin[1], 
                    axis_2d[0], axis_2d[1],
                    head_width=scale*0.15, head_length=scale*0.1, 
                    fc=color, ec=color, alpha=alpha, linewidth=2)
            # Label at the end of arrow
            ax.text(origin[0] + axis_2d[0]*1.2, 
                   origin[1] + axis_2d[1]*1.2,
                   label, color=color, fontsize=12, fontweight='bold')
    
    # ============================================================================
    # PLOT 1: Manipulator Frames
    # ============================================================================
    ax1.set_title('Cup Manipulator Frames (Top View: X-Y plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)  # Both systems now in X-Y plane
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # World frame (origin)
    world_origin = np.array([0.0, 0.0, 0.0])
    draw_frame_2d(ax1, world_origin[[0, 1]], np.eye(3), scale=0.2, 
                 colors=['red', 'blue'], labels=['X_w', 'Y_w'], axis_indices=[0, 1], alpha=0.5)
    ax1.plot(0, 0, 'ko', markersize=8, label='World Origin')
    
    # Manipulator base frame
    base_frame = plant.GetFrameByName("base_mount_manipulator", manipulator.model_instance)
    X_WB = plant.CalcRelativeTransform(context, plant.world_frame(), base_frame)
    base_pos = X_WB.translation()
    base_rot = X_WB.rotation().matrix()
    draw_frame_2d(ax1, base_pos[[0, 1]], base_rot, scale=0.25,
                 colors=['darkred', 'darkblue'], labels=['X_b', 'Y_b'], axis_indices=[0, 1], alpha=0.7)
    ax1.plot(base_pos[0], base_pos[1], 'rs', markersize=10, label='Base Frame')
    
    # Link1 frame
    link1_frame = plant.GetFrameByName("link1", manipulator.model_instance)
    X_WL1 = plant.CalcRelativeTransform(context, plant.world_frame(), link1_frame)
    link1_pos = X_WL1.translation()
    link1_rot = X_WL1.rotation().matrix()
    draw_frame_2d(ax1, link1_pos[[0, 1]], link1_rot, scale=0.3,
                 colors=['crimson', 'cyan'], labels=['X_1', 'Y_1'], axis_indices=[0, 1], alpha=0.9)
    ax1.plot(link1_pos[0], link1_pos[1], 'go', markersize=10, label='Link1 Frame')
    
    # Link2 (EE) frame
    link2_frame = plant.GetFrameByName(manipulator.LINK2_NAME, manipulator.model_instance)
    X_WL2 = plant.CalcRelativeTransform(context, plant.world_frame(), link2_frame)
    link2_pos = X_WL2.translation()
    link2_rot = X_WL2.rotation().matrix()
    draw_frame_2d(ax1, link2_pos[[0, 1]], link2_rot, scale=0.35,
                 colors=['orangered', 'deepskyblue'], labels=['X_2', 'Y_2'], axis_indices=[0, 1])
    ax1.plot(link2_pos[0], link2_pos[1], 'mo', markersize=10, label='Link2 Frame')
    
    # EE position using cup_center frame
    ee_pos = manipulator.get_end_effector_position(plant, context)
    # Get cup_center frame rotation
    cup_center_frame = manipulator.get_end_effector_frame(plant)
    X_WEE = plant.CalcRelativeTransform(context, plant.world_frame(), cup_center_frame)
    ee_rot = X_WEE.rotation().matrix()
    # Draw EE frame
    draw_frame_2d(ax1, ee_pos[[0, 1]], ee_rot, scale=0.25,
                 colors=['gold', 'lime'], labels=['X_ee', 'Y_ee'], axis_indices=[0, 1], alpha=0.8)
    ax1.plot(ee_pos[0], ee_pos[1], 'r*', markersize=20, label='EE Position (cup center)', zorder=10)
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # ============================================================================
    # PLOT 2: Cart Frames  
    # ============================================================================
    ax2.set_title('Cart-Pendulum Frames (Top View: X-Y plane)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X [m]', fontsize=12)
    ax2.set_ylabel('Y [m]', fontsize=12)  # Cart works in X-Y plane
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # World frame
    draw_frame_2d(ax2, world_origin[[0, 1]], np.eye(3), scale=0.2,
                 colors=['red', 'green'], labels=['X_w', 'Y_w'], axis_indices=[0, 1], alpha=0.5)
    ax2.plot(0, 0, 'ko', markersize=8, label='World Origin')
    
    # Cart frame
    try:
        cart_body = plant.GetBodyByName("cart", cart_model)
        cart_frame = cart_body.body_frame()
        X_WCart = plant.CalcRelativeTransform(context, plant.world_frame(), cart_frame)
        cart_pos = X_WCart.translation()
        cart_rot = X_WCart.rotation().matrix()
        draw_frame_2d(ax2, cart_pos[[0, 1]], cart_rot, scale=0.3,
                     colors=['purple', 'orange'], labels=['X_c', 'Y_c'], axis_indices=[0, 1])
        ax2.plot(cart_pos[0], cart_pos[1], 'bs', markersize=12, label='Cart Frame')
        
        # Mark cart position
        ax2.plot(cart_pos[0], cart_pos[1], 'b*', markersize=20, label='Cart Position', zorder=10)
    except Exception as e:
        print(colored(f"Warning: Could not get cart frame: {e}", "yellow"))
    
    # Try to get pendulum frame
    try:
        pend_body = plant.GetBodyByName("pendulum", cart_model)
        pend_frame = pend_body.body_frame()
        X_WPend = plant.CalcRelativeTransform(context, plant.world_frame(), pend_frame)
        pend_pos = X_WPend.translation()
        pend_rot = X_WPend.rotation().matrix()
        draw_frame_2d(ax2, pend_pos[[0, 1]], pend_rot, scale=0.25,
                     colors=['darkviolet', 'gold'], labels=['X_p', 'Y_p'], axis_indices=[0, 1], alpha=0.7)
        ax2.plot(pend_pos[0], pend_pos[1], 'mo', markersize=10, label='Pendulum Frame')
    except Exception as e:
        print(colored(f"Info: No pendulum frame found: {e}", "cyan"))
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # ============================================================================
    # PLOT 3: Combined View - All Frames with Offset to Avoid Overlap
    # ============================================================================
    ax3.set_title('All Frames Combined (Side View: X-Z plane)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X [m]', fontsize=12)
    ax3.set_ylabel('Z [m]', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # World frame at origin
    draw_frame_2d(ax3, world_origin[[0, 2]], np.eye(3), scale=0.15, 
                 colors=['red', 'blue'], labels=['W_x', 'W_z'], axis_indices=[0, 2], alpha=0.4)
    ax3.plot(0, 0, 'ko', markersize=6, label='World Origin', alpha=0.5)
    
    # Manipulator frames (in X-Z plane, actual positions)
    draw_frame_2d(ax3, base_pos[[0, 2]], base_rot, scale=0.18,
                 colors=['darkred', 'darkblue'], labels=['B_x', 'B_z'], axis_indices=[0, 2], alpha=0.6)
    ax3.plot(base_pos[0], base_pos[2], 'r^', markersize=7, label='Base', alpha=0.7)
    
    draw_frame_2d(ax3, link1_pos[[0, 2]], link1_rot, scale=0.20,
                 colors=['crimson', 'cyan'], labels=['L1_x', 'L1_z'], axis_indices=[0, 2], alpha=0.7)
    ax3.plot(link1_pos[0], link1_pos[2], 'gs', markersize=7, label='Link1', alpha=0.7)
    
    draw_frame_2d(ax3, link2_pos[[0, 2]], link2_rot, scale=0.22,
                 colors=['orangered', 'deepskyblue'], labels=['L2_x', 'L2_z'], axis_indices=[0, 2], alpha=0.8)
    ax3.plot(link2_pos[0], link2_pos[2], 'mo', markersize=7, label='Link2', alpha=0.7)
    
    # EE position with offset (cup center)
    # Draw EE frame at offset position
    draw_frame_2d(ax3, ee_pos[[0, 2]], link2_rot, scale=0.18,
                 colors=['gold', 'lime'], labels=['EE_x', 'EE_z'], axis_indices=[0, 2], alpha=0.9)
    ax3.plot(ee_pos[0], ee_pos[2], 'r*', markersize=15, label='EE (cup center)', zorder=10)
    
    # Cart frame - plot in X-Z plane at its Y position (shifted vertically for visibility)
    try:
        # Cart is at height z_offset, position (cart_x, cart_y) in X-Y plane
        # Map cart's X-Y position to X-Z for visualization: (cart_X, cart_Y) → (cart_X, cart_Y_as_Z)
        cart_viz_pos = np.array([cart_pos[0], cart_pos[1]])  # Use Y as Z for visualization
        draw_frame_2d(ax3, cart_viz_pos, cart_rot, scale=0.20,
                     colors=['purple', 'orange'], labels=['C_x', 'C_y'], axis_indices=[0, 1], alpha=0.8)
        ax3.plot(cart_viz_pos[0], cart_viz_pos[1], 'bd', markersize=9, label='Cart (X-Y plane)', zorder=9)
        
        # Draw line connecting EE and Cart to show mapping
        ax3.plot([ee_pos[0], cart_viz_pos[0]], [ee_pos[2], cart_viz_pos[1]], 
                'k--', linewidth=1.5, alpha=0.4, label='EE[X,Z]→Cart[X,Y]')
        
        # Annotate the mapping
        mid_x = (ee_pos[0] + cart_viz_pos[0]) / 2
        mid_z = (ee_pos[2] + cart_viz_pos[1]) / 2
        ax3.annotate('X-Z to X-Y mapping', xy=(mid_x, mid_z), fontsize=9, 
                    ha='center', color='black', alpha=0.6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    except Exception as e:
        print(colored(f"Warning: Could not plot cart in combined view: {e}", "yellow"))
    
    # Add annotations for clarity
    ax3.text(0.02, 0.98, 'Manipulator: X-Z plane\nCart: X-Y plane (Y shown as Z)', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Overall figure title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(colored(f"\n📊 Frame visualization plotted with 3 views", "green"))
    print(colored(f"   Left: Manipulator frames in X-Z plane (after 180° URDF flip)", "cyan"))
    print(colored(f"   Middle: Cart-Pendulum frames in X-Y plane", "cyan"))
    print(colored(f"   Right: Combined view showing all frames and X-Z→X-Y mapping", "cyan"))
    
    return fig


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_lqr_manip_ee_traj_track_results(t, state_data, ref_data, cart_traj_data, 
                                         ee_positions, ee_velocities, force_data, 
                                         impedance_data, manip_state_data, 
                                         manip_desired_state_data, manip_js_torque_data, config):
    """
    Generate comprehensive plots for LQR manipulator end-effector trajectory tracking.
    
    Args:
        t: Time vector
        state_data: Cart-pendulum state [x, y, α, β, ẋ, ẏ, α̇, β̇]
        ref_data: ZFT reference state [x_ref, y_ref, ẋ_ref, ẏ_ref]
        cart_traj_data: Cart trajectory sent to manipulator [x, y, ẋ, ẏ]
        ee_positions: End-effector positions [x_EE, y_EE]
        ee_velocities: End-effector velocities [ẋ_EE, ẏ_EE]
        force_data: Muscle forces [F_x, F_y]
        impedance_data: Impedance forces [F_x_imp, F_y_imp]
        manip_state_data: Manipulator state [q1, q2, q̇1, q̇2]
        manip_desired_state_data: Desired manipulator state from IK [q1_d, q2_d, q̇1_d, q̇2_d, q̈1_d, q̈2_d]
        manip_js_torque_data: Joint-space controller torques [τ1, τ2]
        config: SimulationConfig with target_x, target_y attributes
    """
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(5, 4, figure=fig)
    
    # Row 1: Cart position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, state_data[0, :], 'b-', label='x (cart)', linewidth=2)
    ax1.plot(t, ref_data[0, :], 'r--', label='x_ref', linewidth=1.5)
    ax1.plot(t, cart_traj_data[0, :], 'c-.', label='x (to manip)', linewidth=1.5, alpha=0.7)
    ax1.plot(t, ee_positions[0, :], 'g:', label='x_EE', linewidth=2)
    ax1.axhline(config.target_x, color='m', linestyle=':', label='target')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X Position [m]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Cart X Position')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, state_data[1, :], 'b-', label='y (cart)', linewidth=2)
    ax2.plot(t, ref_data[1, :], 'r--', label='y_ref', linewidth=1.5)
    ax2.plot(t, cart_traj_data[1, :], 'c-.', label='y (to manip)', linewidth=1.5, alpha=0.7)
    ax2.plot(t, ee_positions[1, :], 'g:', label='y_EE', linewidth=2)
    ax2.axhline(config.target_y, color='m', linestyle=':', label='target')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y Position [m]')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Cart Y Position')
    
    # 2D trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state_data[0, :], state_data[1, :], 'b-', label='cart', linewidth=2)
    ax3.plot(ref_data[0, :], ref_data[1, :], 'r--', label='reference', linewidth=1.5)
    ax3.plot(cart_traj_data[0, :], cart_traj_data[1, :], 'c-.', label='to manip', linewidth=1.5, alpha=0.7)
    ax3.plot(ee_positions[0, :], ee_positions[1, :], 'g:', label='EE', linewidth=2)
    ax3.plot(config.target_x, config.target_y, 'm*', markersize=15, label='target')
    ax3.plot(state_data[0, 0], state_data[1, 0], 'ko', markersize=8, label='start')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    ax3.set_title('2D Trajectory')
    
    # Tracking error
    ax4 = fig.add_subplot(gs[0, 3])
    error_x = state_data[0, :] - ref_data[0, :]
    error_y = state_data[1, :] - ref_data[1, :]
    error_mag = np.sqrt(error_x**2 + error_y**2)
    ax4.plot(t, error_mag, 'r-', linewidth=2)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Tracking Error [m]')
    ax4.grid(True)
    ax4.set_title('Position Tracking Error')
    
    # Row 2: Cart velocity
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t, state_data[4, :], 'b-', label='ẋ (cart)', linewidth=2)
    ax5.plot(t, ref_data[2, :], 'r--', label='ẋ_ref', linewidth=1.5)
    ax5.plot(t, cart_traj_data[2, :], 'c-.', label='ẋ (to manip)', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('X Velocity [m/s]')
    ax5.legend()
    ax5.grid(True)
    ax5.set_title('Cart X Velocity')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t, state_data[5, :], 'b-', label='ẏ (cart)', linewidth=2)
    ax6.plot(t, ref_data[3, :], 'r--', label='ẏ_ref', linewidth=1.5)
    ax6.plot(t, cart_traj_data[3, :], 'c-.', label='ẏ (to manip)', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Y Velocity [m/s]')
    ax6.legend()
    ax6.grid(True)
    ax6.set_title('Cart Y Velocity')
    
    # Combined velocity magnitude
    ax7 = fig.add_subplot(gs[1, 2])
    vel_cart = np.sqrt(state_data[4, :]**2 + state_data[5, :]**2)
    vel_ref = np.sqrt(ref_data[2, :]**2 + ref_data[3, :]**2)
    vel_to_manip = np.sqrt(cart_traj_data[2, :]**2 + cart_traj_data[3, :]**2)
    ax7.plot(t, vel_cart, 'b-', label='|v| cart', linewidth=2)
    ax7.plot(t, vel_ref, 'r--', label='|v| ref', linewidth=1.5)
    ax7.plot(t, vel_to_manip, 'c-.', label='|v| to manip', linewidth=1.5, alpha=0.7)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Velocity Magnitude [m/s]')
    ax7.legend()
    ax7.grid(True)
    ax7.set_title('Velocity Magnitude')
    
    # Velocity tracking error
    ax8 = fig.add_subplot(gs[1, 3])
    vel_error = vel_cart - vel_ref
    ax8.plot(t, vel_error, 'r-', linewidth=2)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Velocity Error [m/s]')
    ax8.grid(True)
    ax8.set_title('Velocity Tracking Error')
    
    # Row 3: Pendulum angles and angular velocities
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(t, np.rad2deg(state_data[2, :]), 'b-', label='pitch (α)', linewidth=2)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Pitch Angle [deg]')
    ax9.legend()
    ax9.grid(True)
    ax9.set_title('Pendulum Pitch Angle')
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot(t, np.rad2deg(state_data[3, :]), 'r-', label='roll (β)', linewidth=2)
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Roll Angle [deg]')
    ax10.legend()
    ax10.grid(True)
    ax10.set_title('Pendulum Roll Angle')
    
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(t, np.rad2deg(state_data[6, :]), 'b-', label='α̇', linewidth=2)
    ax11.set_xlabel('Time [s]')
    ax11.set_ylabel('Pitch Angular Velocity [deg/s]')
    ax11.legend()
    ax11.grid(True)
    ax11.set_title('Pendulum Pitch Angular Velocity')
    
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(t, np.rad2deg(state_data[7, :]), 'r-', label='β̇', linewidth=2)
    ax12.set_xlabel('Time [s]')
    ax12.set_ylabel('Roll Angular Velocity [deg/s]')
    ax12.legend()
    ax12.grid(True)
    ax12.set_title('Pendulum Roll Angular Velocity')
    
    # Row 4: Forces
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.plot(t, force_data[0, :], 'b-', label='F_x (muscle)', linewidth=2)
    ax13.plot(t, impedance_data[0, :], 'c--', label='F_x (impedance)', linewidth=1.5, alpha=0.7)
    ax13.set_xlabel('Time [s]')
    ax13.set_ylabel('Force X [N]')
    ax13.legend()
    ax13.grid(True)
    ax13.set_title('X-Direction Forces')
    
    ax14 = fig.add_subplot(gs[3, 1])
    ax14.plot(t, force_data[1, :], 'r-', label='F_y (muscle)', linewidth=2)
    ax14.plot(t, impedance_data[1, :], 'm--', label='F_y (impedance)', linewidth=1.5, alpha=0.7)
    ax14.set_xlabel('Time [s]')
    ax14.set_ylabel('Force Y [N]')
    ax14.legend()
    ax14.grid(True)
    ax14.set_title('Y-Direction Forces')
    
    # Force magnitude
    ax15 = fig.add_subplot(gs[3, 2])
    force_muscle_mag = np.sqrt(force_data[0, :]**2 + force_data[1, :]**2)
    force_impedance_mag = np.sqrt(impedance_data[0, :]**2 + impedance_data[1, :]**2)
    ax15.plot(t, force_muscle_mag, 'b-', label='|F| muscle', linewidth=2)
    ax15.plot(t, force_impedance_mag, 'c--', label='|F| impedance', linewidth=1.5, alpha=0.7)
    ax15.set_xlabel('Time [s]')
    ax15.set_ylabel('Force Magnitude [N]')
    ax15.legend()
    ax15.grid(True)
    ax15.set_title('Force Magnitude')
    
    # Energy-like metric
    ax16 = fig.add_subplot(gs[3, 3])
    cart_kinetic = 0.5 * (state_data[4, :]**2 + state_data[5, :]**2)
    ax16.plot(t, cart_kinetic, 'b-', linewidth=2)
    ax16.set_xlabel('Time [s]')
    ax16.set_ylabel('Kinetic Energy (cart) [normalized]')
    ax16.grid(True)
    ax16.set_title('Cart Kinetic Energy')
    
    # Row 5: Manipulator state (joint angles and velocities, EE position/velocity)
    ax17 = fig.add_subplot(gs[4, 0])
    q1_deg = np.rad2deg(manip_state_data[0, :])
    q2_deg = np.rad2deg(manip_state_data[1, :])
    q1_des_deg = np.rad2deg(manip_desired_state_data[0, :])
    q2_des_deg = np.rad2deg(manip_desired_state_data[1, :])
    ax17.plot(t, q1_deg, 'b-', linewidth=2.5, label='q1 actual', alpha=0.8)
    ax17.plot(t, q2_deg, 'r-', linewidth=2.5, label='q2 actual', alpha=0.8)
    ax17.plot(t, q1_des_deg, 'b--', linewidth=1.5, label='q1 desired (IK)', alpha=0.7)
    ax17.plot(t, q2_des_deg, 'r--', linewidth=1.5, label='q2 desired (IK)', alpha=0.7)
    ax17.set_xlabel('Time [s]')
    ax17.set_ylabel('Joint Angles [deg]')
    ax17.legend(fontsize=8)
    ax17.grid(True)
    ax17.set_title('Manipulator Joint Angles: Actual vs Desired (IK from cart)')
    
    ax18 = fig.add_subplot(gs[4, 1])
    q1_dot_deg = np.rad2deg(manip_state_data[2, :])
    q2_dot_deg = np.rad2deg(manip_state_data[3, :])
    q1_dot_des_deg = np.rad2deg(manip_desired_state_data[2, :])
    q2_dot_des_deg = np.rad2deg(manip_desired_state_data[3, :])
    ax18.plot(t, q1_dot_deg, 'b-', linewidth=2.5, label='q̇1 actual', alpha=0.8)
    ax18.plot(t, q2_dot_deg, 'r-', linewidth=2.5, label='q̇2 actual', alpha=0.8)
    ax18.plot(t, q1_dot_des_deg, 'b--', linewidth=1.5, label='q̇1 desired (IK)', alpha=0.7)
    ax18.plot(t, q2_dot_des_deg, 'r--', linewidth=1.5, label='q̇2 desired (IK)', alpha=0.7)
    ax18.set_xlabel('Time [s]')
    ax18.set_ylabel('Joint Velocities [deg/s]')
    ax18.legend(fontsize=8)
    ax18.grid(True)
    ax18.set_title('Manipulator Joint Velocities: Actual vs Desired (IK from cart)')
    
    ax19 = fig.add_subplot(gs[4, 2])
    ax19.plot(t, ee_positions[0, :], 'b-', linewidth=2, label='EE x')
    ax19.plot(t, ee_positions[1, :], 'r-', linewidth=2, label='EE y')
    ax19.plot(t, state_data[0, :], 'b:', linewidth=1.5, alpha=0.7, label='cart x')
    ax19.plot(t, state_data[1, :], 'r:', linewidth=1.5, alpha=0.7, label='cart y')
    ax19.set_xlabel('Time [s]')
    ax19.set_ylabel('EE Position [m]')
    ax19.legend()
    ax19.grid(True)
    ax19.set_title('Manipulator End-Effector Position vs Cart')
    
    ax20 = fig.add_subplot(gs[4, 3])
    ax20.plot(t, ee_velocities[0, :], 'b-', linewidth=2, label='EE ẋ')
    ax20.plot(t, ee_velocities[1, :], 'r-', linewidth=2, label='EE ẏ')
    ax20.plot(t, state_data[4, :], 'b:', linewidth=1.5, alpha=0.7, label='cart ẋ')
    ax20.plot(t, state_data[5, :], 'r:', linewidth=1.5, alpha=0.7, label='cart ẏ')
    ax20.set_xlabel('Time [s]')
    ax20.set_ylabel('EE Velocity [m/s]')
    ax20.legend()
    ax20.grid(True)
    ax20.set_title('Manipulator End-Effector Velocity vs Cart')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = 'plots/lqr_manip_ee_traj_track_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Main plots saved to {plot_path}", "green"))
    plt.show()
