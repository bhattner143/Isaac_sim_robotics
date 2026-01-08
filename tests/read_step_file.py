"""
Read and inspect STEP (.stp/.step) CAD files using Python

Two modes:
1. Basic mode (no dependencies): Read STEP file header and count entities
2. Full mode (requires pythonocc-core): Complete geometry analysis and visualization
"""

import sys
from pathlib import Path
import re

import re

def read_step_file_basic(step_file_path):
    """
    Read STEP file and extract basic information (no dependencies required)
    
    Args:
        step_file_path: Path to the STEP file
    """
    step_path = Path(step_file_path)
    if not step_path.exists():
        print(f"ERROR: File not found: {step_file_path}")
        return None
    
    print("=" * 70)
    print(f"Reading STEP file: {step_path.name}")
    print("=" * 70)
    
    try:
        with open(step_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print("✓ File loaded successfully")
        print(f"✓ File size: {step_path.stat().st_size / 1024:.2f} KB")
        
        # Extract header information
        header_match = re.search(r'HEADER;(.*?)ENDSEC;', content, re.DOTALL)
        if header_match:
            print("\n" + "-" * 70)
            print("HEADER INFORMATION")
            print("-" * 70)
            
            header = header_match.group(1)
            
            # Extract file description
            file_desc = re.search(r"FILE_DESCRIPTION\s*\(\s*\('([^']+)'", header)
            if file_desc:
                print(f"Description: {file_desc.group(1)}")
            
            # Extract file name
            file_name = re.search(r"FILE_NAME\s*\(\s*'([^']+)'", header)
            if file_name:
                print(f"Name: {file_name.group(1)}")
            
            # Extract timestamp
            timestamp = re.search(r"'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", header)
            if timestamp:
                print(f"Timestamp: {timestamp.group(1)}")
            
            # Extract author
            author = re.search(r"FILE_NAME.*?\(\s*'[^']*',\s*'[^']*',\s*\('([^']+)'", header, re.DOTALL)
            if author:
                print(f"Author: {author.group(1)}")
            
            # Extract schema
            schema = re.search(r"FILE_SCHEMA\s*\(\s*\('([^']+)'", header)
            if schema:
                print(f"Schema: {schema.group(1)}")
        
        # Count entities in DATA section
        data_match = re.search(r'DATA;(.*?)ENDSEC;', content, re.DOTALL)
        if data_match:
            data = data_match.group(1)
            
            print("\n" + "-" * 70)
            print("ENTITY COUNTS")
            print("-" * 70)
            
            # Count different entity types
            entities = {
                'CARTESIAN_POINT': len(re.findall(r'CARTESIAN_POINT', data)),
                'DIRECTION': len(re.findall(r'DIRECTION', data)),
                'VERTEX_POINT': len(re.findall(r'VERTEX_POINT', data)),
                'EDGE_CURVE': len(re.findall(r'EDGE_CURVE', data)),
                'FACE_OUTER_BOUND': len(re.findall(r'FACE_OUTER_BOUND', data)),
                'ADVANCED_FACE': len(re.findall(r'ADVANCED_FACE', data)),
                'CLOSED_SHELL': len(re.findall(r'CLOSED_SHELL', data)),
                'MANIFOLD_SOLID_BREP': len(re.findall(r'MANIFOLD_SOLID_BREP', data)),
                'AXIS2_PLACEMENT_3D': len(re.findall(r'AXIS2_PLACEMENT_3D', data)),
                'CIRCLE': len(re.findall(r'CIRCLE', data)),
                'LINE': len(re.findall(r'LINE', data)),
                'B_SPLINE': len(re.findall(r'B_SPLINE', data)),
            }
            
            for entity_type, count in entities.items():
                if count > 0:
                    print(f"{entity_type:25s}: {count:6d}")
            
            # Total entities
            total_entities = len(re.findall(r'^#\d+', data, re.MULTILINE))
            print(f"\n{'TOTAL ENTITIES':25s}: {total_entities:6d}")
        
        print("\n" + "=" * 70)
        print("BASIC ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nFor detailed geometry analysis, install pythonocc-core:")
        print("  conda install -c conda-forge pythonocc-core")
        print("Then run with --full flag")
        
        return content
        
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return None


def read_step_file(step_file_path):
    """
    Read a STEP file and extract basic information
    
    Args:
        step_file_path: Path to the STEP file
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
        from OCC.Core.TopoDS import topods
    except ImportError:
        print("ERROR: pythonocc-core not installed")
        print("Install with: conda install -c conda-forge pythonocc-core")
        print("Or: pip install pythonocc-core")
        return None
    
    step_path = Path(step_file_path)
    if not step_path.exists():
        print(f"ERROR: File not found: {step_file_path}")
        return None
    
    print("=" * 70)
    print(f"Reading STEP file: {step_path.name}")
    print("=" * 70)
    
    # Create STEP reader
    step_reader = STEPControl_Reader()
    
    # Read the file
    status = step_reader.ReadFile(str(step_path))
    
    if status != IFSelect_RetDone:
        print(f"ERROR: Failed to read STEP file (status: {status})")
        return None
    
    print("✓ File loaded successfully")
    
    # Transfer shapes to document
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    print(f"✓ Shape extracted")
    
    # Count geometric entities
    solids = 0
    faces = 0
    edges = 0
    vertices = 0
    
    # Explore solids
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids += 1
        exp.Next()
    
    # Explore faces
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces += 1
        exp.Next()
    
    # Explore edges
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edges += 1
        exp.Next()
    
    # Explore vertices
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        vertices += 1
        exp.Next()
    
    print("\n" + "-" * 70)
    print("GEOMETRY INFORMATION")
    print("-" * 70)
    print(f"Solids:   {solids}")
    print(f"Faces:    {faces}")
    print(f"Edges:    {edges}")
    print(f"Vertices: {vertices}")
    
    # Calculate properties if we have solids
    if solids > 0:
        try:
            props = GProp_GProps()
            brepgprop_VolumeProperties(shape, props)
            volume = props.Mass()
            
            print("\n" + "-" * 70)
            print("PHYSICAL PROPERTIES")
            print("-" * 70)
            print(f"Volume: {volume:.6f} cubic units")
            
            # Center of mass
            cog = props.CentreOfMass()
            print(f"Center of Mass: ({cog.X():.4f}, {cog.Y():.4f}, {cog.Z():.4f})")
        except Exception as e:
            print(f"\nNote: Could not calculate volume properties: {e}")
    
    # Calculate surface area
    try:
        props_surface = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props_surface)
        surface_area = props_surface.Mass()
        print(f"Surface Area: {surface_area:.6f} square units")
    except Exception as e:
        print(f"Note: Could not calculate surface area: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return shape


def visualize_step_file(step_file_path):
    """
    Read and visualize a STEP file using pythonocc viewer
    
    Args:
        step_file_path: Path to the STEP file
    """
    shape = read_step_file(step_file_path)
    
    if shape is None:
        return
    
    try:
        from OCC.Display.SimpleGui import init_display
        
        print("\nLaunching 3D viewer...")
        print("Close the viewer window to exit.")
        
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(shape, update=True)
        start_display()
        
    except ImportError:
        print("\nNote: 3D viewer not available")
        print("The file was analyzed successfully, but visualization requires display libraries")
    except Exception as e:
        print(f"\nNote: Could not launch viewer: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_step_file.py <path_to_step_file> [--full] [--visualize]")
        print("\nExample:")
        print("  python read_step_file.py model.step           # Basic analysis (no dependencies)")
        print("  python read_step_file.py model.step --full    # Full analysis (requires pythonocc)")
        print("  python read_step_file.py model.stp --visualize # Full + 3D viewer")
        return
    
    step_file = sys.argv[1]
    use_full = "--full" in sys.argv or "-f" in sys.argv
    visualize = "--visualize" in sys.argv or "-v" in sys.argv
    
    if visualize:
        visualize_step_file(step_file)
    elif use_full:
        read_step_file(step_file)
    else:
        # Default: basic mode (no dependencies)
        read_step_file_basic(step_file)


if __name__ == "__main__":
    main()
