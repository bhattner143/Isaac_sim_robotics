from isaacsim import SimulationApp

# Launch Isaac Sim first
simulation_app = SimulationApp({"headless": True})

try:
    from isaacsim.storage.native import get_assets_root_path
    import os
    
    # Get assets root path
    assets_root = get_assets_root_path()
    print(f"\nIsaac Sim Assets Root Path:")
    print(f"  {assets_root}")
    
    # Construct Franka USD path
    franka_relative_path = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    franka_full_path = assets_root + franka_relative_path
    
    print(f"\nFranka USD Relative Path:")
    print(f"  {franka_relative_path}")
    
    print(f"\nFranka USD Full Path:")
    print(f"  {franka_full_path}")
    
    # Check if file exists
    if os.path.exists(franka_full_path):
        print(f"\n✓ File exists!")
        file_size = os.path.getsize(franka_full_path)
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    else:
        print(f"\n✗ File NOT found at this location")
        
        # Try to find it
        print(f"\nSearching for franka.usd...")
        import subprocess
        result = subprocess.run(
            ["find", os.path.expanduser("~/.local/share/ov"), "-name", "franka.usd"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"Found at:")
            for path in result.stdout.strip().split('\n'):
                print(f"  {path}")
        else:
            print(f"  Not found in ~/.local/share/ov")

finally:
    simulation_app.close()
