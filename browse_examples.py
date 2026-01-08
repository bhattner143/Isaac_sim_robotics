"""
ISAAC Sim Examples Browser and Launcher
Browse and run existing ISAAC Sim examples from your installation
"""

import os
import glob

ISAAC_SIM_PATH = os.path.expanduser("~/isaacsim")
EXAMPLES_DIRS = [
    "standalone_examples/tutorials",
    "standalone_examples/api",
    "standalone_examples/benchmarks",
    "standalone_examples/replicator",
]

def find_examples():
    """Find all example Python files"""
    examples = {}
    
    for example_dir in EXAMPLES_DIRS:
        full_path = os.path.join(ISAAC_SIM_PATH, example_dir)
        if not os.path.exists(full_path):
            continue
            
        # Find all .py files recursively
        pattern = os.path.join(full_path, "**/*.py")
        py_files = glob.glob(pattern, recursive=True)
        
        for py_file in py_files:
            # Skip __init__.py and __pycache__
            if "__init__" in py_file or "__pycache__" in py_file:
                continue
                
            # Get relative path from ISAAC_SIM_PATH
            rel_path = os.path.relpath(py_file, ISAAC_SIM_PATH)
            
            # Categorize by directory
            category = example_dir.split("/")[1]  # tutorials, api, benchmarks, etc.
            if category not in examples:
                examples[category] = []
            
            examples[category].append({
                'name': os.path.basename(py_file),
                'path': py_file,
                'rel_path': rel_path
            })
    
    return examples

def print_examples():
    """Print all available examples"""
    examples = find_examples()
    
    print("=" * 80)
    print("ISAAC SIM EXAMPLES")
    print("=" * 80)
    print()
    
    for category in sorted(examples.keys()):
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        for i, example in enumerate(sorted(examples[category], key=lambda x: x['name']), 1):
            print(f"{i:3d}. {example['name']:<50} ")
            print(f"     {example['rel_path']}")
    
    print(f"\n{'='*80}")
    print(f"Total examples found: {sum(len(v) for v in examples.values())}")
    print(f"{'='*80}")

def get_popular_examples():
    """Return a list of popular/recommended examples to start with"""
    return [
        {
            'name': 'Getting Started',
            'path': os.path.join(ISAAC_SIM_PATH, 'standalone_examples/tutorials/getting_started.py'),
            'description': 'Basic tutorial to get started with ISAAC Sim'
        },
        {
            'name': 'Getting Started with Robot',
            'path': os.path.join(ISAAC_SIM_PATH, 'standalone_examples/tutorials/getting_started_robot.py'),
            'description': 'Introduction to robot simulation'
        },
        {
            'name': 'Camera Example',
            'path': os.path.join(ISAAC_SIM_PATH, 'standalone_examples/benchmarks/benchmark_camera.py'),
            'description': 'Camera sensor benchmarking'
        },
    ]

def print_popular_examples():
    """Print recommended examples for beginners"""
    print("\n" + "=" * 80)
    print("RECOMMENDED EXAMPLES FOR GETTING STARTED")
    print("=" * 80 + "\n")
    
    popular = get_popular_examples()
    for i, example in enumerate(popular, 1):
        print(f"{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Path: {example['path']}")
        if os.path.exists(example['path']):
            print(f"   Status: ✓ Available")
        else:
            print(f"   Status: ✗ Not found")
        print()

def print_usage_instructions():
    """Print how to use the examples"""
    print("\n" + "=" * 80)
    print("HOW TO RUN EXAMPLES")
    print("=" * 80 + "\n")
    
    print("METHOD 1: Using the launcher script (recommended)")
    print("-" * 80)
    print("cd ~/Documents/isac_sim_pydrake")
    print("./launch_isaac.sh python <path_to_example>")
    print()
    print("Example:")
    print("./launch_isaac.sh python ~/isaacsim/standalone_examples/tutorials/getting_started.py")
    print()
    
    print("METHOD 2: Using conda environment directly")
    print("-" * 80)
    print("conda activate env_isaacsim")
    print("python <path_to_example>")
    print()
    print("Example:")
    print("conda activate env_isaacsim")
    print("python ~/isaacsim/standalone_examples/tutorials/getting_started.py")
    print()
    
    print("METHOD 3: Copy to your workspace and run")
    print("-" * 80)
    print("cp ~/isaacsim/standalone_examples/tutorials/getting_started.py ~/Documents/isac_sim_pydrake/")
    print("cd ~/Documents/isac_sim_pydrake")
    print("conda activate env_isaacsim")
    print("python getting_started.py")
    print()

if __name__ == "__main__":
    print_popular_examples()
    print_usage_instructions()
    print_examples()
