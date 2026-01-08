from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.robot import Robot
from onshape_robotics_toolkit.utilities.helpers import save_model_as_json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
key_env_path = os.path.join(script_dir, "key.env")

# Initialize the client
client = Client(env=key_env_path)

# Load the Onshape Assembly
robot = Robot.from_url(
    name="cart_pendulum",
    url="https://cad.onshape.com/documents/02830e503a1c2626d58e0780/w/62db7934caa01effe231f1a9/e/4d4867fcb5ca80ef5736211e",
    client=client,
    max_depth=0,
    use_user_defined_root=True,
)

# Save the Assembly as JSON in the script directory
json_path = os.path.join(script_dir, "manipulator_cart_pendulum.json")
save_model_as_json(robot.assembly, json_path)

# Visualize the Assembly Graph (commented out due to threading issues)
png_path = os.path.join(script_dir, "manipulator_cart_pendulum.png")
robot.show_graph(file_name=png_path)

# Save the Robot Object as a URDF File in the script directory
urdf_path = os.path.join(script_dir, "manipulator_cart_pendulum.urdf")
robot.save(file_path=urdf_path)
