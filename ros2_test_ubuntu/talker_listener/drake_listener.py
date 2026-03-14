"""
Drake Listener Launcher — conda Python 3.11
============================================
Launches ros2_drake_listener_node.py using system Python 3.12
(where rclpy is installed) with the ROS 2 environment sourced.

Usage:
    conda activate pydrake
    python ros2_test_ubuntu/script_drake_listener.py

Or use the convenience script:
    bash ros2_test_ubuntu/run_listener.sh
"""

import os
import signal
import subprocess
import sys

# ============================================================================
# PATHS
# ============================================================================
SYSTEM_PYTHON = "/usr/bin/python3"
ROS2_SETUP    = os.path.expanduser("~/ros2_jazzy/install/local_setup.bash")
THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
LISTENER_NODE = os.path.join(os.path.dirname(THIS_DIR), "nodes", "ros2_drake_listener_node.py")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Validate paths
    for path, label in [
        (SYSTEM_PYTHON, "System Python"),
        (ROS2_SETUP,    "ROS 2 setup"),
        (LISTENER_NODE, "Listener node"),
    ]:
        if not os.path.exists(path):
            print(f"[script_drake_listener] ERROR: {label} not found: {path}")
            sys.exit(1)

    print("[script_drake_listener] Launching ROS 2 Listener...")
    print(f"  System Python : {SYSTEM_PYTHON}")
    print(f"  ROS 2 setup   : {ROS2_SETUP}")
    print(f"  Listener node : {LISTENER_NODE}")
    print(f"\nListening on /drake_hello — press Ctrl+C to stop.\n")

    # Source ROS 2 and run listener node with system Python
    cmd = (
        f"source {ROS2_SETUP} && "
        f"{SYSTEM_PYTHON} {LISTENER_NODE}"
    )

    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
    )

    print(f"[script_drake_listener] Listener running (PID={proc.pid})")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[script_drake_listener] Ctrl+C — stopping listener...")
        proc.send_signal(signal.SIGINT)
        proc.wait()
        print("[script_drake_listener] Done.")


if __name__ == '__main__':
    main()
