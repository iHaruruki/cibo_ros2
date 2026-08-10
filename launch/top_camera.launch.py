import os
import glob
import subprocess

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    share_dir = get_package_share_directory("cibo_ros2")
    subprocess.run(["uv", "sync", "--project", share_dir, "--no-editable"], check=True)
    venv_site_pkgs = glob.glob(
        os.path.join(share_dir, ".venv", "lib", "python*", "site-packages")
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = ":".join(venv_site_pkgs + [existing_pythonpath]).strip(":")

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="holistic_pose",
        description="Namespace for the nodes",
    )
    
    mediapipe_node_cmd = Node(
        package="cibo_ros2",
        executable="top_camera_depth.py",
        name="top_camera_depth",
        namespace=namespace,
        additional_env={"PYTHONPATH": new_pythonpath},
    )
        
    return LaunchDescription([
        namespace_cmd,
        mediapipe_node_cmd,
    ])