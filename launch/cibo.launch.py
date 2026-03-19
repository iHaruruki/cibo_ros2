# Copyright (c) 2025
# MIT License

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_file_name = 'cibo.urdf'
    urdf = os.path.join(get_package_share_directory('cibo'), urdf_file_name)
    with open(urdf, 'r') as f:
        robot_desc = f.read()

    rviz_config_dir = os.path.join(
        get_package_share_directory('cibo'),
        'rviz', 'cibo_tf.rviz'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf]
        ),

        # Dual camera sync republisher
        # Node(
        #     package='cibo',
        #     executable='dual_camera_sync_node',
        #     name='dual_camera_sync_node',
        #     output='screen',
        #     parameters=[{'queue_size': 20, 'slop_sec': 0.08, 'restamp': False}],
        # ),

        # front_camera_depth_node -> /sync/camera_01/** を購読
        Node(
            package='cibo',
            executable='front_camera_depth_node',
            name='front_camera_node',
            output='screen',
            # remappings=[
            #     ('/camera_01/color/image_raw', '/sync/camera_01/color/image_raw'),
            #     ('/camera_01/depth/image_raw', '/sync/camera_01/depth/image_raw'),
            #     ('/camera_01/depth/camera_info', '/sync/camera_01/depth/camera_info'),
            # ]
        ),

        # top_camera_depth_node -> /sync/camera_02/** を購読
        Node(
            package='cibo',
            executable='top_camera_depth_node',
            name='top_camera_node',
            output='screen',
            # remappings=[
            #     ('/camera_02/color/image_raw', '/sync/camera_02/color/image_raw'),
            #     ('/camera_02/depth/image_raw', '/sync/camera_02/depth/image_raw'),
            #     ('/camera_02/depth/camera_info', '/sync/camera_02/depth/camera_info'),
            # ]
        ),

        Node(
            package='rviz2', 
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen',
        ),
    ])