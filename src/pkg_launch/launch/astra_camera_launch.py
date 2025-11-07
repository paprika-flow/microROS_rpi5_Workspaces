#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    device_path_arg = DeclareLaunchArgument(
        'device_path',
        default_value='/dev/video0',
        description='Path to the camera device'
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='20.0',
        description='Camera frame rate (Hz)'
    )

    return LaunchDescription([
        # Declare launch arguments
        device_path_arg,
        fps_arg,

        # Node to run the camera publisher
        Node(
            package='pkg_sensors',
            executable='astra_camera_node',  # matches entry_points in setup.py
            name='astra_camera_node',
            output='screen',
            parameters=[{
                'device_path': LaunchConfiguration('device_path'),
                'fps': LaunchConfiguration('fps')
            }]
        )
    ])
