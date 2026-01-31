#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # ----------------------------------------
    # 1. Include Astra Camera Launch File
    # ----------------------------------------
    pkg_launch_share = get_package_share_directory('pkg_launch')
    astra_launch_path = os.path.join(pkg_launch_share, 'launch', 'astra_camera_launch.py')

    astra_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(astra_launch_path)
    )

    # ----------------------------------------
    # 2. Sidewalk Follower Node
    # ----------------------------------------
    follower_node = Node(
        package='pkg_navigation',
        executable='sidewalk_follower',
        name='sidewalk_follower',
        output='screen'
    )

    return LaunchDescription([
        astra_camera,
        follower_node
    ])
