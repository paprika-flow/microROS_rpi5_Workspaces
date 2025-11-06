# my_feature_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the LiDAR launch file
    sllidar_launch_file = os.path.join(
        get_package_share_directory('sllidar_ros2'),
        'launch',
        'sllidar_launch.py'
    )

    return LaunchDescription([
        # Start LiDAR automatically
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sllidar_launch_file),
            launch_arguments={
                'serial_port': '/dev/ttyUSB1', #CHANGE THIS TO YOUR LIDAR PORT
                'serial_baudrate': '115200'
            }.items()
        )
    ])
