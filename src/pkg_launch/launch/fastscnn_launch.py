from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pkg_FastSCNN',
            executable='fast_scnn_node',
            name='fast_scnn_node',
            parameters=[
                {'device': 'cuda'},    # or 'cpu'
                {'weights_path': '/home/clement_workspace/src/pkg_FastSCNN/weights/fast_scnn_citys.pth'},
                {'save_dir': '/home/clement_workspace/src/pkg_FastSCNN/seg_output'}
            ]
        )
    ])
