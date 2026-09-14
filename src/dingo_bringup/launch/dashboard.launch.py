from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'host', default_value='127.0.0.1',
            description='Interface address for the dashboard; use the Tailscale IP for remote access',
        ),
        Node(
            package='dingo_bringup',
            executable='dingo_dashboard.py',
            name='dingo_dashboard',
            output='screen',
            parameters=[{
                'port': 8090,
                'host': LaunchConfiguration('host'),
                'robot_namespace': 'dd100_10000002',
                'scan_topic': '/scan',
                'map_topic': '/dd100_10000002/map',
                'auto_start_map': 'dingo_map',
                'auto_localize': True,
                'auto_relocalize': True,
            }],
        ),
    ])
