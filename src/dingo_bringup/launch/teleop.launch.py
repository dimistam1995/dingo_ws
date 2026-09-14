from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('dingo_bringup'), 'config', 'teleop_twist_joy.yaml'
    ])
    return LaunchDescription([
        Node(package='joy', executable='joy_node', name='joy_node', output='screen'),
        Node(
            package='teleop_twist_joy', executable='teleop_node',
            name='teleop_twist_joy', parameters=[params], output='screen',
            remappings=[('/cmd_vel', '/dd100_10000002/cmd_vel')],
        ),
    ])
