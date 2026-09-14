"""Open the Dingo RViz view with the namespaced Clearpath TF topics."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare('dingo_bringup'), 'config', 'dingo.rviz']
    )
    return LaunchDescription(
        [
            Node(
                package='rviz2',
                executable='rviz2',
                name='dingo_rviz',
                output='screen',
                arguments=[
                    '-d',
                    config,
                    '--ros-args',
                    '-r',
                    '/tf:=/dd100_10000002/tf',
                    '-r',
                    '/tf_static:=/dd100_10000002/tf_static',
                ],
            )
        ]
    )
