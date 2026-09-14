from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'scan_topic', default_value='/scan',
            description='Filtered 2D laser topic used by SLAM',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('clearpath_nav2_demos'), 'launch', 'slam.launch.py'
            ])),
            launch_arguments={
                'setup_path': '/etc/clearpath',
                'use_sim_time': 'false',
                'scan_topic': LaunchConfiguration('scan_topic'),
            }.items(),
        )
    ])
