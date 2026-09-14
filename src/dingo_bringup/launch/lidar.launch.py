"""Clean Hokuyo UTM-30LX-EW bringup for the Dingo workspace."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'hokuyo_ip',
            default_value='192.168.0.10',
            description='IPv4 address of the Hokuyo UTM-30LX-EW',
        ),
        DeclareLaunchArgument(
            'hokuyo_port',
            default_value='10940',
            description='SCIP TCP port of the Hokuyo',
        ),
        DeclareLaunchArgument(
            'laser_frame',
            default_value='laser',
            description='ROS frame attached to the LiDAR scan',
        ),
        Node(
            package='urg_node',
            executable='urg_node_driver',
            name='hokuyo_utm30lx_ew',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                # UTM-30LX-EW: 270 degrees, ±135 degrees.
                'angle_min': -2.356194490192345,
                'angle_max': 2.356194490192345,
                'ip_address': LaunchConfiguration('hokuyo_ip'),
                'ip_port': ParameterValue(
                    LaunchConfiguration('hokuyo_port'), value_type=int
                ),
                'laser_frame_id': LaunchConfiguration('laser_frame'),
                'calibrate_time': False,
                'default_user_latency': 0.0,
                'diagnostics_tolerance': 0.05,
                'diagnostics_window_time': 5.0,
                'error_limit': 4,
                'get_detailed_status': False,
                'publish_intensity': False,
                'publish_multiecho': False,
                'cluster': 1,
                # 40 Hz sensor -> 20 Hz output.  This keeps AMCL and the
                # dashboard supplied with enough scans during a rotation.
                'skip': 1,
            }],
            remappings=[('scan', '/scan')],
        ),
    ])
