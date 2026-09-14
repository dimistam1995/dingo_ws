"""One-command sensor stack for the clean Dingo workspace.

The Clearpath base platform and the Dingo Dashboard are managed separately:
the former by Clearpath system services and the latter by
dingo-dashboard.service. This launch owns only the mounted Hokuyo and D455,
so there are no duplicate platform, dashboard, joystick or camera processes.
"""

import fcntl
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


_SENSOR_LOCK_FD = None


def _acquire_sensor_lock():
    """Allow exactly one mounted-sensor launch on this computer.

    The systemd unit is the normal owner, but this also protects against a
    second manual ``ros2 launch ... stack.launch.py`` from another terminal.
    The advisory lock is released automatically when the launch process exits.
    """
    global _SENSOR_LOCK_FD
    if _SENSOR_LOCK_FD is not None:
        return

    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if not runtime_dir:
        runtime_dir = f'/run/user/{os.getuid()}'
    lock_dir = Path(runtime_dir)
    if not lock_dir.is_dir():
        lock_dir = Path('/tmp')
    lock_path = lock_dir / 'dingo-mounted-sensors.lock'
    lock_fd = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_fd.close()
        raise RuntimeError(
            'Το Dingo sensor stack τρέχει ήδη. '
            'Σταμάτησε πρώτα το dingo-sensors.service ή το άλλο ros2 launch.'
        ) from exc
    _SENSOR_LOCK_FD = lock_fd


def generate_launch_description():
    _acquire_sensor_lock()
    package_share = FindPackageShare('dingo_bringup')

    tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share, 'launch', 'tf.launch.py'
        ])),
    )

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share, 'launch', 'lidar.launch.py'
        ])),
        launch_arguments={
            'hokuyo_ip': LaunchConfiguration('hokuyo_ip'),
            'hokuyo_port': LaunchConfiguration('hokuyo_port'),
            'laser_frame': LaunchConfiguration('laser_frame'),
        }.items(),
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share, 'launch', 'camera.launch.py'
        ])),
        condition=IfCondition(LaunchConfiguration('start_camera')),
    )

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
        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Start the mounted RealSense D455 driver',
        ),
        tf,
        lidar,
        camera,
    ])
