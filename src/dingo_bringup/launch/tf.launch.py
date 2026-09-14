"""Static transforms for the sensors mounted on the Dingo.

The Clearpath platform publishes its TF topics in the robot namespace
(``/dd100_10000002/tf`` and ``/dd100_10000002/tf_static``).  The mounted
sensors must publish into the same TF namespace so that SLAM and Nav2 can
resolve ``base_link -> laser`` and ``base_link -> camera_link``.

The initial mounting values are intentionally kept in sensor_profile.yaml.
They are a provisional installation calibration based on the supplied
measurements/photos and must be refined if mapping shows a registration
error.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


ROBOT_TF_STATIC_TOPIC = '/dd100_10000002/tf_static'


def _mounting(profile, sensor_name):
    mounting = profile['sensors'][sensor_name]['mounting']
    xyz = mounting['xyz_m']
    rpy = mounting['rpy_rad']
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f'{sensor_name} mounting must contain 3 xyz and 3 rpy values')
    if any(value is None for value in (*xyz, *rpy)):
        raise ValueError(f'{sensor_name} mounting calibration still contains null values')
    return mounting['base_frame'], xyz, rpy


def _static_transform(name, parent, child, xyz, rpy):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        output='screen',
        arguments=[
            '--x', str(xyz[0]),
            '--y', str(xyz[1]),
            '--z', str(xyz[2]),
            '--roll', str(rpy[0]),
            '--pitch', str(rpy[1]),
            '--yaw', str(rpy[2]),
            '--frame-id', parent,
            '--child-frame-id', child,
        ],
        # The platform and Clearpath navigation demos remap TF into this
        # topic.  Keep the sensor transforms there as well.
        remappings=[('/tf_static', ROBOT_TF_STATIC_TOPIC)],
    )


def generate_launch_description():
    profile_path = Path(
        get_package_share_directory('dingo_bringup'),
        'config',
        'sensor_profile.yaml',
    )
    with profile_path.open(encoding='utf-8') as profile_file:
        profile = yaml.safe_load(profile_file)

    lidar_parent, lidar_xyz, lidar_rpy = _mounting(profile, 'lidar')
    camera_parent, camera_xyz, camera_rpy = _mounting(profile, 'camera')

    return LaunchDescription([
        _static_transform(
            'dingo_lidar_tf',
            lidar_parent,
            profile['sensors']['lidar']['ros']['frame'],
            lidar_xyz,
            lidar_rpy,
        ),
        _static_transform(
            'dingo_camera_tf',
            camera_parent,
            profile['sensors']['camera']['ros']['camera_link_frame'],
            camera_xyz,
            camera_rpy,
        ),
    ])
