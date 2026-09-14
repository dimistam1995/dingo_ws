from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='camera',
            name='camera',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                # Use the confirmed D455, even if another RealSense is later
                # connected to the computer.
                'serial_no': '151422250402',
                'enable_color': True,
                'enable_depth': True,
                'enable_sync': True,
                'rgb_camera.color_profile': '640,480,15',
                'depth_module.depth_profile': '640,480,15',
                # The application consumes RGB-D, not the raw stereo IR
                # images.  Disabling those output streams removes the
                # incomplete USB IR frames seen on this host while leaving
                # hardware depth enabled.
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth.enable': True,
                'pointcloud.enable': False,
                'enable_gyro': False,
                'enable_accel': False,
                'publish_tf': True,
                'tf_publish_rate': 0.0,
                'tf_prefix': '',
                'base_frame_id': 'link',
            }],
            arguments=['--ros-args', '--log-level', 'info'],
            # RealSense uses an absolute /tf_static publisher.  Put its
            # internal camera_link -> optical-frame tree beside the two
            # mounted-sensor transforms above.
            remappings=[
                ('/tf', '/dd100_10000002/tf'),
                ('/tf_static', '/dd100_10000002/tf_static'),
            ],
        ),
    ])
