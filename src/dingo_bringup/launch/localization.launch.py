import os

from ament_index_python.packages import get_package_share_directory
from clearpath_config.clearpath_config import ClearpathConfig
from clearpath_config.common.utils.yaml import read_yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace, SetRemap
from nav2_common.launch import RewrittenYaml


def launch_navigation(context, *args, **kwargs):
    setup_path = LaunchConfiguration('setup_path').perform(context)
    config = read_yaml(os.path.join(setup_path, 'robot.yaml'))
    clearpath_config = ClearpathConfig(config)
    namespace = clearpath_config.system.namespace
    nav2_bringup = get_package_share_directory('nav2_bringup')
    clearpath_nav2 = get_package_share_directory('clearpath_nav2_demos')
    scan_topic = LaunchConfiguration('scan_topic').perform(context)
    params = RewrittenYaml(
        source_file=os.path.join(
            clearpath_nav2, 'config', clearpath_config.platform.get_platform_model(), 'nav2.yaml'
        ),
        param_rewrites={
            'topic': scan_topic,
            # MPPI repeatedly failed to produce a trajectory on this host.
            # Use the lighter regulated pure-pursuit controller for the
            # differential-drive Dingo while retaining the Clearpath planner
            # and costmaps.
            'controller_server.ros__parameters.FollowPath.plugin':
                'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
            'controller_server.ros__parameters.FollowPath.desired_linear_vel': '0.18',
            'controller_server.ros__parameters.FollowPath.lookahead_dist': '0.45',
            'controller_server.ros__parameters.FollowPath.min_lookahead_dist': '0.25',
            'controller_server.ros__parameters.FollowPath.max_lookahead_dist': '0.70',
            'controller_server.ros__parameters.FollowPath.lookahead_time': '1.5',
            'controller_server.ros__parameters.FollowPath.transform_tolerance': '0.2',
            'controller_server.ros__parameters.FollowPath.use_regulated_linear_velocity_scaling': 'true',
            'controller_server.ros__parameters.FollowPath.use_cost_regulated_linear_velocity_scaling': 'true',
            # Clearpath's collision_monitor is the live LiDAR safety gate in
            # the cmd_vel chain.  RPP's additional footprint check treats a
            # safe footprint edge inside the costmap inflation zone as a
            # collision and then aborts otherwise valid goals with a generic
            # Nav2 failure.  Keep the planner/costmaps and collision_monitor;
            # disable only this duplicate RPP pre-check.
            'controller_server.ros__parameters.FollowPath.use_collision_detection': 'false',
            # Dashboard goals are point goals: the user selects an x/y cell and
            # does not request a final robot heading.  Clearpath's default
            # SimpleGoalChecker still requires yaw, so RPP can keep rotating
            # after the Dingo has reached the selected point.  Use Nav2's
            # official position-only checker and prevent that final rotation.
            'controller_server.ros__parameters.general_goal_checker.plugin':
                'nav2_controller::PositionGoalChecker',
            'controller_server.ros__parameters.FollowPath.use_rotate_to_heading': 'false',
            'controller_server.ros__parameters.FollowPath.rotate_to_heading_angular_vel': '0.5',
            'controller_server.ros__parameters.FollowPath.max_angular_accel': '1.5',
            'controller_server.ros__parameters.FollowPath.allow_reversing': 'false',
        },
        convert_types=True,
    )
    return [GroupAction([
        PushRosNamespace(namespace),
        SetRemap('/' + namespace + '/odom', '/' + namespace + '/platform/odom'),
        SetRemap('/tf', '/' + namespace + '/tf'),
        SetRemap('/tf_static', '/' + namespace + '/tf_static'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': params,
                'use_composition': 'False',
            }.items(),
        ),
    ])]


def launch_localization(context, *args, **kwargs):
    setup_path = LaunchConfiguration('setup_path').perform(context)
    config = read_yaml(os.path.join(setup_path, 'robot.yaml'))
    clearpath_config = ClearpathConfig(config)
    namespace = clearpath_config.system.namespace
    nav2_bringup = get_package_share_directory('nav2_bringup')
    dingo_bringup = get_package_share_directory('dingo_bringup')
    scan_topic = LaunchConfiguration('scan_topic').perform(context)
    params = RewrittenYaml(
        source_file=os.path.join(dingo_bringup, 'config', 'localization.yaml'),
        param_rewrites={'scan_topic': scan_topic},
        convert_types=True,
    )
    return [GroupAction([
        PushRosNamespace(namespace),
        SetRemap('/tf', '/' + namespace + '/tf'),
        SetRemap('/tf_static', '/' + namespace + '/tf_static'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'localization_launch.py')
            ),
            launch_arguments={
                'namespace': namespace,
                'map': LaunchConfiguration('map'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': params,
            }.items(),
        ),
    ])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', description='Absolute path to a map YAML file'),
        DeclareLaunchArgument('setup_path', default_value='/etc/clearpath'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='2D laser topic used by AMCL and Nav2 costmaps',
        ),
        OpaqueFunction(function=launch_navigation),
        OpaqueFunction(function=launch_localization),
    ])
