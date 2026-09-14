#!/usr/bin/env python3
"""Fresh Dingo dashboard: mapping, perception, status and deliberate teleop."""
import json
import base64
import fcntl
import hashlib
import math
import mimetypes
import os
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from ament_index_python.packages import get_package_share_directory
from clearpath_platform_msgs.msg import Power, Temperature
from cv_bridge import CvBridge, CvBridgeError
from foxglove_msgs.msg import RawAudio
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from lifecycle_msgs.msg import State as LifecycleState, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from nav2_msgs.action import ComputePathToPose, DriveOnHeading, NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, CompressedImage, Image, Imu, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException


_DASHBOARD_LOCK_FD = None


def acquire_dashboard_lock():
    """Prevent a second Dashboard process from sharing the ROS graph/HTTP port."""
    global _DASHBOARD_LOCK_FD
    if _DASHBOARD_LOCK_FD is not None:
        return

    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if not runtime_dir:
        runtime_dir = f'/run/user/{os.getuid()}'
    lock_dir = Path(runtime_dir)
    if not lock_dir.is_dir():
        lock_dir = Path('/tmp')
    lock_override = os.environ.get('DINGO_DASHBOARD_LOCK_FILE', '').strip()
    lock_path = Path(lock_override).expanduser() if lock_override else (
        lock_dir / 'dingo-dashboard.lock'
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError(
            'Το Dingo Dashboard τρέχει ήδη. '
            'Χρησιμοποίησε το υπάρχον http://localhost:8090.'
        ) from exc
    _DASHBOARD_LOCK_FD = lock_fd

try:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
except ImportError:  # The official Nav2 path still works without the optional matcher.
    np = None
    distance_transform_edt = None

HTML = (Path(get_package_share_directory('dingo_bringup')) / 'web' / 'index.html').read_text(
    encoding='utf-8'
)


class DashboardNode(Node):
    DEFAULT_SETTINGS = {
        'max_linear_speed': 0.12,
        'max_angular_speed': 0.8,
    }

    # The assistant may control only these known user services.  The service
    # name is always resolved against this tuple before it reaches systemctl;
    # arbitrary commands, paths, shell snippets and unknown services never
    # cross the Tool Gateway boundary.
    TOOL_SERVICE_ALLOWLIST = (
        'dingo-voice.service',
        'dingo-local-llm.service',
        'dingo-dashboard.service',
        'dingo-sensors.service',
        'dingo-object-detector.service',
        'dingo-face-recognition.service',
        'dingo-mic-array.service',
    )
    TOOL_SERVICE_LABELS = {
        'dingo-voice.service': 'φωνητικός βοηθός / STT / TTS',
        'dingo-local-llm.service': 'τοπικό Qwen / FastFlowLM',
        'dingo-dashboard.service': 'Dashboard',
        'dingo-sensors.service': 'αισθητήρες / LiDAR / MCU',
        'dingo-object-detector.service': 'YOLO object detector',
        'dingo-face-recognition.service': 'αναγνώριση προσώπου',
        'dingo-mic-array.service': 'ReSpeaker mic array',
    }
    TOOL_TARGETS = {
        'service', 'camera', 'mapping', 'navigation', 'localization', 'rviz', 'map',
    }
    TOOL_OPERATIONS = {
        'status', 'start', 'stop', 'restart', 'cancel', 'global_localization', 'save',
    }

    # YOLO uses language-independent COCO class ids.  These are the Greek
    # words shown in the Dashboard and returned by the voice assistant.
    VISION_GREEK_LABELS = {
        'person': ('άνθρωπο', 'ανθρώπους'),
        'bicycle': ('ποδήλατο', 'ποδήλατα'),
        'car': ('αυτοκίνητο', 'αυτοκίνητα'),
        'motorcycle': ('μηχανάκι', 'μηχανάκια'),
        'airplane': ('αεροπλάνο', 'αεροπλάνα'),
        'bus': ('λεωφορείο', 'λεωφορεία'),
        'train': ('τρένο', 'τρένα'),
        'truck': ('φορτηγό', 'φορτηγά'),
        'boat': ('βάρκα', 'βάρκες'),
        'traffic light': ('φανάρι', 'φανάρια'),
        'fire hydrant': ('πυροσβεστικό κρουνό', 'πυροσβεστικούς κρουνούς'),
        'stop sign': ('πινακίδα STOP', 'πινακίδες STOP'),
        'parking meter': ('παρκόμετρο', 'παρκόμετρα'),
        'bench': ('παγκάκι', 'παγκάκια'),
        'bird': ('πουλί', 'πουλιά'),
        'cat': ('γάτα', 'γάτες'),
        'dog': ('σκύλο', 'σκύλους'),
        'horse': ('άλογο', 'άλογα'),
        'sheep': ('πρόβατο', 'πρόβατα'),
        'cow': ('αγελάδα', 'αγελάδες'),
        'elephant': ('ελέφαντα', 'ελέφαντες'),
        'bear': ('αρκούδα', 'αρκούδες'),
        'zebra': ('ζέβρα', 'ζέβρες'),
        'giraffe': ('καμηλοπάρδαλη', 'καμηλοπαρδάλεις'),
        'backpack': ('σακίδιο', 'σακίδια'),
        'umbrella': ('ομπρέλα', 'ομπρέλες'),
        'handbag': ('τσάντα', 'τσάντες'),
        'tie': ('γραβάτα', 'γραβάτες'),
        'suitcase': ('βαλίτσα', 'βαλίτσες'),
        'frisbee': ('φρίσμπι', 'φρίσμπι'),
        'skis': ('πέδιλο σκι', 'πέδιλα σκι'),
        'snowboard': ('σανίδα snowboard', 'σανίδες snowboard'),
        'sports ball': ('μπάλα', 'μπάλες'),
        'kite': ('χαρταετό', 'χαρταετούς'),
        'baseball bat': ('ρόπαλο μπέιζμπολ', 'ρόπαλα μπέιζμπολ'),
        'baseball glove': ('γάντι μπέιζμπολ', 'γάντια μπέιζμπολ'),
        'skateboard': ('σκέιτμπορντ', 'σκέιτμπορντ'),
        'surfboard': ('σανίδα σερφ', 'σανίδες σερφ'),
        'tennis racket': ('ρακέτα τένις', 'ρακέτες τένις'),
        'bottle': ('μπουκάλι', 'μπουκάλια'),
        'wine glass': ('ποτήρι κρασιού', 'ποτήρια κρασιού'),
        'cup': ('φλιτζάνι', 'φλιτζάνια'),
        'fork': ('πιρούνι', 'πιρούνια'),
        'knife': ('μαχαίρι', 'μαχαίρια'),
        'spoon': ('κουτάλι', 'κουτάλια'),
        'bowl': ('μπολ', 'μπολ'),
        'banana': ('μπανάνα', 'μπανάνες'),
        'apple': ('μήλο', 'μήλα'),
        'sandwich': ('σάντουιτς', 'σάντουιτς'),
        'orange': ('πορτοκάλι', 'πορτοκάλια'),
        'broccoli': ('μπρόκολο', 'μπρόκολα'),
        'carrot': ('καρότο', 'καρότα'),
        'hot dog': ('χοτ ντογκ', 'χοτ ντογκ'),
        'pizza': ('πίτσα', 'πίτσες'),
        'donut': ('ντόνατ', 'ντόνατ'),
        'cake': ('τούρτα', 'τούρτες'),
        'chair': ('καρέκλα', 'καρέκλες'),
        'couch': ('καναπέ', 'καναπέδες'),
        'potted plant': ('γλάστρα', 'γλάστρες'),
        'bed': ('κρεβάτι', 'κρεβάτια'),
        'dining table': ('τραπέζι', 'τραπέζια'),
        'toilet': ('τουαλέτα', 'τουαλέτες'),
        'tv': ('τηλεόραση', 'τηλεοράσεις'),
        'laptop': ('laptop', 'laptop'),
        'mouse': ('ποντίκι', 'ποντίκια'),
        'remote': ('τηλεχειριστήριο', 'τηλεχειριστήρια'),
        'keyboard': ('πληκτρολόγιο', 'πληκτρολόγια'),
        'cell phone': ('κινητό τηλέφωνο', 'κινητά τηλέφωνα'),
        'microwave': ('φούρνο μικροκυμάτων', 'φούρνους μικροκυμάτων'),
        'oven': ('φούρνο', 'φούρνους'),
        'toaster': ('τοστιέρα', 'τοστιέρες'),
        'sink': ('νεροχύτη', 'νεροχύτες'),
        'refrigerator': ('ψυγείο', 'ψυγεία'),
        'book': ('βιβλίο', 'βιβλία'),
        'clock': ('ρολόι', 'ρολόγια'),
        'vase': ('βάζο', 'βάζα'),
        'scissors': ('ψαλίδι', 'ψαλίδια'),
        'teddy bear': ('αρκουδάκι', 'αρκουδάκια'),
        'hair drier': ('πιστολάκι μαλλιών', 'πιστολάκια μαλλιών'),
        'toothbrush': ('οδοντόβουρτσα', 'οδοντόβουρτσες'),
    }

    # Clearpath's ROS 2 API is a namespaced ROS graph, not a separate HTTP
    # protocol.  Keep the documented interfaces here and merge them with the
    # live graph in /api/clearpath so the Dashboard exposes both the contract
    # and what this particular Dingo actually has online.
    CLEARPATH_TOPIC_CATALOG = (
        ('cmd_vel', 'geometry_msgs/msg/TwistStamped', 'velocity command', 'command', 'System Default'),
        ('diagnostics', 'diagnostics_msgs/msg/DiagnosticArray', 'diagnostic messages', 'read', 'System Default'),
        ('joy_teleop/cmd_vel', 'geometry_msgs/msg/TwistStamped', 'joystick velocity command', 'command', 'System Default'),
        ('joy_teleop/joy', 'sensor_msgs/msg/Joy', 'joystick state', 'read', 'System Default'),
        ('platform/bms/state', 'sensor_msgs/msg/BatteryState', 'battery state', 'read', 'Sensor Data'),
        ('platform/cmd_fans', 'clearpath_platform_msgs/msg/Fans', 'fan command where supported', 'command', 'System Default'),
        ('platform/cmd_lights', 'clearpath_platform_msgs/msg/Lights', 'lighting command where supported', 'command', 'System Default'),
        ('platform/cmd_vel', 'geometry_msgs/msg/TwistStamped', 'velocity after twist mux', 'read', 'System Default'),
        ('platform/dynamic_joint_states', 'control_msgs/msg/DynamicJointState', 'platform dynamic joints', 'read', 'System Default'),
        ('platform/emergency_stop', 'std_msgs/msg/Bool', 'emergency-stop state', 'read', 'Sensor Data'),
        ('platform/joint_states', 'sensor_msgs/msg/JointState', 'platform joint states', 'read', 'System Default'),
        ('platform/motors/cmd', 'clearpath_platform_msgs/msg/Drive', 'individual motor commands', 'read', 'Sensor Data'),
        ('platform/motors/feedback', 'clearpath_platform_msgs/msg/DriveFeedback', 'motor feedback', 'read', 'Sensor Data'),
        ('platform/motors/status', 'clearpath_motor_msgs/msg/PumaMultiStatus', 'motor status where supported', 'read', 'Sensor Data'),
        ('platform/odom', 'nav_msgs/msg/Odometry', 'wheel odometry', 'read', 'System Default'),
        ('platform/odom/filtered', 'nav_msgs/msg/Odometry', 'wheel odometry fused with IMU', 'read', 'System Default'),
        ('platform/wifi_connected', 'std_msgs/msg/Bool', 'Wi-Fi connected state', 'read', 'System Default'),
        ('platform/wifi_status', 'wireless_msgs/msg/Connection', 'Wi-Fi connection status', 'read', 'System Default'),
        ('platform/mcu/status', 'clearpath_platform_msgs/msg/Status', 'MCU status', 'read', 'Sensor Data'),
        ('platform/mcu/status/pinout', 'clearpath_platform_msgs/msg/PinoutState', 'MCU pinout state', 'read', 'Sensor Data'),
        ('platform/mcu/status/power', 'clearpath_platform_msgs/msg/Power', 'robot power status', 'read', 'Sensor Data'),
        ('platform/mcu/status/stop', 'clearpath_platform_msgs/msg/StopStatus', 'stop-loop status', 'read', 'Sensor Data'),
        ('platform/mcu/status/temperature', 'clearpath_platform_msgs/msg/Temperature', 'MCU temperatures', 'read', 'Sensor Data'),
        ('platform/display/status', 'clearpath_platform_msgs/msg/DisplayStatus', 'display status', 'read', 'Sensor Data'),
        ('robot_description', 'std_msgs/msg/String', 'robot description', 'read', 'Transient Local'),
        ('tf', 'tf2_msgs/msg/TFMessage', 'dynamic link transforms', 'read', 'System Default'),
        ('tf_static', 'tf2_msgs/msg/TFMessage', 'static link transforms', 'read', 'Transient Local'),
        ('twist_marker_server/cmd_vel', 'geometry_msgs/msg/TwistStamped', 'RViz twist command', 'command', 'System Default'),
        ('/rosout', 'rcl_interfaces/msg/Log', 'ROS logs', 'read', 'Transient Local'),
        ('sensors/lidar2d_#/scan', 'sensor_msgs/msg/LaserScan', '2D LiDAR scan pattern', 'read', 'System Default'),
        ('sensors/lidar3d_#/points', 'sensor_msgs/msg/PointCloud2', '3D LiDAR point cloud pattern', 'read', 'System Default'),
        ('sensors/lidar3d_#/scan', 'sensor_msgs/msg/LaserScan', '3D LiDAR scan pattern', 'read', 'System Default'),
        ('sensors/lidar3d_#/imu/data_raw', 'sensor_msgs/msg/Imu', '3D LiDAR IMU pattern', 'read', 'Sensor Data'),
        ('sensors/camera_#/color/image', 'sensor_msgs/msg/Image', 'camera RGB image pattern', 'read', 'System Default'),
        ('sensors/camera_#/color/camera_info', 'sensor_msgs/msg/CameraInfo', 'camera RGB info pattern', 'read', 'System Default'),
        ('sensors/camera_#/depth/image', 'sensor_msgs/msg/Image', 'camera depth image pattern', 'read', 'System Default'),
        ('sensors/camera_#/depth/camera_info', 'sensor_msgs/msg/CameraInfo', 'camera depth info pattern', 'read', 'System Default'),
        ('sensors/camera_#/points', 'sensor_msgs/msg/PointCloud2', 'camera point cloud pattern', 'read', 'System Default'),
        ('sensors/camera_#/imu/data_raw', 'sensor_msgs/msg/Imu', 'camera IMU pattern', 'read', 'Sensor Data'),
        ('sensors/imu_#/data_raw', 'sensor_msgs/msg/Imu', 'raw IMU pattern', 'read', 'Sensor Data'),
        ('sensors/imu_#/data', 'sensor_msgs/msg/Imu', 'filtered IMU pattern', 'read', 'System Default'),
        ('sensors/imu_#/mag', 'sensor_msgs/msg/MagneticField', 'magnetometer pattern', 'read', 'System Default'),
        ('sensors/ins_#/imu_0/data', 'sensor_msgs/msg/Imu', 'INS IMU pattern', 'read', 'Sensor Data'),
        ('sensors/ins_#/gps_0/fix', 'sensor_msgs/msg/NavSatFix', 'INS GPS 0 pattern', 'read', 'Sensor Data'),
        ('sensors/ins_#/gps_1/fix', 'sensor_msgs/msg/NavSatFix', 'INS GPS 1 pattern', 'read', 'Sensor Data'),
        ('sensors/ins_#/odom', 'nav_msgs/msg/Odometry', 'INS odometry pattern', 'read', 'Sensor Data'),
        ('sensors/gps_#/fix', 'sensor_msgs/msg/NavSatFix', 'GPS fix pattern', 'read', 'System Default'),
        ('scan', 'sensor_msgs/msg/LaserScan', 'mounted Hokuyo scan alias', 'read', 'Sensor Data'),
    )

    CLEARPATH_SERVICE_CATALOG = (
        ('platform/mcu/configure', 'clearpath_platform_msgs/srv/ConfigureMcu', 'configure MCU', 'command'),
        ('platform/mcu/clear_estop_needs_reset', 'std_srvs/srv/Empty', 'clear MCU estop reset state', 'command'),
        ('platform/pinout/aux_#', 'clearpath_platform_msgs/srv/SetPinout', 'set auxiliary pin pattern', 'command'),
        ('platform/pinout/gpo_#', 'clearpath_platform_msgs/srv/SetPinout', 'set GPIO pin pattern', 'command'),
        ('platform/pinout/user_pwr_ctrl', 'std_srvs/srv/SetBool', 'user power rail control', 'command'),
        ('reinitialize_global_localization', 'std_srvs/srv/Empty', 'AMCL global relocalization', 'command'),
        ('request_nomotion_update', 'std_srvs/srv/Empty', 'AMCL no-motion update', 'command'),
        ('set_initial_pose', 'nav2_msgs/srv/SetInitialPose', 'set AMCL initial pose', 'command'),
        ('map_server/load_map', 'nav2_msgs/srv/LoadMap', 'load a static map', 'command'),
        ('map_server/save_map', 'nav2_msgs/srv/SaveMap', 'save a static map', 'command'),
        ('global_costmap/clear_entirely_global_costmap', 'nav2_msgs/srv/ClearEntireCostmap', 'clear global costmap', 'command'),
        ('local_costmap/clear_entirely_local_costmap', 'nav2_msgs/srv/ClearEntireCostmap', 'clear local costmap', 'command'),
        ('controller_manager/list_controllers', 'controller_manager_msgs/srv/ListControllers', 'list platform controllers', 'read'),
        ('controller_manager/switch_controller', 'controller_manager_msgs/srv/SwitchController', 'switch platform controllers', 'command'),
    )

    CLEARPATH_ACTION_CATALOG = (
        ('navigate_to_pose', 'nav2_msgs/action/NavigateToPose', 'navigate to one pose'),
        ('navigate_through_poses', 'nav2_msgs/action/NavigateThroughPoses', 'navigate through poses'),
        ('compute_path_to_pose', 'nav2_msgs/action/ComputePathToPose', 'compute a path to a pose'),
        ('compute_path_through_poses', 'nav2_msgs/action/ComputePathThroughPoses', 'compute a path through poses'),
        ('follow_path', 'nav2_msgs/action/FollowPath', 'follow a path'),
        ('follow_waypoints', 'nav2_msgs/action/FollowWaypoints', 'follow waypoints'),
        ('spin', 'nav2_msgs/action/Spin', 'spin in place'),
        ('backup', 'nav2_msgs/action/BackUp', 'back up'),
        ('drive_on_heading', 'nav2_msgs/action/DriveOnHeading', 'drive on heading'),
        ('wait', 'nav2_msgs/action/Wait', 'wait'),
        ('assisted_teleop', 'nav2_msgs/action/AssistedTeleop', 'assisted teleoperation'),
        ('smooth_path', 'nav2_msgs/action/SmoothPath', 'smooth a path'),
        ('dock_robot', 'nav2_msgs/action/DockRobot', 'dock robot'),
        ('undock_robot', 'nav2_msgs/action/UndockRobot', 'undock robot'),
        ('follow_gps_waypoints', 'nav2_msgs/action/FollowGPSWaypoints', 'follow GPS waypoints'),
        ('compute_route', 'nav2_msgs/action/ComputeRoute', 'compute a route'),
        ('compute_and_track_route', 'nav2_msgs/action/ComputeAndTrackRoute', 'compute and track a route'),
    )

    def __init__(self):
        super().__init__('dingo_dashboard')
        self.robot_namespace = self.declare_parameter(
            'robot_namespace', 'dd100_10000002'
        ).value.strip('/')
        self.host = self.declare_parameter('host', '127.0.0.1').value
        self.scan_topic = self.declare_parameter(
            'scan_topic', '/scan'
        ).value
        self.map_topic = self.declare_parameter(
            'map_topic', '/dd100_10000002/map'
        ).value
        self.camera_topic = self.declare_parameter(
            'camera_topic', '/camera/camera/color/image_raw/compressed'
        ).value
        self.camera_raw_topic = self.declare_parameter(
            'camera_raw_topic', '/camera/camera/color/image_raw'
        ).value
        self.maps_dir = Path(
            self.declare_parameter(
                'maps_dir', str(Path.home() / 'dingo_ws' / 'maps')
            ).value
        ).expanduser()
        self.auto_start_map = str(
            self.declare_parameter('auto_start_map', 'dingo_map').value or ''
        ).strip()
        if self.auto_start_map.lower() in {'false', 'none', 'off', '0'}:
            self.auto_start_map = ''
        self.auto_localize = bool(
            self.declare_parameter('auto_localize', True).value
        )
        # Keep localization self-healing after startup as well.  AMCL can
        # retain a low-covariance pose after the Dingo is moved by hand or
        # after a stale map->odom transform survives a restart.  The
        # watchdog below only resets AMCL while the platform is stationary;
        # it never commands a search rotation or a navigation goal.
        self.auto_relocalize = bool(
            self.declare_parameter('auto_relocalize', True).value
        )
        # Some integration tests start Nav2 outside the Dashboard process
        # (for example a Gazebo launch).  Keep the normal Dashboard ownership
        # as the default, but allow an explicit external-stack mode so the
        # Dashboard can still send goals and draw paths without launching a
        # second Nav2 instance.
        self.external_navigation = bool(
            self.declare_parameter('external_navigation', False).value
        )
        self.rooms_file = Path(
            self.declare_parameter(
                'rooms_file',
                str(Path.home() / '.config' / 'dingo_dashboard' / 'rooms.json'),
            ).value
        ).expanduser()
        self.settings_file = Path(
            self.declare_parameter(
                'settings_file',
                str(Path.home() / '.config' / 'dingo_dashboard' / 'settings.json'),
            ).value
        ).expanduser()
        self.wake_training_dir = Path(
            self.declare_parameter(
                'wake_training_dir',
                str(Path.home() / 'dingo_ws' / 'training' / 'wake_word_dingo'),
            ).value
        ).expanduser()
        self.wake_training_seconds = max(
            1.8,
            min(
                4.0,
                float(self.declare_parameter('wake_training_seconds', 2.4).value or 2.4),
            ),
        )
        self.wake_training_dir.mkdir(parents=True, exist_ok=True)
        (self.wake_training_dir / 'real_wav').mkdir(parents=True, exist_ok=True)
        self.vision_llm_enabled = bool(
            self.declare_parameter('vision_llm_enabled', True).value
        )
        self.vision_llm_model = str(
            self.declare_parameter('vision_llm_model', 'gemini-2.5-flash').value
            or 'gemini-2.5-flash'
        )
        default_vision_key = os.environ.get(
            'GEMINI_API_KEY_FILE',
            str(Path.home() / '.config' / 'dingo_voice' / 'gemini_api_key'),
        )
        self.vision_llm_key_file = Path(
            self.declare_parameter('vision_llm_api_key_file', default_vision_key).value
        ).expanduser()
        self.vision_llm_timeout_s = float(
            self.declare_parameter('vision_llm_timeout_s', 35.0).value or 35.0
        )
        self.gemini_usage_file = Path(
            self.declare_parameter(
                'gemini_usage_file',
                str(Path.home() / '.config' / 'dingo_dashboard' / 'gemini_usage.json'),
            ).value
        ).expanduser()
        gemini_usage = self.load_gemini_usage()
        prefix = f'/{self.robot_namespace}' if self.robot_namespace else ''
        # Clearpath's BT joystick watchdog normally owns a very high-priority
        # twist_mux lock.  A DualSense can remain Bluetooth-connected while
        # asleep, which makes that watchdog report 0% quality and silently
        # block every Nav2 command.  Dashboard-owned autonomous actions may
        # temporarily lower only this lock; the physical E-stop and safety
        # locks remain at their normal priorities (255 and 254).
        self.autonomous_navigation_bypass_bt_quality = bool(
            self.declare_parameter(
                'autonomous_navigation_bypass_bt_quality', True
            ).value
        )
        self.bt_quality_normal_priority = max(
            0,
            min(
                255,
                int(
                    self.declare_parameter(
                        'bt_quality_normal_priority', 253
                    ).value
                    or 253
                ),
            ),
        )
        self.bt_quality_autonomous_priority = max(
            0,
            min(
                255,
                int(
                    self.declare_parameter(
                        'bt_quality_autonomous_priority', 0
                    ).value
                    or 0
                ),
            ),
        )
        self.twist_mux_node_name = f'{prefix}/twist_mux' or '/twist_mux'
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.autonomous_navigation_lock = threading.Lock()
        self.autonomous_navigation_active = False
        self.autonomous_navigation_priority = None
        self.autonomous_navigation_last_error = None
        self.autonomous_navigation_last_change_at = 0.0
        self.autonomous_navigation_restore_pending = bool(
            self.autonomous_navigation_bypass_bt_quality
        )
        self.autonomous_navigation_restore_attempts = 0
        self.autonomous_navigation_restore_max_attempts = 15
        self.autonomous_navigation_restore_next_at = 0.0
        self.wake_training_lock = threading.Lock()
        self.audio_monitor_lock = threading.Lock()
        self.audio_monitor_clients = set()
        self.wake_training_active = False
        self.wake_training_chunks = []
        self.wake_training_started_at = 0.0
        self.wake_training_phase = None
        self.wake_training_index = None
        self.wake_training_sample_rate = 16000
        self.wake_training_pipeline_thread = None
        self.wake_training = {
            'state': 'idle',
            'phase': None,
            'current_index': None,
            'elapsed_s': 0.0,
            'seconds': self.wake_training_seconds,
            'message': 'Πάτησε ένα κουμπί για να ξεκινήσεις εγγραφή.',
            'last_saved': None,
            'error': None,
            'pipeline_output': None,
        }
        self.tf_buffer = Buffer()
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            f'{prefix}/navigate_to_pose',
        )
        self.navigation_path_client = ActionClient(
            self,
            ComputePathToPose,
            f'{prefix}/compute_path_to_pose',
        )
        self.spin_client = ActionClient(
            self,
            Spin,
            f'{prefix}/spin',
        )
        self.drive_heading_client = ActionClient(
            self,
            DriveOnHeading,
            f'{prefix}/drive_on_heading',
        )
        self.state = {
            'connected': False,
            'battery': None,
            'power': None,
            'temperature': None,
            'vision': {
                'state': 'offline',
                'source': 'YOLO11n · CPU',
                'summary_el': 'Η αναγνώριση αντικειμένων δεν είναι ενεργή.',
                'objects': [],
                'last_update': None,
                'detail_state': 'idle',
                'detail_question': None,
                'detail_answer': None,
                'detail_updated': None,
            },
            'face': {
                'state': 'offline',
                'source': 'YuNet + SFace',
                'message_el': 'Δεν έχει ξεκινήσει η αναγνώριση προσώπου.',
                'faces': [],
                'enrolled': [],
                'enrollment': None,
                'last_update': None,
            },
            'speaker': {
                'state': 'offline',
                'current_name': None,
                'current_score': None,
                'last_update': None,
                'enrolled': [],
                'enrollment': None,
                'detail': 'Δεν έχει ξεκινήσει η αναγνώριση φωνής.',
            },
            'follow': {
                'active': False,
                'state': 'idle',
                'detail': 'Η λειτουργία follow είναι κλειστή.',
                'last_seen': None,
                'target': None,
            },
            'gemini': gemini_usage,
            'odom': None,
            'scan': None,
            'imu': {
                'available': False,
                'frame': None,
                'orientation': None,
                'angular_velocity': None,
                'linear_acceleration': None,
                'rate_hz': None,
                'last_update': None,
                'source': f'/{self.robot_namespace}/sensors/imu_0/data',
            },
            'localization': {
                'amcl_available': False,
                'localized': False,
                'pose': None,
                'covariance': None,
                'tf_map_base_link': None,
                'tf_ok': False,
                'last_update': None,
            },
            'camera': {
                'available': False,
                'topic': self.camera_topic,
                'frame': None,
                'last_frame': None,
            },
            'emergency_stop': None,
            'safety_stop': None,
            'voice': {
                'state': 'offline',
                'detail': 'Δεν έχει ξεκινήσει το voice assistant',
                'stt_model': 'large-v3-turbo',
                'stt_backend': 'vulkan',
                'stt_active_backend': 'vulkan',
                'stt_provider': 'whisper.cpp/Vulkan GPU',
                'stt_fallback_model': 'large-v3-turbo',
                'stt_fallback_reason': None,
                'llm_provider': 'flm',
                'llm_model': 'qwen3.5:9b',
                'llm_tool_calling': 'json_fallback',
                'llm_fallback_active': False,
                'llm_fallback_reason': None,
                'last_transcript': None,
                'transcript_seq': 0,
                'last_reply': None,
                'last_reply_ok': None,
                'last_reply_action': None,
                'reply_seq': 0,
                'last_text_command': None,
                'wake_ready': False,
                'wake_model': 'Alexa',
                'wake_threshold': None,
                'wake_score': 0.0,
                'wake_detected_at': None,
                'audio_rms': 0.0,
                'audio_peak': 0.0,
                'last_audio_at': None,
                'wake_error': None,
                'direction_available': False,
                'direction_enabled': True,
                'direction_angle': None,
                'direction_speech': False,
                'direction_at': None,
                'direction_error': None,
                'led_state': 'unknown',
                'voice_features': {
                    'direction_of_arrival': False,
                    'hardware_vad': False,
                    'beamforming': False,
                    'noise_reduction': False,
                    'echo_cancellation': False,
                    'automatic_gain': False,
                },
            },
            'tool_gateway': {
                'enabled': True,
                'mode': 'allowlist',
                'last_action': None,
                'last_target': None,
                'last_operation': None,
                'last_service': None,
                'last_ok': None,
                'last_detail': None,
                'last_at': None,
                'protections': [
                    'allow-list',
                    'emergency stop',
                    'Nav2/collision checks',
                    'timeouts',
                    'audit log',
                ],
            },
        }
        self.map_state = None
        self.camera_bytes = None
        self.camera_mime = 'image/jpeg'
        self.vision_query_lock = threading.Lock()
        self.vision_query_active = False
        self.vision_query_started_at = 0.0
        self.speaker_query_pending_since = None
        self.follow_active = False
        self.follow_started_at = 0.0
        self.follow_until = 0.0
        self.follow_target_x = None
        self.follow_missing_announced = False
        self.follow_timeout_s = 30.0
        # A detector needs a few frames after the explicit start command.  Do
        # not turn a normal camera/YOLO startup delay into an immediate
        # follow-stop; while searching the commanded velocity is always zero.
        self.follow_acquire_timeout_s = 4.0
        self.follow_lost_timeout_s = 1.2
        self.bridge = CvBridge()
        self.rooms = self.load_rooms()
        self.settings = self.load_settings()
        self.mapping_process = None
        self.camera_process = None
        self.rviz_process = None
        self.map_save_process = None
        self.navigation_process = None
        # A power-off request is one-shot.  This prevents double clicks or
        # repeated HTTP requests from queueing multiple shutdown commands.
        self.poweroff_requested = False
        self.auto_start_attempted = False
        self.auto_start_last_log_at = 0.0
        self.auto_localize_attempted = False
        self.auto_localize_retry_at = 0.0
        self.auto_localize_last_log_at = 0.0
        self.auto_localize_retry_count = 0
        self.auto_localize_max_retries = 3
        self.localization_watchdog_invalid_since = 0.0
        self.localization_watchdog_last_reset_at = 0.0
        self.localization_watchdog_confirm_s = 3.0
        self.localization_watchdog_cooldown_s = 45.0
        self.navigation_goal_handle = None
        self.navigation_result_future = None
        self.navigation_status = 'idle'
        self.navigation_map = None
        self.navigation_initial_pose = None
        self.navigation_initial_pose_sent_at = 0.0
        # Initial pose is only a DDS delivery aid.  Once AMCL has accepted
        # it, stop replaying it so a moving robot is tracked by AMCL's
        # map->odom transform instead of being pulled back to the seed pose.
        self.initial_pose_republish_s = 20.0
        self.navigation_pose_initialized = False
        self.navigation_amcl_pose = None
        self.navigation_amcl_covariance = None
        self.navigation_amcl_received_at = 0.0
        self.navigation_goal = None
        self.navigation_feedback = None
        self.navigation_path_goal_handle = None
        self.navigation_path_result_future = None
        self.navigation_path = None
        self.navigation_path_status = 'idle'
        self.navigation_path_error = None
        self.navigation_path_request_id = 0
        self.spin_goal_handle = None
        self.spin_result_future = None
        self.spin_status = 'idle'
        self.spin_feedback = None
        self.spin_degrees = None
        self.drive_heading_goal_handle = None
        self.drive_heading_result_future = None
        self.drive_heading_status = 'idle'
        self.drive_heading_feedback = None
        self.drive_heading_distance_m = None
        self.patrol_active = False
        self.patrol_queue = []
        self.patrol_current = None
        self.patrol_total = 0
        self.patrol_completed = 0
        self.localization_service_client = self.create_client(
            Empty,
            f'{prefix}/reinitialize_global_localization',
        )
        self.nomotion_update_client = self.create_client(
            Empty,
            f'{prefix}/request_nomotion_update',
        )
        self.localization_lifecycle_nodes = ('map_server', 'amcl')
        self.localization_lifecycle_state_clients = {
            name: self.create_client(GetState, f'{prefix}/{name}/get_state')
            for name in self.localization_lifecycle_nodes
        }
        self.localization_lifecycle_change_clients = {
            name: self.create_client(ChangeState, f'{prefix}/{name}/change_state')
            for name in self.localization_lifecycle_nodes
        }
        self.localization_lifecycle_future = None
        self.localization_lifecycle_node = None
        self.localization_lifecycle_operation = None
        self.localization_lifecycle_states = {
            name: None for name in self.localization_lifecycle_nodes
        }
        self.localization_lifecycle_done = False
        # The Clearpath/Nav2 lifecycle manager can race the platform's first
        # odometry/cmd_vel endpoints on a cold start and leave the navigation
        # nodes configured-but-inactive.  Keep a separate recovery state
        # machine for those nodes; it never sends a NavigateToPose goal.
        self.navigation_lifecycle_nodes = (
            'controller_server',
            'smoother_server',
            'planner_server',
            'route_server',
            'behavior_server',
            'velocity_smoother',
            'collision_monitor',
            'bt_navigator',
            'waypoint_follower',
            'docking_server',
        )
        self.navigation_lifecycle_state_clients = {
            name: self.create_client(GetState, f'{prefix}/{name}/get_state')
            for name in self.navigation_lifecycle_nodes
        }
        self.navigation_lifecycle_change_clients = {
            name: self.create_client(ChangeState, f'{prefix}/{name}/change_state')
            for name in self.navigation_lifecycle_nodes
        }
        self.navigation_lifecycle_future = None
        self.navigation_lifecycle_node = None
        self.navigation_lifecycle_operation = None
        self.navigation_lifecycle_states = {
            name: None for name in self.navigation_lifecycle_nodes
        }
        self.navigation_lifecycle_done = False
        self.navigation_lifecycle_not_before = 0.0
        self.localization_service_future = None
        self.nomotion_update_future = None
        self.localization_last_nomotion_update_at = 0.0
        self.localization_search_active = False
        self.localization_scan_match_active = False
        self.localization_search_started_at = 0.0
        self.localization_good_count = 0
        self.localization_good_required = 5
        # A strong stationary LiDAR match already gives AMCL a very good
        # initial hypothesis.  Three consecutive low-covariance AMCL updates
        # are enough to confirm that hypothesis without making the user wait
        # for the full global-search settling window.
        self.localization_scan_match_good_required = 3
        self.localization_last_pose = None
        self.localization_global_service_response = False
        self.localization_last_global_reset = False
        self.localization_scan_match_attempted = False
        self.localization_allow_motion = False
        self.localization_method = None
        self.localization_fallback_after_s = 5.0
        # The fallback matcher may need a few seconds on the small computer;
        # leave enough time for AMCL to accept its seed before declaring a
        # failed search.
        self.localization_timeout_s = 75.0
        self.localization_angular_speed = 0.16
        # Never draw a scan against a transform that is materially older than
        # the scan itself.  A latest-transform fallback is useful during
        # startup, but while rotating it can make a perfectly good LiDAR
        # appear to jump to another wall.
        self.scan_transform_max_age_s = 0.10
        self.tf_last_update = 0.0
        self.imu_last_update = 0.0
        self.imu_previous_update = 0.0
        self.clearpath_api_cache = None
        self.clearpath_api_cache_at = 0.0
        self.last_map_save_name = None
        self.last_drive_at = 0.0
        self.drive_timeout_s = 0.45

        self.pub = self.create_publisher(TwistStamped, f'{prefix}/cmd_vel', 10)
        self.voice_text_pub = self.create_publisher(
            String, f'{prefix}/voice/text_command', 10
        )
        # Audio captured by a phone/browser is injected into the voice node's
        # normal wake-word path.  It is a separate topic so the physical
        # ReSpeaker stream remains available and is never opened twice.
        self.remote_audio_pub = self.create_publisher(
            RawAudio, f'{prefix}/voice/remote_audio', 10
        )
        self.voice_reply_pub = self.create_publisher(
            String, f'{prefix}/voice/reply', 10
        )
        self.voice_provider_pub = self.create_publisher(
            String, f'{prefix}/voice/provider_command', 10
        )
        self.voice_direction_enable_pub = self.create_publisher(
            Bool,
            f'{prefix}/voice/direction_enable',
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        # Keep the DoA indicator enabled by default so the ReSpeaker shows a
        # green pointer while speech is detected.  The transient-local
        # publisher also gives a later-starting DoA node the current choice
        # immediately; the Dashboard toggle can still turn it off.
        self.voice_direction_enable_pub.publish(Bool(data=True))
        self.face_command_pub = self.create_publisher(
            String, f'{prefix}/face/command', 10
        )
        self.speaker_command_pub = self.create_publisher(
            String, f'{prefix}/voice/speaker_command', 10
        )
        self.create_subscription(
            BatteryState,
            f'{prefix}/platform/bms/state',
            self.battery,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Power,
            f'{prefix}/platform/mcu/status/power',
            self.power,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Temperature,
            f'{prefix}/platform/mcu/status/temperature',
            self.mcu_temperature,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            f'{prefix}/detected_objects',
            self.vision_detections,
            10,
        )
        self.create_subscription(
            Odometry,
            f'{prefix}/platform/odom',
            self.odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            f'{prefix}/sensors/imu_0/data',
            self.imu_data,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            f'{prefix}/platform/emergency_stop',
            self.emergency_stop,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            f'{prefix}/platform/safety_stop',
            self.safety_stop,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/status',
            self.voice_status,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/direction',
            self.voice_direction,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/transcript',
            self.voice_transcript,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/command',
            self.voice_command,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/reply',
            self.voice_reply,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/face/state',
            self.face_state,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/speaker',
            self.speaker_state,
            10,
        )
        self.create_subscription(
            String,
            f'{prefix}/voice/gemini_usage',
            self.gemini_usage,
            10,
        )
        # The voice service already publishes the ReSpeaker stream as
        # RawAudio.  The Dashboard subscribes only while the user records a
        # training clip, so this never competes with the assistant's audio
        # processing and avoids opening the USB device a second time.
        self.create_subscription(
            RawAudio,
            f'{prefix}/voice/audio',
            self.wake_training_audio,
            10,
        )
        path_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            NavPath,
            f'{prefix}/plan',
            self.navigation_plan,
            path_qos,
        )
        self.create_subscription(
            NavPath,
            f'{prefix}/plan_smoothed',
            self.navigation_plan_smoothed,
            path_qos,
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan, qos_profile_sensor_data
        )
        tf_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        tf_static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            TFMessage,
            f'{prefix}/tf',
            self.tf_dynamic,
            tf_qos,
        )
        self.create_subscription(
            TFMessage,
            f'{prefix}/tf_static',
            self.tf_static,
            tf_static_qos,
        )
        if self.camera_topic:
            self.create_subscription(
                CompressedImage,
                self.camera_topic,
                self.camera_compressed,
                qos_profile_sensor_data,
            )
        # Prefer the driver's compressed transport.  Converting every raw
        # frame to JPEG in this single-threaded executor can starve the rest
        # of the dashboard and is unnecessary when compressed is available.
        if self.camera_raw_topic and not self.camera_topic:
            self.create_subscription(
                Image,
                self.camera_raw_topic,
                self.camera_raw,
                qos_profile_sensor_data,
            )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, self.map_topic, self.map, map_qos)
        amcl_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            f'{prefix}/amcl_pose',
            self.amcl_pose,
            amcl_qos,
        )
        self.create_timer(0.1, self.drive_watchdog)
        self.create_timer(0.1, self.follow_tick)
        self.create_timer(0.1, self.localization_tick)
        # AMCL intentionally does not publish /amcl_pose for every laser scan
        # when the robot is stationary. Ask it for an occasional no-motion
        # update so the Dashboard gets a fresh pose heartbeat without moving
        # the Dingo or starting another global search.
        self.create_timer(2.0, self.localization_heartbeat_tick)
        self.create_timer(0.1, self.wake_training_tick)
        self.create_timer(2.0, self.localization_lifecycle_tick)
        self.create_timer(2.0, self.navigation_lifecycle_tick)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            f'{prefix}/initialpose',
            10,
        )
        self.create_timer(0.5, self.publish_initial_pose)
        self.create_timer(2.0, self.auto_start_navigation_tick)
        self.create_timer(2.0, self.auto_localization_tick)
        self.create_timer(2.0, self.localization_watchdog_tick)
        # If the Dashboard was killed while autonomous mode was active, the
        # in-memory twist_mux parameter can outlive it.  Retry a fail-closed
        # restore for a short startup window until twist_mux is discoverable.
        self.create_timer(2.0, self.autonomous_navigation_lock_tick)

    @staticmethod
    def finite_value(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _wake_training_counts_locked(self):
        real_dir = self.wake_training_dir / 'real_wav'
        return {
            'pos': len(list(real_dir.glob('real_pos_*.wav'))),
            'neg': len(list(real_dir.glob('real_neg_*.wav'))),
        }

    @staticmethod
    def _wake_training_target(phase):
        # Keep a comfortable margin of real recordings so the wake-word
        # model sees both voice variation and hard negatives before install.
        return 35 if phase == 'pos' else 60

    def wake_training_snapshot(self):
        with self.wake_training_lock:
            result = dict(self.wake_training)
            counts = self._wake_training_counts_locked()
            result['positives'] = counts['pos']
            result['negatives'] = counts['neg']
            result['positive_target'] = 35
            result['negative_target'] = 60
            result['recording'] = bool(self.wake_training_active)
            if self.wake_training_active:
                result['elapsed_s'] = min(
                    self.wake_training_seconds,
                    max(0.0, time.monotonic() - self.wake_training_started_at),
                )
                result['remaining_s'] = max(
                    0.0, self.wake_training_seconds - result['elapsed_s']
                )
            else:
                result['remaining_s'] = 0.0
            result['training_dir'] = str(self.wake_training_dir)
            return result

    def start_wake_training(self, phase):
        phase = str(phase or '').strip().lower()
        aliases = {
            'positive': 'pos',
            'positives': 'pos',
            'θετικό': 'pos',
            'θετικο': 'pos',
            'negative': 'neg',
            'negatives': 'neg',
            'αρνητικό': 'neg',
            'αρνητικο': 'neg',
        }
        phase = aliases.get(phase, phase)
        if phase not in {'pos', 'neg'}:
            raise ValueError('Η φάση πρέπει να είναι pos ή neg.')
        with self.wake_training_lock:
            if self.wake_training_active:
                raise RuntimeError('Μια εγγραφή βρίσκεται ήδη σε εξέλιξη.')
            if self.wake_training.get('state') == 'training':
                raise RuntimeError('Το μοντέλο εκπαιδεύεται ήδη.')
            counts = self._wake_training_counts_locked()
            target = self._wake_training_target(phase)
            if counts[phase] >= target:
                label = 'θετικών' if phase == 'pos' else 'αρνητικών'
                raise RuntimeError(f'Έχουν ήδη ολοκληρωθεί οι {label} ηχογραφήσεις.')
            self.wake_training_active = True
            self.wake_training_chunks = []
            self.wake_training_started_at = time.monotonic()
            self.wake_training_phase = phase
            self.wake_training_index = counts[phase] + 1
            self.wake_training_sample_rate = 16000
            self.wake_training = {
                'state': 'recording',
                'phase': phase,
                'current_index': self.wake_training_index,
                'elapsed_s': 0.0,
                'seconds': self.wake_training_seconds,
                'message': (
                    'Πες τώρα καθαρά «Hey Dingo».'
                    if phase == 'pos'
                    else 'Πες τώρα μια παρόμοια φράση ή εντολή, όχι «Hey Dingo».'
                ),
                'last_saved': None,
                'error': None,
                'pipeline_output': None,
            }
        return self.wake_training_snapshot()

    def cancel_wake_training(self):
        with self.wake_training_lock:
            if not self.wake_training_active:
                return False
            self.wake_training_active = False
            self.wake_training_chunks = []
            self.wake_training_phase = None
            self.wake_training_index = None
            self.wake_training = {
                **self.wake_training,
                'state': 'idle',
                'phase': None,
                'current_index': None,
                'elapsed_s': 0.0,
                'message': 'Η εγγραφή ακυρώθηκε. Μπορείς να ξαναπατήσεις ένα κουμπί.',
                'error': None,
            }
            return True

    def wake_training_audio(self, message):
        self.broadcast_microphone_audio(message)
        with self.wake_training_lock:
            if not self.wake_training_active or not message.data:
                return
            try:
                channels = max(1, int(message.number_of_channels or 1))
                # RawAudio.data is a byte sequence.  Going through int8 keeps
                # this correct for both uint8 and int8 ROS bindings.
                raw = bytes((int(value) & 0xFF for value in message.data))
                values = np.frombuffer(raw, dtype='<i1')
                if len(values) % 2:
                    values = values[:-1]
                pcm = values.view('<i2')
                if channels > 1:
                    usable = (len(pcm) // channels) * channels
                    pcm = pcm[:usable].reshape(-1, channels)[:, 0]
                if len(pcm) == 0:
                    return
                rate = max(1, int(message.sample_rate or 16000))
                if rate != 16000 and len(pcm) > 1:
                    length = max(1, int(round(len(pcm) * 16000 / rate)))
                    old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
                    new = np.linspace(0.0, 1.0, num=length, endpoint=False)
                    pcm = np.interp(new, old, pcm).astype(np.int16)
                self.wake_training_sample_rate = 16000
                self.wake_training_chunks.append(
                    np.asarray(pcm, dtype='<i2').copy()
                )
            except (TypeError, ValueError, OverflowError):
                # A malformed block must not break the live Dashboard or the
                # assistant; the timer will report if no usable audio arrived.
                return

    def register_audio_monitor(self):
        client = queue.Queue(maxsize=12)
        with self.audio_monitor_lock:
            self.audio_monitor_clients.add(client)
        return client

    def unregister_audio_monitor(self, client):
        with self.audio_monitor_lock:
            self.audio_monitor_clients.discard(client)

    def broadcast_microphone_audio(self, message):
        """Fan out mono ch0 PCM to explicitly requested browser listeners."""
        if not message.data:
            return
        try:
            channels = max(1, int(message.number_of_channels or 1))
            raw = bytes((int(value) & 0xFF for value in message.data))
            pcm = np.frombuffer(raw, dtype='<i1').view('<i2')
            if channels > 1:
                usable = (len(pcm) // channels) * channels
                pcm = pcm[:usable].reshape(-1, channels)[:, 0]
            payload = np.asarray(pcm, dtype='<i2').tobytes()
        except (TypeError, ValueError, OverflowError):
            return
        if not payload:
            return
        with self.audio_monitor_lock:
            clients = tuple(self.audio_monitor_clients)
        for client in clients:
            try:
                client.put_nowait(payload)
            except queue.Full:
                try:
                    client.get_nowait()
                    client.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

    def publish_remote_audio(self, payload, sample_rate=16000):
        """Publish one mono PCM block captured by a browser microphone."""
        if not payload or len(payload) < 2 or len(payload) % 2:
            return
        message = RawAudio()
        message.timestamp = self.get_clock().now().to_msg()
        message.data = list(payload)
        message.format = 'pcm-s16'
        message.sample_rate = int(sample_rate)
        message.number_of_channels = 1
        self.remote_audio_pub.publish(message)

    def wake_training_tick(self):
        with self.wake_training_lock:
            if not self.wake_training_active:
                return
            elapsed = max(0.0, time.monotonic() - self.wake_training_started_at)
            self.wake_training['elapsed_s'] = min(
                self.wake_training_seconds, elapsed
            )
            if elapsed < self.wake_training_seconds:
                return
            phase = self.wake_training_phase
            index = self.wake_training_index
            chunks = list(self.wake_training_chunks)
            self.wake_training_active = False
            self.wake_training_chunks = []
            self.wake_training_phase = None
            self.wake_training_index = None
            self.wake_training['state'] = 'saving'

        path = None
        try:
            if not chunks:
                raise RuntimeError(
                    'Δεν έφτασε ήχος από το ReSpeaker. Έλεγξε ότι το voice service είναι ενεργό.'
                )
            audio = np.concatenate(chunks).astype(np.int16, copy=False)
            if len(audio) < 1600:
                raise RuntimeError('Η ηχογράφηση ήταν πολύ σύντομη.')
            real_dir = self.wake_training_dir / 'real_wav'
            real_dir.mkdir(parents=True, exist_ok=True)
            path = real_dir / f'real_{phase}_{int(index):03d}.wav'
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(np.asarray(audio, dtype='<i2').tobytes())
            with self.wake_training_lock:
                counts = self._wake_training_counts_locked()
                finished = counts['pos'] >= 35 and counts['neg'] >= 60
                self.wake_training = {
                    **self.wake_training,
                    'state': 'complete' if finished else 'idle',
                    'phase': None,
                    'current_index': None,
                    'elapsed_s': self.wake_training_seconds,
                    'message': (
                        'Όλες οι ηχογραφήσεις ολοκληρώθηκαν. Πάτησε «Εκπαίδευση μοντέλου».'
                        if finished
                        else f'Αποθηκεύτηκε ηχογράφηση {phase} {int(index):02d}. Πάτησε ξανά για την επόμενη.'
                    ),
                    'last_saved': str(path),
                    'error': None,
                }
        except (OSError, RuntimeError, ValueError) as exc:
            with self.wake_training_lock:
                self.wake_training = {
                    **self.wake_training,
                    'state': 'error',
                    'phase': None,
                    'current_index': None,
                    'message': str(exc),
                    'error': str(exc),
                    'last_saved': str(path) if path else None,
                }

    def start_wake_training_pipeline(self):
        with self.wake_training_lock:
            if self.wake_training_active:
                raise RuntimeError('Ολοκλήρωσε πρώτα την τρέχουσα εγγραφή.')
            if self.wake_training.get('state') == 'training':
                raise RuntimeError('Το μοντέλο εκπαιδεύεται ήδη.')
            counts = self._wake_training_counts_locked()
            if counts['pos'] < 35 or counts['neg'] < 60:
                raise RuntimeError(
                    f'Χρειάζονται 35 θετικές και 60 αρνητικές ηχογραφήσεις '
                    f'(έχεις {counts["pos"]} και {counts["neg"]}).'
                )
            self.wake_training = {
                **self.wake_training,
                'state': 'training',
                'message': 'Εκπαιδεύω και ελέγχω το μοντέλο «Hey Dingo»…',
                'error': None,
                'pipeline_output': None,
            }
            thread = threading.Thread(
                target=self._run_wake_training_pipeline,
                name='dingo-wake-training',
                daemon=True,
            )
            self.wake_training_pipeline_thread = thread
            thread.start()
        return self.wake_training_snapshot()

    def _run_wake_training_pipeline(self):
        output = []
        installed = False
        restart_ok = None
        try:
            commands = (
                ['generate_data.py'],
                ['extract_features.py'],
                ['train.py'],
                ['evaluate.py', '--model', 'hey_dingo_candidate.onnx'],
            )
            for command in commands:
                result = subprocess.run(
                    [sys.executable, *command],
                    cwd=str(self.wake_training_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=True,
                )
                output.append(result.stdout[-1800:])
                if result.stderr:
                    output.append(result.stderr[-700:])
            evaluation = '\n'.join(output)
            false_positive_match = re.search(
                r'false positives at threshold 0\.50:\s*(\d+)', evaluation
            )
            positive_min_match = re.search(
                r'positive min/mean/max:\s*([0-9.]+)/', evaluation
            )
            false_positives = (
                int(false_positive_match.group(1))
                if false_positive_match
                else None
            )
            positive_min = (
                float(positive_min_match.group(1))
                if positive_min_match
                else None
            )
            # Install only after the recorded negatives produce no threshold
            # false positives and the weakest positive clears the runtime
            # threshold.  Otherwise the candidate remains available for review
            # but cannot wake the robot accidentally.
            if (
                false_positives == 0
                and positive_min is not None
                and positive_min >= 0.5
            ):
                candidate = self.wake_training_dir / 'hey_dingo_candidate.onnx'
                destination = self.wake_training_dir.parents[1] / 'models' / 'wake_words' / 'dingo.onnx'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
                installed = True
                restart = subprocess.run(
                    ['systemctl', '--user', 'restart', 'dingo-voice.service'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                restart_ok = restart.returncode == 0
                if restart.stderr:
                    output.append(restart.stderr[-500:])
            summary = (
                'Έτοιμο: το «Hey Dingo» εγκαταστάθηκε και το voice service '
                'επανεκκινήθηκε.'
                if installed and restart_ok
                else 'Το μοντέλο πέρασε τον έλεγχο και εγκαταστάθηκε. '
                'Χρειάζεται επανεκκίνηση του voice service.'
                if installed
                else 'Το candidate δημιουργήθηκε, αλλά δεν εγκαταστάθηκε '
                'επειδή ο έλεγχος είχε false positives.'
            )
            if false_positives is not None:
                summary += f' False positives: {false_positives}.'
            with self.wake_training_lock:
                self.wake_training = {
                    **self.wake_training,
                    'state': 'complete' if installed else 'error',
                    'message': summary,
                    'error': None if installed else summary,
                    'pipeline_output': '\n'.join(output)[-5000:],
                }
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            output.append(str(exc))
            with self.wake_training_lock:
                self.wake_training = {
                    **self.wake_training,
                    'state': 'error',
                    'message': f'Η εκπαίδευση απέτυχε: {exc}',
                    'error': str(exc),
                    'pipeline_output': '\n'.join(output)[-5000:],
                }
        finally:
            self.wake_training_pipeline_thread = None

    @staticmethod
    def gemini_usage_defaults():
        return {
            'day': None,
            'provider': 'Google Gemini API',
            'model': 'gemini-2.5-flash',
            'status': 'waiting_for_usage',
            'requests_today': 0,
            'requests_session': 0,
            'errors_today': 0,
            'input_tokens_today': 0,
            'output_tokens_today': 0,
            'thinking_tokens_today': 0,
            'total_tokens_today': 0,
            'last_request_at': None,
            'last_request_type': None,
            'last_usage': None,
            'last_error': None,
            # The Gemini API does not expose a remaining-quota number when
            # the client authenticates only with an API key.  Keep this
            # explicit so the UI never presents a guessed percentage.
            'quota_remaining': None,
            'quota_source': 'not_available_with_api_key',
            'quota_note_el': (
                'Το ακριβές υπόλοιπο quota δεν επιστρέφεται από το Gemini API '
                'με απλό API key. Τα όρια ανήκουν στο Google Cloud project '
                'και ελέγχονται στο AI Studio.'
            ),
        }

    def load_gemini_usage(self):
        today = datetime.now().astimezone().date().isoformat()
        result = self.gemini_usage_defaults()
        result['day'] = today
        result['model'] = self.vision_llm_model
        try:
            stored = json.loads(self.gemini_usage_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return result
        if not isinstance(stored, dict) or stored.get('day') != today:
            return result
        integer_fields = (
            'requests_today',
            'errors_today',
            'input_tokens_today',
            'output_tokens_today',
            'thinking_tokens_today',
            'total_tokens_today',
        )
        for field in integer_fields:
            try:
                result[field] = max(0, int(stored.get(field, 0)))
            except (TypeError, ValueError, OverflowError):
                pass
        for field in ('status', 'last_request_type', 'last_error', 'last_request_at'):
            if field in stored:
                result[field] = stored[field]
        if isinstance(stored.get('last_usage'), dict):
            result['last_usage'] = dict(stored['last_usage'])
        return result

    def persist_gemini_usage(self):
        with self.lock:
            current = dict(self.state.get('gemini') or {})
        fields = (
            'day',
            'provider',
            'model',
            'status',
            'requests_today',
            'errors_today',
            'input_tokens_today',
            'output_tokens_today',
            'thinking_tokens_today',
            'total_tokens_today',
            'last_request_at',
            'last_request_type',
            'last_usage',
            'last_error',
        )
        payload = {field: current.get(field) for field in fields}
        try:
            self.gemini_usage_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.gemini_usage_file.with_name(
                self.gemini_usage_file.name + '.tmp'
            )
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            os.chmod(temporary, 0o600)
            temporary.replace(self.gemini_usage_file)
        except OSError as exc:
            self.get_logger().warning(f'Δεν αποθηκεύτηκε το Gemini usage: {exc}')

    def record_gemini_usage(self, usage, request_type='assistant', model=None):
        if not isinstance(usage, dict) or not usage:
            return

        def token_count(key):
            try:
                return max(0, int(usage.get(key)))
            except (TypeError, ValueError, OverflowError):
                return None

        prompt_tokens = token_count('promptTokenCount')
        output_tokens = token_count('candidatesTokenCount')
        thinking_tokens = token_count('thoughtsTokenCount')
        total_tokens = token_count('totalTokenCount')
        known_tokens = [
            value for value in (prompt_tokens, output_tokens, thinking_tokens)
            if value is not None
        ]
        if total_tokens is None and known_tokens:
            total_tokens = sum(known_tokens)
        today = datetime.now().astimezone().date().isoformat()
        now = time.time()
        with self.lock:
            current = dict(self.state.get('gemini') or self.gemini_usage_defaults())
            if current.get('day') != today:
                current = self.gemini_usage_defaults()
                current['day'] = today
            current['model'] = str(model or current.get('model') or self.vision_llm_model)
            current['status'] = 'measured'
            current['requests_today'] = int(current.get('requests_today') or 0) + 1
            current['requests_session'] = int(current.get('requests_session') or 0) + 1
            current['input_tokens_today'] = int(current.get('input_tokens_today') or 0) + (prompt_tokens or 0)
            current['output_tokens_today'] = int(current.get('output_tokens_today') or 0) + (output_tokens or 0)
            current['thinking_tokens_today'] = int(current.get('thinking_tokens_today') or 0) + (thinking_tokens or 0)
            current['total_tokens_today'] = int(current.get('total_tokens_today') or 0) + (total_tokens or 0)
            current['last_request_at'] = now
            current['last_request_type'] = str(request_type or 'assistant')[:40]
            current['last_usage'] = {
                'input_tokens': prompt_tokens,
                'output_tokens': output_tokens,
                'thinking_tokens': thinking_tokens,
                'total_tokens': total_tokens,
            }
            current['last_error'] = None
            self.state['gemini'] = current
        self.persist_gemini_usage()

    def record_gemini_error(self, error):
        today = datetime.now().astimezone().date().isoformat()
        with self.lock:
            current = dict(self.state.get('gemini') or self.gemini_usage_defaults())
            if current.get('day') != today:
                current = self.gemini_usage_defaults()
                current['day'] = today
            current['status'] = 'error'
            current['errors_today'] = int(current.get('errors_today') or 0) + 1
            current['last_error'] = str(error)[:300]
            self.state['gemini'] = current
        self.persist_gemini_usage()

    def gemini_usage(self, msg):
        try:
            payload = json.loads(msg.data or '{}')
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.record_gemini_usage(
            payload.get('usage'),
            payload.get('request_type', 'voice'),
            payload.get('model'),
        )

    def battery(self, msg):
        now = time.time()
        percentage = self.finite_value(msg.percentage)
        voltage = self.finite_value(msg.voltage)
        current = self.finite_value(msg.current)
        with self.lock:
            self.state['connected'] = True
            self.state['battery'] = {
                'percent': round(percentage * 100.0, 1)
                if percentage is not None and percentage >= 0
                else None,
                'voltage': round(voltage, 2) if voltage is not None and voltage >= 0 else None,
                'current': round(current, 3) if current is not None else None,
                'power_w': round(abs(voltage * current), 2)
                if voltage is not None and current is not None
                else None,
                'status': int(msg.power_supply_status),
                'last_update': now,
            }

    def power(self, msg):
        now = time.time()
        def reading(values, index):
            if index >= len(values):
                return None
            value = self.finite_value(values[index])
            return round(value, 3) if value is not None else None

        voltages = list(msg.measured_voltages)
        currents = list(msg.measured_currents)
        battery_voltage = reading(voltages, Power.D100_MEASURED_BATTERY)
        rail_5v_voltage = reading(voltages, Power.D100_MEASURED_5V)
        rail_12v_voltage = reading(voltages, Power.D100_MEASURED_12V)
        total_current = reading(currents, Power.D100_TOTAL_CURRENT)
        computer_current = reading(currents, Power.D100_COMPUTER_CURRENT)
        computer_voltage = rail_12v_voltage or battery_voltage
        with self.lock:
            self.state['connected'] = True
            self.state['power'] = {
                'battery_voltage': battery_voltage,
                'rail_5v_voltage': rail_5v_voltage,
                'rail_12v_voltage': rail_12v_voltage,
                'total_current': total_current,
                'computer_current': computer_current,
                'total_power_w': round(abs(battery_voltage * total_current), 2)
                if battery_voltage is not None and total_current is not None
                else None,
                'computer_voltage': computer_voltage,
                'computer_power_w': round(abs(computer_voltage * computer_current), 2)
                if computer_voltage is not None and computer_current is not None
                else None,
                'source': 'platform/mcu/status/power',
                'last_update': now,
            }

    def mcu_temperature(self, msg):
        now = time.time()
        values = []
        for value in list(msg.temperatures):
            value = self.finite_value(value)
            if value is not None:
                values.append(round(value, 1))
        with self.lock:
            self.state['connected'] = True
            self.state['temperature'] = {
                'values': values,
                'source': 'platform/mcu/status/temperature',
                'frame': msg.header.frame_id or None,
                'last_update': now,
            }

    def imu_data(self, msg):
        """Expose filtered IMU values and a lightweight health rate."""
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = (
            math.copysign(math.pi / 2.0, sinp)
            if abs(sinp) >= 1.0
            else math.asin(sinp)
        )
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        now = time.time()
        with self.lock:
            previous = self.imu_previous_update
            self.imu_previous_update = now
            self.imu_last_update = now
            old_rate = (self.state.get('imu') or {}).get('rate_hz')
            rate = old_rate
            if previous > 0.0 and now > previous:
                instant_rate = 1.0 / (now - previous)
                rate = (
                    instant_rate
                    if old_rate is None
                    else old_rate * 0.8 + instant_rate * 0.2
                )
            self.state['imu'] = {
                'available': True,
                'frame': msg.header.frame_id or 'imu_0_link',
                'orientation': {
                    'roll_deg': round(math.degrees(roll), 2),
                    'pitch_deg': round(math.degrees(pitch), 2),
                    'yaw_deg': round(math.degrees(yaw), 2),
                },
                'angular_velocity': {
                    'x': round(float(msg.angular_velocity.x), 4),
                    'y': round(float(msg.angular_velocity.y), 4),
                    'z': round(float(msg.angular_velocity.z), 4),
                },
                'linear_acceleration': {
                    'x': round(float(msg.linear_acceleration.x), 3),
                    'y': round(float(msg.linear_acceleration.y), 3),
                    'z': round(float(msg.linear_acceleration.z), 3),
                },
                'rate_hz': round(rate, 1) if rate is not None else None,
                'last_update': now,
                'source': f'/{self.robot_namespace}/sensors/imu_0/data',
            }

    @classmethod
    def vision_label(cls, label):
        label = str(label or '').strip().lower()
        return cls.VISION_GREEK_LABELS.get(label, (label or 'αντικείμενο', label or 'αντικείμενα'))

    @classmethod
    def vision_summary(cls, objects):
        counts = Counter(item.get('label', '') for item in objects if item.get('label'))
        if not counts:
            return 'Δεν αναγνωρίζω αντικείμενα αυτή τη στιγμή.'
        parts = []
        for label, count in counts.most_common(8):
            singular, plural = cls.vision_label(label)
            parts.append(f'{count} {singular if count == 1 else plural}')
        return 'Βλέπω ' + ', '.join(parts) + '.'

    def vision_detections(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, list):
            return
        objects = []
        for item in payload[:32]:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip().lower()
            if not label:
                continue
            confidence = self.finite_value(item.get('conf'))
            distance = self.finite_value(item.get('z'))
            if distance is None:
                distance = self.finite_value(item.get('box_distance'))
            x1 = self.finite_value(item.get('x1'))
            y1 = self.finite_value(item.get('y1'))
            x2 = self.finite_value(item.get('x2'))
            y2 = self.finite_value(item.get('y2'))
            img_w = self.finite_value(item.get('img_w'))
            img_h = self.finite_value(item.get('img_h'))
            center_x = self.finite_value(item.get('center_x'))
            center_y = self.finite_value(item.get('center_y'))
            if center_x is None and x1 is not None and x2 is not None:
                center_x = (x1 + x2) / 2.0
            if center_y is None and y1 is not None and y2 is not None:
                center_y = (y1 + y2) / 2.0
            singular, _ = self.vision_label(label)
            detected = {
                'label': label,
                'label_el': singular,
                'confidence': round(confidence, 2) if confidence is not None else None,
                'distance_m': round(distance, 2) if distance is not None else None,
            }
            for key, value in (
                ('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2),
                ('center_x', center_x), ('center_y', center_y),
                ('img_w', img_w), ('img_h', img_h),
            ):
                if value is not None:
                    detected[key] = round(value, 2)
            objects.append(detected)
        with self.lock:
            previous = dict(self.state.get('vision') or {})
            self.state['vision'] = {
                'state': 'ready',
                'source': 'YOLO11n · CPU',
                'summary_el': self.vision_summary(objects),
                'objects': objects,
                'last_update': time.time(),
                'detail_state': previous.get('detail_state', 'idle'),
                'detail_question': previous.get('detail_question'),
                'detail_answer': previous.get('detail_answer'),
                'detail_updated': previous.get('detail_updated'),
            }

    def vision_reply(self):
        with self.lock:
            vision = dict(self.state.get('vision') or {})
            objects = list(vision.get('objects') or [])
            last_update = vision.get('last_update')
            summary = str(vision.get('summary_el') or '')
        try:
            stale = last_update is None or time.time() - float(last_update) > 5.0
        except (TypeError, ValueError):
            stale = True
        if stale:
            self.publish_voice_reply(
                'Η αναγνώριση αντικειμένων δεν είναι ενεργή ή η κάμερα δεν στέλνει εικόνα.',
                ok=False,
                action='vision',
            )
            return
        if not summary:
            summary = self.vision_summary(objects)
        self.publish_voice_reply(summary, action='vision')

    @staticmethod
    def read_host_temperatures():
        readings = []
        seen_names = set()

        def add_reading(name, raw_value):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return
            if abs(value) > 200.0:
                value /= 1000.0
            if not math.isfinite(value) or value < -20.0 or value > 150.0:
                return
            name = str(name or 'temperature').strip()[:48]
            if name in seen_names:
                return
            seen_names.add(name)
            readings.append({'name': name, 'c': round(value, 1)})

        thermal_root = Path('/sys/class/thermal')
        try:
            for temp_path in sorted(thermal_root.glob('thermal_zone*/temp')):
                zone = temp_path.parent
                try:
                    label = (zone / 'type').read_text(encoding='utf-8').strip()
                    raw_value = temp_path.read_text(encoding='utf-8').strip()
                except (OSError, UnicodeError):
                    continue
                add_reading(label or zone.name, raw_value)
        except OSError:
            pass

        # Some systems expose additional sensors through hwmon only.
        hwmon_root = Path('/sys/class/hwmon')
        try:
            for temp_path in sorted(hwmon_root.glob('hwmon*/temp*_input')):
                chip = temp_path.parent
                try:
                    chip_name = (chip / 'name').read_text(encoding='utf-8').strip()
                    label_path = temp_path.with_name(
                        temp_path.name.replace('_input', '_label')
                    )
                    label = (
                        label_path.read_text(encoding='utf-8').strip()
                        if label_path.is_file()
                        else temp_path.stem
                    )
                    raw_value = temp_path.read_text(encoding='utf-8').strip()
                except (OSError, UnicodeError):
                    continue
                add_reading(f'{chip_name or chip.name} {label}', raw_value)
        except OSError:
            pass
        return readings[:16]

    @staticmethod
    def read_meminfo():
        values = {}
        try:
            lines = Path('/proc/meminfo').read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError):
            return values
        for line in lines:
            key, separator, rest = line.partition(':')
            if not separator:
                continue
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                value = float(parts[0])
            except ValueError:
                continue
            if len(parts) > 1 and parts[1].lower() == 'kb':
                value *= 1024.0
            values[key] = value
        return values

    @staticmethod
    def format_gib(value):
        try:
            return f'{float(value) / (1024 ** 3):.1f} GiB'
        except (TypeError, ValueError):
            return 'μη διαθέσιμο'

    def system_processes_text(self):
        """Return a bounded read-only view of Dingo-related processes."""
        try:
            result = subprocess.run(
                ['ps', '-eo', 'comm=,stat=,pcpu=,pmem='],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f'Δεν μπόρεσα να διαβάσω τις διεργασίες: {exc}.'
        keywords = (
            'dingo', 'ros', 'nav2', 'amcl', 'fastflow', 'whisper',
            'yolo', 'realsense', 'respeaker', 'python',
        )
        rows = []
        for line in (result.stdout or '').splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            command = parts[0]
            if not any(keyword in command.lower() for keyword in keywords):
                continue
            rows.append(f'{command} ({parts[2]}% CPU, {parts[3]}% RAM)')
            if len(rows) >= 8:
                break
        return (
            'Dingo διεργασίες: ' + '; '.join(rows) + '.'
            if rows else 'Δεν βρέθηκαν σχετικές ενεργές διεργασίες.'
        )

    @staticmethod
    def _fresh_value(last_update, max_age):
        try:
            return last_update is not None and time.time() - float(last_update) <= max_age
        except (TypeError, ValueError):
            return False

    def system_devices_text(self):
        with self.lock:
            state = {
                key: dict(self.state.get(key) or {})
                for key in ('battery', 'power', 'temperature', 'scan', 'imu', 'camera', 'voice')
            }
        mcu_live = any(
            self._fresh_value(state[key].get('last_update'), 3.0)
            for key in ('battery', 'power', 'temperature')
        )
        scan = state['scan']
        imu = state['imu']
        camera = state['camera']
        voice = state['voice']
        return (
            'Συνδεδεμένα: '
            f'MCU {"live" if mcu_live else "offline"}, '
            f'LiDAR {"live" if self._fresh_value(scan.get("last_update"), 1.5) else "offline"}, '
            f'IMU {"live" if self._fresh_value(imu.get("last_update"), 1.5) else "offline"}, '
            f'RealSense {"live" if camera.get("available") else "offline"}, '
            f'ReSpeaker {"live" if voice.get("wake_ready") else "σε αναμονή"}. '
            'STT: AMD GPU Vulkan · Qwen: AMD NPU.'
        )

    def system_sensors_text(self):
        with self.lock:
            scan = dict(self.state.get('scan') or {})
            imu = dict(self.state.get('imu') or {})
            camera = dict(self.state.get('camera') or {})
            localization = dict(self.state.get('localization') or {})
            emergency = self.state.get('emergency_stop')
            safety = self.state.get('safety_stop')
        navigation = self.navigation_snapshot()
        return (
            'Αισθητήρες: '
            f'LiDAR {scan.get("valid", 0)} έγκυρα σημεία, '
            f'IMU {"live" if self._fresh_value(imu.get("last_update"), 1.5) else "offline"}, '
            f'κάμερα {"live" if camera.get("available") else "offline"}, '
            f'AMCL {"εντοπισμένο" if navigation.get("localized") else "όχι εντοπισμένο"}, '
            f'safety {"μπλοκαρισμένο" if emergency is True or safety is True else "ελεύθερο"}. '
            f'Frame LiDAR: {scan.get("frame") or "—"}.'
        )

    def system_voice_text(self):
        with self.lock:
            voice = dict(self.state.get('voice') or {})
        return (
            f'Voice: {voice.get("state", "unknown")}. '
            f'Wake word: {voice.get("wake_model", "—")} '
            f'({"έτοιμο" if voice.get("wake_ready") else "όχι έτοιμο"}). '
            f'STT {voice.get("stt_model", "—")} σε {voice.get("stt_active_backend") or voice.get("stt_backend", "—")}. '
            f'LLM {voice.get("llm_model", "—")} ({voice.get("llm_provider", "—")}).'
        )

    def system_vision_text(self):
        with self.lock:
            vision = dict(self.state.get('vision') or {})
            objects = [
                dict(item) for item in (vision.get('objects') or [])
                if isinstance(item, dict)
            ]
        labels = [str(item.get('label_el') or item.get('label') or '').strip() for item in objects]
        labels = [label for label in labels if label]
        return (
            f'Vision: {vision.get("state", "offline")}. '
            + (f'Βλέπει: {", ".join(labels[:8])}.' if labels else 'Δεν αναγνωρίζονται αντικείμενα τώρα.')
        )

    def system_gemini_text(self):
        with self.lock:
            gemini = dict(self.state.get('gemini') or {})
        return (
            f'Gemini: {gemini.get("status", "αναμονή")}, '
            f'μοντέλο {gemini.get("model", "gemini-2.5-flash")}, '
            f'{gemini.get("requests_today", 0)} κλήσεις και '
            f'{gemini.get("total_tokens_today", 0)} tokens σήμερα.'
        )

    def system_ros_text(self):
        try:
            api = self.clearpath_api_snapshot()
            runtime = api.get('runtime') or {}
            return (
                f'ROS graph: {len(runtime.get("nodes") or [])} nodes, '
                f'{len(runtime.get("topics") or [])} runtime topics και '
                f'{len(runtime.get("services") or [])} runtime services. '
                f'Clearpath schema: {api.get("schema", "—")}.'
            )
        except Exception as exc:
            return f'Το ROS graph δεν είναι διαθέσιμο: {exc}.'

    def system_services_text(self):
        try:
            statuses = self.tool_services_status()
        except RuntimeError as exc:
            return f'Δεν μπόρεσα να ελέγξω τις υπηρεσίες: {exc}.'
        parts = [f'{item["label"]}: {item["active"]}' for item in statuses]
        return 'Υπηρεσίες Dingo: ' + '; '.join(parts) + '.'

    def system_info_reply(self, target='summary'):
        target = self.normalize_voice_text(target) or 'summary'
        aliases = {
            'disk': 'disk',
            'storage': 'disk',
            'space': 'disk',
            'δισκο': 'disk',
            'χωρο': 'disk',
            'memory': 'memory',
            'ram': 'memory',
            'μνημη': 'memory',
            'cpu': 'cpu',
            'temperature': 'temperature',
            'temperatures': 'temperature',
            'θερμοκρασια': 'temperature',
            'θερμοκρασιες': 'temperature',
            'power': 'power',
            'ρευματα': 'power',
            'καταναλωση': 'power',
            'network': 'network',
            'δικτυο': 'network',
            'services': 'services',
            'service': 'services',
            'υπηρεσιες': 'services',
            'υπηρεσια': 'services',
            'processes': 'processes',
            'process': 'processes',
            'διαδικασιες': 'processes',
            'διεργασιες': 'processes',
            'devices': 'devices',
            'device': 'devices',
            'συσκευες': 'devices',
            'περιφερειακα': 'devices',
            'sensors': 'sensors',
            'sensor': 'sensors',
            'αισθητηρες': 'sensors',
            'λιδαρ': 'sensors',
            'lidar': 'sensors',
            'imu': 'sensors',
            'localization': 'sensors',
            'εντοπισμος': 'sensors',
            'voice': 'voice',
            'φωνη': 'voice',
            'μικροφωνο': 'voice',
            'stt': 'voice',
            'vision': 'vision',
            'camera': 'vision',
            'καμερα': 'vision',
            'yolo': 'vision',
            'gemini': 'gemini',
            'quota': 'gemini',
            'tokens': 'gemini',
            'ros': 'ros',
            'topics': 'ros',
            'services_graph': 'ros',
            'summary': 'summary',
            'all': 'summary',
        }
        target = aliases.get(target, 'summary')

        usage = shutil.disk_usage(str(Path.home()))
        disk_text = (
            f'Δίσκος: {self.format_gib(usage.free)} ελεύθερα από '
            f'{self.format_gib(usage.total)} '
            f'({usage.used / usage.total * 100.0:.0f}% χρησιμοποιείται).'
            if usage.total
            else 'Ο δίσκος δεν είναι διαθέσιμος.'
        )

        memory = self.read_meminfo()
        total_memory = memory.get('MemTotal')
        available_memory = memory.get('MemAvailable', memory.get('MemFree'))
        used_memory = (
            total_memory - available_memory
            if total_memory is not None and available_memory is not None
            else None
        )
        memory_text = (
            f'RAM: {self.format_gib(available_memory)} διαθέσιμη από '
            f'{self.format_gib(total_memory)} '
            f'({self.format_gib(used_memory)} χρησιμοποιούνται).'
            if total_memory is not None and available_memory is not None
            else 'Η RAM δεν είναι διαθέσιμη.'
        )

        try:
            load_1, load_5, load_15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            cpu_text = (
                f'CPU: {cpu_count} λογικοί πυρήνες, load '
                f'{load_1:.2f}/{load_5:.2f}/{load_15:.2f} '
                f'(1/5/15 λεπτά, περίπου {load_1 / cpu_count * 100.0:.0f}% φόρτος).'
            )
        except (AttributeError, OSError):
            cpu_text = 'Η μέτρηση CPU δεν είναι διαθέσιμη.'

        host_temperatures = self.read_host_temperatures()
        with self.lock:
            mcu_temperature = dict(self.state.get('temperature') or {})
            power = dict(self.state.get('power') or {})
            battery = dict(self.state.get('battery') or {})
        temperature_parts = [
            f'{item["name"]}: {item["c"]:.1f}°C'
            for item in host_temperatures
        ]
        for index, value in enumerate(mcu_temperature.get('values') or []):
            temperature_parts.append(f'MCU αισθητήρας {index}: {value:.1f}°C')
        temperature_text = (
            'Θερμοκρασίες: ' + ', '.join(temperature_parts[:8]) + '.'
            if temperature_parts
            else 'Δεν βρέθηκε διαθέσιμος αισθητήρας θερμοκρασίας.'
        )

        power_parts = []
        total_w = power.get('total_power_w')
        total_current = power.get('total_current')
        computer_w = power.get('computer_power_w')
        computer_current = power.get('computer_current')
        if total_w is not None:
            power_parts.append(f'σύνολο {total_w:.1f} W')
        if total_current is not None:
            power_parts.append(f'{total_current:.2f} A συνολικά')
        if computer_w is not None:
            power_parts.append(f'Computer {computer_w:.1f} W')
        if computer_current is not None:
            power_parts.append(f'Computer {computer_current:.2f} A')
        if battery.get('percent') is not None:
            power_parts.append(f'μπαταρία {battery["percent"]:.0f}%')
        power_text = (
            'Κατανάλωση: ' + ', '.join(power_parts) + '.'
            if power_parts
            else 'Δεν έχω διαθέσιμη live μέτρηση κατανάλωσης.'
        )

        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except OSError:
            ip_address = 'μη διαθέσιμη'
        network_text = f'Δίκτυο: όνομα {hostname}, IP {ip_address}.'

        if target == 'disk':
            reply = disk_text
        elif target == 'memory':
            reply = memory_text
        elif target == 'cpu':
            reply = cpu_text
        elif target == 'temperature':
            reply = temperature_text
        elif target == 'power':
            reply = power_text
        elif target == 'network':
            reply = network_text
        elif target == 'services':
            reply = self.system_services_text()
        elif target == 'processes':
            reply = self.system_processes_text()
        elif target == 'devices':
            reply = self.system_devices_text()
        elif target == 'sensors':
            reply = self.system_sensors_text()
        elif target == 'voice':
            reply = self.system_voice_text()
        elif target == 'vision':
            reply = self.system_vision_text()
        elif target == 'gemini':
            reply = self.system_gemini_text()
        elif target == 'ros':
            reply = self.system_ros_text()
        else:
            reply = ' '.join((disk_text, memory_text, cpu_text, temperature_text))
        self.publish_voice_reply(reply, action='system_info')

    @staticmethod
    def normalize_voice_text(value):
        text = unicodedata.normalize('NFD', str(value or '').lower())
        return ''.join(
            char for char in text if not unicodedata.combining(char)
        ).strip()

    def voice_status(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current.update(
                {
                    'state': str(payload.get('state', 'unknown')),
                    'detail': str(payload.get('detail', '')),
                    'device': payload.get('device'),
                    'sample_rate': payload.get('sample_rate'),
                    'channels': payload.get('channels'),
                    'stt_model': payload.get('stt_model'),
                    'stt_backend': payload.get('stt_backend'),
                    'stt_active_backend': payload.get('stt_active_backend'),
                    'stt_provider': payload.get('stt_provider'),
                    'stt_fallback_model': payload.get('stt_fallback_model'),
                    'stt_fallback_reason': payload.get('stt_fallback_reason'),
                    'llm_provider': payload.get('llm_provider'),
                    'llm_model': payload.get('llm_model'),
                    'llm_tool_calling': payload.get(
                        'llm_tool_calling',
                        current.get('llm_tool_calling', 'json_fallback'),
                    ),
                    'llm_fallback_active': bool(payload.get('llm_fallback_active', False)),
                    'llm_fallback_reason': payload.get('llm_fallback_reason'),
                    'wake_ready': bool(payload.get('wake_ready', False)),
                    'wake_model': str(payload.get('wake_model', 'Alexa')),
                    'wake_threshold': payload.get('wake_threshold'),
                    'wake_score': payload.get('wake_score', 0.0),
                    'wake_detected_at': payload.get('wake_detected_at'),
                    'audio_rms': payload.get('audio_rms', 0.0),
                    'audio_peak': payload.get('audio_peak', 0.0),
                    'last_audio_at': payload.get('last_audio_at'),
                    'wake_error': payload.get('wake_error'),
                    'last_transcript': payload.get(
                        'last_transcript', current.get('last_transcript')
                    ),
                }
            )
            self.state['voice'] = current

    def voice_direction(self, msg):
        """Merge the XVF3800 DSP's DoA/VAD and processing capabilities."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        features = payload.get('features')
        if not isinstance(features, dict):
            features = {}
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current.update(
                {
                    'direction_available': bool(payload.get('available', False)),
                    'direction_enabled': bool(payload.get('enabled', False)),
                    'direction_angle': payload.get('angle_deg'),
                    'direction_speech': bool(payload.get('speech', False)),
                    'direction_at': payload.get('angle_at'),
                    'direction_error': payload.get('error'),
                    'led_state': str(payload.get('led') or 'unknown'),
                    'voice_features': {
                        key: bool(features.get(key, False))
                        for key in (
                            'direction_of_arrival',
                            'hardware_vad',
                            'beamforming',
                            'noise_reduction',
                            'echo_cancellation',
                            'automatic_gain',
                        )
                    },
                }
            )
            self.state['voice'] = current

    def voice_transcript(self, msg):
        transcript = str(msg.data or '')[:500]
        words = re.findall(
            r'[^\W_]+', self.normalize_voice_text(transcript), flags=re.UNICODE
        )
        normalized = ' '.join(words)
        if normalized in {'ευχαριστω', 'ευχαριστω πολυ'} or re.fullmatch(
            r'(?:ευχαριστω(?:\s+πολυ)?)(?:\s+ευχαριστω(?:\s+πολυ)?)+',
            normalized,
            flags=re.UNICODE,
        ):
            return
        def wake_variant(word):
            return word.translate(str.maketrans({
                'a': 'α', 'l': 'λ', 'e': 'ε', 'x': 'ξ',
                'k': 'κ', 'h': 'χ',
            }))
        if len(words) >= 2 and all(
            wake_variant(word) in {'αλεξα', 'αλεχα'} for word in words
        ):
            return
        if len(words) >= 2 and not re.search(r'[α-ω]', normalized, flags=re.UNICODE):
            return
        for phrase_size in (1, 2, 3):
            if (
                len(words) >= phrase_size * 3
                and len(words) % phrase_size == 0
                and words == words[:phrase_size] * (len(words) // phrase_size)
            ):
                return
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['last_transcript'] = transcript
            current['transcript_seq'] = int(
                current.get('transcript_seq', 0) or 0
            ) + 1
            self.state['voice'] = current

    def voice_reply(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        text = str(payload.get('text', '') or '')[:500]
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['last_reply'] = text
            current['last_reply_ok'] = bool(payload.get('ok', False))
            current['last_reply_action'] = payload.get('action')
            current['reply_seq'] = int(current.get('reply_seq', 0) or 0) + 1
            self.state['voice'] = current

    def face_state(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            self.state['face'] = {
                'state': str(payload.get('state', 'unknown')),
                'source': str(payload.get('source', 'YuNet + SFace')),
                'message_el': str(payload.get('message_el', '')),
                'faces': list(payload.get('faces') or [])[:16],
                'enrolled': [str(item) for item in (payload.get('enrolled') or [])][:100],
                'enrollment': payload.get('enrollment'),
                'last_update': payload.get('last_update'),
            }

    def speaker_state(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        pending_since = None
        with self.lock:
            self.state['speaker'] = {
                'state': str(payload.get('state', 'unknown')),
                'current_name': payload.get('current_name'),
                'current_score': payload.get('current_score'),
                'last_update': payload.get('last_update'),
                'enrolled': [str(item) for item in (payload.get('enrolled') or [])][:100],
                'enrollment': payload.get('enrollment'),
                'detail': str(payload.get('detail', '')),
            }
            pending_since = self.speaker_query_pending_since
            if pending_since is not None:
                state_name = str(payload.get('state', ''))
                try:
                    updated = float(payload.get('last_update'))
                except (TypeError, ValueError):
                    updated = 0.0
                if state_name in {'error', 'disabled', 'offline'} or (
                    state_name not in {'recognizing', 'enrolling'}
                    and updated >= pending_since
                ):
                    self.speaker_query_pending_since = None
        if pending_since is not None and self.speaker_query_pending_since is None:
            self.publish_voice_reply(
                self.speaker_reply_text(payload), action='speaker_query'
            )

    @staticmethod
    def speaker_reply_text(payload):
        state = str(payload.get('state', ''))
        if state in {'offline', 'disabled', 'error'}:
            return 'Η αναγνώριση φωνής δεν είναι διαθέσιμη.'
        if not payload.get('enrolled'):
            return 'Δεν έχει εγγραφεί ακόμη καμία φωνή στο Dingo.'
        try:
            stale = (
                payload.get('last_update') is None
                or time.time() - float(payload.get('last_update')) > 5.0
            )
        except (TypeError, ValueError):
            stale = True
        if stale:
            return 'Δεν έχω αρκετά πρόσφατο δείγμα φωνής για να πω ποιος μίλησε.'
        name = str(payload.get('current_name') or '').strip()
        return f'Μίλησε ο/η {name}.' if name else 'Άκουσα φωνή, αλλά δεν αναγνωρίζω ποιος μίλησε.'

    def speaker_reply(self):
        with self.lock:
            payload = dict(self.state.get('speaker') or {})
            recognizing = payload.get('state') == 'recognizing'
            if recognizing:
                self.speaker_query_pending_since = time.time()
        if recognizing:
            self.publish_voice_reply(
                'Μισό δευτερόλεπτο, αναλύω ποιος μίλησε.',
                action='speaker_query',
            )
            return
        self.publish_voice_reply(
            self.speaker_reply_text(payload), action='speaker_query'
        )

    def face_reply(self):
        with self.lock:
            payload = dict(self.state.get('face') or {})
        state = str(payload.get('state', ''))
        if state in {'offline', 'loading', 'error'}:
            text = 'Η αναγνώριση προσώπου δεν είναι έτοιμη.'
            ok = False
        elif not payload.get('enrolled'):
            text = 'Δεν έχει εγγραφεί ακόμη κανένα πρόσωπο στο Dingo.'
            ok = False
        else:
            try:
                stale = (
                    payload.get('last_update') is None
                    or time.time() - float(payload.get('last_update')) > 5.0
                )
            except (TypeError, ValueError):
                stale = True
            faces = payload.get('faces') or []
            if stale or not faces:
                text = 'Δεν βλέπω πρόσωπο αυτή τη στιγμή.'
                ok = False
            else:
                names = [
                    str(item.get('name', '')).strip()
                    for item in faces
                    if isinstance(item, dict)
                ]
                known = [name for name in names if name and name != 'άγνωστο πρόσωπο']
                text = (
                    'Μπροστά μου βλέπω: ' + ', '.join(known) + '.'
                    if known
                    else 'Βλέπω πρόσωπο, αλλά δεν αναγνωρίζω ποιος είναι.'
                )
                ok = bool(known)
        self.publish_voice_reply(text, ok=ok, action='face_query')

    def read_vision_api_key(self):
        key = os.environ.get('GEMINI_API_KEY', '').strip()
        if not key:
            try:
                key = self.vision_llm_key_file.read_text(encoding='utf-8').strip()
            except (FileNotFoundError, OSError):
                key = ''
        if not key:
            raise RuntimeError('Δεν βρέθηκε το Gemini API key για ερωτήσεις εικόνας.')
        return key

    def start_vision_question(self, question):
        if not self.vision_llm_enabled:
            raise RuntimeError('Οι λεπτομερείς ερωτήσεις εικόνας είναι απενεργοποιημένες.')
        question = ' '.join(str(question or '').split())[:500]
        if not question:
            question = 'Περιέγραψε σύντομα τι φαίνεται στην εικόνα.'
        with self.vision_query_lock:
            if self.vision_query_active:
                raise RuntimeError('Η προηγούμενη ερώτηση εικόνας δεν έχει ολοκληρωθεί.')
            with self.lock:
                image = self.camera_bytes
                mime = self.camera_mime
                camera = dict(self.state.get('camera') or {})
                if not image:
                    raise RuntimeError('Δεν υπάρχει διαθέσιμη εικόνα από την κάμερα.')
                try:
                    stale = time.time() - float(camera.get('last_frame') or 0.0) > 5.0
                except (TypeError, ValueError):
                    stale = True
                if stale:
                    raise RuntimeError('Η εικόνα της κάμερας είναι παλιά· περίμενε το live stream.')
                self.vision_query_active = True
                self.vision_query_started_at = time.time()
                vision = dict(self.state.get('vision') or {})
                vision.update({
                    'detail_state': 'thinking',
                    'detail_question': question,
                    'detail_answer': None,
                    'detail_updated': None,
                })
                self.state['vision'] = vision
        threading.Thread(
            target=self._run_vision_question,
            args=(question, image, mime),
            daemon=True,
        ).start()
        return {'ok': True, 'queued': True, 'question': question}

    def _run_vision_question(self, question, image, mime):
        try:
            api_key = self.read_vision_api_key()
            payload = {
                'systemInstruction': {
                    'parts': [{
                        'text': (
                            'Απάντησε στα ελληνικά, σύντομα και συγκεκριμένα. '
                            'Περιέγραψε μόνο όσα φαίνονται στην εικόνα· μην επινοείς '
                            'αντικείμενα, πρόσωπα, κείμενο ή αποστάσεις. Αν δεν είσαι '
                            'βέβαιος, πες το καθαρά. Μην δίνεις εντολές κίνησης.'
                        ),
                    }],
                },
                'contents': [{
                    'role': 'user',
                    'parts': [
                        {'text': question},
                        {
                            'inline_data': {
                                'mime_type': mime or 'image/jpeg',
                                'data': base64.b64encode(image).decode('ascii'),
                            },
                        },
                    ],
                }],
                'generationConfig': {
                    'temperature': 0.1,
                    'maxOutputTokens': 256,
                    'thinkingConfig': {'thinkingBudget': 0},
                    'responseMimeType': 'text/plain',
                },
            }
            endpoint = (
                'https://generativelanguage.googleapis.com/v1beta/models/'
                f'{quote(self.vision_llm_model, safe="")}:generateContent'
            )
            request = Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'x-goog-api-key': api_key,
                },
                method='POST',
            )
            try:
                with urlopen(request, timeout=self.vision_llm_timeout_s) as response:
                    body = json.loads(response.read().decode('utf-8'))
            except HTTPError as exc:
                raise RuntimeError(
                    f'Το Gemini vision δεν είναι διαθέσιμο (HTTP {exc.code}).'
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f'Το Gemini vision δεν είναι διαθέσιμο: {exc}') from exc
            self.record_gemini_usage(
                body.get('usageMetadata'),
                request_type='vision',
                model=self.vision_llm_model,
            )
            candidates = body.get('candidates') or []
            parts = candidates[0].get('content', {}).get('parts', []) if candidates else []
            answer = next(
                (str(part.get('text')).strip() for part in parts
                 if isinstance(part, dict) and part.get('text')),
                '',
            )
            if not answer:
                raise RuntimeError('Το Gemini vision δεν επέστρεψε απάντηση.')
            with self.lock:
                vision = dict(self.state.get('vision') or {})
                vision.update({
                    'detail_state': 'ready',
                    'detail_question': question,
                    'detail_answer': answer[:1000],
                    'detail_updated': time.time(),
                })
                self.state['vision'] = vision
            self.publish_voice_reply(answer[:1000], action='vision_question')
        except Exception as exc:
            self.record_gemini_error(exc)
            with self.lock:
                vision = dict(self.state.get('vision') or {})
                vision.update({
                    'detail_state': 'error',
                    'detail_question': question,
                    'detail_answer': str(exc),
                    'detail_updated': time.time(),
                })
                self.state['vision'] = vision
            self.get_logger().warning(f'Vision question failed: {exc}')
            self.publish_voice_reply(
                f'Δεν μπόρεσα να αναλύσω την εικόνα: {exc}',
                ok=False,
                action='vision_question',
            )
        finally:
            with self.vision_query_lock:
                self.vision_query_active = False

    def publish_voice_reply(self, text, ok=True, action=None):
        payload = {
            'ok': bool(ok),
            'text': str(text),
            'action': action,
            'source': 'dashboard',
        }
        self.voice_reply_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

    def record_tool_event(
        self,
        target,
        operation,
        service=None,
        ok=True,
        detail='',
    ):
        """Store a small audit record without retaining arbitrary input.

        The event contains only normalized allow-listed names and a bounded
        result message.  User text, shell fragments and file paths are never
        copied into the gateway state.
        """
        detail = ' '.join(str(detail or '').split())[:180]
        event = {
            'action': 'system_control',
            'target': str(target or '')[:32],
            'operation': str(operation or '')[:32],
            'service': str(service or '')[:80] if service else None,
            'ok': bool(ok),
            'detail': detail,
            'at': time.time(),
        }
        with self.lock:
            gateway = dict(self.state.get('tool_gateway') or {})
            gateway.update(
                {
                    'last_action': event['action'],
                    'last_target': event['target'],
                    'last_operation': event['operation'],
                    'last_service': event['service'],
                    'last_ok': event['ok'],
                    'last_detail': event['detail'],
                    'last_at': event['at'],
                }
            )
            self.state['tool_gateway'] = gateway
        message = (
            f'Tool Gateway {event["target"]}/{event["operation"]}'
            + (f' ({event["service"]})' if event['service'] else '')
            + f': {"OK" if event["ok"] else "REJECTED"}'
        )
        if event['ok']:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)
        return event

    @classmethod
    def normalize_tool_target(cls, value):
        normalized = cls.normalize_voice_text(value)
        normalized = normalized.replace('-', '_').replace(' ', '_')
        aliases = {
            'service': 'service',
            'services': 'service',
            'υπηρεσια': 'service',
            'υπηρεσιες': 'service',
            'systemd': 'service',
            'camera': 'camera',
            'καμερα': 'camera',
            'realsense': 'camera',
            'mapping': 'mapping',
            'slam': 'mapping',
            'χαρτογραφηση': 'mapping',
            'navigation': 'navigation',
            'nav2': 'navigation',
            'πλοηγηση': 'navigation',
            'localization': 'localization',
            'relocalization': 'localization',
            'amcl': 'localization',
            'εντοπισμος': 'localization',
            'rviz': 'rviz',
            'rviz2': 'rviz',
            'map': 'map',
            'χαρτης': 'map',
        }
        return aliases.get(normalized)

    @classmethod
    def normalize_tool_operation(cls, value):
        normalized = cls.normalize_voice_text(value)
        normalized = normalized.replace('-', '_').replace(' ', '_')
        aliases = {
            'status': 'status',
            'state': 'status',
            'check': 'status',
            'ελεγχος': 'status',
            'κατασταση': 'status',
            'start': 'start',
            'run': 'start',
            'enable': 'start',
            'ξεκινα': 'start',
            'εκκινηση': 'start',
            'ενεργοποιηση': 'start',
            'stop': 'stop',
            'close': 'stop',
            'disable': 'stop',
            'σταματημα': 'stop',
            'κλεισιμο': 'stop',
            'restart': 'restart',
            'relaunch': 'restart',
            'επανεκκινηση': 'restart',
            'cancel': 'cancel',
            'ακυρωση': 'cancel',
            'global_localization': 'global_localization',
            'global': 'global_localization',
            'relocalize': 'global_localization',
            'save': 'save',
            'αποθηκευση': 'save',
        }
        return aliases.get(normalized)

    @classmethod
    def normalize_tool_service(cls, value):
        requested = str(value or '').strip().lower()
        if requested in cls.TOOL_SERVICE_ALLOWLIST:
            return requested
        normalized = cls.normalize_voice_text(value)
        normalized = normalized.replace('-', '_').replace(' ', '_')
        aliases = {
            'voice': 'dingo-voice.service',
            'φωνη': 'dingo-voice.service',
            'microphone': 'dingo-voice.service',
            'μικροφωνο': 'dingo-voice.service',
            'alexa': 'dingo-voice.service',
            'respeaker': 'dingo-mic-array.service',
            'qwen': 'dingo-local-llm.service',
            'llm': 'dingo-local-llm.service',
            'local_llm': 'dingo-local-llm.service',
            'fastflow': 'dingo-local-llm.service',
            'dashboard': 'dingo-dashboard.service',
            'ντασμπορντ': 'dingo-dashboard.service',
            'sensors': 'dingo-sensors.service',
            'αισθητηρες': 'dingo-sensors.service',
            'lidar': 'dingo-sensors.service',
            'imu': 'dingo-sensors.service',
            'yolo': 'dingo-object-detector.service',
            'object_detector': 'dingo-object-detector.service',
            'vision': 'dingo-object-detector.service',
            'face': 'dingo-face-recognition.service',
            'face_recognition': 'dingo-face-recognition.service',
            'speaker': 'dingo-face-recognition.service',
            'mic_array': 'dingo-mic-array.service',
            'xvf3800': 'dingo-mic-array.service',
            'xmos': 'dingo-mic-array.service',
        }
        return aliases.get(normalized)

    @staticmethod
    def systemctl_command(*args):
        systemctl = shutil.which('systemctl') or '/usr/bin/systemctl'
        return [systemctl, '--user', *args]

    @staticmethod
    def systemctl_environment():
        environment = os.environ.copy()
        runtime_dir = environment.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}'
        environment.setdefault(
            'DBUS_SESSION_BUS_ADDRESS', f'unix:path={runtime_dir}/bus'
        )
        return environment

    def tool_service_status(self, requested):
        service = self.normalize_tool_service(requested)
        if service is None:
            raise RuntimeError('Η υπηρεσία δεν είναι στη allow-list του Tool Gateway.')
        try:
            active_result = subprocess.run(
                self.systemctl_command('is-active', service),
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                env=self.systemctl_environment(),
            )
            enabled_result = subprocess.run(
                self.systemctl_command('is-enabled', service),
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                env=self.systemctl_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'Δεν μπόρεσα να ελέγξω την υπηρεσία: {exc}') from exc
        active = (active_result.stdout or active_result.stderr).strip().splitlines()
        enabled = (enabled_result.stdout or enabled_result.stderr).strip().splitlines()
        active_state = active[0] if active else 'unknown'
        enabled_state = enabled[0] if enabled else 'unknown'
        if active_state.lower().startswith('failed to connect'):
            active_state = 'unknown'
        if enabled_state.lower().startswith('failed to connect'):
            enabled_state = 'unknown'
        return {
            'service': service,
            'label': self.TOOL_SERVICE_LABELS.get(service, service),
            'active': active_state,
            'enabled': enabled_state,
            'error': (
                (active[0] if active else '')
                if active_state == 'unknown' and active_result.returncode != 0
                else None
            ),
        }

    def tool_services_status(self):
        return [self.tool_service_status(service) for service in self.TOOL_SERVICE_ALLOWLIST]

    def tool_service_control(self, requested, operation):
        service = self.normalize_tool_service(requested)
        if service is None:
            raise RuntimeError('Η υπηρεσία δεν είναι στη allow-list του Tool Gateway.')
        if operation not in {'start', 'restart'}:
            raise RuntimeError(
                'Για υπηρεσίες επιτρέπονται μόνο κατάσταση, εκκίνηση ή επανεκκίνηση. '
                'Το stop υπηρεσίας παραμένει κλειδωμένο.'
            )
        if service == 'dingo-dashboard.service':
            raise RuntimeError(
                'Το Dashboard δεν επανεκκινείται από τη φωνή. Χρησιμοποίησε το ίδιο το Dashboard.'
            )
        try:
            result = subprocess.run(
                self.systemctl_command(operation, service),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                env=self.systemctl_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'Η ενέργεια στην υπηρεσία απέτυχε: {exc}') from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError(
                detail[0][:180] if detail else f'Το systemd επέστρεψε κωδικό {result.returncode}.'
            )
        return service

    def system_control_reply(self, payload):
        """Execute one validated Tool Gateway action and speak its result."""
        target = self.normalize_tool_target(payload.get('target'))
        operation = self.normalize_tool_operation(payload.get('operation')) or 'status'
        raw_service = str(payload.get('service') or '').strip()
        service = self.normalize_tool_service(raw_service)
        map_name = str(payload.get('map_name') or '').strip()[:60]
        if target is None and service is not None:
            target = 'service'
        if raw_service and service is None:
            self.record_tool_event(
                target or 'unknown', operation, ok=False, detail='service not allow-listed'
            )
            self.publish_voice_reply(
                'Απέρριψα υπηρεσία που δεν υπάρχει στη allow-list.',
                ok=False,
                action='system_control',
            )
            return
        if target not in self.TOOL_TARGETS:
            self.record_tool_event('unknown', operation, ok=False, detail='unknown target')
            self.publish_voice_reply(
                'Απέρριψα άγνωστο στόχο Tool Gateway.', ok=False, action='system_control'
            )
            return
        if operation not in self.TOOL_OPERATIONS:
            self.record_tool_event(target, operation, service=service, ok=False, detail='unknown operation')
            self.publish_voice_reply(
                'Απέρριψα άγνωστη ενέργεια Tool Gateway.', ok=False, action='system_control'
            )
            return

        try:
            if target == 'service':
                if operation == 'status':
                    if service:
                        status = self.tool_service_status(service)
                        active = status['active']
                        reply = (
                            f'{status["label"]}: {active}. '
                            f'Ενεργοποίηση στην εκκίνηση: {status["enabled"]}.'
                        )
                    else:
                        statuses = self.tool_services_status()
                        parts = [
                            f'{item["label"]}: {item["active"]}'
                            for item in statuses
                        ]
                        reply = 'Υπηρεσίες Dingo: ' + '; '.join(parts) + '.'
                    self.record_tool_event(
                        target, operation, service=service, ok=True, detail=reply
                    )
                    self.publish_voice_reply(reply, action='system_control')
                    return
                if service is None:
                    raise RuntimeError(
                        'Πες ποια υπηρεσία να ξεκινήσω ή να επανεκκινήσω, για παράδειγμα «το Qwen».'
                    )
                self.tool_service_control(service, operation)
                reply = (
                    f'Έγινε {"επανεκκίνηση" if operation == "restart" else "εκκίνηση"} '
                    f'της {self.TOOL_SERVICE_LABELS.get(service, service)}.'
                )
            elif target == 'camera':
                if operation == 'status':
                    status = self.camera_status()
                    reply = (
                        'Η κάμερα είναι '
                        + ('διαθέσιμη' if status.get('available') else 'μη διαθέσιμη')
                        + '. '
                        + ('Το stream τρέχει.' if status.get('process_running') else 'Το stream είναι κλειστό.')
                    )
                elif operation == 'start':
                    changed = self.start_camera()
                    reply = 'Η κάμερα ξεκινά.' if changed else 'Η κάμερα είναι ήδη ενεργή από το Dingo stack.'
                elif operation == 'stop':
                    changed = self.stop_camera()
                    reply = 'Η κάμερα σταμάτησε.' if changed else 'Δεν υπάρχει Dashboard camera process για σταμάτημα.'
                elif operation == 'restart':
                    self.stop_camera()
                    changed = self.start_camera()
                    reply = 'Η κάμερα επανεκκινήθηκε.' if changed else 'Η κάμερα ανήκει ήδη στο βασικό Dingo stack.'
                else:
                    raise RuntimeError('Για την κάμερα επιτρέπονται κατάσταση, start, stop ή restart.')
            elif target == 'mapping':
                if operation == 'status':
                    status = self.mapping_snapshot()
                    reply = (
                        f'SLAM: {"ενεργό" if status["running"] else "κλειστό"}. '
                        f'Χάρτης διαθέσιμος: {"ναι" if status["map_available"] else "όχι"}.'
                    )
                elif operation == 'start':
                    changed = self.start_mapping()
                    reply = 'Η χαρτογράφηση ξεκίνησε.' if changed else 'Η χαρτογράφηση είναι ήδη ενεργή.'
                elif operation == 'stop':
                    changed = self.stop_mapping()
                    reply = 'Η χαρτογράφηση σταμάτησε.' if changed else 'Η χαρτογράφηση ήταν ήδη κλειστή.'
                elif operation == 'restart':
                    self.stop_mapping()
                    changed = self.start_mapping()
                    reply = 'Η χαρτογράφηση επανεκκινήθηκε.' if changed else 'Η χαρτογράφηση δεν ξεκίνησε.'
                else:
                    raise RuntimeError('Για το SLAM επιτρέπονται κατάσταση, start, stop ή restart.')
            elif target == 'navigation':
                if operation == 'status':
                    status = self.navigation_snapshot()
                    reply = (
                        f'Nav2: {status.get("status", "unknown")}. '
                        f'Εντοπισμένο: {"ναι" if status.get("localized") else "όχι"}. '
                        f'Διαδρομή: {"διαθέσιμη" if status.get("path") else "όχι"}.'
                    )
                elif operation == 'start':
                    changed = self.start_navigation(map_name or None)
                    reply = 'Το Nav2 ξεκινά χωρίς να κινήσει το Dingo.' if changed else 'Το Nav2 είναι ήδη ενεργό.'
                elif operation == 'stop':
                    changed = self.stop_navigation()
                    reply = 'Η αυτόνομη πλοήγηση σταμάτησε.' if changed else 'Το Nav2 ήταν ήδη κλειστό.'
                elif operation == 'cancel':
                    changed = self.cancel_navigation()
                    reply = 'Ο ενεργός στόχος ακυρώθηκε.' if changed else 'Δεν υπήρχε ενεργός στόχος.'
                elif operation == 'restart':
                    current_map = self.navigation_snapshot().get('map') or map_name or None
                    self.stop_navigation()
                    changed = self.start_navigation(current_map)
                    reply = 'Το Nav2 επανεκκινήθηκε χωρίς εντολή κίνησης.' if changed else 'Το Nav2 δεν ξεκίνησε.'
                else:
                    raise RuntimeError('Για το Nav2 επιτρέπονται κατάσταση, start, stop, cancel ή restart.')
            elif target == 'localization':
                if operation == 'status':
                    status = self.navigation_snapshot()
                    reply = (
                        'Το localization είναι '
                        + ('σταθερό.' if status.get('localized') else 'σε αναμονή ή δεν έχει επιβεβαιωθεί.')
                    )
                elif operation in {'start', 'global_localization'}:
                    changed = self.start_global_localization(allow_motion=False)
                    reply = (
                        'Ξεκίνησε επίσημο AMCL global localization χωρίς περιστροφή του ρομπότ.'
                        if changed else 'Το localization είναι ήδη σε εξέλιξη.'
                    )
                else:
                    raise RuntimeError('Για localization επιτρέπονται κατάσταση ή global localization.')
            elif target == 'rviz':
                if operation == 'status':
                    status = self.rviz_status()
                    reply = 'Το RViz2 είναι ' + ('ανοιχτό.' if status.get('process_running') else 'κλειστό.')
                elif operation == 'start':
                    changed = self.start_rviz()
                    reply = 'Το RViz2 ανοίγει.' if changed else 'Το RViz2 είναι ήδη ανοιχτό.'
                elif operation == 'stop':
                    changed = self.stop_rviz()
                    reply = 'Το RViz2 έκλεισε.' if changed else 'Το RViz2 ήταν ήδη κλειστό.'
                elif operation == 'restart':
                    self.stop_rviz()
                    changed = self.start_rviz()
                    reply = 'Το RViz2 επανεκκινήθηκε.' if changed else 'Το RViz2 δεν ξεκίνησε.'
                else:
                    raise RuntimeError('Για RViz2 επιτρέπονται κατάσταση, start, stop ή restart.')
            elif target == 'map':
                if operation == 'status':
                    available = self.map_snapshot() is not None
                    reply = 'Υπάρχει ενεργός χάρτης.' if available else 'Δεν υπάρχει ενεργός χάρτης.'
                elif operation == 'save':
                    result = self.save_map(map_name or 'dingo_map_voice')
                    reply = f'Ο χάρτης αποθηκεύεται ως «{result["name"]}». '
                else:
                    raise RuntimeError('Για τον χάρτη επιτρέπονται κατάσταση ή αποθήκευση.')
            else:
                raise RuntimeError('Ο στόχος δεν υποστηρίζεται.')
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            detail = str(exc)[:180]
            self.record_tool_event(
                target, operation, service=service, ok=False, detail=detail
            )
            self.publish_voice_reply(detail, ok=False, action='system_control')
            return

        self.record_tool_event(
            target, operation, service=service, ok=True, detail=reply
        )
        self.publish_voice_reply(reply, action='system_control')

    def set_voice_direction_enabled(self, enabled):
        """Enable/disable DoA display from the Dashboard only.

        This setting does not publish velocity and therefore cannot turn the
        base by itself.  It only gates the XVF3800 direction/VAD indication.
        """
        value = bool(enabled)
        self.voice_direction_enable_pub.publish(Bool(data=value))
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['direction_enabled'] = value
            self.state['voice'] = current
        return value

    def set_voice_provider(self, requested):
        """Select the Dashboard assistant's local LLM or Gemini at runtime."""
        value = self.normalize_voice_text(requested).replace('-', '_').replace(' ', '_')
        aliases = {
            'local': 'flm',
            'qwen': 'flm',
            'qwen3_5': 'flm',
            'flm': 'flm',
            'gemini': 'gemini',
            'gemini_2_5_flash': 'gemini',
        }
        provider = aliases.get(value)
        if provider is None:
            raise ValueError('Άγνωστη επιλογή LLM. Διάλεξε local ή Gemini.')
        discovery_deadline = time.monotonic() + 2.0
        while (
            self.voice_provider_pub.get_subscription_count() < 1
            and time.monotonic() < discovery_deadline
        ):
            time.sleep(0.1)
        if self.voice_provider_pub.get_subscription_count() < 1:
            raise RuntimeError(
                'Το voice assistant δεν είναι συνδεδεμένο για αλλαγή LLM.'
            )
        self.voice_provider_pub.publish(
            String(data=json.dumps({'provider': provider}, ensure_ascii=False))
        )
        model = 'gemini-2.5-flash' if provider == 'gemini' else 'qwen3.5:9b'
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['llm_provider'] = provider
            current['llm_model'] = model
            self.state['voice'] = current
        return {'provider': provider, 'model': model}

    def set_native_tool_calling(self, enabled):
        """Enable/disable native function calls in the voice assistant."""
        value = bool(enabled)
        discovery_deadline = time.monotonic() + 2.0
        while (
            self.voice_provider_pub.get_subscription_count() < 1
            and time.monotonic() < discovery_deadline
        ):
            time.sleep(0.1)
        if self.voice_provider_pub.get_subscription_count() < 1:
            raise RuntimeError(
                'Το voice assistant δεν είναι συνδεδεμένο για αλλαγή tool calling.'
            )
        self.voice_provider_pub.publish(
            String(data=json.dumps({
                'native_tool_calling': value,
            }, ensure_ascii=False))
        )
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['llm_tool_calling'] = (
                'native' if value else 'json_fallback'
            )
            self.state['voice'] = current
        return {
            'native_tool_calling': value,
            'llm_tool_calling': 'native' if value else 'json_fallback',
        }

    def submit_text_command(self, text):
        text = ' '.join(str(text or '').split())
        if not text:
            raise ValueError('Γράψε πρώτα μια ερώτηση ή εντολή.')
        if len(text) > 500:
            raise ValueError('Η ερώτηση είναι πολύ μεγάλη (μέχρι 500 χαρακτήρες).')
        discovery_deadline = time.monotonic() + 2.0
        while (
            self.voice_text_pub.get_subscription_count() < 1
            and time.monotonic() < discovery_deadline
        ):
            time.sleep(0.1)
        if self.voice_text_pub.get_subscription_count() < 1:
            raise RuntimeError(
                'Το Dingo voice assistant δεν είναι συνδεδεμένο. '
                'Έλεγξε το dingo-voice.service.'
            )
        with self.lock:
            current = dict(self.state.get('voice') or {})
            current['last_text_command'] = text
            self.state['voice'] = current
        self.voice_text_pub.publish(
            String(
                data=json.dumps(
                    {'text': text, 'source': 'dashboard_text'},
                    ensure_ascii=False,
                )
            )
        )
        return {'ok': True, 'queued': True, 'text': text}

    def publish_identity_command(self, publisher, payload, label):
        if publisher.get_subscription_count() < 1:
            raise RuntimeError(f'Η υπηρεσία {label} δεν είναι συνδεδεμένη.')
        publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        return {'ok': True, 'queued': True, 'action': payload.get('action')}

    def submit_face_command(self, payload):
        return self.publish_identity_command(
            self.face_command_pub, payload, 'face recognition'
        )

    def submit_speaker_command(self, payload):
        return self.publish_identity_command(
            self.speaker_command_pub, payload, 'speaker recognition'
        )

    def voice_room(self, requested):
        requested = self.normalize_voice_text(requested)
        with self.lock:
            rooms = list(self.rooms)
        matches = [
            room for room in rooms
            if self.normalize_voice_text(room.get('name')) == requested
        ]
        return matches[0] if len(matches) == 1 else None

    def voice_command(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.publish_voice_reply('Απορρίφθηκε μη έγκυρη voice εντολή.', ok=False)
            return
        if not isinstance(payload, dict):
            self.publish_voice_reply('Απορρίφθηκε μη έγκυρη voice εντολή.', ok=False)
            return

        action = str(payload.get('action', '')).strip()
        if action == 'stop':
            self.stop_follow(silent=True)
            self.cancel_patrol()
            self.cancel_drive_heading()
            self.cancel_spin()
            self.cancel_navigation()
            self.drive(0.0, 0.0)
            self.publish_voice_reply('Το Dingo σταμάτησε.', action='stop')
            return

        if action == 'move_distance':
            if payload.get('confirmed') is not True:
                self.publish_voice_reply(
                    'Η κίνηση σε μέτρα απαιτεί επιβεβαίωση.',
                    ok=False,
                    action=action,
                )
                return
            try:
                distance_m = float(payload.get('distance_m'))
                started = self.start_drive_heading(distance_m)
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                if started:
                    direction = 'μπροστά' if distance_m > 0.0 else 'πίσω'
                    self.publish_voice_reply(
                        f'Κινούμαι {abs(distance_m):g} μέτρα {direction}.',
                        action=action,
                    )
            return

        if action == 'navigate_room':
            # Room navigation is direct. The safety/emergency-stop,
            # localization, Nav2 and active-goal checks below remain mandatory.
            room = self.voice_room(payload.get('room'))
            if room is None:
                self.publish_voice_reply(
                    f'Δεν βρέθηκε αποθηκευμένο δωμάτιο «{payload.get("room", "") }».',
                    ok=False,
                    action=action,
                )
                return
            with self.lock:
                blocked = (
                    self.state.get('emergency_stop') is True
                    or self.state.get('safety_stop') is True
                )
            if blocked:
                self.publish_voice_reply(
                    'Η κίνηση είναι μπλοκαρισμένη από το safety ή emergency stop.',
                    ok=False,
                    action=action,
                )
                return
            try:
                self.send_navigation_goal(room['x'], room['y'])
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                self.publish_voice_reply(
                    f'Πηγαίνω στο «{room["name"]}».', action=action
                )
            return

        if action == 'system_info':
            self.system_info_reply(payload.get('target', 'summary'))
            return

        if action == 'system_control':
            self.system_control_reply(payload)
            return

        if action == 'vision':
            self.vision_reply()
            return

        if action == 'vision_question':
            try:
                self.start_vision_question(payload.get('question'))
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                self.publish_voice_reply(
                    'Πήρα την εικόνα· την αναλύω τώρα.', action=action
                )
            return

        if action == 'face_query':
            self.face_reply()
            return

        if action == 'speaker_query':
            self.speaker_reply()
            return

        if action == 'follow_start':
            if payload.get('confirmed') is not True:
                self.publish_voice_reply(
                    'Το follow-me απαιτεί επιβεβαίωση.', ok=False, action=action
                )
                return
            try:
                started = self.start_follow()
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                if started:
                    self.publish_voice_reply(
                        'Ξεκινώ follow-me με χαμηλή ταχύτητα. Θα σταματήσω αν σε χάσω.',
                        action=action,
                    )
            return

        if action == 'follow_stop':
            self.stop_follow()
            return

        if action == 'patrol':
            if payload.get('confirmed') is not True:
                self.publish_voice_reply(
                    'Η βόλτα απαιτεί φωνητική επιβεβαίωση.',
                    ok=False,
                    action=action,
                )
                return
            try:
                started = self.start_patrol(payload.get('rounds', 1))
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                if started:
                    with self.lock:
                        total = self.patrol_total
                    self.publish_voice_reply(
                        f'Ξεκινάω βόλτα σε {total} αποθηκευμένα σημεία.',
                        action=action,
                    )
            return

        if action == 'rotate':
            if payload.get('confirmed') is not True:
                self.publish_voice_reply(
                    'Η στροφή απαιτεί φωνητική επιβεβαίωση.',
                    ok=False,
                    action=action,
                )
                return
            try:
                degrees = float(payload.get('degrees', 360.0))
                started = self.start_spin(degrees)
            except (RuntimeError, ValueError) as exc:
                self.publish_voice_reply(str(exc), ok=False, action=action)
            else:
                if started:
                    side = 'αριστερά' if degrees >= 0 else 'δεξιά'
                    self.publish_voice_reply(
                        f'Στρίβω {abs(degrees):g} μοίρες {side}.',
                        action=action,
                    )
            return

        if action == 'status':
            navigation = self.navigation_snapshot()
            status = navigation.get('status', 'unknown')
            localized = 'εντοπισμένο' if navigation.get('localized') else 'χωρίς επιβεβαιωμένη θέση'
            self.publish_voice_reply(
                f'Nav2: {status}. Το Dingo είναι {localized}.', action=action
            )
            return

        if action == 'battery':
            with self.lock:
                battery = dict(self.state.get('battery') or {})
                power = dict(self.state.get('power') or {})
            percent = battery.get('percent')
            watts = power.get('total_power_w') or battery.get('power_w')
            if percent is None and watts is None:
                self.publish_voice_reply('Δεν έχω ακόμη διαθέσιμη μέτρηση μπαταρίας.', ok=False, action=action)
            else:
                parts = []
                if percent is not None:
                    parts.append(f'{percent:.0f}%')
                if watts is not None:
                    parts.append(f'{watts:.1f} Watt')
                self.publish_voice_reply(
                    f'Η μπαταρία δείχνει {" και ".join(parts)}.', action=action
                )
            return

        if action == 'where':
            navigation = self.navigation_snapshot()
            pose = navigation.get('amcl_pose') or {}
            if not pose:
                self.publish_voice_reply('Δεν έχω επιβεβαιωμένη θέση στον χάρτη.', ok=False, action=action)
            else:
                self.publish_voice_reply(
                    f'Είμαι στη θέση x {pose.get("x")}, y {pose.get("y")}.',
                    action=action,
                )
            return

        self.publish_voice_reply(
            f'Η voice action «{action}» δεν επιτρέπεται.',
            ok=False,
            action=action,
        )

    def odom(self, msg):
        now = time.time()
        with self.lock:
            self.state['connected'] = True
            self.state['odom'] = {
                'x': round(msg.pose.pose.position.x, 3),
                'y': round(msg.pose.pose.position.y, 3),
                'linear': round(msg.twist.twist.linear.x, 3),
                'angular': round(msg.twist.twist.angular.z, 3),
                'last_update': now,
            }

    def amcl_pose(self, msg):
        rotation = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        covariance = {
            'x': round(float(msg.pose.covariance[0]), 3),
            'y': round(float(msg.pose.covariance[7]), 3),
            'yaw': round(float(msg.pose.covariance[35]), 3),
        }
        pose = {
            'x': round(float(msg.pose.pose.position.x), 3),
            'y': round(float(msg.pose.pose.position.y), 3),
            'yaw': round(yaw, 4),
            'frame': msg.header.frame_id or 'map',
        }
        with self.lock:
            searching = self.localization_search_active
            service_ready = self.localization_global_service_response
            if self.navigation_initial_pose is not None and not searching:
                self.navigation_initial_pose = None
                self.navigation_initial_pose_sent_at = 0.0
            self.navigation_amcl_pose = pose
            self.navigation_amcl_covariance = covariance
            self.navigation_amcl_received_at = time.monotonic()
            self.state['localization'] = {
                'amcl_available': True,
                'localized': bool(self.amcl_pose_is_good(covariance)),
                'pose': dict(pose),
                'covariance': dict(covariance),
                'tf_map_base_link': None,
                'tf_ok': False,
                'last_update': time.time(),
            }
            if searching and service_ready:
                if self.amcl_pose_is_good(covariance):
                    previous = self.localization_last_pose
                    if previous is None:
                        self.localization_good_count = 1
                    elif (
                        math.hypot(pose['x'] - previous['x'], pose['y'] - previous['y'])
                        <= 0.35
                        and abs(self._angle_difference(pose['yaw'], previous['yaw']))
                        <= 0.35
                    ):
                        self.localization_good_count += 1
                    else:
                        self.localization_good_count = 1
                    self.localization_last_pose = pose
                else:
                    self.localization_good_count = 0
                    self.localization_last_pose = None
                required_good = self.localization_good_required
                if self.localization_method == 'scan_match_fallback':
                    required_good = min(
                        required_good,
                        self.localization_scan_match_good_required,
                    )
                localized = self.localization_good_count >= required_good
            else:
                self.localization_good_count = 0
                self.localization_last_pose = None
                localized = False
        if localized:
            # AMCL can occasionally converge to a low-covariance hypothesis
            # which is geometrically impossible (for example with part of
            # the robot footprint inside a mapped wall).  Such a pose makes
            # Nav2 return NO_VALID_PATH immediately, so do not mark it as a
            # usable localization.  Leave the global-localization fallback
            # running so the LiDAR matcher can seed a safer hypothesis.
            if self.map_pose_is_safe(pose):
                self.finish_global_localization(True)
            else:
                with self.lock:
                    if self.localization_search_active:
                        self.localization_good_count = 0
                        self.localization_last_pose = None
                        self.navigation_feedback = {
                            'message': 'Το AMCL βρήκε θέση πάνω σε εμπόδιο· συνεχίζω την αναζήτηση με LiDAR',
                            'method': self.localization_method or 'amcl_global',
                        }

    @staticmethod
    def amcl_pose_is_good(covariance):
        return (
            covariance is not None
            and covariance['x'] <= 0.75
            and covariance['y'] <= 0.75
            and covariance['yaw'] <= 0.75
        )

    def map_pose_is_safe(self, pose, padding_m=0.0):
        """Check that the Dingo footprint lies in known free map cells.

        AMCL covariance alone is not enough to authorize autonomous motion:
        a false global hypothesis can still have a small covariance.  This
        check is intentionally conservative and uses the same DD100 footprint
        used by the Nav2 costmaps.  Unknown cells are rejected as well because
        the robot cannot safely start navigation from an unmapped area.
        """
        if not pose:
            return False
        try:
            x = float(pose['x'])
            y = float(pose['y'])
            yaw = float(pose.get('yaw', 0.0))
        except (KeyError, TypeError, ValueError):
            return False
        with self.lock:
            map_state = self.map_state
        if not map_state:
            return False
        try:
            width = int(map_state['width'])
            height = int(map_state['height'])
            resolution = float(map_state['resolution'])
            origin = map_state['origin']
            origin_x = float(origin['x'])
            origin_y = float(origin['y'])
            data = map_state['data']
        except (KeyError, TypeError, ValueError):
            return False
        if (
            width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(data) < width * height
        ):
            return False

        # DD100 footprint: 0.551 m x 0.517 m.  Do not add a second artificial
        # margin here: the map is 5 cm/cell and Nav2's costmap/inflation layer
        # performs the actual obstacle clearance check.  An extra 2.5 cm
        # margin caused valid AMCL poses to be rejected due to cell rounding.
        half_length = 0.551 / 2.0 + float(padding_m)
        half_width = 0.517 / 2.0 + float(padding_m)
        step = max(resolution / 2.0, 0.025)
        sample_count_x = int(math.ceil((2.0 * half_length) / step))
        sample_count_y = int(math.ceil((2.0 * half_width) / step))
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        for ix in range(sample_count_x + 1):
            local_x = -half_length + (2.0 * half_length * ix / sample_count_x)
            for iy in range(sample_count_y + 1):
                local_y = -half_width + (2.0 * half_width * iy / sample_count_y)
                world_x = x + cosine * local_x - sine * local_y
                world_y = y + sine * local_x + cosine * local_y
                cell_x = int(math.floor((world_x - origin_x) / resolution))
                cell_y = int(math.floor((world_y - origin_y) / resolution))
                if not (0 <= cell_x < width and 0 <= cell_y < height):
                    return False
                value = int(data[cell_y * width + cell_x])
                if value < 0 or value >= 65:
                    return False
        return True

    def scan(self, msg):
        max_display_range = min(
            float(msg.range_max) if msg.range_max > 0 else 8.0, 8.0
        )
        min_range = max(float(msg.range_min), 0.05)
        step = max(1, math.ceil(len(msg.ranges) / 720))
        points = []
        for index in range(0, len(msg.ranges), step):
            distance = float(msg.ranges[index])
            if (
                not math.isfinite(distance)
                or distance < min_range
                or distance > max_display_range
            ):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            points.append(
                [
                    round(distance * math.cos(angle), 3),
                    round(distance * math.sin(angle), 3),
                ]
            )
        with self.lock:
            self.state['connected'] = True
            self.state['scan'] = {
                'count': len(msg.ranges),
                'valid': len(points),
                'frame': msg.header.frame_id,
                'stamp': {
                    'sec': int(msg.header.stamp.sec),
                    'nanosec': int(msg.header.stamp.nanosec),
                },
                'range_max': round(max_display_range, 2),
                'points': points,
                'last_update': time.time(),
            }

    def tf_dynamic(self, msg):
        if msg.transforms:
            self.tf_last_update = time.time()
        for transform in msg.transforms:
            try:
                self.tf_buffer.set_transform(transform, 'dingo_dashboard')
            except (TypeError, ValueError):
                continue

    def tf_static(self, msg):
        if msg.transforms and self.tf_last_update <= 0.0:
            self.tf_last_update = time.time()
        for transform in msg.transforms:
            try:
                self.tf_buffer.set_transform_static(transform, 'dingo_dashboard')
            except (TypeError, ValueError):
                continue

    def camera_compressed(self, msg):
        self.store_camera(
            bytes(msg.data),
            self.mime_for_format(msg.format),
            self.camera_topic,
            msg.header.frame_id,
        )

    def camera_raw(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            compressed = self.bridge.cv2_to_compressed_imgmsg(image, dst_format='jpg')
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.get_logger().warning(f'Could not convert camera frame: {exc}')
            return
        self.store_camera(
            bytes(compressed.data),
            'image/jpeg',
            self.camera_raw_topic,
            msg.header.frame_id,
        )

    def store_camera(self, data, mime, topic, frame):
        if not data:
            return
        with self.lock:
            self.camera_bytes = data
            self.camera_mime = mime
            self.state['camera'] = {
                'available': True,
                'topic': topic,
                'frame': frame,
                'last_frame': time.time(),
            }

    @staticmethod
    def mime_for_format(fmt):
        value = (fmt or '').lower()
        if 'png' in value:
            return 'image/png'
        if 'webp' in value:
            return 'image/webp'
        return 'image/jpeg'

    def emergency_stop(self, msg):
        with self.lock:
            self.state['emergency_stop'] = bool(msg.data)

    def safety_stop(self, msg):
        with self.lock:
            self.state['safety_stop'] = bool(msg.data)

    def navigation_plan(self, msg):
        self.update_navigation_path_from_topic(msg)

    def navigation_plan_smoothed(self, msg):
        self.update_navigation_path_from_topic(msg)

    def update_navigation_path_from_topic(self, msg):
        """Keep the live planner path visible even when ComputePathToPose is absent."""
        points = []
        for pose_stamped in msg.poses or []:
            point = pose_stamped.pose.position
            point_x = float(point.x)
            point_y = float(point.y)
            if math.isfinite(point_x) and math.isfinite(point_y):
                points.append([round(point_x, 3), round(point_y, 3)])
        with self.lock:
            if self.navigation_goal is None or self.navigation_status not in (
                'sending',
                'navigating',
            ):
                return
            if not points:
                return
            if math.hypot(
                points[-1][0] - self.navigation_goal['x'],
                points[-1][1] - self.navigation_goal['y'],
            ) > 1.0:
                return
            self.navigation_path = points
            self.navigation_path_status = 'ready'
            self.navigation_path_error = None

    def map(self, msg):
        with self.lock:
            self.map_state = {
                'width': msg.info.width,
                'height': msg.info.height,
                'resolution': msg.info.resolution,
                'origin': {
                    'x': msg.info.origin.position.x,
                    'y': msg.info.origin.position.y,
                },
                'frame': msg.header.frame_id or 'map',
                'data': list(msg.data),
            }

    def lookup_transform(
        self, target_frame, source_frame, stamp=None, allow_latest_fallback=True
    ):
        target_frame = str(target_frame or '').strip('/')
        source_frame = str(source_frame or '').strip('/')
        if not target_frame or not source_frame:
            return None
        used_latest_fallback = False
        try:
            if stamp:
                stamp_time = Time(
                    seconds=int(stamp.get('sec', 0)),
                    nanoseconds=int(stamp.get('nanosec', 0)),
                )
            else:
                stamp_time = Time()
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp_time,
            )
        except (TransformException, TypeError, ValueError):
            if stamp is None or not allow_latest_fallback:
                return None
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                used_latest_fallback = True
            except (TransformException, TypeError, ValueError):
                return None

        transform = transform_stamped.transform
        rotation = transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        result = {
            'x': round(float(transform.translation.x), 3),
            'y': round(float(transform.translation.y), 3),
            'yaw': round(yaw, 4),
            'frame': target_frame,
        }
        if stamp is not None:
            requested_ns = (
                int(stamp.get('sec', 0)) * 1_000_000_000
                + int(stamp.get('nanosec', 0))
            )
            actual_ns = (
                int(transform_stamped.header.stamp.sec) * 1_000_000_000
                + int(transform_stamped.header.stamp.nanosec)
            )
            result['transform_age_s'] = round(
                (requested_ns - actual_ns) / 1_000_000_000.0, 4
            )
            result['used_latest_fallback'] = used_latest_fallback
        return result

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            if state.get('odom') is not None:
                state['odom'] = dict(state['odom'])
            if state.get('battery') is not None:
                state['battery'] = dict(state['battery'])
            if state.get('power') is not None:
                state['power'] = dict(state['power'])
            if state.get('temperature') is not None:
                state['temperature'] = dict(state['temperature'])
            if state.get('vision') is not None:
                state['vision'] = dict(state['vision'])
                state['vision']['objects'] = [
                    dict(item) for item in (state['vision'].get('objects') or [])
                    if isinstance(item, dict)
                ]
            if state.get('face') is not None:
                state['face'] = dict(state['face'])
                state['face']['faces'] = [
                    dict(item) for item in (state['face'].get('faces') or [])
                    if isinstance(item, dict)
                ]
            if state.get('speaker') is not None:
                state['speaker'] = dict(state['speaker'])
            if state.get('follow') is not None:
                state['follow'] = dict(state['follow'])
                if isinstance(state['follow'].get('target'), dict):
                    state['follow']['target'] = dict(state['follow']['target'])
            if state.get('gemini') is not None:
                state['gemini'] = dict(state['gemini'])
                if isinstance(state['gemini'].get('last_usage'), dict):
                    state['gemini']['last_usage'] = dict(state['gemini']['last_usage'])
            if state.get('scan') is not None:
                state['scan'] = dict(state['scan'])
            if state.get('imu') is not None:
                state['imu'] = dict(state['imu'])
                for key in ('orientation', 'angular_velocity', 'linear_acceleration'):
                    if isinstance(state['imu'].get(key), dict):
                        state['imu'][key] = dict(state['imu'][key])
            if state.get('localization') is not None:
                state['localization'] = dict(state['localization'])
                for key in ('pose', 'covariance', 'tf_map_base_link'):
                    if isinstance(state['localization'].get(key), dict):
                        state['localization'][key] = dict(state['localization'][key])
            if state.get('voice') is not None:
                state['voice'] = dict(state['voice'])
            if state.get('tool_gateway') is not None:
                state['tool_gateway'] = dict(state['tool_gateway'])

        battery = state.get('battery') or {}
        power = state.get('power') or {}
        battery_power = battery.get('power_w')
        total_power = (
            power.get('total_power_w')
            if power.get('total_power_w') is not None
            else battery_power
        )
        total_current = power.get('total_current')
        if total_current is None:
            total_current = battery.get('current')
        power_topic_available = (
            power.get('total_power_w') is not None
            or power.get('computer_power_w') is not None
            or power.get('total_current') is not None
            or power.get('computer_current') is not None
        )
        state['power'] = {
            'available': total_power is not None or power.get('computer_power_w') is not None,
            'total_w': total_power,
            'total_current': total_current,
            # The Power message exposes a computer/12 V rail current. It does
            # not expose the MCU chip's own consumption, so do not label this
            # value as MCU power in the dashboard.
            'computer_w': power.get('computer_power_w'),
            'computer_current': power.get('computer_current'),
            'computer_voltage': power.get('computer_voltage'),
            'mcu_direct_w': None,
            'battery_w': battery_power,
            'battery_current': battery.get('current'),
            'battery_voltage': (
                power.get('battery_voltage')
                if power.get('battery_voltage') is not None
                else battery.get('voltage')
            ),
            'rail_5v_voltage': power.get('rail_5v_voltage'),
            'rail_12v_voltage': power.get('rail_12v_voltage'),
            'last_update': power.get('last_update'),
            'source': (
                power.get('source')
                if power_topic_available
                else 'platform/bms/state estimate'
            ),
        }

        mapping_running = self.process_running('mapping_process')
        with self.lock:
            pose_initialized = self.navigation_pose_initialized
            amcl_pose = (
                dict(self.navigation_amcl_pose)
                if self.navigation_amcl_pose
                else None
            )
            covariance = (
                dict(self.navigation_amcl_covariance)
                if self.navigation_amcl_covariance
                else None
            )
        localization_trusted = bool(
            mapping_running
            or (
                pose_initialized
                and self.amcl_pose_is_good(covariance)
                and self.map_pose_is_safe(amcl_pose)
                and self.lookup_transform('map', 'base_link') is not None
            )
        )
        # Keep the current TF available for diagnostics, but do not project a
        # scan onto the map until AMCL has passed the same in-map safety gate
        # used by navigation.  A stale map->odom transform can otherwise make
        # a perfectly live LiDAR look like it belongs to a different room.
        tf_pose = self.lookup_transform('map', 'base_link')
        state['map_pose_trusted'] = localization_trusted
        state['map_pose'] = (
            tf_pose if localization_trusted else None
        )
        scan = state.get('scan')
        if scan and scan.get('points') and scan.get('frame'):
            transform = (
                self.lookup_transform(
                    'map',
                    scan['frame'],
                    scan.get('stamp'),
                    allow_latest_fallback=True,
                )
                if tf_pose is not None
                else None
            )
            transform_age = (
                abs(float(transform.get('transform_age_s', 0.0)))
                if transform
                else float('inf')
            )
            if (
                localization_trusted
                and transform
                and transform_age <= self.scan_transform_max_age_s
            ):
                cosine = math.cos(transform['yaw'])
                sine = math.sin(transform['yaw'])
                scan['map_points'] = [
                    [
                        round(
                            transform['x'] + cosine * point[0] - sine * point[1],
                            3,
                        ),
                        round(
                            transform['y'] + sine * point[0] + cosine * point[1],
                            3,
                        ),
                ]
                    for point in scan['points']
                ]
                scan['map_frame'] = 'map'
                scan['map_points_trusted'] = bool(localization_trusted)
                scan['map_transform_age_s'] = round(transform_age, 4)
            else:
                scan['map_points'] = []
                scan['map_points_trusted'] = False
                scan['map_transform_age_s'] = (
                    round(transform_age, 4) if math.isfinite(transform_age) else None
                )
        elif scan is not None:
            scan['map_points'] = []
            scan['map_points_trusted'] = False
        localization = state.get('localization') or {}
        now = time.time()
        pose_last_update = localization.get('last_update')
        try:
            pose_age_s = max(0.0, now - float(pose_last_update))
        except (TypeError, ValueError):
            pose_age_s = None
        try:
            tf_age_s = max(0.0, now - float(self.tf_last_update))
        except (TypeError, ValueError):
            tf_age_s = None
        localization.update(
            {
                'localized': localization_trusted,
                'tf_map_base_link': tf_pose,
                'tf_ok': tf_pose is not None,
                # AMCL may correctly leave /amcl_pose unchanged while the
                # Dingo is stationary. Keep that topic age visible without
                # treating it as a localization failure.
                'pose_age_s': round(pose_age_s, 2) if pose_age_s is not None else None,
                'amcl_pose_fresh': bool(
                    pose_age_s is not None and pose_age_s <= 3.0
                ),
                'tf_age_s': round(tf_age_s, 2) if tf_age_s is not None else None,
            }
        )
        state['localization'] = localization

        def health_entry(label, topic, last_update, stale_after):
            if last_update is None:
                return {
                    'label': label,
                    'topic': topic,
                    'status': 'offline',
                    'online': False,
                    'age_s': None,
                    'last_update': None,
                }
            try:
                age = max(0.0, now - float(last_update))
            except (TypeError, ValueError):
                age = None
            online = age is not None and age <= stale_after
            return {
                'label': label,
                'topic': topic,
                'status': 'live' if online else 'stale',
                'online': online,
                'age_s': round(age, 2) if age is not None else None,
                'last_update': last_update,
            }

        battery_state = state.get('battery') or {}
        power_state = state.get('power') or {}
        temperature_state = state.get('temperature') or {}
        mcu_updates = [
            value.get('last_update')
            for value in (battery_state, power_state, temperature_state)
            if value.get('last_update') is not None
        ]
        mcu_last = max(mcu_updates) if mcu_updates else None
        mcu_health = health_entry(
            'MCU / platform',
            f'/{self.robot_namespace}/platform/mcu/status/*',
            mcu_last,
            3.0,
        )
        lidar_health = health_entry(
            'Hokuyo LiDAR', self.scan_topic,
            (state.get('scan') or {}).get('last_update'), 1.0,
        )
        imu_health = health_entry(
            'IMU', f'/{self.robot_namespace}/sensors/imu_0/data',
            (state.get('imu') or {}).get('last_update'), 1.0,
        )
        camera_health = health_entry(
            'RealSense camera', self.camera_topic,
            (state.get('camera') or {}).get('last_frame'), 2.0,
        )
        odom_health = health_entry(
            'Odometry', f'/{self.robot_namespace}/platform/odom',
            (state.get('odom') or {}).get('last_update'), 2.0,
        )
        tf_health = health_entry(
            'TF dynamic', f'/{self.robot_namespace}/tf',
            self.tf_last_update or None, 3.0,
        )
        amcl_pose_health = health_entry(
            'AMCL pose', f'/{self.robot_namespace}/amcl_pose',
            localization.get('last_update'), 3.0,
        )
        amcl_health = dict(amcl_pose_health)
        amcl_health['label'] = 'AMCL localization'
        amcl_health['pose_age_s'] = localization.get('pose_age_s')
        amcl_health['tf_age_s'] = localization.get('tf_age_s')

        odom_state = state.get('odom') or {}
        try:
            linear_speed = abs(float(odom_state.get('linear') or 0.0))
            angular_speed = abs(float(odom_state.get('angular') or 0.0))
        except (TypeError, ValueError):
            linear_speed = float('inf')
            angular_speed = float('inf')
        robot_moving = linear_speed > 0.015 or angular_speed > 0.03
        # A stationary AMCL filter may leave /amcl_pose unchanged for longer
        # than the topic-health timeout. In that case the live map->base_link
        # TF plus live LiDAR/odometry is the meaningful localization heartbeat.
        # While moving, an old AMCL pose remains a real warning.
        localization_live = bool(
            localization_trusted
            and localization.get('amcl_available')
            and tf_health['online']
            and lidar_health['online']
            and odom_health['online']
            and (not robot_moving or amcl_pose_health['online'])
        )
        if localization_live:
            amcl_health.update(
                {
                    'status': 'live',
                    'online': True,
                    'age_s': tf_health.get('age_s'),
                    'last_update': self.tf_last_update or amcl_pose_health.get('last_update'),
                    'detail': (
                        'TF live · pose unchanged while stationary'
                        if not amcl_pose_health['online']
                        else 'pose + TF live'
                    ),
                }
            )
        elif localization_trusted and tf_health['online'] and robot_moving:
            amcl_health['detail'] = 'TF live · AMCL pose stale while moving'

        state['sensor_health'] = {
            'mcu': mcu_health,
            'lidar': lidar_health,
            'imu': imu_health,
            'camera': camera_health,
            'odometry': odom_health,
            'amcl': amcl_health,
            'tf': tf_health,
        }
        return state

    def _clearpath_name(self, relative):
        relative = str(relative or '').strip()
        if not relative:
            return ''
        if relative.startswith('/'):
            return relative
        return f'/{self.robot_namespace}/{relative}'.replace('//', '/')

    def _clearpath_topic_candidates(self, relative):
        if '#' in relative:
            return []
        if relative == 'scan':
            return [
                '/scan',
                self._clearpath_name('sensors/lidar2d_0/scan'),
            ]
        return [self._clearpath_name(relative)]

    @staticmethod
    def _graph_types(graph, name, fallback):
        values = graph.get(name)
        return list(values) if values else [fallback]

    def _clearpath_runtime_name(self, name):
        prefix = f'/{self.robot_namespace}/'
        return (
            name.startswith(prefix)
            or name in {
                '/scan',
                '/map',
                '/diagnostics',
                '/tf',
                '/tf_static',
                '/rosout',
            }
            or name.startswith('/camera/')
        )

    def clearpath_api_snapshot(self):
        """Return the documented Clearpath API plus the live ROS graph."""
        now = time.monotonic()
        if self.clearpath_api_cache is not None and now - self.clearpath_api_cache_at < 2.0:
            return self.clearpath_api_cache

        try:
            topic_graph = {
                name: list(types)
                for name, types in self.get_topic_names_and_types()
            }
        except Exception:
            topic_graph = {}
        try:
            service_graph = {
                name: list(types)
                for name, types in self.get_service_names_and_types()
            }
        except Exception:
            service_graph = {}

        def endpoint_counts(name):
            if not name:
                return 0, 0
            try:
                publishers = int(self.count_publishers(name))
            except Exception:
                publishers = 0
            try:
                subscribers = int(self.count_subscribers(name))
            except Exception:
                subscribers = 0
            return publishers, subscribers

        topics = []
        known_topic_names = set()
        for relative, declared_type, description, access, qos in self.CLEARPATH_TOPIC_CATALOG:
            candidates = self._clearpath_topic_candidates(relative)
            actual = next((name for name in candidates if name in topic_graph), None)
            publishers, subscribers = endpoint_counts(actual)
            topics.append(
                {
                    'relative_name': relative,
                    'ros_name': actual,
                    'type': self._graph_types(topic_graph, actual, declared_type)[0],
                    'declared_type': declared_type,
                    'description': description,
                    'access': access,
                    'qos': qos,
                    'pattern': '#' in relative,
                    'available': actual is not None,
                    'publishers': publishers,
                    'subscribers': subscribers,
                }
            )
            if actual:
                known_topic_names.add(actual)

        runtime_topics = []
        for name in sorted(topic_graph):
            if not self._clearpath_runtime_name(name) or name in known_topic_names:
                continue
            publishers, subscribers = endpoint_counts(name)
            runtime_topics.append(
                {
                    'ros_name': name,
                    'types': list(topic_graph[name]),
                    'publishers': publishers,
                    'subscribers': subscribers,
                }
            )

        services = []
        known_service_names = set()
        for relative, declared_type, description, access in self.CLEARPATH_SERVICE_CATALOG:
            if '#' in relative:
                actual = None
                candidates = []
            else:
                candidates = [self._clearpath_name(relative)]
                actual = next((name for name in candidates if name in service_graph), None)
            services.append(
                {
                    'relative_name': relative,
                    'ros_name': actual,
                    'type': self._graph_types(service_graph, actual, declared_type)[0],
                    'declared_type': declared_type,
                    'description': description,
                    'access': access,
                    'pattern': '#' in relative,
                    'available': actual is not None,
                }
            )
            if actual:
                known_service_names.add(actual)

        runtime_services = [
            {
                'ros_name': name,
                'types': list(service_graph[name]),
            }
            for name in sorted(service_graph)
            if (
                name.startswith(f'/{self.robot_namespace}/')
                or name.startswith('/map_saver/')
            )
            and name not in known_service_names
        ]

        actions = []
        for relative, action_type, description in self.CLEARPATH_ACTION_CATALOG:
            name = self._clearpath_name(relative)
            action_services = (
                f'{name}/_action/send_goal',
                f'{name}/_action/get_result',
                f'{name}/_action/cancel_goal',
            )
            available = any(service in service_graph for service in action_services)
            actions.append(
                {
                    'relative_name': relative,
                    'ros_name': name,
                    'type': action_type,
                    'description': description,
                    'access': 'command',
                    'available': available,
                }
            )

        try:
            nodes = [
                {
                    'name': name,
                    'namespace': namespace,
                    'full_name': f'{namespace.rstrip("/")}/{name}'.replace('//', '/'),
                }
                for name, namespace in self.get_node_names_and_namespaces()
            ]
        except Exception:
            nodes = []

        state = self.snapshot()
        navigation = self.navigation_snapshot()
        scan = state.get('scan') or {}
        with self.lock:
            safety = {
                'emergency_stop': self.state.get('emergency_stop'),
                'safety_stop': self.state.get('safety_stop'),
            }
        api = {
            'schema': 'clearpath-ros2-jazzy',
            'generated_at': time.time(),
            'robot': {
                'model': 'DD100',
                'serial': 'dd100-10000002',
                'namespace': self.robot_namespace,
                'controller': 'ps5',
                'mcu_protocol': 'proton',
            },
            'documentation': {
                'overview': 'https://docs.clearpathrobotics.com/docs/ros/api/overview/',
                'platform': 'https://docs.clearpathrobotics.com/docs/ros/api/platform_api/',
                'mcu': 'https://docs.clearpathrobotics.com/docs/ros/api/mcu_api/',
                'sensors': 'https://docs.clearpathrobotics.com/docs/ros/api/sensors_api/',
            },
            'topics': topics,
            'services': services,
            'actions': actions,
            'runtime': {
                'topics': runtime_topics,
                'services': runtime_services,
                'nodes': nodes,
            },
            'live': {
                'battery': state.get('battery'),
                'power': state.get('power'),
                'odom': state.get('odom'),
                'scan': {
                    'topic': self.scan_topic,
                    'count': scan.get('count'),
                    'valid': scan.get('valid'),
                    'frame': scan.get('frame'),
                    'map_points': len(scan.get('map_points') or []),
                    'map_points_trusted': scan.get('map_points_trusted', False),
                },
                'camera': state.get('camera'),
                'map_pose': state.get('map_pose'),
                'map_pose_trusted': state.get('map_pose_trusted', False),
                'safety': safety,
                'navigation': navigation,
            },
        }
        self.clearpath_api_cache = api
        self.clearpath_api_cache_at = now
        return api

    def map_snapshot(self):
        with self.lock:
            return self.map_state

    def camera_snapshot(self):
        with self.lock:
            return self.camera_bytes, self.camera_mime

    def load_rooms(self):
        try:
            rooms = json.loads(self.rooms_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []
        if not isinstance(rooms, list):
            return []
        valid = []
        for room in rooms[:100]:
            if not isinstance(room, dict):
                continue
            try:
                valid.append(
                    {
                        'id': str(room['id']),
                        'name': str(room['name'])[:48],
                        'x': round(float(room['x']), 3),
                        'y': round(float(room['y']), 3),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return valid

    def write_rooms(self, rooms):
        self.rooms_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.rooms_file.with_name(self.rooms_file.name + '.tmp')
        temporary.write_text(
            json.dumps(rooms, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        temporary.replace(self.rooms_file)

    def rooms_snapshot(self):
        with self.lock:
            return list(self.rooms)

    def load_settings(self):
        settings = dict(self.DEFAULT_SETTINGS)
        try:
            stored = json.loads(self.settings_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return settings
        if not isinstance(stored, dict):
            return settings
        for key, minimum, maximum in (
            ('max_linear_speed', 0.02, 0.25),
            ('max_angular_speed', 0.10, 0.80),
        ):
            try:
                value = float(stored.get(key, settings[key]))
                if math.isfinite(value):
                    settings[key] = max(minimum, min(maximum, value))
            except (TypeError, ValueError):
                pass
        return settings

    def write_settings(self, settings):
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_file.with_name(self.settings_file.name + '.tmp')
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        temporary.replace(self.settings_file)

    def settings_snapshot(self):
        with self.lock:
            return dict(self.settings)

    def update_settings(self, values):
        with self.lock:
            settings = dict(self.settings)
        for key, minimum, maximum in (
            ('max_linear_speed', 0.02, 0.25),
            ('max_angular_speed', 0.10, 0.80),
        ):
            if key not in values:
                continue
            value = float(values[key])
            if not math.isfinite(value):
                raise ValueError('Μη έγκυρη ταχύτητα')
            settings[key] = max(minimum, min(maximum, value))
        with self.lock:
            self.settings = settings
            self.write_settings(dict(self.settings))
            return dict(self.settings)

    def add_room(self, name, x, y):
        name = str(name or '').strip()
        if not name:
            raise ValueError('Το δωμάτιο χρειάζεται όνομα')
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Μη έγκυρη θέση δωματίου')
        room = {
            'id': str(int(time.time() * 1000)),
            'name': name[:48],
            'x': round(x, 3),
            'y': round(y, 3),
        }
        with self.lock:
            self.rooms.append(room)
            self.rooms = self.rooms[-100:]
            rooms = list(self.rooms)
            self.write_rooms(rooms)
        return room

    def remove_room(self, room_id):
        with self.lock:
            original = len(self.rooms)
            self.rooms = [room for room in self.rooms if room['id'] != str(room_id)]
            changed = len(self.rooms) != original
            if changed:
                self.write_rooms(list(self.rooms))
            return changed

    def maps_snapshot(self):
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        maps = []
        for yaml_file in sorted(
            self.maps_dir.glob('*.yaml'),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            maps.append(
                {
                    'name': yaml_file.stem,
                    'updated': int(yaml_file.stat().st_mtime),
                }
            )
        return maps

    @staticmethod
    def ros2_command(*args):
        ros2 = shutil.which('ros2') or '/opt/ros/jazzy/bin/ros2'
        return [ros2, *args]

    def set_autonomous_navigation_mode(self, active, force=False):
        """Toggle only Clearpath's Bluetooth-quality twist_mux lock.

        The Clearpath BT watchdog is intentionally fail-closed for manual
        joystick operation.  It is not a reason to block an explicitly
        requested Nav2 goal when the physical E-stop and safety-stop locks
        are still present.  We change the live twist_mux parameter through
        the ROS 2 CLI so this works from both HTTP threads and ROS callbacks;
        waiting on an rclpy parameter future from a single-threaded callback
        would otherwise deadlock the Dashboard executor.

        A failed parameter update never sends a new autonomous goal.  Every
        terminal/cancel/shutdown path requests the normal priority again, and
        the retry timer handles a twist_mux that is still starting up.
        """
        active = bool(active)
        if not self.autonomous_navigation_bypass_bt_quality:
            with self.lock:
                self.autonomous_navigation_active = False
                self.autonomous_navigation_priority = self.bt_quality_normal_priority
                self.autonomous_navigation_last_error = None
            return True

        priority = (
            self.bt_quality_autonomous_priority
            if active
            else self.bt_quality_normal_priority
        )
        with self.autonomous_navigation_lock:
            with self.lock:
                current_active = self.autonomous_navigation_active
                current_priority = self.autonomous_navigation_priority
                current_error = self.autonomous_navigation_last_error
            if (
                not force
                and current_active == active
                and current_priority == priority
                and current_error is None
            ):
                return True

            try:
                result = subprocess.run(
                    self.ros2_command(
                        'param',
                        'set',
                        self.twist_mux_node_name,
                        'locks.bt_quality.priority',
                        str(priority),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=6,
                    check=False,
                    env=os.environ.copy(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail = str(exc)
                result = None
            else:
                detail = (
                    (result.stderr or result.stdout or '').strip().splitlines()
                    or [f'exit {result.returncode}']
                )[0][:240]

            success = result is not None and result.returncode == 0
            with self.lock:
                # The logical navigation state follows the requested action;
                # priority=None makes an unsuccessful restore visible rather
                # than pretending that the live twist_mux is known-safe.
                self.autonomous_navigation_active = active if success else False
                self.autonomous_navigation_priority = priority if success else None
                self.autonomous_navigation_last_error = None if success else detail
                self.autonomous_navigation_last_change_at = time.time()
                self.autonomous_navigation_restore_pending = not success
                if success:
                    self.autonomous_navigation_restore_attempts = 0
                    self.autonomous_navigation_restore_next_at = 0.0

            if success:
                self.get_logger().info(
                    'twist_mux BT-quality lock priority set to '
                    f'{priority} (autonomous={active})'
                )
            else:
                self.get_logger().warning(
                    'Could not set twist_mux BT-quality lock priority '
                    f'to {priority}: {detail}'
                )
            return success

    def autonomous_navigation_lock_tick(self):
        """Retry the fail-closed BT-quality priority after startup/errors."""
        with self.lock:
            pending = self.autonomous_navigation_restore_pending
            active = self.autonomous_navigation_active
            attempts = self.autonomous_navigation_restore_attempts
            next_at = self.autonomous_navigation_restore_next_at
        if not pending or active or attempts >= self.autonomous_navigation_restore_max_attempts:
            return
        now = time.monotonic()
        if next_at and now < next_at:
            return
        with self.lock:
            self.autonomous_navigation_restore_attempts += 1
            attempt = self.autonomous_navigation_restore_attempts
            self.autonomous_navigation_restore_next_at = now + 2.0
        if self.set_autonomous_navigation_mode(False, force=True):
            self.get_logger().debug(
                f'Restored normal BT-quality lock priority on attempt {attempt}'
            )

    def start_process(self, attribute, args):
        with self.process_lock:
            process = getattr(self, attribute)
            if process is not None and process.poll() is None:
                return False
            try:
                process = subprocess.Popen(
                    self.ros2_command(*args),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                self.get_logger().error(f'Could not start {args}: {exc}')
                return False
            setattr(self, attribute, process)
            return True

    def stop_process(self, attribute):
        with self.process_lock:
            process = getattr(self, attribute)
            setattr(self, attribute, None)
        if process is None:
            return False
        if process.poll() is not None:
            return True
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        return True

    def start_mapping(self):
        if self.navigation_stack_running():
            raise RuntimeError(
                'Το Nav2 τρέχει ήδη. Κλείσε πρώτα το Nav2 πριν ξεκινήσεις SLAM.'
            )
        return self.start_process(
            'mapping_process',
            ['launch', 'dingo_bringup', 'slam.launch.py', 'scan_topic:=/scan'],
        )

    def stop_mapping(self):
        return self.stop_process('mapping_process')

    def start_camera(self):
        # The normal Dingo stack owns the RealSense process.  Do not launch a
        # second driver from the dashboard when that stack is already live.
        if self.camera_driver_active():
            return False
        return self.start_process(
            'camera_process',
            ['launch', 'dingo_bringup', 'camera.launch.py'],
        )

    def stop_camera(self):
        return self.stop_process('camera_process')

    def start_rviz(self):
        """Start the real RViz2 view used by the Dashboard tab."""
        return self.start_process(
            'rviz_process',
            ['run', 'dingo_bringup', 'rviz_dashboard_bridge.sh'],
        )

    def stop_rviz(self):
        return self.stop_process('rviz_process')

    def rviz_status(self):
        config = (
            Path(get_package_share_directory('dingo_bringup'))
            / 'config'
            / 'dingo.rviz'
        )
        return {
            'available': bool(
                shutil.which('rviz2')
                or Path('/opt/ros/jazzy/bin/rviz2').is_file()
            ),
            'process_running': self.process_running('rviz_process'),
            'web_ready': self.rviz_web_ready(),
            'mode': 'native_rviz2_over_novnc',
            'config': str(config),
            'fixed_frame': 'map',
            'web_url': 'rviz/vnc.html?autoconnect=true&resize=scale&path=rviz-ws',
            'topics': {
                'map': self.map_topic,
                'scan': self.scan_topic,
                'tf': f'/{self.robot_namespace}/tf',
                'tf_static': f'/{self.robot_namespace}/tf_static',
            },
        }

    @staticmethod
    def rviz_web_ready():
        try:
            with socket.create_connection(('127.0.0.1', 8091), timeout=0.15):
                return True
        except OSError:
            return False

    def proxy_rviz_websocket(self, client_socket, request_headers):
        """Proxy the same-origin noVNC WebSocket to websockify."""
        upstream = None
        try:
            upstream = socket.create_connection(('127.0.0.1', 8091), timeout=3)
            headers = [
                'GET /websockify HTTP/1.1',
                'Host: 127.0.0.1:8091',
            ]
            for name in (
                'Upgrade',
                'Connection',
                'Sec-WebSocket-Key',
                'Sec-WebSocket-Version',
                'Sec-WebSocket-Protocol',
                'Origin',
            ):
                value = request_headers.get(name)
                if value:
                    headers.append(f'{name}: {value}')
            upstream.sendall(('\r\n'.join(headers) + '\r\n\r\n').encode())
            response = bytearray()
            while b'\r\n\r\n' not in response and len(response) < 65536:
                chunk = upstream.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
            client_socket.sendall(response)
            if not response.startswith(b'HTTP/1.1 101'):
                return
            client_socket.settimeout(None)
            upstream.settimeout(None)
            sockets = [client_socket, upstream]
            while True:
                readable, _, exceptional = select.select(sockets, [], sockets, 60)
                if exceptional:
                    break
                if not readable:
                    continue
                for source in readable:
                    payload = source.recv(65536)
                    if not payload:
                        return
                    destination = upstream if source is client_socket else client_socket
                    destination.sendall(payload)
        except (OSError, ValueError):
            return
        finally:
            if upstream is not None:
                try:
                    upstream.close()
                except OSError:
                    pass

    def camera_driver_active(self):
        try:
            return any(
                self.count_publishers(topic) > 0
                for topic in (self.camera_raw_topic, self.camera_topic)
                if topic
            )
        except (RuntimeError, AttributeError):
            return False

    def map_file_for_name(self, name):
        name = str(name or '').strip()
        if name.lower().endswith('.yaml'):
            name = name[:-5]
        safe_name = re.sub(r'[^\w.-]+', '_', name, flags=re.UNICODE).strip('._')
        if not safe_name:
            raise RuntimeError('Διάλεξε πρώτα έναν αποθηκευμένο χάρτη')
        map_file = self.maps_dir / f'{safe_name}.yaml'
        if not map_file.is_file():
            raise RuntimeError(f'Δεν βρέθηκε ο χάρτης «{safe_name}»')
        return map_file

    def start_navigation(self, map_name):
        if self.follow_active:
            self.stop_follow(silent=True)
        map_name = str(map_name or self.auto_start_map or 'dingo_map').strip()
        map_file = self.map_file_for_name(map_name)
        if self.navigation_stack_running():
            return False
        mapping_running = self.process_running('mapping_process')
        # A map->base_link transform from an old localization session is not a
        # safe initial pose for a newly selected map. Only carry a pose across
        # when we are switching directly from the currently running SLAM map.
        initial_pose = self.lookup_transform('map', 'base_link') if mapping_running else None
        if mapping_running:
            self.stop_mapping()
        started = self.start_process(
            'navigation_process',
            [
                'launch',
                'dingo_bringup',
                'localization.launch.py',
                f'map:={map_file}',
                'scan_topic:=/scan',
            ],
        )
        if started:
            with self.lock:
                self.navigation_map = map_file.stem
                self.navigation_status = 'starting'
                self.navigation_goal_handle = None
                self.navigation_result_future = None
                self.navigation_initial_pose = initial_pose
                self.navigation_initial_pose_sent_at = 0.0
                self.navigation_pose_initialized = bool(initial_pose)
                self.navigation_amcl_pose = None
                self.navigation_amcl_covariance = None
                self.navigation_amcl_received_at = 0.0
                self.localization_service_future = None
                self.localization_lifecycle_future = None
                self.localization_lifecycle_node = None
                self.localization_lifecycle_operation = None
                self.localization_lifecycle_states = {
                    name: None for name in self.localization_lifecycle_nodes
                }
                self.localization_lifecycle_done = False
                self.navigation_lifecycle_future = None
                self.navigation_lifecycle_node = None
                self.navigation_lifecycle_operation = None
                self.navigation_lifecycle_states = {
                    name: None for name in self.navigation_lifecycle_nodes
                }
                self.navigation_lifecycle_done = False
                # Give Nav2's own lifecycle manager a chance to start first;
                # the recovery state machine takes over only after this short
                # cold-start grace period.
                self.navigation_lifecycle_not_before = time.monotonic() + 8.0
                self.localization_search_active = False
                self.localization_search_started_at = 0.0
                self.localization_good_count = 0
                self.localization_last_pose = None
                self.localization_global_service_response = False
                self.localization_last_global_reset = False
                self.localization_scan_match_attempted = False
                self.localization_allow_motion = False
                self.localization_method = None
                self.navigation_goal = None
                self.navigation_feedback = None
                self.auto_localize_attempted = False
                self.auto_localize_retry_at = 0.0
                self.auto_localize_last_log_at = 0.0
                self.auto_localize_retry_count = 0
                self.localization_watchdog_invalid_since = 0.0
                self.localization_watchdog_last_reset_at = 0.0
        return started

    def auto_start_navigation_tick(self):
        """Start Nav2 once at boot, after the mounted sensors are visible.

        This deliberately starts localization only. It never creates a
        NavigateToPose goal and therefore cannot command autonomous motion.
        """
        if not self.auto_start_map or self.auto_start_attempted:
            return
        if self.process_running('mapping_process') or self.process_running(
            'navigation_process'
        ):
            # A user-selected mapping/navigation session takes precedence over
            # the boot default.
            self.auto_start_attempted = True
            return
        try:
            self.map_file_for_name(self.auto_start_map)
        except RuntimeError as exc:
            now = time.monotonic()
            if now - self.auto_start_last_log_at >= 30.0:
                self.get_logger().warning(
                    f'Auto-start map unavailable ({self.auto_start_map}): {exc}'
                )
                self.auto_start_last_log_at = now
            return
        with self.lock:
            sensors_ready = (
                # Nav2 may start before the LiDAR driver has connected. Keep
                # the stack alive so automatic Global Localization can begin
                # as soon as the first real scan arrives; no goal is accepted
                # until localization_is_good() succeeds.
                self.state['odom'] is not None
            )
        if not sensors_ready or self.lookup_transform('odom', 'base_link') is None:
            return
        try:
            started = self.start_navigation(self.auto_start_map)
        except RuntimeError as exc:
            self.get_logger().error(f'Could not auto-start Nav2: {exc}')
            started = False
        self.auto_start_attempted = True
        if started:
            self.get_logger().info(
                f'Auto-starting Nav2 with map: {self.auto_start_map}'
            )
        else:
            self.get_logger().warning('Nav2 auto-start did not start a process')

    def _request_localization_transition(self, node_name, transition_id):
        client = self.localization_lifecycle_change_clients[node_name]
        if not client.service_is_ready():
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        try:
            future = client.call_async(request)
        except Exception as exc:
            self.get_logger().debug(
                f'Could not request {node_name} lifecycle transition: {exc}'
            )
            return False
        with self.lock:
            self.localization_lifecycle_future = future
            self.localization_lifecycle_node = node_name
            self.localization_lifecycle_operation = 'change'
        return True

    def localization_lifecycle_tick(self):
        """Recover localization lifecycle nodes if Nav2 autostart races them.

        Clearpath launches the official Nav2 lifecycle manager, but on a cold
        start its autostart request can arrive before map_server/amcl expose
        their services.  Keep the recovery local and explicit: configure and
        activate only the two localization nodes, never navigation motion.
        """
        if not self.navigation_stack_running():
            return
        with self.lock:
            future = self.localization_lifecycle_future
            node_name = self.localization_lifecycle_node
            operation = self.localization_lifecycle_operation
            states = dict(self.localization_lifecycle_states)
            done = self.localization_lifecycle_done
        if done:
            return
        if future is not None:
            if not future.done():
                return
            try:
                result = future.result()
            except Exception as exc:
                self.get_logger().debug(
                    f'Localization lifecycle {operation} failed for {node_name}: {exc}'
                )
                with self.lock:
                    self.localization_lifecycle_future = None
                    self.localization_lifecycle_node = None
                    self.localization_lifecycle_operation = None
                return
            with self.lock:
                self.localization_lifecycle_future = None
                self.localization_lifecycle_node = None
                self.localization_lifecycle_operation = None
            if operation == 'get':
                state_id = int(result.current_state.id)
                with self.lock:
                    self.localization_lifecycle_states[node_name] = state_id
                if state_id == LifecycleState.PRIMARY_STATE_UNCONFIGURED:
                    self._request_localization_transition(
                        node_name, Transition.TRANSITION_CONFIGURE
                    )
                elif state_id == LifecycleState.PRIMARY_STATE_INACTIVE:
                    self._request_localization_transition(
                        node_name, Transition.TRANSITION_ACTIVATE
                    )
            elif operation == 'change' and not bool(result.success):
                self.get_logger().debug(
                    f'Localization lifecycle transition rejected for {node_name}'
                )
            return

        next_node = next(
            (
                name for name in self.localization_lifecycle_nodes
                if states.get(name) != LifecycleState.PRIMARY_STATE_ACTIVE
            ),
            None,
        )
        if next_node is None:
            with self.lock:
                self.localization_lifecycle_done = True
            self.get_logger().info('Nav2 localization lifecycle is active')
            return
        client = self.localization_lifecycle_state_clients[next_node]
        if not client.service_is_ready():
            return
        try:
            future = client.call_async(GetState.Request())
        except Exception as exc:
            self.get_logger().debug(
                f'Could not query {next_node} lifecycle state: {exc}'
            )
            return
        with self.lock:
            self.localization_lifecycle_future = future
            self.localization_lifecycle_node = next_node
            self.localization_lifecycle_operation = 'get'

    def _request_navigation_transition(self, node_name, transition_id):
        client = self.navigation_lifecycle_change_clients[node_name]
        if not client.service_is_ready():
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        try:
            future = client.call_async(request)
        except Exception as exc:
            self.get_logger().debug(
                f'Could not request {node_name} navigation transition: {exc}'
            )
            return False
        with self.lock:
            self.navigation_lifecycle_future = future
            self.navigation_lifecycle_node = node_name
            self.navigation_lifecycle_operation = 'change'
        return True

    def navigation_lifecycle_tick(self):
        """Recover Nav2 navigation nodes left inactive by a cold-start race."""
        if not self.navigation_stack_running():
            return
        if time.monotonic() < self.navigation_lifecycle_not_before:
            return
        with self.lock:
            future = self.navigation_lifecycle_future
            node_name = self.navigation_lifecycle_node
            operation = self.navigation_lifecycle_operation
            states = dict(self.navigation_lifecycle_states)
            done = self.navigation_lifecycle_done
        if done:
            return
        if future is not None:
            if not future.done():
                return
            try:
                result = future.result()
            except Exception as exc:
                self.get_logger().debug(
                    f'Navigation lifecycle {operation} failed for {node_name}: {exc}'
                )
                with self.lock:
                    self.navigation_lifecycle_future = None
                    self.navigation_lifecycle_node = None
                    self.navigation_lifecycle_operation = None
                return
            with self.lock:
                self.navigation_lifecycle_future = None
                self.navigation_lifecycle_node = None
                self.navigation_lifecycle_operation = None
            if operation == 'get':
                state_id = int(result.current_state.id)
                with self.lock:
                    self.navigation_lifecycle_states[node_name] = state_id
                if state_id == LifecycleState.PRIMARY_STATE_UNCONFIGURED:
                    self._request_navigation_transition(
                        node_name, Transition.TRANSITION_CONFIGURE
                    )
                elif state_id == LifecycleState.PRIMARY_STATE_INACTIVE:
                    self._request_navigation_transition(
                        node_name, Transition.TRANSITION_ACTIVATE
                    )
            elif operation == 'change' and not bool(result.success):
                self.get_logger().debug(
                    f'Navigation lifecycle transition rejected for {node_name}'
                )
            return

        next_node = next(
            (
                name for name in self.navigation_lifecycle_nodes
                if states.get(name) != LifecycleState.PRIMARY_STATE_ACTIVE
            ),
            None,
        )
        if next_node is None:
            with self.lock:
                self.navigation_lifecycle_done = True
            self.get_logger().info('Nav2 navigation lifecycle is active')
            return
        client = self.navigation_lifecycle_state_clients[next_node]
        if not client.service_is_ready():
            return
        try:
            future = client.call_async(GetState.Request())
        except Exception as exc:
            self.get_logger().debug(
                f'Could not query {next_node} navigation lifecycle state: {exc}'
            )
            return
        with self.lock:
            self.navigation_lifecycle_future = future
            self.navigation_lifecycle_node = next_node
            self.navigation_lifecycle_operation = 'get'

    def auto_localization_tick(self):
        """Automatically seed AMCL after the boot map has started."""
        if not self.auto_localize or not self.navigation_stack_running():
            return
        # Localization is independent of NavigateToPose.  During startup the
        # navigation action server can still be unavailable while map_server
        # and AMCL are already active; do not block automatic AMCL startup on
        # the navigation action lifecycle.
        with self.lock:
            if (
                self.auto_localize_attempted
                or self.localization_search_active
                or self.localization_scan_match_active
                or self.navigation_pose_initialized
                or self.navigation_initial_pose is not None
            ):
                return
            inputs_ready = bool(self.map_state and self.state.get('scan'))
        if not inputs_ready:
            return
        now = time.monotonic()
        if now < self.auto_localize_retry_at:
            return
        try:
            # Boot localization must never command a rotation by itself.  It
            # uses AMCL global reset and stationary LiDAR matching first.
            started = self.start_global_localization(allow_motion=False)
        except RuntimeError as exc:
            self.auto_localize_retry_at = now + 5.0
            if now - self.auto_localize_last_log_at >= 30.0:
                self.get_logger().warning(f'Automatic AMCL localization retry: {exc}')
                self.auto_localize_last_log_at = now
            return
        if not started:
            self.auto_localize_retry_at = now + 5.0
            return
        self.auto_localize_attempted = True
        if started:
            self.get_logger().info('Automatic AMCL localization started')

    def localization_watchdog_tick(self):
        """Re-localize automatically when AMCL's pose is no longer usable.

        A global reset at boot is not enough when somebody picks up the Dingo,
        the map is changed, or an old map->odom transform leaves AMCL with a
        numerically precise but geometrically impossible pose.  In that case
        the Dashboard used to keep drawing the live scan at the stale pose.
        Wait for a fresh scan and a stationary base, then use the same
        official AMCL reset as the manual Global Localization button.  The
        scan matcher remains a stationary fallback; this method never enables
        the optional rotation search.
        """
        if not self.auto_relocalize or not self.navigation_stack_running():
            return

        with self.lock:
            if (
                self.localization_search_active
                or self.localization_scan_match_active
                or self.localization_method == 'manual_initial_pose'
            ):
                self.localization_watchdog_invalid_since = 0.0
                return
            pose_initialized = self.navigation_pose_initialized
            pose = dict(self.navigation_amcl_pose) if self.navigation_amcl_pose else None
            covariance = (
                dict(self.navigation_amcl_covariance)
                if self.navigation_amcl_covariance
                else None
            )
            odom = dict(self.state.get('odom') or {})
            scan = dict(self.state.get('scan') or {})
            map_available = self.map_state is not None

        # The startup timer owns the first localization attempt.  Do not race
        # it while AMCL is still coming up.  After an attempted search has
        # failed, however, keep the watchdog alive even if AMCL has no pose at
        # all, so the next automatic retry does not depend on a manual button.
        if (
            not map_available
            or (
                not self.auto_localize_attempted
                and (not pose_initialized or not pose)
            )
        ):
            self.localization_watchdog_invalid_since = 0.0
            return

        try:
            linear_speed = abs(float(odom.get('linear') or 0.0))
            angular_speed = abs(float(odom.get('angular') or 0.0))
        except (TypeError, ValueError):
            return
        if linear_speed > 0.015 or angular_speed > 0.03:
            # Let AMCL finish tracking while the platform is moving.  Start
            # the stationary timer again once it has stopped.
            self.localization_watchdog_invalid_since = 0.0
            return

        try:
            scan_age = time.time() - float(scan.get('last_update'))
        except (TypeError, ValueError):
            scan_age = float('inf')
        if scan_age > 3.0:
            # A reset cannot converge without a live scan; wait for the Hokuyo
            # driver instead of repeatedly resetting AMCL offline.
            self.localization_watchdog_invalid_since = 0.0
            return

        pose_valid = bool(
            self.amcl_pose_is_good(covariance)
            and self.map_pose_is_safe(pose)
            and self.lookup_transform('map', 'base_link') is not None
        )
        now = time.monotonic()
        if pose_valid:
            self.localization_watchdog_invalid_since = 0.0
            return

        if self.localization_watchdog_invalid_since <= 0.0:
            self.localization_watchdog_invalid_since = now
            return
        if now - self.localization_watchdog_invalid_since < self.localization_watchdog_confirm_s:
            return
        if now - self.localization_watchdog_last_reset_at < self.localization_watchdog_cooldown_s:
            return

        self.localization_watchdog_last_reset_at = now
        self.localization_watchdog_invalid_since = 0.0
        try:
            started = self.start_global_localization(allow_motion=False)
        except RuntimeError as exc:
            self.get_logger().warning(
                f'Automatic AMCL re-localization could not start: {exc}'
            )
            return
        if started:
            self.get_logger().warning(
                'AMCL pose left the current map; automatic global re-localization started'
            )

    def publish_initial_pose(self):
        with self.lock:
            pose = (
                dict(self.navigation_initial_pose)
                if self.navigation_initial_pose
                else None
            )
            sent_at = self.navigation_initial_pose_sent_at
        if pose is None or not self.navigation_stack_running():
            return
        now = time.monotonic()
        if sent_at and now - sent_at > self.initial_pose_republish_s:
            with self.lock:
                self.navigation_initial_pose = None
                self.navigation_initial_pose_sent_at = 0.0
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = pose['x']
        msg.pose.pose.position.y = pose['y']
        msg.pose.pose.orientation.z = math.sin(pose['yaw'] / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose['yaw'] / 2.0)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.10
        self.initial_pose_pub.publish(msg)
        with self.lock:
            if self.navigation_initial_pose is not None:
                self.navigation_initial_pose_sent_at = sent_at or now
                # A scan-match seed is provisional while global localization
                # is active.  Only AMCL convergence may make it trusted.
                if not self.localization_search_active:
                    self.navigation_pose_initialized = True

    def send_initial_pose(self, x, y, yaw=0.0):
        try:
            x = float(x)
            y = float(y)
            yaw = float(yaw)
        except (TypeError, ValueError) as exc:
            raise ValueError('Η αρχική θέση δεν είναι έγκυρη') from exc
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError('Η αρχική θέση δεν είναι έγκυρη')
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        self.cancel_navigation()
        with self.lock:
            self.localization_search_active = False
            self.localization_scan_match_active = False
            self.localization_service_future = None
            self.localization_lifecycle_future = None
            self.localization_lifecycle_node = None
            self.localization_lifecycle_operation = None
            self.localization_lifecycle_states = {
                name: None for name in self.localization_lifecycle_nodes
            }
            self.localization_lifecycle_done = False
            self.nomotion_update_future = None
            self.navigation_amcl_pose = None
            self.navigation_amcl_covariance = None
            self.navigation_amcl_received_at = 0.0
            self.localization_good_count = 0
            self.localization_last_pose = None
            self.localization_global_service_response = False
            self.localization_last_global_reset = False
            self.localization_scan_match_attempted = False
            self.localization_allow_motion = False
            self.localization_method = 'manual_initial_pose'
            self.navigation_initial_pose = {
                'x': round(x, 3),
                'y': round(y, 3),
                'yaw': round(yaw, 4),
                'frame': 'map',
            }
            self.navigation_initial_pose_sent_at = 0.0
            self.navigation_goal = None
            self.navigation_feedback = None
            self.navigation_pose_initialized = True
            self.navigation_status = 'ready'
        self.publish_initial_pose()
        return True

    @staticmethod
    def _angle_difference(first, second):
        return math.atan2(math.sin(first - second), math.cos(first - second))

    def _scan_match_pose(self):
        """Find a good map pose from the current 2D scan, without moving Dingo.

        This is deliberately a conservative helper for the Dashboard's global
        localization button.  AMCL remains the estimator and the official
        Clearpath/Nav2 parameters remain unchanged; this matcher only supplies
        an initial pose when the static map has a strong geometric candidate.
        """
        if np is None or distance_transform_edt is None:
            return None
        with self.lock:
            map_state = dict(self.map_state) if self.map_state else None
            scan_state = dict(self.state.get('scan') or {})
        if not map_state or not scan_state.get('points') or not scan_state.get('frame'):
            return None
        try:
            width = int(map_state['width'])
            height = int(map_state['height'])
            resolution = float(map_state['resolution'])
            origin_x = float(map_state['origin']['x'])
            origin_y = float(map_state['origin']['y'])
            grid = np.asarray(map_state['data'], dtype=np.int16).reshape(height, width)
            scan_points = np.asarray(scan_state['points'], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None
        if width < 10 or height < 10 or resolution <= 0.0:
            return None

        # Match the base pose, not the laser pose.  The measured static TF is
        # therefore part of the calculation, including a possible x/y/yaw
        # offset on a custom sensor bracket.
        base_to_laser = self.lookup_transform('base_link', scan_state['frame'])
        if base_to_laser is None:
            return None
        sensor_x = float(base_to_laser['x'])
        sensor_y = float(base_to_laser['y'])
        sensor_yaw = float(base_to_laser['yaw'])
        cosine = math.cos(sensor_yaw)
        sine = math.sin(sensor_yaw)
        base_points = np.empty_like(scan_points)
        base_points[:, 0] = (
            cosine * scan_points[:, 0] - sine * scan_points[:, 1] + sensor_x
        )
        base_points[:, 1] = (
            sine * scan_points[:, 0] + cosine * scan_points[:, 1] + sensor_y
        )
        ranges = np.hypot(base_points[:, 0], base_points[:, 1])
        usable = (
            np.isfinite(ranges)
            & (ranges >= 0.35)
            & (ranges <= min(float(scan_state.get('range_max') or 8.0), 6.0))
        )
        base_points = base_points[usable]
        if len(base_points) < 80:
            return None

        # Evenly reduce the scan for the global pass, then use all usable
        # points in the local refinement.  This keeps a phone request fast.
        global_points = base_points[::max(1, math.ceil(len(base_points) / 180))]
        occupied = grid >= 65
        distance_map = distance_transform_edt(~occupied).astype(np.float32) * resolution
        map_xs = origin_x + (np.arange(0, width, 3, dtype=np.float32) + 0.5) * resolution
        map_ys = origin_y + (np.arange(0, height, 3, dtype=np.float32) + 0.5) * resolution

        footprint_half_length = 0.551 / 2.0
        footprint_half_width = 0.517 / 2.0
        # The matcher uses the map's native cell scale as a fast pre-check.
        # map_pose_is_safe() performs the final half-cell-resolution check
        # before a pose is released to navigation.
        footprint_step = max(resolution, 0.05)
        footprint_count_x = int(
            math.ceil((2.0 * footprint_half_length) / footprint_step)
        )
        footprint_count_y = int(
            math.ceil((2.0 * footprint_half_width) / footprint_step)
        )
        footprint_samples = [
            (
                -footprint_half_length
                + (2.0 * footprint_half_length * sample_x / footprint_count_x),
                -footprint_half_width
                + (2.0 * footprint_half_width * sample_y / footprint_count_y),
            )
            for sample_x in range(footprint_count_x + 1)
            for sample_y in range(footprint_count_y + 1)
        ]

        def footprint_clear(x, y, yaw):
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            for local_x, local_y in footprint_samples:
                world_x = x + cosine * local_x - sine * local_y
                world_y = y + sine * local_x + cosine * local_y
                cell_x = int(math.floor((world_x - origin_x) / resolution))
                cell_y = int(math.floor((world_y - origin_y) / resolution))
                if not (0 <= cell_x < width and 0 <= cell_y < height):
                    return False
                if int(grid[cell_y, cell_x]) < 0 or int(grid[cell_y, cell_x]) >= 65:
                    return False
            return True

        def scalar_score(x, y, yaw, points):
            c = math.cos(yaw)
            s = math.sin(yaw)
            world_x = x + c * points[:, 0] - s * points[:, 1]
            world_y = y + s * points[:, 0] + c * points[:, 1]
            cols = np.rint((world_x - origin_x) / resolution).astype(np.int32)
            rows = np.rint((world_y - origin_y) / resolution).astype(np.int32)
            valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            safe_cols = np.clip(cols, 0, width - 1)
            safe_rows = np.clip(rows, 0, height - 1)
            distances = distance_map[safe_rows, safe_cols]
            distances = np.where(valid, distances, 0.60)
            mean_distance = float(np.minimum(distances, 0.60).mean())
            close_12 = float(np.mean(distances < 0.12))
            close_25 = float(np.mean(distances < 0.25))
            return (
                mean_distance
                + 0.05 * float(1.0 - np.mean(valid))
                - 0.08 * close_12
                - 0.03 * close_25,
                mean_distance,
                close_12,
                close_25,
                float(np.mean(valid)),
            )

        def global_candidates():
            candidates = []
            for yaw_degrees in range(-180, 180, 10):
                yaw = math.radians(yaw_degrees)
                c = math.cos(yaw)
                s = math.sin(yaw)
                local_x = c * global_points[:, 0] - s * global_points[:, 1]
                local_y = s * global_points[:, 0] + c * global_points[:, 1]
                for row_index in range(0, height, 3):
                    world_x = map_xs[:, None] + local_x[None, :]
                    world_y = map_ys[row_index // 3] + local_y[None, :]
                    cols = np.rint((world_x - origin_x) / resolution).astype(np.int32)
                    rows = np.rint((world_y - origin_y) / resolution).astype(np.int32)
                    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
                    safe_cols = np.clip(cols, 0, width - 1)
                    safe_rows = np.clip(rows, 0, height - 1)
                    distances = distance_map[safe_rows, safe_cols]
                    distances = np.where(valid, distances, 0.60)
                    mean_distance = np.minimum(distances, 0.60).mean(axis=1)
                    close_12 = (distances < 0.12).mean(axis=1)
                    close_25 = (distances < 0.25).mean(axis=1)
                    valid_fraction = valid.mean(axis=1)
                    center_cols = np.arange(0, width, 3)
                    center_ok = grid[row_index, center_cols] == 0
                    # Keep the whole DD100 footprint in known free space;
                    # checking only the centre can select a wall-adjacent
                    # hypothesis with an artificially perfect scan score.
                    clearance_x = int(
                        math.ceil(
                            (
                                abs(c) * footprint_half_length
                                + abs(s) * footprint_half_width
                            )
                            / resolution
                        )
                    )
                    clearance_y = int(
                        math.ceil(
                            (
                                abs(s) * footprint_half_length
                                + abs(c) * footprint_half_width
                            )
                            / resolution
                        )
                    )
                    for offset_y in range(-clearance_y, clearance_y + 1):
                        cell_row = row_index + offset_y
                        if not (0 <= cell_row < height):
                            center_ok[:] = False
                            break
                        for offset_x in range(-clearance_x, clearance_x + 1):
                            cell_col = center_cols + offset_x
                            inside = (cell_col >= 0) & (cell_col < width)
                            safe_cols = np.clip(cell_col, 0, width - 1)
                            center_ok &= inside & (grid[cell_row, safe_cols] == 0)
                    score = (
                        mean_distance
                        + 0.05 * (1.0 - valid_fraction)
                        - 0.08 * close_12
                        - 0.03 * close_25
                        + np.where(center_ok, 0.0, 0.40)
                    )
                    for index in np.argsort(score)[:3]:
                        candidates.append(
                            (
                                float(score[index]),
                                float(map_xs[index]),
                                float(map_ys[row_index // 3]),
                                yaw,
                            )
                        )
            candidates.sort(key=lambda item: item[0])
            distinct = []
            for candidate in candidates:
                if all(
                    math.hypot(candidate[1] - other[1], candidate[2] - other[2]) > 0.45
                    or abs(math.degrees(self._angle_difference(candidate[3], other[3]))) > 18.0
                    for other in distinct
                ):
                    distinct.append(candidate)
                if len(distinct) >= 6:
                    break
            return distinct

        coarse = global_candidates()
        if not coarse:
            return None
        refined = []
        for _, coarse_x, coarse_y, coarse_yaw in coarse:
            # The map resolution is 5 cm.  Searching at that resolution is
            # enough for AMCL to refine the hypothesis, and avoids spending
            # tens of seconds checking nearly identical poses.
            x_values = np.arange(coarse_x - 0.25, coarse_x + 0.251, 0.05)
            y_values = np.arange(coarse_y - 0.25, coarse_y + 0.251, 0.05)
            yaw_values = np.arange(
                coarse_yaw - math.radians(8.0),
                coarse_yaw + math.radians(8.01),
                math.radians(1.5),
            )
            for yaw in yaw_values:
                for y in y_values:
                    for x in x_values:
                        if not footprint_clear(float(x), float(y), float(yaw)):
                            continue
                        score, mean_distance, close_12, close_25, valid_fraction = scalar_score(
                            float(x), float(y), float(yaw), base_points
                        )
                        col = int(round((float(x) - origin_x) / resolution))
                        row = int(round((float(y) - origin_y) / resolution))
                        if (
                            row < 0
                            or row >= height
                            or col < 0
                            or col >= width
                            or grid[row, col] != 0
                        ):
                            score += 0.40
                        refined.append(
                            (
                                score,
                                mean_distance,
                                close_12,
                                close_25,
                                valid_fraction,
                                float(x),
                                float(y),
                                float(yaw),
                            )
                        )
        refined.sort(key=lambda item: item[0])
        distinct = []
        for candidate in refined:
            if all(
                math.hypot(candidate[5] - other[5], candidate[6] - other[6]) > 0.45
                or abs(math.degrees(self._angle_difference(candidate[7], other[7]))) > 18.0
                for other in distinct
            ):
                distinct.append(candidate)
            if len(distinct) >= 2:
                break
        if not distinct:
            return None
        best = distinct[0]
        second = distinct[1] if len(distinct) > 1 else None
        # Require a genuinely strong wall registration and a little separation
        # from another distant hypothesis before handing it to AMCL.
        if best[1] > 0.16 or best[2] < 0.60 or best[4] < 0.90:
            return None
        if second is not None and second[0] - best[0] < 0.012:
            return None
        return {
            'x': round(best[5], 3),
            'y': round(best[6], 3),
            'yaw': round(math.atan2(math.sin(best[7]), math.cos(best[7])), 4),
            'score': round(best[0], 4),
            'mean_distance': round(best[1], 3),
            'wall_match': round(best[2], 3),
        }

    def start_global_localization(self, allow_motion=False):
        """Start the official AMCL global relocalization sequence.

        A scan-match pose is only a fallback hypothesis.  It must never mark
        the robot localized by itself: AMCL has to accept it and publish a
        stable, low-covariance pose first.
        """
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        with self.lock:
            if self.localization_search_active or self.localization_scan_match_active:
                return False
        self.cancel_navigation()
        if not self.localization_service_client.wait_for_service(timeout_sec=5.0):
            with self.lock:
                self.navigation_status = 'failed'
                self.navigation_feedback = {
                    'error': 'Το επίσημο AMCL global localization service δεν είναι διαθέσιμο',
                }
            raise RuntimeError('Το AMCL global localization service δεν είναι διαθέσιμο')

        if allow_motion and not self.set_autonomous_navigation_mode(True):
            raise RuntimeError(
                'Δεν ενεργοποιήθηκε το autonomous mode του Clearpath· '
                'δεν ξεκινά η περιστροφή localization για λόγους ασφάλειας.'
            )

        now = time.monotonic()
        with self.lock:
            # Drop every old pose before resetting AMCL.  Otherwise the last
            # map->odom transform can look like a successful new localization.
            self.navigation_initial_pose = None
            self.navigation_initial_pose_sent_at = 0.0
            self.navigation_pose_initialized = False
            self.navigation_amcl_pose = None
            self.navigation_amcl_covariance = None
            self.navigation_amcl_received_at = 0.0
            self.localization_good_count = 0
            self.localization_last_pose = None
            self.localization_global_service_response = False
            self.localization_last_global_reset = False
            self.localization_scan_match_attempted = False
            self.localization_allow_motion = bool(allow_motion)
            self.localization_method = 'amcl_global'
            self.localization_search_active = True
            self.localization_scan_match_active = False
            self.localization_search_started_at = now
            self.localization_watchdog_invalid_since = 0.0
            self.localization_watchdog_last_reset_at = now
            self.localization_last_nomotion_update_at = 0.0
            self.nomotion_update_future = None
            self.navigation_status = 'localizing'
            self.navigation_goal = None
            self.navigation_feedback = {
                'message': 'Ξεκινά το επίσημο AMCL global localization',
                'method': 'amcl_global',
            }
        try:
            future = self.localization_service_client.call_async(Empty.Request())
            with self.lock:
                self.localization_service_future = future
            future.add_done_callback(self.global_localization_response_cb)
        except Exception as exc:
            self.finish_global_localization(
                False,
                f'Αποτυχία global localization: {exc}',
            )
            raise RuntimeError(f'Αποτυχία global localization: {exc}') from exc
        return True

    def global_localization_response_cb(self, future):
        try:
            future.result()
        except Exception as exc:
            self.finish_global_localization(
                False,
                f'Το AMCL global localization απέτυχε: {exc}',
            )
            return
        with self.lock:
            if self.localization_search_active:
                self.localization_global_service_response = True
                self.localization_last_global_reset = True
                self.navigation_feedback = {
                    'message': 'Το AMCL έκανε global reset· περιμένω σταθερή θέση από το LiDAR',
                    'method': 'amcl_global',
                }
        self.get_logger().info('AMCL global localization reset acknowledged')

    def _seed_scan_match_pose(self, candidate):
        """Give AMCL a provisional scan-match hypothesis after global reset."""
        with self.lock:
            if not self.localization_search_active:
                return False
            self.navigation_initial_pose = {
                'x': round(float(candidate['x']), 3),
                'y': round(float(candidate['y']), 3),
                'yaw': round(float(candidate['yaw']), 4),
                'frame': 'map',
            }
            self.navigation_initial_pose_sent_at = 0.0
            self.navigation_pose_initialized = False
            self.localization_method = 'scan_match_fallback'
            self.localization_last_pose = None
            self.localization_good_count = 0
            self.navigation_feedback = {
                'message': 'Βρέθηκε υπόθεση LiDAR· περιμένω επιβεβαίωση από AMCL',
                'method': 'scan_match_fallback',
                'match_score': candidate['score'],
                'wall_match': candidate['wall_match'],
            }
        self.publish_initial_pose()
        self.request_nomotion_update()
        return True

    def finish_global_localization(self, success, error=''):
        with self.lock:
            active = self.localization_search_active
            autonomous_mode_was_active = self.autonomous_navigation_active
            method = self.localization_method or 'amcl_global'
            self.localization_search_active = False
            self.localization_scan_match_active = False
            self.localization_service_future = None
            self.nomotion_update_future = None
            self.localization_global_service_response = False
            self.localization_scan_match_attempted = False
            self.localization_allow_motion = False
            self.localization_last_pose = None
            self.navigation_initial_pose = None
            self.navigation_initial_pose_sent_at = 0.0
            if success:
                # The optional LiDAR matcher can provide AMCL with a seed, but
                # only AMCL's stable pose/covariance confirms localization.
                method = 'amcl_global'
                self.localization_method = method
                self.auto_localize_retry_count = 0
                self.navigation_pose_initialized = True
                self.navigation_status = 'ready'
                self.navigation_feedback = {
                    'message': 'Το AMCL επιβεβαίωσε σταθερή θέση στον χάρτη',
                    'method': method,
                }
            else:
                self.navigation_pose_initialized = False
                self.navigation_status = 'failed'
                self.navigation_feedback = {'error': error or 'Δεν βρέθηκε η θέση του Dingo'}
                if (
                    active
                    and self.auto_localize
                    and self.auto_localize_retry_count < self.auto_localize_max_retries
                ):
                    self.auto_localize_retry_count += 1
                    retry_number = self.auto_localize_retry_count
                    self.auto_localize_attempted = False
                    self.auto_localize_retry_at = time.monotonic() + 15.0
                    self.navigation_feedback.update(
                        {
                            'message': (
                                'Το localization θα επαναληφθεί αυτόματα '
                                'χωρίς κίνηση'
                            ),
                            'retry_in_s': 15,
                            'retry_number': retry_number,
                        }
                    )
                    self.get_logger().warning(
                        'AMCL localization failed; scheduling safe automatic '
                        f'retry {retry_number}/{self.auto_localize_max_retries}'
                    )
        if success:
            self.get_logger().info(
                f'AMCL global localization confirmed ({method})'
            )
        elif active:
            self.get_logger().warning(
                f'AMCL global localization failed: {error or "unknown error"}'
            )
        if active or success:
            self.drive(0.0, 0.0)
        if autonomous_mode_was_active:
            self.set_autonomous_navigation_mode(False)

    def localization_tick(self):
        with self.lock:
            active = self.localization_search_active
            matching = self.localization_scan_match_active
            started_at = self.localization_search_started_at
            service_response = self.localization_global_service_response
            scan_match_attempted = self.localization_scan_match_attempted
            allow_motion = self.localization_allow_motion
            motion_blocked = (
                self.state['emergency_stop'] is True
                or self.state['safety_stop'] is True
            )
        if matching or not active:
            return
        if not self.navigation_stack_running():
            self.finish_global_localization(False, 'Το Nav2 σταμάτησε κατά την αναζήτηση')
            return
        if time.monotonic() - started_at >= self.localization_timeout_s:
            self.finish_global_localization(
                False,
                'Δεν βρέθηκε σταθερή θέση. Δοκίμασε ξανά σε πιο ανοιχτό σημείο.',
            )
            return
        if not service_response:
            return

        now = time.monotonic()
        if not scan_match_attempted and now - started_at >= self.localization_fallback_after_s:
            with self.lock:
                if not self.localization_search_active:
                    return
                self.localization_scan_match_active = True
            candidate = None
            match_started_at = time.monotonic()
            try:
                candidate = self._scan_match_pose()
            except Exception as exc:
                self.get_logger().warning(f'LiDAR map matcher unavailable: {exc}')
            match_duration = time.monotonic() - match_started_at
            with self.lock:
                self.localization_scan_match_active = False
                self.localization_scan_match_attempted = True
            if candidate is not None:
                self.get_logger().info(
                    'LiDAR map matcher found a candidate in '
                    f'{match_duration:.1f}s '
                    f'(mean error {candidate["mean_distance"]:.3f}m, '
                    f'wall match {candidate["wall_match"]:.2f})'
                )
                self._seed_scan_match_pose(candidate)
                return
            self.get_logger().warning(
                f'LiDAR map matcher found no candidate after {match_duration:.1f}s'
            )
            with self.lock:
                if self.localization_search_active:
                    self.navigation_feedback = {
                        'message': 'Το AMCL δεν έχει συγκλίνει ακόμη· δεν βρέθηκε μοναδική στατική υπόθεση LiDAR',
                        'method': 'amcl_global',
                    }

        with self.lock:
            last_update = self.localization_last_nomotion_update_at
        if now - last_update >= 0.5:
            self.request_nomotion_update()
        if allow_motion and not motion_blocked:
            self.drive(0.0, self.localization_angular_speed)
        elif allow_motion and motion_blocked:
            with self.lock:
                if self.localization_search_active:
                    self.navigation_feedback = {
                        'message': 'Το AMCL global search περιμένει απελευθέρωση E-stop για περιστροφή',
                        'method': 'amcl_global',
                    }

    def localization_heartbeat_tick(self):
        """Keep AMCL's pose topic fresh while the stationary robot is localized.

        AMCL normally publishes a new pose after it observes enough motion.
        That is correct for the filter, but a Dashboard that treats the topic
        timestamp as the whole localization state can incorrectly show an
        interruption while the robot is sitting still.  AMCL exposes an
        official no-motion update service for exactly this case.  Use it only
        after localization has succeeded and only while odometry says the
        robot is stationary; this never commands motion.
        """
        if not self.navigation_stack_running():
            return
        with self.lock:
            if (
                self.localization_search_active
                or self.localization_scan_match_active
                or not self.navigation_pose_initialized
            ):
                return
            pose = dict(self.navigation_amcl_pose) if self.navigation_amcl_pose else None
            covariance = (
                dict(self.navigation_amcl_covariance)
                if self.navigation_amcl_covariance
                else None
            )
            odom = dict(self.state.get('odom') or {})
            last_request = self.localization_last_nomotion_update_at
        try:
            linear_speed = abs(float(odom.get('linear') or 0.0))
            angular_speed = abs(float(odom.get('angular') or 0.0))
        except (TypeError, ValueError):
            return
        if linear_speed > 0.015 or angular_speed > 0.03:
            return
        if not pose or not self.amcl_pose_is_good(covariance):
            return
        if not self.map_pose_is_safe(pose):
            return
        if self.lookup_transform('map', 'base_link') is None:
            return
        if time.monotonic() - last_request < 2.0:
            return
        self.request_nomotion_update()

    def request_nomotion_update(self):
        if not self.nomotion_update_client.service_is_ready():
            return False
        with self.lock:
            current = self.nomotion_update_future
            if current is not None and not current.done():
                return False
        try:
            future = self.nomotion_update_client.call_async(Empty.Request())
        except Exception as exc:
            self.get_logger().debug(f'AMCL no-motion update unavailable: {exc}')
            return False
        with self.lock:
            self.nomotion_update_future = future
            self.localization_last_nomotion_update_at = time.monotonic()
        return True

    def localization_is_good(self):
        with self.lock:
            pose_initialized = self.navigation_pose_initialized
            covariance = dict(self.navigation_amcl_covariance) if self.navigation_amcl_covariance else None
            pose = dict(self.navigation_amcl_pose) if self.navigation_amcl_pose else None
        return bool(
            pose_initialized
            and self.amcl_pose_is_good(covariance)
            and self.map_pose_is_safe(pose)
            and self.lookup_transform('map', 'base_link') is not None
        )

    def navigation_feedback_cb(self, feedback_message):
        feedback = feedback_message.feedback
        with self.lock:
            self.navigation_feedback = {
                'distance_remaining': round(float(feedback.distance_remaining), 2),
                'recoveries': int(feedback.number_of_recoveries),
                'current_x': round(float(feedback.current_pose.pose.position.x), 3),
                'current_y': round(float(feedback.current_pose.pose.position.y), 3),
            }
            if self.navigation_status not in ('canceling', 'canceled'):
                self.navigation_status = 'navigating'

    def navigation_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:  # rclpy action futures surface transport errors here
            with self.lock:
                self.navigation_goal_handle = None
                self.navigation_status = 'failed'
                self.navigation_feedback = {'error': str(exc)}
            self.clear_navigation_path()
            self.set_autonomous_navigation_mode(False)
            return

        if not goal_handle.accepted:
            with self.lock:
                self.navigation_goal_handle = None
                self.navigation_status = 'rejected'
                self.navigation_feedback = {'error': 'Ο στόχος απορρίφθηκε από το Nav2'}
            self.clear_navigation_path()
            self.set_autonomous_navigation_mode(False)
            return

        with self.lock:
            self.navigation_goal_handle = goal_handle
            self.navigation_status = 'navigating'
        result_future = goal_handle.get_result_async()
        with self.lock:
            self.navigation_result_future = result_future
        result_future.add_done_callback(self.navigation_result_cb)

    def navigation_result_cb(self, future):
        status = None
        error_code = 0
        try:
            result = future.result()
            status = result.status
            if status == GoalStatus.STATUS_SUCCEEDED:
                navigation_status = 'succeeded'
            elif status == GoalStatus.STATUS_CANCELED:
                navigation_status = 'canceled'
            else:
                navigation_status = 'failed'
            error = getattr(result.result, 'error_msg', '')
            error_code = int(getattr(result.result, 'error_code', 0) or 0)
            if navigation_status == 'failed' and not error:
                error = 'Το Nav2 δεν ολοκλήρωσε τον στόχο'
                if error_code:
                    error += f' (κωδικός {error_code})'
        except Exception as exc:  # keep the Dashboard alive if Nav2 disappears
            navigation_status = 'failed'
            error = str(exc)
        with self.lock:
            self.navigation_goal_handle = None
            self.navigation_result_future = None
            self.navigation_status = navigation_status
            # A completed goal must not remain drawn on the live map.  The
            # robot marker is sourced from the current AMCL TF, while the
            # goal overlay is only for an active request.
            self.navigation_goal = None
            self.navigation_feedback = {'error': error} if error else None
        self.clear_navigation_path()

        if navigation_status == 'failed':
            self.get_logger().warning(
                f'Nav2 navigation failed (status={status}, error_code={error_code}): {error}'
            )
            self.publish_voice_reply(
                f'Η διαδρομή δεν εκτελέστηκε: {error}.',
                ok=False,
                action='navigate_room',
            )

        with self.lock:
            patrol_active = self.patrol_active
            if patrol_active and navigation_status == 'succeeded':
                self.patrol_completed += 1
        # A terminal Nav2 result must return twist_mux to the fail-closed
        # joystick-quality priority before another goal or manual teleop can
        # begin.  Patrol will explicitly re-enable the mode for its next leg.
        self.set_autonomous_navigation_mode(False)
        if patrol_active and navigation_status == 'succeeded':
            self.start_next_patrol_goal()
        elif patrol_active:
            self.cancel_patrol()
            reason = error or 'ο στόχος δεν ολοκληρώθηκε'
            self.publish_voice_reply(
                f'Η βόλτα σταμάτησε: {reason}.',
                ok=False,
                action='patrol',
            )

    def start_patrol(self, rounds=1):
        try:
            rounds = int(rounds or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError('Ο αριθμός γύρων δεν είναι έγκυρος') from exc
        rounds = max(1, min(rounds, 3))
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        if not self.navigation_client.wait_for_server(timeout_sec=1.0):
            raise RuntimeError('Το Nav2 δεν είναι ακόμη έτοιμο· περίμενε λίγο')
        if not self.localization_is_good():
            raise RuntimeError(
                'Περίμενε να ολοκληρωθεί το αυτόματο Global Localization'
            )
        with self.lock:
            rooms = [dict(room) for room in self.rooms]
            blocked = (
                self.state.get('emergency_stop') is True
                or self.state.get('safety_stop') is True
            )
            if blocked:
                raise RuntimeError(
                    'Η κίνηση είναι μπλοκαρισμένη από το safety ή emergency stop.'
                )
            if self.navigation_goal_handle is not None:
                raise RuntimeError('Υπάρχει ενεργός στόχος· ακύρωσέ τον πρώτα')
            if self.drive_heading_goal_handle is not None or self.drive_heading_status in (
                'sending',
                'moving',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή κίνηση σε απόσταση· ακύρωσέ την πρώτα')
            if self.spin_goal_handle is not None or self.spin_status in (
                'sending',
                'spinning',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή στροφή· ακύρωσέ την πρώτα')
            if self.patrol_active:
                raise RuntimeError('Η βόλτα είναι ήδη σε εξέλιξη')
        if len(rooms) < 2:
            raise RuntimeError(
                'Χρειάζονται τουλάχιστον δύο αποθηκευμένα δωμάτια για βόλτα.'
            )

        route = [dict(room) for _ in range(rounds) for room in rooms]
        with self.lock:
            self.patrol_active = True
            self.patrol_queue = route
            self.patrol_current = None
            self.patrol_total = len(route)
            self.patrol_completed = 0
        return self.start_next_patrol_goal()

    def start_next_patrol_goal(self):
        with self.lock:
            if not self.patrol_active:
                return False
            if not self.patrol_queue:
                total = self.patrol_total
                completed = self.patrol_completed
                self.patrol_active = False
                self.patrol_current = None
                done = True
                room = None
            else:
                room = dict(self.patrol_queue.pop(0))
                self.patrol_current = dict(room)
                done = False
        if done:
            self.publish_voice_reply(
                f'Η βόλτα ολοκληρώθηκε: {completed}/{total} σημεία.',
                action='patrol',
            )
            return False
        try:
            self.send_navigation_goal(room['x'], room['y'], internal=True)
        except (RuntimeError, ValueError) as exc:
            self.cancel_patrol()
            self.publish_voice_reply(
                f'Η βόλτα δεν ξεκίνησε: {exc}.',
                ok=False,
                action='patrol',
            )
            return False
        return True

    def cancel_patrol(self):
        with self.lock:
            active = self.patrol_active
            self.patrol_active = False
            self.patrol_queue = []
            self.patrol_current = None
        return active

    def drive_heading_feedback_cb(self, feedback_message):
        feedback = feedback_message.feedback
        with self.lock:
            self.drive_heading_feedback = {
                'distance_traveled': round(float(feedback.distance_traveled), 3),
            }
            if self.drive_heading_status not in ('canceling', 'canceled'):
                self.drive_heading_status = 'moving'

    def drive_heading_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self.lock:
                self.drive_heading_goal_handle = None
                self.drive_heading_result_future = None
                self.drive_heading_status = 'failed'
                self.drive_heading_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            self.publish_voice_reply(
                f'Η κίνηση απέτυχε: {exc}.',
                ok=False,
                action='move_distance',
            )
            return
        if not goal_handle.accepted:
            with self.lock:
                self.drive_heading_goal_handle = None
                self.drive_heading_result_future = None
                self.drive_heading_status = 'rejected'
                self.drive_heading_feedback = {
                    'error': 'Η κίνηση απορρίφθηκε από το Nav2'
                }
            self.set_autonomous_navigation_mode(False)
            self.publish_voice_reply(
                'Η κίνηση απορρίφθηκε από το Nav2.',
                ok=False,
                action='move_distance',
            )
            return

        with self.lock:
            cancel_requested = self.drive_heading_status == 'canceling'
            self.drive_heading_goal_handle = goal_handle
            self.drive_heading_status = 'canceling' if cancel_requested else 'moving'
        if cancel_requested:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return
        result_future = goal_handle.get_result_async()
        with self.lock:
            self.drive_heading_result_future = result_future
        result_future.add_done_callback(self.drive_heading_result_cb)

    def drive_heading_result_cb(self, future):
        try:
            result = future.result()
            status = result.status
            error_code = int(getattr(result.result, 'error_code', 0) or 0)
            error = str(getattr(result.result, 'error_msg', '') or '')
            if status == GoalStatus.STATUS_SUCCEEDED:
                drive_status = 'succeeded'
                ok = True
                text = 'Η κίνηση ολοκληρώθηκε.'
            elif status == GoalStatus.STATUS_CANCELED:
                drive_status = 'canceled'
                ok = False
                text = 'Η κίνηση ακυρώθηκε.'
            else:
                drive_status = 'failed'
                ok = False
                if error_code == 723:
                    text = 'Η κίνηση σταμάτησε: εντοπίστηκε εμπόδιο από το Nav2.'
                elif error_code == 721:
                    text = 'Η κίνηση σταμάτησε επειδή έληξε ο χρόνος ασφαλείας.'
                else:
                    detail = f': {error}' if error else ''
                    text = f'Η κίνηση απέτυχε{detail}.'
        except Exception as exc:
            drive_status = 'failed'
            ok = False
            text = f'Η κίνηση απέτυχε: {exc}.'
        with self.lock:
            self.drive_heading_goal_handle = None
            self.drive_heading_result_future = None
            self.drive_heading_status = drive_status
            if not self.drive_heading_feedback:
                self.drive_heading_feedback = {'error': text} if not ok else None
        self.set_autonomous_navigation_mode(False)
        self.publish_voice_reply(text, ok=ok, action='move_distance')

    def start_drive_heading(self, distance_m):
        try:
            distance_m = float(distance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError('Η απόσταση δεν είναι έγκυρη') from exc
        if (
            not math.isfinite(distance_m)
            or abs(distance_m) < 0.05
            or abs(distance_m) > 3.0
        ):
            raise ValueError('Η απόσταση πρέπει να είναι από 5 εκατοστά έως 3 μέτρα')
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        if not self.drive_heading_client.wait_for_server(timeout_sec=1.0):
            raise RuntimeError('Το Nav2 δεν παρέχει ακόμη την κίνηση σε απόσταση')
        with self.lock:
            blocked = (
                self.state.get('emergency_stop') is True
                or self.state.get('safety_stop') is True
            )
            if blocked:
                raise RuntimeError(
                    'Η κίνηση είναι μπλοκαρισμένη από το safety ή emergency stop.'
                )
            if self.follow_active:
                raise RuntimeError('Το follow-me είναι ενεργό· σταμάτησέ το πρώτα')
            if self.navigation_goal_handle is not None or self.patrol_active:
                raise RuntimeError('Υπάρχει ενεργή πλοήγηση· ακύρωσέ την πρώτα')
            if self.spin_goal_handle is not None or self.spin_status in (
                'sending',
                'spinning',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή στροφή· ακύρωσέ την πρώτα')
            if self.drive_heading_goal_handle is not None or self.drive_heading_status in (
                'sending',
                'moving',
                'canceling',
            ):
                raise RuntimeError('Η κίνηση σε απόσταση είναι ήδη σε εξέλιξη')
            limits = dict(self.settings)
            max_speed = float(limits.get('max_linear_speed', 0.15) or 0.15)
            speed = max(0.05, min(0.18, abs(max_speed)))
            goal = DriveOnHeading.Goal()
            goal.target.x = distance_m
            goal.target.y = 0.0
            goal.target.z = 0.0
            goal.speed = speed if distance_m > 0.0 else -speed
            allowance = max(15.0, min(90.0, abs(distance_m) / speed * 4.0 + 8.0))
            goal.time_allowance.sec = int(allowance)
            goal.time_allowance.nanosec = int(
                (allowance - int(allowance)) * 1_000_000_000
            )
            self.drive_heading_status = 'sending'
            self.drive_heading_feedback = None
            self.drive_heading_distance_m = round(distance_m, 3)
        if not self.set_autonomous_navigation_mode(True):
            with self.lock:
                self.drive_heading_status = 'failed'
                self.drive_heading_feedback = {
                    'error': 'Το autonomous mode δεν ενεργοποιήθηκε'
                }
            raise RuntimeError(
                'Δεν ενεργοποιήθηκε το autonomous mode του Clearpath· '
                'δεν στάλθηκε η κίνηση για λόγους ασφάλειας.'
            )
        try:
            future = self.drive_heading_client.send_goal_async(
                goal,
                feedback_callback=self.drive_heading_feedback_cb,
            )
            future.add_done_callback(self.drive_heading_goal_response_cb)
        except Exception as exc:
            with self.lock:
                self.drive_heading_status = 'failed'
                self.drive_heading_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            raise RuntimeError(f'Δεν στάλθηκε η κίνηση στο Nav2: {exc}') from exc
        return True

    def cancel_drive_heading(self):
        with self.lock:
            goal_handle = self.drive_heading_goal_handle
            active = self.drive_heading_status in ('sending', 'moving', 'canceling')
            if active:
                self.drive_heading_status = 'canceling'
        if goal_handle is None:
            if active:
                self.set_autonomous_navigation_mode(False)
            return active
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:
            with self.lock:
                self.drive_heading_status = 'failed'
                self.drive_heading_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            return False
        self.set_autonomous_navigation_mode(False)
        return True

    def spin_feedback_cb(self, feedback_message):
        feedback = feedback_message.feedback
        with self.lock:
            self.spin_feedback = {
                'angular_distance_traveled': round(
                    float(feedback.angular_distance_traveled), 3
                ),
            }

    def spin_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self.lock:
                self.spin_goal_handle = None
                self.spin_result_future = None
                self.spin_status = 'failed'
                self.spin_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            self.publish_voice_reply(
                f'Η στροφή απέτυχε: {exc}.', ok=False, action='rotate'
            )
            return
        if not goal_handle.accepted:
            with self.lock:
                self.spin_goal_handle = None
                self.spin_result_future = None
                self.spin_status = 'rejected'
                self.spin_feedback = {'error': 'Η στροφή απορρίφθηκε από το Nav2'}
            self.set_autonomous_navigation_mode(False)
            self.publish_voice_reply(
                'Η στροφή απορρίφθηκε από το Nav2.', ok=False, action='rotate'
            )
            return

        with self.lock:
            cancel_requested = self.spin_status == 'canceling'
            self.spin_goal_handle = goal_handle
            self.spin_status = 'canceling' if cancel_requested else 'spinning'
        if cancel_requested:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return
        result_future = goal_handle.get_result_async()
        with self.lock:
            self.spin_result_future = result_future
        result_future.add_done_callback(self.spin_result_cb)

    def spin_result_cb(self, future):
        try:
            result = future.result()
            status = result.status
            if status == GoalStatus.STATUS_SUCCEEDED:
                spin_status = 'succeeded'
                ok = True
                text = 'Η στροφή ολοκληρώθηκε.'
            elif status == GoalStatus.STATUS_CANCELED:
                spin_status = 'canceled'
                ok = False
                text = 'Η στροφή ακυρώθηκε.'
            else:
                spin_status = 'failed'
                ok = False
                error = str(getattr(result.result, 'error_msg', '') or '')
                text = f'Η στροφή απέτυχε{": " + error if error else ""}.'
        except Exception as exc:
            spin_status = 'failed'
            ok = False
            text = f'Η στροφή απέτυχε: {exc}.'
        with self.lock:
            self.spin_goal_handle = None
            self.spin_result_future = None
            self.spin_status = spin_status
        self.set_autonomous_navigation_mode(False)
        self.publish_voice_reply(text, ok=ok, action='rotate')

    def start_spin(self, degrees=360.0):
        try:
            degrees = float(degrees)
        except (TypeError, ValueError) as exc:
            raise ValueError('Η στροφή πρέπει να έχει έγκυρες μοίρες') from exc
        if not math.isfinite(degrees) or abs(degrees) < 1.0 or abs(degrees) > 360.0:
            raise ValueError('Η στροφή πρέπει να είναι από 1 έως 360 μοίρες')
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        if not self.spin_client.wait_for_server(timeout_sec=1.0):
            raise RuntimeError('Το Nav2 δεν παρέχει ακόμη την εντολή Spin')
        if not self.localization_is_good():
            raise RuntimeError(
                'Περίμενε να ολοκληρωθεί το αυτόματο Global Localization'
            )
        with self.lock:
            blocked = (
                self.state.get('emergency_stop') is True
                or self.state.get('safety_stop') is True
            )
            if blocked:
                raise RuntimeError(
                    'Η κίνηση είναι μπλοκαρισμένη από το safety ή emergency stop.'
                )
            if self.navigation_goal_handle is not None or self.patrol_active:
                raise RuntimeError('Υπάρχει ενεργή πλοήγηση· ακύρωσέ την πρώτα')
            if self.drive_heading_goal_handle is not None or self.drive_heading_status in (
                'sending',
                'moving',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή κίνηση σε απόσταση· ακύρωσέ την πρώτα')
            if self.spin_goal_handle is not None or self.spin_status in (
                'sending',
                'spinning',
                'canceling',
            ):
                raise RuntimeError('Η στροφή είναι ήδη σε εξέλιξη')
            goal = Spin.Goal()
            goal.target_yaw = float(math.radians(degrees))
            allowance = max(20.0, min(120.0, abs(goal.target_yaw) / 0.25 + 10.0))
            goal.time_allowance.sec = int(allowance)
            goal.time_allowance.nanosec = int(
                (allowance - int(allowance)) * 1_000_000_000
            )
            self.spin_status = 'sending'
            self.spin_feedback = None
            self.spin_degrees = round(degrees, 1)
        if not self.set_autonomous_navigation_mode(True):
            with self.lock:
                self.spin_status = 'failed'
                self.spin_feedback = {
                    'error': 'Το autonomous mode δεν ενεργοποιήθηκε'
                }
            raise RuntimeError(
                'Δεν ενεργοποιήθηκε το autonomous mode του Clearpath· '
                'δεν στάλθηκε η στροφή για λόγους ασφάλειας.'
            )
        try:
            future = self.spin_client.send_goal_async(
                goal,
                feedback_callback=self.spin_feedback_cb,
            )
            future.add_done_callback(self.spin_goal_response_cb)
        except Exception as exc:
            with self.lock:
                self.spin_status = 'failed'
                self.spin_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            raise RuntimeError(f'Δεν στάλθηκε η στροφή στο Nav2: {exc}') from exc
        return True

    def cancel_spin(self):
        with self.lock:
            goal_handle = self.spin_goal_handle
            active = self.spin_status in ('sending', 'spinning', 'canceling')
            if active:
                self.spin_status = 'canceling'
        if goal_handle is None:
            if active:
                self.set_autonomous_navigation_mode(False)
            return active
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:
            with self.lock:
                self.spin_status = 'failed'
                self.spin_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            return False
        self.set_autonomous_navigation_mode(False)
        return True

    def clear_navigation_path(self):
        """Invalidate and cancel the current planner request, if any."""
        with self.lock:
            path_goal_handle = self.navigation_path_goal_handle
            self.navigation_path_request_id += 1
            self.navigation_path_goal_handle = None
            self.navigation_path_result_future = None
            self.navigation_path = None
            self.navigation_path_status = 'idle'
            self.navigation_path_error = None
        if path_goal_handle is not None:
            try:
                path_goal_handle.cancel_goal_async()
            except Exception:
                pass

    def request_navigation_path(self, x, y):
        """Ask Nav2 for the planner path that corresponds to the active goal."""
        with self.lock:
            previous_handle = self.navigation_path_goal_handle
            self.navigation_path_request_id += 1
            request_id = self.navigation_path_request_id
            self.navigation_path_goal_handle = None
            self.navigation_path_result_future = None
            self.navigation_path = None
            self.navigation_path_status = 'computing'
            self.navigation_path_error = None
        if previous_handle is not None:
            try:
                previous_handle.cancel_goal_async()
            except Exception:
                pass

        try:
            if not self.navigation_path_client.wait_for_server(timeout_sec=1.0):
                raise RuntimeError('Το Nav2 δεν παρέχει ακόμη planner path')
            start = self.lookup_transform('map', 'base_link')
            if start is None:
                raise RuntimeError('Δεν βρέθηκε η τρέχουσα θέση του Dingo στον χάρτη')

            stamp = self.get_clock().now().to_msg()
            request = ComputePathToPose.Goal()
            request.goal = PoseStamped()
            request.goal.header.frame_id = 'map'
            request.goal.header.stamp = stamp
            request.goal.pose.position.x = float(x)
            request.goal.pose.position.y = float(y)
            request.goal.pose.orientation.z = math.sin(float(start['yaw']) / 2.0)
            request.goal.pose.orientation.w = math.cos(float(start['yaw']) / 2.0)
            request.start = PoseStamped()
            request.start.header.frame_id = 'map'
            request.start.header.stamp = stamp
            request.start.pose.position.x = float(start['x'])
            request.start.pose.position.y = float(start['y'])
            request.start.pose.orientation.z = math.sin(float(start['yaw']) / 2.0)
            request.start.pose.orientation.w = math.cos(float(start['yaw']) / 2.0)
            request.use_start = True
            request.planner_id = ''
            future = self.navigation_path_client.send_goal_async(request)
            with self.lock:
                if request_id != self.navigation_path_request_id:
                    return False
                self.navigation_path_result_future = future
            future.add_done_callback(
                lambda result_future: self.navigation_path_goal_response_cb(
                    result_future, request_id
                )
            )
            return True
        except Exception as exc:
            with self.lock:
                if (
                    request_id == self.navigation_path_request_id
                    and self.navigation_path_status != 'ready'
                ):
                    self.navigation_path_status = 'failed'
                    self.navigation_path_error = str(exc)
            self.get_logger().warning(f'Could not compute Nav2 path: {exc}')
            return False

    def navigation_path_goal_response_cb(self, future, request_id):
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self.lock:
                if request_id == self.navigation_path_request_id:
                    self.navigation_path_result_future = None
                    self.navigation_path_status = 'failed'
                    self.navigation_path_error = str(exc)
            return

        with self.lock:
            current = request_id == self.navigation_path_request_id
        if not current:
            if goal_handle is not None and goal_handle.accepted:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
            return
        if goal_handle is None or not goal_handle.accepted:
            with self.lock:
                self.navigation_path_goal_handle = None
                self.navigation_path_result_future = None
                self.navigation_path_status = 'failed'
                self.navigation_path_error = 'Ο planner του Nav2 απέρριψε τον υπολογισμό διαδρομής'
            return

        result_future = goal_handle.get_result_async()
        with self.lock:
            if request_id != self.navigation_path_request_id:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return
            self.navigation_path_goal_handle = goal_handle
            self.navigation_path_result_future = result_future
        result_future.add_done_callback(
            lambda path_future: self.navigation_path_result_cb(path_future, request_id)
        )

    def navigation_path_result_cb(self, future, request_id):
        path = []
        error = ''
        succeeded = False
        try:
            result = future.result()
            result_message = result.result
            succeeded = result.status == GoalStatus.STATUS_SUCCEEDED
            error = str(getattr(result_message, 'error_msg', '') or '')
            error_code = int(getattr(result_message, 'error_code', 0) or 0)
            if not succeeded and not error:
                error = f'Ο planner δεν βρήκε διαδρομή (κωδικός {error_code})'
            for pose_stamped in getattr(result_message.path, 'poses', []) or []:
                point = pose_stamped.pose.position
                point_x = float(point.x)
                point_y = float(point.y)
                if math.isfinite(point_x) and math.isfinite(point_y):
                    path.append([round(point_x, 3), round(point_y, 3)])
            if succeeded and not path:
                error = 'Ο planner επέστρεψε κενή διαδρομή'
        except Exception as exc:
            error = str(exc)

        with self.lock:
            if request_id != self.navigation_path_request_id:
                return
            self.navigation_path_goal_handle = None
            self.navigation_path_result_future = None
            self.navigation_path = path or None
            self.navigation_path_status = 'ready' if succeeded and path else 'failed'
            self.navigation_path_error = error or None

    def send_navigation_goal(self, x, y, internal=False):
        if self.follow_active:
            self.stop_follow(silent=True)
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError('Ο στόχος πρέπει να έχει έγκυρες συντεταγμένες') from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Ο στόχος πρέπει να έχει έγκυρες συντεταγμένες')
        if not self.navigation_stack_running():
            raise RuntimeError('Πάτησε πρώτα «Ενεργοποίηση Nav2»')
        if not self.navigation_client.wait_for_server(timeout_sec=1.0):
            raise RuntimeError('Το Nav2 δεν είναι ακόμη έτοιμο· περίμενε λίγο')
        if not self.localization_is_good():
            raise RuntimeError(
                'Περίμενε να ολοκληρωθεί το αυτόματο Global Localization'
            )
        with self.lock:
            if self.navigation_goal_handle is not None or self.navigation_status in (
                'sending',
                'navigating',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργός στόχος· ακύρωσέ τον πρώτα')
            if self.drive_heading_goal_handle is not None or self.drive_heading_status in (
                'sending',
                'moving',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή κίνηση σε απόσταση· ακύρωσέ την πρώτα')
            if self.patrol_active and not internal:
                raise RuntimeError('Η βόλτα είναι ήδη σε εξέλιξη')
            if self.spin_goal_handle is not None or self.spin_status in (
                'sending',
                'spinning',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή στροφή· ακύρωσέ την πρώτα')
            pose = self.lookup_transform('map', 'base_link')
            yaw = float(pose['yaw']) if pose else 0.0
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = x
            goal.pose.pose.position.y = y
            goal.pose.pose.position.z = 0.0
            goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
            self.navigation_goal = {
                'x': round(x, 3),
                'y': round(y, 3),
                'yaw': round(yaw, 4),
                'frame': 'map',
            }
            self.navigation_feedback = None
            self.navigation_status = 'sending'
        if not self.set_autonomous_navigation_mode(True):
            with self.lock:
                self.navigation_status = 'failed'
                self.navigation_goal = None
                self.navigation_feedback = {
                    'error': (
                        'Το autonomous mode δεν ενεργοποιήθηκε· '
                        'δεν στάλθηκε εντολή κίνησης'
                    )
                }
            self.clear_navigation_path()
            raise RuntimeError(
                'Δεν ενεργοποιήθηκε το autonomous mode του Clearpath· '
                'δεν στάλθηκε ο στόχος για λόγους ασφάλειας.'
            )
        try:
            future = self.navigation_client.send_goal_async(
                goal,
                feedback_callback=self.navigation_feedback_cb,
            )
            future.add_done_callback(self.navigation_goal_response_cb)
        except Exception as exc:
            with self.lock:
                self.navigation_status = 'failed'
                self.navigation_feedback = {'error': str(exc)}
                self.navigation_goal = None
            self.clear_navigation_path()
            self.set_autonomous_navigation_mode(False)
            raise RuntimeError(f'Δεν στάλθηκε ο στόχος στο Nav2: {exc}') from exc
        self.request_navigation_path(x, y)
        return True

    def cancel_navigation(self):
        self.cancel_patrol()
        with self.lock:
            goal_handle = self.navigation_goal_handle
            autonomous_mode_was_active = self.autonomous_navigation_active
            if goal_handle is not None:
                self.navigation_status = 'canceling'
        self.clear_navigation_path()
        if goal_handle is None:
            if autonomous_mode_was_active:
                self.set_autonomous_navigation_mode(False)
            return False
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:
            with self.lock:
                self.navigation_status = 'failed'
                self.navigation_feedback = {'error': str(exc)}
            self.set_autonomous_navigation_mode(False)
            return False
        self.set_autonomous_navigation_mode(False)
        return True

    def stop_navigation(self):
        self.stop_follow(silent=True)
        self.cancel_drive_heading()
        self.cancel_spin()
        self.cancel_navigation()
        stopped = self.stop_process('navigation_process')
        self.set_autonomous_navigation_mode(False)
        with self.lock:
            self.localization_search_active = False
            self.localization_scan_match_active = False
            self.localization_service_future = None
            self.nomotion_update_future = None
            self.localization_good_count = 0
            self.localization_last_pose = None
            self.localization_global_service_response = False
            self.localization_scan_match_attempted = False
            self.localization_allow_motion = False
            self.localization_method = None
            if stopped or self.navigation_status in (
                'starting',
                'ready',
                'sending',
                'navigating',
                'canceling',
            ):
                self.navigation_status = 'idle'
            self.navigation_goal_handle = None
            self.navigation_result_future = None
            self.navigation_goal = None
            self.navigation_feedback = None
            self.navigation_initial_pose = None
            self.navigation_initial_pose_sent_at = 0.0
            self.navigation_pose_initialized = False
            self.navigation_amcl_pose = None
            self.navigation_amcl_covariance = None
            self.navigation_amcl_received_at = 0.0
            self.spin_goal_handle = None
            self.spin_result_future = None
            self.spin_status = 'idle'
            self.spin_feedback = None
            self.spin_degrees = None
            self.drive_heading_goal_handle = None
            self.drive_heading_result_future = None
            self.drive_heading_status = 'idle'
            self.drive_heading_feedback = None
            self.drive_heading_distance_m = None
        self.drive(0.0, 0.0)
        return stopped

    def navigation_snapshot(self):
        running = self.navigation_stack_running()
        action_ready = self.navigation_client.server_is_ready()
        spin_action_ready = self.spin_client.server_is_ready()
        drive_heading_action_ready = self.drive_heading_client.server_is_ready()
        restore_after_snapshot = False
        with self.lock:
            if not running and self.navigation_status in (
                'starting',
                'ready',
                'sending',
                'navigating',
                'canceling',
            ):
                self.navigation_status = 'idle'
                restore_after_snapshot = True
            status = self.navigation_status
            goal = dict(self.navigation_goal) if self.navigation_goal else None
            feedback = (
                dict(self.navigation_feedback) if self.navigation_feedback else None
            )
            path = (
                [list(point) for point in self.navigation_path]
                if self.navigation_path
                else None
            )
            path_status = self.navigation_path_status
            path_error = self.navigation_path_error
            map_name = self.navigation_map
            initial_pose = (
                dict(self.navigation_initial_pose)
                if self.navigation_initial_pose
                else None
            )
            pose_initialized = self.navigation_pose_initialized
            amcl_pose = (
                dict(self.navigation_amcl_pose)
                if self.navigation_amcl_pose
                else None
            )
            covariance = (
                dict(self.navigation_amcl_covariance)
                if self.navigation_amcl_covariance
                else None
            )
            localization_searching = (
                self.localization_search_active or self.localization_scan_match_active
            )
            localization_method = self.localization_method
            localization_good_count = self.localization_good_count
            localization_good_required = (
                min(
                    self.localization_good_required,
                    self.localization_scan_match_good_required,
                )
                if localization_method == 'scan_match_fallback'
                else self.localization_good_required
            )
            localization_global_service_response = (
                self.localization_global_service_response
            )
            localization_last_global_reset = self.localization_last_global_reset
            localization_allow_motion = self.localization_allow_motion
            navigation_lifecycle_active = self.navigation_lifecycle_done
            spin_status = self.spin_status
            spin_feedback = (
                dict(self.spin_feedback) if self.spin_feedback else None
            )
            spin_degrees = self.spin_degrees
            drive_heading_status = self.drive_heading_status
            drive_heading_feedback = (
                dict(self.drive_heading_feedback)
                if self.drive_heading_feedback
                else None
            )
            drive_heading_distance_m = self.drive_heading_distance_m
            patrol_active = self.patrol_active
            patrol_current = (
                dict(self.patrol_current) if self.patrol_current else None
            )
            patrol_total = self.patrol_total
            patrol_completed = self.patrol_completed
            autonomous_navigation_active = self.autonomous_navigation_active
            autonomous_navigation_priority = self.autonomous_navigation_priority
            autonomous_navigation_last_error = self.autonomous_navigation_last_error
        if restore_after_snapshot:
            self.set_autonomous_navigation_mode(False)
        if running and action_ready and status == 'starting':
            status = 'ready'
            with self.lock:
                self.navigation_status = status
        return {
            'running': running,
            'action_ready': bool(action_ready),
            'navigation_lifecycle_active': bool(navigation_lifecycle_active),
            'status': status,
            'map': map_name,
            'localized': bool(
                pose_initialized
                and self.amcl_pose_is_good(covariance)
                and self.map_pose_is_safe(amcl_pose)
                and self.lookup_transform('map', 'base_link') is not None
            ),
            'localization_searching': localization_searching,
            'localization_method': localization_method,
            'localization_good_count': localization_good_count,
            'localization_good_required': localization_good_required,
            'global_localization_service_ready': bool(
                self.localization_service_client.service_is_ready()
            ),
            'global_localization_reset_done': localization_global_service_response,
            'global_localization_last_reset_done': localization_last_global_reset,
            'global_localization_allow_motion': localization_allow_motion,
            'amcl_pose': amcl_pose,
            'localization_covariance': covariance,
            'initial_pose': initial_pose,
            'goal': goal,
            'feedback': feedback,
            'path': path,
            'path_status': path_status,
            'path_error': path_error,
            'autonomous_mode': {
                'active': bool(autonomous_navigation_active),
                'bt_quality_bypassed': bool(
                    autonomous_navigation_active
                    and autonomous_navigation_priority
                    == self.bt_quality_autonomous_priority
                ),
                'bt_quality_priority': autonomous_navigation_priority,
                'last_error': autonomous_navigation_last_error,
                'safety_locks_retained': True,
            },
            'spin': {
                'action_ready': bool(spin_action_ready),
                'status': spin_status,
                'degrees': spin_degrees,
                'feedback': spin_feedback,
            },
            'drive_on_heading': {
                'action_ready': bool(drive_heading_action_ready),
                'status': drive_heading_status,
                'distance_m': drive_heading_distance_m,
                'feedback': drive_heading_feedback,
            },
            'patrol': {
                'active': patrol_active,
                'current': patrol_current,
                'total': patrol_total,
                'completed': patrol_completed,
            },
        }

    def process_running(self, attribute):
        with self.process_lock:
            process = getattr(self, attribute)
            if process is not None and process.poll() is not None:
                setattr(self, attribute, None)
                process = None
            return process is not None

    def navigation_stack_running(self):
        """Return whether Dashboard-owned or externally launched Nav2 is up."""
        return self.external_navigation or self.process_running('navigation_process')

    def mapping_snapshot(self):
        return {
            'running': self.process_running('mapping_process'),
            'saving': self.process_running('map_save_process'),
            'last_save': self.last_map_save_name,
            'map_available': self.map_snapshot() is not None,
        }

    def camera_status(self):
        with self.lock:
            status = dict(self.state['camera'])
        status['process_running'] = (
            self.process_running('camera_process') or self.camera_driver_active()
        )
        return status

    def save_map(self, name):
        with self.lock:
            if self.map_state is None:
                raise RuntimeError('Δεν υπάρχει ακόμη ενεργός χάρτης')
        name = str(name or '').strip()
        safe_name = re.sub(r'[^\w.-]+', '_', name, flags=re.UNICODE).strip('._')
        safe_name = safe_name[:60] or 'dingo_map'
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        with self.process_lock:
            if self.map_save_process is not None and self.map_save_process.poll() is None:
                raise RuntimeError('Η αποθήκευση χάρτη είναι ήδη σε εξέλιξη')
            try:
                process = subprocess.Popen(
                    self.ros2_command(
                        'run',
                        'nav2_map_server',
                        'map_saver_cli',
                        '-t',
                        self.map_topic,
                        '-f',
                        str(self.maps_dir / safe_name),
                        '--ros-args',
                        # The map topic is transient-local, but a freshly
                        # started map_saver still needs time for DDS discovery
                        # before it receives the retained map.
                        '-p',
                        'save_map_timeout:=15.0',
                        '-p',
                        'map_subscribe_transient_local:=true',
                    ),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeError(f'Αποτυχία εκκίνησης map saver: {exc}') from exc
            self.map_save_process = process
            self.last_map_save_name = safe_name
        return {'name': safe_name}

    def start_follow(self):
        with self.lock:
            if self.state.get('emergency_stop') is True or self.state.get('safety_stop') is True:
                raise RuntimeError('Η κίνηση είναι μπλοκαρισμένη από το safety ή emergency stop.')
            if self.follow_active:
                raise RuntimeError('Το follow-me είναι ήδη ενεργό.')
            if self.navigation_goal_handle is not None or self.patrol_active:
                raise RuntimeError('Υπάρχει ενεργή πλοήγηση· σταμάτησέ την πρώτα.')
            if self.drive_heading_goal_handle is not None or self.drive_heading_status in (
                'sending',
                'moving',
                'canceling',
            ):
                raise RuntimeError('Υπάρχει ενεργή κίνηση σε απόσταση· σταμάτησέ την πρώτα.')
            if self.spin_goal_handle is not None or self.spin_status in ('sending', 'spinning', 'canceling'):
                raise RuntimeError('Υπάρχει ενεργή στροφή· σταμάτησέ την πρώτα.')
            self.follow_active = True
            self.follow_started_at = time.monotonic()
            self.follow_until = self.follow_started_at + self.follow_timeout_s
            self.follow_target_x = None
            self.follow_missing_announced = False
            self.state['follow'] = {
                'active': True,
                'state': 'searching',
                'detail': 'Αναζητώ άνθρωπο στην κάμερα.',
                'last_seen': None,
                'target': None,
            }
        # Clear any preceding teleop command before the controller starts its
        # own 10 Hz decisions.
        self.drive(0.0, 0.0)
        return True

    def stop_follow(self, silent=False):
        with self.lock:
            active = self.follow_active
            self.follow_active = False
            self.follow_until = 0.0
            self.follow_target_x = None
            self.follow_missing_announced = False
            self.state['follow'] = {
                'active': False,
                'state': 'idle',
                'detail': 'Η λειτουργία follow είναι κλειστή.',
                'last_seen': None,
                'target': None,
            }
        self.drive(0.0, 0.0)
        if not silent:
            self.publish_voice_reply(
                'Το follow-me σταμάτησε.' if active else 'Το follow-me δεν ήταν ενεργό.',
                action='follow_stop',
            )
        return active

    def hold_follow_search(self, detail='Αναζητώ άνθρωπο στην κάμερα.'):
        """Keep follow active briefly while the detector reacquires the person.

        Searching never drives the robot.  Initial acquisition gets a short
        grace period after the explicit command; after a target was acquired,
        only a brief detector dropout is tolerated.  A prolonged loss returns
        False so the caller can stop the mode and announce the safety stop.
        """
        self.drive(0.0, 0.0)
        with self.lock:
            if not self.follow_active:
                return False
            target_locked = self.follow_target_x is not None
            started_at = self.follow_started_at
            follow_state = dict(self.state.get('follow') or {})
        last_seen = self.finite_value(follow_state.get('last_seen'))
        initial_search = (
            not target_locked
            and time.monotonic() - started_at < self.follow_acquire_timeout_s
        )
        recent_loss = (
            target_locked
            and last_seen is not None
            and time.time() - last_seen < self.follow_lost_timeout_s
        )
        if not (initial_search or recent_loss):
            return False
        with self.lock:
            self.state['follow'] = {
                'active': True,
                'state': 'searching',
                'detail': detail,
                'last_seen': last_seen,
                'target': None,
            }
        return True

    def follow_tick(self):
        with self.lock:
            active = self.follow_active
            until = self.follow_until
            vision = dict(self.state.get('vision') or {})
            blocked = (
                self.state.get('emergency_stop') is True
                or self.state.get('safety_stop') is True
            )
        if not active:
            return
        if blocked:
            self.stop_follow(silent=True)
            self.publish_voice_reply(
                'Το follow-me σταμάτησε επειδή ενεργοποιήθηκε safety ή emergency stop.',
                ok=False,
                action='follow_stop',
            )
            return
        if until and time.monotonic() >= until:
            self.stop_follow(silent=True)
            self.publish_voice_reply(
                'Το follow-me έληξε για ασφάλεια. Ξεκίνα το ξανά όταν είσαι έτοιμος.',
                ok=False,
                action='follow_stop',
            )
            return
        try:
            vision_age = time.time() - float(vision.get('last_update') or 0.0)
        except (TypeError, ValueError):
            vision_age = float('inf')
        objects = [
            item for item in (vision.get('objects') or [])
            if isinstance(item, dict) and item.get('label') == 'person'
        ]
        if vision_age > self.follow_lost_timeout_s or not objects:
            if self.hold_follow_search():
                return
            self.stop_follow(silent=True)
            self.publish_voice_reply(
                'Δεν σε βλέπω και σταμάτησα για ασφάλεια.',
                ok=False,
                action='follow_stop',
            )
            return

        def object_center(item):
            value = item.get('center_x')
            if value is None and item.get('x1') is not None and item.get('x2') is not None:
                value = (float(item['x1']) + float(item['x2'])) / 2.0
            return float(value) if value is not None else None

        width = next(
            (float(item.get('img_w')) for item in objects if item.get('img_w')),
            640.0,
        )
        with self.lock:
            target_x = self.follow_target_x if self.follow_target_x is not None else width / 2.0
        people = [(item, object_center(item)) for item in objects]
        people = [(item, center) for item, center in people if center is not None]
        if not people:
            if self.hold_follow_search('Δεν βλέπω καθαρά τη θέση σου· ψάχνω.'):
                return
            self.stop_follow(silent=True)
            self.publish_voice_reply('Δεν βλέπω καθαρά τη θέση σου και σταμάτησα.', ok=False, action='follow_stop')
            return
        person, center = min(people, key=lambda pair: abs(pair[1] - target_x))
        distance = self.finite_value(person.get('distance_m'))
        error = center - width / 2.0
        angular = max(-0.55, min(0.55, -error / max(width / 2.0, 1.0) * 0.55))
        linear = 0.0
        if distance is not None:
            if distance > 1.15 and abs(error) < width * 0.22:
                linear = min(0.12, max(0.0, (distance - 1.05) * 0.18))
            elif distance < 0.82:
                linear = 0.0
        self.drive(linear, angular)
        with self.lock:
            self.follow_target_x = center
            self.state['follow'] = {
                'active': True,
                'state': 'tracking',
                'detail': 'Ακολουθώ τον άνθρωπο με χαμηλή ταχύτητα.',
                'last_seen': time.time(),
                'target': {
                    'center_x': round(center, 1),
                    'image_width': round(width, 1),
                    'distance_m': round(distance, 2) if distance is not None else None,
                    'linear_m_s': round(linear, 3),
                    'angular_rad_s': round(angular, 3),
                },
            }

    def drive(self, linear, angular):
        if not rclpy.ok():
            return False
        try:
            linear, angular = float(linear), float(angular)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid drive command') from exc
        if not math.isfinite(linear) or not math.isfinite(angular):
            raise ValueError('invalid drive command')
        with self.lock:
            blocked = (
                self.state['emergency_stop'] is True
                or self.state['safety_stop'] is True
            )
            limits = dict(self.settings)
        if blocked:
            linear, angular = 0.0, 0.0
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = max(
            -limits['max_linear_speed'], min(limits['max_linear_speed'], linear)
        )
        msg.twist.angular.z = max(
            -limits['max_angular_speed'], min(limits['max_angular_speed'], angular)
        )
        try:
            self.pub.publish(msg)
        except Exception as exc:
            # During systemd shutdown the ROS context may close between the
            # guard above and publish; the child process cleanup must continue.
            self.get_logger().debug(f'Ignoring drive publish during shutdown: {exc}')
            return False
        with self.lock:
            self.last_drive_at = (
                time.monotonic() if (msg.twist.linear.x or msg.twist.angular.z) else 0.0
            )
        return True

    def drive_watchdog(self):
        with self.lock:
            expired = bool(
                self.last_drive_at
                and time.monotonic() - self.last_drive_at > self.drive_timeout_s
            )
            if expired:
                self.last_drive_at = 0.0
        if expired:
            self.drive(0.0, 0.0)

    def request_system_poweroff(self):
        """Request a host shutdown after stopping all robot motion.

        The command is deliberately fixed and never accepts shell text or a
        path from the browser.  The Dashboard user is allowed to run this
        exact systemctl command through the host's NOPASSWD sudo policy.  A
        short delay lets the HTTP 202 response reach the browser before the
        system begins stopping the Dashboard service.
        """
        with self.lock:
            if self.poweroff_requested:
                return False
            self.poweroff_requested = True
        self.get_logger().warning('Mini PC poweroff requested from Dashboard')
        # Leave the base stopped even if an autonomous action was active.
        self.stop_follow(silent=True)
        self.cancel_drive_heading()
        self.cancel_spin()
        self.cancel_navigation()
        self.drive(0.0, 0.0)

        def execute_poweroff():
            time.sleep(0.45)
            try:
                result = subprocess.run(
                    [
                        '/usr/bin/sudo',
                        '-n',
                        '/usr/bin/systemctl',
                        '--no-block',
                        'poweroff',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                    env=self.systemctl_environment(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.get_logger().error(f'Mini PC poweroff command failed: {exc}')
                return
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                self.get_logger().error(
                    'Mini PC poweroff command failed: '
                    + (detail[0][:180] if detail else f'exit {result.returncode}')
                )

        threading.Thread(
            target=execute_poweroff,
            name='dingo-poweroff',
            daemon=True,
        ).start()
        return True

    def shutdown_processes(self):
        self.stop_follow(silent=True)
        self.stop_process('mapping_process')
        self.stop_navigation()
        self.stop_process('camera_process')
        self.stop_rviz()
        self.stop_process('map_save_process')
        # Keep the physical Clearpath default if the Dashboard is stopped by
        # systemd, Ctrl-C, or a ROS shutdown before a Nav2 result arrives.
        self.set_autonomous_navigation_mode(False, force=True)


def serve(node, port):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def send(self, code, value, content_type='application/json', headers=None):
            body = value if isinstance(value, bytes) else value.encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            for key, header_value in (headers or {}).items():
                self.send_header(key, header_value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        @staticmethod
        def json(value):
            return json.dumps(value, ensure_ascii=False)

        def read_json(self):
            length = min(int(self.headers.get('Content-Length', '0')), 65536)
            return json.loads(self.rfile.read(length) or '{}')

        def serve_rviz_asset(self, path):
            root = Path('/usr/share/novnc').resolve()
            relative = (
                unquote(path[len('/rviz/'):]).lstrip('/')
                if path.startswith('/rviz/')
                else 'vnc.html'
            )
            candidate = (root / relative).resolve()
            if root != candidate and root not in candidate.parents:
                return self.send(403, '{"error":"forbidden"}')
            if candidate.is_dir():
                candidate = candidate / 'index.html'
            if not candidate.is_file():
                return self.send(404, '{}')
            content_type = (
                mimetypes.guess_type(str(candidate))[0]
                or 'application/octet-stream'
            )
            return self.send(
                200,
                candidate.read_bytes(),
                content_type,
                {'Cache-Control': 'no-cache'},
            )

        def serve_microphone_websocket(self):
            """Stream ch0 PCM to a browser that explicitly opted in."""
            if self.headers.get('Upgrade', '').lower() != 'websocket':
                return self.send(400, '{"error":"websocket upgrade required"}')
            key = self.headers.get('Sec-WebSocket-Key', '').strip()
            if not key:
                return self.send(400, '{"error":"missing websocket key"}')
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()
                ).digest()
            ).decode()
            self.send_response(101, 'Switching Protocols')
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', accept)
            self.end_headers()
            client = node.register_audio_monitor()
            try:
                while True:
                    try:
                        payload = client.get(timeout=10.0)
                    except queue.Empty:
                        continue
                    size = len(payload)
                    if size < 126:
                        header = bytes((0x82, size))
                    elif size < 65536:
                        header = bytes((0x82, 126)) + struct.pack('!H', size)
                    else:
                        header = bytes((0x82, 127)) + struct.pack('!Q', size)
                    self.connection.sendall(header + payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                node.unregister_audio_monitor(client)

        @staticmethod
        def recv_websocket_bytes(connection, count):
            chunks = []
            remaining = count
            while remaining:
                chunk = connection.recv(remaining)
                if not chunk:
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            return b''.join(chunks)

        def serve_microphone_upload_websocket(self):
            """Receive mono PCM captured by a phone and publish it to ROS."""
            if self.headers.get('Upgrade', '').lower() != 'websocket':
                return self.send(400, '{"error":"websocket upgrade required"}')
            key = self.headers.get('Sec-WebSocket-Key', '').strip()
            if not key:
                return self.send(400, '{"error":"missing websocket key"}')
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()
                ).digest()
            ).decode()
            self.send_response(101, 'Switching Protocols')
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', accept)
            self.end_headers()
            self.connection.settimeout(2.0)
            try:
                while True:
                    header = self.recv_websocket_bytes(self.connection, 2)
                    if not header:
                        break
                    first, second = header
                    opcode = first & 0x0F
                    length = second & 0x7F
                    masked = bool(second & 0x80)
                    if length == 126:
                        raw_length = self.recv_websocket_bytes(self.connection, 2)
                        if not raw_length:
                            break
                        length = struct.unpack('!H', raw_length)[0]
                    elif length == 127:
                        raw_length = self.recv_websocket_bytes(self.connection, 8)
                        if not raw_length:
                            break
                        length = struct.unpack('!Q', raw_length)[0]
                    # Phone microphone blocks are normally 2–8 KB. Refuse a
                    # giant frame before allocating memory for an untrusted
                    # browser connection.
                    if length > 65536:
                        break
                    mask = self.recv_websocket_bytes(self.connection, 4) if masked else b''
                    payload = self.recv_websocket_bytes(self.connection, length) or b''
                    if masked:
                        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
                    if opcode == 0x8:  # close
                        break
                    if opcode == 0x9:  # ping
                        pong = bytes((0x8A, len(payload))) + payload
                        self.connection.sendall(pong)
                        continue
                    if opcode != 0x2 or not payload:
                        continue
                    node.publish_remote_audio(payload, 16000)
            except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/api/state':
                return self.send(200, self.json(node.snapshot()))
            if path == '/api/microphone-ws':
                return self.serve_microphone_websocket()
            if path == '/api/microphone-upload':
                return self.serve_microphone_upload_websocket()
            if path == '/api/vision':
                state = node.snapshot().get('vision') or {}
                return self.send(200, self.json(state))
            if path == '/api/face':
                return self.send(200, self.json(node.snapshot().get('face') or {}))
            if path == '/api/speaker':
                return self.send(200, self.json(node.snapshot().get('speaker') or {}))
            if path == '/api/follow':
                return self.send(200, self.json(node.snapshot().get('follow') or {}))
            if path == '/api/gemini':
                return self.send(200, self.json(node.snapshot().get('gemini') or {}))
            if path == '/api/tool-gateway':
                return self.send(200, self.json(node.snapshot().get('tool_gateway') or {}))
            if path == '/api/wake-training':
                return self.send(200, self.json(node.wake_training_snapshot()))
            if path == '/rviz-ws':
                if self.headers.get('Upgrade', '').lower() != 'websocket':
                    return self.send(400, '{"error":"websocket upgrade required"}')
                node.proxy_rviz_websocket(self.connection, self.headers)
                return
            if path == '/rviz' or path.startswith('/rviz/'):
                return self.serve_rviz_asset(path)
            if path == '/api/map':
                value = node.map_snapshot()
                return self.send(200, self.json(value) if value else '{}')
            if path == '/api/rooms':
                return self.send(200, self.json(node.rooms_snapshot()))
            if path == '/api/maps':
                return self.send(200, self.json(node.maps_snapshot()))
            if path == '/api/settings':
                return self.send(200, self.json(node.settings_snapshot()))
            if path == '/api/mapping':
                return self.send(200, self.json(node.mapping_snapshot()))
            if path == '/api/navigation':
                return self.send(200, self.json(node.navigation_snapshot()))
            if path == '/api/clearpath':
                return self.send(200, self.json(node.clearpath_api_snapshot()))
            if path == '/api/clearpath/topics':
                value = node.clearpath_api_snapshot()
                return self.send(
                    200,
                    self.json(
                        {
                            'schema': value['schema'],
                            'robot': value['robot'],
                            'topics': value['topics'],
                            'runtime_topics': value['runtime']['topics'],
                        }
                    ),
                )
            if path == '/api/clearpath/services':
                value = node.clearpath_api_snapshot()
                return self.send(
                    200,
                    self.json(
                        {
                            'schema': value['schema'],
                            'robot': value['robot'],
                            'services': value['services'],
                            'runtime_services': value['runtime']['services'],
                        }
                    ),
                )
            if path == '/api/clearpath/actions':
                value = node.clearpath_api_snapshot()
                return self.send(
                    200,
                    self.json(
                        {
                            'schema': value['schema'],
                            'robot': value['robot'],
                            'actions': value['actions'],
                        }
                    ),
                )
            if path == '/api/clearpath/status':
                value = node.clearpath_api_snapshot()
                return self.send(
                    200,
                    self.json(
                        {
                            'schema': value['schema'],
                            'robot': value['robot'],
                            'live': value['live'],
                        }
                    ),
                )
            if path == '/api/camera/status':
                return self.send(200, self.json(node.camera_status()))
            if path == '/api/rviz':
                return self.send(200, self.json(node.rviz_status()))
            if path == '/api/camera.jpg':
                image, mime = node.camera_snapshot()
                if not image:
                    return self.send(404, '{"error":"camera unavailable"}')
                return self.send(
                    200,
                    image,
                    mime,
                    {'Cache-Control': 'no-store, max-age=0'},
                )
            if path == '/' or path == '/index.html':
                return self.send(
                    200,
                    HTML,
                    'text/html; charset=utf-8',
                    {'Cache-Control': 'no-store, no-cache, must-revalidate'},
                )
            return self.send(404, '{}')

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                data = self.read_json()
                if path == '/api/drive':
                    if abs(float(data.get('linear', 0) or 0)) > 1e-6 or abs(float(data.get('angular', 0) or 0)) > 1e-6:
                        node.stop_follow(silent=True)
                    node.drive(data.get('linear', 0), data.get('angular', 0))
                    return self.send(204, b'')
                if path == '/api/voice/direction':
                    enabled = node.set_voice_direction_enabled(
                        bool(data.get('enabled', False))
                    )
                    return self.send(
                        200,
                        self.json({'ok': True, 'enabled': enabled}),
                    )
                if path == '/api/voice/provider':
                    result = node.set_voice_provider(data.get('provider'))
                    return self.send(
                        200,
                        self.json({'ok': True, **result}),
                    )
                if path == '/api/voice/native-tools':
                    result = node.set_native_tool_calling(
                        bool(data.get('enabled', False))
                    )
                    return self.send(
                        200,
                        self.json({'ok': True, **result}),
                    )
                if path == '/api/system':
                    if data.get('action') != 'poweroff':
                        return self.send(400, '{"error":"invalid system action"}')
                    if data.get('confirmed') is not True:
                        return self.send(
                            400,
                            '{"error":"Ο τερματισμός απαιτεί επιβεβαίωση"}',
                        )
                    if not node.request_system_poweroff():
                        return self.send(
                            409,
                            '{"error":"Ο τερματισμός έχει ήδη ζητηθεί"}',
                        )
                    return self.send(
                        202,
                        self.json(
                            {
                                'ok': True,
                                'action': 'poweroff',
                                'message': 'Το Mini PC θα τερματιστεί τώρα.',
                            }
                        ),
                    )
                if path == '/api/assistant':
                    result = node.submit_text_command(data.get('text'))
                    return self.send(202, self.json(result))
                if path == '/api/wake-training':
                    action = str(data.get('action', 'status')).strip().lower()
                    if action == 'start':
                        result = node.start_wake_training(data.get('phase'))
                    elif action == 'cancel':
                        result = {
                            'ok': True,
                            'changed': node.cancel_wake_training(),
                            **node.wake_training_snapshot(),
                        }
                    elif action == 'train':
                        result = node.start_wake_training_pipeline()
                    elif action == 'status':
                        result = node.wake_training_snapshot()
                    else:
                        return self.send(400, '{"error":"invalid wake training action"}')
                    return self.send(202 if action in {'start', 'train'} else 200, self.json(result))
                if path == '/api/vision/ask':
                    result = node.start_vision_question(data.get('question', data.get('text', '')))
                    return self.send(202, self.json(result))
                if path == '/api/identity/face':
                    result = node.submit_face_command(data)
                    return self.send(202, self.json(result))
                if path == '/api/identity/speaker':
                    result = node.submit_speaker_command(data)
                    return self.send(202, self.json(result))
                if path == '/api/follow':
                    action = str(data.get('action', '')).strip().lower()
                    if action == 'start':
                        if data.get('confirmed') is not True:
                            return self.send(400, '{"error":"Το follow-me απαιτεί επιβεβαίωση"}')
                        changed = node.start_follow()
                    elif action == 'stop':
                        changed = node.stop_follow()
                    else:
                        return self.send(400, '{"error":"invalid follow action"}')
                    return self.send(200, self.json({'ok': True, 'changed': changed}))
                if path == '/api/settings':
                    return self.send(200, self.json(node.update_settings(data)))
                if path == '/api/mapping':
                    action = data.get('action')
                    if action == 'start':
                        started = node.start_mapping()
                    elif action == 'stop':
                        started = node.stop_mapping()
                    else:
                        return self.send(400, '{"error":"invalid mapping action"}')
                    return self.send(200, self.json({'ok': True, 'changed': started}))
                if path == '/api/navigation':
                    action = data.get('action')
                    if action == 'start':
                        changed = node.start_navigation(data.get('map'))
                    elif action == 'stop':
                        changed = node.stop_navigation()
                    elif action == 'goal':
                        changed = node.send_navigation_goal(
                            data.get('x'), data.get('y')
                        )
                    elif action == 'initial_pose':
                        changed = node.send_initial_pose(
                            data.get('x'), data.get('y'), data.get('yaw', 0.0)
                        )
                    elif action == 'global_localization':
                        changed = node.start_global_localization(
                            allow_motion=bool(data.get('allow_motion', False))
                        )
                    elif action == 'cancel':
                        changed = node.cancel_navigation()
                    else:
                        return self.send(400, '{"error":"invalid navigation action"}')
                    return self.send(200, self.json({'ok': True, 'changed': changed}))
                if path == '/api/map/save':
                    result = node.save_map(data.get('name'))
                    return self.send(202, self.json(result))
                if path == '/api/camera/control':
                    action = data.get('action')
                    if action == 'start':
                        changed = node.start_camera()
                    elif action == 'stop':
                        changed = node.stop_camera()
                    else:
                        return self.send(400, '{"error":"invalid camera action"}')
                    return self.send(200, self.json({'ok': True, 'changed': changed}))
                if path == '/api/rviz':
                    action = data.get('action')
                    if action == 'start':
                        changed = node.start_rviz()
                    elif action == 'stop':
                        changed = node.stop_rviz()
                    elif action == 'restart':
                        node.stop_rviz()
                        changed = node.start_rviz()
                    else:
                        return self.send(400, '{"error":"invalid RViz action"}')
                    return self.send(
                        200,
                        self.json(
                            {
                                'ok': True,
                                'changed': changed,
                                'status': node.rviz_status(),
                            }
                        ),
                    )
                if path == '/api/rooms':
                    if data.get('action', 'add') == 'delete':
                        changed = node.remove_room(data.get('id'))
                        return self.send(200, self.json({'ok': True, 'changed': changed}))
                    room = node.add_room(data.get('name'), data.get('x'), data.get('y'))
                    return self.send(201, self.json(room))
                return self.send(404, '{}')
            except (ValueError, TypeError, json.JSONDecodeError, OverflowError) as exc:
                return self.send(400, self.json({'error': str(exc)}))
            except RuntimeError as exc:
                return self.send(409, self.json({'error': str(exc)}))

    ThreadingHTTPServer((node.host, port), Handler).serve_forever()


def main():
    try:
        acquire_dashboard_lock()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    rclpy.init()
    node = DashboardNode()
    port = node.declare_parameter('port', 8090).value
    threading.Thread(target=serve, args=(node, int(port)), daemon=True).start()
    node.get_logger().info(f'Dingo dashboard: http://{node.host}:{port}')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown_processes()
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
