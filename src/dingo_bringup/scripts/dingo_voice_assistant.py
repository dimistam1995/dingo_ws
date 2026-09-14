#!/usr/bin/env python3
"""Safe voice interface for the Clearpath Dingo.

The node deliberately has a narrow responsibility:

* capture audio from the named ReSpeaker USB device;
* publish the raw PCM blocks and a transcript topic;
* ask the configured local or cloud LLM for a native function/tool call;
* publish only allow-listed commands for the Dashboard to execute.

It never executes shell commands and never publishes velocity directly.  The
Dashboard remains the single motion authority; navigation still passes
through its localization, Nav2 and safety checks.
"""

import base64
import asyncio
import fcntl
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import numpy as np
except ImportError:  # The node can still report a missing audio dependency.
    np = None

try:
    import sounddevice as sd
except ImportError:  # The node can still wait for an installation fix.
    sd = None

try:
    from scipy.signal import butter, lfilter, lfilter_zi
except ImportError:  # The wake detector can still report a missing filter dependency.
    butter = lfilter = lfilter_zi = None

import rclpy
from foxglove_msgs.msg import RawAudio
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


_VOICE_LOCK_FD = None


def acquire_voice_lock():
    """Allow one voice assistant process on the computer."""
    global _VOICE_LOCK_FD
    if _VOICE_LOCK_FD is not None:
        return

    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if not runtime_dir:
        runtime_dir = f'/run/user/{os.getuid()}'
    lock_dir = Path(runtime_dir)
    if not lock_dir.is_dir():
        lock_dir = Path('/tmp')
    lock_override = os.environ.get('DINGO_VOICE_LOCK_FILE', '').strip()
    lock_path = Path(lock_override).expanduser() if lock_override else (
        lock_dir / 'dingo-voice-assistant.lock'
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError(
            'Το Dingo voice assistant τρέχει ήδη σε άλλο process.'
        ) from exc
    _VOICE_LOCK_FD = lock_fd


def normalize_text(value):
    """Lowercase text and remove Greek accent marks for matching."""
    text = unicodedata.normalize('NFD', str(value or '').lower())
    return ''.join(char for char in text if not unicodedata.combining(char))


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class VoiceAssistantNode(Node):
    ALLOWED_INTENTS = {
        'stop',
        'status',
        'battery',
        'where',
        'navigate_room',
        'move_distance',
        'system_info',
        'system_control',
        'patrol',
        'rotate',
        'vision',
        'vision_question',
        'face_query',
        'speaker_query',
        'follow_start',
        'follow_stop',
        'time',
        'answer',
        'unknown',
    }
    INTENT_ALIASES = {
        'go_to_room': 'navigate_room',
        'goto_room': 'navigate_room',
        'go': 'navigate_room',
        'navigate': 'navigate_room',
        'move': 'move_distance',
        'drive': 'move_distance',
        'move_forward': 'move_distance',
        'move_backward': 'move_distance',
        'drive_on_heading': 'move_distance',
        'cancel': 'stop',
        'emergency_stop': 'stop',
        'location': 'where',
        'system': 'system_info',
        'system_status': 'system_info',
        'control_system': 'system_control',
        'manage_system': 'system_control',
        'tour': 'patrol',
        'spin': 'rotate',
        'turn': 'rotate',
        'see': 'vision',
        'look': 'vision',
        'objects': 'vision',
        'look_closer': 'vision_question',
        'describe': 'vision_question',
        'who_spoke': 'speaker_query',
        'who_is_there': 'face_query',
        'follow': 'follow_start',
    }
    # These are the only robot/system capabilities exposed to the LLM.  The
    # model can request one of these functions, but it never receives shell,
    # ROS, filesystem or arbitrary network access.  The resulting intent is
    # still passed through validate_intent() and the existing safety checks
    # before the Dashboard can execute anything.
    NATIVE_TOOL_INTENTS = {
        'stop_robot': 'stop',
        'get_status': 'status',
        'get_battery': 'battery',
        'get_location': 'where',
        'navigate_to_room': 'navigate_room',
        'drive_distance': 'move_distance',
        'drive_on_heading': 'move_distance',
        'get_system_info': 'system_info',
        'control_system': 'system_control',
        'start_patrol': 'patrol',
        'rotate_in_place': 'rotate',
        'describe_scene': 'vision',
        'ask_about_camera': 'vision_question',
        'recognize_face': 'face_query',
        'identify_speaker': 'speaker_query',
        'start_follow_me': 'follow_start',
        'stop_follow_me': 'follow_stop',
        'get_time': 'time',
    }
    POSITIVE_WORDS = {
        'ναι', 'ναι προχωρα', 'προχωρα', 'προχώρα', 'yes', 'ok', 'okay',
        'ενταξει', 'εντάξει', 'βεβαια', 'βέβαια',
    }
    NEGATIVE_WORDS = {
        'οχι', 'όχι', 'ακυρο', 'άκυρο', 'cancel', 'no', 'σταματα', 'σταμάτα',
    }

    def __init__(self):
        super().__init__('dingo_voice_assistant')
        self.robot_namespace = str(
            self.declare_parameter('robot_namespace', 'dd100_10000002').value
        ).strip('/')
        prefix = f'/{self.robot_namespace}' if self.robot_namespace else ''

        # ReSpeaker is intentionally required by name.  With the default
        # value, an absent array never falls back to the Mini PC's microphone.
        self.microphone_name = str(
            self.declare_parameter('microphone_name', 'ReSpeaker').value or ''
        ).strip()
        self.microphone_backend = str(
            self.declare_parameter('microphone_backend', 'auto').value or 'auto'
        ).strip().lower()
        if self.microphone_backend not in {'auto', 'alsa', 'pipewire'}:
            self.get_logger().warning(
                f'Άγνωστο microphone_backend «{self.microphone_backend}». '
                'Χρησιμοποιώ auto.'
            )
            self.microphone_backend = 'auto'
        self.microphone_channels = int(
            self.declare_parameter('microphone_channels', 2).value or 2
        )
        # The current direct-ALSA XVF3800 endpoint carries speech on channels
        # 0/1, while channels 2–5 are effectively silent. Use the calibrated
        # channel 0 for VAD, Whisper and the fast wake detector alike; using a
        # silent auxiliary channel makes the detector miss every «Alexa».
        # Mono browser audio is automatically mapped to its only channel by
        # block_to_mono().
        self.microphone_auto_fallback = bool(
            self.declare_parameter('microphone_auto_fallback', True).value
        )
        try:
            self.microphone_gain = max(
                0.5,
                min(
                    4.0,
                    float(self.declare_parameter('microphone_gain', 2.0).value),
                ),
            )
        except (TypeError, ValueError):
            self.microphone_gain = 2.0
        microphone_channel = self.declare_parameter(
            'microphone_channel', 0
        ).value
        self.microphone_channel = int(
            0 if microphone_channel is None else microphone_channel
        )
        stt_channel = self.declare_parameter('stt_channel', 0).value
        self.stt_channel = int(
            0 if stt_channel is None else stt_channel
        )
        try:
            wake_channel = int(
                self.declare_parameter('wake_channel', 0).value
            )
        except (TypeError, ValueError):
            wake_channel = -1
        self.wake_channel = (
            self.microphone_channel if wake_channel < 0 else wake_channel
        )
        # The lightweight openWakeWord model is the first and normally only
        # wake gate.  A Whisper fallback can be enabled for diagnostics, but
        # it is disabled by default: seeded/short Whisper decodes may turn
        # room noise or a speaker echo into a false «Alexa» and make the
        # robot answer without an explicit wake word.
        # openWakeWord's official Alexa detector is trained on English TTS
        # "Alexa".  A Greek pronunciation («αλεχα»/«αλεξα») can miss its
        # threshold, so the fallback remains an opt-in recovery path after
        # the user has verified the dedicated detector.
        self.wake_fallback_stt = bool(
            self.declare_parameter('wake_fallback_stt', False).value
        )
        self.wake_fallback_model_name = str(
            self.declare_parameter('wake_fallback_model', 'small').value
            or 'small'
        ).strip()
        # A noisy room can produce several VAD segments while one GPU wake
        # check is still running.  Keep a short guard between fallback checks;
        # openWakeWord remains always-on and is still the fast path.
        try:
            self.wake_fallback_cooldown_s = max(
                0.5,
                min(
                    10.0,
                    float(
                        self.declare_parameter(
                            'wake_fallback_cooldown_s', 2.5
                        ).value
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.wake_fallback_cooldown_s = 2.5
        self.target_sample_rate = 16000
        try:
            self.wake_highpass_hz = max(
                0.0,
                min(
                    1000.0,
                    float(
                        # Optional rumble filter used only by the wake path.
                        self.declare_parameter('wake_highpass_hz', 250.0).value
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.wake_highpass_hz = 250.0
        self._wake_hp_b = None
        self._wake_hp_a = None
        self._wake_hp_zi_template = None
        self._wake_hp_zi = None
        self.configure_wake_highpass()
        self.capture_rate_preference = int(
            self.declare_parameter('capture_rate', 0).value or 0
        )
        self.publish_raw_audio = bool(
            self.declare_parameter('publish_raw_audio', True).value
        )
        self.tts_enabled = bool(
            self.declare_parameter('tts_enabled', True).value
        )
        # Keep the selected online TTS as the only speech path.  The former
        # local Piper voice was a backup voice and is intentionally removed;
        # a TTS outage is reported instead of switching to another voice.
        self.tts_backend = str(
            self.declare_parameter(
                'tts_backend',
                os.environ.get('DINGO_TTS_BACKEND', 'edge'),
            ).value or 'edge'
        ).strip().lower()
        if self.tts_backend not in {'edge', 'gemini'}:
            self.get_logger().warning(
                f'Το TTS backend «{self.tts_backend}» δεν είναι διαθέσιμο. '
                'Το Piper έχει αφαιρεθεί· χρησιμοποιώ Edge TTS.'
            )
            self.tts_backend = 'edge'
        self.tts_cloud_model = str(
            self.declare_parameter(
                'tts_cloud_model',
                os.environ.get(
                    'DINGO_TTS_CLOUD_MODEL',
                    'gemini-3.1-flash-tts-preview',
                ),
            ).value or 'gemini-3.1-flash-tts-preview'
        ).strip()
        self.tts_cloud_fallback_model = str(
            self.declare_parameter(
                'tts_cloud_fallback_model',
                os.environ.get(
                    'DINGO_TTS_CLOUD_FALLBACK_MODEL',
                    'gemini-2.5-flash-preview-tts',
                ),
            ).value or 'gemini-2.5-flash-preview-tts'
        ).strip()
        self.tts_voice_name = str(
            self.declare_parameter(
                'tts_voice',
                os.environ.get('DINGO_TTS_VOICE', 'Aoede'),
            ).value or 'Iapetus'
        ).strip()
        self.tts_edge_voice = str(
            self.declare_parameter(
                'tts_edge_voice',
                os.environ.get('DINGO_TTS_EDGE_VOICE', 'el-GR-AthinaNeural'),
            ).value or 'el-GR-AthinaNeural'
        ).strip()
        try:
            self.tts_timeout_s = max(
                5.0,
                min(
                    60.0,
                    float(self.declare_parameter('tts_timeout_s', 30.0).value),
                ),
            )
        except (TypeError, ValueError):
            self.tts_timeout_s = 30.0
        self.tts_output_device = str(
            self.declare_parameter('tts_output_device', '').value or ''
        ).strip()
        try:
            self.tts_volume = max(
                0.0,
                min(1.0, float(self.declare_parameter('tts_volume', 1.0).value)),
            )
        except (TypeError, ValueError):
            self.tts_volume = 1.0

        self.stt_model_name = str(
            self.declare_parameter('stt_model', 'large-v3-turbo').value or 'large-v3-turbo'
        ).strip()
        self.stt_backend = str(
            self.declare_parameter('stt_backend', 'cpu').value or 'cpu'
        ).strip().lower()
        if self.stt_backend not in {'cpu', 'npu', 'vulkan'}:
            self.get_logger().warning(
                f'Άγνωστο stt_backend «{self.stt_backend}». Χρησιμοποιώ CPU.'
            )
            self.stt_backend = 'cpu'
        self.stt_fallback_model_name = str(
            self.declare_parameter(
                'stt_fallback_model',
                os.environ.get('DINGO_STT_FALLBACK_MODEL', 'large-v3-turbo'),
            ).value or 'large-v3-turbo'
        ).strip()
        default_vulkan_cli = os.environ.get(
            'DINGO_VULKAN_WHISPER_CLI',
            str(
                Path.home()
                / '.local'
                / 'share'
                / 'dingo-whisper-vulkan'
                / 'bin'
                / 'whisper-cli'
            ),
        )
        self.vulkan_whisper_cli = Path(
            self.declare_parameter('vulkan_whisper_cli', default_vulkan_cli).value
        ).expanduser()
        default_vulkan_model = os.environ.get(
            'DINGO_VULKAN_WHISPER_MODEL',
            str(
                Path.home()
                / '.cache'
                / 'dingo'
                / 'whisper-vulkan'
                / 'ggml-large-v3-turbo.bin'
            ),
        )
        self.vulkan_model_path = Path(
            self.declare_parameter('vulkan_model_path', default_vulkan_model).value
        ).expanduser()
        try:
            self.vulkan_threads = max(
                1,
                min(
                    32,
                    int(self.declare_parameter('vulkan_threads', 8).value or 8),
                ),
            )
        except (TypeError, ValueError):
            self.vulkan_threads = 8
        try:
            self.vulkan_beam_size = max(
                1,
                min(
                    10,
                    int(self.declare_parameter('vulkan_beam_size', 5).value or 5),
                ),
            )
        except (TypeError, ValueError):
            self.vulkan_beam_size = 5
        try:
            self.vulkan_timeout_s = max(
                10.0,
                min(
                    300.0,
                    float(self.declare_parameter('vulkan_timeout_s', 60.0).value),
                ),
            )
        except (TypeError, ValueError):
            self.vulkan_timeout_s = 60.0
        self.npu_python = str(
            self.declare_parameter(
                'npu_python',
                os.environ.get('DINGO_NPU_PYTHON', '/home/dimi/ryzenai_venv/bin/python'),
            ).value or '/home/dimi/ryzenai_venv/bin/python'
        ).strip()
        self.npu_model_dir = str(
            self.declare_parameter(
                'npu_model_dir',
                os.environ.get(
                    'DINGO_NPU_MODEL_DIR',
                    str(Path.home() / '.cache' / 'home_robot' / 'whisper-npu' / 'medium'),
                ),
            ).value
        ).strip()
        self.npu_tokenizer_dir = str(
            self.declare_parameter(
                'npu_tokenizer_dir',
                os.environ.get(
                    'DINGO_NPU_TOKENIZER_DIR',
                    str(Path.home() / '.cache' / 'home_robot' / 'whisper-tokenizer' / 'medium'),
                ),
            ).value
        ).strip()
        self.npu_config_dir = str(
            self.declare_parameter(
                'npu_config_dir',
                os.environ.get(
                    'DINGO_NPU_CONFIG_DIR',
                    '/home/dimi/RyzenAI-SW/Demos/ASR/Whisper/config',
                ),
            ).value
        ).strip()
        self.npu_cache_dir = str(
            self.declare_parameter(
                'npu_cache_dir',
                os.environ.get(
                    'DINGO_NPU_CACHE_DIR',
                    str(Path.home() / '.cache' / 'home_robot' / 'whisper-npu' / 'cache'),
                ),
            ).value
        ).strip()
        self.stt_language = str(
            self.declare_parameter('stt_language', 'el').value or 'el'
        ).strip()
        self.llm_provider = str(
            self.declare_parameter('llm_provider', 'flm').value or 'flm'
        ).strip().lower()
        if self.llm_provider not in {'flm', 'gemini', 'ollama'}:
            self.get_logger().warning(
                f'Άγνωστος llm_provider «{self.llm_provider}». Χρησιμοποιώ τοπικό FLM.'
            )
            self.llm_provider = 'flm'
        default_llm_url = (
            'http://127.0.0.1:52625/v1/chat/completions'
            if self.llm_provider == 'flm'
            else 'http://127.0.0.1:11434/api/chat'
        )
        self.llm_url = str(
            self.declare_parameter(
                'llm_url', os.environ.get('DINGO_LLM_URL', default_llm_url)
            ).value
        ).rstrip('/')
        self.llm_model = str(
            self.declare_parameter(
                'llm_model', os.environ.get('DINGO_LLM_MODEL', 'qwen3.5:9b')
            ).value or 'qwen3.5:9b'
        )
        self.native_tool_calling = bool(
            self.declare_parameter('native_tool_calling', False).value
        )
        # Keep separate local/Gemini settings so the Dashboard can switch the
        # assistant brain at runtime without restarting the audio pipeline.
        local_url_default = os.environ.get(
            'DINGO_LOCAL_LLM_URL',
            self.llm_url
            if self.llm_provider != 'gemini'
            else 'http://127.0.0.1:52625/v1/chat/completions',
        )
        local_model_default = os.environ.get(
            'DINGO_LOCAL_LLM_MODEL',
            self.llm_model if self.llm_provider != 'gemini' else 'qwen3.5:9b',
        )
        self.local_llm_url = str(
            self.declare_parameter('local_llm_url', local_url_default).value
            or local_url_default
        ).rstrip('/')
        self.local_llm_model = str(
            self.declare_parameter('local_llm_model', local_model_default).value
            or local_model_default
        )
        self.gemini_llm_model = str(
            self.declare_parameter(
                'gemini_llm_model',
                os.environ.get('DINGO_GEMINI_LLM_MODEL', 'gemini-2.5-flash'),
            ).value
            or 'gemini-2.5-flash'
        )
        self.llm_config_lock = threading.Lock()
        self.llm_fallback_active = False
        self.llm_fallback_reason = ''
        if self.llm_provider == 'gemini':
            self.llm_model = self.gemini_llm_model
        default_key_file = os.environ.get(
            'GEMINI_API_KEY_FILE',
            str(Path.home() / '.config' / 'dingo_voice' / 'gemini_api_key'),
        )
        self.gemini_api_key_file = Path(
            self.declare_parameter('gemini_api_key_file', default_key_file).value
        ).expanduser()
        self.llm_timeout_s = float(
            self.declare_parameter('llm_timeout_s', 35.0).value or 35.0
        )
        self.rooms_file = Path(
            self.declare_parameter(
                'rooms_file',
                str(Path.home() / '.config' / 'dingo_dashboard' / 'rooms.json'),
            ).value
        ).expanduser()
        self.wake_model_name = str(
            self.declare_parameter('wake_model_name', 'Alexa').value or 'Alexa'
        ).strip() or 'Alexa'
        wake_words = str(
            self.declare_parameter(
                # Whisper often writes the English wake word with Greek
                # letters when the command is spoken in Greek. Keep the
                # dedicated Alexa detector, but accept those STT spellings as
                # equivalent wake words in the fallback path.
                'wake_words', 'alexa,αλεξα,αλεχα'
            ).value
        )
        self.wake_words = tuple(
            word for word in (normalize_text(item).strip() for item in wake_words.split(','))
            if word
        )
        self.wake_enabled = bool(
            self.declare_parameter('wake_enabled', True).value
        )
        self.wake_threshold = max(
            0.05,
            min(
                0.99,
                float(self.declare_parameter('wake_threshold', 0.50).value or 0.50),
            ),
        )
        try:
            self.wake_model_vad_threshold = max(
                0.0,
                min(
                    0.99,
                    float(
                        self.declare_parameter(
                            'wake_model_vad_threshold', 0.0
                        ).value
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.wake_model_vad_threshold = 0.0
        self.wake_confirmation_hits = max(
            1,
            min(
                6,
                int(self.declare_parameter('wake_confirmation_hits', 3).value or 3),
            ),
        )
        self.wake_confirmation_window_s = max(
            0.2,
            min(
                3.0,
                float(
                    self.declare_parameter(
                        'wake_confirmation_window_s', 0.8
                    ).value
                    or 0.8
                ),
            ),
        )
        self.wake_cooldown_s = max(
            0.5,
            min(
                10.0,
                float(self.declare_parameter('wake_cooldown_s', 1.5).value or 1.5),
            ),
        )
        self.wake_listen_window_s = max(
            2.0,
            min(
                20.0,
                float(
                    self.declare_parameter('wake_listen_window_s', 8.0).value
                    or 8.0
                ),
            ),
        )
        default_wake_model = str(
            Path.home() / 'dingo_ws' / 'models' / 'wake_words' / 'alexa.onnx'
        )
        self.wake_model_path = Path(
            self.declare_parameter('wake_model_path', default_wake_model).value
        ).expanduser()
        self.confirmation_timeout_s = float(
            self.declare_parameter('confirmation_timeout_s', 20.0).value or 20.0
        )
        self.speaker_enabled = bool(
            self.declare_parameter('speaker_enabled', True).value
        )
        self.speaker_threshold = float(
            self.declare_parameter('speaker_threshold', 0.72).value or 0.72
        )
        self.speaker_enroll_count = max(
            3, min(8, int(self.declare_parameter('speaker_enroll_count', 3).value or 3))
        )
        default_speaker_gallery = str(
            Path.home() / '.config' / 'dingo_identity' / 'speakers.json'
        )
        self.speaker_gallery_path = Path(
            self.declare_parameter('speaker_gallery', default_speaker_gallery).value
        ).expanduser()

        self.raw_audio_pub = self.create_publisher(
            RawAudio, f'{prefix}/voice/audio', 10
        )
        self.transcript_pub = self.create_publisher(
            String, f'{prefix}/voice/transcript', 10
        )
        self.command_pub = self.create_publisher(
            String, f'{prefix}/voice/command', 10
        )
        self.create_subscription(
            String, f'{prefix}/voice/text_command', self.text_command, 10
        )
        # Optional transcript injection for Gazebo/CI diagnostics. The
        # production audio path calls handle_transcript() after Whisper; this
        # hook is off by default and lets a simulator exercise that exact
        # wake-word/confirmation path without a physical microphone.
        self.transcript_input_topic = str(
            self.declare_parameter('transcript_input_topic', '').value or ''
        ).strip()
        if self.transcript_input_topic:
            self.create_subscription(
                String,
                self.transcript_input_topic,
                self.transcript_input,
                10,
            )
        self.create_subscription(
            String, f'{prefix}/voice/provider_command', self.provider_command, 10
        )
        self.reply_pub = self.create_publisher(
            String, f'{prefix}/voice/reply', 10
        )
        self.speaker_state_pub = self.create_publisher(
            String, f'{prefix}/voice/speaker', 10
        )
        self.gemini_usage_pub = self.create_publisher(
            String, f'{prefix}/voice/gemini_usage', 10
        )
        self.create_subscription(
            String, f'{prefix}/voice/speaker_command', self.speaker_command, 10
        )
        # The Dashboard also publishes replies on this topic (for example,
        # the live result of «τι βλέπεις;»).  The voice node speaks those
        # replies while keeping the text reply as the source of truth.
        self.create_subscription(
            String, f'{prefix}/voice/reply', self.voice_reply, 10
        )
        # The Dashboard may be opened after the voice service.  Keep the
        # latest state latched so a reload immediately shows microphone and
        # wake-word status instead of waiting for a new utterance.
        status_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_pub = self.create_publisher(
            String, f'{prefix}/voice/status', status_qos
        )

        self.audio_queue = queue.Queue(maxsize=80)
        # Browser/phone microphone blocks arrive on a separate ROS topic. They
        # use the same wake-word + VAD + Whisper pipeline as the ReSpeaker,
        # while the physical capture stream keeps running as a fallback.
        self.remote_audio_queue = queue.Queue(maxsize=80)
        self.remote_audio_last_at = 0.0
        self.create_subscription(
            RawAudio,
            f'{prefix}/voice/remote_audio',
            self.remote_audio_callback,
            10,
        )
        # The Anker speaker is connected to the Mini PC's AUX output rather
        # than the XVF3800 playback reference. Ignore captured frames while
        # the robot is speaking (and briefly after) so the assistant cannot
        # transcribe its own TTS as a new command.
        self.audio_ignore_until = 0.0
        self.stream_lock = threading.Lock()
        self.stream = None
        self.stream_reader_thread = None
        self.stream_reader_stop = threading.Event()
        self.stream_is_blocking_reader = False
        # A USB audio reconnect can leave PortAudio's stream object alive
        # while its file descriptor points to the removed ALSA node.  The
        # watchdog below closes that stale stream so ensure_stream() can open
        # the newly enumerated ReSpeaker automatically.
        self.stream_started_at = 0.0
        self.capture_watchdog_s = 8.0
        self.capture_rate = 0
        self.capture_channels = 0
        self.capture_device = ''
        self.last_device_check_at = 0.0
        self.last_portaudio_refresh_at = 0.0
        self.last_device_error = ''
        self.pipewire_configured = False
        self.pipewire_last_configure_at = 0.0
        self.pipewire_source_name = ''
        self.status_state = ''
        self.status_detail = ''
        self.last_transcript = ''
        if self.stt_backend == 'npu':
            self.stt_active_backend = 'npu'
            self.stt_provider = 'pending'
        elif self.stt_backend == 'vulkan':
            self.stt_active_backend = 'vulkan'
            self.stt_provider = 'whisper.cpp/Vulkan GPU'
        else:
            self.stt_active_backend = 'cpu'
            self.stt_provider = 'faster-whisper'
        self.stt_fallback_reason = ''
        self.audio_rms = 0.0
        self.audio_peak = 0.0
        self.last_audio_at = 0.0
        self.last_meter_publish_at = 0.0
        self.wake_model = None
        self.wake_model_error = ''
        self.wake_last_score = 0.0
        self.wake_segment_max_score = 0.0
        self.wake_last_detected_at = 0.0
        self.wake_last_trigger_monotonic = 0.0
        self.wake_command_until = 0.0
        self.wake_ack_sent = False
        self.wake_score_hits = deque()
        self.wake_fallback_last_started_at = 0.0
        self.current_segment_wake_authorized = False
        self.last_segment_wake_authorized = False
        self.wake_audio_buffer = (
            np.zeros(0, dtype=np.int16) if np is not None else None
        )
        self.wake_chime_lock = threading.Lock()

        self.pre_roll = deque(maxlen=24)
        self.speech_blocks = []
        self.speech_active = False
        self.speech_duration_s = 0.0
        self.silence_duration_s = 0.0
        self.noise_rms = 0.004
        try:
            self.vad_threshold = max(
                0.005,
                min(
                    0.2,
                    float(self.declare_parameter('vad_threshold', 0.015).value),
                ),
            )
        except (TypeError, ValueError):
            self.vad_threshold = 0.015
        self.vad_start_blocks = max(
            1,
            min(
                10,
                int(self.declare_parameter('vad_start_blocks', 3).value or 3),
            ),
        )
        self.min_speech_s = 0.35
        self.max_speech_s = 8.0
        self.end_silence_s = 0.80
        self.speech_start_streak = 0

        self.worker_lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker_future = None
        self.worker_kind = None
        self.worker_wake_authorized = False
        # A wake-only utterance and the command that follows it are normally
        # two VAD segments. Whisper may still be decoding the first one when
        # the second finishes, so retain one authorised command instead of
        # dropping it as "busy".
        self.pending_stt_samples = None
        self.pending_stt_wake_authorized = False
        self.whisper_model = None
        self.wake_whisper_model = None
        self.wake_whisper_lock = threading.Lock()
        self.npu_whisper = None
        self.npu_disabled = False
        self.vulkan_disabled = False
        self.vulkan_backend_reported = False
        self.speaker_worker = ThreadPoolExecutor(max_workers=1)
        self.speaker_future = None
        self.speaker_future_lock = threading.Lock()
        self.speaker_recognizer = None
        self.speaker_import_error = ''
        self.speaker_enrollment = None
        if self.speaker_enabled:
            try:
                from dingo_speaker_identity import SpeakerRecognizer

                self.speaker_recognizer = SpeakerRecognizer(
                    self.speaker_gallery_path, self.speaker_threshold
                )
            except Exception as exc:
                self.speaker_import_error = str(exc)
        self.speaker_state = {
            'state': (
                'ready' if self.speaker_recognizer is not None
                else 'error' if self.speaker_enabled else 'disabled'
            ),
            'current_name': None,
            'current_score': None,
            'last_update': None,
            'enrolled': self.speaker_names(),
            'enrollment': None,
            'detail': (
                'Η αναγνώριση φωνής είναι έτοιμη.'
                if self.speaker_recognizer is not None
                else self.speaker_import_error
                if self.speaker_enabled
                else 'Η αναγνώριση φωνής είναι απενεργοποιημένη.'
            ),
        }
        self.publish_speaker_state(force=True)
        self.tts_worker = ThreadPoolExecutor(max_workers=1)
        # The Anker speaker is outside the ReSpeaker AEC reference path. Guard
        # capture as soon as a reply is queued, including TTS synthesis time.
        self.tts_busy = threading.Event()
        self.tts_error_logged = False
        self.pending_command = None
        self.pending_expires_at = 0.0
        self.load_wake_model()

        # The high-frequency timer only drains a bounded number of blocks per
        # tick.  Audio capture stays in PortAudio's callback thread, while
        # Whisper and the LLM stay in the single worker thread.
        self.create_timer(0.05, self.tick)
        self.create_timer(2.0, self.ensure_stream)
        self.create_timer(2.0, self.publish_speaker_state)
        self.publish_status(
            'waiting_for_microphone',
            f'Περιμένω συσκευή με όνομα «{self.microphone_name}»',
        )

    def load_wake_model(self):
        """Load the configured local wake-word model when it exists."""
        if not self.wake_enabled:
            self.wake_model_error = 'wake detector disabled'
            return
        if not self.wake_model_path.is_file():
            self.wake_model_error = f'Δεν υπάρχει ακόμη {self.wake_model_path}'
            self.get_logger().warning(
                'Dedicated wake word is not ready: λείπει το μοντέλο '
                f'«{self.wake_model_path}». Οι φωνητικές εντολές παραμένουν '
                'κλειστές μέχρι να εγκατασταθεί έγκυρο μοντέλο.'
            )
            return
        try:
            from openwakeword.model import Model

            self.wake_model = Model(
                wakeword_model_paths=[str(self.wake_model_path)],
                # External VAD already gates complete speech segments.  The
                # XVF3800's processed ASR beam can be quiet enough that
                # openWakeWord's optional Silero VAD suppresses a real
                # wake-word before the detector sees it, so keep this filter
                # disabled unless explicitly requested.
                vad_threshold=self.wake_model_vad_threshold,
            )
            self.wake_model_error = ''
            self.get_logger().info(
                f'Wake word ready: «{self.wake_model_name}» από {self.wake_model_path}'
            )
        except Exception as exc:  # noqa: BLE001
            self.wake_model_error = str(exc)
            self.get_logger().error(
                f'Αποτυχία φόρτωσης wake model «{self.wake_model_path}»: {exc}'
            )

    def publish_status(self, state, detail='', *, force=False, log=True):
        state = str(state)
        detail = str(detail or '')
        if not force and state == self.status_state and detail == self.status_detail:
            return
        self.status_state = state
        self.status_detail = detail
        payload = {
            'state': state,
            'detail': detail,
            'device': self.capture_device or None,
            'sample_rate': self.capture_rate or None,
            'channels': self.capture_channels or None,
            'stt_model': self.stt_model_name,
            'stt_backend': self.stt_backend,
            'stt_active_backend': self.stt_active_backend,
            'stt_provider': self.stt_provider,
            'stt_fallback_model': self.stt_fallback_model_name,
            'wake_fallback_model': self.wake_fallback_model_name,
            'stt_fallback_reason': self.stt_fallback_reason or None,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'llm_tool_calling': 'native' if self.native_tool_calling else 'json_fallback',
            'llm_fallback_active': bool(self.llm_fallback_active),
            'llm_fallback_reason': self.llm_fallback_reason or None,
            'last_transcript': self.last_transcript or None,
            'wake_ready': bool(self.wake_model is not None),
            'wake_model': self.wake_model_name,
            'wake_model_path': str(self.wake_model_path),
            'wake_threshold': self.wake_threshold,
            'wake_model_vad_threshold': self.wake_model_vad_threshold,
            'wake_confirmation_hits': self.wake_confirmation_hits,
            'wake_confirmation_window_s': self.wake_confirmation_window_s,
            'wake_channel': self.wake_channel,
            'stt_channel': self.stt_channel,
            'wake_fallback_stt': self.wake_fallback_stt,
            'wake_highpass_hz': self.wake_highpass_hz,
            'wake_score': round(float(self.wake_last_score), 4),
            'wake_segment_max_score': round(
                float(self.wake_segment_max_score), 4
            ),
            'wake_detected_at': self.wake_last_detected_at or None,
            'audio_rms': round(float(self.audio_rms), 5),
            'audio_peak': round(float(self.audio_peak), 5),
            'last_audio_at': self.last_audio_at or None,
            'wake_error': self.wake_model_error or None,
        }
        self.status_pub.publish(String(data=compact_json(payload)))
        if log:
            if detail:
                self.get_logger().info(f'Voice: {state} — {detail}')
            else:
                self.get_logger().info(f'Voice: {state}')

    def speaker_names(self):
        if self.speaker_recognizer is None:
            return []
        try:
            return self.speaker_recognizer.gallery.names()
        except Exception:
            return []

    def publish_speaker_state(self, force=False):
        del force  # Kept in the signature so state transitions can be explicit at call sites.
        state = dict(self.speaker_state)
        state['enrolled'] = self.speaker_names()
        enrollment = state.get('enrollment')
        if enrollment is not None:
            state['enrollment'] = dict(enrollment)
        self.speaker_state_pub.publish(String(data=compact_json(state)))

    def set_speaker_state(self, state, detail=None, **updates):
        if self.speaker_recognizer is None and self.speaker_enabled:
            state = 'error'
        self.speaker_state['state'] = str(state)
        if detail is not None:
            self.speaker_state['detail'] = str(detail)
        self.speaker_state.update(updates)
        self.publish_speaker_state(force=True)

    def speaker_command(self, msg):
        """Handle explicit Dashboard enrollment/maintenance commands."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.publish_reply('Η εντολή αναγνώρισης φωνής δεν είναι έγκυρη.', ok=False)
            return
        if not isinstance(payload, dict):
            self.publish_reply('Η εντολή αναγνώρισης φωνής δεν είναι έγκυρη.', ok=False)
            return
        action = str(payload.get('action', '')).strip().lower()
        if self.speaker_recognizer is None:
            self.publish_reply(
                'Η αναγνώριση φωνής δεν είναι διαθέσιμη σε αυτό το σύστημα.',
                ok=False,
                action=action or None,
            )
            return
        try:
            if action == 'enroll':
                name = self.speaker_recognizer.gallery._clean_name(payload.get('name'))
                requested = int(payload.get('samples') or self.speaker_enroll_count)
                required = max(3, min(8, requested))
                self.speaker_enrollment = {
                    'name': name,
                    'embeddings': [],
                    'required': required,
                }
                self.set_speaker_state(
                    'enrolling',
                    f'Πες {required} διαφορετικές φράσεις για «{name}».',
                    current_name=None,
                    current_score=None,
                    last_update=None,
                    enrollment={
                        'name': name,
                        'captured': 0,
                        'required': required,
                    },
                )
                self.publish_reply(
                    f'Ξεκίνησε η εγγραφή φωνής για «{name}». Πες {required} διαφορετικές φράσεις.',
                    action='speaker_enroll',
                )
            elif action == 'cancel':
                active = self.speaker_enrollment is not None
                self.speaker_enrollment = None
                self.set_speaker_state(
                    'ready',
                    'Η εγγραφή φωνής ακυρώθηκε.' if active else 'Δεν υπάρχει ενεργή εγγραφή φωνής.',
                    enrollment=None,
                )
                if active:
                    self.publish_reply('Η εγγραφή φωνής ακυρώθηκε.', action='speaker_cancel')
            elif action == 'forget':
                name = self.speaker_recognizer.gallery._clean_name(payload.get('name'))
                removed = self.speaker_recognizer.gallery.remove(name)
                self.set_speaker_state(
                    'ready',
                    f'Διαγράφηκε η φωνή «{name}».' if removed else f'Δεν βρήκα εγγραφή για «{name}».',
                )
                self.publish_reply(
                    f'Διαγράφηκε η φωνή «{name}».' if removed else f'Δεν βρήκα εγγραφή για «{name}».',
                    ok=removed,
                    action='speaker_forget',
                )
            elif action == 'query':
                self.publish_reply(self.speaker_query_text(), action='speaker_query')
            else:
                self.publish_reply('Άγνωστη εντολή αναγνώρισης φωνής.', ok=False)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.publish_reply(str(exc), ok=False, action=action or None)

    def speaker_query_text(self):
        state = self.speaker_state
        if state.get('state') in {'disabled', 'error'}:
            return 'Η αναγνώριση φωνής δεν είναι διαθέσιμη.'
        if not state.get('enrolled'):
            return 'Δεν έχει εγγραφεί ακόμη καμία φωνή στο Dingo.'
        last_update = state.get('last_update')
        try:
            stale = last_update is None or time.time() - float(last_update) > 5.0
        except (TypeError, ValueError):
            stale = True
        if stale:
            return 'Δεν έχω αρκετά πρόσφατο δείγμα φωνής για να πω ποιος μίλησε.'
        name = state.get('current_name')
        if name:
            return f'Μίλησε ο/η {name}.'
        return 'Άκουσα φωνή, αλλά δεν αναγνωρίζω ποιος μίλησε.'

    def queue_speaker_analysis(self, samples):
        if not self.speaker_enabled or self.speaker_recognizer is None:
            return
        with self.speaker_future_lock:
            if self.speaker_future is not None and not self.speaker_future.done():
                return
            self.set_speaker_state(
                'recognizing',
                'Αναλύω το τελευταίο δείγμα φωνής…',
            )
            try:
                self.speaker_future = self.speaker_worker.submit(
                    self.speaker_recognizer.identify, samples
                )
            except RuntimeError:
                self.speaker_future = None

    def poll_speaker_worker(self):
        with self.speaker_future_lock:
            future = self.speaker_future
            if future is None or not future.done():
                return
            self.speaker_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.set_speaker_state('error', f'Σφάλμα αναγνώρισης φωνής: {exc}')
            self.get_logger().warning(f'Speaker recognition error: {exc}')
            return

        now = time.time()
        name = result.get('name')
        score = result.get('score')
        self.speaker_state['current_name'] = name
        self.speaker_state['current_score'] = (
            round(float(score), 3) if score is not None else None
        )
        self.speaker_state['last_update'] = now
        enrollment = self.speaker_enrollment
        if enrollment is not None:
            enrollment['embeddings'].append(result['embedding'])
            captured = len(enrollment['embeddings'])
            required = enrollment['required']
            self.speaker_state['enrollment'] = {
                'name': enrollment['name'],
                'captured': captured,
                'required': required,
            }
            if captured >= required:
                enrolled_name = self.speaker_recognizer.enroll_embeddings(
                    enrollment['name'], enrollment['embeddings']
                )
                self.speaker_enrollment = None
                self.speaker_state['enrollment'] = None
                self.set_speaker_state(
                    'ready',
                    f'Η φωνή «{enrolled_name}» αποθηκεύτηκε.',
                    current_name=enrolled_name,
                    current_score=1.0,
                    last_update=now,
                )
                self.publish_reply(
                    f'Ολοκληρώθηκε η εγγραφή φωνής για «{enrolled_name}».',
                    action='speaker_enroll',
                )
                return
            self.set_speaker_state(
                'enrolling',
                f'Δείγμα {captured}/{required}. Πες άλλη διαφορετική φράση.',
                last_update=now,
            )
            return
        self.set_speaker_state(
            'ready',
            f'Αναγνωρίστηκε ο/η {name}.' if name else 'Άγνωστη φωνή.',
            last_update=now,
        )

    def publish_reply(self, text, ok=True, action=None):
        payload = {
            'ok': bool(ok),
            'text': str(text),
            'action': action,
            'source': 'voice_assistant',
        }
        self.reply_pub.publish(String(data=compact_json(payload)))
        self.get_logger().info(f'Voice reply: {text}')
        self.queue_tts(text)

    def voice_reply(self, msg):
        """Speak replies produced by the Dashboard (not our own echo)."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get('source') == 'voice_assistant':
            return
        self.queue_tts(payload.get('text', ''))

    def provider_command(self, msg):
        """Switch LLM provider/tool mode without restarting the voice node."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            payload = msg.data
        requested = payload.get('provider') if isinstance(payload, dict) else payload
        tool_value = (
            payload.get('native_tool_calling', payload.get('native_tools'))
            if isinstance(payload, dict)
            else None
        )
        provider = None
        if requested is not None and str(requested).strip():
            requested = normalize_text(requested).replace('-', '_').replace(' ', '_')
            aliases = {
                'local': 'flm',
                'qwen': 'flm',
                'qwen3_5': 'flm',
                'flm': 'flm',
                'gemini': 'gemini',
                'gemini_2_5_flash': 'gemini',
            }
            provider = aliases.get(requested)
            if provider is None:
                self.get_logger().warning(
                    f'Άγνωστος provider από Dashboard: {requested}'
                )
                return
        if provider is None and tool_value is None:
            self.get_logger().warning('Το Dashboard έστειλε κενή ρύθμιση LLM')
            return
        tool_changed = tool_value is not None
        if isinstance(tool_value, str):
            native_tools = normalize_text(tool_value) in {
                '1', 'true', 'yes', 'on', 'native', 'enabled', 'ενεργο'
            }
        else:
            native_tools = bool(tool_value)
        with self.llm_config_lock:
            if provider is not None:
                self.llm_provider = provider
                # A manual provider selection starts a fresh provider session;
                # clear any previous automatic-fallback indicator.
                self.llm_fallback_active = False
                self.llm_fallback_reason = ''
                if provider == 'gemini':
                    self.llm_model = self.gemini_llm_model
                else:
                    self.llm_url = self.local_llm_url
                    self.llm_model = self.local_llm_model
            if tool_changed:
                self.native_tool_calling = native_tools
        tool_label = (
            'native tool calling ενεργό'
            if self.native_tool_calling
            else 'JSON fallback ενεργό'
        )
        if provider is None:
            label = f'{tool_label} · {self.llm_model}'
        else:
            label = (
                f'Gemini {self.llm_model}'
                if provider == 'gemini'
                else f'Τοπικό LLM {self.llm_model}'
            )
            if tool_changed:
                label += f' · {tool_label}'
        self.get_logger().info(f'Ρύθμιση LLM άλλαξε σε {label}')
        self.publish_status(
            self.status_state or 'listening',
            f'Ενεργό: {label}',
            force=True,
            log=False,
        )

    def queue_tts(self, text):
        if not self.tts_enabled:
            return
        text = ' '.join(str(text or '').split())[:500]
        if not text:
            return
        if sd is None or np is None:
            self.log_tts_error_once(
                'Η εκφώνηση είναι απενεργοποιημένη: λείπει το sounddevice ή το numpy.'
            )
            return
        try:
            self.tts_busy.set()
            self.tts_worker.submit(self.speak_text, text)
        except RuntimeError:
            # The executor is shutting down; the text reply was already sent.
            self.tts_busy.clear()

    def log_tts_error_once(self, message):
        if self.tts_error_logged:
            return
        self.tts_error_logged = True
        self.get_logger().warning(message)

    def speak_text(self, text):
        """Run one queued TTS response and always release the audio guard."""
        try:
            self._speak_text_impl(text)
        finally:
            self.tts_busy.clear()

    def _speak_text_impl(self, text):
        if self.tts_backend == 'edge':
            try:
                audio, sample_rate = self.synthesize_edge_tts(text)
                self.play_audio(audio, sample_rate)
                self.get_logger().info(
                    f'TTS: καθαρή ελληνική γυναικεία φωνή Edge ({self.tts_edge_voice})'
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.log_tts_error_once(
                    f'Edge TTS απέτυχε ({exc}). Δεν υπάρχει εφεδρική φωνή.'
                )
                return

        if self.tts_backend == 'gemini':
            try:
                audio, sample_rate, model = self.synthesize_gemini_tts(text)
                self.play_audio(audio, sample_rate)
                self.get_logger().info(
                    f'TTS: εκφωνήθηκε καθαρή ελληνική απάντηση ({model})'
                )
                return
            except Exception as exc:
                self.log_tts_error_once(
                    f'Gemini TTS απέτυχε ({exc}). Δεν υπάρχει εφεδρική φωνή.'
                )
                return

        self.log_tts_error_once(f'Άγνωστο TTS backend: {self.tts_backend}')

    def synthesize_edge_tts(self, text):
        """Return clear Greek PCM audio from Microsoft's Edge Greek voice.

        Edge TTS is used only for speech synthesis, not for LLM reasoning, so
        it does not consume the Gemini quota. There is no local voice fallback.
        """
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError('Λείπει το πακέτο edge-tts.') from exc

        async def collect_audio():
            communicator = edge_tts.Communicate(
                text,
                voice=self.tts_edge_voice,
                connect_timeout=max(5, int(self.tts_timeout_s)),
                receive_timeout=max(5, int(self.tts_timeout_s)),
            )
            chunks = []
            async for chunk in communicator.stream():
                if chunk.get('type') == 'audio' and chunk.get('data'):
                    chunks.append(chunk['data'])
            return b''.join(chunks)

        encoded_audio = asyncio.run(collect_audio())
        if not encoded_audio:
            raise RuntimeError('Το Edge TTS δεν επέστρεψε ήχο.')

        decoded = subprocess.run(
            [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', 'pipe:0', '-f', 'f32le', '-ac', '1', '-ar', '24000',
                'pipe:1',
            ],
            input=encoded_audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(5, float(self.tts_timeout_s)),
        )
        if decoded.returncode != 0 or not decoded.stdout:
            detail = decoded.stderr.decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'Αποτυχία αποκωδικοποίησης Edge TTS: {detail[:180]}')
        audio = np.frombuffer(decoded.stdout, dtype='<f4').copy()
        if audio.size == 0:
            raise RuntimeError('Το Edge TTS επέστρεψε άδειο ήχο.')
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False), 24000

    def synthesize_gemini_tts(self, text):
        """Return natural Greek PCM audio, trying the fast TTS models in order."""
        api_key = self.read_gemini_api_key()
        models = [self.tts_cloud_model]
        if self.tts_cloud_fallback_model not in models:
            models.append(self.tts_cloud_fallback_model)
        last_error = None
        for model in models:
            payload = {
                'model': model,
                'input': (
                    'Διάβασε ακριβώς το παρακάτω κείμενο στα ελληνικά. '
                    'Χρησιμοποίησε φυσική γυναικεία φωνή, καθαρή νεοελληνική '
                    'προφορά, '
                    'σωστό ελληνικό τονισμό, ήρεμο ρυθμό και καθαρή άρθρωση. '
                    'Μην προσθέσεις ή αλλάξεις λέξεις. Κείμενο: «'
                    f'{text}»'
                ),
                'response_format': {'type': 'audio'},
                'generation_config': {
                    'speech_config': [{'voice': self.tts_voice_name}],
                },
            }
            endpoint = 'https://generativelanguage.googleapis.com/v1beta/interactions'
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
                with urlopen(request, timeout=self.tts_timeout_s) as response:
                    body = json.loads(response.read().decode('utf-8'))
                audio_part = body.get('output_audio')
                if not isinstance(audio_part, dict) or not audio_part.get('data'):
                    for step in body.get('steps') or []:
                        for part in step.get('content') or []:
                            if (
                                isinstance(part, dict)
                                and part.get('type') == 'audio'
                                and part.get('data')
                            ):
                                audio_part = part
                    
                if not isinstance(audio_part, dict) or not audio_part.get('data'):
                    raise RuntimeError('Το Gemini TTS δεν επέστρεψε ήχο')
                raw_audio = base64.b64decode(audio_part['data'], validate=True)
                if len(raw_audio) < 2:
                    raise RuntimeError('Το Gemini TTS επέστρεψε άδειο ήχο')
                sample_rate = int(audio_part.get('sample_rate') or 24000)
                channels = max(1, int(audio_part.get('channels') or 1))
                usable_bytes = len(raw_audio) - (len(raw_audio) % 2)
                audio = np.frombuffer(
                    raw_audio[:usable_bytes], dtype='<i2'
                ).astype(np.float32) / 32768.0
                if channels > 1:
                    frame_count = audio.size // channels
                    audio = audio[:frame_count * channels].reshape(
                        frame_count, channels
                    ).mean(axis=1)
                audio = np.clip(audio, -1.0, 1.0).astype(
                    np.float32, copy=False
                )
                self.publish_gemini_usage(
                    body, request_type='tts', model=model
                )
                return audio, sample_rate, model
            except HTTPError as exc:
                last_error = RuntimeError(
                    f'Gemini TTS HTTP {exc.code} ({model})'
                )
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = RuntimeError(
                    f'Gemini TTS δεν είναι διαθέσιμο ({model}): {exc}'
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(
                    f'Μη έγκυρη απάντηση Gemini TTS ({model}): {exc}'
                )
        raise last_error or RuntimeError('Το Gemini TTS δεν είναι διαθέσιμο')

    def play_audio(self, audio, sample_rate):
        if self.tts_volume != 1.0:
            audio = np.clip(audio * self.tts_volume, -1.0, 1.0)
        sample_rate = max(1, int(sample_rate))
        frame_count = int(np.asarray(audio).shape[0]) if np.asarray(audio).ndim else 0
        playback_duration = frame_count / float(sample_rate)
        wake_session_active = time.monotonic() <= self.wake_command_until
        self.publish_status('speaking', 'Εκφωνώ την απάντηση', force=True)

        # The Anker speaker is connected to the Mini PC AUX output, so its
        # playback is not present on the XVF3800 AEC reference channel.  Mute
        # the software capture gate for the *whole* utterance.  The previous
        # 250 ms gate expired while TTS was still speaking, making the robot
        # transcribe itself, fill the STT worker with background jobs, and
        # miss the user's real command.
        self.audio_ignore_until = max(
            self.audio_ignore_until,
            time.monotonic() + playback_duration + 0.8,
        )
        # Do not join speech captured just before playback to audio captured
        # after it.  Keep the wake command window itself alive; only the
        # incomplete VAD segment is discarded.
        self.speech_active = False
        self.speech_start_streak = 0
        self.speech_blocks = []
        self.speech_duration_s = 0.0
        self.silence_duration_s = 0.0
        self.current_segment_wake_authorized = False
        self.last_segment_wake_authorized = False
        self.pre_roll.clear()
        try:
            sd.play(
                audio,
                samplerate=sample_rate,
                device=self.tts_output_device or None,
                blocking=True,
            )
        finally:
            self.tts_busy.clear()
            self.audio_ignore_until = max(
                self.audio_ignore_until, time.monotonic() + 0.8
            )
            # A wake-only acknowledgement must not consume most of the
            # command window.  Give the user a fresh full window after the
            # spoken acknowledgement and its echo guard have completed.
            if wake_session_active:
                self.wake_command_until = max(
                    self.wake_command_until,
                    self.audio_ignore_until + self.wake_listen_window_s,
                )
            self.publish_status('listening', 'Περιμένω την επόμενη εντολή', force=True)

    def configure_pipewire_microphone(self):
        """Select the six-channel XVF3800 source for Pulse/PipeWire clients.

        The XVF3800 exposes six capture channels, but WirePlumber may restore
        the card in a surround profile that opens without delivering frames.
        The pro-audio profile exposes the raw six-channel endpoint reliably.
        The voice node then uses the Pulse compatibility device, keeping
        capture independent of which physical ALSA node PipeWire owns.
        """
        if self.microphone_backend not in {'auto', 'pipewire'}:
            return False
        microphone_name = normalize_text(self.microphone_name)
        if 'respeaker' not in microphone_name and 'xvf3800' not in microphone_name:
            return False
        if self.pipewire_configured:
            return True
        now = time.monotonic()
        if now - self.pipewire_last_configure_at < 4.0:
            return False
        self.pipewire_last_configure_at = now

        try:
            cards = subprocess.run(
                ['pactl', 'list', 'cards', 'short'],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            card_name = ''
            # The first column of `pactl list cards short` is a PipeWire
            # object id, not the ALSA card selector accepted by amixer.  The
            # XVF3800 exposes the stable ALSA id `Array`.
            card_index = 'Array'
            for line in cards.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2:
                    continue
                if 'respeaker' in normalize_text(fields[1]) or 'xvf3800' in normalize_text(fields[1]):
                    card_name = fields[1]
                    break
            if not card_name:
                return False

            profile = 'pro-audio'
            profile_result = subprocess.run(
                ['pactl', 'set-card-profile', card_name, profile],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if profile_result.returncode != 0:
                return False

            # PCM-1 is the XVF3800 USB playback level. Keep the hardware
            # playback controls enabled and at full level; PipeWire/TTS still
            # controls the application volume separately.
            # Seeed's official 6-channel FAQ also requires the Headset
            # capture switches/volumes (numid 8/10) or some USB channels
            # stay silent after a firmware flash.
            if card_index:
                for control, value in (
                    ('numid=3', 'on,on'),
                    ('numid=4', 'on'),
                    ('numid=5', '60,60'),
                    ('numid=6', '60'),
                    ('numid=8', 'on,on,on,on,on,on'),
                    ('numid=9', 'on'),
                    ('numid=10', '60,60,60,60,60,60'),
                    ('numid=11', '60'),
                ):
                    subprocess.run(
                        ['amixer', '-c', card_index, 'cset', control, value],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2.0,
                    )

            sources = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            source_name = ''
            fallback_source = ''
            for line in sources.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2:
                    continue
                candidate = fields[1]
                normalized = normalize_text(candidate)
                if 'respeaker' not in normalized and 'xvf3800' not in normalized:
                    continue
                if 'pro-input' in candidate:
                    source_name = candidate
                    break
                fallback_source = candidate
            source_name = source_name or fallback_source
            if not source_name:
                return False

            default_result = subprocess.run(
                ['pactl', 'set-default-source', source_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if default_result.returncode != 0:
                return False
            self.pipewire_source_name = source_name

            # Route speech output to the Mini PC's analog output when no
            # explicit TTS device was configured.  The user's external
            # speaker is connected there through AUX.  Keep the XVF3800
            # playback sink as a fallback for setups that connect a speaker
            # directly to the ReSpeaker.
            if not self.tts_output_device:
                sinks = subprocess.run(
                    ['pactl', 'list', 'sinks', 'short'],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                output_sink = ''
                respeaker_sink = ''
                for line in sinks.stdout.splitlines():
                    fields = line.split()
                    if len(fields) < 2:
                        continue
                    candidate = fields[1]
                    normalized = normalize_text(candidate)
                    if 'analog-stereo' in normalized and 'pci-' in normalized:
                        output_sink = candidate
                        break
                    if (
                        ('respeaker' in normalized or 'xvf3800' in normalized)
                        and 'pro-output' in normalized
                    ):
                        respeaker_sink = candidate
                output_sink = output_sink or respeaker_sink
                if output_sink:
                    output_result = subprocess.run(
                        ['pactl', 'set-default-sink', output_sink],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                    )
                    if output_result.returncode == 0:
                        self.get_logger().info(
                            'PipeWire: TTS έξοδος στο Mini PC analog/AUX'
                            if 'analog-stereo' in normalize_text(output_sink)
                            else 'PipeWire: TTS έξοδος στο reSpeaker XVF3800 Pro'
                        )
            self.pipewire_configured = True
            self.get_logger().info(
                'PipeWire: χρησιμοποιώ reSpeaker XVF3800 σε pro-audio / 6 κανάλια'
            )
            return True
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return False

    def release_pipewire_microphone(self):
        """Release the XVF3800 ALSA device for the calibrated direct path."""
        if self.microphone_backend != 'alsa':
            return
        try:
            cards = subprocess.run(
                ['pactl', 'list', 'cards', 'short'],
                check=False, capture_output=True, text=True, timeout=2.0,
            )
            for line in cards.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2:
                    continue
                card = fields[1]
                normalized = normalize_text(card)
                if 'respeaker' in normalized or 'xvf3800' in normalized:
                    subprocess.run(
                        ['pactl', 'set-card-profile', card, 'off'],
                        check=False, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=2.0,
                    )
                    return
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass

    def ensure_stream(self):
        if sd is None or np is None:
            self.publish_status(
                'dependency_error',
                'Χρειάζονται τα Python packages sounddevice και numpy',
            )
            return
        with self.stream_lock:
            active_stream = self.stream
            stream_started_at = self.stream_started_at

        if active_stream is not None:
            now = time.monotonic()
            # During TTS playback the capture gate deliberately skips audio
            # blocks (the speaker is not on the ReSpeaker AEC reference), so
            # last_audio_at is expected to pause.  Wait until playback and its
            # echo guard finish before judging the ALSA stream stale.
            if self.tts_busy.is_set() or now < self.audio_ignore_until:
                return
            # Give a newly opened stream time to deliver its first block. Once
            # it has run for a few seconds, an unchanged last_audio_at means
            # the USB/ALSA endpoint disappeared or PortAudio stopped calling
            # the callback.  Do not leave the node falsely reporting
            # "listening" forever in that state.
            started_recently = (
                stream_started_at > 0.0
                and now - stream_started_at < self.capture_watchdog_s
            )
            audio_recently = (
                self.last_audio_at > 0.0
                and time.time() - self.last_audio_at < self.capture_watchdog_s
            )
            if started_recently or audio_recently:
                return
            self.reset_capture_stream(
                active_stream,
                'Το stream του ReSpeaker σταμάτησε· επανασύνδεση USB ήχου…',
            )

        now = time.monotonic()
        if now - self.last_device_check_at < 1.5:
            return
        self.last_device_check_at = now
        try:
            # The historical home_robot node opened the XVF3800 directly via
            # ALSA. Release PipeWire's always-open pro-audio node first so
            # PortAudio can see `hw:1,0` and deliver all six channels.
            self.release_pipewire_microphone()
            pipewire_ready = self.configure_pipewire_microphone()
            devices = sd.query_devices()
            needle = normalize_text(self.microphone_name)
            named_candidates = []
            pipewire_candidates = []
            for index, info in enumerate(devices):
                name = str(info.get('name', ''))
                if int(info.get('max_input_channels', 0) or 0) <= 0:
                    continue
                normalized_name = normalize_text(name).strip()
                if normalized_name in {'pulse', 'default'}:
                    pipewire_candidates.append((index, info))
                if not needle or needle in normalized_name:
                    named_candidates.append((index, info))
            candidates = (
                # On this host both ALSA/Pulse aliases are exposed.  The
                # `pulse` alias can open a much quieter compatibility stream
                # than PipeWire's `default` source, even though both report
                # the same device.  Prefer the endpoint that follows the
                # configured default source; keep `pulse` as a fallback.
                sorted(
                    pipewire_candidates,
                    key=lambda item: (
                        normalize_text(str(item[1].get('name', ''))).strip()
                        != 'default'
                    ),
                ) + named_candidates
                if pipewire_ready
                else named_candidates
            )
            if not candidates and self.microphone_backend == 'alsa':
                # PortAudio snapshots ALSA devices when it is initialized.
                # During boot the voice service can start a few seconds before
                # the USB array appears, leaving query_devices() permanently
                # stale even though arecord already sees the ReSpeaker. With
                # no stream open it is safe to refresh that registry.
                if now - self.last_portaudio_refresh_at >= 5.0:
                    self.last_portaudio_refresh_at = now
                    try:
                        sd._terminate()
                        sd._initialize()
                        devices = sd.query_devices()
                        named_candidates = [
                            (index, info)
                            for index, info in enumerate(devices)
                            if int(info.get('max_input_channels', 0) or 0) > 0
                            and (
                                not needle
                                or needle
                                in normalize_text(str(info.get('name', '')))
                            )
                        ]
                        candidates = named_candidates
                    except Exception:
                        candidates = []
            if not candidates and self.microphone_backend == 'alsa':
                # PortAudio can hide a busy/disabled ALSA card from the
                # enumeration even though the stable device name remains
                # usable.  The previous home_robot node opened this endpoint
                # directly, so keep the same explicit fallback.
                direct_names = []
                try:
                    listed = subprocess.run(
                        ['arecord', '-l'], check=False, capture_output=True,
                        text=True, timeout=2.0,
                    )
                    for line in listed.stdout.splitlines():
                        match = re.search(
                            r'card\s+(\d+):.*device\s+(\d+):', line,
                            flags=re.IGNORECASE,
                        )
                        normalized_line = normalize_text(line)
                        if match and (
                            needle in normalized_line
                            or 'xvf3800' in normalized_line
                        ):
                            direct_names.append(
                                f'hw:{match.group(1)},{match.group(2)}'
                            )
                except (FileNotFoundError, subprocess.SubprocessError, OSError):
                    pass
                direct_names.extend(('hw:Array,0', 'hw:1,0'))
                for direct_name in dict.fromkeys(direct_names):
                    try:
                        direct_info = sd.query_devices(direct_name)
                        if int(direct_info.get('max_input_channels', 0) or 0) > 0:
                            candidates = [(direct_name, direct_info)]
                            break
                    except Exception:
                        continue
            if not candidates:
                self.publish_status(
                    'waiting_for_microphone',
                    f'Δεν βρέθηκε ακόμη «{self.microphone_name}»',
                )
                return

            device_index, info = candidates[0]
            max_channels = int(info.get('max_input_channels', 0) or 0)
            channels = self.microphone_channels or max_channels
            channels = max(1, min(channels, max_channels))
            device_name = str(info.get('name', device_index))
            use_blocking_reader = (
                pipewire_ready
                and normalize_text(device_name).strip() in {'pulse', 'default'}
            )
            default_rate = int(round(float(info.get('default_samplerate', 48000))))
            rates = []
            if self.capture_rate_preference > 0:
                rates.append(self.capture_rate_preference)
            rates.extend([16000, 48000, default_rate])
            seen_rates = set()
            last_error = None
            for rate in rates:
                if rate in seen_rates or rate <= 0:
                    continue
                seen_rates.add(rate)
                stream = None
                try:
                    # The PipeWire/Pulse ALSA bridge does not reliably wake
                    # blocking reads at a 20 ms period on this host.  A
                    # 100 ms read is still responsive enough for wake/VAD and
                    # matches the period that successfully returns XVF3800
                    # frames from the same device.
                    blocksize = (
                        max(1024, int(rate * 0.1))
                        if use_blocking_reader
                        else max(256, int(rate * 0.02))
                    )
                    stream = sd.InputStream(
                        device=device_index,
                        samplerate=rate,
                        channels=channels,
                        dtype='int16',
                        blocksize=blocksize,
                        # PortAudio's callback mode can open the Pulse
                        # compatibility device but never deliver callbacks on
                        # this host.  Its blocking read API does deliver the
                        # XVF3800 frames, so use a small reader thread for
                        # PipeWire and retain callback mode for direct ALSA.
                        callback=None if use_blocking_reader else self.audio_callback,
                    )
                    stream.start()
                    with self.stream_lock:
                        self.stream = stream
                        self.stream_is_blocking_reader = use_blocking_reader
                        self.stream_started_at = time.monotonic()
                        self.capture_rate = int(rate)
                        self.capture_channels = int(channels)
                        if pipewire_ready and normalize_text(device_name).strip() in {
                            'pulse', 'default'
                        }:
                            self.capture_device = (
                                f'{self.microphone_name} μέσω PipeWire ({device_name})'
                            )
                        else:
                            self.capture_device = device_name
                    if use_blocking_reader:
                        self.stream_reader_stop.clear()
                        reader = threading.Thread(
                            target=self.read_stream_blocking,
                            args=(stream, blocksize),
                            name='dingo-pipewire-capture',
                            daemon=True,
                        )
                        self.stream_reader_thread = reader
                        reader.start()
                    self.publish_status(
                        'listening',
                        f'{self.capture_device} · {rate} Hz · {channels} ch',
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
            detail = f'Αποτυχία ανοίγματος ReSpeaker: {last_error}'
            if detail != self.last_device_error:
                self.last_device_error = detail
                self.get_logger().warning(detail)
            self.publish_status('microphone_error', detail)
        except Exception as exc:
            detail = f'Αποτυχία ανίχνευσης audio device: {exc}'
            if detail != self.last_device_error:
                self.last_device_error = detail
                self.get_logger().warning(detail)
            self.publish_status('microphone_error', detail)

    def read_stream_blocking(self, stream, blocksize):
        """Read PipeWire/Pulse frames without PortAudio callback mode."""
        try:
            while not self.stream_reader_stop.is_set():
                block, overflowed = stream.read(blocksize)
                if block is None or len(block) == 0:
                    continue
                self.audio_callback(block, len(block), None, overflowed)
        except Exception as exc:
            if not self.stream_reader_stop.is_set():
                self.get_logger().warning(f'PipeWire capture σταμάτησε: {exc}')
                self.publish_status('microphone_error', f'PipeWire capture: {exc}')
        finally:
            with self.stream_lock:
                if self.stream is stream:
                    self.stream = None
                    self.stream_is_blocking_reader = False
                    self.stream_started_at = 0.0
                    self.capture_rate = 0
                    self.capture_channels = 0
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def reset_capture_stream(self, stream, detail):
        """Close a dead capture stream and let the timer open it again."""
        with self.stream_lock:
            if self.stream is not stream:
                return False
            blocking_reader = self.stream_is_blocking_reader
            self.stream = None
            self.stream_is_blocking_reader = False
            self.stream_started_at = 0.0
            self.capture_rate = 0
            self.capture_channels = 0
            if blocking_reader:
                self.stream_reader_stop.set()
        try:
            if blocking_reader and hasattr(stream, 'abort'):
                stream.abort()
            else:
                stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        # Discard blocks read from the old descriptor; the next stream starts
        # with a clean wake/VAD buffer on the following timer tick.
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.get_logger().warning(detail)
        self.publish_status('microphone_error', detail)
        return True

    def audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            # Do not call ROS logging from PortAudio's callback thread.
            pass
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # Dropping the newest block is safer than blocking the audio
            # callback and causing a cascading overrun.
            pass

    def remote_audio_callback(self, message):
        """Convert a browser's mono pcm-s16 block to the normal int16 shape."""
        if np is None or not message.data:
            return
        try:
            if str(message.format or 'pcm-s16').lower() not in {'pcm-s16', 's16le'}:
                return
            channels = max(1, int(message.number_of_channels or 1))
            raw = bytes((int(value) & 0xFF for value in message.data))
            values = np.frombuffer(raw, dtype='<i2')
            if channels > 1:
                usable = (len(values) // channels) * channels
                values = values[:usable].reshape(-1, channels)[:, 0]
            if len(values) < 2:
                return
            rate = max(1, int(message.sample_rate or 16000))
            if rate != self.target_sample_rate:
                length = max(1, int(round(len(values) * self.target_sample_rate / rate)))
                old = np.linspace(0.0, 1.0, num=len(values), endpoint=False)
                new = np.linspace(0.0, 1.0, num=length, endpoint=False)
                values = np.interp(new, old, values).astype(np.int16)
            block = np.asarray(values, dtype='<i2').reshape(-1, 1)
            self.remote_audio_last_at = time.monotonic()
            try:
                self.remote_audio_queue.put_nowait(block)
            except queue.Full:
                try:
                    self.remote_audio_queue.get_nowait()
                    self.remote_audio_queue.put_nowait(block)
                except queue.Empty:
                    pass
        except (TypeError, ValueError, OverflowError):
            return

    def publish_audio_block(self, block):
        if not self.publish_raw_audio:
            return
        raw = np.ascontiguousarray(block, dtype='<i2').tobytes()
        message = RawAudio()
        message.timestamp = self.get_clock().now().to_msg()
        message.data = list(raw)
        message.format = 'pcm-s16'
        message.sample_rate = int(self.capture_rate)
        message.number_of_channels = int(self.capture_channels)
        self.raw_audio_pub.publish(message)

    def tick(self):
        self.poll_worker()
        self.poll_speaker_worker()
        remote_blocks = []
        for _ in range(12):
            try:
                remote_blocks.append(self.remote_audio_queue.get_nowait())
            except queue.Empty:
                break
        # While a phone is actively streaming, discard stale hardware blocks
        # so the two microphones do not interleave into one utterance.
        remote_active = time.monotonic() - self.remote_audio_last_at < 0.75
        if remote_active:
            while True:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        for block in remote_blocks:
            if self.tts_busy.is_set() or time.monotonic() < self.audio_ignore_until:
                continue
            self.feed_wake(block)
            self.feed_vad(block)
        if not remote_active:
            for _ in range(12):
                try:
                    block = self.audio_queue.get_nowait()
                except queue.Empty:
                    break
                self.publish_audio_block(block)
                if self.tts_busy.is_set() or time.monotonic() < self.audio_ignore_until:
                    continue
                self.feed_wake(block)
                self.feed_vad(block)
        if self.pending_command and time.monotonic() > self.pending_expires_at:
            self.pending_command = None
            self.pending_expires_at = 0.0
            self.publish_reply('Η επιβεβαίωση έληξε. Πες ξανά την εντολή.', ok=False)

    def configure_wake_highpass(self):
        """Prepare the measured XVF3800 wake-channel noise filter."""
        if (
            self.wake_highpass_hz <= 0.0
            or butter is None
            or lfilter is None
            or lfilter_zi is None
        ):
            self._wake_hp_b = None
            self._wake_hp_a = None
            self._wake_hp_zi_template = None
            self._wake_hp_zi = None
            if self.wake_highpass_hz > 0.0:
                self.get_logger().warning(
                    'Το scipy δεν είναι διαθέσιμο — το high-pass του wake detector '
                    'μένει απενεργοποιημένο.'
                )
            return
        self._wake_hp_b, self._wake_hp_a = butter(
            2,
            self.wake_highpass_hz / (self.target_sample_rate / 2.0),
            btype='high',
        )
        self._wake_hp_zi_template = lfilter_zi(
            self._wake_hp_b, self._wake_hp_a
        )
        self._wake_hp_zi = None

    def block_to_mono(self, block, channel=None):
        values = np.asarray(block, dtype=np.float32)
        if values.ndim == 2:
            # The XVF3800 exposes six channels.  Select the calibrated
            # consumer channel; averaging causes phase cancellation and noise.
            channel = self.microphone_channel if channel is None else channel
            if channel < 0 or channel >= values.shape[1]:
                channel = 0
            selected = values[:, channel]
            if (
                self.microphone_auto_fallback
                and channel == 1
                and values.shape[1] > 1
            ):
                # Compatibility fallback for a legacy stereo/PipeWire state
                # where channel 1 is silent but channel 0 is live. The
                # calibrated six-channel setup uses processed channel 0 for
                # STT/VAD and the guarded Whisper wake fallback.
                # ``block`` is signed int16 PCM.  Compare levels in the same
                # normalized units used by VAD below; comparing raw int16
                # values with a 0–1 threshold made this fallback effectively
                # unreachable whenever AUX1 contained even a trace of noise.
                selected_rms = (
                    math.sqrt(float(np.mean(np.square(selected)))) / 32768.0
                )
                alternate = values[:, 0]
                alternate_rms = (
                    math.sqrt(float(np.mean(np.square(alternate)))) / 32768.0
                )
                # The microphone gain is applied after this choice.  A clean
                # but naturally quieter ASR beam must not be replaced by the
                # clipped AUX0 signal simply because the pre-gain level is
                # below the configured VAD threshold.
                gain = max(1.0, float(self.microphone_gain or 1.0))
                speech_floor = max(
                    0.0025,
                    float(getattr(self, 'vad_threshold', 0.015) or 0.015) / gain,
                )
                if (
                    selected_rms < speech_floor * 2.0
                    and alternate_rms >= speech_floor
                    and alternate_rms >= selected_rms * 2.0
                ):
                    values = alternate
                else:
                    values = selected
            else:
                values = selected
        values /= 32768.0
        if self.microphone_gain != 1.0:
            # Apply a bounded digital gain before VAD/wake/STT and clip only
            # at the int16 full-scale boundary so normal speech is usable.
            values = np.clip(values * self.microphone_gain, -1.0, 1.0)
        if self.capture_rate == self.target_sample_rate:
            return values
        if self.capture_rate <= 0 or len(values) < 2:
            return values
        output_length = max(
            1, int(round(len(values) * self.target_sample_rate / self.capture_rate))
        )
        old_positions = np.linspace(0.0, 1.0, num=len(values), endpoint=False)
        new_positions = np.linspace(0.0, 1.0, num=output_length, endpoint=False)
        return np.interp(new_positions, old_positions, values).astype(np.float32)

    def play_wake_chime(self):
        """Play an audible wake confirmation on the system's default speaker."""
        if sd is None or not self.wake_chime_lock.acquire(blocking=False):
            return
        self.audio_ignore_until = max(
            self.audio_ignore_until, time.monotonic() + 0.25
        )
        try:
            # Use a distinct two-tone signal and route it through Pulse/PipeWire
            # explicitly.  The robot's speaker is connected to the Mini PC AUX
            # sink; relying only on PortAudio's implicit default could select a
            # different output or make the short chime inaudible.
            rate = 44100
            parts = []
            for index, (frequency, duration) in enumerate(
                ((880.0, 0.11), (1320.0, 0.16))
            ):
                count = max(1, int(rate * duration))
                t = np.arange(count, dtype=np.float32) / rate
                envelope = np.sin(np.pi * np.linspace(0.0, 1.0, count))
                parts.append((0.34 * envelope * np.sin(
                    2.0 * np.pi * frequency * t
                )).astype(np.float32))
                if index == 0:
                    parts.append(np.zeros(int(rate * 0.045), dtype=np.float32))
            tone = np.concatenate(parts)
            pcm = np.clip(tone * 32767.0, -32768, 32767).astype('<i2')
            result = subprocess.run(
                [
                    'paplay', '--raw', '--format=s16le',
                    f'--rate={rate}', '--channels=1',
                    '--device=@DEFAULT_SINK@',
                ],
                input=pcm.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=2.0,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors='replace').strip())
        except Exception:
            try:
                sd.play(
                    tone,
                    samplerate=rate,
                    device=self.tts_output_device or 'default',
                    blocking=True,
                )
            except Exception:
                pass
        finally:
            self.audio_ignore_until = max(
                self.audio_ignore_until, time.monotonic() + 0.65
            )
            self.wake_chime_lock.release()

    def activate_wake(self, score):
        now = time.monotonic()
        # A wake session already has an eight-second command window.  Do not
        # start another chime/reply when the fallback STT re-hears its own
        # wake-word prompt or room echo during that same window.
        if now <= self.wake_command_until:
            return False
        if now - self.wake_last_trigger_monotonic < self.wake_cooldown_s:
            return False
        self.wake_last_score = float(score)
        self.wake_last_detected_at = time.time()
        self.wake_last_trigger_monotonic = now
        self.wake_command_until = now + self.wake_listen_window_s
        self.wake_ack_sent = False
        if self.speech_active:
            # VAD can start a few blocks before the detector recognises the
            # end of the wake phrase. Authorise that same segment as soon as
            # the detector confirms it.
            self.current_segment_wake_authorized = True
        self.publish_status(
            'wake_detected',
            f'Σε ακούω — {self.wake_model_name} ({float(score):.2f})',
            force=True,
        )
        threading.Thread(target=self.play_wake_chime, daemon=True).start()
        return True

    def feed_wake(self, block):
        """Run the configured wake detector on 80 ms, 16 kHz PCM frames."""
        if self.wake_model is None or np is None:
            return
        mono = self.block_to_mono(block, self.wake_channel)
        if len(mono) == 0:
            return
        if self._wake_hp_b is not None:
            if self._wake_hp_zi is None:
                self._wake_hp_zi = self._wake_hp_zi_template * mono[0]
            mono, self._wake_hp_zi = lfilter(
                self._wake_hp_b,
                self._wake_hp_a,
                mono,
                zi=self._wake_hp_zi,
            )
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        self.wake_audio_buffer = np.concatenate((self.wake_audio_buffer, pcm))
        while len(self.wake_audio_buffer) >= 1280:
            chunk = self.wake_audio_buffer[:1280]
            self.wake_audio_buffer = self.wake_audio_buffer[1280:]
            try:
                predictions = self.wake_model.predict(chunk)
                score = max(
                    (float(value) for value in predictions.values()),
                    default=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                self.wake_model_error = str(exc)
                self.get_logger().error(f'Wake detector stopped: {exc}')
                self.wake_model = None
                return
            self.wake_last_score = score
            self.wake_segment_max_score = max(
                self.wake_segment_max_score, score
            )
            if score >= self.wake_threshold:
                now = time.monotonic()
                if now <= self.wake_command_until:
                    self.wake_score_hits.clear()
                    continue
                self.wake_score_hits.append(now)
                cutoff = now - self.wake_confirmation_window_s
                while self.wake_score_hits and self.wake_score_hits[0] < cutoff:
                    self.wake_score_hits.popleft()
                if len(self.wake_score_hits) < self.wake_confirmation_hits:
                    continue
                self.wake_score_hits.clear()
                self.activate_wake(score)

    def feed_vad(self, block):
        if np is None or self.capture_rate <= 0:
            return
        mono = self.block_to_mono(block, self.stt_channel)
        if len(mono) == 0:
            return
        duration = len(mono) / float(self.target_sample_rate)
        rms = math.sqrt(float(np.mean(np.square(mono))))
        self.audio_rms = rms
        self.audio_peak = float(np.max(np.abs(mono)))
        self.last_audio_at = time.time()
        now = time.monotonic()
        if now - self.last_meter_publish_at >= 0.25:
            self.last_meter_publish_at = now
            self.publish_status(
                self.status_state or 'listening',
                self.status_detail,
                force=True,
                log=False,
            )
        threshold = max(self.vad_threshold, self.noise_rms * 3.0)
        speaking = rms >= threshold

        if not self.speech_active:
            self.pre_roll.append(mono)
            if not speaking:
                self.speech_start_streak = 0
                self.noise_rms = 0.97 * self.noise_rms + 0.03 * rms
                return
            self.speech_start_streak += 1
            if self.speech_start_streak < self.vad_start_blocks:
                return
            self.speech_start_streak = 0
            self.speech_active = True
            self.wake_segment_max_score = 0.0
            self.speech_blocks = list(self.pre_roll)
            self.speech_duration_s = sum(
                len(item) for item in self.speech_blocks
            ) / float(self.target_sample_rate)
            self.silence_duration_s = 0.0
            self.pre_roll.clear()
            self.current_segment_wake_authorized = (
                self.wake_model is not None
                and time.monotonic() <= self.wake_command_until
            )
            self.publish_status(
                'speech_detected',
                f'Άκουσα φωνή — ελέγχω αν είπες «{self.wake_model_name}»',
                force=True,
            )
            return

        self.speech_blocks.append(mono)
        self.speech_duration_s += duration
        if speaking:
            self.silence_duration_s = 0.0
        else:
            self.silence_duration_s += duration
        if (
            self.silence_duration_s >= self.end_silence_s
            or self.speech_duration_s >= self.max_speech_s
        ):
            self.finish_speech_segment()

    def finish_speech_segment(self):
        blocks = self.speech_blocks
        self.last_segment_wake_authorized = self.current_segment_wake_authorized
        self.current_segment_wake_authorized = False
        self.speech_active = False
        self.speech_start_streak = 0
        self.speech_blocks = []
        self.speech_duration_s = 0.0
        self.silence_duration_s = 0.0
        self.pre_roll.clear()
        if not blocks:
            return
        samples = np.concatenate(blocks).astype(np.float32, copy=False)
        if len(samples) / float(self.target_sample_rate) < self.min_speech_s:
            self.last_segment_wake_authorized = False
            self.publish_status('listening', 'Η φράση ήταν πολύ σύντομη')
            return
        if self.wake_model is None:
            # Fail closed while the configured wake model is missing or invalid.
            # The microphone meter and RawAudio topic remain live for training,
            # but speech must never reach Whisper/LLM by accident.
            self.last_segment_wake_authorized = False
            self.publish_status(
                'listening',
                f'Το ReSpeaker ακούει — περιμένω έγκυρο wake model «{self.wake_model_name}»',
            )
            return
        if not self.last_segment_wake_authorized:
            if self.wake_fallback_stt:
                now = time.monotonic()
                if (
                    now - self.wake_fallback_last_started_at
                    < self.wake_fallback_cooldown_s
                ):
                    self.publish_status(
                        'listening',
                        f'Άκουσα φωνή — περιμένω «{self.wake_model_name}»',
                    )
                    return
                self.wake_fallback_last_started_at = now
                # The detector remains the fast path.  Whisper is only run
                # after VAD finds a complete speech segment, and
                # handle_transcript() keeps the wake-word privacy gate: speech
                # without «Alexa» never reaches the LLM or motion commands.
                if not self.submit_work('wake_stt', self.transcribe_wake, samples):
                    self.publish_status(
                        'busy',
                        'Ο προηγούμενος φωνητικός κύκλος δεν τελείωσε ακόμη',
                    )
                else:
                    self.publish_status(
                        'transcribing',
                        f'Ελέγχω αν ειπώθηκε «{self.wake_model_name}»',
                    )
            else:
                # Keep the detector always-on, but do not send ordinary room
                # speech to Whisper/LLM. This is both a privacy guard and a
                # CPU guard when the fallback is disabled.
                self.publish_status(
                    'listening',
                    f'Άκουσα φωνή — περίμενα να πεις «{self.wake_model_name}» '
                    f'(score {float(self.wake_segment_max_score):.2f})',
                )
            return
        self.queue_speaker_analysis(samples)
        if not self.submit_work(
            'stt',
            self.transcribe,
            samples,
            wake_authorized=self.last_segment_wake_authorized,
        ):
            if (
                self.last_segment_wake_authorized
                and self.pending_stt_samples is None
            ):
                self.pending_stt_samples = samples.copy()
                self.pending_stt_wake_authorized = True
                self.publish_status(
                    'queued_command',
                    'Άκουσα την εντολή — περιμένει για μεταγραφή',
                )
            else:
                self.publish_status(
                    'busy',
                    'Ο προηγούμενος φωνητικός κύκλος δεν τελείωσε ακόμη',
                )
        else:
            self.publish_status('transcribing', 'Μετατροπή φωνής σε κείμενο')

    def submit_work(
        self, kind, function, argument, *, wake_authorized=False
    ):
        with self.worker_lock:
            if self.worker_future is not None and not self.worker_future.done():
                return False
            self.worker_future = self.worker.submit(function, argument)
            self.worker_kind = kind
            self.worker_wake_authorized = bool(wake_authorized)
            return True

    def poll_worker(self):
        with self.worker_lock:
            future = self.worker_future
            kind = self.worker_kind
            wake_authorized = self.worker_wake_authorized
            if future is None:
                pending = self.pending_stt_samples
                pending_authorized = self.pending_stt_wake_authorized
                if pending is not None:
                    self.pending_stt_samples = None
                    self.pending_stt_wake_authorized = False
                    self.worker_future = self.worker.submit(
                        self.transcribe, pending
                    )
                    self.worker_kind = 'stt'
                    self.worker_wake_authorized = pending_authorized
                    self.publish_status(
                        'transcribing', 'Μετατροπή της εντολής σε κείμενο'
                    )
                return
            if not future.done():
                return
            self.worker_future = None
            self.worker_kind = None
            self.worker_wake_authorized = False
        try:
            result = future.result()
        except Exception as exc:
            self.publish_status('error', str(exc))
            self.publish_reply(f'Παρουσιάστηκε σφάλμα φωνής: {exc}', ok=False)
            return
        if kind in {'stt', 'wake_stt'}:
            self.handle_transcript(
                result,
                detector_authorized=(
                    wake_authorized if kind == 'stt' else False
                ),
            )
        elif kind == 'llm':
            self.handle_intent(result)

    def transcribe_wake(self, samples):
        """Recover a missed wake word without exposing ordinary room speech.

        This fallback is deliberately restricted to VAD-complete segments that
        the dedicated wake detector missed.  The current ReSpeaker's ASR beam
        can be quiet enough for openWakeWord to miss a real Greek pronunciation
        of Alexa, while the installed large-v3-turbo model recognizes it
        reliably.  Reuse the configured accelerated STT backend where
        available; ``handle_transcript()`` still requires the explicit wake
        word before publishing text, contacting an LLM, or moving the robot.
        """
        if self.stt_backend == 'vulkan' and not self.vulkan_disabled:
            self.get_logger().info(
                'Wake fallback: έλεγχος με VULKAN Whisper '
                f'({self.stt_model_name})'
            )
            try:
                # This is a *verification* pass, not final command STT.  Do
                # not seed whisper.cpp with «Alexa»: a prompt-biased decode
                # can turn room noise or speaker echo into a false wake.
                # handle_transcript() still requires the wake word at the
                # beginning before anything can reach the LLM or Dashboard.
                transcript = self.transcribe_vulkan(
                    samples, include_wake_prompt=False
                )
                self.stt_active_backend = 'vulkan'
                self.stt_fallback_reason = ''
                return transcript
            except Exception as exc:  # noqa: BLE001
                self.vulkan_disabled = True
                self.stt_active_backend = 'cpu'
                self.stt_provider = 'faster-whisper'
                self.stt_fallback_reason = str(exc)[:240]
                self.get_logger().warning(
                    f'Το Vulkan Whisper δεν είναι διαθέσιμο για wake check '
                    f'({exc}). Χρησιμοποιώ CPU {self.wake_fallback_model_name}.'
                )

        if self.stt_backend == 'npu':
            self.get_logger().info(
                f'Wake fallback: έλεγχος με NPU Whisper '
                f'({self.stt_model_name})'
            )
            return self.transcribe(samples)

        from faster_whisper import WhisperModel

        with self.wake_whisper_lock:
            # When the final STT also uses the same CPU model, share one
            # instance. This keeps the wake fallback fast without consuming
            # another 1–2 GB of RAM for an identical Whisper model.
            if (
                self.stt_backend == 'cpu'
                and self.stt_model_name == self.wake_fallback_model_name
            ):
                if self.whisper_model is None:
                    self.whisper_model = WhisperModel(
                        self.stt_model_name,
                        device='cpu',
                        compute_type='int8',
                    )
                self.wake_whisper_model = self.whisper_model
            elif self.wake_whisper_model is None:
                self.wake_whisper_model = WhisperModel(
                    self.wake_fallback_model_name,
                    device='cpu',
                    compute_type='int8',
                )
            model = self.wake_whisper_model
        self.get_logger().info(
            f'Wake fallback Whisper έτοιμο σε CPU '
            f'({self.wake_fallback_model_name})'
        )
        # The fallback is deliberately a clean, independent wake check. Do
        # not seed Whisper with the wake word: doing so makes short silence
        # segments hallucinate «Alexa». Silero VAD and segment confidence
        # filtering keep the fallback from turning room noise into a wake.
        segments, _ = model.transcribe(
            samples,
            language=self.stt_language,
            beam_size=1,
            hotwords=None,
            vad_filter=True,
            vad_parameters={
                'min_silence_duration_ms': 350,
                'speech_pad_ms': 120,
            },
            no_speech_threshold=0.60,
            initial_prompt=None,
            condition_on_previous_text=False,
        )
        accepted = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            try:
                no_speech_prob = float(getattr(segment, 'no_speech_prob', 0.0) or 0.0)
            except (TypeError, ValueError):
                no_speech_prob = 1.0
            try:
                avg_logprob = float(getattr(segment, 'avg_logprob', 0.0) or 0.0)
            except (TypeError, ValueError):
                avg_logprob = -99.0
            try:
                compression_ratio = float(
                    getattr(segment, 'compression_ratio', 0.0) or 0.0
                )
            except (TypeError, ValueError):
                compression_ratio = 99.0
            if (
                no_speech_prob > 0.60
                or avg_logprob < -1.25
                or compression_ratio > 2.4
            ):
                continue
            accepted.append(text)
        return ' '.join(accepted).strip()

    def transcribe_vulkan(self, samples, include_wake_prompt=True):
        """Transcribe one VAD-complete segment with whisper.cpp on the GPU.

        The live audio path gives us a float32 mono NumPy array, while
        whisper.cpp's CLI accepts a 16-bit WAV file.  The temporary file is
        deleted immediately after the subprocess exits.  The CLI is required
        to report that it initialized Vulkan; otherwise a silent CPU fallback
        would make the Dashboard claim GPU acceleration when it is not active.
        """
        if np is None:
            raise RuntimeError('Λείπει το NumPy για το Vulkan Whisper.')
        if not self.vulkan_whisper_cli.is_file():
            raise RuntimeError(
                f'Λείπει το whisper.cpp GPU executable: {self.vulkan_whisper_cli}'
            )
        if not self.vulkan_model_path.is_file():
            raise RuntimeError(
                f'Λείπει το whisper.cpp GPU model: {self.vulkan_model_path}'
            )

        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(values) == 0:
            return ''
        pcm = np.clip(values * 32767.0, -32768, 32767).astype('<i2')
        with tempfile.TemporaryDirectory(prefix='dingo-whisper-vulkan-') as temp_dir:
            audio_path = Path(temp_dir) / 'segment.wav'
            with wave.open(str(audio_path), 'wb') as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(self.target_sample_rate)
                audio_file.writeframes(pcm.tobytes())

            env = os.environ.copy()
            library_dir = str(self.vulkan_whisper_cli.parent)
            existing_library_path = env.get('LD_LIBRARY_PATH', '').strip()
            env['LD_LIBRARY_PATH'] = (
                library_dir
                if not existing_library_path
                else f'{library_dir}{os.pathsep}{existing_library_path}'
            )
            command = [
                str(self.vulkan_whisper_cli),
                '-m', str(self.vulkan_model_path),
                '-f', str(audio_path),
                '-l', self.stt_language,
                '-t', str(self.vulkan_threads),
                '-bs', str(self.vulkan_beam_size),
                '-bo', str(self.vulkan_beam_size),
                '-nt',
            ]
            if include_wake_prompt:
                command.extend([
                    '--prompt',
                    f'{self.wake_model_name}, {", ".join(self.wake_words)}.',
                ])
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    timeout=self.vulkan_timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f'Το Vulkan Whisper ξεπέρασε το όριο των '
                    f'{self.vulkan_timeout_s:.0f}s.'
                ) from exc

        stdout = (completed.stdout or '').strip()
        stderr = completed.stderr or ''
        output = f'{stdout}\n{stderr}'
        if completed.returncode != 0:
            detail = ' '.join(output.split())[-500:]
            raise RuntimeError(
                f'Το whisper.cpp Vulkan τερμάτισε με κωδικό '
                f'{completed.returncode}: {detail}'
            )
        if 'using Vulkan' not in output:
            raise RuntimeError(
                'Το whisper.cpp δεν επιβεβαίωσε ενεργό Vulkan GPU backend.'
            )

        if not self.vulkan_backend_reported:
            device_line = next(
                (
                    line.strip()
                    for line in stderr.splitlines()
                    if line.strip().startswith('ggml_vulkan: 0 =')
                ),
                '',
            )
            device_name = device_line.split('=', 1)[1].split('|', 1)[0].strip() \
                if '=' in device_line else 'AMD GPU'
            self.stt_provider = f'whisper.cpp/Vulkan GPU ({device_name})'
            self.vulkan_backend_reported = True
            self.get_logger().info(
                f'Whisper {self.stt_model_name} ενεργό σε GPU μέσω Vulkan: '
                f'{device_name}'
            )
        return stdout

    def transcribe(self, samples):
        if self.stt_backend == 'npu' and not self.npu_disabled:
            try:
                if self.npu_whisper is None:
                    from dingo_npu_whisper_client import NPUWhisperClient

                    self.npu_whisper = NPUWhisperClient(
                        self.npu_python,
                        self.stt_model_name,
                        self.stt_language,
                        self.npu_model_dir,
                        self.npu_tokenizer_dir,
                        self.npu_config_dir,
                        self.npu_cache_dir,
                    )
                    self.stt_active_backend = 'npu'
                    self.stt_provider = json.dumps(
                        self.npu_whisper.providers,
                        ensure_ascii=False,
                        separators=(',', ':'),
                    )
                    self.get_logger().info(
                        f'Whisper {self.stt_model_name} ενεργό στο AMD NPU '
                        f'(VitisAI): {self.npu_whisper.providers}'
                    )
                return self.npu_whisper.transcribe(samples)
            except Exception as exc:  # noqa: BLE001
                self.npu_disabled = True
                self.stt_active_backend = 'cpu'
                self.stt_provider = 'faster-whisper'
                self.stt_fallback_reason = str(exc)[:240]
                if self.npu_whisper is not None:
                    try:
                        self.npu_whisper.close()
                    except Exception:
                        pass
                    self.npu_whisper = None
                self.get_logger().warning(
                    f'Το AMD NPU Whisper δεν είναι διαθέσιμο ({exc}). '
                    f'Χρησιμοποιώ CPU {self.stt_fallback_model_name}.'
                )

        if self.stt_backend == 'vulkan' and not self.vulkan_disabled:
            try:
                transcript = self.transcribe_vulkan(samples)
                self.stt_active_backend = 'vulkan'
                self.stt_fallback_reason = ''
                return transcript
            except Exception as exc:  # noqa: BLE001
                self.vulkan_disabled = True
                self.stt_active_backend = 'cpu'
                self.stt_provider = 'faster-whisper'
                self.stt_fallback_reason = str(exc)[:240]
                self.get_logger().warning(
                    f'Το whisper.cpp Vulkan GPU δεν είναι διαθέσιμο ({exc}). '
                    f'Χρησιμοποιώ CPU {self.stt_fallback_model_name}.'
                )

        if self.whisper_model is None:
            from faster_whisper import WhisperModel

            cpu_model_name = (
                self.stt_fallback_model_name
                if self.stt_backend in {'npu', 'vulkan'}
                else self.stt_model_name
            )
            self.whisper_model = WhisperModel(
                cpu_model_name,
                device='cpu',
                compute_type='int8',
            )
        segments, _ = self.whisper_model.transcribe(
            samples,
            language=self.stt_language,
            beam_size=5,
            hotwords=', '.join(self.wake_words) or None,
            # The external VAD finds a candidate segment; the Whisper VAD
            # trims the pre-roll/trailing silence that otherwise causes
            # multilingual models to repeat unrelated text or emit
            # hallucinations from the speaker's residual echo.
            vad_filter=True,
            vad_parameters={
                'min_silence_duration_ms': 350,
                'speech_pad_ms': 120,
            },
            no_speech_threshold=0.60,
            initial_prompt=f'{self.wake_model_name}, {", ".join(self.wake_words)}.',
            condition_on_previous_text=False,
        )
        return ' '.join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()

    def strip_wake_word(self, text):
        normalized = normalize_text(text)
        alternatives = '|'.join(
            re.escape(word) for word in sorted(self.wake_words, key=len, reverse=True)
        )
        match = re.search(
            rf'(?<!\w)(?:{alternatives})(?!\w)',
            normalized,
        )
        if match is not None:
            prefix = normalized[:match.start()].strip(' ,.!?;:-')
            # Whisper sometimes hallucinates «αν δένω, ντιγκο, ντιγκο…» on
            # room noise. A wake word must be the first spoken token (an
            # optional English «hey» is harmless); do not send the hallucinated
            # remainder to the LLM or to a motion action.
            if prefix and prefix not in {'hey', 'ε', 'ει', 'εε'}:
                return None
            remainder = normalized[match.end():].strip(' ,.!?;:-')
            remainder = re.sub(
                rf'^(?:(?:{alternatives})[ ,.!?;:-]*)+$',
                '',
                remainder,
            ).strip()
            return remainder
        return None

    @staticmethod
    def is_stt_hallucination(text):
        """Reject common Whisper hallucinations from speech STT.

        Whisper can emit subtitle-credit text, repeated filler words, or a
        short gratitude phrase when the captured segment is mostly silence,
        room noise, or speaker echo. This must be filtered before
        ``voice/transcript`` is published, otherwise the Dashboard displays it
        as if the user had spoken it and the LLM may answer a sentence that was
        never said.
        """
        words = re.findall(r'[^\W_]+', normalize_text(text), flags=re.UNICODE)
        normalized = ' '.join(words)
        if re.search(
            r'(?<!\w)υποτιτλοι\s+author(?:\s+)?wave(?!\w)',
            normalized,
            flags=re.UNICODE,
        ):
            return True
        # The installed large-v3-turbo repeatedly produced this phrase from
        # the speaker tail/room silence. Ignore it even when it occurs once,
        # because it is not an actionable command and should not reach Chat.
        if normalized in {'ευχαριστω', 'ευχαριστω πολυ'}:
            return True
        if re.fullmatch(
            r'(?:ευχαριστω(?:\s+πολυ)?)(?:\s+ευχαριστω(?:\s+πολυ)?)+',
            normalized,
            flags=re.UNICODE,
        ):
            return True
        def wake_variant(word):
            return normalize_text(word).translate(str.maketrans({
                'a': 'α', 'l': 'λ', 'e': 'ε', 'x': 'ξ',
                'k': 'κ', 'h': 'χ',
            }))
        # Whisper sometimes mixes Latin and Greek characters and repeats the
        # wake word. That is a wake signal, not a command to send to the LLM.
        if len(words) >= 2 and all(
            wake_variant(word) in {'αλεξα', 'αλεχα'} for word in words
        ):
            return True
        # The configured language is Greek. A multi-word result containing no
        # Greek letters (for example Icelandic «Hvað er það?») is a model
        # hallucination from silence/echo, not a valid Greek user command.
        if len(words) >= 2 and not re.search(r'[α-ω]', normalized, flags=re.UNICODE):
            return True
        # Reject exact short n-gram loops such as «τα πάντα» repeated many
        # times, another characteristic silence hallucination.
        if len(words) >= 3:
            for phrase_size in (1, 2, 3):
                if len(words) % phrase_size:
                    continue
                repetitions = len(words) // phrase_size
                if repetitions >= 3 and words == words[:phrase_size] * repetitions:
                    return True
        return False

    def is_positive(self, text):
        return normalize_text(text).strip() in {
            normalize_text(word) for word in self.POSITIVE_WORDS
        }

    def is_negative(self, text):
        return normalize_text(text).strip() in {
            normalize_text(word) for word in self.NEGATIVE_WORDS
        }

    @staticmethod
    def local_system_control(normalized):
        """Recognize explicit, non-motion system-management commands.

        This is deliberately conservative.  A target and an operation must be
        present before a command is sent to the Dashboard Tool Gateway; a
        vague phrase such as «κάνε κάτι» still goes to the normal LLM answer
        path instead of becoming a system action.
        """
        normalized = normalize_text(normalized)
        if any(phrase in normalized for phrase in (
            'ποιος μιλησε',
            'ποια φωνη ακουσες',
            'ποια φωνη ηταν',
            'ποιος μιλαει',
            'ποιος μιλα',
        )):
            return None
        service = ''
        service_terms = (
            (
                'dingo-mic-array.service',
                ('mic array', 'array μικροφων', 'μικροφωνικ', 'xmos', 'xvf3800', 'respeaker'),
            ),
            (
                'dingo-voice.service',
                ('φωνη', 'voice', 'μικροφων', 'microphone', 'alexa'),
            ),
            (
                'dingo-local-llm.service',
                ('qwen', 'llm', 'τοπικο μοντελο', 'local llm', 'fastflow'),
            ),
            (
                'dingo-dashboard.service',
                ('dashboard', 'ντασμπορντ'),
            ),
            (
                'dingo-sensors.service',
                ('αισθητηρ', 'sensors', 'lidar', 'λιδαρ', 'imu'),
            ),
            (
                'dingo-object-detector.service',
                ('yolo', 'αντικειμενα', 'object detector', 'αναγνωριση αντικειμενων'),
            ),
            (
                'dingo-face-recognition.service',
                ('face', 'προσωπ', 'αναγνωριση προσωπου'),
            ),
        )
        for candidate, terms in service_terms:
            if any(term in normalized for term in terms):
                service = candidate
                break

        target = ''
        if any(term in normalized for term in (
            'global localization',
            'αυτοματο localization',
            'αυτοματο εντοπισ',
            'παγκοσμιο εντοπισ',
            'βρες τη θεση',
            'βρες την θεση',
            'ξαναβρες τη θεση',
            'relocaliz',
            'amcl',
        )):
            target = 'localization'
        elif any(term in normalized for term in (
            'καμερα', 'realsense', 'camera', 'εικονα',
        )):
            target = 'camera'
        elif any(term in normalized for term in (
            'slam', 'χαρτογραφ', 'mapping', 'χαρτη',
        )):
            target = 'mapping'
        elif any(term in normalized for term in (
            'nav2', 'πλοηγ', 'navigation', 'πλοηση', 'αυτονομη πλοηγηση',
        )):
            target = 'navigation'
        elif any(term in normalized for term in (
            'rviz', 'rviz2', 'οπτικοποιηση',
        )):
            target = 'rviz'
        elif any(term in normalized for term in (
            'αποθηκευσε χαρτη', 'αποθηκευση χαρτη', 'save map',
        )):
            target = 'map'
        elif service:
            target = 'service'

        if not target:
            return None

        if target == 'localization' and any(term in normalized for term in (
            'global localization', 'αυτοματο', 'παγκοσμιο', 'βρες τη θεση',
            'βρες την θεση', 'relocaliz', 'amcl',
        )):
            operation = 'global_localization'
        elif any(term in normalized for term in (
            'κατασταση', 'status', 'δουλευει', 'λειτουργει', 'τι τρεχει',
            'ειναι ενεργο', 'ειναι ανοιχτο', 'ειναι συνδεμενο', 'ειναι συνδεδεμενο',
        )):
            operation = 'status'
        elif any(term in normalized for term in (
            'επανεκ', 'restart', 'ξαναξεκι', 'ξανα ξεκι', 'reboot service',
        )):
            operation = 'restart'
        elif any(term in normalized for term in (
            'ακυρω', 'cancel', 'ακυρωση στοχου',
        )):
            operation = 'cancel'
        elif any(term in normalized for term in (
            'σταματα', 'τερματισε', 'τερματισ', 'κλεισε', 'απενεργοποι',
            'stop', 'close',
        )):
            operation = 'stop'
        elif any(term in normalized for term in (
            'ξεκι', 'ενεργοποι', 'ανοιξε', 'άνοιξε', 'start', 'run',
        )):
            operation = 'start'
        elif target == 'map' and any(term in normalized for term in (
            'αποθηκευ', 'save',
        )):
            operation = 'save'
        else:
            # A direct «έλεγξε την κάμερα / το localization» is read-only.
            operation = 'status'

        if target == 'localization' and operation == 'start':
            operation = 'global_localization'
        if target == 'navigation' and operation == 'global_localization':
            target = 'localization'
        if target == 'service' and operation in {'stop', 'cancel', 'save', 'global_localization'}:
            return None
        return {
            'action': 'system_control',
            'target': target,
            'operation': operation,
            'service': service,
        }

    @staticmethod
    def local_system_target(normalized):
        """Recognize common live-system questions without using the LLM."""
        if any(phrase in normalized for phrase in (
            'ελευθερο χωρο',
            'χωρο ελευθερο',
            'ποσο χωρο',
            'δισκο',
            'αποθηκευτικ',
            'storage',
            'disk',
        )):
            return 'disk'
        if any(phrase in normalized for phrase in (
            'μνημη',
            'ram',
            'ραμ',
        )):
            return 'memory'
        if any(phrase in normalized for phrase in (
            'θερμοκρα',
            'θερμοκρασια',
            'ζεστο',
            'θερμοκρασιες',
        )):
            return 'temperature'
        if any(phrase in normalized for phrase in (
            'cpu',
            'επεξεργαστη',
            'φορτιο του συστηματος',
            'φορτο cpu',
        )):
            return 'cpu'
        if any(phrase in normalized for phrase in (
            'watt',
            'βατ',
            'αμπερ',
            'ρευμ',
            'καταναλων',
            'καταναλωση',
        )):
            return 'power'
        if any(phrase in normalized for phrase in (
            'ip',
            'δικτυο',
            'wifi',
            'wi fi',
        )):
            return 'network'
        if any(phrase in normalized for phrase in (
            'υπηρεσι',
            'service',
            'systemd',
            'τι τρεχει',
            'ποια υπηρεσι',
        )):
            return 'services'
        if any(phrase in normalized for phrase in (
            'διαδικασι',
            'διεργασι',
            'process',
            'processes',
            'τι τρεχει στον υπολογιστη',
        )):
            return 'processes'
        if any(phrase in normalized for phrase in (
            'συσκευ',
            'περιφερειακ',
            'τι ειναι συνδεμενο',
            'τι ειναι συνδεδεμενο',
            'hardware',
            'devices',
        )):
            return 'devices'
        if any(phrase in normalized for phrase in (
            'αισθητηρ',
            'lidar',
            'λιδαρ',
            'imu',
            'sensors',
            'localization',
            'εντοπισ',
            'amcl',
            'θεση στον χαρτη',
        )):
            return 'sensors'
        if any(phrase in normalized for phrase in (
            'μικροφων',
            'φωνητικ',
            'wake word',
            'wakeword',
            'alexa',
            'respeaker',
            'stt',
        )):
            return 'voice'
        if any(phrase in normalized for phrase in (
            'yolo',
            'καμερα',
            'camera',
            'αντικειμεν',
            'vision',
        )):
            return 'vision'
        if any(phrase in normalized for phrase in (
            'gemini',
            'quota',
            'οριο του gemini',
            'tokens',
            'τοκεν',
        )):
            return 'gemini'
        if any(phrase in normalized for phrase in (
            'ros',
            'topic',
            'topics',
            'action server',
            'ros graph',
        )):
            return 'ros'
        if any(phrase in normalized for phrase in (
            'πληροφοριες συστηματος',
            'στοιχεια συστηματος',
            'τι εχει το συστημα',
        )):
            return 'summary'
        return None

    @staticmethod
    def is_vision_query(normalized):
        return any(phrase in normalized for phrase in (
            'τι βλεπεις',
            'τι βλεπουμε',
            'ποια αντικειμενα',
            'τι αντικειμενα',
            'τι υπαρχει μπροστα',
            'τι υπαρχει γυρω',
            'τι βλεπεισ',
        ))

    @staticmethod
    def is_detailed_vision_query(normalized):
        return any(phrase in normalized for phrase in (
            'περιγραψε',
            'περιγραφη της σκηνης',
            'αναλυτικα τι βλεπεις',
            'πες μου αναλυτικα',
            'τι ειναι αυτο',
            'τι γραφει',
            'διαβασε το',
            'τι χρωμα',
            'τι υπαρχει στη σκηνη',
        ))

    @staticmethod
    def is_face_query(normalized):
        return any(phrase in normalized for phrase in (
            'ποιος ειναι μπροστα',
            'ποια ειναι μπροστα',
            'ποιον βλεπεις',
            'ποια προσωπα',
            'αναγνωρισε το προσωπο',
            'ποιος ειναι απεναντι',
        ))

    @staticmethod
    def is_speaker_query(normalized):
        return any(phrase in normalized for phrase in (
            'ποιος μιλησε',
            'ποιος μιλησε τωρα',
            'ποια φωνη ακουσες',
            'ποια φωνη ηταν',
            'ποιος μιλαει',
            'ποιος μιλα',
        ))

    @staticmethod
    def is_follow_start(normalized):
        return any(phrase in normalized for phrase in (
            'ακολουθησε με',
            'ακολουθησε μεσα',
            'ελα μαζι μου',
            'follow me',
        ))

    @staticmethod
    def is_follow_stop(normalized):
        return any(phrase in normalized for phrase in (
            'σταματα να με ακολουθεις',
            'σταματησε να με ακολουθεις',
            'μη με ακολουθεις',
            'stop following',
        ))

    def request_follow_confirmation(self):
        self.request_motion_confirmation(
            {
                'action': 'follow_start',
                'confirmed': False,
                'source': 'voice_assistant',
            },
            'Να ξεκινήσω να σε ακολουθώ με χαμηλή ταχύτητα; Πες «ναι» για επιβεβαίωση.',
        )

    @staticmethod
    def local_motion_command(normalized):
        """Recognize the two deliberately limited voice motion commands."""
        if any(phrase in normalized for phrase in (
            'κανε μια βολτα',
            'κανε βολτα',
            'βολτα μεσα στο σπιτι',
            'περιπολια',
            'γυρο στο σπιτι',
        )):
            return {'action': 'patrol', 'rounds': 1}

        if not any(phrase in normalized for phrase in (
            'στροφη',
            'περιστροφη',
            'στριψε',
            'γυρνα επιτοπου',
            'γυρισε επιτοπου',
            'περιστρεψου',
        )):
            return None

        degrees = 360.0
        match = re.search(
            r'(-?\d+(?:[.,]\d+)?)\s*(?:μοιρ|degree|deg)',
            normalized,
        )
        if match:
            try:
                degrees = float(match.group(1).replace(',', '.'))
            except ValueError:
                degrees = 360.0
        if 'δεξια' in normalized:
            degrees = -abs(degrees)
        elif 'αριστερα' in normalized:
            degrees = abs(degrees)
        return {'action': 'rotate', 'degrees': degrees}

    @staticmethod
    def local_distance_command(normalized):
        """Recognize an explicit short forward/backward distance command.

        Keep this deterministic so «κάνε ένα μέτρο μπροστά» cannot be
        misclassified as room navigation or spend an LLM request.  The
        Dashboard performs the collision-checked Nav2 action after the
        normal confirmation gate.
        """
        normalized = normalize_text(normalized)
        forward = any(phrase in normalized for phrase in (
            'μπροστα',
            'εμπρος',
            'προς τα εμπρος',
            'προχωρα',
        ))
        backward = any(phrase in normalized for phrase in (
            'πισω',
            'οπισθεν',
            'προς τα πισω',
        ))
        if forward == backward:
            return None

        number_words = {
            'μηδεν': 0.0,
            'ενα': 1.0,
            'μια': 1.0,
            'μιας': 1.0,
            'εναν': 1.0,
            'δυο': 2.0,
            'τρεις': 3.0,
            'τρια': 3.0,
            'τεσσερα': 4.0,
            'τεσσερις': 4.0,
            'πεντε': 5.0,
            'μισο': 0.5,
        }
        word_alternatives = '|'.join(
            sorted(
                (re.escape(word) for word in number_words),
                key=len,
                reverse=True,
            )
        )
        number_pattern = rf'(?P<number>-?\d+(?:[.,]\d+)?|{word_alternatives})'
        unit_pattern = (
            r'(?:μετρ\w*|meter\w*|centimeter\w*|cm\b|m\b|'
            r'εκατοστ\w*)'
        )
        match = re.search(
            rf'{number_pattern}\s*(?P<unit>{unit_pattern})',
            normalized,
        )
        if match is None:
            return None
        raw_number = match.group('number')
        try:
            distance = float(raw_number.replace(',', '.'))
        except (AttributeError, TypeError, ValueError):
            distance = number_words.get(raw_number)
        if distance is None or not math.isfinite(distance):
            return None
        unit = match.group('unit')
        if unit.startswith(('cm', 'centimeter', 'εκατοστ')):
            distance /= 100.0
        if distance <= 0.0:
            return None
        if backward:
            distance = -distance
        return {'action': 'move_distance', 'distance_m': distance}

    def local_room_command(self, normalized):
        """Route unambiguous Greek room commands without spending LLM tokens."""
        if not any(marker in normalized for marker in (
            'πηγαινε', 'παμε', 'στειλε', 'μετακινησου', 'οδηγησε',
            'κατευθυνσου', 'προς', 'go to', 'navigate',
        )):
            return None
        rooms = sorted(self.known_rooms(), key=lambda item: len(normalize_text(item)), reverse=True)
        for room in rooms:
            if any(
                form and form in normalized
                for form in self.room_lookup_forms(room)
            ):
                return room
        return None

    @staticmethod
    def room_lookup_forms(value):
        """Return safe nominative/accusative forms for a saved room name."""
        normalized = normalize_text(value).strip()
        if not normalized:
            return set()
        forms = {normalized}
        if normalized.endswith('ς'):
            forms.add(normalized[:-1])
        elif normalized.endswith('ο'):
            forms.add(f'{normalized}ς')
        return forms

    def request_motion_confirmation(self, command, question):
        """Dispatch a motion command immediately.

        The user has explicitly chosen hands-free motion.  Keep the old
        method name because the validated motion helpers still call it, but
        do not ask for a second spoken confirmation.  Distance limits,
        emergency-stop checks, localization and Nav2 collision handling stay
        active in the Dashboard.
        """
        del question
        command = dict(command)
        command['confirmed'] = True
        command.setdefault('source', 'voice_assistant')
        self.pending_command = None
        self.pending_expires_at = 0.0
        self.publish_command(command)

    def request_patrol_confirmation(self, rounds=1):
        rooms = self.known_rooms()
        if len(rooms) < 2:
            self.publish_reply(
                'Χρειάζονται τουλάχιστον δύο αποθηκευμένα δωμάτια για βόλτα.',
                ok=False,
                action='patrol',
            )
            return
        try:
            rounds = int(rounds or 1)
        except (TypeError, ValueError):
            rounds = 1
        rounds = max(1, min(rounds, 3))
        room_list = ', '.join(rooms[:8])
        if len(rooms) > 8:
            room_list += ', …'
        self.request_motion_confirmation(
            {
                'action': 'patrol',
                'rounds': rounds,
                'confirmed': False,
                'source': 'voice_assistant',
            },
            f'Να κάνει βόλτα στα αποθηκευμένα δωμάτια ({room_list}); Πες «ναι» για επιβεβαίωση.',
        )

    def request_rotate_confirmation(self, degrees=360.0, direction=''):
        try:
            degrees = float(degrees if degrees is not None else 360.0)
        except (TypeError, ValueError):
            degrees = 360.0
        if not math.isfinite(degrees) or abs(degrees) < 1.0 or abs(degrees) > 360.0:
            self.publish_reply(
                'Η στροφή πρέπει να είναι από 1 έως 360 μοίρες.',
                ok=False,
                action='rotate',
            )
            return
        direction = normalize_text(direction)
        if direction in {'δεξια', 'right', 'clockwise', 'cw'}:
            degrees = -abs(degrees)
        elif direction in {'αριστερα', 'left', 'counterclockwise', 'ccw'}:
            degrees = abs(degrees)
        side = 'αριστερά' if degrees > 0 else 'δεξιά'
        self.request_motion_confirmation(
            {
                'action': 'rotate',
                'degrees': round(degrees, 1),
                'confirmed': False,
                'source': 'voice_assistant',
            },
            f'Να στρίψω {abs(degrees):g} μοίρες {side}; Πες «ναι» για επιβεβαίωση.',
        )

    def request_distance_confirmation(self, distance_m):
        try:
            distance_m = float(distance_m)
        except (TypeError, ValueError):
            self.publish_reply(
                'Δεν κατάλαβα την απόσταση. Πες, για παράδειγμα, «ένα μέτρο μπροστά».',
                ok=False,
                action='move_distance',
            )
            return
        if (
            not math.isfinite(distance_m)
            or abs(distance_m) < 0.05
            or abs(distance_m) > 3.0
        ):
            self.publish_reply(
                'Η χειροκίνητη κίνηση πρέπει να είναι από 5 εκατοστά έως 3 μέτρα.',
                ok=False,
                action='move_distance',
            )
            return
        direction = 'μπροστά' if distance_m > 0.0 else 'πίσω'
        unit = (
            'μέτρο'
            if math.isclose(abs(distance_m), 1.0, abs_tol=0.005)
            else 'μέτρα'
        )
        self.request_motion_confirmation(
            {
                'action': 'move_distance',
                'distance_m': round(distance_m, 3),
                'confirmed': False,
                'source': 'voice_assistant',
            },
            f'Να κινηθώ {abs(distance_m):g} {unit} {direction}; '
            'Πες «ναι» για επιβεβαίωση.',
        )

    def handle_confirmation(self, text):
        if self.is_positive(text):
            command = dict(self.pending_command)
            self.pending_command = None
            self.pending_expires_at = 0.0
            command['confirmed'] = True
            self.publish_reply(
                'Επιβεβαιώθηκε. Εκτελώ την εντολή.',
                action=command.get('action'),
            )
            self.publish_command(command)
        elif self.is_negative(text):
            self.pending_command = None
            self.pending_expires_at = 0.0
            self.publish_reply('Ακυρώθηκε.', ok=True)
        else:
            self.publish_reply(
                'Πες «ναι» για επιβεβαίωση ή «όχι» για ακύρωση.',
                ok=False,
            )

    def handle_command_text(self, command_text):
        command_text = ' '.join(str(command_text or '').split())
        if not command_text:
            self.publish_reply('Γράψε ή πες μου τι θέλεις.', ok=False)
            return

        normalized = normalize_text(command_text)
        system_control = self.local_system_control(normalized)
        if system_control is not None:
            system_control['confirmed'] = True
            system_control['source'] = 'voice_assistant'
            self.publish_command(system_control)
            return
        if any(phrase in normalized for phrase in ('σταματα', 'stop', 'ακυρωσε')):
            self.publish_command(
                {'action': 'stop', 'confirmed': True, 'source': 'voice_assistant'}
            )
            self.publish_reply('Σταματάω τώρα.', action='stop')
            return

        if self.is_speaker_query(normalized):
            self.publish_command(
                {
                    'action': 'speaker_query',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return

        if self.is_face_query(normalized):
            self.publish_command(
                {
                    'action': 'face_query',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return

        if self.is_follow_start(normalized):
            self.request_follow_confirmation()
            return

        distance = self.local_distance_command(normalized)
        if distance is not None:
            self.request_distance_confirmation(distance['distance_m'])
            return

        room = self.local_room_command(normalized)
        if room is not None:
            # The common «πήγαινε στην κουζίνα» path is deterministic and does
            # not wait for Gemini/Qwen to guess an intent or room name.
            self.handle_intent({'intent': 'navigate_room', 'room': room})
            return

        if self.is_detailed_vision_query(normalized):
            self.publish_command(
                {
                    'action': 'vision_question',
                    'question': command_text,
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return

        target = self.local_system_target(normalized)
        if target is not None:
            self.publish_command(
                {
                    'action': 'system_info',
                    'target': target,
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return

        if self.is_vision_query(normalized):
            self.publish_command(
                {
                    'action': 'vision',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return

        motion = self.local_motion_command(normalized)
        if motion is not None:
            if motion['action'] == 'patrol':
                self.request_patrol_confirmation(motion.get('rounds', 1))
            else:
                self.request_rotate_confirmation(motion.get('degrees', 360.0))
            return

        # The local clock is a deterministic, always-available capability.
        # Handle it before the LLM so a simple time question never depends on
        # network access or model availability.
        if any(phrase in normalized for phrase in (
            'τι ωρα',
            'ποια ωρα',
            'τι ημερομηνια',
            'ποια ημερομηνια',
            'τι μερα ειναι',
        )):
            current = datetime.now().astimezone()
            self.publish_reply(
                f'Είναι {current:%H:%M}, {current:%d/%m/%Y}.',
                action='time',
            )
            return

        if not self.submit_work('llm', self.ask_llm, command_text):
            self.publish_status('busy', 'Περίμενε να ολοκληρωθεί η προηγούμενη εντολή')
            self.publish_reply(
                'Περίμενε να ολοκληρωθεί η προηγούμενη εντολή.',
                ok=False,
            )
        else:
            self.publish_status('thinking', f'Ερμηνεύω: «{command_text}»')

    def handle_transcript(self, text, detector_authorized=None):
        text = ' '.join(str(text or '').split())
        if detector_authorized is None:
            detector_authorized = self.last_segment_wake_authorized
        detector_authorized = bool(detector_authorized)
        if text and self.is_stt_hallucination(text):
            words = re.findall(
                r'[^\W_]+', normalize_text(text), flags=re.UNICODE
            )
            def wake_variant(word):
                return normalize_text(word).translate(str.maketrans({
                    'a': 'α', 'l': 'λ', 'e': 'ε', 'x': 'ξ',
                    'k': 'κ', 'h': 'χ',
                }))
            detector_confirmed_wake_only = (
                detector_authorized
                and bool(words)
                and all(
                    wake_variant(word) in {'αλεξα', 'αλεχα'}
                    for word in words
                )
            )
            if detector_confirmed_wake_only:
                # The dedicated detector already proved this was a real wake.
                # Whisper commonly repeats a prompted one-word transcript;
                # treat it as a wake-only segment and keep the command window.
                text = self.wake_model_name
            else:
                # Do not publish known Whisper subtitle hallucinations to the
                # Dashboard and never let them reach the LLM or command parser.
                self.last_transcript = ''
                self.last_segment_wake_authorized = False
                self.publish_status(
                    'wake_rejected',
                    'Δεν αναγνώρισα «Alexa» — αγνόησα θόρυβο ή ηχείο',
                )
                self.get_logger().warning(
                    'Απορρίφθηκε γνωστή hallucination του STT πριν εμφανιστεί στο Chat.'
                )
                return
        if not text:
            self.last_segment_wake_authorized = False
            self.publish_status(
                'wake_rejected',
                'Ξύπνησα, αλλά δεν άκουσα καθαρά την εντολή'
                if self.wake_model is not None
                else 'Δεν αναγνωρίστηκε ομιλία',
            )
            return

        if self.pending_command is not None:
            self.handle_confirmation(text)
            return

        command_text = self.strip_wake_word(text)
        wake_authorized = (
            command_text is not None
            or detector_authorized
        )
        self.last_segment_wake_authorized = False
        if command_text is None and not wake_authorized:
            # Listening is continuous, but the LLM is only contacted after
            # the explicit wake word to avoid sending nearby conversations.
            # Do not publish this rejected background speech as a Dashboard
            # chat message either.
            self.publish_status(
                'wake_rejected',
                f'Άκουσα φωνή — δεν αναγνωρίστηκε «{self.wake_model_name}»',
            )
            return
        if command_text is not None and not detector_authorized:
            # The CPU Whisper fallback recognized the wake word even though
            # openWakeWord did not.  Give the same audible acknowledgement
            # and command window as the dedicated detector.
            self.activate_wake(0.0)
        if command_text is None:
            # The dedicated detector already heard the wake word. Whisper may omit
            # the short wake word, so the rest of the utterance is still a
            # valid command.
            command_text = text
        if not command_text:
            # Whisper fallback may recognize only the wake word (sometimes
            # repeated several times).  It is a control signal, not a user
            # chat message.  A wake session gets one acknowledgement only.
            if not self.wake_ack_sent:
                self.wake_ack_sent = True
                self.publish_reply('Σε ακούω. Πες μου τι θέλεις.', ok=True)
            return

        # Publish only an authorized command/question.  Wake-only checks and
        # ordinary room speech must not appear as user messages in the Chat.
        self.last_transcript = text
        self.transcript_pub.publish(String(data=text))
        self.handle_command_text(command_text)

    def transcript_input(self, msg):
        """Feed a simulator/diagnostic transcript through the real STT path."""
        self.handle_transcript(msg.data)

    def text_command(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            payload = {'text': msg.data}
        if not isinstance(payload, dict):
            self.publish_reply('Το κείμενο της εντολής δεν είναι έγκυρο.', ok=False)
            return

        text = ' '.join(str(payload.get('text', '') or '').split())
        self.last_transcript = text
        self.transcript_pub.publish(String(data=text))
        if not text:
            self.publish_reply('Γράψε πρώτα μια ερώτηση ή εντολή.', ok=False)
            return
        if self.pending_command is not None:
            self.handle_confirmation(text)
            return

        # Text commands are already explicit, so they do not require the
        # spoken wake word.  If the user includes it anyway, remove it.
        command_text = self.strip_wake_word(text) or text
        self.handle_command_text(command_text)

    def known_rooms(self):
        try:
            values = json.loads(self.rooms_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []
        if not isinstance(values, list):
            return []
        return [
            str(item.get('name', '')).strip()[:48]
            for item in values
            if isinstance(item, dict) and str(item.get('name', '')).strip()
        ][:100]

    def llm_system_prompt(self, rooms):
        return (
            'Είσαι ο ασφαλής βοηθός/intent router για Clearpath Dingo. '
            'Για εντολές συστήματος ή ρομπότ χρησιμοποίησε το κατάλληλο native tool '
            'από αυτά που σου δίνονται. Για απλή γενική ερώτηση απάντησε σύντομα στα ελληνικά. '
            'Αν ο provider δεν υποστηρίζει native tools, επέστρεψε μόνο JSON intent, χωρίς markdown. '
            'Intents: stop,status,battery,where,navigate_room,move_distance,system_info,system_control,patrol,rotate,vision,'
            'vision_question,face_query,speaker_query,follow_start,follow_stop,time,answer,unknown. '
            'Κανόνες: navigate_room=μόνο room από τη λίστα· system_info target=disk,memory,cpu,'
            'temperature,power,network,services,processes,devices,sensors,voice,vision,gemini,ros ή summary· '
            'system_control είναι η μόνη πρόσβαση ελέγχου συστήματος και έχει target=service,camera,mapping,'
            'navigation,localization,rviz ή map και operation=status,start,stop,restart,cancel,global_localization,save. '
            'Για target=service χρησιμοποίησε μόνο ένα από τα allow-listed service names: '
            'dingo-voice.service,dingo-local-llm.service,dingo-dashboard.service,dingo-sensors.service,'
            'dingo-object-detector.service,dingo-face-recognition.service,dingo-mic-array.service. '
            'Ποτέ μην επινοείς service/path/command. Το system_control δεν είναι shell και δεν δέχεται '
            'ROS topics, αρχεία, συντεταγμένες, reboot ή shutdown. '
            'move_distance έχει distance_m σε μέτρα '
            '(θετικό μπροστά, αρνητικό πίσω) και θέλει επιβεβαίωση· '
            'patrol/rotate/follow_start θέλουν επιβεβαίωση· '
            'rotate έχει degrees 1-360 και direction left/right· vision_question έχει question· '
            'answer είναι σύντομη γενική απάντηση στα ελληνικά· time για ώρα/ημερομηνία. '
            'Για live intents μην επινοείς μετρήσεις. Όλα τα κείμενα σύντομα, '
            'σωστά ελληνικά με τόνους. Ποτέ shell commands, ROS topics ή συντεταγμένες. '
            'Schema: {"intent":"...","room":"","target":"","operation":"status",'
            '"service":"","map_name":"","reply":"","question":"",'
            '"degrees":360,"direction":"","rounds":1,"distance_m":1}. '
            f'Δωμάτια: {json.dumps(rooms, ensure_ascii=False)}'
        )

    def read_gemini_api_key(self):
        key = os.environ.get('GEMINI_API_KEY', '').strip()
        if not key:
            try:
                key = self.gemini_api_key_file.read_text(encoding='utf-8').strip()
            except (FileNotFoundError, OSError):
                key = ''
        if not key:
            raise RuntimeError(
                'Λείπει το Gemini API key. Βάλ’ το στο '
                f'{self.gemini_api_key_file} (μόνο το key, με δικαιώματα 600).'
            )
        return key

    @staticmethod
    def parse_json_response(content):
        if not isinstance(content, str):
            raise RuntimeError('Το LLM επέστρεψε μη έγκυρο περιεχόμενο')
        content = content.strip()
        if content.startswith('```'):
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError('Το LLM δεν επέστρεψε έγκυρο JSON intent') from exc

    @classmethod
    def native_tool_specs(cls):
        """Return the provider-neutral, allow-listed function definitions."""
        return [
            {
                'name': 'stop_robot',
                'description': 'Σταμάτησε αμέσως το Dingo.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'get_status',
                'description': 'Δώσε την τρέχουσα κατάσταση του Dingo.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'get_battery',
                'description': 'Δώσε μπαταρία, τάση και κατανάλωση όταν είναι διαθέσιμες.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'get_location',
                'description': 'Δώσε τη γνωστή θέση του Dingo στον χάρτη.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'navigate_to_room',
                'description': 'Πήγαινε σε δωμάτιο που υπάρχει στη λίστα χαρτογράφησης.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'room': {'type': 'string', 'description': 'Το όνομα του δωματίου.'},
                    },
                    'required': ['room'],
                },
            },
            {
                'name': 'drive_distance',
                'description': (
                    'Κινήσου σε ευθεία για συγκεκριμένη απόσταση. '
                    'Θετική απόσταση σημαίνει μπροστά και αρνητική πίσω. '
                    'Θα ζητηθεί επιβεβαίωση πριν κινηθεί.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'distance_m': {
                            'type': 'number',
                            'minimum': -3,
                            'maximum': 3,
                            'description': 'Απόσταση σε μέτρα: θετική μπροστά, αρνητική πίσω.',
                        },
                    },
                    'required': ['distance_m'],
                },
            },
            {
                'name': 'get_system_info',
                'description': 'Δώσε μετρήσεις του υπολογιστή και του ρομπότ.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'target': {
                            'type': 'string',
                            'enum': [
                                'disk', 'memory', 'cpu', 'temperature',
                                'power', 'network', 'services', 'processes',
                                'devices', 'sensors', 'voice', 'vision',
                                'gemini', 'ros', 'summary', 'all',
                            ],
                        },
                    },
                    'required': ['target'],
                },
            },
            {
                'name': 'control_system',
                'description': (
                    'Έλεγξε ένα επιτρεπόμενο υποσύστημα του Dingo. '
                    'Δεν εκτελεί shell, αρχεία ή αυθαίρετες εντολές. '
                    'Οι κινήσεις και το localization κρατούν τους ελέγχους ασφαλείας.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'target': {
                            'type': 'string',
                            'enum': [
                                'service', 'camera', 'mapping', 'navigation',
                                'localization', 'rviz', 'map',
                            ],
                        },
                        'operation': {
                            'type': 'string',
                            'enum': [
                                'status', 'start', 'stop', 'restart',
                                'cancel', 'global_localization', 'save',
                            ],
                        },
                        'service': {
                            'type': 'string',
                            'description': 'Μόνο allow-listed όνομα υπηρεσίας, όταν target=service.',
                        },
                        'map_name': {
                            'type': 'string',
                            'description': 'Όνομα αποθηκευμένου χάρτη, ποτέ διαδρομή αρχείου.',
                        },
                    },
                    'required': ['target', 'operation'],
                },
            },
            {
                'name': 'start_patrol',
                'description': 'Κάνε βόλτα/περιπολία. Θα ζητηθεί επιβεβαίωση πριν κινηθεί.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'rounds': {
                            'type': 'integer',
                            'minimum': 1,
                            'maximum': 3,
                            'description': 'Πλήθος γύρων, από 1 έως 3.',
                        },
                    },
                },
            },
            {
                'name': 'rotate_in_place',
                'description': 'Κάνε στροφή επιτόπου. Θα ζητηθεί επιβεβαίωση.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'degrees': {
                            'type': 'number',
                            'minimum': 1,
                            'maximum': 360,
                            'description': 'Γωνία στροφής σε μοίρες.',
                        },
                        'direction': {
                            'type': 'string',
                            'enum': ['left', 'right'],
                        },
                    },
                    'required': ['degrees', 'direction'],
                },
            },
            {
                'name': 'describe_scene',
                'description': 'Περιέγραψε τι βλέπει η κάμερα.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'ask_about_camera',
                'description': 'Απάντησε σε ερώτηση για τη ζωντανή εικόνα της κάμερας.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'question': {'type': 'string'},
                    },
                    'required': ['question'],
                },
            },
            {
                'name': 'recognize_face',
                'description': 'Αναγνώρισε το πρόσωπο που βλέπει η κάμερα.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'identify_speaker',
                'description': 'Αναγνώρισε ποιος μίλησε, όταν υπάρχει εκπαιδευμένο voice profile.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'start_follow_me',
                'description': 'Ξεκίνα λειτουργία ακολούθησέ με. Θα ζητηθεί επιβεβαίωση.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'stop_follow_me',
                'description': 'Σταμάτησε τη λειτουργία ακολούθησέ με.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'get_time',
                'description': 'Δώσε την τοπική ώρα και ημερομηνία του υπολογιστή.',
                'parameters': {'type': 'object', 'properties': {}},
            },
        ]

    @classmethod
    def gemini_tool_definitions(cls):
        def gemini_schema(schema):
            if not isinstance(schema, dict):
                return schema
            converted = {}
            for key, value in schema.items():
                if key == 'type' and isinstance(value, str):
                    converted[key] = value.upper()
                elif isinstance(value, dict):
                    converted[key] = gemini_schema(value)
                elif isinstance(value, list):
                    converted[key] = [gemini_schema(item) for item in value]
                else:
                    converted[key] = value
            return converted

        return [
            {
                'name': spec['name'],
                'description': spec['description'],
                'parameters': gemini_schema(spec['parameters']),
            }
            for spec in cls.native_tool_specs()
        ]

    @classmethod
    def openai_tool_definitions(cls):
        return [
            {
                'type': 'function',
                'function': {
                    'name': spec['name'],
                    'description': spec['description'],
                    'parameters': spec['parameters'],
                },
            }
            for spec in cls.native_tool_specs()
        ]

    @classmethod
    def tool_call_to_intent(cls, name, arguments=None):
        """Convert a provider function call to the existing safe intent shape."""
        intent = cls.NATIVE_TOOL_INTENTS.get(str(name or '').strip())
        if not intent:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        result = {'intent': intent, 'reply': ''}
        if intent == 'navigate_room':
            result['room'] = str(arguments.get('room', '') or '')
        elif intent == 'move_distance':
            result['distance_m'] = arguments.get(
                'distance_m', arguments.get('distance')
            )
        elif intent == 'system_info':
            result['target'] = str(arguments.get('target', 'summary') or 'summary')
        elif intent == 'system_control':
            result['target'] = str(arguments.get('target', '') or '')
            result['operation'] = str(arguments.get('operation', 'status') or 'status')
            result['service'] = str(arguments.get('service', '') or '')
            result['map_name'] = str(arguments.get('map_name', '') or '')
        elif intent == 'patrol':
            result['rounds'] = arguments.get('rounds', 1)
        elif intent == 'rotate':
            result['degrees'] = arguments.get('degrees', 360)
            direction = normalize_text(arguments.get('direction', 'left')).strip()
            result['direction'] = direction or 'left'
            if direction.startswith('right'):
                try:
                    result['degrees'] = -abs(float(result['degrees']))
                except (TypeError, ValueError):
                    result['degrees'] = -360.0
        elif intent == 'vision_question':
            result['question'] = str(arguments.get('question', '') or '')
        return result

    @classmethod
    def gemini_function_call_to_intent(cls, parts):
        for part in parts or []:
            if not isinstance(part, dict):
                continue
            function_call = part.get('functionCall')
            if not isinstance(function_call, dict):
                continue
            result = cls.tool_call_to_intent(
                function_call.get('name'), function_call.get('args')
            )
            if result is not None:
                return result
        return None

    @classmethod
    def openai_tool_call_to_intent(cls, message):
        if not isinstance(message, dict):
            return None
        for tool_call in message.get('tool_calls') or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get('function') or tool_call
            if not isinstance(function, dict):
                continue
            result = cls.tool_call_to_intent(
                function.get('name'),
                function.get('arguments', function.get('args')),
            )
            if result is not None:
                return result
        return None

    @classmethod
    def parse_model_content(cls, content):
        """Accept legacy JSON fallback or a normal Greek answer."""
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('Το LLM δεν επέστρεψε απάντηση')
        try:
            parsed = cls.parse_json_response(content)
        except RuntimeError:
            return {
                'intent': 'answer',
                'reply': content.strip()[:240],
            }
        return parsed

    def ask_llm(self, command_text):
        rooms = self.known_rooms()
        system_prompt = self.llm_system_prompt(rooms)
        # Hold the configuration lock for the complete request.  A Dashboard
        # switch therefore waits for the current answer instead of mixing a
        # provider with another provider's URL/model halfway through a call.
        with self.llm_config_lock:
            if self.llm_provider == 'gemini':
                try:
                    return self.ask_gemini(command_text, system_prompt)
                except Exception as exc:  # noqa: BLE001
                    # Gemini quota/rate-limit, API-key, network and transient
                    # cloud failures must not turn a valid Dingo command into
                    # a user-visible error. Retry the exact same request with
                    # the local Qwen 9B endpoint instead.
                    self.llm_provider = 'flm'
                    self.llm_url = self.local_llm_url
                    self.llm_model = self.local_llm_model
                    self.llm_fallback_active = True
                    self.llm_fallback_reason = type(exc).__name__
                    self.get_logger().warning(
                        'Το Gemini απέτυχε· συνεχίζω αυτόματα με τοπικό '
                        f'Qwen {self.local_llm_model}: {exc}'
                    )
                    self.publish_status(
                        self.status_state or 'thinking',
                        'Το Gemini δεν είναι διαθέσιμο. Συνεχίζω αυτόματα με '
                        f'τοπικό Qwen {self.local_llm_model}.',
                        force=True,
                        log=False,
                    )
                    return self.ask_flm(command_text, system_prompt)
            if self.llm_provider == 'flm':
                return self.ask_flm(command_text, system_prompt)
            return self.ask_ollama(command_text, system_prompt)

    def publish_gemini_usage(self, body, request_type='assistant', model=None):
        usage = body.get('usageMetadata') if isinstance(body, dict) else None
        if not isinstance(usage, dict) or not usage:
            interactions_usage = body.get('usage') if isinstance(body, dict) else None
            if isinstance(interactions_usage, dict):
                # The Interactions API uses snake_case counters, while the
                # Dashboard already consumes the generateContent names.
                usage = {
                    'promptTokenCount': interactions_usage.get('total_input_tokens'),
                    'candidatesTokenCount': interactions_usage.get('total_output_tokens'),
                    'thoughtsTokenCount': interactions_usage.get('total_thought_tokens'),
                    'totalTokenCount': interactions_usage.get('total_tokens'),
                }
        if not isinstance(usage, dict) or not usage:
            return
        payload = {
            'source': 'voice_assistant',
            'model': model or self.llm_model,
            'request_type': request_type,
            # Usage metadata contains counts only; never publish the API key
            # or the user's command text on the telemetry topic.
            'usage': {
                key: usage.get(key)
                for key in (
                    'promptTokenCount',
                    'candidatesTokenCount',
                    'thoughtsTokenCount',
                    'totalTokenCount',
                )
                if usage.get(key) is not None
            },
        }
        self.gemini_usage_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

    def ask_gemini(self, command_text, system_prompt):
        api_key = self.read_gemini_api_key()
        schema = {
            'type': 'object',
            'properties': {
                'intent': {
                    'type': 'string',
                    'enum': sorted(self.ALLOWED_INTENTS),
                },
                'room': {'type': 'string'},
                'target': {'type': 'string'},
                'operation': {'type': 'string'},
                'service': {'type': 'string'},
                'map_name': {'type': 'string'},
                'reply': {'type': 'string'},
                'question': {'type': 'string'},
                'degrees': {'type': 'number'},
                'direction': {'type': 'string'},
                'rounds': {'type': 'integer'},
                'distance_m': {'type': 'number'},
            },
            'required': ['intent', 'room', 'reply'],
        }
        legacy_payload = {
            'systemInstruction': {
                'parts': [{'text': system_prompt}],
            },
            'contents': [{
                'role': 'user',
                'parts': [{'text': command_text}],
            }],
            'generationConfig': {
                'temperature': 0,
                'maxOutputTokens': 192,
                'thinkingConfig': {'thinkingBudget': 0},
                'responseMimeType': 'application/json',
                'responseSchema': schema,
            },
        }
        if self.native_tool_calling:
            payload = {
                'systemInstruction': {
                    'parts': [{'text': system_prompt}],
                },
                'contents': [{
                    'role': 'user',
                    'parts': [{'text': command_text}],
                }],
                'tools': [{
                    'functionDeclarations': self.gemini_tool_definitions(),
                }],
                'toolConfig': {
                    'functionCallingConfig': {'mode': 'AUTO'},
                },
                'generationConfig': {
                    'temperature': 0,
                    'maxOutputTokens': 192,
                    'thinkingConfig': {'thinkingBudget': 0},
                },
            }
        else:
            payload = legacy_payload
        endpoint = (
            'https://generativelanguage.googleapis.com/v1beta/models/'
            f'{quote(self.llm_model, safe="")}:generateContent'
        )

        def post(request_payload):
            request = Request(
                endpoint,
                data=json.dumps(request_payload, ensure_ascii=False).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'x-goog-api-key': api_key,
                },
                method='POST',
            )
            try:
                with urlopen(request, timeout=self.llm_timeout_s) as response:
                    return json.loads(response.read().decode('utf-8'))
            except HTTPError:
                raise
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    f'Το Gemini δεν είναι διαθέσιμο ({self.llm_model}): {exc}'
                ) from exc

        try:
            body = post(payload)
        except HTTPError as exc:
            # Some OpenAI-compatible/local providers and older Gemini models
            # reject the tools field. Keep the service usable with the old
            # structured-JSON contract in that case.
            if not self.native_tool_calling or exc.code not in {400, 404, 422, 501}:
                raise RuntimeError(
                    f'Το Gemini δεν είναι διαθέσιμο ({self.llm_model}, HTTP {exc.code})'
                ) from exc
            self.get_logger().warning(
                f'Το Gemini απέρριψε native tools (HTTP {exc.code})· '
                'χρησιμοποιώ προσωρινά JSON fallback.'
            )
            try:
                body = post(legacy_payload)
            except HTTPError as legacy_exc:
                raise RuntimeError(
                    f'Το Gemini δεν είναι διαθέσιμο '
                    f'({self.llm_model}, HTTP {legacy_exc.code})'
                ) from legacy_exc

        self.publish_gemini_usage(body, request_type='voice_command')
        candidates = body.get('candidates') or []
        if not candidates:
            raise RuntimeError('Το Gemini δεν επέστρεψε υποψήφια απάντηση')
        parts = candidates[0].get('content', {}).get('parts', [])
        native_intent = self.gemini_function_call_to_intent(parts)
        if native_intent is not None:
            return native_intent
        content = next(
            (part.get('text') for part in parts if isinstance(part, dict) and part.get('text')),
            '',
        )
        return self.parse_model_content(content)

    def ask_ollama(self, command_text, system_prompt):
        legacy_payload = {
            'model': self.llm_model,
            'stream': False,
            'format': 'json',
            'options': {'temperature': 0},
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': command_text},
            ],
        }
        payload = legacy_payload
        if self.native_tool_calling:
            payload = {
                'model': self.llm_model,
                'stream': False,
                'messages': legacy_payload['messages'],
                'tools': self.openai_tool_definitions(),
            }

        def post(request_payload):
            request = Request(
                self.llm_url,
                data=json.dumps(request_payload, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            try:
                with urlopen(request, timeout=self.llm_timeout_s) as response:
                    return json.loads(response.read().decode('utf-8'))
            except HTTPError:
                raise
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    f'Το Ollama δεν είναι διαθέσιμο ({self.llm_model}): {exc}'
                ) from exc

        try:
            body = post(payload)
        except HTTPError as exc:
            if not self.native_tool_calling or exc.code not in {400, 404, 422, 501}:
                raise RuntimeError(
                    f'Το Ollama δεν είναι διαθέσιμο ({self.llm_model}, HTTP {exc.code})'
                ) from exc
            self.get_logger().warning(
                f'Το Ollama απέρριψε native tools (HTTP {exc.code})· '
                'χρησιμοποιώ JSON fallback.'
            )
            try:
                body = post(legacy_payload)
            except HTTPError as legacy_exc:
                raise RuntimeError(
                    f'Το Ollama δεν είναι διαθέσιμο '
                    f'({self.llm_model}, HTTP {legacy_exc.code})'
                ) from legacy_exc

        message = body.get('message', {}) if isinstance(body, dict) else {}
        native_intent = self.openai_tool_call_to_intent(message)
        if native_intent is not None:
            return native_intent
        content = message.get('content', '') if isinstance(message, dict) else ''
        return self.parse_model_content(content)

    def ask_flm(self, command_text, system_prompt):
        """Ask the local FastFlowLM OpenAI-compatible endpoint."""
        legacy_payload = {
            'model': self.llm_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': command_text},
            ],
            'stream': False,
            'temperature': 0,
            'max_tokens': 384,
        }
        payload = legacy_payload
        if self.native_tool_calling:
            payload = dict(legacy_payload)
            payload['tools'] = self.openai_tool_definitions()

        def post(request_payload):
            request = Request(
                self.llm_url,
                data=json.dumps(request_payload, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            try:
                with urlopen(request, timeout=self.llm_timeout_s) as response:
                    return json.loads(response.read().decode('utf-8'))
            except HTTPError:
                raise
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    f'Το τοπικό LLM δεν είναι διαθέσιμο ({self.llm_model}): {exc}'
                ) from exc

        try:
            body = post(payload)
        except HTTPError as exc:
            if not self.native_tool_calling or exc.code not in {400, 404, 422, 501}:
                raise RuntimeError(
                    f'Το τοπικό LLM δεν είναι διαθέσιμο '
                    f'({self.llm_model}, HTTP {exc.code})'
                ) from exc
            self.get_logger().warning(
                f'Το FastFlowLM απέρριψε native tools (HTTP {exc.code})· '
                'χρησιμοποιώ JSON fallback.'
            )
            try:
                body = post(legacy_payload)
            except HTTPError as legacy_exc:
                raise RuntimeError(
                    f'Το τοπικό LLM δεν είναι διαθέσιμο '
                    f'({self.llm_model}, HTTP {legacy_exc.code})'
                ) from legacy_exc

        choices = body.get('choices') if isinstance(body, dict) else None
        message = choices[0].get('message') if choices else None
        native_intent = self.openai_tool_call_to_intent(message)
        if native_intent is not None:
            return native_intent
        content = message.get('content', '') if isinstance(message, dict) else ''
        if isinstance(content, list):
            content = ''.join(
                item.get('text', '')
                for item in content
                if isinstance(item, dict) and item.get('text')
            )
        return self.parse_model_content(content)

    def validate_intent(self, value):
        if not isinstance(value, dict):
            return {
                'intent': 'unknown',
                'room': '',
                'target': '',
                'operation': '',
                'service': '',
                'map_name': '',
                'reply': '',
                'question': '',
                'degrees': None,
                'direction': '',
                'rounds': 1,
                'distance_m': None,
            }
        intent = normalize_text(value.get('intent', '')).replace(' ', '_')
        intent = self.INTENT_ALIASES.get(intent, intent)
        if intent not in self.ALLOWED_INTENTS:
            intent = 'unknown'
        room = str(value.get('room', '') or '').strip()[:48]
        raw_target = value.get('target', '')
        target = normalize_text(raw_target).strip()[:32]
        operation = normalize_text(value.get('operation', '')).strip()[:32]
        service = str(value.get('service', '') or '').strip()[:80]
        map_name = str(value.get('map_name', '') or '').strip()[:60]
        reply = str(value.get('reply', '') or '').strip()[:240]
        question = str(value.get('question', '') or '').strip()[:500]
        try:
            degrees = float(value.get('degrees'))
        except (TypeError, ValueError):
            degrees = None
        if degrees is not None and not math.isfinite(degrees):
            degrees = None
        # Small local models occasionally place the turn amount in target
        # despite the schema. Recover it safely instead of silently turning
        # a requested 90° rotation into the default 360° rotation.
        if degrees is None and intent == 'rotate':
            try:
                candidate = float(str(raw_target).replace(',', '.'))
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and math.isfinite(candidate):
                degrees = candidate
        try:
            rounds = int(value.get('rounds') or 1)
        except (TypeError, ValueError):
            rounds = 1
        try:
            distance_m = float(value.get('distance_m'))
        except (TypeError, ValueError):
            distance_m = None
        if distance_m is not None and not math.isfinite(distance_m):
            distance_m = None
        return {
            'intent': intent,
            'room': room,
            'target': target,
            'operation': operation,
            'service': service,
            'map_name': map_name,
            'reply': reply,
            'question': question,
            'degrees': degrees,
            'direction': normalize_text(value.get('direction', '')).strip()[:24],
            'rounds': max(1, min(rounds, 3)),
            'distance_m': distance_m,
        }

    def resolve_room(self, requested):
        requested_forms = self.room_lookup_forms(requested)
        matches = [
            room for room in self.known_rooms()
            if requested_forms.intersection(self.room_lookup_forms(room))
        ]
        return matches[0] if len(matches) == 1 else None

    def handle_intent(self, value):
        intent = self.validate_intent(value)
        action = intent['intent']
        if action == 'stop':
            self.publish_command(
                {'action': 'stop', 'confirmed': True, 'source': 'voice_assistant'}
            )
            self.publish_reply('Σταματάω τώρα.', action='stop')
            return
        if action in {'status', 'battery', 'where'}:
            self.publish_command(
                {'action': action, 'confirmed': True, 'source': 'voice_assistant'}
            )
            return
        if action == 'system_info':
            self.publish_command(
                {
                    'action': action,
                    'target': intent['target'] or 'summary',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action == 'system_control':
            self.publish_command(
                {
                    'action': action,
                    'target': intent['target'],
                    'operation': intent['operation'] or 'status',
                    'service': intent['service'],
                    'map_name': intent['map_name'],
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action == 'vision':
            self.publish_command(
                {
                    'action': action,
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action == 'vision_question':
            self.publish_command(
                {
                    'action': action,
                    'question': intent['question'] or intent['reply'] or 'Περιέγραψε την εικόνα.',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action in {'face_query', 'speaker_query'}:
            self.publish_command(
                {
                    'action': action,
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action == 'follow_start':
            self.request_follow_confirmation()
            return
        if action == 'follow_stop':
            self.publish_command(
                {
                    'action': 'follow_stop',
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        if action == 'answer':
            self.publish_reply(
                intent['reply'] or 'Δεν έχω διαθέσιμη απάντηση για αυτό.',
                ok=True,
                action=action,
            )
            return
        if action == 'time':
            current = datetime.now().astimezone()
            self.publish_reply(
                f'Είναι {current:%H:%M}, {current:%d/%m/%Y}.',
                ok=True,
                action=action,
            )
            return
        if action == 'move_distance':
            self.request_distance_confirmation(intent.get('distance_m'))
            return
        if action == 'patrol':
            self.request_patrol_confirmation(intent.get('rounds', 1))
            return
        if action == 'rotate':
            self.request_rotate_confirmation(
                intent.get('degrees') or 360.0,
                intent.get('direction', ''),
            )
            return
        if action == 'navigate_room':
            room = self.resolve_room(intent['room'])
            if room is None:
                self.publish_reply(
                    f'Δεν βρήκα αποθηκευμένο δωμάτιο «{intent["room"]}».',
                    ok=False,
                    action=action,
                )
                return
            self.publish_command(
                {
                    'action': 'navigate_room',
                    'room': room,
                    'confirmed': True,
                    'source': 'voice_assistant',
                }
            )
            return
        self.publish_reply(
            intent['reply'] or f'Δεν κατάλαβα την εντολή. Δοκίμασε ξανά με «{self.wake_model_name}».',
            ok=False,
            action='unknown',
        )

    def publish_command(self, command):
        allowed = {
            'stop',
            'status',
            'battery',
            'where',
            'navigate_room',
            'move_distance',
            'system_info',
            'system_control',
            'patrol',
            'rotate',
            'vision',
            'vision_question',
            'face_query',
            'speaker_query',
            'follow_start',
            'follow_stop',
        }
        action = str(command.get('action', ''))
        if action not in allowed:
            self.get_logger().warning(f'Απορρίφθηκε άγνωστη voice action: {action}')
            return
        if action in {'patrol', 'rotate', 'follow_start', 'move_distance'} and command.get('confirmed') is not True:
            self.get_logger().warning('Απορρίφθηκε εντολή κίνησης χωρίς επιβεβαίωση')
            return
        payload = {
            'action': action,
            'room': str(command.get('room', '') or '')[:48],
            'target': normalize_text(command.get('target', '')).strip()[:32],
            'operation': normalize_text(command.get('operation', '')).strip()[:32],
            'service': str(command.get('service', '') or '').strip()[:80],
            'map_name': str(command.get('map_name', '') or '').strip()[:60],
            'question': str(command.get('question', '') or '')[:500],
            'degrees': command.get('degrees'),
            'rounds': command.get('rounds', 1),
            'distance_m': command.get('distance_m'),
            'confirmed': bool(command.get('confirmed', False)),
            'source': 'voice_assistant',
        }
        if action == 'rotate':
            try:
                degrees = float(payload['degrees'])
            except (TypeError, ValueError):
                self.get_logger().warning('Απορρίφθηκε rotate χωρίς έγκυρες μοίρες')
                return
            if not math.isfinite(degrees) or abs(degrees) < 1.0 or abs(degrees) > 360.0:
                self.get_logger().warning('Απορρίφθηκε rotate εκτός ορίου 1–360 μοιρών')
                return
            payload['degrees'] = round(degrees, 1)
        if action == 'patrol':
            try:
                payload['rounds'] = max(1, min(3, int(payload['rounds'] or 1)))
            except (TypeError, ValueError):
                payload['rounds'] = 1
        if action == 'move_distance':
            try:
                distance_m = float(payload['distance_m'])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    'Απορρίφθηκε move_distance χωρίς έγκυρη απόσταση'
                )
                return
            if (
                not math.isfinite(distance_m)
                or abs(distance_m) < 0.05
                or abs(distance_m) > 3.0
            ):
                self.get_logger().warning(
                    'Απορρίφθηκε move_distance εκτός ορίου 0.05–3 μέτρων'
                )
                return
            payload['distance_m'] = round(distance_m, 3)
        self.command_pub.publish(String(data=compact_json(payload)))

    def close(self):
        self.stream_reader_stop.set()
        with self.stream_lock:
            stream = self.stream
            self.stream = None
            blocking_reader = self.stream_is_blocking_reader
            self.stream_is_blocking_reader = False
        if stream is not None:
            try:
                if blocking_reader and hasattr(stream, 'abort'):
                    # PortAudio may be blocked inside stream.read().  abort()
                    # wakes that read immediately; close() alone can leave
                    # the service stuck until systemd's stop timeout.
                    stream.abort()
                else:
                    stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        reader = self.stream_reader_thread
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        self.stream_reader_thread = None
        self.worker.shutdown(wait=False, cancel_futures=True)
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        self.tts_worker.shutdown(wait=False, cancel_futures=True)
        self.speaker_worker.shutdown(wait=False, cancel_futures=True)
        if self.npu_whisper is not None:
            try:
                self.npu_whisper.close()
            except Exception:
                pass


def main():
    try:
        acquire_voice_lock()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    rclpy.init()
    node = VoiceAssistantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
