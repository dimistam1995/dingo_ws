#!/usr/bin/env python3
"""Hardware microphone status and direction-of-arrival for the Dingo.

The ReSpeaker XVF3800 exposes its DoA/VAD state through a small USB vendor
control endpoint in addition to the normal ALSA audio stream.  Keeping that
access in its own node means the voice assistant can continue to use the
beamformed channel while the Dashboard receives a reliable direction and
hardware-health signal.

This node is deliberately read-only with respect to the robot base: it never
publishes velocity and it never turns the Dingo.  It may drive the XVF3800's
LED ring as a local listening indicator.

Published (under the configured robot namespace):
  voice/direction (std_msgs/String, JSON)

The JSON includes the current angle (0-359 degrees, 0 is the front of the
array), the DSP speech flag, device availability, and the processing features
provided by the XVF3800 firmware.  The Dashboard can therefore distinguish
"the USB microphone is present" from "the DSP currently hears speech".
"""

import json
import struct
import threading
import time

try:
    import usb.core
    import usb.util
except ImportError:  # The node can remain visible as unavailable.
    usb = None

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


VID = 0x2886
PID = 0x001A
USB_TIMEOUT_MS = 100_000

# XVF3800 firmware parameter IDs.
DOA_VALUE = (20, 18, 2)          # resource, command, two uint16 values
LED_EFFECT = (20, 12)
LED_BRIGHTNESS = (20, 13)
LED_SPEED = (20, 15)
LED_COLOR = (20, 16)
LED_DOA_COLOR = (20, 17)

LED_OFF = 0
LED_BREATH = 1
LED_SINGLE = 3
LED_DOA = 4

# LED_COLOR is a conventional 0xRRGGBB value.  The USB transport serializes
# the uint32 little-endian, but the firmware interprets the numeric value as
# RRGGBB.  Keep the named colours in their normal RGB form so the visible
# ring matches the voice state.
COLOR_BLUE = 0x0000FF
COLOR_RED = 0xFF0000
COLOR_GREEN = 0x00FF00
COLOR_DARK = 0x111111


class XVF3800:
    """Small, serialised wrapper around the XVF3800 vendor USB controls."""

    def __init__(self, device):
        self.device = device
        self.lock = threading.Lock()

    def _ctrl_in(self, command, resource, length):
        return self.device.ctrl_transfer(
            usb.util.CTRL_IN
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            0x80 | command,
            resource,
            length,
            USB_TIMEOUT_MS,
        )

    def _ctrl_out(self, command, resource, payload):
        self.device.ctrl_transfer(
            usb.util.CTRL_OUT
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            command,
            resource,
            payload,
            USB_TIMEOUT_MS,
        )

    def read_doa(self):
        """Return ``(speech_detected, angle_degrees)`` from the DSP."""
        resource, command, count = DOA_VALUE
        with self.lock:
            response = self._ctrl_in(command, resource, count * 2 + 1)
        raw = bytes(response)
        if len(raw) < 5:
            raise RuntimeError(f'Το XVF3800 επέστρεψε μόνο {len(raw)} bytes DoA')
        angle, speech = struct.unpack_from('<HH', raw, 1)
        return bool(speech), float(angle % 360)

    def _write_u8(self, item, value):
        resource, command = item
        with self.lock:
            self._ctrl_out(command, resource, bytes([max(0, min(255, int(value)))]))

    def _write_u32(self, item, value):
        resource, command = item
        with self.lock:
            self._ctrl_out(command, resource, struct.pack('<I', int(value) & 0xFFFFFFFF))

    def set_led_state(self, state, brightness):
        """Set a conservative local LED indication; failures are propagated."""
        self._write_u8(LED_BRIGHTNESS, brightness)
        if state == 'listening':
            # Keep the idle/listening indicator steady.  Animation is reserved
            # for the processing state so the user can tell when Whisper/LLM
            # is actually working.
            self._write_u32(LED_COLOR, COLOR_BLUE)
            self._write_u8(LED_EFFECT, LED_SINGLE)
        elif state == 'processing':
            # A breathing red light is the only normal animated state.
            self._write_u32(LED_COLOR, COLOR_RED)
            self._write_u8(LED_SPEED, 8)
            self._write_u8(LED_EFFECT, LED_BREATH)
        elif state == 'direction':
            # Let the XVF3800's DSP move the pointer to the current direction
            # of arrival.  The pointer is green while the array is hearing a
            # voice; the dark background keeps the direction easy to see.
            resource, command = LED_DOA_COLOR
            with self.lock:
                self._ctrl_out(
                    command,
                    resource,
                    struct.pack('<II', COLOR_DARK, COLOR_GREEN),
                )
            self._write_u8(LED_SPEED, 8)
            self._write_u8(LED_EFFECT, LED_DOA)
        elif state == 'error':
            self._write_u32(LED_COLOR, COLOR_RED)
            self._write_u8(LED_EFFECT, LED_SINGLE)
        elif state == 'muted':
            self._write_u32(LED_COLOR, COLOR_RED)
            self._write_u8(LED_EFFECT, LED_SINGLE)
        elif state == 'off':
            self._write_u8(LED_EFFECT, LED_OFF)
        else:
            self._write_u8(LED_EFFECT, LED_OFF)

    def close(self):
        if usb is not None:
            try:
                usb.util.dispose_resources(self.device)
            except Exception:
                pass


def find_device():
    if usb is None:
        return None
    device = usb.core.find(idVendor=VID, idProduct=PID)
    return XVF3800(device) if device is not None else None


class DingoMicArrayNode(Node):
    """Publish XVF3800 DoA/VAD and control only its local LED ring."""

    def __init__(self):
        super().__init__('dingo_mic_array')
        namespace = str(
            self.declare_parameter('robot_namespace', 'dd100_10000002').value
            or ''
        ).strip('/')
        prefix = f'/{namespace}' if namespace else ''
        self.poll_hz = max(
            2.0,
            min(20.0, float(self.declare_parameter('poll_hz', 10.0).value or 10.0)),
        )
        self.led_enabled = bool(self.declare_parameter('led_enabled', True).value)
        # The DSP can always be queried, but publishing/using the direction
        # is a user-controlled feature.  It starts OFF so the Dashboard is
        # the only place that enables it deliberately.
        self.direction_enabled = bool(
            self.declare_parameter('direction_enabled', False).value
        )
        self.led_brightness = max(
            0,
            min(255, int(self.declare_parameter('led_brightness', 150).value or 150)),
        )

        self.direction_pub = self.create_publisher(
            String, f'{prefix}/voice/direction', 10
        )
        direction_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool,
            f'{prefix}/voice/direction_enable',
            self.direction_enable,
            direction_qos,
        )
        self.create_subscription(
            # voice/status is TRANSIENT_LOCAL so a restarted mic-array node
            # immediately receives the assistant's current state.  With the
            # default VOLATILE QoS it missed the latched "listening" state
            # and left the ring in its previous firmware colour.
            String, f'{prefix}/voice/status', self.voice_status, direction_qos
        )

        self.device = None
        self.device_name = None
        self.last_angle = None
        self.last_speech = False
        self.last_angle_at = None
        self.last_error = ''
        self.voice_state = 'idle'
        self.led_state = 'idle'
        self.last_led_state = None
        self.last_publish_at = 0.0
        self.last_missing_publish_at = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.publish_direction(force=True)
        self.poll_thread = threading.Thread(target=self.poll_loop, daemon=True)
        self.poll_thread.start()
        self.get_logger().info(
            'XVF3800 microphone service started: DoA/VAD + local listening indicator'
        )

    def desired_led_state(self, voice_state=None, hardware_speech=None):
        """Resolve the LED state from the DSP and voice-processing states.

        The XVF3800 firmware already provides VAD and DoA.  Its speech flag
        drives the immediate green direction pointer, while Whisper/LLM
        states take priority and keep the complete ring red during work.
        """
        state = self.voice_state if voice_state is None else str(voice_state or '')
        speech = self.last_speech if hardware_speech is None else bool(hardware_speech)
        if state in {'transcribing', 'thinking', 'busy', 'speaking'}:
            return 'processing'
        if state in {'error', 'microphone_error', 'dependency_error'}:
            return 'error'
        if self.direction_enabled and (
            speech or state in {'wake_detected', 'speech_detected'}
        ):
            return 'direction'
        if state in {'listening', 'wake_detected', 'speech_detected'}:
            return 'listening'
        return 'idle'

    def voice_status(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        state = str(payload.get('state', '') or '')
        with self.lock:
            self.voice_state = state or self.voice_state
            desired = self.desired_led_state()
            changed = desired != self.led_state
            self.led_state = desired
            device = self.device
        if changed and device is not None and self.led_enabled:
            self.apply_led(device, desired)
        self.publish_direction(force=True)

    def direction_enable(self, message):
        enabled = bool(message.data)
        with self.lock:
            changed = enabled != self.direction_enabled
            self.direction_enabled = enabled
            device = self.device
            desired = self.desired_led_state()
            led_changed = desired != self.led_state
            self.led_state = desired
        if (changed or led_changed) and device is not None and self.led_enabled:
            self.apply_led(device, desired)
        self.publish_direction(force=True)
        self.get_logger().info(
            f'Κατεύθυνση φωνής: {"ΕΝΕΡΓΗ" if enabled else "ΚΛΕΙΣΤΗ"}'
        )

    def apply_led(self, device, state):
        try:
            # The DoA pointer is only shown after the user enables the
            # feature. Listening/processing colours remain available even
            # while direction is disabled.
            effective_state = state
            if state == 'direction' and not self.direction_enabled:
                effective_state = 'listening'
            elif state == 'idle':
                effective_state = 'off'
            device.set_led_state(effective_state, self.led_brightness)
            self.last_led_state = effective_state
        except Exception as exc:
            self.get_logger().warning(f'Δεν μπόρεσα να αλλάξω το LED του XVF3800: {exc}')

    def payload(self, *, available, state, speech, angle, error=None):
        enabled = bool(self.direction_enabled)
        return {
            'state': state,
            'available': bool(available),
            'enabled': enabled,
            'device': self.device_name,
            'angle_deg': (
                round(float(angle), 1)
                if enabled and angle is not None
                else None
            ),
            'speech': bool(speech) if enabled else False,
            'angle_at': self.last_angle_at,
            'error': error or None,
            'features': {
                'direction_of_arrival': bool(available and enabled),
                'hardware_vad': bool(available and enabled),
                'beamforming': bool(available),
                'noise_reduction': bool(available),
                'echo_cancellation': bool(available),
                'automatic_gain': bool(available),
            },
            'led': self.last_led_state or self.led_state,
            'source': 'xvf3800_dsp',
        }

    def publish_direction(self, force=False):
        now = time.time()
        if not force and now - self.last_publish_at < 0.20:
            return
        self.last_publish_at = now
        with self.lock:
            available = self.device is not None
            payload = self.payload(
                available=available,
                state='ready' if available else 'waiting_for_respeaker',
                speech=self.last_speech,
                angle=self.last_angle,
                error=self.last_error,
            )
        self.direction_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        )

    def poll_loop(self):
        interval = 1.0 / self.poll_hz
        while rclpy.ok() and not self.stop_event.is_set():
            with self.lock:
                device = self.device
            if device is None:
                device = find_device()
                if device is None:
                    with self.lock:
                        self.last_error = (
                            'Δεν βρέθηκε το ReSpeaker XVF3800 μέσω USB.'
                        )
                    if time.time() - self.last_missing_publish_at >= 2.0:
                        self.last_missing_publish_at = time.time()
                        self.publish_direction(force=True)
                    time.sleep(2.0)
                    continue
                with self.lock:
                    self.device = device
                    self.device_name = 'reSpeaker XVF3800 4-Mic Array'
                    self.last_error = ''
                if self.led_enabled:
                    self.apply_led(device, self.led_state)
                self.publish_direction(force=True)
                self.get_logger().info('Βρέθηκε το ReSpeaker XVF3800 και το DoA είναι ενεργό.')

            try:
                speech, angle = device.read_doa()
                with self.lock:
                    self.last_speech = speech
                    self.last_angle = angle
                    self.last_angle_at = time.time()
                    self.last_error = ''
                    desired = self.desired_led_state(hardware_speech=speech)
                    led_changed = desired != self.led_state
                    self.led_state = desired
                if led_changed and self.led_enabled:
                    self.apply_led(device, desired)
                self.publish_direction()
            except Exception as exc:  # USB unplug/reset: retry cleanly.
                self.get_logger().warning(f'Σφάλμα ανάγνωσης DoA από XVF3800: {exc}')
                with self.lock:
                    self.last_error = str(exc)
                    self.device = None
                    self.device_name = None
                try:
                    device.close()
                except Exception:
                    pass
                self.publish_direction(force=True)
            time.sleep(interval)

    def destroy_node(self):
        self.stop_event.set()
        with self.lock:
            device = self.device
            self.device = None
        if device is not None:
            if self.led_enabled:
                try:
                    device.set_led_state('idle', self.led_brightness)
                except Exception:
                    pass
            device.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DingoMicArrayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
