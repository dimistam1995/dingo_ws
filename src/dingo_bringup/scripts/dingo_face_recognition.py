#!/usr/bin/env python3
"""Fresh on-device face detection and recognition for the Dingo camera.

The node is deliberately independent from the previous project.  It uses the
OpenCV Zoo YuNet detector and SFace recognizer, keeps a small local gallery of
normalized face features, and only starts enrollment after an explicit
Dashboard command.  It publishes state/boxes; it does not speak on every
frame and it never drives the robot.
"""

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def package_model_path(filename):
    try:
        root = Path(get_package_share_directory('dingo_bringup'))
    except Exception:
        root = Path(__file__).resolve().parents[2]
    return root / 'models' / 'identity' / filename


class FaceGallery:
    """Persistent name -> normalized SFace feature mapping."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.lock = threading.RLock()
        self.features = {}
        self.load()

    @staticmethod
    def clean_name(name):
        value = ' '.join(str(name or '').split()).strip()
        if not value:
            raise ValueError('Χρειάζεται όνομα για την εγγραφή προσώπου.')
        if len(value) > 48:
            raise ValueError('Το όνομα είναι πολύ μεγάλο (μέχρι 48 χαρακτήρες).')
        return value

    @staticmethod
    def normalize(values):
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError('Το χαρακτηριστικό προσώπου είναι κενό.')
        return (vector / norm).astype(np.float32).tolist()

    def load(self):
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            payload = {}
        raw = payload.get('faces', {}) if isinstance(payload, dict) else {}
        loaded = {}
        if isinstance(raw, dict):
            for raw_name, values in raw.items():
                try:
                    loaded[self.clean_name(raw_name)] = self.normalize(values)
                except (TypeError, ValueError):
                    continue
        with self.lock:
            self.features = loaded

    def names(self):
        with self.lock:
            return sorted(self.features)

    def save(self):
        with self.lock:
            payload = {'version': 1, 'faces': dict(sorted(self.features.items()))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{self.path.name}.', suffix='.tmp', dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def upsert(self, name, feature):
        name = self.clean_name(name)
        with self.lock:
            self.features[name] = self.normalize(feature)
        self.save()
        return name

    def remove(self, name):
        name = self.clean_name(name)
        with self.lock:
            existed = self.features.pop(name, None) is not None
        if existed:
            self.save()
        return existed

    def match(self, feature):
        query = np.asarray(self.normalize(feature), dtype=np.float32)
        best_name = None
        best_score = -1.0
        with self.lock:
            items = list(self.features.items())
        for name, values in items:
            candidate = np.asarray(values, dtype=np.float32)
            if candidate.shape != query.shape:
                continue
            score = float(np.dot(query, candidate))
            if score > best_score:
                best_name, best_score = name, score
        return best_name, best_score if best_name is not None else None


class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('dingo_face_recognition')
        self.robot_namespace = str(
            self.declare_parameter('robot_namespace', 'dd100_10000002').value
        ).strip('/')
        prefix = f'/{self.robot_namespace}' if self.robot_namespace else ''
        self.camera_topic = str(
            self.declare_parameter(
                'camera_topic', '/camera/camera/color/image_raw'
            ).value
        )
        self.process_every_n = max(
            1, int(self.declare_parameter('process_every_n', 5).value or 5)
        )
        self.detection_score = float(
            self.declare_parameter('detection_score', 0.65).value or 0.65
        )
        self.recognition_threshold = float(
            self.declare_parameter('recognition_threshold', 0.40).value or 0.40
        )
        self.enroll_frames = max(
            3, int(self.declare_parameter('enroll_frames', 5).value or 5)
        )
        self.min_face_size = max(
            30, int(self.declare_parameter('min_face_size', 60).value or 60)
        )
        default_gallery = str(
            Path.home() / '.config' / 'dingo_identity' / 'faces.json'
        )
        self.gallery = FaceGallery(
            self.declare_parameter('face_gallery', default_gallery).value
        )
        default_yunet = str(
            package_model_path('yunet_face_detection.onnx')
        )
        default_sface = str(
            package_model_path('sface_face_recognition.onnx')
        )
        self.yunet_model = Path(
            self.declare_parameter('yunet_model', default_yunet).value
        ).expanduser()
        self.sface_model = Path(
            self.declare_parameter('sface_model', default_sface).value
        ).expanduser()

        self.state_pub = self.create_publisher(
            String, f'{prefix}/face/state', 10
        )
        self.reply_pub = self.create_publisher(
            String, f'{prefix}/voice/reply', 10
        )
        self.create_subscription(
            Image, self.camera_topic, self.camera, qos_profile_sensor_data
        )
        self.create_subscription(
            String, f'{prefix}/face/command', self.command, 10
        )

        self.bridge = CvBridge()
        self.detector = None
        self.recognizer = None
        self.input_size = None
        self.frame_count = 0
        self.last_frame_at = 0.0
        self.last_publish_at = 0.0
        self.faces = []
        self.enrollment = None
        self.status_state = 'loading'
        self.status_message = 'Φορτώνω YuNet και SFace…'
        self.load_error = None
        self.publish_state(force=True)
        self.load_models()
        self.create_timer(1.0, self.timer)

    def load_models(self):
        try:
            if not self.yunet_model.is_file():
                raise RuntimeError(f'Δεν βρέθηκε YuNet: {self.yunet_model}')
            if not self.sface_model.is_file():
                raise RuntimeError(f'Δεν βρέθηκε SFace: {self.sface_model}')
            self.detector = cv2.FaceDetectorYN.create(
                str(self.yunet_model), '', (320, 240),
                self.detection_score, 0.3, 5000,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                str(self.sface_model), ''
            )
            self.status_state = 'waiting_for_camera'
            self.status_message = 'Τα μοντέλα είναι έτοιμα· περιμένω RealSense.'
            self.get_logger().info(
                f'Face recognition ready: YuNet + SFace · {len(self.gallery.names())} enrolled'
            )
        except Exception as exc:
            self.detector = None
            self.recognizer = None
            self.load_error = str(exc)
            self.status_state = 'error'
            self.status_message = f'Αποτυχία μοντέλων προσώπου: {exc}'
            self.get_logger().error(self.status_message)
        self.publish_state(force=True)

    def publish_reply(self, text, ok=True, action=None):
        payload = {
            'ok': bool(ok),
            'text': str(text),
            'action': action,
            'source': 'face_recognition',
        }
        self.reply_pub.publish(String(data=compact_json(payload)))

    def state_payload(self):
        enrollment = None
        if self.enrollment:
            enrollment = {
                'name': self.enrollment['name'],
                'captured': len(self.enrollment['features']),
                'required': self.enroll_frames,
            }
        return {
            'state': self.status_state,
            'source': 'YuNet + SFace',
            'message_el': self.status_message,
            'faces': list(self.faces),
            'enrolled': self.gallery.names(),
            'enrollment': enrollment,
            'last_update': self.last_frame_at or None,
        }

    def publish_state(self, force=False):
        now = time.time()
        if not force and now - self.last_publish_at < 0.15:
            return
        self.last_publish_at = now
        self.state_pub.publish(String(data=compact_json(self.state_payload())))

    def timer(self):
        if self.status_state == 'waiting_for_camera' and self.last_frame_at:
            if time.time() - self.last_frame_at > 5.0:
                self.status_message = 'Τα μοντέλα είναι έτοιμα· περιμένω RealSense.'
        self.publish_state(force=True)

    def command(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.publish_reply('Η εντολή προσώπου δεν είναι έγκυρη.', ok=False)
            return
        if not isinstance(payload, dict):
            self.publish_reply('Η εντολή προσώπου δεν είναι έγκυρη.', ok=False)
            return
        action = str(payload.get('action', '')).strip().lower()
        try:
            if action == 'enroll':
                self.start_enrollment(payload.get('name'))
            elif action == 'cancel':
                self.cancel_enrollment()
            elif action == 'forget':
                name = self.gallery.clean_name(payload.get('name'))
                if self.gallery.remove(name):
                    self.publish_reply(f'Διαγράφηκε το πρόσωπο «{name}».', action='face_forget')
                else:
                    self.publish_reply(f'Δεν βρήκα εγγραφή για «{name}».', ok=False, action='face_forget')
                self.publish_state(force=True)
            elif action == 'query':
                self.publish_reply(self.query_text(), action='face_query')
            else:
                self.publish_reply('Άγνωστη εντολή προσώπου.', ok=False)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.publish_reply(str(exc), ok=False, action=action or None)

    def start_enrollment(self, name):
        if self.detector is None or self.recognizer is None:
            raise RuntimeError('Η αναγνώριση προσώπου δεν έχει φορτώσει.')
        name = self.gallery.clean_name(name)
        self.enrollment = {'name': name, 'features': []}
        self.status_state = 'enrolling'
        self.status_message = (
            f'Κοίτα την κάμερα για {self.enroll_frames} καθαρά δείγματα για «{name}».'
        )
        self.publish_reply(
            f'Ξεκίνησε η εγγραφή προσώπου για «{name}». Κοίτα την κάμερα μέχρι να ολοκληρωθεί.',
            action='face_enroll',
        )
        self.publish_state(force=True)

    def cancel_enrollment(self):
        was_active = self.enrollment is not None
        self.enrollment = None
        if self.detector is not None:
            self.status_state = 'ready' if self.last_frame_at else 'waiting_for_camera'
            self.status_message = 'Η εγγραφή προσώπου ακυρώθηκε.' if was_active else 'Δεν υπάρχει ενεργή εγγραφή.'
        if was_active:
            self.publish_reply('Η εγγραφή προσώπου ακυρώθηκε.', action='face_cancel')
        self.publish_state(force=True)

    def largest_face(self, faces):
        candidates = []
        for row in faces or []:
            values = np.asarray(row).reshape(-1)
            if values.size < 15:
                continue
            x, y, width, height, score = map(float, values[:5])
            if score < self.detection_score or width < self.min_face_size or height < self.min_face_size:
                continue
            candidates.append((width * height, values))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def detect(self, frame):
        height, width = frame.shape[:2]
        if self.input_size != (width, height):
            self.detector.setInputSize((width, height))
            self.input_size = (width, height)
        result = self.detector.detect(frame)
        raw_faces = result[1] if isinstance(result, tuple) else result
        if raw_faces is None:
            return []
        return [np.asarray(item).reshape(-1) for item in raw_faces]

    def feature_for(self, frame, face):
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned)
            return FaceGallery.normalize(feature)
        except cv2.error:
            return None

    def recognize(self, frame, face):
        feature = self.feature_for(frame, face)
        if feature is None:
            return None, None
        name, score = self.gallery.match(feature)
        if score is None or score < self.recognition_threshold:
            name = 'άγνωστο πρόσωπο'
        return name, score

    @staticmethod
    def box(face, width, height):
        x, y, box_width, box_height = [float(value) for value in face[:4]]
        x1 = max(0, min(width, int(round(x))))
        y1 = max(0, min(height, int(round(y))))
        x2 = max(x1, min(width, int(round(x + box_width))))
        y2 = max(y1, min(height, int(round(y + box_height))))
        return x1, y1, x2, y2

    def camera(self, msg):
        self.frame_count += 1
        self.last_frame_at = time.time()
        if self.detector is None or self.recognizer is None:
            self.status_state = 'error'
            self.publish_state()
            return
        if self.frame_count % self.process_every_n:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = np.ascontiguousarray(frame)
            detected = self.detect(frame)
        except (CvBridgeError, cv2.error, ValueError, TypeError) as exc:
            self.status_state = 'error'
            self.status_message = f'Σφάλμα κάμερας προσώπου: {exc}'
            self.get_logger().warning(self.status_message)
            self.publish_state(force=True)
            return

        height, width = frame.shape[:2]
        output = []
        for face in detected[:16]:
            if face.size < 15 or float(face[14]) < self.detection_score:
                continue
            name, score = self.recognize(frame, face)
            x1, y1, x2, y2 = self.box(face, width, height)
            output.append(
                {
                    'name': name or 'άγνωστο πρόσωπο',
                    'score': round(float(score), 3) if score is not None and math.isfinite(float(score)) else None,
                    'detection_score': round(float(face[14]), 3),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                }
            )
        self.faces = output
        self.status_state = 'enrolling' if self.enrollment else 'ready'
        if self.enrollment:
            candidate = self.largest_face(detected)
            if candidate is not None:
                feature = self.feature_for(frame, candidate)
                if feature is not None:
                    self.enrollment['features'].append(feature)
                    captured = len(self.enrollment['features'])
                    self.status_message = (
                        f'Εγγραφή «{self.enrollment["name"]}»: '
                        f'{captured}/{self.enroll_frames} δείγματα.'
                    )
                    if captured >= self.enroll_frames:
                        name = self.enrollment['name']
                        self.gallery.upsert(name, np.mean(self.enrollment['features'], axis=0))
                        self.enrollment = None
                        self.status_state = 'ready'
                        self.status_message = f'Το πρόσωπο «{name}» αποθηκεύτηκε.'
                        self.publish_reply(
                            f'Ολοκληρώθηκε η εγγραφή προσώπου για «{name}».',
                            action='face_enroll',
                        )
                else:
                    self.status_message = 'Βλέπω πρόσωπο αλλά δεν πήρα καθαρό χαρακτηριστικό.'
            else:
                self.status_message = 'Κοίτα την κάμερα· δεν βρήκα αρκετά καθαρό πρόσωπο.'
        elif output:
            self.status_message = f'Εντοπίστηκαν {len(output)} πρόσωπα.'
        else:
            self.status_message = 'Δεν εντοπίστηκε πρόσωπο.'
        self.publish_state(force=True)

    def query_text(self):
        if self.status_state in {'loading', 'error'}:
            return 'Η αναγνώριση προσώπου δεν είναι έτοιμη.'
        if not self.gallery.names():
            return 'Δεν έχει εγγραφεί ακόμη κανένα πρόσωπο στο Dingo.'
        if not self.faces:
            return 'Δεν βλέπω πρόσωπο αυτή τη στιγμή.'
        names = [item['name'] for item in self.faces]
        known = [name for name in names if name != 'άγνωστο πρόσωπο']
        if known:
            return 'Μπροστά μου βλέπω: ' + ', '.join(known) + '.'
        return 'Βλέπω πρόσωπο, αλλά δεν αναγνωρίζω ποιος είναι.'


def main():
    rclpy.init()
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
