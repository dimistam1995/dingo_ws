#!/usr/bin/env python3
"""Record real Hey Dingo positives and hard negatives from the ReSpeaker."""

import argparse
import os
import time
import wave

import numpy as np
import sounddevice as sd

try:
    import rclpy
    from foxglove_msgs.msg import RawAudio
except ImportError:
    rclpy = None
    RawAudio = None


def find_device(name):
    needle = name.lower()
    for index, info in enumerate(sd.query_devices()):
        if info.get('max_input_channels', 0) > 0 and needle in info['name'].lower():
            return index
    raise RuntimeError(f'Δεν βρέθηκε συσκευή εισόδου «{name}»')


def record_direct(path, device, seconds):
    frames = int(seconds * 16000)
    audio = sd.rec(
        frames,
        samplerate=16000,
        channels=2,
        dtype='int16',
        device=device,
        blocking=True,
    )[:, 0]
    with wave.open(path, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.asarray(audio, dtype='<i2').tobytes())


class RosAudioRecorder:
    """Capture the already-open stream owned by dingo-voice.service."""

    def __init__(self, topic):
        if rclpy is None or RawAudio is None:
            raise RuntimeError('Το ROS 2 RawAudio δεν είναι διαθέσιμο.')
        self.node = rclpy.create_node('dingo_wake_training_recorder')
        self.chunks = []
        self.active = False
        self.subscription = self.node.create_subscription(
            RawAudio, topic, self._audio, 10
        )

    def _audio(self, message):
        if not self.active or not message.data:
            return
        channels = max(1, int(message.number_of_channels or 1))
        values = np.frombuffer(bytes(message.data), dtype='<i1')
        # RawAudio.data is a byte array. Reinterpret the bytes as signed 16-bit
        # PCM before selecting channel 0, the XVF3800 processed beam.
        if len(values) % 2:
            values = values[:-1]
        pcm = values.view('<i2')
        if channels > 1:
            pcm = pcm.reshape(-1, channels)[:, 0]
        self.chunks.append(np.asarray(pcm, dtype='<i2').copy())

    def record(self, path, seconds):
        self.chunks = []
        self.active = True
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.active = False
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)
        with wave.open(path, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(np.asarray(audio, dtype='<i2').tobytes())

    def close(self):
        self.node.destroy_node()


def collect(label, count, seconds, prompt, record):
    for index in range(1, count + 1):
        input(f'[{label} {index}/{count}] {prompt}  [Enter για εγγραφή] ')
        path = os.path.join('real_wav', f'real_{label}_{index:03d}.wav')
        record(path, seconds)
        print(f'Αποθηκεύτηκε {path}')
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-name', default='ReSpeaker')
    parser.add_argument('--positives', type=int, default=20)
    parser.add_argument('--negatives', type=int, default=30)
    parser.add_argument('--seconds', type=float, default=2.4)
    parser.add_argument(
        '--ros-topic', default='/dd100_10000002/voice/audio',
        help='RawAudio topic owned by dingo-voice.service',
    )
    args = parser.parse_args()
    os.makedirs('real_wav', exist_ok=True)
    ros_recorder = None
    if rclpy is not None:
        try:
            rclpy.init()
            ros_recorder = RosAudioRecorder(args.ros_topic)
            record = ros_recorder.record
            print(f'Χρησιμοποιώ το live audio topic {args.ros_topic}.')
        except Exception as exc:  # noqa: BLE001
            if rclpy.ok():
                rclpy.shutdown()
            print(f'ROS capture unavailable ({exc}); χρησιμοποιώ απευθείας ReSpeaker.')
    if ros_recorder is None:
        device = find_device(args.device_name)
        record = lambda path, seconds: record_direct(path, device, seconds)
    try:
        print('Θετικά: πες μόνο «Hey Dingo» με διαφορετική ένταση, απόσταση και γωνία.')
        collect('pos', args.positives, args.seconds, 'Πες καθαρά «Hey Dingo»', record)
        print(
            'Αρνητικά: πες φράσεις χωρίς «Hey Dingo», ειδικά «Dingo», '
            '«Hey Dino», «Ντίνο» και κανονικές εντολές.'
        )
        collect(
            'neg', args.negatives,
            args.seconds,
            'Πες μία φράση χωρίς «Hey Dingo»',
            record,
        )
    finally:
        if ros_recorder is not None:
            ros_recorder.close()
            if rclpy.ok():
                rclpy.shutdown()
    print('Τώρα τρέξε extract_features.py και train.py.')


if __name__ == '__main__':
    main()
