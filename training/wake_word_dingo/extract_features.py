#!/usr/bin/env python3
"""Create openWakeWord (16, 96) embeddings with microphone-like augmentation."""

import os
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter
from openwakeword.utils import AudioFeatures


WINDOW = 32000
TRAILING = (0, 3200)
N_AUGMENTED = 4
N_SILENCE = 60
FRAME = 320
PAD = 1600


def read_wav(path):
    with wave.open(path, 'rb') as stream:
        rate = stream.getframerate()
        width = stream.getsampwidth()
        channels = stream.getnchannels()
        frames = stream.getnframes()
        raw = stream.readframes(frames)
    if rate != 16000 or width != 2:
        raise ValueError(f'{path}: expected 16 kHz, 16-bit WAV')
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio


def trim_silence(audio, threshold=150, pad=800):
    voiced = np.flatnonzero(np.abs(audio) > threshold)
    if len(voiced) == 0:
        return audio
    return audio[max(0, voiced[0] - pad):min(len(audio), voiced[-1] + pad)]


def fit_window(audio):
    if len(audio) <= WINDOW:
        return audio
    count = len(audio) // FRAME
    if count:
        frames = audio[:count * FRAME].reshape(count, FRAME).astype(np.float64)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        for ratio in (0.15, 0.25, 0.35):
            voiced = np.flatnonzero(rms > max(rms.max() * ratio, 150))
            if len(voiced):
                start = max(0, voiced[0] * FRAME - PAD)
                end = min(len(audio), (voiced[-1] + 1) * FRAME + PAD)
                candidate = audio[start:end]
                if len(candidate) <= WINDOW:
                    return candidate
    return audio


def make_buffer(audio, trailing):
    available = WINDOW - trailing
    if len(audio) > available:
        return None
    output = np.zeros(WINDOW, dtype=np.int16)
    start = available - len(audio)
    output[start:start + len(audio)] = audio
    return output


def augment(buffer, noise, rng):
    values = buffer.astype(np.float64)
    current = np.sqrt((values ** 2).mean()) + 1e-6
    target = 10 ** rng.uniform(np.log10(80), np.log10(4500))
    values *= target / current
    if rng.random() < 0.5:
        length = int(rng.integers(800, 3200))
        tau = length / rng.uniform(2, 5)
        impulse = rng.standard_normal(length) * np.exp(-np.arange(length) / tau)
        impulse /= np.sqrt((impulse ** 2).sum()) + 1e-9
        values = fftconvolve(values, impulse, mode='full')[:len(values)]
    if rng.random() < 0.5:
        cutoff = rng.uniform(3000, 7000)
        b, a = butter(2, cutoff / 8000, btype='low')
        values = lfilter(b, a, values)
    if len(noise) < len(values):
        noise = np.resize(noise, len(values))
    start = int(rng.integers(0, len(noise) - len(values) + 1))
    room = noise[start:start + len(values)].astype(np.float64)
    signal_rms = np.sqrt((values ** 2).mean()) + 1e-6
    room_rms = np.sqrt((room ** 2).mean()) + 1e-6
    snr = rng.uniform(5, 25)
    values += room * (signal_rms / (10 ** (snr / 20)) / room_rms)
    return np.clip(values, -32768, 32767).astype(np.int16)


def load_noise():
    path = os.path.join('noise', 'room_noise.wav')
    if os.path.exists(path):
        return read_wav(path)
    rng = np.random.default_rng(42)
    # A quiet synthetic room floor is only a bootstrap. Real room noise is
    # collected by record_real.py and becomes more valuable after retraining.
    return (rng.standard_normal(64000) * 45).astype(np.int16)


def main():
    manifest = []
    with open('manifest.tsv', encoding='utf-8') as source:
        for line in source:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                manifest.append((parts[0], parts[1]))
    noise = load_noise()
    rng = np.random.default_rng(0)
    buffers, labels, clean = [], [], []
    skipped = 0
    for label, path in manifest:
        audio = fit_window(trim_silence(read_wav(path)))
        if len(audio) > WINDOW:
            skipped += 1
            continue
        for trailing in TRAILING:
            base = make_buffer(audio, trailing)
            if base is None:
                continue
            value = 1 if label == 'pos' else 0
            buffers.append(base)
            labels.append(value)
            clean.append(1)
            for _ in range(N_AUGMENTED):
                buffers.append(augment(base, noise, rng))
                labels.append(value)
                clean.append(0)
    for _ in range(N_SILENCE):
        buffers.append((rng.standard_normal(WINDOW) * 45).astype(np.int16))
        labels.append(0)
        clean.append(0)
    clips = np.stack(buffers)
    print(
        f'Examples: {len(labels)}; positive={sum(labels)}; '
        f'negative={len(labels) - sum(labels)}; skipped={skipped}'
    )
    features = AudioFeatures().embed_clips(clips, batch_size=64)
    np.savez(
        'features.npz',
        X=features.astype(np.float32),
        y=np.asarray(labels, dtype=np.float32),
        clean=np.asarray(clean, dtype=np.float32),
    )
    print(f'Saved features.npz with shape {features.shape}')


if __name__ == '__main__':
    main()

