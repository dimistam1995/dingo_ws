#!/usr/bin/env python3
"""Run a quick streamed score check for the Hey Dingo model."""

import argparse
import wave

import numpy as np
from openwakeword.model import Model


def read(path):
    with wave.open(path, 'rb') as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)


def score(model, audio):
    audio = audio[-32000:]
    padded = np.zeros(32000, dtype=np.int16)
    padded[-len(audio):] = audio
    model.reset()
    for _ in range(6):
        model.predict(np.zeros(1280, dtype=np.int16))
    # The runtime triggers on any frame that crosses the threshold.  The
    # previous evaluator kept only the final frame, which could hide an
    # earlier false positive and approve an unsafe wake model.
    scores = []
    for start in range(0, len(padded), 1280):
        result = model.predict(padded[start:start + 1280])
        scores.append(float(list(result.values())[0]))
    return max(scores, default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='hey_dingo_candidate.onnx')
    args = parser.parse_args()
    model = Model(wakeword_model_paths=[args.model])
    positives, negatives = [], []
    with open('manifest.tsv', encoding='utf-8') as source:
        for line in source:
            label, path = line.strip().split('\t')
            value = score(model, read(path))
            (positives if label == 'pos' else negatives).append(value)
    print(
        f'positive min/mean/max: {min(positives):.3f}/'
        f'{np.mean(positives):.3f}/{max(positives):.3f}'
    )
    print(
        f'negative min/mean/max: {min(negatives):.3f}/'
        f'{np.mean(negatives):.3f}/{max(negatives):.3f}'
    )
    print(f'false positives at threshold 0.50: {sum(value >= 0.5 for value in negatives)}')


if __name__ == '__main__':
    main()
