#!/usr/bin/env python3
"""Train and export the small Hey Dingo openWakeWord classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class DingoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, value):
        return self.net(value)


def run(seed, x_train, y_train, x_val, y_val):
    torch.manual_seed(seed)
    model = DingoClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive_fraction = float(y_train.mean())
    positive_weight = 0.5 / max(positive_fraction, 1e-6)
    negative_weight = 0.5 / max(1.0 - positive_fraction, 1e-6)
    best_loss, best_state = float('inf'), None
    batch_size = 128
    for _ in range(40):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            output = model(
                x_train[indices] + torch.randn_like(x_train[indices]) * 0.01
            ).squeeze(-1)
            weights = y_train[indices] * positive_weight + (
                1.0 - y_train[indices]
            ) * negative_weight
            loss = nn.functional.binary_cross_entropy(
                output, y_train[indices], weight=weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = model(x_val).squeeze(-1)
            val_loss = nn.functional.binary_cross_entropy(value, y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: item.detach().clone()
                          for key, item in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        value = model(x_val).squeeze(-1).numpy()
    return best_loss, model, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--install',
        action='store_true',
        help='copy the candidate to models/wake_words after real-data review',
    )
    args = parser.parse_args()
    data = np.load('features.npz')
    x, y = data['X'], data['y']
    clean = data['clean'] if 'clean' in data else np.ones_like(y)
    x_train, x_val, y_train, y_val, _, clean_val = train_test_split(
        x, y, clean, test_size=0.2, random_state=42, stratify=y
    )
    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    best = None
    for seed in range(6):
        loss, model, output = run(seed, x_train, y_train, x_val, y_val)
        clean_mask = clean_val > 0.5
        clean_output = output[clean_mask]
        clean_y = y_val.numpy()[clean_mask]
        negatives = clean_output[clean_y < 0.5]
        positives = clean_output[clean_y > 0.5]
        false_positive = int((negatives > 0.5).sum())
        false_negative = int((positives < 0.5).sum())
        worst_negative = float(negatives.max()) if len(negatives) else 0.0
        recall_floor = float(np.quantile(positives, 0.10)) if len(positives) else 0.0
        margin = recall_floor - worst_negative
        print(
            f'seed={seed} margin={margin:+.3f} '
            f'clean_fp={false_positive} clean_fn={false_negative} '
            f'worst_negative={worst_negative:.3f} loss={loss:.4f}'
        )
        ranking = (margin, -loss)
        if best is None or ranking > best[0]:
            best = (ranking, model, seed)
    _, model, seed = best
    model.eval()
    workspace = Path(__file__).resolve().parents[2]
    destination = Path('hey_dingo_candidate.onnx')
    if args.install:
        destination = workspace / 'models' / 'wake_words' / 'dingo.onnx'
        destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 16, 96, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        destination,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
    )
    print(f'Picked seed {seed}; exported {destination}')
    if not args.install:
        print('Candidate is not installed. Run evaluate.py, then use --install '
              'only after real positive/negative recordings pass.')


if __name__ == '__main__':
    main()
