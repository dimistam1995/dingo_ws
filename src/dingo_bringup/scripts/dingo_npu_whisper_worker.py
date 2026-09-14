#!/usr/bin/env python3
"""JSON-lines server for AMD's VitisAI Whisper ONNX models."""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000

# The worker is started with RyzenAI's Python, but these defaults also make a
# direct diagnostic invocation behave like the service.
_XRT = '/opt/xilinx/xrt'
os.environ.setdefault('XILINX_XRT', _XRT)
_VENV = str(Path(sys.executable).resolve().parents[1])
_LIB_DIRS = [
    f'{_VENV}/deployment/lib',
    f'{_VENV}/onnxruntime/lib',
    f'{_VENV}/lib/python3.12/site-packages/flexmlrt/lib',
    f'{_VENV}/lib/python3.12/site-packages/flexml/flexml_extras/lib',
    f'{_VENV}/lib/python3.12/site-packages/voe/lib',
    f'{_XRT}/lib',
]
os.environ['LD_LIBRARY_PATH'] = ':'.join(_LIB_DIRS) + ':' + os.environ.get(
    'LD_LIBRARY_PATH', ''
)
os.environ['PATH'] = f'{_XRT}/bin:' + os.environ.get('PATH', '')
os.environ['PYTHONPATH'] = f'{_XRT}/python:' + os.environ.get('PYTHONPATH', '')


class WhisperONNX:
    def __init__(self, model_dir, tokenizer_dir, config_dir, cache_dir,
                 model_size, language):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        model_dir = Path(model_dir)
        tokenizer_dir = Path(tokenizer_dir)
        config_dir = Path(config_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # AMD publishes the newer large-v3 export with model-specific file
        # names, while the medium/turbo exports use the generic names.
        # Accept both layouts so the configured model is selected explicitly
        # instead of silently falling back to another model.
        encoder_path = next(
            (model_dir / name for name in (
                'encoder_model.onnx',
                'large_v3_encoder.onnx',
            ) if (model_dir / name).exists()),
            model_dir / 'encoder_model.onnx',
        )
        decoder_path = next(
            (model_dir / name for name in (
                'decoder_model.onnx',
                'large_v3_decoder.onnx',
            ) if (model_dir / name).exists()),
            model_dir / 'decoder_model.onnx',
        )
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(
                f'Λείπουν τα AMD Whisper ONNX αρχεία από {model_dir}'
            )
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f'Λείπει το Whisper tokenizer: {tokenizer_dir}')

        encoder_config = config_dir / 'vitisai_config_whisper_encoder.json'
        decoder_config = config_dir / 'vitisai_config_whisper_decoder.json'
        if not encoder_config.exists() or not decoder_config.exists():
            raise FileNotFoundError(
                f'Λείπουν τα VitisAI Whisper configs από {config_dir}'
            )

        def providers(config, key):
            return [
                ('VitisAIExecutionProvider', {
                    'config_file': str(config),
                    'cache_dir': str(cache_dir),
                    'cache_key': key,
                }),
                'CPUExecutionProvider',
            ]

        started = time.monotonic()
        self.encoder = ort.InferenceSession(
            str(encoder_path),
            providers=providers(
                encoder_config, f'whisper_{model_size}_encoder'
            ),
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path),
            providers=providers(
                decoder_config, f'whisper_{model_size}_decoder'
            ),
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(tokenizer_dir), local_files_only=True
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True
        )
        self.tokenizer.set_prefix_tokens(language=language, task='transcribe')
        self.initial_tokens = list(self.tokenizer.prefix_tokens)
        self.eos_token = self.tokenizer.eos_token_id
        self.no_speech_token = self.tokenizer.convert_tokens_to_ids(
            '<|nospeech|>'
        )
        if not isinstance(self.no_speech_token, int):
            self.no_speech_token = -1
        self.begin_suppress_tokens = set()
        self.suppress_tokens = set()
        generation_config = tokenizer_dir / 'generation_config.json'
        if generation_config.exists():
            try:
                config = json.loads(generation_config.read_text())
                self.begin_suppress_tokens = {
                    int(token) for token in config.get(
                        'begin_suppress_tokens', []
                    )
                }
                self.suppress_tokens = {
                    int(token) for token in config.get(
                        'suppress_tokens', []
                    )
                }
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        self.timestamp_begin = self.tokenizer.convert_tokens_to_ids(
            '<|0.00|>'
        )

        decoder_inputs = self.decoder.get_inputs()
        input_ids_input = next(
            (
                item for item in decoder_inputs
                if item.type == 'tensor(int64)'
            ),
            decoder_inputs[0],
        )
        self.input_ids_name = input_ids_input.name
        self.encoder_out_name = next(
            (
                item.name for item in decoder_inputs
                if item.name != self.input_ids_name
            ),
            decoder_inputs[1].name,
        )
        shape = input_ids_input.shape
        self.max_length = (
            shape[1]
            if len(shape) > 1 and isinstance(shape[1], int)
            else 448
        )
        self.providers = {
            'encoder': self.encoder.get_providers(),
            'decoder': self.decoder.get_providers(),
        }
        print(json.dumps({
            'ready': True,
            'providers': self.providers,
            'startup_s': round(time.monotonic() - started, 1),
        }, ensure_ascii=False), flush=True)

    def _decoder_logits(self, encoder_out, tokens):
        decoder_input = np.full(
            (1, self.max_length), self.eos_token, dtype=np.int64
        )
        decoder_input[0, :len(tokens)] = tokens
        outputs = self.decoder.run(None, {
            self.input_ids_name: decoder_input,
            self.encoder_out_name: encoder_out,
        })
        return outputs[0][0, len(tokens) - 1]

    def _suppress_invalid_tokens(self, logits, first_token):
        logits = np.array(logits, copy=True)
        for token in self.suppress_tokens:
            if 0 <= token < len(logits):
                logits[token] = -np.inf
        if first_token:
            for token in self.begin_suppress_tokens:
                if 0 <= token < len(logits):
                    logits[token] = -np.inf
        # The application uses transcription without timestamps.  Prevent
        # timestamp tokens from becoming ordinary text/control output.
        if isinstance(self.timestamp_begin, int) and self.timestamp_begin > 0:
            logits[self.timestamp_begin:] = -np.inf
        return logits

    def transcribe(self, audio):
        # The outer voice node has VAD, but keep a second gate here.  Whisper
        # is generative and can turn silence/echo into captions; never pass
        # those hallucinations to the LLM or to a robot command.
        if audio is None or len(audio) == 0:
            return ''
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if not np.isfinite(rms) or rms < 0.0035:
            return ''
        features = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors='np'
        )['input_features']
        encoder_out = self.encoder.run(
            None,
            {self.encoder.get_inputs()[0].name: features},
        )[0]
        tokens = list(self.initial_tokens)
        logits = self._decoder_logits(encoder_out, tokens)
        if 0 <= self.no_speech_token < len(logits):
            shifted = logits.astype(np.float64) - float(np.max(logits))
            probabilities = np.exp(shifted)
            probabilities /= max(float(np.sum(probabilities)), 1e-12)
            if float(probabilities[self.no_speech_token]) >= 0.65:
                return ''
        for _ in range(len(tokens), self.max_length):
            logits = self._suppress_invalid_tokens(
                logits, first_token=(len(tokens) == len(self.initial_tokens))
            )
            next_token = int(np.argmax(logits))
            if next_token == self.eos_token:
                break
            tokens.append(next_token)
            logits = self._decoder_logits(encoder_out, tokens)
        return self.tokenizer.decode(
            tokens[len(self.initial_tokens):], skip_special_tokens=True
        ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--model-size', default='medium')
    parser.add_argument('--language', default='el')
    args = parser.parse_args()
    try:
        model = WhisperONNX(
            args.model_dir,
            args.tokenizer_dir,
            args.config_dir,
            args.cache_dir,
            args.model_size,
            args.language,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            'error': f'{type(exc).__name__}: {exc}'
        }), flush=True)
        raise

    for line in sys.stdin:
        try:
            request = json.loads(line)
            audio = np.frombuffer(
                base64.b64decode(request['audio']),
                dtype=np.float32,
            )
            print(json.dumps({
                'text': model.transcribe(audio),
            }, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({
                'error': f'{type(exc).__name__}: {exc}'
            }), flush=True)


if __name__ == '__main__':
    main()
