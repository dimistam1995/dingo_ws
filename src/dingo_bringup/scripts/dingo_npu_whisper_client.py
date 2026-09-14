"""Isolated client for the AMD VitisAI Whisper worker.

The voice node runs with the ROS Python environment, while AMD's Ryzen AI
runtime lives in its dedicated virtual environment. Inference therefore runs
in a child process and communicates over JSON lines.
"""

import base64
import json
import os
import select
import subprocess
import time


class NPUWhisperClient:
    def __init__(self, python, model_size, language, model_dir,
                 tokenizer_dir, config_dir, cache_dir):
        worker = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'dingo_npu_whisper_worker.py',
        )
        if not os.path.isfile(worker):
            raise FileNotFoundError(f'Λείπει ο NPU worker: {worker}')
        if not os.path.isfile(python):
            raise FileNotFoundError(f'Λείπει το NPU Python runtime: {python}')

        env = os.environ.copy()
        xrt = '/opt/xilinx/xrt'
        venv = os.path.dirname(os.path.dirname(os.path.abspath(python)))
        site = os.path.join(venv, 'lib', 'python3.12', 'site-packages')
        library_dirs = [
            os.path.join(venv, 'deployment', 'lib'),
            os.path.join(venv, 'onnxruntime', 'lib'),
            os.path.join(site, 'flexmlrt', 'lib'),
            os.path.join(site, 'flexml', 'flexml_extras', 'lib'),
            os.path.join(site, 'voe', 'lib'),
            os.path.join(xrt, 'lib'),
        ]
        env['XILINX_XRT'] = xrt
        env['PATH'] = f'{xrt}/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = ':'.join(library_dirs) + ':' + env.get(
            'LD_LIBRARY_PATH', ''
        )
        flexmlrt = os.path.join(site, 'flexmlrt', 'lib', 'libflexmlrt.so')
        if os.path.exists(flexmlrt):
            env['LD_PRELOAD'] = flexmlrt + (
                ':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else ''
            )
        env['PYTHONPATH'] = f'{xrt}/python:' + env.get('PYTHONPATH', '')
        env['PYTHONUNBUFFERED'] = '1'

        self._proc = subprocess.Popen(
            [
                python,
                worker,
                '--model-dir', os.path.expanduser(model_dir),
                '--tokenizer-dir', os.path.expanduser(tokenizer_dir),
                '--config-dir', os.path.expanduser(config_dir),
                '--cache-dir', os.path.expanduser(cache_dir),
                '--model-size', str(model_size),
                '--language', str(language),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Keep the JSON-lines stdout channel clean.  The AMD runtime can
            # be verbose during graph compilation; inheriting/redirecting it
            # avoids filling a pipe and blocking the worker before readiness.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            ready = self._read_json(900.0)
            if ready is None:
                stderr = self._read_stderr()
                raise RuntimeError(
                    'Ο NPU Whisper worker τερμάτισε πριν είναι έτοιμος'
                    + (f': {stderr}' if stderr else '')
                )
            if not ready.get('ready'):
                raise RuntimeError(
                    ready.get('error', 'Αποτυχία NPU Whisper worker')
                )
            self.providers = ready.get('providers', {})
            provider_text = json.dumps(self.providers, ensure_ascii=False)
            if 'VitisAIExecutionProvider' not in provider_text:
                raise RuntimeError(
                    'Το Whisper φορτώθηκε χωρίς VitisAIExecutionProvider '
                    f'(μόνο CPU): {provider_text}'
                )
        except Exception:
            self.close()
            raise

    def _readline(self, timeout):
        if self._proc.poll() is not None:
            return ''
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(
                f'Ο NPU Whisper worker δεν απάντησε σε {timeout:.0f}s'
            )
        return self._proc.stdout.readline().strip()

    def _read_json(self, timeout):
        """Read the next JSON message, ignoring AMD runtime diagnostics."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self._readline(remaining)
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # VitisAI may print compilation/profiling lines to stdout
                # before the worker's JSON-lines protocol message.
                continue

    def _read_stderr(self):
        if self._proc.stderr is None:
            return ''
        try:
            return self._proc.stderr.read().strip()[-500:]
        except (OSError, ValueError):
            return ''

    def transcribe(self, audio):
        if self._proc.poll() is not None:
            detail = self._read_stderr()
            raise RuntimeError(
                'Ο NPU Whisper worker σταμάτησε'
                + (f': {detail}' if detail else '')
            )
        payload = base64.b64encode(
            audio.astype('float32').tobytes()
        ).decode('ascii')
        self._proc.stdin.write(json.dumps({'audio': payload}) + '\n')
        self._proc.stdin.flush()
        result = self._read_json(180.0)
        if result is None:
            detail = self._read_stderr()
            raise RuntimeError(
                'Ο NPU Whisper worker επέστρεψε κενή απάντηση'
                + (f': {detail}' if detail else '')
            )
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result.get('text', '').strip()

    def close(self):
        proc = getattr(self, '_proc', None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
