#!/usr/bin/env python3
"""Small, local speaker gallery used by the Dingo voice assistant.

This module intentionally contains no ROS or audio-device code.  The voice
node owns capture and transcription; this class only turns a finished 16 kHz
speech segment into an embedding and compares it with an explicitly enrolled
gallery.  The gallery stores only normalized embeddings and names, never raw
recordings.
"""

import json
import os
import tempfile
import threading
from pathlib import Path


class SpeakerGallery:
    """Persistent name -> normalized embedding mapping."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.lock = threading.RLock()
        self.embeddings = {}
        self.load()

    @staticmethod
    def _clean_name(name):
        value = ' '.join(str(name or '').split()).strip()
        if not value:
            raise ValueError('Χρειάζεται όνομα για την εγγραφή φωνής.')
        if len(value) > 48:
            raise ValueError('Το όνομα είναι πολύ μεγάλο (μέχρι 48 χαρακτήρες).')
        return value

    @staticmethod
    def _normalize(values):
        try:
            import numpy as np

            vector = np.asarray(values, dtype=np.float32).reshape(-1)
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError('Μη έγκυρο embedding φωνής.') from exc
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError('Το embedding φωνής είναι κενό.')
        return (vector / norm).astype(np.float32).tolist()

    def load(self):
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            payload = {}
        raw = payload.get('speakers', {}) if isinstance(payload, dict) else {}
        loaded = {}
        if isinstance(raw, dict):
            for raw_name, values in raw.items():
                try:
                    name = self._clean_name(raw_name)
                    loaded[name] = self._normalize(values)
                except ValueError:
                    continue
        with self.lock:
            self.embeddings = loaded

    def names(self):
        with self.lock:
            return sorted(self.embeddings)

    def save(self):
        with self.lock:
            payload = {
                'version': 1,
                'speakers': dict(sorted(self.embeddings.items())),
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{self.path.name}.',
            suffix='.tmp',
            dir=str(self.path.parent),
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

    def upsert(self, name, embedding):
        name = self._clean_name(name)
        normalized = self._normalize(embedding)
        with self.lock:
            self.embeddings[name] = normalized
        self.save()
        return name

    def remove(self, name):
        name = self._clean_name(name)
        with self.lock:
            existed = self.embeddings.pop(name, None) is not None
        if existed:
            self.save()
        return existed

    def match(self, embedding):
        """Return ``(name, cosine_score)`` for the closest enrolled voice."""
        normalized = self._normalize(embedding)
        try:
            import numpy as np

            query = np.asarray(normalized, dtype=np.float32)
        except (ImportError, ValueError) as exc:
            raise ValueError('Δεν είναι διαθέσιμο το numpy για αναγνώριση φωνής.') from exc
        best_name = None
        best_score = -1.0
        with self.lock:
            items = list(self.embeddings.items())
        for name, values in items:
            candidate = np.asarray(values, dtype=np.float32)
            if candidate.shape != query.shape:
                continue
            score = float(np.dot(query, candidate))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score if best_name is not None else None


class SpeakerRecognizer:
    """Lazy CPU speaker encoder plus gallery matching."""

    def __init__(self, gallery_path, threshold=0.72):
        self.gallery = SpeakerGallery(gallery_path)
        self.threshold = float(threshold)
        self.encoder = None
        self.encoder_lock = threading.Lock()

    def _get_encoder(self):
        with self.encoder_lock:
            if self.encoder is None:
                # Resemblyzer's reference audio helper imports librosa.  On
                # this ROS image librosa's optional numba/coverage pair is
                # incompatible, so patch only the mel frontend with the
                # equivalent NumPy/STFT implementation below.  The trained
                # Resemblyzer encoder itself remains unchanged and CPU-only.
                from resemblyzer import VoiceEncoder, audio as resemblyzer_audio

                resemblyzer_audio.wav_to_mel_spectrogram = self._mel_spectrogram

                self.encoder = VoiceEncoder(device='cpu', verbose=False)
            return self.encoder

    @staticmethod
    def _mel_spectrogram(wav):
        """Librosa-compatible 16 kHz / 40-bin mel frontend without librosa."""
        import numpy as np

        values = np.asarray(wav, dtype=np.float32).reshape(-1)
        sample_rate = 16000
        n_fft = 400
        hop = 160
        n_mels = 40
        if values.size == 0:
            return np.zeros((0, n_mels), dtype=np.float32)
        # librosa's default center=True, pad_mode='constant'.
        padded = np.pad(values, (n_fft // 2, n_fft // 2), mode='constant')
        frame_count = 1 + max(0, (len(padded) - n_fft) // hop)
        positions = (
            np.arange(frame_count)[:, None] * hop
            + np.arange(n_fft)[None, :]
        )
        window = np.hanning(n_fft).astype(np.float32)
        frames = padded[positions] * window
        power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2

        def hz_to_mel(frequency):
            frequency = np.asarray(frequency, dtype=np.float32)
            return np.where(
                frequency < 1000.0,
                frequency / (200.0 / 3.0),
                15.0 + 27.0 * np.log10(np.maximum(frequency, 1.0) / 1000.0),
            )

        def mel_to_hz(mel):
            mel = np.asarray(mel, dtype=np.float32)
            return np.where(
                mel < 15.0,
                mel * (200.0 / 3.0),
                1000.0 * np.power(10.0, (mel - 15.0) / 27.0),
            )

        mel_edges = np.linspace(
            float(hz_to_mel(0.0)),
            float(hz_to_mel(sample_rate / 2.0)),
            n_mels + 2,
        )
        hz_edges = mel_to_hz(mel_edges)
        bins = np.floor((n_fft + 1) * hz_edges / sample_rate).astype(int)
        frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for index in range(n_mels):
            left, center, right = bins[index:index + 3]
            if center > left:
                filters[index, left:center] = (
                    frequencies[left:center] - hz_edges[index]
                ) / (hz_edges[index + 1] - hz_edges[index])
            if right > center:
                filters[index, center:right] = (
                    hz_edges[index + 2] - frequencies[center:right]
                ) / (hz_edges[index + 2] - hz_edges[index + 1])
            # librosa's default Slaney normalization makes each triangle's
            # area comparable across the mel scale.
            width = hz_edges[index + 2] - hz_edges[index]
            if width > 0:
                filters[index] *= 2.0 / width
        return np.maximum(0.0, power @ filters.T).astype(np.float32)

    @staticmethod
    def _preprocess(values):
        """Normalize a 16 kHz segment without librosa's optional dependencies."""
        import numpy as np

        values = np.asarray(values, dtype=np.float32).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1.5:
            values = values / 32768.0
        if values.size:
            rms = float(np.sqrt(np.mean(np.square(values))))
            if rms > 1e-6:
                target_rms = 10 ** (-30.0 / 20.0)
                if rms < target_rms:
                    values = values * min(8.0, target_rms / rms)
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def embed(self, samples):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                'Λείπει το resemblyzer ή το numpy για αναγνώριση φωνής.'
            ) from exc
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size < 4800:  # 0.3 s at 16 kHz is too short for a stable voice vector.
            raise ValueError('Το ηχητικό απόσπασμα είναι πολύ μικρό.')
        wav = self._preprocess(values)
        if wav.size < 4800:
            raise ValueError('Το ηχητικό απόσπασμα είναι πολύ μικρό.')
        embedding = self._get_encoder().embed_utterance(wav)
        return np.asarray(embedding, dtype=np.float32)

    def identify(self, samples):
        embedding = self.embed(samples)
        name, score = self.gallery.match(embedding)
        recognized = (
            name
            if name is not None and score is not None and score >= self.threshold
            else None
        )
        return {
            'name': recognized,
            'candidate': name,
            'score': score,
            'embedding': embedding.tolist(),
        }

    def enroll_embeddings(self, name, embeddings):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Λείπει το numpy για εγγραφή φωνής.') from exc
        vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in embeddings]
        vectors = [item for item in vectors if item.size]
        if not vectors:
            raise ValueError('Δεν συγκεντρώθηκε έγκυρο δείγμα φωνής.')
        dimensions = {item.size for item in vectors}
        if len(dimensions) != 1:
            raise ValueError('Τα δείγματα φωνής έχουν διαφορετική μορφή.')
        mean = np.mean(np.stack(vectors), axis=0)
        return self.gallery.upsert(name, mean)
