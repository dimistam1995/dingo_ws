#!/usr/bin/env python3
"""Generate Greek/English synthetic data for the Hey Dingo wake word."""

import asyncio
import glob
import os
import subprocess

import edge_tts


RAW_DIR = 'raw'
WAV_DIR = 'wav'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)

GREEK_VOICES = ['el-GR-AthinaNeural', 'el-GR-NestorasNeural']
ENGLISH_VOICES = ['en-US-AriaNeural', 'en-US-GuyNeural']
VARIANTS = [('-15%', '-25Hz'), ('+0%', '+0Hz'), ('+15%', '+25Hz')]

POSITIVE_GREEK = [
    'Έι Ντίγκο', 'Έι Ντίγκο!', 'Ει Ντινγκο', 'Έι Ντίγκο άκου με',
    'Έι Ντίγκο έλα εδώ', 'Έι Ντίγκο με ακούς',
]
POSITIVE_ENGLISH = [
    'Hey Dingo', 'Hey Dingo!', 'Hey Dingo listen', 'Hey Dingo come here',
]

NEGATIVE_GREEK = [
    'Γεια σου', 'Πήγαινε στην κουζίνα', 'Σταμάτα', 'Τι ώρα είναι',
    'Κάνε μια βόλτα', 'Κάνε μια στροφή', 'Ποιος μίλησε τώρα',
    'Πόση μνήμη έχω', 'Πόσος χώρος μένει', 'Τι βλέπεις',
    'Ντίνο', 'Ντίνο μου', 'Ντίνα', 'Νίκο', 'Νίκο μου', 'Ντίνο άκου',
    'Ντίγκο', 'Έι Ντίνο', 'Έι Ντίνα', 'Έι Νίκο',
    'Δεν καταλαβαίνω', 'Ευχαριστώ πολύ', 'Πού είναι το ρομπότ',
    'Το ρομπότ πήγε στην κουζίνα', 'Άκου με ρομπότ',
]
NEGATIVE_ENGLISH = [
    'Hello', 'Stop', 'Go to the kitchen', 'What time is it',
    'What do you see', 'Dino', 'Dina', 'Niko', 'Dino listen',
    'Hey Dino', 'Hey Dina', 'Hey Niko', 'Dingo', 'Robot', 'My robot',
    'Thank you', 'Come here',
]


JOB_COUNTER = 0


def jobs_for(label, texts, voices, variants):
    global JOB_COUNTER
    result = []
    for voice in voices:
        for text in texts:
            for rate, pitch in variants:
                result.append((label, text, voice, rate, pitch,
                               f'hey_{label}_{JOB_COUNTER:05d}_{voice}'))
                JOB_COUNTER += 1
    return result


JOBS = []
JOBS.extend(jobs_for('pos', POSITIVE_GREEK, GREEK_VOICES, VARIANTS))
JOBS.extend(jobs_for('pos', POSITIVE_ENGLISH, ENGLISH_VOICES, VARIANTS))
JOBS.extend(jobs_for('neg', NEGATIVE_GREEK, GREEK_VOICES, VARIANTS))
JOBS.extend(jobs_for('neg', NEGATIVE_ENGLISH, ENGLISH_VOICES, VARIANTS))

SEM = asyncio.Semaphore(8)


async def generate_one(job):
    label, phrase, voice, rate, pitch, name = job
    mp3_path = os.path.join(RAW_DIR, f'{name}.mp3')
    wav_path = os.path.join(WAV_DIR, f'{name}.wav')
    if os.path.exists(wav_path):
        return label, wav_path
    async with SEM:
        try:
            await edge_tts.Communicate(
                phrase, voice, rate=rate, pitch=pitch
            ).save(mp3_path)
            subprocess.run(
                [
                    'ffmpeg', '-y', '-loglevel', 'error', '-i', mp3_path,
                    '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', wav_path,
                ],
                check=True,
            )
            return label, wav_path
        except Exception as exc:  # noqa: BLE001
            print(f'FAILED {name}: {exc}')
            return None


async def main():
    print(
        f'Generating {len(JOBS)} clips '
        f'(positive={sum(j[0] == "pos" for j in JOBS)}, '
        f'negative={sum(j[0] == "neg" for j in JOBS)})'
    )
    generated = await asyncio.gather(*(generate_one(job) for job in JOBS))
    # The Dashboard recorder writes directly to real_wav/.  Discover those
    # files here instead of relying on an old manifest, otherwise a perfectly
    # valid real recording would silently be omitted from retraining.
    real = []
    for label in ('pos', 'neg'):
        for path in sorted(glob.glob(os.path.join('real_wav', f'real_{label}_*.wav'))):
            real.append((label, path))
    with open('manifest.tsv', 'w', encoding='utf-8') as manifest:
        for item in generated:
            if item is not None:
                manifest.write(f'{item[0]}\t{item[1]}\n')
        for label, path in real:
            manifest.write(f'{label}\t{path}\n')
    print(f'Done. Kept {len(real)} real recordings.')


if __name__ == '__main__':
    asyncio.run(main())
