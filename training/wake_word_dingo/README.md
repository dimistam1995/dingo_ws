# Dingo wake word

This is the Dingo workspace's own wake-word pipeline. It creates a small
openWakeWord-compatible ONNX detector for `Hey Dingo`, `Έι Ντίγκο` and close
Greek pronunciations.

The XVF3800 USB stream is used as 16 kHz mono PCM from channel 0. The model
is intentionally separate from Whisper: the always-on detector is lightweight
and the slower speech-to-text model runs only after a wake event.

## Build the first model

```bash
cd /home/dimi/dingo_ws/training/wake_word_dingo
python3 generate_data.py
python3 extract_features.py
python3 train.py
python3 evaluate.py
```

The model is written to:

```text
/home/dimi/dingo_ws/models/wake_words/dingo.onnx
```

The first model is a bootstrap model made from TTS voices. For reliable use
on this particular room and speaker, record real positive examples and hard
negative examples with:

```bash
python3 record_real.py --positives 20 --negatives 30
python3 extract_features.py
python3 train.py --install
```

The recorder normally reads the live `RawAudio` topic from the running voice
service, so the service does not need to be stopped. Press Enter before each
take, speak only after the prompt, and keep the microphone in its normal robot
position.

The positive examples should contain only the exact phrase `Hey Dingo`; vary
distance, angle and volume. The negative examples should contain similar sounds
(`Dingo`, `Hey Dino`, `Ντίνο`, `Ντίνα`, `Νίκο`, etc.) and ordinary robot
commands. This prevents a short syllable or a nearby conversation from waking
the robot.
