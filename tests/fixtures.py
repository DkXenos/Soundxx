"""Shared test audio.

data/speech_16k.wav: 6 s cut from silero-vad's tests/data/test.wav (MIT,
Silero Team). Layout: speech 0-1.75 s, silence 1.75-4.5 s, speech 4.5-6 s.
"""
import wave
from pathlib import Path

import numpy as np

SPEECH_WAV = Path(__file__).parent / "data" / "speech_16k.wav"
SPEECH_SPANS = [(0.0, 1.75), (4.5, 6.0)]
SILENCE_SPAN = (1.75, 4.5)


def load_speech() -> np.ndarray:
    with wave.open(str(SPEECH_WAV)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def tone(freq: float, seconds: float, rate: int = 16000, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def frames(x: np.ndarray, size: int = 512):
    for i in range(0, len(x) - size + 1, size):
        yield x[i:i + size]
