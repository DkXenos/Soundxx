"""Voice activity detection (phase 2, not implemented yet).

Plan: silero-vad via ONNX Runtime. It consumes exactly 512-sample frames at
16 kHz, the same as our block size, so each pipeline block gets one
speech probability with no rebuffering.
"""
from __future__ import annotations

import numpy as np


class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.5, samplerate: int = 16_000):
        self.threshold = threshold
        self.samplerate = samplerate

    def speech_probability(self, frame: np.ndarray) -> float:
        """Probability in [0, 1] that this 512-sample frame contains speech."""
        raise NotImplementedError("VAD arrives in phase 2")

    def reset(self) -> None:
        raise NotImplementedError("VAD arrives in phase 2")
