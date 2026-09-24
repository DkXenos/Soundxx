"""Speaker embeddings (phase 2, not implemented yet).

Plan: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) over a ~1.0 s
window of VAD-positive audio, recomputed every few blocks. The device is
chosen by benchmark: for one-second inputs MPS transfer overhead can make it
slower than the CPU, so "auto" times both at startup and keeps the faster.
"""
from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 192


class SpeakerEmbedder:
    def __init__(self, device: str = "auto"):
        self.device = device

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """~1 s of 16 kHz speech -> L2-normalised float32 vector of EMBEDDING_DIM."""
        raise NotImplementedError("embeddings arrive in phase 2")
