"""Step 1: speech regions. The live chain's silero VAD, run over the whole recording."""
from __future__ import annotations

import numpy as np

from ..config import ANALYSIS_FRAME, ANALYSIS_RATE
from ..vad import SileroVAD

FRAME_S = ANALYSIS_FRAME / ANALYSIS_RATE


def vad_probs(audio: np.ndarray, progress=None) -> np.ndarray:
    model = SileroVAD()
    n = len(audio) // ANALYSIS_FRAME
    probs = np.zeros(n, np.float32)
    for i in range(n):
        probs[i] = model(audio[i * ANALYSIS_FRAME:(i + 1) * ANALYSIS_FRAME])
        if progress is not None and i % 2000 == 0:
            progress(i / max(n, 1), f"finding speech: {i * FRAME_S / 60:.0f} of {n * FRAME_S / 60:.0f} min")
    return probs


def speech_regions(probs: np.ndarray, threshold: float = 0.5, min_speech_s: float = 0.25,
                   min_silence_s: float = 0.3, pad_s: float = 0.1) -> list[tuple[float, float]]:
    """Hysteresis (enter at ``threshold``, leave below threshold - 0.15, like silero's own segmenter),
    then gaps shorter than ``min_silence_s`` closed, regions shorter than ``min_speech_s`` dropped, padded."""
    neg = max(threshold - 0.15, 0.01)
    regions, speaking, start = [], False, 0
    for i, p in enumerate(probs):
        if not speaking and p >= threshold:
            speaking, start = True, i
        elif speaking and p < neg:
            speaking = False
            regions.append([start, i])
    if speaking:
        regions.append([start, len(probs)])
    merged: list[list[int]] = []
    gap = round(min_silence_s / FRAME_S)
    for a, b in regions:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    total = len(probs) * FRAME_S
    out = []
    for a, b in merged:
        if (b - a) * FRAME_S < min_speech_s:
            continue
        out.append((max(0.0, a * FRAME_S - pad_s), min(total, b * FRAME_S + pad_s)))
    # Padding can make neighbours touch: join them.
    joined: list[tuple[float, float]] = []
    for a, b in out:
        if joined and a <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(b, joined[-1][1]))
        else:
            joined.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in joined]


def run(folder, audio: np.ndarray, progress, threshold: float = 0.5) -> dict:
    probs = vad_probs(audio, progress)
    regions = speech_regions(probs, threshold)
    duration = len(audio) / ANALYSIS_RATE
    speech = sum(b - a for a, b in regions)
    return {"regions": regions, "duration_s": round(duration, 2), "speech_s": round(speech, 2),
            "threshold": threshold, "frame_s": FRAME_S,
            # Kept at 10 per second for the eval sweeps (re-thresholding without re-running the model).
            "probs_100ms": [round(float(x), 3) for x in _decimate(probs, 3)]}


def _decimate(probs: np.ndarray, k: int) -> np.ndarray:
    n = len(probs) // k
    return probs[:n * k].reshape(n, k).max(axis=1) if n else probs
