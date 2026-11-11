"""Shared bits of the after-class pass: reading the recording, step files, progress, memory."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from ..config import ANALYSIS_RATE
from ..settings import write_json_atomic

log = logging.getLogger("focus_ear.post")

POST_DIR = "post"
STEPS = ("regions", "diarize", "asr", "merge")
OUTPUTS = {"regions": "post/regions.json", "diarize": "post/diarization.json", "asr": "post/asr.json",
           "merge": "final.json"}


def step_path(folder: Path, step: str) -> Path:
    return folder / OUTPUTS[step]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_step(folder: Path, step: str, data: dict) -> Path:
    p = step_path(folder, step)
    write_json_atomic(p, data, indent=None)
    return p


def load_audio(folder: Path) -> np.ndarray:
    """The session's audio.wav as 16 kHz mono float32 (resampled if it isn't)."""
    import soundfile as sf

    x, fs = sf.read(folder / "audio.wav", dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if fs != ANALYSIS_RATE:
        import soxr
        x = soxr.resample(x, fs, ANALYSIS_RATE).astype(np.float32)
    return np.ascontiguousarray(x)


class Progress:
    """Writes post/progress.json (step, fraction, message) at most twice a second, for the app."""

    def __init__(self, folder: Path, step: str):
        self.path = folder / POST_DIR / "progress.json"
        self.step = step
        self._last = 0.0

    def __call__(self, fraction: float, message: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.5:
            return
        self._last = now
        try:
            write_json_atomic(self.path, {"step": self.step, "fraction": round(min(max(fraction, 0.0), 1.0), 3),
                                          "message": message, "updated": time.time()}, indent=None)
        except OSError:
            pass


class PeakMemory:
    """Samples this process's memory footprint every 0.25 s; .peak_mb is the highest seen."""

    def __init__(self) -> None:
        from ..health import memory_mb
        self._read = memory_mb
        self.peak_mb = memory_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(1)
        self.peak_mb = max(self.peak_mb, self._read())

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self.peak_mb = max(self.peak_mb, self._read())
