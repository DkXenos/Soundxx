"""Bring an existing recording (QuickTime .m4a, a phone's .mp3, a .wav…) in as a session."""
from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import ANALYSIS_RATE
from ..sessions import AUDIO_FILE, SessionStore, wav_seconds


def decode(path: Path) -> np.ndarray:
    """Any audio file -> 16 kHz mono float32. soundfile first (wav, flac, mp3, ogg), else PyAV (m4a, aac, …)."""
    import soxr
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:  # noqa: BLE001 - not a format libsndfile knows
        import av
        chunks, fs = [], None
        with av.open(str(path)) as f:
            stream = f.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=ANALYSIS_RATE)
            for frame in f.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
        x, fs = (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), ANALYSIS_RATE
    if fs != ANALYSIS_RATE:
        x = soxr.resample(x, fs, ANALYSIS_RATE)
    return np.ascontiguousarray(x, dtype=np.float32)


def import_recording(store: SessionStore, path: Path, title: str = "", course: str = "") -> dict:
    x = decode(path)
    if len(x) < ANALYSIS_RATE:
        raise ValueError("shorter than a second, or no audio track")
    started = datetime.fromtimestamp(path.stat().st_mtime - len(x) / ANALYSIS_RATE)
    data = store.create(title=title, course=course, started=started, status="recorded",
                        imported_file=str(path), devices={"input": f"imported: {path.name}"})
    out = store.path(data["id"]) / AUDIO_FILE
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ANALYSIS_RATE)
        w.writeframes(pcm.tobytes())
    return store.update(data["id"], duration_s=round(wav_seconds(out), 1), ended=data["started"])
