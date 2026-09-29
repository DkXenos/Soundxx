"""The unprocessed mic of every live run, saved for replaying through --file.

A pipeline raw tap: the worker hands it each block exactly as the mic
delivered it, before the high-pass or anything else. It's resampled to
16 kHz mono (the models' rate; this is for analysis, not listening) and
queued; a thread of its own writes it to a 16-bit WAV about once a second.
Each write rewrites the WAV header's lengths and is fsynced, so after a
crash the file is valid and ends at most a second or two early.
"""
from __future__ import annotations

import logging
import os
import threading
import wave
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soxr

if TYPE_CHECKING:
    from .pipeline import BlockContext

log = logging.getLogger(__name__)


class RawRecorder:
    """AudioTap: raw mic blocks -> ``path`` (a new 16-bit mono WAV at ``rate``)."""

    name = "recorder"

    def __init__(self, path: Path, in_rate: int, rate: int = 16_000, flush_s: float = 1.0,
                 max_queue_s: float = 30.0):
        self.path = path
        self.rate = rate
        self._resampler = soxr.ResampleStream(in_rate, rate, 1, dtype="float32") if in_rate != rate else None
        self._file = open(path, "xb")  # never overwrite an earlier lecture
        self._wav = wave.open(self._file, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(rate)
        self._queue: deque[np.ndarray] = deque()
        self._queued = 0
        self._max_queued = int(max_queue_s * rate)
        self._lock = threading.Lock()      # worker <-> writer thread; never taken by an audio callback
        self._flush_s = flush_s
        self._stop = threading.Event()
        # Read by the UI; only ever increase.
        self.frames = 0                    # written to disk
        self.dropped = 0                   # samples lost because the disk fell 30 s behind
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    @property
    def seconds(self) -> float:
        return self.frames / self.rate

    def observe(self, block: np.ndarray, ctx: BlockContext | None = None) -> None:
        """Worker thread: resample (always a copy) and queue. Never touches the disk."""
        x = block.copy() if self._resampler is None else self._resampler.resample_chunk(block)
        if not len(x):
            return
        with self._lock:
            self._queue.append(x)
            self._queued += len(x)
            while self._queued > self._max_queued:
                self._queued -= len(old := self._queue.popleft())
                self.dropped += len(old)

    def close(self) -> None:
        """Write what's queued and close the file (idempotent)."""
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._flush_s):
                self._write()
            self._write()
        except OSError as exc:  # disk full, unplugged volume: the audio must carry on regardless
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("recording stopped: %s (%s so far is saved in %s)", self.error,
                      _duration(self.seconds), self.path)
        finally:
            try:
                self._wav.close()
                self._file.close()
            except OSError:
                pass

    def _write(self) -> None:
        with self._lock:
            chunks, self._queue, self._queued = list(self._queue), deque(), 0
        if not chunks:
            return
        pcm = np.round(np.clip(np.concatenate(chunks), -1.0, 1.0) * 32767).astype("<i2")
        self._wav.writeframes(pcm.tobytes())  # also rewrites the header's lengths
        self._file.flush()
        os.fsync(self._file.fileno())
        self.frames += len(pcm)


def _duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
