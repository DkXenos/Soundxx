"""Worker thread: mic ring -> processing chain -> buds ring, plus timing metrics.

All heavy work (ML included, once it exists) runs here, never in an audio
callback. The chain is a list of AudioStage objects; an empty chain is pure
passthrough.

Extension points, deliberately not implemented yet:
  * Noise suppression: an AudioStage placed ahead of the gain stage.
  * Overlapping-speech separation: an AudioStage that replaces gating and
    returns only the selected speaker's separated signal.
  * Transcription: an AudioTap. It sees every processed block plus its
    BlockContext (speaker label included), and does its work off-thread.
"""
from __future__ import annotations

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .audio_io import AudioEngine, AudioStats
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class BlockContext:
    index: int          # running block counter
    adc_time: float     # time.monotonic() at which block[0] hit the mic's ADC
    samplerate: int
    # Filled in by the analysis stages once they exist:
    is_speech: bool | None = None
    speaker_id: int | None = None


class AudioStage(Protocol):
    """Transforms audio on its way to the buds. Must return a block of the same length."""
    name: str

    def process(self, block: np.ndarray, ctx: BlockContext) -> np.ndarray: ...

    def reset(self) -> None: ...


class AudioTap(Protocol):
    """Observes processed audio without changing it.

    Must return quickly: queue the work for another thread. ``block`` is
    reused after the call, so copy it if you keep it.
    """
    name: str

    def observe(self, block: np.ndarray, ctx: BlockContext) -> None: ...


class StageTimer:
    """Per-stage processing time, accumulated between reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # worker <-> reporter; never taken in an audio callback
        self._totals: dict[str, list[float]] = {}  # name -> [count, sum, max]

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            t = self._totals.setdefault(name, [0, 0.0, 0.0])
            t[0] += 1
            t[1] += seconds
            t[2] = max(t[2], seconds)

    def drain(self) -> dict[str, tuple[float, float]]:
        """Returns {stage: (mean_ms, max_ms)} since the last drain."""
        with self._lock:
            totals, self._totals = self._totals, {}
        return {name: (1e3 * s / n, 1e3 * m) for name, (n, s, m) in totals.items()}


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else -200.0


class Pipeline:
    SILENT_MIC_WARN_S = 2.0

    def __init__(self, engine: AudioEngine, cfg: Config,
                 stages: list[AudioStage] | None = None, taps: list[AudioTap] | None = None):
        self.engine = engine
        self.cfg = cfg
        self.stages = list(stages or [])
        self.taps = list(taps or [])
        self.timer = StageTimer()

        # Written by the worker, read by UI/metrics.
        self.in_db = -200.0
        self.out_db = -200.0
        self.blocks = 0
        self.skipped = 0            # samples skipped because the worker fell behind
        self.backlog_sum = 0        # input ring fill at each read, for the mean
        self.heard_signal = False   # any non-zero sample yet?

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def mic_silent(self) -> bool:
        """True if the mic has produced only exact zeros for a while.

        macOS delivers digital silence instead of an error when the terminal
        app lacks microphone permission; a real room is never exactly zero.
        """
        seconds = self.blocks * self.cfg.blocksize / self.cfg.samplerate
        return not self.heard_signal and seconds > self.SILENT_MIC_WARN_S

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pipeline-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        cfg, engine = self.cfg, self.engine
        bs = cfg.blocksize
        block = np.zeros(bs, dtype=np.float32)
        max_backlog = int(cfg.max_backlog_ms * cfg.samplerate / 1000)

        while not self._stop.is_set():
            backlog = engine.input_backlog()
            if backlog < bs:
                time.sleep(0.002)
                continue
            if backlog > max_backlog:
                # Fell behind real time: drop the oldest audio rather than let
                # latency grow without bound.
                self.skipped += engine.skip_input(backlog - bs)
                backlog = bs

            t_start = time.perf_counter()
            adc_time = engine.read_input(block)
            self.backlog_sum += backlog
            peak = float(np.abs(block).max())
            self.heard_signal = self.heard_signal or peak > 0.0
            self.in_db = _db(float(np.sqrt(np.mean(block * block))))

            ctx = BlockContext(index=self.blocks, adc_time=adc_time, samplerate=cfg.samplerate)
            out = block
            for stage in self.stages:
                t = time.perf_counter()
                out = stage.process(out, ctx)
                self.timer.add(stage.name, time.perf_counter() - t)
            for tap in self.taps:
                tap.observe(out, ctx)

            engine.write_output(out, adc_time)
            self.out_db = _db(float(np.sqrt(np.mean(out * out))))
            self.blocks += 1
            self.timer.add("total", time.perf_counter() - t_start)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    interval_s: float
    e2e_ms: float | None            # measured mic ADC -> buds DAC (as CoreAudio reports it)
    # Where that latency comes from (approximate; should roughly sum to e2e):
    in_device_ms: float
    in_queue_ms: float
    proc_ms: float
    out_buffer_ms: float
    out_device_ms: float
    stages: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (mean, max) ms
    rtf: float = 0.0                # processing time / audio time; must stay well below 1
    in_overflows: int = 0
    in_dropped_ms: float = 0.0
    out_underruns: int = 0
    out_underflows: int = 0
    out_skipped_ms: float = 0.0
    worker_skipped_ms: float = 0.0


class MetricsCollector:
    """Turns the ever-increasing counters into per-interval figures."""

    def __init__(self, engine: AudioEngine, pipeline: Pipeline):
        self.engine = engine
        self.pipeline = pipeline
        self._prev_stats = copy.copy(engine.stats)
        self._prev_blocks = pipeline.blocks
        self._prev_backlog = pipeline.backlog_sum
        self._prev_skipped = pipeline.skipped
        self._prev_time = time.monotonic()

    def snapshot(self) -> MetricsSnapshot:
        fs = self.engine.cfg.samplerate
        bs = self.engine.cfg.blocksize
        now = time.monotonic()
        cur: AudioStats = copy.copy(self.engine.stats)
        prev = self._prev_stats
        p = self.pipeline
        blocks = p.blocks - self._prev_blocks

        stages = p.timer.drain()
        proc_ms = stages.get("total", (0.0, 0.0))[0]
        e2e_n = cur.e2e_count - prev.e2e_count
        lvl_n = cur.out_level_count - prev.out_level_count
        interval = now - self._prev_time

        snap = MetricsSnapshot(
            interval_s=interval,
            e2e_ms=1e3 * (cur.e2e_sum - prev.e2e_sum) / e2e_n if e2e_n else None,
            in_device_ms=1e3 * cur.in_device_latency,
            # Time a block sat in the ring after its callback (the backlog includes the block itself).
            in_queue_ms=1e3 * ((p.backlog_sum - self._prev_backlog) / blocks - bs) / fs if blocks else 0.0,
            proc_ms=proc_ms,
            out_buffer_ms=1e3 * (cur.out_level_sum - prev.out_level_sum) / lvl_n / fs if lvl_n else 0.0,
            out_device_ms=1e3 * cur.out_device_latency,
            stages=stages,
            rtf=proc_ms / (1e3 * bs / fs),
            in_overflows=cur.in_overflows - prev.in_overflows,
            in_dropped_ms=1e3 * (cur.in_dropped - prev.in_dropped) / fs,
            out_underruns=cur.out_underruns - prev.out_underruns,
            out_underflows=cur.out_underflows - prev.out_underflows,
            out_skipped_ms=1e3 * (cur.out_skipped - prev.out_skipped) / fs,
            worker_skipped_ms=1e3 * (p.skipped - self._prev_skipped) / fs,
        )
        self._prev_stats = cur
        self._prev_blocks = p.blocks
        self._prev_backlog = p.backlog_sum
        self._prev_skipped = p.skipped
        self._prev_time = now
        return snap
