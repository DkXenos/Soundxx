"""Worker thread: mic ring -> analysis + processing chain -> buds ring, plus metrics.

All heavy work runs here, never in an audio callback. Each block (~32 ms at
the stream rate) goes through two paths:

  analysis  block -> 16 kHz resampler -> 512-sample frames -> FrameAnalyzers
            (Stage 1 VAD; Stage 3 speaker embeddings) which annotate the
            block's BlockContext
  audio     block -> AudioStages (Stage 2 noise gate; Stage 4 speaker gain)
            which read the context and change the audio -> output ring

Extension points, deliberately not implemented yet:
  * Noise suppression: an AudioStage ahead of the gate.
  * Overlapping-speech separation: an AudioStage that replaces gating and
    returns only the selected speaker's separated signal.
  * Transcription: an AudioTap. It sees every processed block plus its
    BlockContext (speaker label included), and does its work off-thread.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import soxr

from .config import ANALYSIS_FRAME, ANALYSIS_RATE, Config

if TYPE_CHECKING:
    from .audio_io import AudioEngine, AudioStats

log = logging.getLogger(__name__)

MIC_SILENT_HINT = ("the mic is delivering pure digital silence: give your terminal app microphone "
                   "access (System Settings > Privacy & Security > Microphone), then restart it")


@dataclass
class BlockContext:
    index: int          # running block counter
    adc_time: float     # time.monotonic() at which block[0] hit the mic's ADC
    samplerate: int     # stream (playback) rate of the block
    # Analysis results. A block doesn't always complete a model frame, so
    # each block starts from the previous block's values.
    vad_prob: float = 0.0           # Stage 1
    is_speech: bool = False         # Stage 1, with hysteresis
    speaker_id: int | None = None   # Stage 3


class FrameAnalyzer(Protocol):
    """Consumes the 16 kHz analysis stream in 512-sample frames and annotates the context.

    Runs synchronously on the worker thread, so it must stay well under a
    frame's 32 ms. A heavy model (Stage 3's speaker embeddings) hands its work
    to its own thread and applies results as they arrive; give such an
    analyzer a close() method and the pipeline stops the thread with it.
    """
    name: str

    def analyze(self, frame: np.ndarray, ctx: BlockContext) -> None: ...

    def reset(self) -> None: ...


class AudioStage(Protocol):
    """Transforms audio on its way to the buds. Must return a block of the same length.

    A stage that delays the audio (lookahead) also sets ``delay_samples``, so
    latency is still measured correctly.
    """
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


class AnalysisFeed:
    """Resamples stream-rate audio to 16 kHz and cuts it into 512-sample model frames.

    soxr's streaming resampler emits in bursts (about 540 samples on most
    calls, occasionally none), so one block can yield zero, one or two frames.
    """

    def __init__(self, in_rate: int):
        self._resampler = None
        if in_rate != ANALYSIS_RATE:
            self._resampler = soxr.ResampleStream(in_rate, ANALYSIS_RATE, 1, dtype="float32")
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        x = block if self._resampler is None else self._resampler.resample_chunk(block)
        buf = np.concatenate((self._pending, x))  # always a copy: ``block`` gets reused
        n = len(buf) // ANALYSIS_FRAME
        self._pending = buf[n * ANALYSIS_FRAME:]
        return [buf[i * ANALYSIS_FRAME:(i + 1) * ANALYSIS_FRAME] for i in range(n)]


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

    def __init__(self, engine: AudioEngine, cfg: Config, analyzers: list[FrameAnalyzer] | None = None,
                 stages: list[AudioStage] | None = None, taps: list[AudioTap] | None = None):
        """Create after engine.start(): the stream rate is only known then."""
        self.engine = engine
        self.cfg = cfg
        self.analyzers = list(analyzers or [])
        self.stages = list(stages or [])
        self.taps = list(taps or [])
        self.timer = StageTimer()
        self._feed = AnalysisFeed(engine.samplerate)
        # Total lookahead delay of the stages: output samples were captured this much earlier.
        self.delay_samples = sum(getattr(stage, "delay_samples", 0) for stage in self.stages)
        self._ctx = BlockContext(index=-1, adc_time=0.0, samplerate=engine.samplerate)

        # Written by the worker, read by UI/metrics. Counters only increase.
        self.ctx: BlockContext | None = None   # the latest block's context
        self.in_db = -200.0
        self.out_db = -200.0
        self.blocks = 0
        self.speech_blocks = 0
        self.vad_prob_sum = 0.0
        self.dropped = 0            # samples dropped because the worker fell behind
        self.drop_events = 0
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
        seconds = self.blocks * self.cfg.block_ms / 1000
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
        for analyzer in self.analyzers:
            close = getattr(analyzer, "close", None)  # analyzers with their own threads
            if close is not None:
                close()

    def _run(self) -> None:
        engine = self.engine
        bs = engine.blocksize
        block = np.zeros(bs, dtype=np.float32)
        max_backlog = max(2 * bs, int(self.cfg.max_backlog_ms * engine.samplerate / 1000))

        while not self._stop.is_set():
            backlog = engine.input_backlog()
            if backlog < bs:
                time.sleep(0.002)
                continue
            if backlog > max_backlog:
                # Fell behind real time: drop the oldest audio, keeping the
                # newest block, rather than let latency grow.
                self.dropped += engine.skip_input(backlog - bs)
                self.drop_events += 1
                backlog = bs

            t_start = time.perf_counter()
            adc_time = engine.read_input(block)
            self.backlog_sum += backlog
            out = self.process_block(block, adc_time)
            engine.write_output(out, adc_time - self.delay_samples / engine.samplerate)
            self.timer.add("total", time.perf_counter() - t_start)

    def process_block(self, block: np.ndarray, adc_time: float = 0.0) -> np.ndarray:
        """Analyse one stream-rate block and run it through the stages.

        May modify ``block`` in place; returns the audio to play. Separate
        from the ring handling so it can be driven offline in tests.
        """
        timer = self.timer
        self.heard_signal = self.heard_signal or bool(block.any())
        self.in_db = _db(float(np.sqrt(np.mean(block * block))))
        ctx = self._ctx = dataclasses.replace(self._ctx, index=self.blocks, adc_time=adc_time)

        t = time.perf_counter()
        frames = self._feed.push(block)
        timer.add("resample", time.perf_counter() - t)
        for frame in frames:
            for analyzer in self.analyzers:
                t = time.perf_counter()
                analyzer.analyze(frame, ctx)
                timer.add(analyzer.name, time.perf_counter() - t)

        out = block
        for stage in self.stages:
            t = time.perf_counter()
            out = stage.process(out, ctx)
            timer.add(stage.name, time.perf_counter() - t)
        for tap in self.taps:
            tap.observe(out, ctx)

        self.out_db = _db(float(np.sqrt(np.mean(out * out))))
        self.speech_blocks += ctx.is_speech
        self.vad_prob_sum += ctx.vad_prob
        self.ctx = ctx
        self.blocks += 1
        return out


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
    delay_ms: float                 # lookahead delay lines in the stages
    out_buffer_ms: float
    out_device_ms: float
    stages: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (mean, max) ms
    block_ms: float = 0.0
    rtf: float = 0.0                # processing time / audio time; must stay well below 1
    speech_pct: float = 0.0         # share of blocks the VAD called speech
    vad_prob_mean: float = 0.0
    in_overflows: int = 0
    in_dropped_ms: float = 0.0
    out_underruns: int = 0
    out_underflows: int = 0
    out_skipped_ms: float = 0.0
    worker_dropped_ms: float = 0.0
    worker_drop_events: int = 0


class MetricsCollector:
    """Turns the ever-increasing counters into per-interval figures."""

    def __init__(self, engine: AudioEngine, pipeline: Pipeline):
        self.engine = engine
        self.pipeline = pipeline
        self._prev_stats = copy.copy(engine.stats)
        self._prev = self._pipeline_counters()
        self._prev_time = time.monotonic()

    def _pipeline_counters(self) -> dict[str, float]:
        p = self.pipeline
        return {"blocks": p.blocks, "backlog": p.backlog_sum, "speech": p.speech_blocks,
                "prob": p.vad_prob_sum, "dropped": p.dropped, "drops": p.drop_events}

    def snapshot(self) -> MetricsSnapshot:
        fs, bs = self.engine.samplerate, self.engine.blocksize
        now = time.monotonic()
        cur: AudioStats = copy.copy(self.engine.stats)
        prev = self._prev_stats
        counters = self._pipeline_counters()
        d = {k: counters[k] - self._prev[k] for k in counters}
        blocks = d["blocks"]

        stages = self.pipeline.timer.drain()
        proc_ms = stages.get("total", (0.0, 0.0))[0]
        block_ms = 1e3 * bs / fs
        e2e_n = cur.e2e_count - prev.e2e_count
        lvl_n = cur.out_level_count - prev.out_level_count

        snap = MetricsSnapshot(
            interval_s=now - self._prev_time,
            e2e_ms=1e3 * (cur.e2e_sum - prev.e2e_sum) / e2e_n if e2e_n else None,
            in_device_ms=1e3 * cur.in_device_latency,
            # Time a block sat in the ring after its callback (the backlog includes the block itself).
            in_queue_ms=1e3 * (d["backlog"] / blocks - bs) / fs if blocks else 0.0,
            proc_ms=proc_ms,
            delay_ms=1e3 * self.pipeline.delay_samples / fs,
            out_buffer_ms=1e3 * (cur.out_level_sum - prev.out_level_sum) / lvl_n / fs if lvl_n else 0.0,
            out_device_ms=1e3 * cur.out_device_latency,
            stages=stages,
            block_ms=block_ms,
            rtf=proc_ms / block_ms,
            speech_pct=100.0 * d["speech"] / blocks if blocks else 0.0,
            vad_prob_mean=d["prob"] / blocks if blocks else 0.0,
            in_overflows=cur.in_overflows - prev.in_overflows,
            in_dropped_ms=1e3 * (cur.in_dropped - prev.in_dropped) / fs,
            out_underruns=cur.out_underruns - prev.out_underruns,
            out_underflows=cur.out_underflows - prev.out_underflows,
            out_skipped_ms=1e3 * (cur.out_skipped - prev.out_skipped) / fs,
            worker_dropped_ms=1e3 * d["dropped"] / fs,
            worker_drop_events=int(d["drops"]),
        )
        self._prev_stats = cur
        self._prev = counters
        self._prev_time = now
        return snap
