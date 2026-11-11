"""Worker thread: mic ring -> analysis + processing chain -> buds ring, plus metrics.

All heavy work runs here, never in an audio callback. Each block (~32 ms at
the stream rate) is conditioned, then goes through two paths.

Sample rates. "Stream rate" is the mic's native rate, 44.1 kHz on a MacBook;
both PortAudio streams run at it, so CoreAudio never resamples.

  mic 44.1k ─► high-pass 80 Hz (44.1k) ─► raw ──┬─────────────────────────────────────────────────────────► raw
                                                 └─► polyphase ─► 48k ─► DeepFilterNet3 ─► 48k ─► polyphase ─► 44.1k ─► denoised
                                                     (10 ms frames at 48 kHz; the whole detour is a fixed 41 ms at 44.1 kHz)

  --denoise-scope    analysis (VAD, embeddings, Whisper)    playback
  all                denoised                                denoised
  playback           raw, undelayed                          denoised
  models             denoised                                raw, delayed 41 ms to stay in step

  analysis  44.1k ─► soxr ─► 16k ─► 512-sample frames ─► FrameAnalyzers
            (Stage 1 VAD; Stage 3 speaker embeddings) annotate the block's
            BlockContext; then Stage 5's utterance segmenter reads those
            annotations and hands 16 kHz utterances to the ASR thread
  playback  44.1k ─► AudioStages (Stage 2 gate; Stage 4 speaker gain; AGC;
            limiter), which read the context ─► clip check ─► output ring ─► Buds, 44.1k

Under "playback" the models hear each sound 41 ms before the Buds play it,
which works like free lookahead for the gate. Under "models" the raw audio
is delayed by the denoiser's delay so decisions still land on the audio they
were made from; so every scope adds the same latency.

  --mode listen      no speaker stages and no lookahead: high-pass, denoise,
                     gate, AGC, limiter. Adds the A/B switch below.
  --mode full        everything above.

Around the chain:
  raw taps   see each block exactly as the mic delivered it, before the
             high-pass (the raw recording).
  A/B        (listen mode) crossfades the output between the processed
             chain and the raw mic delayed by the same amount, so switching
             is click-free and the timing doesn't jump.
  late bypass  the denoiser's model runs on its own thread (live), and the
             worker waits for it only as long as the audio already queued for
             the output lasts. A block that misses that deadline plays
             undenoised but still in step, instead of the output running dry.
             Counted in ``denoise_skips``.

Pause (set_paused): the worker keeps reading the mic, so the level meter
stays alive, but nothing is recorded, analysed, transcribed or played. The
utterance being collected is closed first, so it still gets transcribed.
``media_time`` doesn't advance while paused.

Stage swaps (set_stages): the playback stages can be replaced between two
blocks (switching between listen and focus mode mid-session). The delays
are recomputed, so the latency accounting and the A/B delay stay right.

Extension points, deliberately not implemented yet:
  * Overlapping-speech separation: an AudioStage that replaces gating and
    returns only the selected speaker's separated signal.
  * Anything that needs the processed (gated, attenuated) audio: an AudioTap.
    Transcription is a FrameAnalyzer instead, because Whisper wants the
    16 kHz speech, not what the buds hear.
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
from .dsp import ClipMeter
from .gain import DelayLine, SmoothGain

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
    # Position of block[0] in the recording, in seconds: stops while paused,
    # so transcript times stay positions in audio.wav. Equals the file
    # position in --file mode.
    media_time: float = 0.0
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

    SCOPES = ("all", "playback", "models")

    def __init__(self, engine: AudioEngine, cfg: Config, analyzers: list[FrameAnalyzer] | None = None,
                 stages: list[AudioStage] | None = None, taps: list[AudioTap] | None = None,
                 highpass: AudioStage | None = None, denoiser: AudioStage | None = None,
                 denoise_scope: str = "all", raw_taps: list[AudioTap] | None = None, ab: bool = False):
        """Create after engine.start(): the stream rate is only known then.

        ``highpass`` and ``denoiser`` condition the input before both paths
        (see the module docstring); either may be None. ``raw_taps`` observe
        the unprocessed mic. ``ab`` adds the processed/raw switch
        (set_processing).
        """
        if denoise_scope not in self.SCOPES:
            raise ValueError(f"denoise_scope must be one of {self.SCOPES}")
        self.engine = engine
        self.cfg = cfg
        self.analyzers = list(analyzers or [])
        self.stages = list(stages or [])
        self.taps = list(taps or [])
        self.raw_taps = list(raw_taps or [])
        self.highpass = highpass
        self.denoiser = denoiser
        self.denoise_scope = denoise_scope
        self._raw_delay = (DelayLine(denoiser.delay_samples)
                           if denoiser is not None and denoise_scope == "models" else None)
        self.timer = StageTimer()
        self._feed = AnalysisFeed(engine.samplerate)
        # Where the playback path's delay comes from, in samples: output samples
        # were captured this much earlier (processing time and buffers aside).
        self._compute_delays()
        # A/B: the raw mic, delayed like the processed path, crossfaded in by the UI.
        self.processing = True      # what the UI asked for; the crossfade follows it
        self._ab_gain = SmoothGain(engine.samplerate, cfg.ramp_ms) if ab else None
        self._ab_delay = DelayLine(self.delay_samples) if ab else None
        self.paused = False         # what the UI asked for; the worker follows on the next block
        self._was_paused = False
        self._media_samples = 0     # samples that went into the recording (not while paused)
        self._new_stages: list[AudioStage] | None = None
        # Where full scale is checked. Only "output" is real clipping (the DAC
        # can't go past 1.0, so it gets clamped there); the rest say where it started.
        self.clips = {name: ClipMeter(name, limit) for name, limit in
                      (("mic", 0.999), ("denoise", 1.0), ("output", 1.0))}
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
        self.errors = 0             # blocks a stage failed on (played unprocessed)
        self.last_error: str | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _compute_delays(self) -> None:
        self.delays = {name: n for name, n in
                       [("denoise", self.denoiser.delay_samples if self.denoiser is not None else 0)]
                       + [(stage.name, getattr(stage, "delay_samples", 0)) for stage in self.stages] if n}
        self.delay_samples = sum(self.delays.values())

    @property
    def media_time(self) -> float:
        """Seconds recorded so far (pauses excluded)."""
        return self._media_samples / self.engine.samplerate

    def set_paused(self, paused: bool) -> None:
        """Stop (True) or resume recording, analysis and playback; the mic meter keeps running."""
        self.paused = bool(paused)

    def set_stages(self, stages: list[AudioStage]) -> None:
        """Replace the playback stages from the next block on (any thread)."""
        self._new_stages = list(stages)

    def _apply_new_stages(self) -> None:
        stages, self._new_stages = self._new_stages, None
        self.stages = stages
        self._compute_delays()
        if self._ab_delay is not None:
            self._ab_delay = DelayLine(self.delay_samples)

    def _enter_pause(self) -> None:
        for x in self.analyzers:
            pause = getattr(x, "pause", None)  # the segmenter closes its utterance
            if pause is not None:
                pause()

    @property
    def mic_silent(self) -> bool:
        """True if the mic has produced only exact zeros for a while.

        macOS delivers digital silence instead of an error when the terminal
        app lacks microphone permission; a real room is never exactly zero.
        """
        seconds = self.blocks * self.cfg.block_ms / 1000
        return not self.heard_signal and seconds > self.SILENT_MIC_WARN_S

    @property
    def ab(self) -> bool:
        """Whether the processed/raw switch exists (listen mode)."""
        return self._ab_gain is not None

    @property
    def denoise_skips(self) -> int:
        """Blocks played undenoised because the model missed its deadline; only increases."""
        return getattr(self.denoiser, "late_blocks", 0)

    # The rest of the chain after the denoiser, plus scheduling slack: the
    # denoiser gets whatever the output's queue leaves after this.
    DEADLINE_MARGIN_S = 0.006
    MIN_DEADLINE_S = 0.001

    def _denoise_deadline(self) -> float | None:
        """Seconds the worker can wait for the denoiser before the output runs dry; None: no limit."""
        headroom = getattr(self.engine, "output_headroom", None)
        seconds = headroom() if headroom is not None else None
        if seconds is None or not self.cfg.late_bypass:
            return None
        return max(self.MIN_DEADLINE_S, seconds - self.DEADLINE_MARGIN_S)

    def set_processing(self, on: bool) -> None:
        """A/B: True plays the processed chain, False the raw mic; the worker crossfades over ramp_ms."""
        self.processing = bool(on)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pipeline-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for x in self.analyzers + self.raw_taps + self.taps + [self.denoiser]:
            close = getattr(x, "close", None)  # those with threads or files of their own
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
            raw = block.copy()
            try:
                out = self.process_block(block, adc_time)
            except Exception as exc:  # noqa: BLE001 - a broken stage must not silence the audio
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self.errors == 1:
                    log.exception("processing failed; playing the audio unprocessed while it keeps failing")
                out = np.clip(raw, -1.0, 1.0)
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
        self.clips["mic"].check(block)
        if self._new_stages is not None:
            self._apply_new_stages()
        if self.paused:
            if not self._was_paused:
                self._was_paused = True
                self._enter_pause()
            self.blocks += 1
            self.out_db = -200.0
            return np.zeros_like(block)
        self._was_paused = False
        ctx = self._ctx = dataclasses.replace(self._ctx, index=self.blocks, adc_time=adc_time,
                                              media_time=self._media_samples / self._ctx.samplerate)
        self._media_samples += len(block)
        for tap in self.raw_taps:
            tap.observe(block, ctx)
        dry = block.copy() if self._ab_gain is not None else None  # the stages may work in place

        raw = block
        if self.highpass is not None:
            t = time.perf_counter()
            raw = self.highpass.process(raw, ctx)
            timer.add(self.highpass.name, time.perf_counter() - t)
        analysis = playback = raw
        if self.denoiser is not None:
            t = time.perf_counter()
            deadline = self._denoise_deadline()
            denoised = (self.denoiser.process(raw, ctx) if deadline is None
                        else self.denoiser.process(raw, ctx, deadline=deadline))
            timer.add(self.denoiser.name, time.perf_counter() - t)
            self.clips["denoise"].check(denoised)
            scope = self.denoise_scope
            analysis = raw if scope == "playback" else denoised
            if scope == "models":
                playback = self._raw_delay.process(raw)
            else:
                playback = denoised

        t = time.perf_counter()
        frames = self._feed.push(analysis)
        timer.add("resample", time.perf_counter() - t)
        for frame in frames:
            for analyzer in self.analyzers:
                t = time.perf_counter()
                analyzer.analyze(frame, ctx)
                timer.add(analyzer.name, time.perf_counter() - t)

        out = playback
        for stage in self.stages:
            t = time.perf_counter()
            out = stage.process(out, ctx)
            timer.add(stage.name, time.perf_counter() - t)
        if self._ab_gain is not None:
            dry = self._ab_delay.process(dry)  # always runs, so the raw side stays in step
            self._ab_gain.set_target(1.0 if self.processing else 0.0)
            if self._ab_gain.current != 1.0 or self._ab_gain.target != 1.0:
                wet = self._ab_gain.process(np.ones(len(out), np.float32))  # the crossfade curve
                out = dry + wet * (out - dry)
        if self.clips["output"].check(out):
            out = np.clip(out, -1.0, 1.0)  # what the DAC would do anyway; counted and reported
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
    delay_ms: float                 # every delay in the playback path (denoiser, lookaheads, limiter)
    out_buffer_ms: float
    out_device_ms: float
    stages: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (mean, max) ms
    delays: dict[str, float] = field(default_factory=dict)  # delay_ms by stage, ms
    denoise_skips: int = 0          # late blocks passed through undenoised this interval
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
    clips: dict[str, int] = field(default_factory=dict)  # samples at full scale this interval, per point


DELAY_LABELS = {"speaker-gain": "speaker lookahead", "gate": "gate lookahead"}


def latency_breakdown(snap: MetricsSnapshot) -> str:
    """'mic 33 + queue 0 + proc 8 + denoise 38 + limiter 2 + out-buf 40 + out-dev 33': where e2e goes.

    Approximate: the parts are measured separately and should roughly sum
    to the measured end-to-end figure.
    """
    delays = snap.delays or ({"delays": snap.delay_ms} if snap.delay_ms else {})
    parts = [("mic", snap.in_device_ms), ("queue", snap.in_queue_ms), ("proc", snap.proc_ms),
             *((DELAY_LABELS.get(name, name), ms) for name, ms in delays.items()),
             ("out-buf", snap.out_buffer_ms), ("out-dev", snap.out_device_ms)]
    return " + ".join(f"{label} {ms:.0f}" for label, ms in parts)


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
                "prob": p.vad_prob_sum, "dropped": p.dropped, "drops": p.drop_events,
                "skips": p.denoise_skips, **{f"clip_{name}": meter.count for name, meter in p.clips.items()}}

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
            delays={name: 1e3 * n / fs for name, n in self.pipeline.delays.items()},
            denoise_skips=int(d["skips"]),
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
            clips={k[5:]: int(v) for k, v in d.items() if k.startswith("clip_") and v},
        )
        self._prev_stats = cur
        self._prev = counters
        self._prev_time = now
        return snap
