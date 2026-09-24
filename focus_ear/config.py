"""Runtime configuration shared by every module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ANALYSIS_RATE = 16_000  # silero-vad and ECAPA-TDNN both expect 16 kHz mono
ANALYSIS_FRAME = 512    # 32 ms at 16 kHz: silero-vad's frame size
DATA_DIR = Path.home() / ".focus-ear"


@dataclass
class Config:
    # Devices: name substring or PortAudio index. None input = built-in mic.
    input_device: str | None = None
    output_device: str = "Mewo"
    allow_speakers: bool = False

    # Streams run at the mic's native rate (None) so playback keeps its full
    # bandwidth; the models get a resampled 16 kHz copy.
    samplerate: int | None = None
    block_ms: float = 32.0         # one worker block ~ one VAD frame
    latency: str | float = "low"   # PortAudio suggested latency: "low", "high" or seconds

    # Buffering. buffer_ms is the output jitter cushion (adds latency, absorbs
    # worker hiccups). If the worker falls more than max_backlog_ms behind the
    # mic, it drops audio to catch up instead of letting latency grow.
    buffer_ms: float = 64.0
    max_backlog_ms: float = 100.0

    # Stage 1: voice activity detection
    vad_threshold: float = 0.5

    # Stage 2: noise gate
    gate: bool = True
    gate_attenuation_db: float = -18.0
    gate_hangover_ms: float = 200.0
    gate_lookahead_ms: float = 0.0
    ramp_ms: float = 50.0

    # Stage 3: who is speaking. Defaults come from a simulated reverberant
    # lecture (see README): 1.0 s windows with a 0.65 threshold split a
    # lecturer into many "speakers".
    speakers: bool = True
    device: str = "auto"            # "auto" (MPS if available), "mps" or "cpu"
    embed_window_s: float = 1.5
    embed_hop_s: float = 0.25
    cluster_threshold: float = 0.45
    merge_threshold: float = 0.55
    cluster_alpha: float = 0.1
    max_speakers: int = 8
    speaker_timeout_s: float = 120.0
    min_sightings: int = 3

    # Stage 4: per-speaker gain
    passthrough: bool = False       # no audio processing at all (analysis still runs)
    boost_db: float = 0.0
    attenuation_db: float = -20.0
    lookahead_ms: float = 250.0

    tui: bool = True
    debug: bool = False
