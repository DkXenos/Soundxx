"""Runtime configuration shared by every module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000  # silero-vad and ECAPA-TDNN both expect 16 kHz mono
BLOCK_SIZE = 512      # 32 ms; also silero-vad's native frame size at 16 kHz
DATA_DIR = Path.home() / ".focus-ear"


@dataclass
class Config:
    # Devices: name substring or PortAudio index. None input = built-in mic.
    input_device: str | None = None
    output_device: str = "Galaxy Buds"
    allow_speakers: bool = False

    # Stream shape
    samplerate: int = SAMPLE_RATE
    blocksize: int = BLOCK_SIZE
    latency: str | float = "low"   # PortAudio suggested latency: "low", "high" or seconds

    # Buffering. buffer_ms is the output jitter cushion (adds latency, absorbs
    # worker hiccups); max_backlog_ms is how far the worker may fall behind the
    # mic before it skips ahead to stay real-time.
    buffer_ms: float = 64.0
    max_backlog_ms: float = 300.0

    # Gating (phase 2+)
    passthrough: bool = False
    attenuation_db: float = -20.0
    boost_db: float = 0.0
    threshold: float = 0.65

    debug: bool = False
