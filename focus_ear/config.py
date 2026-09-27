"""Runtime configuration shared by every module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ANALYSIS_RATE = 16_000  # silero-vad and ECAPA-TDNN both expect 16 kHz mono
ANALYSIS_FRAME = 512    # 32 ms at 16 kHz: silero-vad's frame size
DATA_DIR = Path.home() / ".focus-ear"
SESSIONS_DIR = DATA_DIR / "sessions"

# Whisper continues the style of its prompt: spoken, code-mixed Indonesian
# with English technical terms left in English. Measured on synthesised
# code-mixed lecture speech (README), this sentence raised the average
# logprob on every clip. A keyword-list prompt was parroted back verbatim
# on noise; this one returned nothing there, where no prompt at all
# hallucinated "Terima kasih."
DEFAULT_PROMPT = ("Oke, jadi di kuliah ini kita pakai Python. Kita define function dulu, terus kita pakai "
                  "for loop, lalu return value-nya.")


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

    # Input conditioning, before everything else
    highpass_hz: float = 80.0       # 0 = off
    denoise: bool = True            # DeepFilterNet3
    denoise_mix: float = 1.0        # 1 = fully denoised, 0 = original
    # Who hears the denoised audio: "all" | "playback" | "models". "playback"
    # because on synthesised code-mixed lecture speech in noise (README),
    # denoising what the models hear wrecked Whisper (avg logprob -0.90 ->
    # -4.05) and merged the student into the lecturer (5 clusters -> 1).
    denoise_scope: str = "playback"
    latency_budget_ms: float = 500.0

    # Output conditioning, last before the Buds
    agc: bool = True
    agc_target_db: float = -23.0    # speech level (RMS, dBFS) the AGC aims for
    limiter: bool = True
    limiter_ceiling_db: float = -6.0  # no sample goes above this

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

    # Stage 5: transcription of the selected speaker (everyone while none is selected)
    transcribe: bool = True
    asr_model: str = "mlx-community/whisper-small-mlx"
    language: str = "id"            # a Whisper language code, or "auto"
    initial_prompt: str = DEFAULT_PROMPT
    utterance_gap_ms: float = 800.0
    max_utterance_s: float = 20.0
    preroll_ms: float = 300.0
    min_utterance_ms: float = 400.0
    asr_queue: int = 4              # utterances waiting for Whisper before the oldest is dropped
    save: bool = True               # write ~/.focus-ear/sessions/*.md and *.jsonl

    tui: bool = True
    debug: bool = False
