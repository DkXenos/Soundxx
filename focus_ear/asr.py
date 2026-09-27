"""Stage 5: speech recognition backends. Whisper on MLX (Apple GPU), else faster-whisper on the CPU.

faster-whisper is built on CTranslate2, which has no Metal support, so on a
Mac it runs CPU-only; it is only the fallback for when mlx-whisper can't
load. On an M3, whisper-small on MLX transcribes a 10 s utterance in ~0.4 s.

A backend must be created and used on one thread: MLX binds its arrays to
the thread that made them ("There is no Stream(gpu, 1) in current thread"
otherwise). The Transcriber loads it on its own thread for that reason.
Models load from the local Hugging Face cache first, so a lecture hall
without wifi still works once a model has been downloaded.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ANALYSIS_RATE

log = logging.getLogger(__name__)


@dataclass
class AsrResult:
    text: str
    language: str                       # the language decoded as (given, or detected for "auto")
    avg_logprob: float | None           # token-weighted over Whisper's segments
    no_speech_prob: float | None
    compression_ratio: float | None
    lang_probs: dict[str, float] | None = None  # top 3, when language identification ran


class AsrBackend(Protocol):
    description: str

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None,
                   detect: bool = False) -> AsrResult:
        """``language`` None detects it. ``detect`` also reports lang_probs for a given language."""
        ...


def _top(probs: dict[str, float], n: int = 3) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:n]}


def _result(text: str, language: str, segments: list[tuple[int, float, float, float]],
            lang_probs: dict[str, float] | None) -> AsrResult:
    """``segments``: (tokens, avg_logprob, no_speech_prob, compression_ratio) per Whisper segment."""
    tokens = sum(n for n, *_ in segments)
    avg = sum(n * lp for n, lp, _, _ in segments) / tokens if tokens else None
    return AsrResult(
        text=text.strip(), language=language, avg_logprob=avg,
        no_speech_prob=max((ns for _, _, ns, _ in segments), default=None),
        compression_ratio=max((cr for *_, cr in segments), default=None),
        lang_probs=_top(lang_probs) if lang_probs else None)


def _local_or_download(repo: str) -> str:
    path = Path(repo).expanduser()
    if path.exists():
        return str(path)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        log.info("downloading %s (first run only)…", repo)
        return snapshot_download(repo)


class MlxWhisper:
    CACHE_LIMIT = 256 * 2**20

    def __init__(self, model: str):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        self._mx = mx
        self._transcribe = mlx_whisper.transcribe
        # MLX keeps freed GPU buffers for reuse, without limit: utterances of
        # varying length grew it from 0.75 to 1.8 GB over 60 of them, about
        # 1 GB more per 90-minute lecture. Capped, Whisper was no slower.
        mx.set_cache_limit(self.CACHE_LIMIT)
        self.path = _local_or_download(model)
        # transcribe() loads through the same cache, in float16 (its fp16 default).
        self.model = ModelHolder.get_model(self.path, mx.float16)
        self.description = f"{model} (MLX, GPU)"
        # The first call compiles the GPU kernels (~1.7 s for small): pay it now, not on the first utterance.
        self.transcribe(np.zeros(ANALYSIS_RATE, np.float32), "en", None)

    def _language_probs(self, audio: np.ndarray) -> dict[str, float]:
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        # Exactly as transcribe() does it for language=None: the first 30 s, silence-padded.
        mel = log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self._mx.float16)
        _, probs = self.model.detect_language(mel)
        return probs

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None,
                   detect: bool = False) -> AsrResult:
        probs = None
        if language is None or detect:
            probs = self._language_probs(audio)  # ~0.15 s for small; transcribe() would do it anyway for None
            language = language or max(probs, key=probs.get)
        # Utterances are under 30 s, so conditioning on previous windows never applies;
        # False also stops a hallucination in one window from seeding the next.
        r = self._transcribe(audio, path_or_hf_repo=self.path, language=language, initial_prompt=prompt,
                             condition_on_previous_text=False, verbose=None)
        segments = [(len(s["tokens"]), s["avg_logprob"], s["no_speech_prob"], s["compression_ratio"])
                    for s in r["segments"]]
        return _result(r["text"], language, segments, probs)


def faster_whisper_size(model: str) -> str:
    """'mlx-community/whisper-small-mlx' -> 'small', 'whisper-large-v3-turbo' -> 'large-v3-turbo'."""
    name = model.rstrip("/").rsplit("/", 1)[-1].lower()
    name = re.sub(r"^whisper-", "", name)
    return re.sub(r"-(mlx|fp16|fp32|q4|q8|\d+bit)\b.*$", "", name)


class FasterWhisper:
    def __init__(self, size: str, cpu_threads: int = 4):
        from faster_whisper import WhisperModel

        kwargs = dict(device="cpu", compute_type="int8", cpu_threads=cpu_threads)
        try:
            self.model = WhisperModel(size, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - not cached (the error type varies with the hub version)
            log.info("downloading faster-whisper %s (first run only)…", size)
            self.model = WhisperModel(size, **kwargs)
        self.description = f"faster-whisper {size} (CTranslate2, CPU int8)"

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None,
                   detect: bool = False) -> AsrResult:
        probs = None
        if language is not None and detect:
            _, _, all_probs = self.model.detect_language(audio)
            probs = dict(all_probs)
        # Greedy, like mlx-whisper (which has no beam search), and fast enough for a CPU.
        segments, info = self.model.transcribe(audio, language=language, initial_prompt=prompt, beam_size=1,
                                               condition_on_previous_text=False, vad_filter=False)
        segments = list(segments)  # the generator does the decoding
        if info.all_language_probs:
            probs = dict(info.all_language_probs)
        return _result("".join(s.text for s in segments), info.language,
                       [(len(s.tokens), s.avg_logprob, s.no_speech_prob, s.compression_ratio) for s in segments],
                       probs)


def load_backend(model: str) -> AsrBackend:
    """mlx-whisper, or faster-whisper on the CPU if that fails. Raises if neither loads."""
    try:
        return MlxWhisper(model)
    except Exception as exc:  # noqa: BLE001 - any MLX failure means "try the CPU"
        log.warning("mlx-whisper couldn't load %s (%s: %s)", model, type(exc).__name__, exc)
    size = faster_whisper_size(model)
    log.warning("DOWNGRADE: transcribing with faster-whisper %s on the CPU (int8) instead of the GPU. "
                "Expect it to be several times slower and to compete with the audio for CPU time.", size)
    return FasterWhisper(size)
