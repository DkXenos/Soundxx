"""Step 3: the final transcript, whisper-large-v3-turbo on MLX over the whole recording.

  * Chunks follow the speaker turns (so a chunk is one voice and gets one
    language), cut to at most ``max_chunk_s`` at the longest pause, or at
    the quietest moment if a turn has no pause at all.
  * Temperature 0 only: greedy decoding, the same text on every run. The
    live model's sampled fallback made identical runs differ (STATUS.md).
  * Word timestamps, so every word can take the speaker of the moment it
    was said (step 4) and a click on a line plays from its first word.
  * Language per chunk: Indonesian unless Whisper's language
    identification is confident it's English (``en_threshold``). Forced
    Indonesian on an English question turns it into nonsense or a
    translation.
  * A segment is flagged when Whisper's own failure tests trip: compression
    ratio above 2.4 (a repetition loop) or mean log-probability below -1.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..config import ANALYSIS_RATE, DEFAULT_PROMPT

log = logging.getLogger("focus_ear.post")

MODEL = "mlx-community/whisper-large-v3-turbo"
GARBLED_RATIO = 2.4
LOW_LOGPROB = -1.0
PAD_S = 0.15


def chunks(turns: list[dict], regions: list[tuple[float, float]], probs_100ms: list[float] | None,
           max_chunk_s: float = 25.0) -> list[dict]:
    """[{start, end, speaker}] covering every turn, each at most max_chunk_s long."""
    regions = [tuple(r) for r in regions]
    out = []
    for t in turns:
        out += _split(t["start"], t["end"], t["speaker"], regions, probs_100ms, max_chunk_s)
    return out


def _split(a: float, b: float, speaker: str, regions, probs, max_s: float) -> list[dict]:
    if b - a <= max_s:
        return [{"start": a, "end": b, "speaker": speaker}]
    # Gaps between speech regions inside (a, b): the best place to cut, latest first within the limit.
    gaps = [(r0[1], r1[0]) for r0, r1 in zip(regions, regions[1:]) if a < r0[1] and r1[0] < b]
    limit = a + max_s
    candidates = [(g0 + g1) / 2 for g0, g1 in gaps if a + 5.0 <= (g0 + g1) / 2 <= limit]
    if candidates:
        # The longest pause in the last 10 s before the limit, else the latest pause.
        late = [(g1 - g0, (g0 + g1) / 2) for g0, g1 in gaps if limit - 10.0 <= (g0 + g1) / 2 <= limit]
        cut = max(late)[1] if late else max(candidates)
    elif probs:
        lo, hi = int((limit - 5.0) * 10), int(limit * 10)
        seg = probs[max(lo, 0):max(hi, lo + 1)]
        cut = (lo + int(np.argmin(seg))) / 10 if len(seg) else limit
    else:
        cut = limit
    return [{"start": a, "end": cut, "speaker": speaker}] + _split(cut, b, speaker, regions, probs, max_s)


class FinalAsr:
    def __init__(self, model: str = MODEL):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        from ..asr import _local_or_download

        self._mx = mx
        self._transcribe = mlx_whisper.transcribe
        mx.set_cache_limit(512 * 2**20)
        self.name = model
        self.path = _local_or_download(model)
        self.model = ModelHolder.get_model(self.path, mx.float16)

    def language(self, audio: np.ndarray) -> dict[str, float]:
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        mel = log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self._mx.float16)
        _, probs = self.model.detect_language(mel)
        return probs

    def transcribe(self, audio: np.ndarray, language: str, prompt: str | None) -> dict:
        return self._transcribe(audio, path_or_hf_repo=self.path, language=language, initial_prompt=prompt,
                                temperature=0.0, condition_on_previous_text=False, word_timestamps=True,
                                verbose=None)


def run(folder, audio: np.ndarray, turns: list[dict], regions: list, probs_100ms, progress,
        model: str = MODEL, language: str = "id", en_threshold: float = 0.6, prompt: str = DEFAULT_PROMPT,
        max_chunk_s: float = 25.0) -> dict:
    progress(0.0, "loading the large speech model…", force=True)
    t0 = time.perf_counter()
    asr = FinalAsr(model)
    load_s = time.perf_counter() - t0
    parts = chunks(turns, regions, probs_100ms, max_chunk_s)
    total_audio = sum(c["end"] - c["start"] for c in parts) or 1.0
    done_audio = 0.0
    out = []
    t1 = time.perf_counter()
    for i, c in enumerate(parts):
        a = max(0.0, c["start"] - PAD_S)
        b = min(len(audio) / ANALYSIS_RATE, c["end"] + PAD_S)
        x = audio[int(a * ANALYSIS_RATE):int(b * ANALYSIS_RATE)]
        if len(x) < int(0.3 * ANALYSIS_RATE):
            continue
        lang, probs = language, None
        if language == "id" and en_threshold < 1.0:
            p = asr.language(x)
            probs = {k: round(float(v), 3) for k, v in sorted(p.items(), key=lambda kv: -kv[1])[:3]}
            if p.get("en", 0.0) >= en_threshold:
                lang = "en"
        r = asr.transcribe(x, lang, prompt if lang == "id" else None)
        segs = []
        for s in r["segments"]:
            words = [[round(a + w["start"], 3), round(a + w["end"], 3), w["word"], round(float(w["probability"]), 3)]
                     for w in s.get("words", [])]
            if not words and not s["text"].strip():
                continue
            segs.append({"start": round(a + s["start"], 3), "end": round(a + s["end"], 3), "text": s["text"].strip(),
                         "words": words, "avg_logprob": round(float(s["avg_logprob"]), 3),
                         "no_speech_prob": round(float(s["no_speech_prob"]), 3),
                         "compression_ratio": round(float(s["compression_ratio"]), 3),
                         "flagged": bool(s["compression_ratio"] > GARBLED_RATIO or s["avg_logprob"] < LOW_LOGPROB)})
        out.append({"start": c["start"], "end": c["end"], "speaker": c["speaker"], "language": lang,
                    "lang_probs": probs, "segments": segs})
        done_audio += c["end"] - c["start"]
        elapsed = time.perf_counter() - t1
        eta = elapsed / done_audio * (total_audio - done_audio) if done_audio else 0
        progress(done_audio / total_audio, f"writing the transcript: {done_audio / 60:.0f} of "
                 f"{total_audio / 60:.0f} min of speech, about {eta / 60:.0f} min left")
    return {"model": model, "chunks": out, "language": language, "en_threshold": en_threshold,
            "temperature": 0.0, "prompt": prompt, "load_s": round(load_s, 1),
            "decode_s": round(time.perf_counter() - t1, 1), "speech_s": round(total_audio, 1)}
