"""Stage 3: who is speaking. ECAPA-TDNN speaker embeddings over speech windows.

EcapaEmbedder wraps SpeechBrain's speechbrain/spkrec-ecapa-voxceleb (192-dim
voiceprints). SpeakerAnalyzer is the pipeline FrameAnalyzer that decides
when to embed and hands the result to the SpeakerTracker; the embedding
itself runs on a thread of its own. torch is imported only when the model
loads, so the analyzer can be tested without it.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import warnings
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .config import ANALYSIS_FRAME, ANALYSIS_RATE, DATA_DIR

if TYPE_CHECKING:
    from .clustering import SpeakerTracker
    from .pipeline import BlockContext

log = logging.getLogger(__name__)

EMBEDDING_DIM = 192
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = DATA_DIR / "models" / "spkrec-ecapa-voxceleb"


class EcapaEmbedder:
    """16 kHz speech -> L2-normalised 192-dim voiceprint.

    ``device="auto"`` uses Apple's GPU (MPS) when available. On an M3, a 1.5 s
    window takes ~10 ms (MPS) or ~18 ms (CPU) back to back, but 32-40 ms on
    either when called every 250 ms as the pipeline does. If MPS fails the
    warm-up, it falls back to the CPU. The first run downloads the model
    (~80 MB) into ~/.focus-ear/models; after that it loads offline.
    """

    def __init__(self, device: str = "auto", model_dir: Path = MODEL_DIR):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._torch = torch
        # SpeechBrain's feature extractor trips a harmless torch.stft deprecation warning.
        warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        local = (model_dir / "hyperparams.yaml").exists()
        source = str(model_dir) if local else MODEL_SOURCE
        if not local:
            log.info("downloading %s to %s (first run only)", MODEL_SOURCE, model_dir)

        t0 = time.perf_counter()
        for dev in dict.fromkeys([device, "cpu"]):  # the requested device, then CPU as a fallback
            try:
                if dev == "cpu":
                    torch.set_num_threads(2)  # leave cores for the audio threads
                self.model = EncoderClassifier.from_hparams(
                    source=source, savedir=str(model_dir), run_opts={"device": dev},
                    local_strategy=LocalStrategy.COPY)
                self.model.eval()
                self.device = dev
                self(np.zeros(ANALYSIS_RATE, dtype=np.float32))  # warm-up; also proves the device works
                break
            except Exception as exc:  # noqa: BLE001 - any backend failure means "try the CPU"
                if dev == "cpu":
                    raise
                log.warning("speaker model failed on %s (%s); falling back to CPU", dev, exc)
        self.load_s = time.perf_counter() - t0
        log.info("speaker model: ECAPA-TDNN on %s, loaded in %.2f s", self.device.upper(), self.load_s)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(self.device)
            e = self.model.encode_batch(x).reshape(-1).cpu().numpy()
        return e / max(float(np.linalg.norm(e)), 1e-12)


class SpeakerAnalyzer:
    """Pipeline FrameAnalyzer: sets ``ctx.speaker_id`` to the confirmed speaker talking.

    Keeps the last ``window_s`` of 16 kHz frames with the VAD's decision for
    each. Every ``hop_s`` it takes the window's speech frames (silence frames
    are left out) and has them embedded and matched to a speaker. It skips a
    window that is mostly silence (less than ``min_speech`` speech). A window
    matching an unconfirmed cluster leaves ``speaker_id`` unchanged, so noise
    bursts and mixed windows don't flip the gain.

    After ``forget_after_s`` of silence ``speaker_id`` goes back to None
    (unknown, played at full volume) until the next voice is identified, and
    the speech before the pause is dropped from the window. Turns are usually
    separated by a pause, so when the selected lecturer resumes after a
    question they aren't held at the student's attenuation, nor matched to
    the student by a window straddling both, while the model catches up.

    Inference runs on its own thread (``background=True``), not the worker.
    ECAPA takes ~10 ms back to back, but 35-100 ms when called every 250 ms
    (Apple Silicon clocks down between bursts). That would stall the audio
    path past its jitter buffer and cause dropouts. The worker hands a window
    over without waiting. If the previous one is still being embedded, the
    new window is dropped and counted, so a slow model costs accuracy, never
    latency. The result lands on the next frame after it's ready.
    ``background=False`` embeds inline, for deterministic tests.
    """

    name = "speaker"

    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], tracker: SpeakerTracker,
                 window_s: float = 1.5, hop_s: float = 0.25, min_speech: float = 0.5,
                 forget_after_s: float = 0.5, background: bool = True):
        frame_s = ANALYSIS_FRAME / ANALYSIS_RATE
        self.embed = embed
        self.tracker = tracker
        self.frame_s = frame_s
        self.min_speech_frames = math.ceil(min_speech * round(window_s / frame_s))
        self._window: deque[tuple[np.ndarray, bool]] = deque(maxlen=round(window_s / frame_s))
        self._hop = max(1, round(hop_s / frame_s))
        self._since_hop = 0
        self._forget_after = max(1, round(forget_after_s / frame_s))
        self._silent_frames = 0
        # Read by metrics/UI; only ever increase.
        self.embedded = 0
        self.skipped_silent = 0
        self.dropped = 0
        self.last_ms = 0.0

        # Hand-off to the inference thread. Only the worker sets _busy, only
        # the inference thread clears it; results come back as (seq, speaker id).
        self._background = background
        self._job: tuple[np.ndarray, float] | None = None
        self._busy = False
        self._wake = threading.Event()
        self._result: tuple[int, int | None] = (0, None)
        self._applied = 0
        self._closing = False
        self._thread: threading.Thread | None = None

    def analyze(self, frame: np.ndarray, ctx: BlockContext) -> None:
        seq, speaker_id = self._result
        if seq != self._applied:
            self._applied = seq
            if speaker_id is not None:
                ctx.speaker_id = speaker_id

        self._silent_frames = 0 if ctx.is_speech else self._silent_frames + 1
        if self._silent_frames == self._forget_after:
            ctx.speaker_id = None
            self._window = deque(((f, False) for f, _ in self._window), maxlen=self._window.maxlen)

        self._window.append((frame, ctx.is_speech))
        rms = float(np.sqrt(np.mean(frame * frame)))
        self.tracker.update_level(20 * math.log10(max(rms, 1e-10)), ctx.is_speech, self.frame_s)

        self._since_hop += 1
        if self._since_hop < self._hop or len(self._window) < self._window.maxlen:
            return
        self._since_hop = 0
        speech = [f for f, is_speech in self._window if is_speech]
        if len(speech) < self.min_speech_frames:
            self.skipped_silent += 1
            return
        audio = np.concatenate(speech)
        if not self._background:
            speaker_id = self._identify(audio, ctx.adc_time)
            if speaker_id is not None:
                ctx.speaker_id = speaker_id
            return
        if self._busy:
            self.dropped += 1  # the model is still on the previous window
            return
        self._busy = True
        self._job = (audio, ctx.adc_time)
        if self._thread is None:
            self._thread = threading.Thread(target=self._serve, name="speaker-embed", daemon=True)
            self._thread.start()
        self._wake.set()

    def _identify(self, audio: np.ndarray, when: float) -> int | None:
        t = time.perf_counter()
        embedding = self.embed(audio)
        self.last_ms = 1e3 * (time.perf_counter() - t)
        self.embedded += 1
        return self.tracker.observe(embedding, when)

    def _serve(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._closing:
                return
            audio, when = self._job
            try:
                speaker_id = self._identify(audio, when)
            except Exception:  # noqa: BLE001 - keep the thread alive; the next window may work
                log.exception("speaker embedding failed")
                speaker_id = None
            self._result = (self._result[0] + 1, speaker_id)
            self._busy = False

    def close(self) -> None:
        """Stop the inference thread (waits for the window in progress)."""
        if self._thread is not None:
            self._closing = True
            self._wake.set()
            self._thread.join(timeout=2)
            self._thread = None

    def reset(self) -> None:
        self._window.clear()
        self._since_hop = 0
