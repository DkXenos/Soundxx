"""Stage 1: voice activity detection with silero-vad, run through ONNX Runtime.

The silero-vad pip package hard-depends on torch even for ONNX use, so we
ship only its ONNX model (models/silero_vad.onnx, v6.2.3, MIT) and feed it
with numpy the same way silero's own OnnxWrapper does.

Everything here runs on the pipeline worker thread, never in an audio callback.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from .config import ANALYSIS_FRAME, ANALYSIS_RATE

if TYPE_CHECKING:
    from .pipeline import BlockContext

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"
_CONTEXT = 64  # samples of the previous frame the model expects prepended (16 kHz)


class SileroVAD:
    """Streaming speech probability, one 512-sample 16 kHz frame per call.

    The model is recurrent: call it on consecutive frames of one stream and
    reset() between unrelated streams.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        opts = ort.SessionOptions()
        # One frame is ~0.1 ms of work; extra threads only add scheduling jitter.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(ANALYSIS_RATE, dtype=np.int64)
        self._input = np.zeros((1, _CONTEXT + ANALYSIS_FRAME), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._input[:] = 0.0

    def __call__(self, frame: np.ndarray) -> float:
        """Probability in [0, 1] that ``frame`` (512 samples, 16 kHz, float32) is speech."""
        if len(frame) != ANALYSIS_FRAME:
            raise ValueError(f"silero-vad needs {ANALYSIS_FRAME}-sample frames, got {len(frame)}")
        x = self._input
        x[0, :_CONTEXT] = x[0, -_CONTEXT:]  # tail of the previous frame
        x[0, _CONTEXT:] = frame
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        return float(out[0, 0])


class SpeechDetector:
    """Turns per-frame probabilities into speech/silence with hysteresis.

    Enters speech at ``threshold`` and leaves it only below
    ``threshold - 0.15``, as silero's own segmenter does, so a probability
    hovering around the threshold doesn't flicker.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.speaking = False

    def update(self, prob: float) -> bool:
        if self.speaking:
            self.speaking = prob >= self.neg_threshold
        else:
            self.speaking = prob >= self.threshold
        return self.speaking

    def reset(self) -> None:
        self.speaking = False


class VadAnalyzer:
    """Pipeline FrameAnalyzer: sets ``ctx.vad_prob`` and ``ctx.is_speech``."""

    name = "vad"

    def __init__(self, threshold: float = 0.5, model_path: Path = MODEL_PATH):
        self.model = SileroVAD(model_path)
        self.detector = SpeechDetector(threshold)

    def analyze(self, frame: np.ndarray, ctx: BlockContext) -> None:
        ctx.vad_prob = self.model(frame)
        ctx.is_speech = self.detector.update(ctx.vad_prob)

    def reset(self) -> None:
        self.model.reset()
        self.detector.reset()
