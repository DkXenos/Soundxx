"""Run with: venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from fixtures import SILENCE_SPAN, frames, load_speech, tone
from focus_ear.pipeline import BlockContext
from focus_ear.vad import SileroVAD, SpeechDetector, VadAnalyzer

FPS = 16000 / 512  # frames per second


def probabilities(vad, x):
    return np.array([vad(f) for f in frames(x)])


class SileroVADTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vad = SileroVAD()

    def setUp(self):
        self.vad.reset()

    def test_silence_is_not_speech(self):
        self.assertLess(probabilities(self.vad, np.zeros(32000, np.float32)).max(), 0.1)

    def test_steady_tones_are_not_speech(self):
        for freq in (440, 1000):
            self.vad.reset()
            self.assertLess(probabilities(self.vad, tone(freq, 2.0)).max(), 0.1, f"{freq} Hz")

    def test_recorded_speech_is_detected_and_the_pause_is_not(self):
        p = probabilities(self.vad, load_speech())
        speech = p[int(0.25 * FPS):int(1.5 * FPS)]
        pause = p[int((SILENCE_SPAN[0] + 0.25) * FPS):int((SILENCE_SPAN[1] - 0.25) * FPS)]
        self.assertGreater(np.mean(speech > 0.5), 0.8)
        self.assertLess(np.mean(pause > 0.5), 0.1)

    def test_reset_makes_results_reproducible(self):
        x = load_speech()[:16000]
        first = probabilities(self.vad, x)
        self.vad.reset()
        np.testing.assert_allclose(probabilities(self.vad, x), first, atol=1e-6)

    def test_rejects_wrong_frame_size(self):
        with self.assertRaises(ValueError):
            self.vad(np.zeros(480, np.float32))


class SpeechDetectorTest(unittest.TestCase):
    def test_hysteresis(self):
        d = SpeechDetector(threshold=0.5)  # leaves speech below 0.35
        states = [d.update(p) for p in (0.4, 0.6, 0.45, 0.36, 0.34, 0.45)]
        self.assertEqual(states, [False, True, True, True, False, False])


class VadAnalyzerTest(unittest.TestCase):
    def test_annotates_context(self):
        a = VadAnalyzer()
        ctx = BlockContext(index=0, adc_time=0.0, samplerate=16000)
        for f in frames(load_speech()[:16000]):
            a.analyze(f, ctx)
        self.assertTrue(ctx.is_speech)
        self.assertGreater(ctx.vad_prob, 0.5)


if __name__ == "__main__":
    unittest.main()
