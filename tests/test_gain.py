"""Run with: venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from focus_ear.gain import DelayLine, NoiseGate, SmoothGain, db_to_gain, gain_to_db
from focus_ear.pipeline import BlockContext

FS = 44100


def gain_curve(g: SmoothGain, n: int, block: int = 441) -> np.ndarray:
    """The per-sample gain SmoothGain applies over n samples, measured on a unit signal."""
    return np.concatenate([g.process(np.ones(block, np.float32)) for _ in range(n // block)])


class SmoothGainTest(unittest.TestCase):
    def test_db_helpers(self):
        self.assertAlmostEqual(db_to_gain(-20), 0.1)
        self.assertAlmostEqual(gain_to_db(0.5), -6.0206, places=3)

    def test_linear_ramp_reaches_target_in_ramp_time(self):
        g = SmoothGain(FS, ramp_ms=50.0, gain=1.0)
        g.set_target(0.0)
        curve = gain_curve(g, 4410)
        n = g.ramp_samples  # 2205 samples = 50 ms
        np.testing.assert_allclose(np.diff(curve[:n]), -1.0 / n, atol=1e-6)
        self.assertEqual(curve[n - 1], 0.0)
        self.assertTrue(np.all(curve[n:] == 0.0))

    def test_retarget_mid_ramp_is_continuous(self):
        g = SmoothGain(FS, ramp_ms=50.0, gain=1.0)
        g.set_target(0.1)
        a = gain_curve(g, 882)  # 20 ms into the ramp
        g.set_target(1.0)
        b = gain_curve(g, 3528)
        curve = np.concatenate([[1.0], a, b])
        # No step bigger than the steepest ramp can take: that's what makes it click-free.
        self.assertLessEqual(np.abs(np.diff(curve)).max(), 0.9 / g.ramp_samples + 1e-6)
        self.assertEqual(g.current, 1.0)

    def test_unity_is_bit_exact_and_in_place(self):
        g = SmoothGain(FS)
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        y = x.copy()
        self.assertIs(g.process(y), y)
        np.testing.assert_array_equal(y, x)


class DelayLineTest(unittest.TestCase):
    def test_delays_by_exact_samples_across_blocks(self):
        d = DelayLine(100)
        x = np.arange(1000, dtype=np.float32)
        y = np.concatenate([d.process(x[i:i + 300].copy()) for i in range(0, 900, 300)])
        np.testing.assert_array_equal(y[:100], 0)
        np.testing.assert_array_equal(y[100:], x[:800])


class NoiseGateTest(unittest.TestCase):
    BLOCK = 1411  # 32 ms at 44.1 kHz

    def run_gate(self, gate, speech_flags):
        """Feed unit blocks with the given VAD decisions; returns the per-sample gain."""
        out = []
        for i, speaking in enumerate(speech_flags):
            ctx = BlockContext(index=i, adc_time=0.0, samplerate=FS, is_speech=speaking)
            out.append(gate.process(np.ones(self.BLOCK, np.float32), ctx).copy())
        return np.concatenate(out)

    def test_silence_is_attenuated(self):
        g = self.run_gate(NoiseGate(FS, attenuation_db=-18), [False] * 10)
        np.testing.assert_allclose(g, db_to_gain(-18), rtol=1e-6)

    def test_opens_on_speech_with_a_50ms_ramp(self):
        gate = NoiseGate(FS, attenuation_db=-18, ramp_ms=50)
        g = self.run_gate(gate, [False] * 3 + [True] * 5)
        onset = 3 * self.BLOCK
        self.assertAlmostEqual(g[onset - 1], db_to_gain(-18), places=6)
        self.assertEqual(g[onset + gate.gain.ramp_samples - 1], 1.0)
        self.assertLess(np.abs(np.diff(g)).max(), 1.0 / gate.gain.ramp_samples + 1e-6)

    def test_hangover_holds_the_gate_open_then_closes_smoothly(self):
        gate = NoiseGate(FS, attenuation_db=-18, hangover_ms=200, ramp_ms=50)
        g = self.run_gate(gate, [True] * 5 + [False] * 15)
        speech_end = 5 * self.BLOCK
        still_open = speech_end + int(0.19 * FS)
        self.assertTrue(np.all(g[speech_end:still_open] == 1.0), "clipped the ending")
        closed = speech_end + int((0.2 + 0.032 + 0.05) * FS)  # hangover + a block + ramp
        self.assertAlmostEqual(g[closed], db_to_gain(-18), places=6)
        self.assertLess(np.abs(np.diff(g)).max(), 1.0 / gate.gain.ramp_samples + 1e-6)

    def test_lookahead_opens_before_the_audio_arrives(self):
        gate = NoiseGate(FS, attenuation_db=-18, ramp_ms=50, lookahead_ms=64)
        blocks = [np.zeros(self.BLOCK, np.float32)] * 4 + [np.ones(self.BLOCK, np.float32)] * 4
        out = np.concatenate([
            gate.process(b.copy(), BlockContext(index=i, adc_time=0.0, samplerate=FS, is_speech=i >= 4))
            for i, b in enumerate(blocks)])
        first_sound = 4 * self.BLOCK + gate.delay._buf.size
        self.assertTrue(np.all(out[:first_sound] == 0.0))
        # By the time the (delayed) onset plays, the gate is already most of the way open.
        self.assertGreater(out[first_sound], 0.9)


if __name__ == "__main__":
    unittest.main()
