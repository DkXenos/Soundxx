"""Run with: venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from fixtures import tone
from focus_ear.dsp import Agc, ClipMeter, HighPass, Limiter
from focus_ear.gain import db_to_gain
from focus_ear.pipeline import BlockContext

FS, BLOCK = 44100, 1411


def blocks(x, n=BLOCK):
    for i in range(0, len(x) - n + 1, n):
        yield x[i:i + n].copy()


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def ctx(speech=True, speaker=None):
    return BlockContext(0, 0.0, FS, is_speech=speech, speaker_id=speaker)


class HighPassTest(unittest.TestCase):
    def run_hp(self, freq):
        hp = HighPass(FS, 80.0)
        x = tone(freq, 3.0, FS)
        y = np.concatenate([hp.process(b) for b in blocks(x)])
        return db(y[FS:]) - db(x[FS:len(y)])  # after the filter settles

    def test_removes_mains_hum_and_keeps_speech(self):
        self.assertLess(self.run_hp(50), -15)
        self.assertLess(self.run_hp(60), -9)
        self.assertGreater(self.run_hp(200), -0.5)
        self.assertAlmostEqual(self.run_hp(1000), 0.0, delta=0.05)

    def test_streaming_matches_one_pass(self):
        x = np.random.default_rng(0).standard_normal(FS).astype(np.float32)
        streamed = np.concatenate([HighPass(FS).process(x)])
        hp = HighPass(FS)
        chunked = np.concatenate([hp.process(x[i:i + 1000]) for i in range(0, len(x), 1000)])
        np.testing.assert_allclose(chunked, streamed, atol=1e-5)


class LimiterTest(unittest.TestCase):
    def test_nothing_ever_exceeds_the_ceiling(self):
        lim = Limiter(FS, ceiling_db=-6.0)
        rng = np.random.default_rng(1)
        quiet = 0.05 * rng.standard_normal(FS)
        cough = 1.5 * rng.standard_normal(FS // 4) * np.hanning(FS // 4)  # +20 dB burst, peaks past 0 dBFS
        x = np.concatenate((quiet, cough, quiet, 3.0 * tone(100, 0.5, FS), quiet)).astype(np.float32)
        y = np.concatenate([lim.process(b) for b in blocks(x)])
        self.assertLessEqual(np.abs(y).max(), db_to_gain(-6.0) + 1e-6)
        self.assertGreater(lim.limited, 0)

    def test_below_the_ceiling_it_is_bit_exact_just_delayed(self):
        lim = Limiter(FS, ceiling_db=-6.0)
        x = (0.3 * np.random.default_rng(2).standard_normal(3 * BLOCK)).clip(-0.49, 0.49).astype(np.float32)
        y = np.concatenate([lim.process(b) for b in blocks(x)])
        np.testing.assert_array_equal(y[lim.L:], x[:len(y) - lim.L])
        self.assertEqual(lim.limited, 0)

    def test_gain_recovers_fully_after_the_peak(self):
        lim = Limiter(FS, ceiling_db=-6.0, release_ms=80)
        rng = np.random.default_rng(6)
        burst = rng.standard_normal(BLOCK).astype(np.float32)
        quiet = (0.1 * rng.standard_normal(40 * BLOCK)).astype(np.float32)
        out = [lim.process(b) for b in blocks(np.concatenate((burst, quiet)))]
        self.assertEqual(lim.gain_db, 0.0)
        limited = lim.limited
        tail = np.concatenate([lim.process(b) for b in blocks(quiet[:5 * BLOCK])])
        self.assertEqual(lim.limited, limited)  # back to exactly unity: nothing more counted...
        np.testing.assert_array_equal(tail[lim.L:], quiet[:5 * BLOCK - lim.L])  # ...and bit-exact again
        self.assertTrue(out)


class AgcTest(unittest.TestCase):
    def test_brings_quiet_speech_to_the_target_slowly(self):
        agc = Agc(FS, BLOCK, target_db=-23.0)
        x = (0.01 * np.random.default_rng(3).standard_normal(20 * FS)).astype(np.float32)  # -40 dBFS
        out = [agc.process(b, ctx()) for b in blocks(x)]
        first_second = np.concatenate(out[:int(FS / BLOCK)])
        self.assertLess(db(first_second), -34)  # no jump: at most 6 dB/s up
        self.assertAlmostEqual(db(np.concatenate(out[-int(FS / BLOCK):])), -23.0, delta=1.0)

    def test_holds_during_silence_and_for_attenuated_speakers(self):
        agc = Agc(FS, BLOCK, target_db=-23.0, adapt=lambda c: c.speaker_id != 2)
        loud = (0.3 * np.random.default_rng(4).standard_normal(FS)).astype(np.float32)
        for b in blocks(loud):
            agc.process(b, ctx(speech=False))       # room noise between sentences
            agc.process(b, ctx(speaker=2))          # a student at -20 dB
        self.assertEqual(agc.gain_db, 0.0)

    def test_gain_is_capped(self):
        agc = Agc(FS, BLOCK, target_db=-23.0, max_gain_db=24.0)
        x = (1e-4 * np.random.default_rng(5).standard_normal(30 * FS)).astype(np.float32)  # -80 dBFS
        for b in blocks(x):
            agc.process(b, ctx())
        self.assertAlmostEqual(agc.gain_db, 24.0, delta=0.01)


class ClipMeterTest(unittest.TestCase):
    def test_counts_full_scale_samples(self):
        m = ClipMeter("output")
        self.assertEqual(m.check(np.array([0.5, -0.9], np.float32)), 0)
        self.assertEqual(m.check(np.array([1.0, -1.2, 0.1], np.float32)), 2)
        self.assertEqual((m.count, m.peak), (2, np.float32(1.2)))


if __name__ == "__main__":
    unittest.main()
