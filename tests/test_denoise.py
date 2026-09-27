"""Run with: venv/bin/python -m unittest discover -s tests"""
import types
import unittest

import numpy as np

from fixtures import load_speech, tone
from focus_ear.config import Config
from focus_ear.pipeline import Pipeline


def _model():
    try:
        from focus_ear.denoise import load_model
        return load_model()
    except Exception:  # noqa: BLE001 - not installed / not downloaded
        return None


MODEL = _model()


class ResamplerTest(unittest.TestCase):
    def test_flat_passband_and_exact_counts_whatever_the_chunking(self):
        from focus_ear.denoise import PolyphaseResampler

        for a, b in ((44100, 48000), (48000, 44100)):
            for freq in (100, 1000, 8000, 15000):
                x = tone(freq, 2.0, a, amp=0.5)
                r = PolyphaseResampler(a, b)
                y = np.concatenate([r.process(x[i:i + 997]) for i in range(0, len(x), 997)])
                self.assertEqual(len(y), -(-len(x) * b // a))
                gain = 20 * np.log10(np.std(y[b // 4:-b // 4]) / np.std(x))
                self.assertAlmostEqual(gain, 0.0, delta=0.05, msg=f"{a}->{b} at {freq} Hz")

    def test_chunk_size_does_not_change_the_output(self):
        from focus_ear.denoise import PolyphaseResampler

        x = np.random.default_rng(0).standard_normal(20000).astype(np.float32)
        one = PolyphaseResampler(44100, 48000).process(x)
        r = PolyphaseResampler(44100, 48000)
        many = np.concatenate([r.process(x[i:i + 1411]) for i in range(0, len(x), 1411)])
        np.testing.assert_allclose(many, one, atol=1e-6)


@unittest.skipIf(MODEL is None, "DeepFilterNet not installed or not downloaded")
class DeepFilterStreamTest(unittest.TestCase):
    def test_streaming_equals_the_offline_model(self):
        import soxr
        import torch

        from focus_ear.denoise import DeepFilterStream, _import_df

        model, params = MODEL
        enh = _import_df()
        x = soxr.resample(load_speech(), 16000, 48000).astype(np.float32)
        x += 0.02 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
        _, df_state, _ = enh.init_df(log_level="WARNING", log_file=None)
        offline = enh.enhance(model, df_state, torch.from_numpy(x)[None]).numpy()[0]
        s = DeepFilterStream(model, params)
        y = np.concatenate([s.process(x[i:i + 1536]) for i in range(0, len(x), 1536)])  # not hop-aligned
        a, b = y[s.delay:], offline[:len(y) - s.delay]
        mid = slice(48000, len(a) - 4800)  # offline pads the last frames differently
        np.testing.assert_allclose(a[mid], b[mid], atol=1e-5)
        self.assertEqual(s.delay, 1440)  # 30 ms: STFT overlap + two frames of lookahead

    def test_it_removes_noise(self):
        from focus_ear.denoise import Denoiser

        d = Denoiser(44100, 1411, *MODEL)
        noise = (0.02 * np.random.default_rng(1).standard_normal(3 * 44100)).astype(np.float32)
        y = np.concatenate([d.process(noise[i:i + 1411]) for i in range(0, len(noise) - 1411, 1411)])
        self.assertLess(np.std(y[44100:]), np.std(noise) / 10)  # >20 dB quieter


@unittest.skipIf(MODEL is None, "DeepFilterNet not installed or not downloaded")
class DenoiserTest(unittest.TestCase):
    FS, BLOCK = 44100, 1411

    def run_blocks(self, d, x):
        return np.concatenate([d.process(x[i:i + self.BLOCK]) for i in range(0, len(x) - self.BLOCK + 1,
                                                                                self.BLOCK)])

    def test_delay_stays_fixed_for_minutes(self):
        # soxr's streaming resamplers wandered by hundreds of samples every ~14 s;
        # the FIFO then ran dry, clicking and shifting the delay.
        from focus_ear.denoise import Denoiser

        d = Denoiser(self.FS, self.BLOCK, *MODEL, bypass=True)
        x = (0.1 * np.random.default_rng(2).standard_normal(180 * self.FS)).astype(np.float32)
        y = self.run_blocks(d, x)
        self.assertEqual(d.underflows, 0)
        for t in (5, 60, 120, 175):
            a = x[t * self.FS:t * self.FS + 4000]
            lag = int(np.argmax(np.correlate(y[t * self.FS:t * self.FS + 4000 + 4000], a, mode="valid")))
            self.assertEqual(lag, d.delay_samples, f"at {t} s")
        self.assertLess(d.delay_ms, 45)

    def test_runs_on_a_fresh_thread(self):
        # The pipeline worker is a new thread: torch's grad mode is on there even though
        # load_model() turned it off on the main thread. This crashed the live worker once.
        import threading

        import torch

        from focus_ear.denoise import Denoiser

        d = Denoiser(self.FS, self.BLOCK, *MODEL)
        errors = []

        def work():
            torch.set_grad_enabled(True)
            try:
                for _ in range(10):
                    d.process(np.zeros(self.BLOCK, np.float32))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        t = threading.Thread(target=work)
        t.start()
        t.join()
        self.assertEqual(errors, [])

    def test_every_block_comes_back_the_same_size(self):
        from focus_ear.denoise import Denoiser

        d = Denoiser(self.FS, self.BLOCK, *MODEL)
        for n in (self.BLOCK, 1000, 1800, 1):
            self.assertEqual(len(d.process(np.zeros(n, np.float32))), n)

    def test_mix_zero_is_the_original_delayed(self):
        from focus_ear.denoise import Denoiser

        x = (0.1 * np.random.default_rng(3).standard_normal(2 * self.FS)).astype(np.float32)
        mixed = self.run_blocks(Denoiser(self.FS, self.BLOCK, *MODEL, mix=0.0), x)
        bypass = self.run_blocks(Denoiser(self.FS, self.BLOCK, *MODEL, bypass=True), x)
        np.testing.assert_allclose(mixed, bypass, atol=1e-6)


class FakeDenoiser:
    """Negates the signal and delays it, so tests can tell which path saw what."""
    name = "denoise"

    def __init__(self, delay):
        from focus_ear.gain import DelayLine
        self.delay_samples = delay
        self._d = DelayLine(delay)

    def process(self, block, ctx=None):
        return -self._d.process(block)


class Recorder:
    name = "recorder"

    def __init__(self):
        self.frames = []

    def analyze(self, frame, ctx):
        self.frames.append(frame.copy())

    def reset(self):
        pass


class ScopeTest(unittest.TestCase):
    """--denoise-scope routes the denoised audio to the models, the ears, or both."""

    def run_scope(self, scope):
        engine = types.SimpleNamespace(samplerate=16000, blocksize=512)
        rec = Recorder()
        pipe = Pipeline(engine, Config(), analyzers=[rec], denoiser=FakeDenoiser(1024), denoise_scope=scope)
        x = np.linspace(0.01, 0.5, 512 * 8, dtype=np.float32)
        out = np.concatenate([pipe.process_block(x[i:i + 512].copy()) for i in range(0, len(x), 512)])
        return x, np.concatenate(rec.frames), out, pipe

    def test_all(self):
        x, heard, played, pipe = self.run_scope("all")
        np.testing.assert_allclose(heard[1024:], -x[:-1024])
        np.testing.assert_allclose(played[1024:], -x[:-1024])
        self.assertEqual(pipe.delay_samples, 1024)

    def test_playback(self):
        x, heard, played, _ = self.run_scope("playback")
        np.testing.assert_allclose(heard, x)                  # models: raw and early
        np.testing.assert_allclose(played[1024:], -x[:-1024])  # ears: denoised

    def test_models(self):
        x, heard, played, pipe = self.run_scope("models")
        np.testing.assert_allclose(heard[1024:], -x[:-1024])  # models: denoised
        np.testing.assert_allclose(played[1024:], x[:-1024])  # ears: raw, delayed to stay in step
        self.assertEqual(pipe.delay_samples, 1024)


if __name__ == "__main__":
    unittest.main()
