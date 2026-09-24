"""Run with: venv/bin/python -m unittest discover -s tests"""
import json
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from fixtures import load_speech, tone
from focus_ear.clustering import OnlineClusterer, SpeakerTracker
from focus_ear.config import Config
from focus_ear.embeddings import MODEL_DIR, SpeakerAnalyzer
from focus_ear.gain import SpeakerGain, db_to_gain
from focus_ear.pipeline import BlockContext, Pipeline
from focus_ear.profiles import load_profiles, save_profiles

FS = 44100
BLOCK = 1411


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(192)
    return v / np.linalg.norm(v)


VOICE_A, VOICE_B = unit(1), unit(2)


def pitch_embedder(audio: np.ndarray) -> np.ndarray:
    """Stand-in for ECAPA: low tones are voice A, high tones voice B."""
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * 16000 / len(audio)
    return VOICE_A if peak_hz < 500 else VOICE_B


class AlwaysSpeech:
    """Stand-in VAD (silero rightly ignores pure tones)."""
    name = "vad"

    def analyze(self, frame, ctx):
        ctx.is_speech, ctx.vad_prob = True, 1.0

    def reset(self):
        pass


def tracker(**kw):
    return SpeakerTracker(OnlineClusterer(threshold=0.5, min_count=3, **kw), 1.0, db_to_gain(-20))


class SpeakerAnalyzerTest(unittest.TestCase):
    def feed(self, analyzer, n_frames, speech=True):
        ctx = BlockContext(index=0, adc_time=0.0, samplerate=16000, is_speech=speech)
        frame = tone(200, 0.032)[:512]
        for _ in range(n_frames):
            analyzer.analyze(frame, ctx)
        return ctx

    def test_embeds_when_the_window_fills_then_once_per_hop(self):
        calls = []
        a = SpeakerAnalyzer(lambda x: calls.append(len(x)) or VOICE_A, tracker(), window_s=1.5, hop_s=0.25,
                            background=False)
        self.feed(a, 46)  # window = 47 frames
        self.assertEqual(calls, [])
        self.feed(a, 1)
        self.assertEqual(calls, [47 * 512])
        self.feed(a, 8 * 5)
        self.assertEqual(len(calls), 6)

    def test_sets_speaker_id_only_once_confirmed(self):
        a = SpeakerAnalyzer(lambda x: VOICE_A, tracker(), window_s=1.0, hop_s=0.25, background=False)
        ctx = self.feed(a, 31 + 8)  # two voiceprints: not confirmed yet
        self.assertIsNone(ctx.speaker_id)
        ctx = self.feed(a, 8)
        self.assertEqual(ctx.speaker_id, 1)

    def test_speaker_is_forgotten_after_a_pause(self):
        a = SpeakerAnalyzer(lambda x: VOICE_A, tracker(), window_s=1.0, hop_s=0.25, forget_after_s=0.5,
                            background=False)
        ctx = self.feed(a, 31 + 16)
        self.assertEqual(ctx.speaker_id, 1)
        ctx.is_speech = False
        for _ in range(15):
            a.analyze(np.zeros(512, np.float32), ctx)
        self.assertEqual(ctx.speaker_id, 1)  # a short pause keeps them
        a.analyze(np.zeros(512, np.float32), ctx)
        self.assertIsNone(ctx.speaker_id)    # 0.5 s of silence: unknown again
        # ...and the next voiceprint holds only post-pause speech, not the previous turn's.
        self.assertEqual(sum(s for _, s in a._window), 0)

    def test_mostly_silent_windows_are_skipped(self):
        calls = []
        a = SpeakerAnalyzer(lambda x: calls.append(1) or VOICE_A, tracker(), window_s=1.0, hop_s=0.25,
                            background=False)
        self.feed(a, 100, speech=False)
        self.assertEqual(calls, [])
        self.assertGreater(a.skipped_silent, 0)

    def test_background_inference_never_blocks_and_drops_windows_while_busy(self):
        def slow_embed(x):
            time.sleep(0.2)
            return VOICE_A

        a = SpeakerAnalyzer(slow_embed, tracker(), window_s=1.0, hop_s=0.25)
        t = time.perf_counter()
        self.feed(a, 31 + 8 * 4)  # 5 windows due at once; the first keeps the model busy
        self.assertLess(time.perf_counter() - t, 0.1, "the worker waited for inference")
        self.assertEqual(a.dropped, 4)
        for _ in range(3):  # let three windows through, one at a time, to confirm the speaker
            while a._busy:
                time.sleep(0.01)
            self.feed(a, 8)
        while a._busy:
            time.sleep(0.01)
        ctx = self.feed(a, 1)  # the finished result is applied on the next frame
        a.close()
        self.assertEqual(ctx.speaker_id, 1)
        self.assertEqual(a.embedded, 4)


class SpeakerGainTest(unittest.TestCase):
    def test_delays_audio_and_holds_gain_when_speaker_unknown(self):
        targets = {None: None, 1: 1.0, 2: 0.1}
        g = SpeakerGain(FS, targets.get, lookahead_ms=250, ramp_ms=50)
        self.assertEqual(g.delay_samples, round(0.25 * FS))
        ctx = BlockContext(index=0, adc_time=0.0, samplerate=FS, speaker_id=2)
        g.process(np.ones(FS, np.float32), ctx)          # ramps to 0.1
        ctx.speaker_id = None
        out = g.process(np.ones(BLOCK, np.float32), ctx)  # unknown: holds 0.1
        np.testing.assert_allclose(out, 0.1, rtol=1e-6)


class ProfilesTest(unittest.TestCase):
    def test_round_trip_and_bad_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "speakers.json"
            self.assertEqual(load_profiles(path), [])
            save_profiles([("Prof", VOICE_A.astype(np.float32))], path)
            [(name, v)] = load_profiles(path)
            self.assertEqual(name, "Prof")
            np.testing.assert_allclose(v, VOICE_A, atol=1e-6)

            data = json.loads(path.read_text())
            data["model"] = "some/other-model"
            path.write_text(json.dumps(data))
            self.assertEqual(load_profiles(path), [])  # incompatible voiceprints are ignored
            path.write_text("{not json")
            self.assertEqual(load_profiles(path), [])  # damaged file: ignored, not a crash


class SpeakerPipelineTest(unittest.TestCase):
    """Stages 3+4 end to end, offline: a low voice (A) and a high voice (B) take turns."""

    def test_selected_speaker_at_unity_others_attenuated(self):
        t = tracker()
        engine = types.SimpleNamespace(samplerate=FS, blocksize=BLOCK)
        gain = SpeakerGain(FS, t.gain_target, lookahead_ms=250)
        analyzer = SpeakerAnalyzer(pitch_embedder, t, background=False)
        pipe = Pipeline(engine, Config(), analyzers=[AlwaysSpeech(), analyzer], stages=[gain])
        turns = [(200, 4.0), (900, 4.0), (200, 4.0)]  # A, B, A
        audio = np.concatenate([tone(f, s, rate=FS) for f, s in turns])
        gains = []
        for i in range(0, len(audio) - BLOCK + 1, BLOCK):
            if i == 3 * FS // BLOCK * BLOCK:
                self.assertEqual(t.select(1), "Focusing on Speaker 1")  # pick A after 3 s
            pipe.process_block(audio[i:i + BLOCK].copy())
            gains.append(gain.gain.current)
        gains = np.array(gains)
        per_s = FS / BLOCK

        def at(sec):
            return gains[int(sec * per_s)]

        self.assertEqual(at(3.9), 1.0)                            # A, selected
        self.assertAlmostEqual(at(7.5), db_to_gain(-20), 6)       # B, attenuated
        self.assertEqual(at(11.5), 1.0)                           # A again
        # Identification lag: the switch lands within ~1.5 s of B starting.
        switch = np.argmax(gains[int(4 * per_s):] < 0.5) / per_s
        self.assertLess(switch, 1.5)
        self.assertEqual(pipe.delay_samples, round(0.25 * FS))


@unittest.skipUnless((MODEL_DIR / "hyperparams.yaml").exists(), "ECAPA model not downloaded yet")
class EcapaEmbedderTest(unittest.TestCase):
    def test_unit_192_dim_and_deterministic(self):
        from focus_ear.embeddings import EcapaEmbedder

        e = EcapaEmbedder("auto")
        x = load_speech()[:24000]
        v = e(x)
        self.assertEqual(v.shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        np.testing.assert_allclose(e(x), v, atol=1e-4)
        self.assertIn(e.device, ("mps", "cpu"))


if __name__ == "__main__":
    unittest.main()
