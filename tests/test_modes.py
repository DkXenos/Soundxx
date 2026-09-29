"""Listen/full modes, the A/B switch, the denoiser's deadline, the raw recording. Run with:
venv/bin/python -m unittest discover -s tests"""
import tempfile
import time
import types
import unittest
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import soxr

from focus_ear.config import Config
from focus_ear.gain import DelayLine
from focus_ear.main import build_parser, config_from_args
from focus_ear.pipeline import MetricsSnapshot, Pipeline, latency_breakdown
from focus_ear.recorder import RawRecorder
from focus_ear.transcript import TranscriptWriter, session_stem
from test_denoise import MODEL, FakeDenoiser


def parse(*argv):
    return config_from_args(build_parser().parse_args(list(argv)))


class ModeTest(unittest.TestCase):
    def test_live_runs_default_to_listen(self):
        cfg = parse()
        self.assertEqual(cfg.mode, "listen")
        self.assertFalse(cfg.speakers)            # no speaker model
        self.assertEqual(cfg.lookahead_ms, 0.0)   # nothing to align
        self.assertEqual(cfg.gate_attenuation_db, -12.0)
        self.assertEqual(cfg.buffer_ms, 40.0)
        self.assertTrue(cfg.record)
        self.assertTrue(cfg.transcribe)

    def test_full_mode_is_the_chain_as_it_was(self):
        cfg = parse("--mode", "full")
        full = Config()
        for key in ("speakers", "lookahead_ms", "gate_attenuation_db", "buffer_ms", "drift_slack_blocks"):
            self.assertEqual(getattr(cfg, key), getattr(full, key), key)
        self.assertEqual((cfg.lookahead_ms, cfg.gate_attenuation_db, cfg.buffer_ms), (250.0, -18.0, 64.0))

    def test_flags_beat_the_mode_defaults(self):
        cfg = parse("--buffer-ms", "80", "--gate-attenuation", "-6", "--lookahead", "100")
        self.assertEqual((cfg.buffer_ms, cfg.gate_attenuation_db, cfg.lookahead_ms), (80.0, -6.0, 100.0))
        self.assertFalse(parse("--mode", "full", "--no-speakers").speakers)
        self.assertFalse(parse("--no-record").record)

    def test_file_mode_keeps_the_full_analysis_unless_asked(self):
        self.assertEqual(parse("--file", "x.wav").mode, "full")
        self.assertTrue(parse("--file", "x.wav").speakers)
        self.assertFalse(parse("--file", "x.wav", "--mode", "listen").speakers)


class HalfInPlace:
    """A stage that works in place, as the gains do: the raw side must not see it."""
    name, delay_samples = "half", 0

    def process(self, block, ctx):
        block *= 0.5
        return block


class Tap:
    name = "tap"

    def __init__(self):
        self.blocks = []

    def observe(self, block, ctx):
        self.blocks.append(block.copy())


def sine(n, fs=16000, freq=50.0, amp=0.5):
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / fs)).astype(np.float32)


class AbSwitchTest(unittest.TestCase):
    FS, B, DELAY = 16000, 512, 1024

    def make(self, **kw):
        engine = types.SimpleNamespace(samplerate=self.FS, blocksize=self.B)
        return Pipeline(engine, Config(), stages=[HalfInPlace()], denoiser=FakeDenoiser(self.DELAY), **kw)

    def run_blocks(self, pipe, x):
        return np.concatenate([pipe.process_block(x[i:i + self.B].copy()) for i in range(0, len(x), self.B)])

    def test_raw_is_the_unprocessed_mic_in_step_and_the_switch_is_smooth(self):
        pipe = self.make(ab=True)
        self.assertTrue(pipe.ab)
        x = sine(40 * self.B)
        first = self.run_blocks(pipe, x[:16 * self.B])
        np.testing.assert_allclose(first[self.DELAY:], -0.5 * x[:16 * self.B - self.DELAY], atol=1e-6)  # processed

        pipe.set_processing(False)
        second = self.run_blocks(pipe, x[16 * self.B:32 * self.B])
        ramp = round(self.FS * pipe.cfg.ramp_ms / 1000)
        start = 16 * self.B
        np.testing.assert_allclose(second[ramp:], x[start + ramp - self.DELAY:32 * self.B - self.DELAY], atol=1e-6)

        pipe.set_processing(True)
        third = self.run_blocks(pipe, x[32 * self.B:])
        np.testing.assert_allclose(third[ramp:], -0.5 * x[32 * self.B + ramp - self.DELAY:-self.DELAY], atol=1e-6)

        # No click: the output never steps by more than the signal itself does (a hard switch jumps ~0.75).
        out = np.concatenate((first, second, third))
        signal_step = np.max(np.abs(np.diff(x)))
        self.assertLess(np.max(np.abs(np.diff(out[self.DELAY:]))), 1.2 * signal_step)

    def test_without_ab_nothing_changes(self):
        pipe = self.make()
        self.assertFalse(pipe.ab)
        x = sine(8 * self.B)
        out = self.run_blocks(pipe, x)
        np.testing.assert_allclose(out[self.DELAY:], -0.5 * x[:-self.DELAY], atol=1e-6)

    def test_raw_taps_see_the_mic_before_anything_touches_it(self):
        tap = Tap()
        engine = types.SimpleNamespace(samplerate=self.FS, blocksize=self.B)
        pipe = Pipeline(engine, Config(), stages=[HalfInPlace()], raw_taps=[tap])  # no high-pass or denoiser
        x = sine(4 * self.B)
        self.run_blocks(pipe, x)
        np.testing.assert_array_equal(np.concatenate(tap.blocks), x)


class DeadlineDenoiser:
    name, delay_samples = "denoise", 0

    def __init__(self):
        self.deadlines = []

    def process(self, block, ctx=None, deadline=None):
        self.deadlines.append(deadline)
        return block


class DeadlineTest(unittest.TestCase):
    def run_one(self, cfg=None, **engine):
        d = DeadlineDenoiser()
        eng = types.SimpleNamespace(samplerate=16000, blocksize=512, **engine)
        Pipeline(eng, cfg or Config(), denoiser=d).process_block(np.zeros(512, np.float32))
        return d.deadlines[0]

    def test_the_denoiser_waits_only_as_long_as_the_output_lasts(self):
        self.assertAlmostEqual(self.run_one(output_headroom=lambda: 0.030), 0.030 - Pipeline.DEADLINE_MARGIN_S)
        self.assertEqual(self.run_one(output_headroom=lambda: 0.002), Pipeline.MIN_DEADLINE_S)

    def test_no_deadline_while_nothing_plays_or_offline(self):
        self.assertIsNone(self.run_one(output_headroom=lambda: None))  # Buds away: nothing to run dry
        self.assertIsNone(self.run_one())                               # --file
        self.assertIsNone(self.run_one(Config(late_bypass=False), output_headroom=lambda: 0.03))


@unittest.skipIf(MODEL is None, "DeepFilterNet not installed or not downloaded")
class ThreadedDenoiserTest(unittest.TestCase):
    FS, B = 44100, 1411

    def test_same_output_as_inline_when_on_time(self):
        from focus_ear.denoise import Denoiser

        x = (0.05 * np.random.default_rng(4).standard_normal(40 * self.B)).astype(np.float32)
        inline = Denoiser(self.FS, self.B, *MODEL)
        threaded = Denoiser(self.FS, self.B, *MODEL, threaded=True)
        try:
            for i in range(0, len(x), self.B):
                np.testing.assert_allclose(threaded.process(x[i:i + self.B], deadline=5.0),
                                           inline.process(x[i:i + self.B]), atol=1e-6)
            self.assertEqual(threaded.late_blocks, 0)
        finally:
            threaded.close()

    def test_a_late_block_plays_raw_in_step_and_the_model_carries_on(self):
        from focus_ear.denoise import Denoiser

        d = Denoiser(self.FS, self.B, *MODEL, threaded=True)
        x = (0.05 * np.random.default_rng(5).standard_normal(80 * self.B)).astype(np.float32)
        raw = DelayLine(d.delay_samples)
        try:
            outs = []
            for k, i in enumerate(range(0, len(x), self.B)):
                expected_raw = raw.process(x[i:i + self.B])
                out = d.process(x[i:i + self.B], deadline=0.0 if k == 30 else None)  # block 30 can't make it
                if k == 30:
                    np.testing.assert_array_equal(out, expected_raw)
                outs.append(out)
            self.assertEqual(d.late_blocks, 1)
            after = np.concatenate(outs[50:])
            self.assertLess(np.std(after), np.std(x) / 3)  # still denoising afterwards
        finally:
            d.close()


class RecorderTest(unittest.TestCase):
    def test_16k_wav_readable_while_recording_and_complete_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.wav"
            rec = RawRecorder(path, 48000, flush_s=0.05)
            x = sine(3 * 48000, fs=48000, freq=440, amp=0.3)
            for i in range(0, len(x), 1536):
                rec.observe(x[i:i + 1536])
            time.sleep(0.3)
            with wave.open(str(path)) as w:  # a crash now would leave this: a valid file
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
                self.assertGreater(w.getnframes(), 2.9 * 16000)
            rec.close()
            with wave.open(str(path)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
            self.assertAlmostEqual(len(pcm) / 16000, 3.0, delta=0.05)
            ref = soxr.resample(x, 48000, 16000)
            n = min(len(pcm), len(ref)) - 1000
            lag = max(range(-20, 21), key=lambda k: float(pcm[1000 + k:n + k] @ ref[1000:n]))
            self.assertLessEqual(abs(lag), 20)
            np.testing.assert_allclose(pcm[1000 + lag:n + lag], ref[1000:n], atol=2e-3)
            self.assertAlmostEqual(rec.seconds, len(pcm) / 16000)
            with self.assertRaises(FileExistsError):
                RawRecorder(path, 48000)  # never overwrites a lecture

    def test_a_stalled_disk_costs_the_oldest_audio_not_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = RawRecorder(Path(tmp) / "s.wav", 16000, flush_s=60, max_queue_s=1.0)
            for _ in range(40):
                rec.observe(np.zeros(1600, np.float32))  # 4 s, nothing written yet
            self.assertEqual(rec.dropped, 3 * 16000)
            rec.close()
            self.assertAlmostEqual(rec.seconds, 1.0)


class SessionFilesTest(unittest.TestCase):
    def test_transcript_and_recording_share_a_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, started = Path(tmp), datetime(2026, 9, 29, 9, 14)
            (d / "2026-09-29_0914.wav").touch()  # an earlier run's recording, without a transcript
            stem = session_stem(d, started)
            self.assertEqual(stem.name, "2026-09-29_0914_2")
            w = TranscriptWriter(d, started, "live", "m", "id", stem=stem, recording=d / f"{stem.name}.wav")
            w.close()
            self.assertEqual(w.md_path.name, "2026-09-29_0914_2.md")
            self.assertIn("**Recording:** 2026-09-29_0914_2.wav", w.md_path.read_text())


class LatencyBreakdownTest(unittest.TestCase):
    def test_every_delay_is_named(self):
        snap = MetricsSnapshot(interval_s=1, e2e_ms=140, in_device_ms=34, in_queue_ms=0, proc_ms=9, delay_ms=40,
                               out_buffer_ms=28, out_device_ms=33, delays={"denoise": 38.0, "limiter": 2.0})
        self.assertEqual(latency_breakdown(snap),
                         "mic 34 + queue 0 + proc 9 + denoise 38 + limiter 2 + out-buf 28 + out-dev 33")
        snap.delays = {"speaker-gain": 250.0}
        self.assertIn("speaker lookahead 250", latency_breakdown(snap))

    def test_pipeline_reports_its_delays_by_stage(self):
        from focus_ear.dsp import Limiter
        from focus_ear.gain import SpeakerGain

        engine = types.SimpleNamespace(samplerate=16000, blocksize=512)
        pipe = Pipeline(engine, Config(), denoiser=FakeDenoiser(608),
                        stages=[SpeakerGain(16000, lambda _: 1.0, 250), Limiter(16000)])
        self.assertEqual(pipe.delays, {"denoise": 608, "speaker-gain": 4000, "limiter": 32})
        self.assertEqual(pipe.delay_samples, 608 + 4000 + 32)


if __name__ == "__main__":
    unittest.main()
