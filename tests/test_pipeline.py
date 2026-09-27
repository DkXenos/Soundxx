"""Run with: venv/bin/python -m unittest discover -s tests"""
import time
import types
import unittest

import numpy as np
import soxr

from fixtures import SILENCE_SPAN, SPEECH_SPANS, load_speech, tone
from focus_ear.audio_io import RingBuffer
from focus_ear.config import Config
from focus_ear.gain import NoiseGate, db_to_gain
from focus_ear.pipeline import AnalysisFeed, Pipeline
from focus_ear.vad import VadAnalyzer

FS = 44100
BLOCK = 1411


class AnalysisFeedTest(unittest.TestCase):
    def test_44k_stream_becomes_512_sample_16k_frames_at_the_right_pitch(self):
        feed = AnalysisFeed(FS)
        x = tone(1000, 3.0, rate=FS)
        frames = [f for i in range(0, len(x) - BLOCK + 1, BLOCK) for f in feed.push(x[i:i + BLOCK])]
        self.assertTrue(all(len(f) == 512 for f in frames))
        self.assertAlmostEqual(len(frames), 3.0 * 16000 / 512, delta=3)
        y = np.concatenate(frames[10:])
        peak_hz = np.argmax(np.abs(np.fft.rfft(y))) * 16000 / len(y)
        self.assertAlmostEqual(peak_hz, 1000, delta=5)

    def test_16k_input_passes_through_and_frames_survive_block_reuse(self):
        feed = AnalysisFeed(16000)
        block = np.ones(512, np.float32)
        (frame,) = feed.push(block)
        block[:] = 0.0
        self.assertTrue(np.all(frame == 1.0))


class OfflinePipelineTest(unittest.TestCase):
    """The real VAD + gate chain on recorded speech at 44.1 kHz, no audio devices."""

    def run_chain(self, gate):
        speech = soxr.resample(load_speech(), 16000, FS).astype(np.float32)
        engine = types.SimpleNamespace(samplerate=FS, blocksize=BLOCK)
        pipe = Pipeline(engine, Config(), analyzers=[VadAnalyzer()], stages=[gate] if gate else [])
        out, gains = [], []
        for i in range(0, len(speech) - BLOCK + 1, BLOCK):
            block = speech[i:i + BLOCK].copy()
            out.append(pipe.process_block(block).copy())
            gains.append(gate.gain.current if gate else 1.0)
        return speech[:len(out) * BLOCK], np.concatenate(out), np.array(gains), pipe

    def block_range(self, start_s, end_s):
        return slice(int(start_s * FS / BLOCK), int(end_s * FS / BLOCK))

    def test_gate_passes_speech_and_attenuates_the_pause(self):
        gate = NoiseGate(FS, attenuation_db=-18)
        _, _, gains, pipe = self.run_chain(gate)
        speech = self.block_range(SPEECH_SPANS[0][0] + 0.3, SPEECH_SPANS[0][1] - 0.2)
        pause = self.block_range(SILENCE_SPAN[0] + 0.5, SILENCE_SPAN[1] - 0.2)
        self.assertGreater(np.mean(gains[speech] == 1.0), 0.9)
        self.assertGreater(np.mean(np.isclose(gains[pause], db_to_gain(-18))), 0.9)
        self.assertGreater(pipe.speech_blocks, 0)

    def test_no_gate_leaves_audio_untouched(self):
        original, out, _, pipe = self.run_chain(None)
        np.testing.assert_array_equal(out, original)
        self.assertGreater(pipe.speech_blocks, 0)  # VAD still observes


class FakeEngine:
    """Just the worker-facing half of AudioEngine, backed by real rings."""
    samplerate, blocksize = FS, BLOCK

    def __init__(self):
        self.in_ring, self.out_ring = RingBuffer(4 * FS), RingBuffer(4 * FS)

    def input_backlog(self):
        return self.in_ring.available()

    def read_input(self, out):
        self.in_ring.read_into(out)
        return 0.0

    def skip_input(self, n):
        return self.in_ring.skip(n)

    def write_output(self, block, adc_time):
        self.out_ring.write(block)


class WorkerTest(unittest.TestCase):
    def test_falling_behind_drops_old_audio_instead_of_adding_latency(self):
        engine = FakeEngine()
        ramp = np.arange(10 * BLOCK, dtype=np.float32) / (10 * BLOCK)  # below full scale: the output is clamped there
        engine.in_ring.write(ramp)  # 320 ms behind
        pipe = Pipeline(engine, Config(max_backlog_ms=100))
        pipe.start()
        time.sleep(0.2)
        pipe.stop()
        self.assertEqual(pipe.drop_events, 1)
        self.assertEqual(pipe.dropped, 9 * BLOCK)
        self.assertEqual(pipe.blocks, 1)
        played = np.zeros(BLOCK, np.float32)
        engine.out_ring.read_into(played)
        np.testing.assert_array_equal(played, ramp[9 * BLOCK:])  # the newest block

    def test_a_failing_stage_plays_the_audio_unprocessed_instead_of_killing_the_worker(self):
        class Broken:
            name, delay_samples = "broken", 0

            def process(self, block, ctx):
                raise RuntimeError("model exploded")

        engine = FakeEngine()
        pipe = Pipeline(engine, Config(max_backlog_ms=1000), stages=[Broken()])
        signal = np.linspace(-0.5, 0.5, 4 * BLOCK, dtype=np.float32)
        with self.assertLogs("focus_ear.pipeline", "ERROR"):
            pipe.start()
            engine.in_ring.write(signal)
            deadline = time.monotonic() + 2
            while pipe.errors < 4 and time.monotonic() < deadline:
                time.sleep(0.01)
            pipe.stop()
        self.assertEqual(pipe.errors, 4)
        self.assertIn("model exploded", pipe.last_error)
        played = np.zeros(4 * BLOCK, np.float32)
        engine.out_ring.read_into(played)
        np.testing.assert_array_equal(played, signal)


if __name__ == "__main__":
    unittest.main()
