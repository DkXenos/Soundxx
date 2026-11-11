"""Engine changes for the app: stable speaker keys, naming by id, pause, stage swaps, output switching,
preflight and the headless engine's command handling. Run with: venv/bin/python -m unittest discover -s tests"""
import logging
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from fixtures import SPEECH_SPANS, load_speech
from focus_ear import audio_io, preflight
from focus_ear.audio_io import AudioEngine
from focus_ear.clustering import OnlineClusterer, SpeakerTracker
from focus_ear.config import Config
from focus_ear.gain import DelayLine
from focus_ear.pipeline import Pipeline
from focus_ear.transcript import UtteranceSegmenter
from focus_ear.vad import VadAnalyzer
from test_audio_io import FakeSoundDevice, FastEngine


def voice(i, dim=192, noise=0.0, seed=None):
    rng = np.random.default_rng(1000 + i)
    v = rng.standard_normal(dim)
    if noise:
        v = v + noise * np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def tracker(**kw):
    return SpeakerTracker(OnlineClusterer(threshold=0.5, merge_threshold=0.6, min_count=1, **kw))


class StableKeysTest(unittest.TestCase):
    def test_keys_stay_with_a_speaker_when_someone_listed_before_them_goes(self):
        t = tracker(timeout_s=10)
        for i in range(3):
            t.observe(voice(i), now=float(i))
        keys = {v.label: v.key for v in t.snapshot(now=3).speakers}
        self.assertEqual(keys, {"Speaker 1": 1, "Speaker 2": 2, "Speaker 3": 3})
        t.observe(voice(2), now=11.5)  # 1 times out (last heard at 0), 2 at 1.0 too
        t.observe(voice(2), now=11.6)
        keys = {v.label: v.key for v in t.snapshot(now=12).speakers}
        self.assertEqual(keys, {"Speaker 3": 3})  # not renumbered to 1
        t.observe(voice(4), now=12)
        keys = {v.label: v.key for v in t.snapshot(now=12).speakers}
        self.assertEqual(keys["Speaker 4"], 1)  # the lowest free key
        self.assertEqual(t.select(3), "Focusing on Speaker 3")
        self.assertEqual(t.select(2), "No speaker 2")

    def test_select_and_name_by_id_and_remembering_is_optional(self):
        t = tracker()
        t.observe(voice(0), now=0)
        t.observe(voice(1), now=1)
        self.assertEqual(t.select_id(2), "Focusing on Speaker 2")
        self.assertEqual(t.snapshot().selected_label, "Speaker 2")
        self.assertIn("no longer", t.select_id(99))
        self.assertEqual(t.name_speaker(1, "Mahasiswa A", remember=False), [])
        profiles = t.name_speaker(2, "  Pak Budi ", remember=True)
        self.assertEqual([n for n, _ in profiles], ["Pak Budi"])
        labels = {x["id"]: (x["label"], x["enrolled"]) for x in t.export()}
        self.assertEqual(labels, {1: ("Mahasiswa A", False), 2: ("Pak Budi", True)})
        self.assertIsNone(t.name_speaker(99, "x"))
        self.assertIsNone(t.name_speaker(1, "   "))

    def test_a_named_speaker_is_not_forgotten_even_unremembered(self):
        t = tracker(timeout_s=5)
        t.observe(voice(0), now=0)
        t.name_speaker(1, "Ani", remember=False)
        t.observe(voice(1), now=100)
        self.assertIn("Ani", [v.label for v in t.snapshot(now=100).speakers])


class PauseTest(unittest.TestCase):
    """Pausing in the silence of the speech fixture: what follows keeps its position in the recording."""

    def test_paused_audio_is_not_analysed_and_times_are_recording_positions(self):
        speech = load_speech()
        engine = types.SimpleNamespace(samplerate=16000, blocksize=512)
        utts = []
        seg = UtteranceSegmenter(utts.append, min_ms=200)
        tap = types.SimpleNamespace(name="rec", n=0)
        tap.observe = lambda block, ctx: setattr(tap, "n", tap.n + len(block))
        pipe = Pipeline(engine, Config(), analyzers=[VadAnalyzer(), seg], raw_taps=[tap])
        loud = (0.3 * np.sin(np.arange(512) / 3)).astype(np.float32)
        pause_at = int(3.0 * 16000 / 512)
        outs = []
        for i in range(0, len(speech) - 511, 512):
            if i // 512 == pause_at:
                pipe.set_paused(True)
                for _ in range(int(3.0 * 16000 / 512)):  # 3 s of loud audio while paused
                    out = pipe.process_block(loud.copy())
                    outs.append(float(np.abs(out).max()))
                self.assertGreater(pipe.in_db, -20)  # the meter still moves
                pipe.set_paused(False)
            pipe.process_block(speech[i:i + 512].copy())
        seg.flush()
        self.assertEqual(max(outs), 0.0)  # nothing played while paused
        self.assertAlmostEqual(tap.n / 16000, (len(speech) // 512) * 512 / 16000, places=2)  # nothing recorded
        self.assertEqual(len(utts), 2)
        for u, (a, b) in zip(utts, SPEECH_SPANS):
            self.assertAlmostEqual(u.start, a, delta=0.3)
            self.assertAlmostEqual(u.end, b, delta=0.3)

    def test_pausing_closes_the_utterance_in_progress(self):
        engine = types.SimpleNamespace(samplerate=16000, blocksize=512)
        utts = []
        seg = UtteranceSegmenter(utts.append, min_ms=200)
        speech = load_speech()
        pipe = Pipeline(engine, Config(), analyzers=[VadAnalyzer(), seg])
        for i in range(0, int(1.5 * 16000), 512):
            pipe.process_block(speech[i:i + 512].copy())
        self.assertEqual(utts, [])
        pipe.set_paused(True)
        pipe.process_block(np.zeros(512, np.float32))
        self.assertEqual(len(utts), 1)  # handed to Whisper at the pause, not lost


class StageSwapTest(unittest.TestCase):
    def test_swapping_stages_recomputes_the_playback_delay(self):
        engine = types.SimpleNamespace(samplerate=16000, blocksize=512)
        delay = DelayLine(4000)
        delay.name = "speaker-gain"
        delay.delay_samples = 4000
        delay_stage = types.SimpleNamespace(name="speaker-gain", delay_samples=4000,
                                            process=lambda b, ctx: delay.process(b))
        pipe = Pipeline(engine, Config(), stages=[])
        self.assertEqual(pipe.delay_samples, 0)
        pipe.set_stages([delay_stage])
        self.assertEqual(pipe.delay_samples, 0)  # applied by the worker, at the next block
        pipe.process_block(np.ones(512, np.float32))
        self.assertEqual(pipe.delay_samples, 4000)
        self.assertEqual(pipe.delays, {"speaker-gain": 4000})


class SetOutputTest(unittest.TestCase):
    def wait_for(self, cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not cond() and time.monotonic() < deadline:
            time.sleep(0.01)
        return cond()

    def test_switch_to_none_and_back_while_the_mic_keeps_running(self):
        sd = FakeSoundDevice()
        sd.buds_awake = True
        probe = lambda timeout=5.0: [{"name": d["name"], "inputs": d["max_input_channels"],  # noqa: E731
                                      "outputs": d["max_output_channels"]} for d in sd.query_devices()]
        with mock.patch.object(audio_io, "sd", sd), mock.patch.object(audio_io, "probe_devices", probe), \
                self.assertLogs("focus_ear.audio_io", logging.INFO):
            engine = FastEngine(Config(output_device="Mewo"))
            engine.start()
            try:
                self.assertTrue(self.wait_for(lambda: engine.state == "running"))
                engine.set_output(None)
                self.assertTrue(self.wait_for(lambda: engine.state == "waiting"))
                self.assertIsNone(engine.output_name)
                beat = engine.stats.in_heartbeat
                self.assertTrue(self.wait_for(lambda: engine.stats.in_heartbeat > beat + 0.1))  # mic alive
                time.sleep(0.2)
                self.assertEqual(engine.state, "waiting")  # no output wanted: stays so, no reconnect
                engine.set_output("Mewo")
                self.assertTrue(self.wait_for(lambda: engine.state == "running"))
                self.assertEqual(engine.output_name, "Mewo Buds")
                self.assertEqual(engine.stats.reconnects, 0)  # chosen switches aren't reconnects
            finally:
                engine.stop()


class PreflightTest(unittest.TestCase):
    DEVICES = [{"name": "MacBook Pro Microphone", "inputs": 1, "outputs": 0, "samplerate": 44100},
               {"name": "MacBook Pro Speakers", "inputs": 0, "outputs": 2, "samplerate": 44100},
               {"name": "Mewo", "inputs": 1, "outputs": 2, "samplerate": 48000}]

    def levels(self, checks):
        return {c["id"]: c["level"] for c in checks}

    def test_buds_present_is_ok_loudspeakers_block_absent_buds_warn(self):
        lv = self.levels(preflight.run_preflight(self.DEVICES, "Mewo", -40.0, False, [], "authorized", lid=False))
        self.assertEqual((lv["mic"], lv["output"], lv["permission"], lv["signal"]), ("ok",) * 4)
        lv = self.levels(preflight.run_preflight(self.DEVICES, "MacBook Pro Speakers", -40.0, False, [], "authorized", lid=False))
        self.assertEqual(lv["output"], "blocker")
        checks = preflight.run_preflight(self.DEVICES[:2], "Mewo", -40.0, False, [], "authorized", lid=False)
        self.assertEqual(self.levels(checks)["output"], "warning")
        self.assertTrue(preflight.can_start(checks))

    def test_silent_mic_denied_permission_and_no_mic_block(self):
        self.assertFalse(preflight.can_start(preflight.run_preflight(self.DEVICES, "Mewo", None, True, [], "authorized")))
        self.assertFalse(preflight.can_start(preflight.run_preflight(self.DEVICES, "Mewo", -40, False, [], "denied")))
        self.assertFalse(preflight.can_start(preflight.run_preflight(self.DEVICES[1:], "Mewo", -40, False, [], "authorized")))
        self.assertFalse(preflight.can_start(preflight.run_preflight(None, "Mewo")))

    def test_a_closed_lid_blocks(self):
        self.assertEqual(preflight.check_lid(True).level, "blocker")
        self.assertEqual(preflight.check_lid(False).level, "ok")

    def test_low_disk_blocks_and_interrupted_sessions_warn(self):
        usage = types.SimpleNamespace(free=1.5e9, total=500e9, used=0)
        with mock.patch.object(preflight.shutil, "disk_usage", return_value=usage):
            self.assertEqual(preflight.check_disk().level, "blocker")
        usage.free = 4e9
        with mock.patch.object(preflight.shutil, "disk_usage", return_value=usage):
            self.assertEqual(preflight.check_disk().level, "warning")
        self.assertEqual(preflight.check_interrupted(["x"]).level, "warning")

    def test_model_cache_detection(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            self.assertFalse(preflight.hf_cached("a/b", hub))
            snap = hub / "models--a--b" / "snapshots" / "123"
            snap.mkdir(parents=True)
            (snap / "weights.safetensors").write_bytes(b"0" * 2_000_000)
            self.assertTrue(preflight.hf_cached("a/b", hub))


class HeadlessCommandTest(unittest.TestCase):
    """Engine.handle with a stand-in session: commands reach the tracker, engine and pipeline."""

    def make(self):
        from focus_ear import headless
        proto = types.SimpleNamespace(sent=[])
        proto.send = lambda type_, **f: proto.sent.append((type_, f))
        eng = headless.Engine(proto, "listen", "Mewo", False)
        t = tracker()
        t.observe(voice(0), now=0)
        audio = types.SimpleNamespace(cfg=Config(output_device="Mewo"), samplerate=16000, blocksize=512,
                                      switched=[])
        audio.set_output = audio.switched.append
        pipe = Pipeline(types.SimpleNamespace(samplerate=16000, blocksize=512), Config())
        eng.session = types.SimpleNamespace(tracker=t, engine=audio, pipeline=pipe, gate=None, speaker_gain=None,
                                            agc=None, limiter=None)
        eng.phase = "standby"
        return eng, proto

    def test_commands(self):
        eng, proto = self.make()
        with mock.patch("focus_ear.headless.update_settings") as upd, \
                mock.patch("focus_ear.headless.save_profiles") as save:
            self.assertTrue(eng.handle({"cmd": "select", "speaker": 1}))
            self.assertEqual(eng.session.tracker.selected_id, 1)
            eng.handle({"cmd": "rename", "speaker": 1, "name": "Pak Budi", "remember": True})
            save.assert_called_once()
            self.assertEqual(eng.session.tracker.label(1), "Pak Budi")
            eng.handle({"cmd": "set_output", "device": None})
            self.assertEqual(eng.session.engine.switched, [None])
            upd.assert_any_call(output_device=None)
            eng.handle({"cmd": "set_mode", "mode": "full"})
            self.assertEqual(eng.mode, "full")
            self.assertIsNotNone(eng.session.speaker_gain)
            self.assertEqual(eng.session.pipeline._new_stages[1].name, "speaker-gain")
            eng.handle({"cmd": "set_mode", "mode": "listen"})
            self.assertIsNone(eng.session.speaker_gain)
            eng.handle({"cmd": "pause"})  # nothing recording: ignored, no error
            eng.handle({"cmd": "bogus"})
            self.assertFalse(eng.handle({"cmd": "stop"}))
        levels = [f["level"] for t, f in proto.sent if t == "event"]
        self.assertNotIn("error", levels)
        self.assertIn("warning", levels)  # the unknown command

    def test_mode_overrides_change_only_the_stage_settings(self):
        from focus_ear.headless import app_config, mode_overrides
        cfg = app_config("listen", "Mewo")
        self.assertTrue(cfg.speakers)            # speakers identified in both modes
        self.assertEqual(cfg.buffer_ms, 64.0)    # one buffer for both
        listen = mode_overrides(cfg, "listen")
        full = mode_overrides(cfg, "full")
        self.assertEqual((listen.gate_attenuation_db, listen.lookahead_ms), (-12.0, 0.0))
        self.assertEqual((full.gate_attenuation_db, full.lookahead_ms), (-18.0, 250.0))
        self.assertEqual(listen.buffer_ms, full.buffer_ms)


if __name__ == "__main__":
    unittest.main()
