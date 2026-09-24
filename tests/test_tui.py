"""Run with: venv/bin/python -m unittest discover -s tests"""
import types
import unittest

import numpy as np

from focus_ear.clustering import OnlineClusterer, SpeakerTracker
from focus_ear.config import Config
from focus_ear.pipeline import BlockContext, MetricsSnapshot
from focus_ear.tui import FocusEarApp


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(192)
    return v / np.linalg.norm(v)


def fake_session():
    tracker = SpeakerTracker(OnlineClusterer(threshold=0.5, min_count=3), 1.0, 0.1)
    for seed, t0 in ((1, 0), (2, 10)):
        for i in range(3):
            tracker.observe(unit(seed), t0 + i)
    saved = []
    snap = MetricsSnapshot(interval_s=1, e2e_ms=612, in_device_ms=34, in_queue_ms=0, proc_ms=2,
                           delay_ms=250, out_buffer_ms=48, out_device_ms=280, block_ms=32)
    session = types.SimpleNamespace(
        cfg=Config(),
        engine=types.SimpleNamespace(state="running", error=None, input_name="MacBook Pro Microphone",
                                     output_name="Mewo", samplerate=44100),
        pipeline=types.SimpleNamespace(delay_samples=11025, drop_events=0, mic_silent=False, in_db=-40.0,
                                       ctx=BlockContext(0, 0.0, 44100, vad_prob=0.9, is_speech=True)),
        metrics=types.SimpleNamespace(snapshot=lambda: snap),
        gate=None, speaker_gain=None, tracker=tracker,
        analyzer=types.SimpleNamespace(dropped=0, last_ms=9.4, embedded=6, skipped_silent=0),
        embed_device="MPS", save_profiles=saved.append, log_metrics=lambda snap: None)
    return session, saved


class TuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_select_name_reset_debug_quit(self):
        session, saved = fake_session()
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            await pilot.press("2")
            self.assertEqual(session.tracker.snapshot().selected_label, "Speaker 2")
            await pilot.press("0")
            self.assertIsNone(session.tracker.snapshot().selected_label)

            await pilot.press("1", "n")
            self.assertTrue(app.query_one("#name").display)
            # Digits and 'q' must type into the box, not select or quit.
            await pilot.press(*"Prof 2q", "enter")
            self.assertFalse(app.query_one("#name").display)
            self.assertEqual([name for name, _ in saved[0]], ["Prof 2q"])
            self.assertEqual(session.tracker.snapshot().selected_label, "Prof 2q")

            await pilot.press("n", "x", "escape")  # cancelled: nothing saved
            self.assertFalse(app.query_one("#name").display)
            self.assertEqual(len(saved), 1)

            debug = app.query_one("#debug")
            shown = debug.display
            await pilot.press("d")
            self.assertEqual(debug.display, not shown)

            await pilot.press("r")
            labels = [v.label for v in session.tracker.snapshot().speakers]
            self.assertEqual(labels, ["Prof 2q"])  # only the enrolled speaker survives a reset

            await pilot.press("q")
        self.assertEqual(app.return_code, 0)

    async def test_name_without_selection_warns_instead_of_prompting(self):
        session, saved = fake_session()
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            await pilot.press("n")
            self.assertFalse(app.query_one("#name").display)

    async def test_fatal_audio_error_exits_with_code_2(self):
        session, _ = fake_session()
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            session.engine.error = RuntimeError("CoreAudio stopped responding")
            await pilot.pause(0.3)
        self.assertEqual(app.return_code, 2)


if __name__ == "__main__":
    unittest.main()
