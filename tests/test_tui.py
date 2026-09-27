"""Run with: venv/bin/python -m unittest discover -s tests"""
import time
import types
import unittest

import numpy as np

from focus_ear.audio_io import AudioStats
from focus_ear.clustering import OnlineClusterer, SpeakerTracker
from focus_ear.config import Config
from focus_ear.pipeline import BlockContext, MetricsSnapshot
from focus_ear.transcript import TranscriptLine
from focus_ear.tui import FocusEarApp


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(192)
    return v / np.linalg.norm(v)


class FakeTranscriber:
    description, writer = "whisper-small (MLX, GPU)", None
    pending = dropped = done = failed = 0
    last_audio_s = last_processing_s = 0.0

    def __init__(self):
        self.lines = []

    def drain_lines(self):
        lines, self.lines = self.lines, []
        return lines


def line(start, speaker, text, garbled=False):
    return TranscriptLine(start=start, end=start + 2, speaker_id=1, speaker=speaker, text=text, language="id",
                          lang_probs=None, avg_logprob=-0.3, no_speech_prob=0.0, compression_ratio=1.2,
                          audio_s=2.3, queue_s=0.0, processing_s=0.2, model="m", garbled=garbled)


def fake_session(transcriber=None):
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
                                     output_name="Mewo", samplerate=44100, blocksize=1411,
                                     stats=AudioStats(), input_backlog=lambda: 0,
                                     out_ring=types.SimpleNamespace(available=lambda: 2822)),
        pipeline=types.SimpleNamespace(delay_samples=11025, drop_events=0, mic_silent=False, in_db=-40.0,
                                       ctx=BlockContext(0, 0.0, 44100, vad_prob=0.9, is_speech=True),
                                       clips={}, errors=0, last_error=None, denoise_scope="playback"),
        started=time.monotonic(),
        metrics=types.SimpleNamespace(snapshot=lambda: snap),
        gate=None, speaker_gain=None, tracker=tracker,
        analyzer=types.SimpleNamespace(dropped=0, last_ms=9.4, embedded=6, skipped_silent=0),
        embed_device="MPS", save_profiles=saved.append, log_metrics=lambda snap: None,
        transcriber=transcriber, asr_error=None,
        segmenter=types.SimpleNamespace(recording_s=0.0, too_short=0, other_speaker=0) if transcriber else None)
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
            if debug.display:
                await pilot.pause(1.2)  # a metrics tick: the health line renders
                self.assertIn("up 0:", app.health)

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

    async def test_transcript_panel_shows_lines_and_t_toggles_it(self):
        transcriber = FakeTranscriber()
        session, _ = fake_session(transcriber)
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            panel = app.query_one("#transcript")
            self.assertTrue(panel.display)
            transcriber.lines = [line(65, "Speaker 1", "kita pakai loop untuk iterate array"),
                                 line(70, "Speaker 1", "destand-destand", garbled=True)]
            transcriber.pending = 1
            await pilot.pause(0.3)
            text = "\n".join(strip.text for strip in panel.lines)
            self.assertIn("00:01:05 Speaker 1: kita pakai loop untuk iterate array", text)
            self.assertIn("(unclear)", text)
            self.assertNotIn("destand", text)
            self.assertIn("transcribing", str(app.query_one("#asr-status").render()))

            await pilot.press("t")
            self.assertFalse(panel.display)
            transcriber.lines = [line(80, "Speaker 1", "while hidden")]
            await pilot.pause(0.3)
            await pilot.press("t")  # lines that arrived while hidden are there
            self.assertIn("while hidden", "\n".join(strip.text for strip in panel.lines))

    async def test_t_without_transcription_warns(self):
        session, _ = fake_session()
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            self.assertFalse(app.query_one("#transcript").display)
            await pilot.press("t")
            self.assertFalse(app.query_one("#transcript").display)

    async def test_fatal_audio_error_exits_with_code_2(self):
        session, _ = fake_session()
        app = FocusEarApp(session)
        async with app.run_test() as pilot:
            session.engine.error = RuntimeError("CoreAudio stopped responding")
            await pilot.pause(0.3)
        self.assertEqual(app.return_code, 2)


if __name__ == "__main__":
    unittest.main()
