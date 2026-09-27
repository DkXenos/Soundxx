"""Run with: venv/bin/python -m unittest discover -s tests"""
import contextlib
import io
import json
import logging
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

from fixtures import SPEECH_SPANS, SPEECH_WAV
from focus_ear import asr
from focus_ear.asr import AsrResult, faster_whisper_size
from focus_ear.pipeline import BlockContext
from focus_ear.transcript import (FRAME_S, AsrUnavailable, Transcriber, TranscriptWriter, Utterance,
                                  UtteranceSegmenter)


def seconds(n_frames):
    return n_frames * FRAME_S


class Feed:
    """Drives a segmenter frame by frame. Frame i is filled with the value i, so tests can see which
    frames an utterance contains."""

    def __init__(self, **kwargs):
        self.out = []
        self.selected = None
        self.seg = UtteranceSegmenter(self.out.append, selected=lambda: self.selected, **kwargs)
        self.i = 0

    def run(self, n, speech=True, speaker=None):
        for _ in range(n):
            ctx = BlockContext(index=self.i, adc_time=self.i * FRAME_S, samplerate=16000,
                               is_speech=speech, speaker_id=speaker)
            self.seg.analyze(np.full(512, float(self.i), np.float32), ctx)
            self.i += 1
        return self

    @staticmethod
    def frames(utt):
        return sorted(set(utt.audio[::512].astype(int)))


class SegmenterTest(unittest.TestCase):
    def test_gap_closes_an_utterance_with_preroll_and_a_short_tail(self):
        f = Feed().run(50, speech=False).run(100).run(24, speech=False)
        self.assertEqual(f.out, [])  # 24 frames = 768 ms < 800 ms gap: still open
        f.run(1, speech=False)
        (utt,) = f.out
        frames = Feed.frames(utt)
        self.assertEqual(frames[0], 40)  # 10 frames (320 ms) of pre-roll before onset at frame 50
        self.assertEqual(frames[-1], 149 + 6)  # ~200 ms kept after the last speech frame
        self.assertAlmostEqual(utt.start, seconds(50))
        self.assertAlmostEqual(utt.end, seconds(150))

    def test_pause_shorter_than_the_gap_keeps_one_utterance(self):
        f = Feed().run(5, speech=False).run(40).run(20, speech=False).run(40).run(30, speech=False)
        self.assertEqual(len(f.out), 1)
        self.assertEqual(len(Feed.frames(f.out[0])), 5 + 40 + 20 + 40 + 6)

    def test_under_400_ms_of_speech_is_discarded(self):
        f = Feed().run(20, speech=False).run(12).run(30, speech=False)  # 384 ms
        self.assertEqual((f.out, f.seg.too_short), ([], 1))
        f.run(13).run(30, speech=False)  # 416 ms
        self.assertEqual(len(f.out), 1)

    def test_length_cap_cuts_at_the_latest_pause_without_losing_audio(self):
        cap = int(20 / FRAME_S)  # 625 frames
        f = Feed().run(500).run(3, speech=False).run(400).run(30, speech=False)
        self.assertEqual(len(f.out), 2)
        first, second = map(Feed.frames, f.out)
        self.assertLess(len(first), cap)
        self.assertEqual(first[-1], 501)  # cut at the pause (frames 500-502), not mid-word at the cap
        self.assertEqual(second[0], 502)
        self.assertEqual(first + second, list(range(0, 903 + 6)))  # nothing lost, nothing twice

    def test_length_cap_without_a_pause_is_a_hard_cut(self):
        f = Feed().run(1300).run(30, speech=False)
        lengths = [len(Feed.frames(u)) for u in f.out]
        self.assertEqual(lengths[:2], [625, 625])
        self.assertEqual(sum(lengths), 1300 + 6)

    def test_only_the_selected_speaker_is_kept(self):
        f = Feed()
        f.selected = 1
        f.run(30, speaker=None).run(60, speaker=2).run(30, speech=False)  # student
        f.run(30, speaker=None).run(60, speaker=1).run(30, speech=False)  # lecturer
        (utt,) = f.out
        self.assertEqual(utt.speaker_id, 1)
        self.assertEqual(f.seg.other_speaker, 1)
        # Identification lagged the lecturer's onset: those frames are still in the utterance.
        self.assertAlmostEqual(utt.start, seconds(120))

    def test_unidentified_utterance_belongs_to_the_previous_speaker(self):
        f = Feed()
        f.selected = 1
        f.run(60, speaker=1).run(30, speech=False)
        f.run(20, speaker=None).run(30, speech=False)  # "Ya." - too short to be identified
        self.assertEqual([u.speaker_id for u in f.out], [1, 1])

    def test_everyone_mode_keeps_all_speakers(self):
        f = Feed().run(60, speaker=2).run(30, speech=False).run(60, speaker=1).run(30, speech=False)
        self.assertEqual([u.speaker_id for u in f.out], [2, 1])

    def test_speaker_change_without_a_pause_splits_near_the_change(self):
        f = Feed().run(100, speaker=1).run(4, speech=False).run(24, speaker=1)  # B starts at frame 104
        f.run(40, speaker=2).run(30, speech=False)  # ...identified 24 frames (~0.75 s) late
        a, b = f.out
        self.assertEqual((a.speaker_id, b.speaker_id), (1, 2))
        self.assertEqual(Feed.frames(a)[-1], 102)  # split at the breath before B
        self.assertEqual(Feed.frames(b)[0], 103)

    def test_one_misidentified_window_does_not_split(self):
        f = Feed().run(100, speaker=1).run(8, speaker=2).run(100, speaker=1).run(30, speech=False)
        (utt,) = f.out
        self.assertEqual(utt.speaker_id, 1)

    def test_close_flushes_the_open_utterance(self):
        f = Feed().run(40)
        f.seg.close()
        self.assertEqual(len(f.out), 1)
        self.assertEqual(f.seg.recording_s, 0.0)


# --------------------------------------------------------------------------

class FakeBackend:
    description = "fake"

    def __init__(self, delay=0.0, text="halo"):
        self.delay, self.text = delay, text
        self.seen = []
        self.thread = None

    def transcribe(self, audio, language, prompt, detect=False):
        self.thread = threading.current_thread().name
        self.seen.append(len(audio))
        time.sleep(self.delay)
        return AsrResult(self.text, language or "id", -0.25, 0.01, 1.2, {"id": 0.9} if detect else None)


def utt(n, speaker=1):
    return Utterance(audio=np.zeros(n, np.float32), start=1.0, end=2.0, speaker_id=speaker)


class TranscriberTest(unittest.TestCase):
    def test_full_queue_drops_the_oldest_and_submit_never_waits(self):
        backend = FakeBackend(delay=0.2)
        tr = Transcriber("m", queue_size=2, load=lambda model: backend)
        tr.submit(utt(1))
        while not backend.seen:  # the first is being transcribed
            time.sleep(0.005)
        t = time.perf_counter()
        for n in range(2, 7):
            tr.submit(utt(n))
        self.assertLess(time.perf_counter() - t, 0.05)
        self.assertTrue(tr.close(timeout=5))
        # The first was already being transcribed; of the rest only the newest two survived.
        self.assertEqual(backend.seen, [1, 5, 6])
        self.assertEqual((tr.dropped, tr.done), (3, 3))
        self.assertEqual(backend.thread, "asr")

    def test_file_mode_waits_instead_of_dropping(self):
        backend = FakeBackend(delay=0.02)
        tr = Transcriber("m", queue_size=1, drop_oldest=False, load=lambda model: backend)
        for n in range(1, 7):
            tr.submit(utt(n))
        tr.close()
        self.assertEqual(backend.seen, [1, 2, 3, 4, 5, 6])
        self.assertEqual(tr.dropped, 0)

    def test_lines_carry_labels_and_timings(self):
        tr = Transcriber("m", language=None, detect_language=True, load=lambda model: FakeBackend(),
                         label_of=lambda sid: f"Prof {sid}")
        tr.submit(utt(16000, speaker=3))
        tr.close()
        (line,) = tr.drain_lines()
        self.assertEqual((line.speaker, line.text, line.language), ("Prof 3", "halo", "id"))
        self.assertEqual(line.lang_probs, {"id": 0.9})
        self.assertAlmostEqual(line.audio_s, 1.0)
        self.assertEqual(tr.drain_lines(), [])

    def test_no_backend_raises(self):
        def fail(model):
            raise RuntimeError("no model")
        with self.assertRaisesRegex(AsrUnavailable, "no model"):
            Transcriber("m", load=fail)

    def test_a_failing_utterance_does_not_stop_the_thread(self):
        backend = FakeBackend()
        calls = []

        def flaky(audio, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return FakeBackend.transcribe(backend, audio, *args, **kwargs)
        backend.transcribe = flaky
        tr = Transcriber("m", load=lambda model: backend)
        with self.assertLogs("focus_ear.transcript", logging.ERROR):
            tr.submit(utt(1))
            tr.submit(utt(2))
            tr.close()
        self.assertEqual((tr.failed, tr.done), (1, 1))


class WriterTest(unittest.TestCase):
    def test_markdown_and_jsonl_are_written_as_they_happen(self):
        with tempfile.TemporaryDirectory() as tmp:
            started = datetime(2026, 9, 26, 9, 14)
            tr = Transcriber("m", load=lambda model: FakeBackend(text="kita pakai loop"),
                             label_of=lambda sid: {1: "Prof. Lee", 2: "Speaker 2"}[sid])
            tr.writer = w = TranscriptWriter(Path(tmp), started, "live, MacBook Pro Microphone", "whisper-small", "id")
            self.assertEqual(w.md_path.name, "2026-09-26_0914.md")
            tr.submit(utt(100, speaker=1))
            deadline = time.monotonic() + 5
            while tr.done < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            # Already on disk before the session ends.
            self.assertIn("**[00:00:01] Prof. Lee:** kita pakai loop", w.md_path.read_text())
            self.assertIn("(listed when the session ends)", w.md_path.read_text())

            tr.backend.text = ""  # Whisper heard no words: JSONL only
            tr.submit(utt(100, speaker=2))
            tr.close()
            md = w.md_path.read_text()
            self.assertTrue(md.startswith("# Lecture transcript: 2026-09-26 09:14\n"))
            self.assertIn("- **Speakers:** Prof. Lee\n", md)
            self.assertEqual(md.count("**[00:"), 1)
            records = [json.loads(x) for x in w.jsonl_path.read_text().splitlines()]
            self.assertEqual([r["text"] for r in records], ["kita pakai loop", ""])
            self.assertEqual(set(records[0]), {
                "start", "end", "speaker_id", "speaker", "text", "language", "lang_probs", "avg_logprob",
                "no_speech_prob", "compression_ratio", "audio_s", "queue_s", "processing_s", "model", "garbled"})

            again = TranscriptWriter(Path(tmp), started, "x", "m", "id")  # same minute: no overwrite
            self.assertEqual(again.md_path.name, "2026-09-26_0914_2.md")
            again.close()

    def test_repetition_loops_show_as_unclear(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend(text="destand-" * 40)
            backend.transcribe = lambda *a, **k: AsrResult("destand-" * 40, "id", -0.2, 0.05, 17.8)
            tr = Transcriber("m", load=lambda model: backend)
            tr.writer = w = TranscriptWriter(Path(tmp), datetime(2026, 9, 26, 9, 14), "x", "m", "id")
            tr.submit(utt(100))
            tr.close()
            self.assertIn("*(unclear)*", w.md_path.read_text())
            self.assertTrue(json.loads(w.jsonl_path.read_text())["garbled"])


class BackendTest(unittest.TestCase):
    def test_faster_whisper_names(self):
        self.assertEqual(faster_whisper_size("mlx-community/whisper-small-mlx"), "small")
        self.assertEqual(faster_whisper_size("mlx-community/whisper-medium-mlx-8bit"), "medium")
        self.assertEqual(faster_whisper_size("mlx-community/whisper-large-v3-turbo"), "large-v3-turbo")
        self.assertEqual(faster_whisper_size("mlx-community/whisper-large-v3-mlx"), "large-v3")

    def test_falls_back_to_faster_whisper_and_says_so(self):
        with mock.patch.object(asr, "MlxWhisper", side_effect=RuntimeError("Metal unavailable")), \
                mock.patch.object(asr, "FasterWhisper", side_effect=lambda size: f"fw-{size}"), \
                self.assertLogs("focus_ear.asr", logging.WARNING) as logs:
            self.assertEqual(asr.load_backend("mlx-community/whisper-small-mlx"), "fw-small")
        self.assertIn("DOWNGRADE", "\n".join(logs.output))


class ShutdownTest(unittest.TestCase):
    """Ctrl-C must always leave a complete transcript, even mid-utterance or pressed twice."""

    def test_quitting_mid_utterance_still_transcribes_it(self):
        import types

        from focus_ear.config import Config
        from focus_ear.main import finish_transcription
        from focus_ear.pipeline import Pipeline

        class AlwaysSpeech:
            name = "vad"

            def analyze(self, frame, ctx):
                ctx.is_speech = True

            def reset(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            tr = Transcriber("m", load=lambda model: FakeBackend(text="jadi kompleksitasnya O n"))
            tr.writer = w = TranscriptWriter(Path(tmp), datetime(2026, 9, 26, 9, 14), "x", "m", "id")
            seg = UtteranceSegmenter(tr.submit)
            pipe = Pipeline(types.SimpleNamespace(samplerate=16000, blocksize=512), Config(),
                            analyzers=[AlwaysSpeech(), seg])
            for i in range(60):  # 2 s of speech, still going: nothing has been submitted yet
                pipe.process_block(np.full(512, 0.1, np.float32), adc_time=i * 0.032)
            self.assertEqual(tr.done, 0)
            pipe.stop()  # what quitting does: closes the analyzers, which flushes the open utterance
            finish_transcription(tr, timeout=5)
            self.assertIn("jadi kompleksitasnya O n", w.md_path.read_text())
            self.assertIn("- **Speakers:**", w.md_path.read_text())

    def test_second_ctrl_c_skips_waiting_but_still_closes_the_files(self):
        from focus_ear.main import Interrupts, finish_transcription

        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend(delay=0.3, text="halo")
            tr = Transcriber("m", queue_size=10, drop_oldest=False, load=lambda model: backend,
                             label_of=lambda sid: "Prof")
            tr.writer = w = TranscriptWriter(Path(tmp), datetime(2026, 9, 26, 9, 14), "x", "m", "id")
            for n in range(1, 8):
                tr.submit(utt(n))
            hurry = Interrupts()
            hurry.count = 1  # Ctrl-C pressed again while waiting
            t = time.perf_counter()
            with self.assertLogs("focus_ear", logging.WARNING):
                finish_transcription(tr, timeout=60, interrupts=hurry)
            self.assertLess(time.perf_counter() - t, 2.0)
            self.assertLess(tr.done, 7)
            md = w.md_path.read_text()
            self.assertNotIn("(listed when the session ends)", md)  # closed properly, heading complete
            self.assertEqual(md.count("**[00:"), w.lines)

    def test_interrupts_are_counted_not_raised(self):
        import os
        import signal

        from focus_ear.main import Interrupts

        with Interrupts() as stop:
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        self.assertEqual(stop.count, 2)
        with self.assertRaises(KeyboardInterrupt):  # the previous handler is back
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)


def _mlx_model_cached(repo="mlx-community/whisper-small-mlx"):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_mlx_model_cached(), "whisper-small-mlx not downloaded")
class RealModelTest(unittest.TestCase):
    def test_file_mode_transcribes_the_speech_and_writes_both_files(self):
        from focus_ear import main
        # A real stream, not a MagicMock: libraries that log to sys.stdout (loguru) would
        # otherwise treat the mock as a file name and create it.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(main, "SESSIONS_DIR", Path(tmp)), \
                contextlib.redirect_stdout(io.StringIO()):
            logging.getLogger("focus_ear").setLevel(logging.WARNING)
            code = main.run_file(main.Config(speakers=False, language="auto"), str(SPEECH_WAV))
            self.assertEqual(code, 0)
            (jsonl,) = Path(tmp).glob("*_speech_16k.jsonl")
            records = [json.loads(x) for x in jsonl.read_text().splitlines()]
            # One utterance per speech span of the fixture, at the right times.
            self.assertEqual(len(records), len(SPEECH_SPANS))
            for r, (a, b) in zip(records, SPEECH_SPANS):
                self.assertAlmostEqual(r["start"], a, delta=0.3)
                self.assertAlmostEqual(r["end"], b, delta=0.3)
                self.assertTrue(r["text"])
                self.assertGreater(r["avg_logprob"], -1.0)
            self.assertTrue(jsonl.with_suffix(".md").read_text().count("**[00:00:0"), len(SPEECH_SPANS))


if __name__ == "__main__":
    unittest.main()
