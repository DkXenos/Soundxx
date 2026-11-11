"""Run with: venv/bin/python -m unittest discover -s tests"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from focus_ear import settings as settings_mod
from focus_ear.sessions import SessionError, SessionStore, read_jsonl, repair_wav_header, slugify, wav_seconds


def write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), np.int16).tobytes())


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SessionStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_names_the_folder_after_time_and_title_and_never_reuses_one(self):
        t = datetime(2026, 11, 12, 9, 0)
        a = self.store.create(title="Machine Learning (Kelas B)", course="ML", started=t)
        b = self.store.create(title="Machine Learning (Kelas B)", course="ML", started=t)
        self.assertEqual(a["id"], "2026-11-12_0900_machine-learning-kelas-b")
        self.assertEqual(b["id"], "2026-11-12_0900_machine-learning-kelas-b_2")
        self.assertEqual(self.store.load(a["id"])["status"], "recording")
        self.assertEqual(self.store.load(a["id"])["course"], "ML")

    def test_ids_cannot_escape_the_sessions_folder(self):
        for bad in ("../x", "2026-11-12_0900/../../etc", "", "foo"):
            with self.assertRaises(SessionError):
                self.store.path(bad)

    def test_update_is_atomic_and_rejects_unknown_status(self):
        sid = self.store.create(title="x")["id"]
        self.store.update(sid, status="recorded", duration_s=12.5)
        self.assertEqual(self.store.load(sid)["duration_s"], 12.5)
        with self.assertRaises(SessionError):
            self.store.update(sid, status="exploded")
        leftovers = [p.name for p in self.store.path(sid).iterdir() if p.name != "session.json"]
        self.assertEqual(leftovers, [])  # no temp files left behind

    def test_list_is_newest_first_with_derived_flags(self):
        a = self.store.create(title="old", started=datetime(2026, 1, 1, 8, 0), status="ready")
        b = self.store.create(title="new", started=datetime(2026, 2, 1, 8, 0), status="ready")
        write_wav(self.store.path(b["id"]) / "audio.wav", 1.0)
        listed = self.store.list()
        self.assertEqual([x["id"] for x in listed], [b["id"], a["id"]])
        self.assertTrue(listed[0]["has_audio"])
        self.assertFalse(listed[1]["has_audio"])

    def test_a_recording_session_whose_engine_died_is_interrupted_and_recovers(self):
        live = self.store.create(title="live", engine_pid=os.getpid())   # this process: alive
        dead = self.store.create(title="dead", engine_pid=dead_pid())
        audio = self.store.path(dead["id"]) / "audio.wav"
        write_wav(audio, 3.0)
        with open(audio, "ab") as f:  # what a crash leaves: data written after the header's lengths
            f.write(np.zeros(16000, np.int16).tobytes())
        self.assertEqual(self.store.interrupted(), [dead["id"]])
        data = self.store.recover(dead["id"])
        self.assertEqual(data["status"], "recorded")
        self.assertTrue(data["interrupted"])
        self.assertAlmostEqual(data["duration_s"], 4.0, places=1)
        with wave.open(str(audio)) as w:
            self.assertEqual(w.getnframes(), 4 * 16000)  # header repaired
        self.assertEqual(self.store.interrupted(), [])
        self.assertEqual(self.store.recover(live["id"])["status"], "recording")  # alive: untouched

    def test_flat_files_of_earlier_versions_are_imported_once(self):
        (self.root / "2026-09-26_1411.md").write_text("# t\n")
        (self.root / "2026-09-26_1411.jsonl").write_text(json.dumps({"start": 1, "end": 7.5, "text": "x"}) + "\n")
        write_wav(self.root / "2026-09-29_0800.wav", 2.0)
        (self.root / "notes.txt").write_text("not a session")
        ids = self.store.import_flat()
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.store.import_flat(), [])
        by = {self.store.load(i)["imported_from"]: self.store.load(i) for i in ids}
        self.assertEqual(by["2026-09-26_1411"]["status"], "ready")
        self.assertEqual(by["2026-09-26_1411"]["duration_s"], 7.5)
        self.assertTrue((self.store.path(by["2026-09-26_1411"]["id"]) / "live.jsonl").exists())
        self.assertEqual(by["2026-09-29_0800"]["status"], "recorded")
        self.assertTrue((self.root / "notes.txt").exists())

    def test_read_jsonl_skips_a_line_cut_off_by_a_crash(self):
        p = self.root / "x.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "te')
        self.assertEqual([r["a"] for r in read_jsonl(p)], [1, 2])

    def test_slugify_and_wav_seconds(self):
        self.assertEqual(slugify("Struktur Data — Pertemuan 3!"), "struktur-data-pertemuan-3")
        p = self.root / "a.wav"
        write_wav(p, 1.5)
        self.assertAlmostEqual(wav_seconds(p), 1.5)
        repair_wav_header(p)  # already right: unchanged
        self.assertAlmostEqual(wav_seconds(p), 1.5)


class SettingsTest(unittest.TestCase):
    def test_defaults_then_remembered_choices_and_recent_courses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            self.assertEqual(settings_mod.load_settings(path)["output_device"], "Mewo")
            settings_mod.update_settings(path, output_device="ZoomAudioDevice", course="ML")
            settings_mod.update_settings(path, course="Struktur Data")
            settings_mod.update_settings(path, course="ML", bogus=1)
            s = settings_mod.load_settings(path)
            self.assertEqual(s["output_device"], "ZoomAudioDevice")
            self.assertEqual(s["courses"], ["ML", "Struktur Data"])
            self.assertNotIn("bogus", s)
            path.write_text("{broken")
            self.assertEqual(settings_mod.load_settings(path)["output_device"], "Mewo")


if __name__ == "__main__":
    unittest.main()
