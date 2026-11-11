"""The app's server, the review document and notes parsing. Run with: venv/bin/python -m unittest discover -s tests"""
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from focus_ear import notes, review
from focus_ear.sessions import SessionStore

FINAL = {
    "speakers": {"S1": {"label": "Dosen", "role": "lecturer", "profile": None},
                 "S2": {"label": "Mahasiswa 1", "role": "student", "profile": None},
                 "S3": {"label": "Mahasiswa 2", "role": "student", "profile": None}},
    "lines": [{"id": "L1", "start": 1.0, "end": 4.0, "speaker": "S1", "text": "Selamat pagi.", "flagged": False},
              {"id": "L2", "start": 5.0, "end": 7.0, "speaker": "S2", "text": "Pak, bedanya apa?", "flagged": True},
              {"id": "L3", "start": 8.0, "end": 12.0, "speaker": "S1", "text": "Linked list itu node.", "flagged": False},
              {"id": "L4", "start": 13.0, "end": 14.0, "speaker": "S3", "text": "Is it linear?", "flagged": False}],
}


def make_session(store: SessionStore, seconds: float = 15.0) -> str:
    sid = store.create(title="Struktur Data", course="SD", status="ready", duration_s=seconds)["id"]
    folder = store.path(sid)
    with wave.open(str(folder / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(seconds * 16000), np.int16).tobytes())
    (folder / "final.json").write_text(json.dumps(FINAL))
    rng = np.random.default_rng(0)
    np.savez(folder / "speakers.npz", ids=np.array(["S1", "S2", "S3"]), labels=np.array(["Dosen", "M1", "M2"]),
             centroids=rng.standard_normal((3, 192)).astype(np.float32))
    return sid


class ReviewDocTest(unittest.TestCase):
    def test_every_edit_and_its_undo(self):
        doc = review.doc_from_final(FINAL)
        original = json.loads(json.dumps(doc))
        ops = [{"op": "rename_speaker", "speaker": "S1", "label": "Pak Budi"},
               {"op": "move_line", "line": "L2", "speaker": "S3"},
               {"op": "edit_text", "line": "L3", "text": "Linked list itu terdiri dari node."},
               {"op": "merge_speakers", "source": "S3", "target": "S2"},
               {"op": "move_line", "line": "L4", "speaker": "new"}]
        for op in ops:
            review.apply(doc, op)
        self.assertEqual(doc["speakers"]["S1"]["label"], "Pak Budi")
        self.assertNotIn("S3", doc["speakers"])
        self.assertEqual([ln["speaker"] for ln in doc["lines"]], ["S1", "S2", "S1", "S4"])  # S3 is never reused
        self.assertEqual(doc["speakers"]["S4"]["label"], "Mahasiswa 2")
        self.assertTrue(doc["lines"][2]["edited"])
        for _ in ops:
            self.assertIsNotNone(review.undo(doc))
        self.assertIsNone(review.undo(doc))
        doc.pop("history"), original.pop("history"), doc.pop("next_speaker")
        self.assertEqual(doc, original)

    def test_bad_edits_are_refused(self):
        doc = review.doc_from_final(FINAL)
        for op in ({"op": "rename_speaker", "speaker": "S1", "label": "  "}, {"op": "merge_speakers", "source": "S1", "target": "S1"},
                   {"op": "edit_text", "line": "L99", "text": "x"}, {"op": "explode"}):
            with self.assertRaises(review.EditError):
                review.apply(doc, op)
        self.assertEqual(doc["history"], [])

    def test_live_transcript_becomes_a_document_and_markdown(self):
        records = [{"start": 0.5, "end": 2, "speaker_id": 2, "speaker": "Speaker 2", "text": "halo", "garbled": False},
                   {"start": 3, "end": 4, "speaker_id": None, "speaker": None, "text": "", "garbled": False},
                   {"start": 5, "end": 6, "speaker_id": None, "speaker": None, "text": "aaa aaa", "garbled": True}]
        doc = review.doc_from_live(records)
        self.assertEqual([ln["speaker"] for ln in doc["lines"]], ["S2", "S0"])
        self.assertEqual(doc["lines"][1]["text"], "(unclear)")
        md = review.render_markdown(doc, {"title": "T", "started": "2026-11-12T09:00:00+07:00"})
        self.assertIn("**[00:00:00] Speaker 2:** halo", md)
        self.assertIn("live (not processed yet)", md)


class NotesParseTest(unittest.TestCase):
    LINES = review.text_only(review.doc_from_final(FINAL))

    def test_chapters_take_their_time_from_the_transcript_and_bad_ids_are_dropped(self):
        answer = "```json\n" + json.dumps({
            "summary": "Tentang linked list.", "key_points": ["a", ""],
            "chapters": [{"title": "Linked list", "line_id": "L3", "start": 999}, {"title": "Hallucinated", "line_id": "L77"},
                         {"title": "Pembuka", "line_id": "L1"}],
            "terms": [{"term": "node", "explanation": "x"}], "tasks": [],
            "questions": [{"line_id": "L2", "question": "bedanya?", "answer": "node"}]}) + "\n```"
        n = notes.parse_notes(answer, self.LINES)
        self.assertEqual([(c["title"], c["start"]) for c in n["chapters"]], [("Pembuka", 1.0), ("Linked list", 8.0)])
        self.assertEqual(n["key_points"], ["a"])
        self.assertEqual(n["questions"][0]["start"], 5.0)

    def test_unusable_answers_raise(self):
        for bad in ("no json here", "{broken", json.dumps({"summary": "", "chapters": []})):
            with self.assertRaises(notes.NotesError):
                notes.parse_notes(bad, self.LINES)

    def test_generate_offline_queues_and_online_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            sid = make_session(store)

            def offline(prompt, model):
                raise notes.Offline("no connection")
            self.assertEqual(notes.generate_for_session(store, sid), "skipped")  # 13 words: not worth a call
            with mock.patch.object(notes, "call_claude", offline):
                self.assertEqual(notes.generate_for_session(store, sid, force=True), "queued")
            seen = {}

            def fake(prompt, model):
                seen["prompt"] = prompt
                return json.dumps({"summary": "S", "chapters": [{"title": "A", "line_id": "L1"}]}), {"model": "m"}
            with mock.patch.object(notes, "call_claude", fake):
                self.assertEqual(notes.generate_for_session(store, sid, force=True), "ready")
            self.assertIn("[L3 00:00:08] Dosen: Linked list itu node.", seen["prompt"])
            self.assertNotIn("audio", seen["prompt"].lower())
            self.assertTrue((store.path(sid) / "notes.md").exists())
            self.assertEqual(store.load(sid)["notes_status"], "ready")


class ServerTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from focus_ear.app.server import create_app
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.tmp.name))
        self.sid = make_session(self.store)
        self.settings = mock.patch("focus_ear.app.server.load_settings",
                                   return_value={"output_device": "Mewo", "mode": "listen", "courses": ["SD"],
                                                 "auto_process": False, "auto_notes": False, "notes_model": None})
        self.settings.start()
        self.app = create_app(store=self.store, token="t0k", port=1234)
        self.client = TestClient(self.app).__enter__()
        self.h = {"X-Token": "t0k"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.settings.stop()
        self.tmp.cleanup()

    def test_token_and_host_are_required(self):
        self.assertEqual(self.client.get("/api/sessions").status_code, 401)
        self.assertEqual(self.client.get("/api/sessions", headers={"X-Token": "nope"}).status_code, 401)
        self.assertEqual(self.client.get("/api/sessions", headers={**self.h, "Host": "evil.example:1234"}).status_code, 403)
        self.assertEqual(self.client.get("/api/sessions", headers=self.h).status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)  # the page itself carries no data

    def test_list_detail_edit_undo_export(self):
        r = self.client.get("/api/sessions", headers=self.h).json()
        self.assertEqual([s["id"] for s in r["sessions"]], [self.sid])
        d = self.client.get(f"/api/sessions/{self.sid}", headers=self.h).json()
        self.assertEqual(d["transcript"]["source"], "final")
        self.assertEqual(len(d["transcript"]["lines"]), 4)
        self.assertFalse(d["transcript"]["can_undo"])
        ok = self.client.post(f"/api/sessions/{self.sid}/edit", headers=self.h,
                              json={"op": "rename_speaker", "speaker": "S1", "label": "Bu Rina"})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post(f"/api/sessions/{self.sid}/edit", headers=self.h, json={"op": "edit_text", "line": "L9"})
        self.assertEqual(bad.status_code, 400)
        md = self.client.get(f"/api/sessions/{self.sid}/export/transcript.md?token=t0k").text
        self.assertIn("Bu Rina:", md)
        self.client.post(f"/api/sessions/{self.sid}/undo", headers=self.h)
        self.assertIn("Dosen:", self.client.get(f"/api/sessions/{self.sid}/export/transcript.md?token=t0k").text)
        ref = self.client.get(f"/api/sessions/{self.sid}/export/reference.json?token=t0k").json()
        self.assertEqual(ref["lines"][0]["speaker"], "Dosen")
        self.client.patch(f"/api/sessions/{self.sid}", headers=self.h, json={"title": "Linked list", "bogus": 1})
        self.assertEqual(self.store.load(self.sid)["title"], "Linked list")

    def test_audio_supports_range_requests_for_seeking(self):
        r = self.client.get(f"/api/sessions/{self.sid}/audio?token=t0k", headers={"Range": "bytes=1000-1999"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(len(r.content), 1000)

    def test_remember_a_voice_saves_a_profile_and_marks_the_speaker(self):
        saved = {}
        with mock.patch("focus_ear.profiles.load_profiles", return_value=[("Old", np.ones(192, np.float32))]), \
                mock.patch("focus_ear.profiles.save_profiles", side_effect=lambda p: saved.setdefault("p", p)):
            r = self.client.post(f"/api/sessions/{self.sid}/remember", headers=self.h, json={"speaker": "S1", "name": "Pak Budi"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([n for n, _ in saved["p"]], ["Old", "Pak Budi"])
        self.assertAlmostEqual(float(np.linalg.norm(saved["p"][1][1])), 1.0, places=5)
        doc = self.client.get(f"/api/sessions/{self.sid}", headers=self.h).json()["transcript"]
        self.assertEqual(doc["speakers"]["S1"]["label"], "Pak Budi")
        self.assertTrue(doc["speakers"]["S1"]["remembered"])

    def test_live_commands_need_a_running_engine_and_refuse_loudspeakers(self):
        self.assertEqual(self.client.post("/api/live/command", headers=self.h, json={"cmd": "pause"}).status_code, 409)
        self.assertEqual(self.client.post("/api/live/command", headers=self.h, json={"cmd": "rm -rf"}).status_code, 400)
        r = self.client.post("/api/live/command", headers=self.h, json={"cmd": "set_output", "device": "MacBook Pro Speakers"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.post("/api/live/start", headers=self.h, json={}).status_code, 409)

    def test_preflight_and_recover(self):
        devs = [{"name": "MacBook Pro Microphone", "inputs": 1, "outputs": 0},
                {"name": "MacBook Pro Speakers", "inputs": 0, "outputs": 2}]
        with mock.patch("focus_ear.app.server.probe_devices", return_value=devs), \
                mock.patch("focus_ear.audio_io.lid_closed", return_value=False):
            pf = self.client.get("/api/preflight?output=Mewo", headers=self.h).json()
            out = self.client.get("/api/devices", headers=self.h).json()["outputs"]
        self.assertTrue(pf["can_start"])
        self.assertEqual({c["id"]: c["level"] for c in pf["checks"]}["output"], "warning")
        self.assertEqual(out, [{"name": "MacBook Pro Speakers", "loudspeaker": True}])
        sid = self.store.create(title="crashed", engine_pid=999999)["id"]
        r = self.client.post(f"/api/sessions/{sid}/recover", headers=self.h).json()
        self.assertTrue(r["interrupted"])

    def test_websocket_needs_the_token(self):
        from starlette.websockets import WebSocketDisconnect
        with self.client.websocket_connect("/ws?token=t0k") as ws:
            self.assertEqual(ws.receive_json()["type"], "hello")
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws?token=bad") as ws:
                ws.receive_json()


if __name__ == "__main__":
    unittest.main()
