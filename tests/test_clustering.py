"""Run with: venv/bin/python -m unittest discover -s tests"""
import unittest
from pathlib import Path

import numpy as np

from focus_ear.clustering import OnlineClusterer, SpeakerTracker

# Real ECAPA voiceprints (1.5 s windows) of four macOS voices, clean and with
# simulated lecture-hall reverb and noise. 60 consecutive windows each.
ECAPA = np.load(Path(__file__).parent / "data" / "ecapa_embeddings.npz")
VOICES = ["Daniel", "Samantha", "Rishi", "Moira"]
LECTURE = [(0, 20), (1, 10), (0, 15), (2, 8), (0, 10), (3, 8), (1, 8), (0, 10)]  # (voice, windows)


def voice(seed: int, dim: int = 192) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, noise: float, rng) -> np.ndarray:
    return v + noise * rng.standard_normal(v.shape) / np.sqrt(v.size)


def run_lecture(c: OnlineClusterer, cond: str) -> list[tuple[int, int | None]]:
    pos = [0] * 4
    out = []
    for t, spk in enumerate(s for s, n in LECTURE for _ in range(n)):
        e = ECAPA[f"{cond}_{VOICES[spk]}"][pos[spk]].astype(np.float32)
        pos[spk] += 1
        out.append((spk, c.assign(e, now=t * 0.25).speaker_id))
    return out


class OnlineClustererTest(unittest.TestCase):
    def test_similar_joins_dissimilar_founds_a_new_speaker(self):
        c = OnlineClusterer(threshold=0.5)
        a, b = voice(1), voice(2)
        rng = np.random.default_rng(0)
        ids = [c.assign(near(v, 0.3, rng), now=i).speaker_id for i, v in enumerate([a, a, b, a, b])]
        self.assertEqual(ids, [1, 1, 2, 1, 2])
        self.assertTrue(all(np.isclose(np.linalg.norm(s.centroid), 1) for s in c.speakers.values()))

    def test_running_mean_then_exponential_average(self):
        c = OnlineClusterer(threshold=-1.0, alpha=0.1)  # everything joins speaker 1
        a, b = voice(1), voice(2)
        c.assign(a, 0)
        c.assign(b, 1)  # second embedding weighs 1/2, not alpha
        np.testing.assert_allclose(c.speakers[1].centroid, (a + b) / np.linalg.norm(a + b), atol=1e-6)

    def test_speakers_are_confirmed_after_min_count_sightings(self):
        c = OnlineClusterer(threshold=0.5, min_count=3)
        a = voice(1)
        c.assign(a, 0)
        c.assign(a, 1)
        self.assertFalse(c.confirmed(c.speakers[1]))
        c.assign(a, 2)
        self.assertTrue(c.confirmed(c.speakers[1]))

    def test_converging_clusters_merge_into_the_protected_one(self):
        c = OnlineClusterer(threshold=0.5, merge_threshold=0.9)
        a = voice(1)
        c.assign(a, 0)
        c.assign(voice(2), 1)
        for _ in range(2):
            c.assign(a, 2)  # speaker 1 is confirmed, speaker 2 isn't
        # Speaker 2's voiceprint has since drifted onto speaker 1's (one voice, split early on).
        c.speakers[2].centroid = near(a, 0.1, np.random.default_rng(0)).astype(np.float32)
        c.speakers[2].centroid /= np.linalg.norm(c.speakers[2].centroid)
        result = c.assign(a, 3, protected=frozenset({2}))  # 2 is selected, so 2 survives
        self.assertEqual(result.merged, [(1, 2)])
        self.assertEqual(result.speaker_id, 2)
        self.assertEqual(list(c.speakers), [2])
        self.assertEqual(c.speakers[2].count, 5)

    def test_cap_evicts_unconfirmed_first_never_protected_or_enrolled(self):
        c = OnlineClusterer(threshold=0.9, max_speakers=3, min_count=2)
        c.add_enrolled("Prof", voice(0))              # id 1
        for _ in range(2):
            c.assign(voice(1), 0)                     # id 2, confirmed
        c.assign(voice(2), 1)                         # id 3, unconfirmed
        c.assign(voice(3), 2, protected=frozenset({2}))  # full: evicts 3
        self.assertEqual(sorted(c.speakers), [1, 2, 4])
        self.assertIsNone(c.assign(voice(4), 3, protected=frozenset({2, 4})).speaker_id)

    def test_prune_timeouts(self):
        c = OnlineClusterer(threshold=0.9, timeout_s=120, phantom_timeout_s=30, min_count=2)
        c.add_enrolled("Prof", voice(0))              # 1: enrolled, never pruned
        c.assign(voice(1), 0)
        c.assign(voice(1), 0)                         # 2: confirmed
        c.assign(voice(2), 0)                         # 3: unconfirmed
        c.assign(voice(3), 0)
        c.assign(voice(3), 0)                         # 4: confirmed but selected
        self.assertEqual(c.prune(now=60, protected=frozenset({4})), [3])
        self.assertEqual(c.prune(now=200, protected=frozenset({4})), [2])
        self.assertEqual(sorted(c.speakers), [1, 4])

    def test_reset_keeps_enrolled_speakers_with_their_saved_voiceprint(self):
        c = OnlineClusterer(threshold=-1.0)
        prof = c.add_enrolled("Prof", voice(0))
        c.assign(voice(5), 0)  # drags Prof's centroid
        self.assertLess(float(prof.centroid @ voice(0)), 0.999)
        c.reset()
        self.assertEqual(list(c.speakers), [prof.id])
        np.testing.assert_allclose(prof.centroid, voice(0), atol=1e-6)
        self.assertEqual(prof.count, 0)

    def test_real_voiceprints_in_a_reverberant_hall_give_one_cluster_per_person(self):
        assignments = run_lecture(OnlineClusterer(threshold=0.45, merge_threshold=0.55), "hall")
        lecturer_ids = {sid for spk, sid in assignments[20:] if spk == 0}
        self.assertEqual(lecturer_ids, {assignments[0][1]}, "the lecturer was split")
        c = OnlineClusterer(threshold=0.45, merge_threshold=0.55)
        run_lecture(c, "hall")
        self.assertEqual(sorted(s.count for s in c.speakers.values()), [8, 8, 18, 55])

    def test_the_original_065_threshold_fragments_the_same_hall_recording(self):
        # Documents why the default isn't 0.65: see README "Choosing the defaults".
        c = OnlineClusterer(threshold=0.65, merge_threshold=1.01)
        run_lecture(c, "hall")
        self.assertGreaterEqual(len(c.speakers), 7)


class SpeakerTrackerTest(unittest.TestCase):
    def make(self, **kw):
        c = OnlineClusterer(threshold=0.5, min_count=3, **kw)
        return SpeakerTracker(c, boost_gain=1.0, attenuation_gain=0.1), c

    def confirm(self, tracker, v, t0=0.0):
        ids = [tracker.observe(v, t0 + i) for i in range(3)]
        return ids[-1]

    def test_observe_reports_only_confirmed_speakers(self):
        tracker, _ = self.make()
        self.assertIsNone(tracker.observe(voice(1), 0))
        self.assertIsNone(tracker.observe(voice(1), 1))
        self.assertEqual(tracker.observe(voice(1), 2), 1)

    def test_selection_and_gain_targets(self):
        tracker, _ = self.make()
        a = self.confirm(tracker, voice(1))
        b = self.confirm(tracker, voice(2), 10)
        self.assertEqual(tracker.gain_target(b), 1.0)      # passthrough: everyone at unity
        self.assertEqual(tracker.select(1), "Focusing on Speaker 1")
        self.assertEqual(tracker.gain_target(a), 1.0)
        self.assertEqual(tracker.gain_target(b), 0.1)
        self.assertEqual(tracker.gain_target(None), 1.0)  # not identified yet: full volume
        self.assertEqual(tracker.select(7), "No speaker 7")
        tracker.select(0)
        self.assertEqual(tracker.gain_target(b), 1.0)

    def test_snapshot_hides_unconfirmed_and_marks_talking(self):
        tracker, _ = self.make()
        self.confirm(tracker, voice(1))
        tracker.observe(voice(2), 5)  # one sighting only
        tracker.update_level(-30.0, True, 0.032)
        snap = tracker.snapshot(now=2.5)
        self.assertEqual([v.label for v in snap.speakers], ["Speaker 1"])
        self.assertEqual(snap.unconfirmed, 1)
        self.assertTrue(snap.speakers[0].talking)
        self.assertAlmostEqual(snap.speakers[0].level_db, -30.0)
        self.assertEqual(snap.last_similarities[0][0], "Speaker 1")

    def test_enrolled_speaker_is_auto_selected_unless_the_user_chose(self):
        tracker, c = self.make()
        c.add_enrolled("Prof", voice(1))
        self.confirm(tracker, voice(1))
        self.assertEqual(tracker.snapshot().selected_label, "Prof")
        self.assertIn("Recognised Prof", tracker.drain_events()[0])

        tracker, c = self.make()
        c.add_enrolled("Prof", voice(1))
        tracker.select(0)  # user asked for passthrough: respect it
        self.confirm(tracker, voice(1))
        self.assertIsNone(tracker.snapshot().selected_label)

    def test_enroll_selected_names_them_and_returns_profiles_to_save(self):
        tracker, c = self.make()
        self.assertIsNone(tracker.enroll_selected("Prof"))  # nobody selected
        self.confirm(tracker, voice(1))
        tracker.select(1)
        profiles = tracker.enroll_selected("  Prof  ")
        self.assertEqual([name for name, _ in profiles], ["Prof"])
        np.testing.assert_allclose(profiles[0][1], c.speakers[1].centroid)
        self.assertEqual(tracker.snapshot().speakers[0].label, "Prof")


if __name__ == "__main__":
    unittest.main()
