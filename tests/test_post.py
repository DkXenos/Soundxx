"""Metrics on hand-made cases, and the after-class pass's logic without models.
Run with: venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from focus_ear.eval.metrics import cer, der, lines_to_turns, normalize, term_recall, wer
from focus_ear.post import diarize, final_asr, merge, regions


class MetricsTest(unittest.TestCase):
    def test_wer_counts_each_kind_of_error(self):
        r = wer("kita pakai linked list hari ini", "kita pakai link list hari")
        self.assertEqual((r.substitutions, r.deletions, r.insertions), (1, 1, 0))
        self.assertAlmostEqual(r.rate, 2 / 6)
        self.assertEqual(wer("A, b. C!", "a b c").rate, 0.0)  # case and punctuation don't count
        self.assertEqual(wer("a b", "a x b").insertions, 1)
        self.assertEqual(wer("", "").rate, 0.0)

    def test_cer_and_normalize(self):
        self.assertEqual(normalize("Value-nya O(1), 'kan?"), ["value-nya", "o", "1", "kan"])
        self.assertAlmostEqual(cer("array", "away").rate, 2 / 5)

    def test_term_recall_counts_occurrences(self):
        r = term_recall(["linked list", "array", "stack"], "linked list dan array, array lagi", "link list dan array")
        self.assertEqual((r["found"], r["total"]), (1, 3))
        self.assertEqual(sorted(r["missed"]), ["array", "linked list"])

    def test_der_maps_labels_and_scores_errors(self):
        ref = [{"start": 0, "end": 10, "speaker": "Dosen"}, {"start": 10, "end": 14, "speaker": "Mhs"}]
        same = [{"start": 0, "end": 10, "speaker": 7}, {"start": 10, "end": 14, "speaker": 3}]
        d = der(ref, same, collar=0)
        self.assertAlmostEqual(d["der"], 0.0)
        self.assertEqual(d["mapping"], {"7": "Dosen", "3": "Mhs"})
        split = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 5, "end": 10, "speaker": "B"},
                 {"start": 10, "end": 14, "speaker": "C"}]
        d = der(ref, split, collar=0)
        self.assertAlmostEqual(d["confusion"], 5 / 14, places=2)  # half the lecturer under a third label
        self.assertEqual(d["per_speaker"]["Dosen"]["split_into"], 2)
        self.assertAlmostEqual(d["per_speaker"]["Dosen"]["coverage"], 0.5, places=2)
        miss = [{"start": 0, "end": 10, "speaker": "A"}, {"start": 15, "end": 16, "speaker": "A"}]
        d = der(ref, miss, collar=0)
        self.assertAlmostEqual(d["missed"], 4 / 14, places=2)
        self.assertAlmostEqual(d["false_alarm"], 1 / 14, places=2)

    def test_collar_forgives_boundaries(self):
        ref = [{"start": 0, "end": 10, "speaker": "a"}, {"start": 10, "end": 20, "speaker": "b"}]
        hyp = [{"start": 0, "end": 10.2, "speaker": "x"}, {"start": 10.2, "end": 20, "speaker": "y"}]
        self.assertGreater(der(ref, hyp, collar=0)["der"], 0)
        self.assertAlmostEqual(der(ref, hyp, collar=0.25)["der"], 0.0)

    def test_lines_to_turns_joins_one_speakers_lines(self):
        t = lines_to_turns([{"start": 0, "end": 2, "speaker": "a"}, {"start": 2.3, "end": 4, "speaker": "a"},
                            {"start": 5, "end": 6, "speaker": "b"}])
        self.assertEqual([(x["start"], x["end"], x["speaker"]) for x in t], [(0, 4, "a"), (5, 6, "b")])


class RegionsTest(unittest.TestCase):
    def test_hysteresis_gap_closing_min_length_and_padding(self):
        f = regions.FRAME_S
        p = np.zeros(200, np.float32)
        p[10:40] = 0.9      # speech
        p[40:45] = 0.4      # dips, but above the exit threshold: still speech
        p[45:50] = 0.1      # 160 ms gap: closed (< 300 ms)
        p[50:80] = 0.8
        p[150:153] = 0.9    # 96 ms blip: dropped
        r = regions.speech_regions(p, threshold=0.5, min_speech_s=0.25, min_silence_s=0.3, pad_s=0.1)
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0][0], 10 * f - 0.1, places=2)
        self.assertAlmostEqual(r[0][1], 80 * f + 0.1, places=2)


class DiarizeLogicTest(unittest.TestCase):
    def test_windows_cover_regions(self):
        w = diarize.windows([(0.0, 0.3), (1.0, 2.0), (3.0, 6.0)])
        self.assertEqual(w[0], (1.0, 2.0))          # short region: one window of its own
        self.assertNotIn((0.0, 0.3), w)              # too short to embed
        self.assertEqual(w[1], (3.0, 4.5))
        self.assertAlmostEqual(w[-1][1], 6.0)

    def test_clustering_finds_voices_and_folds_stray_windows(self):
        rng = np.random.default_rng(0)
        centres = rng.standard_normal((2, 192))
        emb = np.concatenate([centres[0] + 0.3 * rng.standard_normal((40, 192)),
                              centres[1] + 0.3 * rng.standard_normal((10, 192)),
                              rng.standard_normal((1, 192))])  # one stray window
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = diarize.cluster(emb, threshold=0.85, min_cluster_windows=3)
        self.assertEqual(len(set(labels.tolist())), 2)
        self.assertEqual(int(labels[0]), 0)  # biggest first
        self.assertTrue((labels[:40] == 0).all() and (labels[40:50] == 1).all())

    def test_short_runs_are_absorbed_into_turns(self):
        wins = [(t, t + 1.5) for t in np.arange(0, 10, 0.75)]
        labels = np.zeros(len(wins), np.int32)
        labels[6] = 1  # one window says someone else: smoothed away
        turns = diarize.turns_from_windows([(0.0, 11.5)], wins, labels, min_turn_s=1.0)  # its run: 0.75 s
        self.assertEqual([t["cluster"] for t in turns], [0])

    def test_names_saved_voice_then_lecturer_then_students_in_order(self):
        c = np.eye(3, 192, dtype=np.float32)
        named = diarize.name_clusters(c, speech_s=[100, 10, 20], first_s=[0, 5, 3], profiles=[("Pak Budi", c[2])])
        labels = {x["cluster"]: x["label"] for x in named}
        self.assertEqual(labels, {0: "Dosen", 1: "Mahasiswa 1", 2: "Pak Budi"})
        named = diarize.name_clusters(c, [100, 10, 20], [0, 5, 3], profiles=[("Pak Budi", c[0])])
        labels = {x["cluster"]: x["label"] for x in named}
        self.assertEqual(labels, {0: "Pak Budi", 1: "Mahasiswa 2", 2: "Mahasiswa 1"})


class ChunkAndMergeTest(unittest.TestCase):
    def test_long_turns_are_cut_at_pauses_under_the_limit(self):
        regs = [(0, 12), (12.6, 24), (24.8, 40)]
        ch = final_asr.chunks([{"start": 0, "end": 40, "speaker": "S1"}], regs, None, max_chunk_s=25)
        self.assertTrue(all(c["end"] - c["start"] <= 25 for c in ch))
        self.assertAlmostEqual(ch[0]["end"], 24.4)  # the pause between 24 and 24.8
        self.assertEqual(ch[-1]["end"], 40)

    def test_words_take_the_speaker_of_their_moment_and_padding_never_duplicates(self):
        asr = {"chunks": [
            {"start": 0, "end": 5, "speaker": "S1", "language": "id", "segments": [
                {"flagged": False, "avg_logprob": -0.2, "words": [[0.1, 0.5, " Halo", 0.9], [0.6, 1.0, " semua.", 0.9],
                                                                  [4.9, 5.2, " Pak", 0.5]]}]},
            {"start": 5, "end": 9, "speaker": "S2", "language": "id", "segments": [
                {"flagged": True, "avg_logprob": -1.2, "words": [[4.9, 5.2, " Pak,", 0.8], [5.3, 5.8, " kenapa?", 0.8],
                                                                 [8.0, 8.5, " Oke", 0.9]]}]}]}
        turns = [{"start": 0, "end": 5.0, "speaker": "S1"}, {"start": 5.0, "end": 7.5, "speaker": "S2"},
                 {"start": 7.6, "end": 9, "speaker": "S1"}]
        lines = merge.build_lines(asr, turns)
        self.assertEqual([(ln["speaker"], ln["text"]) for ln in lines],
                         [("S1", "Halo semua."), ("S2", "Pak, kenapa?"), ("S1", "Oke")])  # one "Pak", not two
        self.assertTrue(lines[1]["flagged"])
        self.assertEqual([ln["id"] for ln in lines], ["L1", "L2", "L3"])

    def test_a_long_pause_starts_a_new_line(self):
        asr = {"chunks": [{"start": 0, "end": 10, "speaker": "S1", "language": "id", "segments": [
            {"flagged": False, "avg_logprob": -0.1, "words": [[0, 0.5, " a", 1], [0.6, 1, " b", 1], [3, 3.5, " c", 1]]}]}]}
        lines = merge.build_lines(asr, [{"start": 0, "end": 10, "speaker": "S1"}], gap_s=1.0)
        self.assertEqual([ln["text"] for ln in lines], ["a b", "c"])


if __name__ == "__main__":
    unittest.main()
