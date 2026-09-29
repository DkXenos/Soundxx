# focus-ear: status audit

**Date:** 2026-09-29 · **Commit:** `36e749d` · **Machine:** Apple M3, 8 GB, macOS 26

This audit only reports; nothing in the code was changed. Every number
below was measured today, unless it's marked as coming from the README or
the log.

## The short version

- **All of it is written. None of it has been checked against reality.**
  - All five stages exist, plus denoising, AGC, the limiter and hardening.
  - All 105 tests pass.
  - But every tuning decision was made on macOS text-to-speech voices with
    synthetic noise or reverb.
  - No real lecture recording has ever gone through it.
- **Nothing from Stage 3 onwards has been heard through the Buds.** The log
  shows Mewo connected only on 2026-09-24, during the Stage 1–2 runs. Every
  run since then had no output device, a virtual one, or used `--file`.
- **There is no evaluation harness.**
  - No WER, no DER, no Hungarian matching, and no sweep scripts, in the
    repo or anywhere on disk.
  - The README's tuning tables came from ad-hoc scripts in `/tmp`, and
    most of those are already gone.
- **The one metric used to compare settings is noisy.**
  - `--file`'s "mean avg_logprob" came out at −0.27, −0.95, −2.15 and
    −0.74 on four identical runs of the same clip.
  - Whisper's temperature fallback samples randomly.
- **Speaker attribution is the weak link, and it drives both outputs.**
  - At −1 dB SNR the lecturer splits into several clusters, and some of
    their speech gets labelled as the student.
  - With the lecturer selected, roughly 30% of their speech in a replay
    never reached the transcript.
  - At −6 dB SNR the transcript is hallucinated nonsense, with or without
    denoising.

---

## 1. What exists

There are no `TODO`/`FIXME` markers and no stub functions. Everything
listed as "complete" is working code; the caveats are about validation.

| Module | What it does | State |
|---|---|---|
| `__init__.py`, `__main__.py` | Version string and the `python -m focus_ear` entry point | Complete |
| `config.py` | `Config` dataclass with every default, and the Whisper prompt sentence | Complete |
| `main.py` (807 lines) | CLI, model loading, wiring, plain status loop, `--file` mode, latency estimate, clean shutdown with the two-Ctrl-C logic | Complete. `run()`, `run_plain()` and `build_session()` have no unit tests |
| `audio_io.py` | Lock-free SPSC rings, device matching, separate input and output streams, a supervisor that reconnects the Buds via a subprocess probe and a PortAudio re-init, end-to-end latency accounting | Complete. Reconnects are tested only against a fake `sounddevice`; a real Buds reconnect was never logged |
| `pipeline.py` | Worker thread, 16 kHz analysis feed, the analyzer/stage/tap interfaces, the three denoise scopes, metrics | Complete. The `AudioTap` interface has no implementations. The overlapping-speech separation "extension point" is a docstring only |
| `vad.py` | silero-vad v6.2.3 through ONNX Runtime (vendored model), with hysteresis | Complete |
| `gain.py` | `SmoothGain` (50 ms linear ramps), `DelayLine`, `NoiseGate`, `SpeakerGain` | Complete. `SpeakerGain`'s "keep the current gain on None" branch is never reached: the tracker never returns None |
| `embeddings.py` | ECAPA-TDNN wrapper (MPS, falling back to CPU); `SpeakerAnalyzer` with its own inference thread | Complete |
| `clustering.py` | Online clusterer (threshold, merging, expiry, cap) and the thread-safe `SpeakerTracker` | Complete |
| `profiles.py` | Named voiceprints in `~/.focus-ear/speakers.json` | Complete, **but never used for real**: the file doesn't exist, so nobody has ever been enrolled |
| `denoise.py` | DeepFilterNet3 reimplemented as a stream (the package only does whole files), an exact polyphase resampler, a fixed-delay wrapper | Complete. It depends on deepfilternet 0.5.6's internal layer layout (asserts) and on a torchaudio import shim |
| `dsp.py` | 80 Hz high-pass, AGC, lookahead limiter, clip meters | Complete |
| `health.py` | Memory, GPU memory, uptime, the health line | Complete. No tests. Reads macOS `rusage_info_v2` at a hard-coded offset |
| `transcript.py` | Utterance segmenter, ASR thread with a bounded queue, Markdown and JSONL writer | Complete |
| `asr.py` | mlx-whisper backend; faster-whisper CPU fallback | MLX is complete. **The fallback only works online**: its model (`faster-whisper small`) isn't cached, and the fallback test uses a mock |
| `tui.py` | Textual UI: speaker table, transcript panel, debug panel, naming | Complete. Tested headless with fakes; never run with the Buds connected |

Dead or unused code: every `reset()` on the analyzers and stages (the app
never calls them; `r` only resets the tracker), the `AudioTap` interface,
and `Denoiser(bypass=True)`, which only tests use.

## 2. Stage-by-stage status

| Stage | Status | What's missing or caveated |
|---|---|---|
| **1. VAD** (silero, worker thread) | **DONE** | Runs inline on the worker at 0.23–0.34 ms per frame. Uses the vendored ONNX model, not the `silero-vad` package |
| **2. Noise gate** (ramps, hangover) | **DONE** | 50 ms linear ramps, 200 ms hangover, optional lookahead. The last stage anyone heard through the Buds (09-24, no metrics logged). It's a VAD gate: during speech, all room noise passes at full level |
| **3. ECAPA + online clustering** | **DONE in code, unvalidated** | Defaults changed from the spec (see §4). Validated only on synthetic voices. Fragments in noise (§5) |
| **4. Per-speaker gain + TUI + enrollment** | **DONE in code, unvalidated** | Gain and TUI work in tests and headless. **Enrollment has never been used** (no `speakers.json`). **Never run through the Buds.** Number keys can shift mid-lecture: key *n* is the *n*-th confirmed speaker in insertion order, so a merge or expiry renumbers everyone after it |
| **Transcription** (mlx-whisper, segmentation, session files) | **DONE in code, synthetic-only** | The only live session (09-26 14:11, 13 s) produced 0 lines. Output is nondeterministic (§5). `--language id` turns English speakers into garbage |
| **Denoising** (DeepFilterNet, scope, high-pass, limiter) | **DONE** | **By default the models never hear denoised audio**: `--denoise-scope playback` gives only your ears the denoised signal. Costs 2–3× what the README says (§5) |
| **Evaluation harness** (WER, DER, Hungarian, `--file`, sweeps) | **PARTIAL** | `--file` mode works and prints proxy stats (VAD %, cluster count, mean match cosine, mean logprob). **Not started:** WER, DER, Hungarian speaker mapping, reference transcripts or RTTM, parameter sweeps, committed eval scripts. focus-ear also can't save the raw mic audio of a live session, so a live run leaves nothing to re-evaluate |

## 3. What you can run right now

Legend: ✓ = exercised today · (t) = covered by unit tests only · (r) =
wired through `config_from_args`, checked by reading the code only.

| Flag | What it does | Checked |
|---|---|---|
| `-h`, `--help` | Help text (not in the README) | ✓ |
| `--list-devices` | Print devices and mark the ones that would be used | ✓ |
| `--input-device NAME\|INDEX` | Mic (defaults to the built-in mic, never the Buds' mic) | (t) |
| `--output-device NAME\|INDEX` | Output by name substring (default `Mewo`) | ✓ |
| `--passthrough` | No audio processing at all; analysis still runs. No effect in `--file` mode | ✓ (file) |
| `--no-tui` | One-line status instead of the UI; no speaker selection | ✓ |
| `--highpass HZ` | High-pass cutoff; `0` turns it off | ✓ |
| `--denoise on\|off` | DeepFilterNet3 | ✓ |
| `--denoise-mix 0-1` | Blend of denoised and original | ✓ |
| `--denoise-scope all\|playback\|models` | Who gets denoised audio. In `--file`, `playback` means no denoiser at all | ✓ |
| `--latency-budget MS` | Startup warning threshold only | (r) |
| `--no-agc`, `--agc-target DBFS` | AGC off, or its target level | (t) |
| `--no-limiter`, `--limiter-ceiling DBFS` | Limiter off, or its ceiling | (t) |
| `--vad-threshold P` | Speech probability threshold (exit threshold is P − 0.15) | (t) |
| `--no-gate`, `--gate-attenuation DB`, `--gate-hangover-ms MS`, `--gate-lookahead-ms MS` | Gate controls | (t) |
| `--no-speakers` | Skip stages 3–4 | ✓ |
| `--device auto\|mps\|cpu` | Where ECAPA runs | (r) |
| `--cluster-threshold`, `--merge-threshold`, `--embed-window`, `--embed-hop`, `--min-sightings`, `--max-speakers` | Clustering controls | ✓ |
| `--speaker-timeout SEC` | Forget unnamed speakers after this long unheard | (t) |
| `--boost DB`, `--attenuation DB` | Gain for the selected speaker, and for everyone else | (t) |
| `--lookahead MS` | Speaker-gain delay (default 250 ms) | (t) |
| `--no-transcribe` | No Whisper | ✓ |
| `--asr-model REPO\|PATH` | Whisper model. A bad name falls back to faster-whisper, fails, and exits 2 with a clear error | ✓ |
| `--language id\|en\|auto` | Decode language | ✓ |
| `--initial-prompt TEXT` | Whisper context | (r) |
| `--utterance-gap MS`, `--max-utterance SEC` | Segmenter controls | (t) |
| `--no-save` | Don't write session files | ✓ |
| `--file PATH` | Offline run over a recording | ✓ |
| `--buffer-ms MS` | Output jitter cushion | (r) |
| `--latency low\|high\|SEC` | PortAudio latency hint | (r) |
| `--allow-speakers` | Permit loudspeaker output | (t) |
| `--debug` | Debug panel, per-second timing lines, and a health line every 30 s in `~/.focus-ear/focus-ear.log` | ✓ |

**Keyboard (TUI):**

| Key | Action |
|---|---|
| `1`–`9` | Focus on the *n*-th shown speaker |
| `0` | Everyone at full volume (the gate stays on) |
| `n` | Name the selected speaker. `Enter` saves, `Esc` cancels (the README doesn't mention `Esc`) |
| `r` | Forget unnamed speakers |
| `t` | Show or hide the transcript panel |
| `d` | Show or hide the debug panel |
| `q`, `Ctrl-C` | Quit, transcribing the utterance in progress. A second Ctrl-C skips the wait but still closes the files |
| SIGTERM, SIGHUP | Same as quit |

**The README claims things that aren't true, or weren't measured:**

- **"If mlx-whisper can't load… usually the model isn't downloaded yet and
  there's no internet"** (implying faster-whisper covers that case). It
  doesn't: faster-whisper's model isn't cached either, so offline the
  fallback fails too.
- **"`--file` … repeated runs give the same result."** That's true for
  speaker clustering. It's false for the transcript text (§5).
- **"Roughly 650 ms" latency, and the example header's "612 ms".** These
  are estimates. Latency through the Buds has never been measured.
- **Denoise "2.7–2.9 ms per 32 ms block".** Today it averaged 5.5–8.5 ms
  live, with spikes to 49–96 ms. The startup self-benchmark still reports
  3.0–3.7 ms, so it understates the steady-state cost.
- **`--buffer-ms 64`.** The measured mean output buffer is about 117 ms:
  the 64 ms prefill plus up to 3 blocks of drift slack.
- **"Playback runs at the mic's own 44.1 kHz."** Today the mic
  negotiated 48 kHz (1536-sample blocks). It still works, but the README's
  1411-sample and 1795-sample figures don't apply.

**Works but isn't documented:**

- Exit codes: 2 for a device or model error, 130 for an interrupted
  `--file`, 3 when CoreAudio is wedged.
- `--file` logs its progress every 10%.
- Some parameters exist only in `Config`, with no flag:

| Parameter | Value |
|---|---|
| pre-roll | 300 ms |
| minimum utterance | 400 ms |
| ASR queue | 4 |
| block | 32 ms |
| max backlog | 100 ms |
| ramp | 50 ms |
| cluster α | 0.1 |
| phantom-cluster expiry | 30 s |
| forget-speaker-after-silence | 0.5 s |

## 4. Known problems

### Tests

**105 tests, 105 pass, 0 skipped, 21.3 s.** The real-model tests (ECAPA,
DeepFilterNet, whisper-small) all ran, because the models are cached.

| File | Tests | Covers |
|---|---|---|
| `test_transcript.py` | 24 | Segmenter edge cases, queue drop and wait, writer, shutdown, a real Whisper run over `--file` |
| `test_clustering.py` | 14 | Clusterer and tracker, on real ECAPA voiceprints **of synthetic voices** |
| `test_audio_io.py` | 11 | Rings, device matching, reconnect against a fake `sounddevice` |
| `test_denoise.py` | 11 | Streaming equals offline, delay fixed for 3 minutes, scopes, resampler |
| `test_dsp.py` | 9 | High-pass, limiter ceiling, AGC, clip meter |
| `test_gain.py` | 9 | Ramps, hangover, lookahead |
| `test_speakers.py` | 9 | Analyzer cadence and threading, profiles, one real-ECAPA smoke test |
| `test_vad.py` | 7 | Silence, tones, a 6 s recorded clip |
| `test_pipeline.py` | 6 | Resampling feed, offline chain, worker drop and error paths |
| `test_tui.py` | 5 | Keys, naming, exit codes, headless |

**Untested:**

- **Accuracy of any kind.** No WER, DER or attribution assertions against
  a reference.
- Real devices and the real latency accounting.
- `main.run`, `run_plain`, `build_session`, `check_latency_budget`.
- `health.py`.
- Enrollment carrying over to the next session on real voices.
- The real faster-whisper fallback.
- Any real human speech longer than 6 s.

### Fragile or speculative code

- **`denoise.py`** mirrors `DfNet.forward` layer by layer and asserts on
  deepfilternet 0.5.6's internals. It's installed with `--no-deps` against
  that package's `numpy<2` pin, and imports through a torchaudio shim. Any
  version bump will likely break it.
- **Speaker attribution feeds two things.** The same per-window decision
  sets the Buds gain, and the per-utterance majority vote decides what gets
  transcribed. A wrong label mutes the lecturer in your ears and drops them
  from the transcript.
- **Worker spikes exceed the output cushion.** The worst single blocks
  were 57–70 ms live and 116 ms in the replay (the denoiser), against a
  64 ms cushion. No underruns showed up in 45 s to a virtual device.
  Behaviour with real Bluetooth is unknown.
- **The faster-whisper fallback is speculative.** It was never used for
  real and doesn't work offline.
- **Speaker number keys renumber** after merges and expiries (see Stage 4).

### Dependencies

- Everything in both requirements files is installed at the pinned
  versions. `pip check` reports only the expected deepfilternet pin
  conflicts (numpy, packaging).
- **Installed but effectively unused:** `faster-whisper` and
  `ctranslate2`, fallback only, with no model cached. `pyflakes` is
  installed but isn't in the requirements (a dev tool).
- **Nothing needed for evaluation is installed** (no WER library, no DER
  tooling). `scipy` is installed, and its `linear_sum_assignment` would
  cover the Hungarian matching.
- **Models cached:**
  - ECAPA (`~/.focus-ear/models`)
  - whisper-small-mlx
  - DeepFilterNet3
- **Not cached:** faster-whisper small, and any larger Whisper model.

### Where the design moved away from the spec

I don't have your original prompts, so this list comes from the README,
code comments and my notes from earlier sessions.

1. **Clustering defaults.**
   - Spec: 1.0 s window, 0.65 threshold.
   - Now: 1.5 s window, 0.45 threshold, plus merging at 0.55, 3 sightings
     before a speaker is shown, and forgetting who was talking after 0.5 s
     of silence.
   - Why: in a simulated reverberant hall with synthetic voices, the spec
     values fragmented one lecturer into many speakers.
   - Today, with the spec-like 1.0 s / 0.5 on the clean clip: 4 clusters
     for 2 real speakers.
2. **Embeddings run on their own thread, not the worker.** Running them
   inline caused 1–3 dropouts a second, because Apple Silicon clocks down
   between bursts.
3. **Silero runs from a vendored ONNX model, not the `silero-vad`
   package.** The reason was to avoid torch. That reason no longer holds:
   torch is required anyway, for ECAPA and DeepFilterNet.
4. **The models don't hear denoised audio by default** (`--denoise-scope
   playback`).
   - On the −1 dB synthetic clip, denoising wrecked Whisper and merged the
     student into the lecturer.
   - Caveat: that comparison rests on single runs of the nondeterministic
     logprob metric. The "denoised = nonsense" result is probably robust;
     the exact numbers aren't.
5. **DeepFilterNet is reimplemented as a stream,** with a custom
   resampler, because the package only works on whole files. The package's
   dependency pins are overridden.
6. **Streams run at the mic's native rate** (44.1 or 48 kHz), as two
   separate streams rather than one duplex stream, so the mic survives the
   Buds sleeping.
7. **Whisper is forced to Indonesian (`id`).** English students come out
   as garbage whenever everyone is being transcribed.
8. **"Isolating a speaker" is turn-level gain switching on the whole room
   mix.**
   - Nothing is separated. Overlapping speech gets whichever gain the
     current window earned.
   - While the lecturer talks, everything else in the room plays at full
     volume.
   - An unidentified speaker plays at full volume, so every student's
     first ~1 s gets through by design.
   - Separation exists only as a docstring.

## 5. Real measurements

**Conditions:** 2026-09-29, M3 8 GB. The Buds weren't connected (Mewo
isn't in the device list). The mic ran at 48 kHz with 32 ms blocks, in a
quiet room.

**What I ran:**

| Run | Setup | Length |
|---|---|---|
| **A** | Default config, live mic, output absent (waiting for Mewo) | 131 s |
| **B** | Live mic, output to the virtual `ZoomAudioDevice` (inaudible; Zoom not running) | 45 s |
| **C** | Real-time replay of a synthetic noisy lecture (−1 dB SNR, 70 s) through the full default live chain | 70 s |
| `--file` | 35 s clips | — |

The room was silent in A and B (VAD 0% speech), so embeddings and Whisper
only ran under load in C and in the `--file` runs.

### Latency

| | Result |
|---|---|
| **Mic → DAC, measured (run B, no Bluetooth)** | **449 ms mean** (416–467) |
| Startup estimate, same run | 485 ms |
| Where it goes (one sample second) | See the next table |
| **Bluetooth to the Buds** | **Not measured.** The README assumes about 250 ms, which would put you at roughly 700 ms in your ears |

The sample second breaks down like this (roughly 467 ms):

| Component | ms |
|---|---|
| Mic | 33 |
| Processing | 8 |
| Speaker lookahead 250 + denoiser 38 + limiter 2 | 290 |
| Output buffer | 117 |
| Output device | 33 |

### Worker time per 32 ms block

Mean of 1-second means / worst single block. The budget is 32 ms; the
output cushion is 64 ms.

| Stage | A (live, silent) | B (live, virtual out) | C (replay, speech) |
|---|---|---|---|
| denoise | 7.14 / 49.3 | 8.49 / 54.6 | 5.50 / 96.0 |
| vad | 0.33 / 24.8 | 0.34 / 21.0 | 0.23 / 9.2 |
| highpass | 0.15 / 8.7 | 0.18 / 1.3 | 0.09 / 5.9 |
| limiter | 0.18 / 2.6 | 0.18 / 5.2 | 0.16 / 4.9 |
| resample, gate, speaker-gain, agc, speaker, utterance | ≤ 0.04 each | ≤ 0.04 each | ≤ 0.03 each |
| **total** | **8.08 / 70.0** | **9.48 / 56.9** | **6.16 / 115.7** |

### Drops

| Run | Result |
|---|---|
| A | 0 worker drops, 0 input overflows. Underruns can't be measured without an output |
| B | 0 underruns, 0 underflows, 0 overflows, 0 worker drops in 45 s |
| C | 0 worker drops, 0 embedding drops, 0 transcript drops, 0 processing errors |

### Models under load (run C)

- **ECAPA on MPS:**
  - 221 windows, all embedded, none dropped.
  - 21.4 ms median, 50.7 ms p90, 133.7 ms max, against a 250 ms hop.
- **Whisper-small on MLX:**
  - 12 utterances, 0.23–0.41 s each for 1–10 s of audio.
  - The exception was 2.03 s, on the one hallucinated utterance.
  - Queue wait was 0 s.
- **`--file` speed:**
  - 35 s of audio took 4.9–5.0 s (7× real time), clean and at −1 dB.
  - At −6 dB it took 11.6 s (3×): hallucinations decode slowly.

### Memory and CPU

Memory is the process footprint, the same measure as Activity Monitor's
"Memory" column.

| Situation | Footprint |
|---|---|
| Live, idle | 2.28–2.48 GB (MLX 718 MB, MPS 85 MB) |
| Replay C, models loaded | 2.57 GB |
| Replay C, after 70 s | 2.85 GB |
| `--file`, peak | 3.6–3.7 GB |

- CPU was 22–29% of one core in run A, with an idle room.
- The 45-minute soak (the README reports a 3.1 GB plateau) was not re-run.

### Accuracy

This is a 35 s synthetic clip: a TTS Indonesian lecturer, then a TTS
English student question, then the lecturer again. I checked it by eye
against the script; there's no WER tooling.

| Clip | SNR | Clusters founded / shown (2 real) | Attribution | Transcript |
|---|---|---|---|---|
| clean | — | 2 / 2 | Correct | Lecturer good, with term errors ("iterathe array", "oen" for O(n), "binar research"). The student line changes every run |
| noisy5 | −1.0 dB | 5 / 2 | The lecturer's "Hari ini kita pakai loop…" went to a phantom Speaker 2; "Pertanyaan bagus" went to the student | Partly right ("kompresitasnya", "ketingin" for linked list) |
| noisy0 | −6.0 dB | 5 / 3 | — | Hallucinated nonsense ("Terima kasih", Korean characters) |
| noisy0, `--denoise-scope all` | −6.0 dB | 3 / 1 (student merged in) | — | Nonsense, then a repetition loop |

- **With the lecturer selected (replay C, −1 dB; selected at 35 s):**
  - 3 utterances in the second half were discarded as "other speaker".
  - 2 of the 3 were the lecturer ("selamat pagi… hari ini kita pakai
    loop…" and "Pertanyaan bagus. Untuk linked list").
  - Roughly 8 of the lecturer's ~27 s of speech in that half never reached
    the transcript.
- **Nondeterminism:**
  - I ran the default `--file` on the clean clip 4 times with identical
    settings.
  - Mean avg_logprob came out at −0.27, −0.95, −2.15 and −0.74, and the
    English line was different every time.
  - One run marked it *(unclear)*. That run scored best, because the
    metric rewards the garbled case.
  - Cause: mlx-whisper's temperature fallback (0.0 → 1.0) samples randomly
    once greedy decoding fails.
  - Lecturer-only comparisons (the prompt wording) probably decode at
    temperature 0 and are stable. Anything that averages in the English
    line isn't reliable from one run.

## 6. What to do next

The goal is to filter noise, isolate one speaker and transcribe them in a
real lecture hall. These are the three blockers, in order.

1. **Get real data, and a way to score it.**
   - Nothing has been measured on a real lecture, so every threshold is a
     guess tuned on TTS voices.
   - Record 10–20 minutes from your usual seat on the MacBook mic.
     focus-ear can't record raw audio yet, so use QuickTime or `sox`.
   - Hand-label a few minutes: the words, and who spoke when.
   - Build the harness you listed: WER; DER with Hungarian speaker mapping
     (`scipy.optimize.linear_sum_assignment`, already installed); and
     fixed-seed or temperature-0 decoding, or repeated runs, so
     comparisons mean something.
   - Commit the scripts.
   - Until this exists, you can't tell whether any change helps.
2. **Speaker attribution is more broken than it looks, and everything
   depends on it.**
   - At −1 dB the lecturer fragments, and parts of them go to the student.
     With the lecturer selected, about 30% of their speech was dropped
     from the transcript.
   - The physics works against you: in a hall the lecturer is metres away
     and reverberant, while the students near you are close and loud.
   - "Isolation" only switches the volume of the whole room mix per turn.
     It can't reduce chatter while the lecturer talks.
   - The real hall's SNR will decide which of the fixes is needed:
     - more robust per-utterance attribution
     - enrolling the lecturer from a clean stretch
     - transcribing everything and labelling afterwards
     - better input, such as a phone or mic nearer the lecturer, likely
       the single biggest lever
   - At −6 dB nothing in the current chain works.
3. **Run the full stack on the Buds.**
   - Stages 3–5, denoise, AGC, limiter, TUI, reconnects and enrollment
     have never played through them.
   - Latency is 449 ms before Bluetooth, so probably around 700 ms in your
     ears. That's unmeasured, and may be too much to listen to over the
     live voice.
   - Worker spikes (57–116 ms against a 64 ms cushion) could cause
     dropouts that nobody has listened for.
   - This is cheap to do, and could turn up something that makes the
     whole thing unusable.

---

*How this was measured:*

- Tests: `python -m unittest discover -s tests -v`.
- Live runs: `python -m focus_ear --no-tui --debug --no-save`, the second
  with `--output-device ZoomAudioDevice`, stopped with SIGTERM and SIGINT
  (both exited cleanly in about 1 s).
- Replay: a wall-clock-paced feed into the real worker with the full
  default chain.
- `--file` runs on the clean, −1 dB and −6 dB clips.
- The measurement scripts are in this session's `/tmp` scratchpad and
  aren't committed.
