# Soundxx: focus-ear

Real-time voice isolation for a noisy lecture hall. focus-ear listens through
the MacBook's built-in mic, tells the people in the room apart, and plays the
room to your Galaxy Buds with the speaker you pick at full volume and
everyone else turned down.

**Status:** all four stages are built.

1. Speech detection
2. Noise gate
3. Speaker identification
4. Per-speaker gain, with an interactive terminal UI

Extension points exist for noise suppression, overlapping-speech separation
and transcription, but those aren't implemented.

## Setup

Needs macOS on Apple Silicon and Python 3.11.

```sh
cd Soundxx
python3.11 -m venv venv          # skip if venv/ already exists
source venv/bin/activate
pip install -r requirements.txt
```

**First run downloads the speaker model** (~80 MB, `speechbrain/spkrec-ecapa-voxceleb`)
into `~/.focus-ear/models/`, so do it once with internet access. After that
it loads offline. That matters in a lecture hall without wifi.

**Microphone permission.** The first run triggers a macOS prompt for your
terminal app (Terminal, iTerm, or VS Code). If it's denied, macOS feeds the
app pure digital silence instead of raising an error; focus-ear detects this
and tells you. To fix it: System Settings → Privacy & Security → Microphone
→ enable your terminal, then restart the terminal.

## Using it

```sh
python -m focus_ear                  # interactive UI
python -m focus_ear --debug          # same, with the debug panel open
python -m focus_ear --list-devices   # check the mic and the Buds ("Mewo") are found
```

```
MacBook Pro Microphone → ● Mewo   latency 612 ms (incl. 250 ms lookahead)   drops: audio 0 · embeddings 0
mic  -40 dB   SPEECH    gate open   focus: Speaker 1, others -20 dB

key  speaker                   level                        spoken
0      Everyone (passthrough)
1    ▶ Speaker 1               ████████████········  -32 dB  ◀ talking  4:12
2      Speaker 2               ██··················               0:07
 1-9 Select   0 Everyone   n Name speaker   r Reset   d Debug   q Quit
```

1. **Wait for the speakers to appear.** A voice gets a row after it's been
   heard 3 times (about 1–2 s of speech), so a cough or a door doesn't
   create a phantom speaker. The lecturer is usually the one with the
   loudest bar and the most `spoken` time.
2. **Press their number.** They play at full volume and everyone else at
   −20 dB. `0` goes back to everyone at full volume; the noise gate stays on.
3. **Press `n` to name them** (for example "Prof. Lee"). They're saved to
   `~/.focus-ear/speakers.json`. Next lecture they're recognised and
   selected automatically once they've spoken a few times. Auto-selection
   never overrides a choice you've made in that session.
4. `r` forgets everyone except named speakers. `d` shows timings, the speech
   probability, and the cosine similarity of the last voiceprint to each
   speaker (useful for tuning, below). `q` quits.

`--no-tui` gives a one-line status instead of the interactive UI; you can't
select a speaker there, but a named speaker is still auto-selected.
`--no-speakers` skips stages 3–4 entirely (no model, no added latency).
`--passthrough` processes nothing but still shows speech and speakers.

## What to expect

### Latency

About 100 ms from mic to macOS's output, plus the 250 ms `--lookahead`, plus
Bluetooth (~250 ms for the Buds). That comes to roughly 600 ms in your ears.
You'll hear the lecturer directly and again through the Buds; if the Buds
have ANC, turn it on.

### The identification lag

Nobody can be identified until they've spoken for a moment. A voiceprint
covers 1.5 s, and a new voice has to fill about half of it before it wins.
The lookahead delays the audio so decisions land closer to the audio they
describe. Measured on synthetic voices taking turns with 0.8 s pauses,
running the real app in real time:

| `--lookahead` | lecturer returning after a question | a student taking over | added latency |
|---|---|---|---|
| 250 ms (default) | ~0.05 s at the wrong gain | ~1.0–1.25 s at full volume | 250 ms |
| 750 ms | ~0.05 s | ~0.65–0.85 s | 750 ms |

After 0.5 s of silence focus-ear forgets who was talking and plays the next
voice at full volume until it's identified. So the person you selected is
never held down while the model catches up, as long as there was a pause.
The cost is that each student's first second gets through. If someone
interrupts with no pause, the previous speaker's gain applies for about a
second.

### Choosing the defaults

The spec asked for a 1.0 s window and a 0.65 cosine threshold. Measured
with real ECAPA voiceprints of synthetic voices, clean and with simulated
lecture-hall reverb and noise, that combination splits one voice into many
"speakers". In the reverberant simulation, the selected lecturer ended up at
full volume only 8% of the time.

Each result below reads as: lecturer at full volume / student speech turned
down / clusters found, with 4 real speakers.

| window | threshold | merge | clean | hall |
|---|---|---|---|---|
| 1.0 s | 0.65 | none | 53% / 88% / 7.6 | 8% / 96% / 7.3 |
| 1.0 s | 0.40 | 0.55 | 100% / 81% / 3.5 | 83% / 84% / 6.3 |
| **1.5 s** | **0.45** | **0.55** | **100% / 84% / 3.6** | **97% / 85% / 4.8** |

Hence the defaults: `--embed-window 1.5`, `--cluster-threshold 0.45`, and
merging of clusters whose voiceprints converge (`--merge-threshold 0.55`).
The student figures top out around 85% because of the identification lag
above.

These are synthetic voices in a simulated room. Your hall will differ, so
tune with `d` open:

| symptom | what to change |
|---|---|
| One person keeps turning into new speakers | Lower `--cluster-threshold` (0.40) or raise `--embed-window` |
| Two people share one row | Raise `--cluster-threshold` (0.5–0.55) |
| Rows appear for noises | Raise `--min-sightings` |
| The gate chops words | Raise `--gate-hangover-ms`, or lower `--vad-threshold` |
| The gate opens on rustling | Raise `--vad-threshold` to 0.6–0.7 |

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--list-devices` | | Print devices and which ones would be used |
| `--input-device NAME\|INDEX` | built-in mic | Mic to record from |
| `--output-device NAME\|INDEX` | `Mewo` | Output, by case-insensitive name substring |
| `--passthrough` | | No audio processing (analysis still shown) |
| `--no-tui` | | One-line status instead of the UI |
| `--vad-threshold P` | `0.5` | Speech probability that counts as speech |
| `--no-gate` | | Don't attenuate non-speech |
| `--gate-attenuation DB` | `-18` | Gain when nobody is speaking |
| `--gate-hangover-ms MS` | `200` | Keep the gate open this long after speech |
| `--gate-lookahead-ms MS` | `0` | Open the gate before a word starts (adds latency) |
| `--no-speakers` | | Skip speaker identification (stages 3–4) |
| `--device auto\|mps\|cpu` | `auto` | Where the speaker model runs; `auto` is the Apple GPU when available |
| `--cluster-threshold COS` | `0.45` | Similarity needed to count as a known speaker |
| `--merge-threshold COS` | `0.55` | Merge speakers whose voiceprints converge this far |
| `--embed-window SEC` | `1.5` | Speech per voiceprint |
| `--embed-hop SEC` | `0.25` | How often a voiceprint is taken |
| `--max-speakers N` | `8` | Speakers tracked at once (1–9) |
| `--speaker-timeout SEC` | `120` | Forget unnamed speakers not heard this long |
| `--min-sightings N` | `3` | Voiceprints before a speaker is shown |
| `--boost DB` | `0` | Gain for the selected speaker |
| `--attenuation DB` | `-20` | Gain for everyone else |
| `--lookahead MS` | `250` | Audio delay that aligns speaker decisions (see the table above) |
| `--buffer-ms MS` | `64` | Output jitter buffer; raise it if you hear dropouts |
| `--latency low\|high\|SEC` | `low` | PortAudio device buffer hint |
| `--allow-speakers` | | Permit loudspeaker output (refused by default: feedback) |
| `--debug` | | Debug panel on, and per-second timing in `~/.focus-ear/focus-ear.log` |

## Troubleshooting

- **The Buds sound like a phone call.** Another app opened the Buds' mic,
  which switches Bluetooth to hands-free mode. focus-ear never records from
  them. Close Zoom/Discord/dictation, or set the Mac's input to the built-in
  mic.
- **Dropouts.** In the debug panel (`d`), check `underrun` and the worker
  timings. Try `--buffer-ms 120`.
- **`embeddings … dropped` climbing.** The model can't keep up (the Mac is
  busy or throttling). Audio isn't affected; identification just gets
  coarser. `--embed-hop 0.5` halves the load.
- **The Buds went to sleep.** The header shows "waiting for 'Mewo'".
  Playback resumes about 2 s after they reconnect; speakers and selection
  are kept.
- **The mic works in other apps but focus-ear reports silence.** Give *this*
  terminal microphone permission (see Setup).

## Design

```
                        ┌──────────── worker thread ─────────────────────────────┐
built-in mic ─► in_ring ┤ analysis: resample to 16 kHz → 512-sample frames         │
 (44.1 kHz)   (callback │   → VAD (silero, ONNX) ─► speech yes/no                 │
               copies)  │   → every 0.25 s: speech frames of the last 1.5 s ──────┼─► speaker-embed thread
                        │                                                         │   ECAPA (MPS) → cluster
                        │ audio: gate → 250 ms delay → speaker gain ─► out_ring ─┼─► output callback ─► Buds
                        └─────────────────────────────────────────────────────────┘
          UI (main thread) ◄── SpeakerTracker snapshot, 10×/s (lock never taken by audio callbacks)
          supervisor thread: detects dead streams, probes for the Buds, reconnects
```

- **The audio callbacks only copy** into and out of lock-free
  single-producer/single-consumer rings. No inference, allocation, logging
  or locks.
- **Native-rate streams.** Playback runs at the mic's own 44.1 kHz. Only the
  models see a 16 kHz copy, made by a streaming soxr resampler.
- **VAD without torch.** silero-vad's ONNX model is vendored
  (`focus_ear/models`, MIT) and run directly by ONNX Runtime. The
  `silero-vad` pip package would pull in torch.
- **Speaker embeddings run on their own thread, not the worker.** ECAPA
  takes ~10 ms back to back, but 35–100 ms when called every 250 ms: Apple
  Silicon clocks down between bursts, and thread QoS doesn't help. Inline
  inference stalled the audio path and caused 1–3 dropouts a second. Now
  the worker hands a window over without waiting. If the previous one is
  still being embedded, the window is dropped and counted (`drops:
  embeddings`), so a slow model costs accuracy, never latency.
- **Online clustering.**
  - Each voiceprint joins the closest speaker above `--cluster-threshold`.
    That speaker's centroid is a running mean at first, then an exponential
    average with α = 0.1.
  - A voiceprint below the threshold starts a new speaker.
  - Speakers whose centroids converge get merged.
  - Confirmed speakers are forgotten after 120 s unheard, unconfirmed ones
    after 30 s. Named and selected speakers never are.
  - At the cap, unconfirmed speakers are evicted first.
- **Gains.** Every change is a 50 ms linear ramp (`SmoothGain`), whether it
  comes from the gate or from a change of speaker, so there are no clicks.
- **Two independent streams**, not one duplex stream, so the mic keeps
  running while the Buds sleep. A throwaway subprocess checks for them every
  2 s.

| Module | Role |
|---|---|
| `audio_io.py` | Ring buffers, device matching, streams, reconnects |
| `pipeline.py` | Worker, 16 kHz analysis feed, stage/analyzer/tap interfaces, metrics |
| `vad.py` | silero-vad through ONNX Runtime |
| `embeddings.py` | ECAPA embedder, speaker analyzer, inference thread |
| `clustering.py` | Online clusterer; the thread-safe tracker shared with the UI |
| `gain.py` | `SmoothGain`, `DelayLine`, `NoiseGate`, `SpeakerGain` |
| `profiles.py` | Named speakers in `~/.focus-ear/speakers.json` |
| `tui.py` | Textual UI |
| `main.py` | CLI, wiring, plain status mode |

**Extension points** (in `pipeline.py`, not implemented):

- noise suppression: an `AudioStage` before the gate
- overlapping-speech separation: an `AudioStage` that replaces the speaker gain
- transcription: an `AudioTap`, which sees each block with its speaker label

## Tests

```sh
python -m unittest discover -s tests
```

- **VAD:** silence, tones, and a recorded speech clip.
- **Gain:** ramp continuity, hangover, lookahead.
- **Clustering:** real ECAPA voiceprints, clean and hall, plus a regression
  test showing why 0.65 fragments.
- **Tracker:** selection, auto-selection, enrollment.
- **Speaker analyzer:** hop cadence, silence skipping, background inference
  and drop counting, forgetting after a pause.
- **Pipeline:** the full chain offline.
- **TUI:** keys, naming (digits and `q` type into the name box), exit codes,
  driven headless with Textual's pilot.
- **Model:** the real ECAPA test runs when the model has been downloaded.
