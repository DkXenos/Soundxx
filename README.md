# Soundxx: focus-ear

Real-time voice isolation for a noisy lecture hall. focus-ear listens through
the MacBook's built-in mic, tells the people in the room apart, and plays the
room to your Galaxy Buds with the speaker you pick boosted and everyone else
turned down.

**Status: phase 1, audio I/O with pure passthrough.** Mic → Buds works,
including recovery when the Buds go to sleep. Speaker detection, gating and
the interactive UI come next (see [Roadmap](#roadmap)).

## Setup

Needs macOS on Apple Silicon and Python 3.11. `portaudio` and `ffmpeg` from
Homebrew are fine to have, but the `sounddevice` wheel bundles its own
PortAudio.

```sh
cd Soundxx
python3.11 -m venv venv          # skip if venv/ already exists
source venv/bin/activate
pip install -r requirements.txt
```

**Microphone permission.** The first run triggers a macOS permission prompt
for your terminal app (Terminal, iTerm, or VS Code if you run it from the VS
Code terminal). If you deny it, macOS doesn't raise an error. It feeds the
app pure digital silence, which focus-ear detects and reports. To fix it:
System Settings → Privacy & Security → Microphone → enable your terminal,
then restart the terminal.

## Verify the Bluetooth path (phase 1)

1. Connect the Buds to the Mac.
2. Check that both devices are seen:

   ```sh
   python -m focus_ear --list-devices
   ```

   The built-in mic should be marked `<- input` and the Buds `<- output`. If
   your Buds' name doesn't contain "Galaxy Buds", pass
   `--output-device "<part of the name>"` (or the device index).
3. Run it, with the Buds in your ears:

   ```sh
   python -m focus_ear --passthrough
   ```

   You should hear the room through the Buds. The status line shows the mic
   level, the output device and the measured latency. Ctrl-C quits.
4. Put the Buds in their case, then take them out again. The status line
   switches to "waiting for 'Galaxy Buds'" and resumes about 2 s after they
   reconnect, with no restart needed.

For per-stage timing, add `--debug` (see below).

### What latency to expect

Measured on this M3 with a silent virtual output device: about **110 ms**
from mic to output. That's roughly 40 ms mic buffering, 50 ms jitter
buffer, 35 ms output device, and 0.2 ms of our processing. Bluetooth then
adds its own codec and radio delay, typically 150–250 ms for Galaxy Buds on
a Mac. macOS reports only part of that, so the figure shown may
underestimate what you hear.

You'll hear the lecturer twice: directly through the air, and ~0.3 s later
through the Buds. If your model has ANC, turn it on to suppress the direct
path.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--list-devices` | | Print devices (index, channels, rate) and which ones would be used |
| `--input-device NAME\|INDEX` | built-in mic | Mic to record from |
| `--output-device NAME\|INDEX` | `Galaxy Buds` | Output, by case-insensitive name substring |
| `--passthrough` | | No filtering (phase 1 always runs this way) |
| `--attenuation DB` | `-20` | Gain for non-selected speakers *(phase 2)* |
| `--boost DB` | `0` | Gain for the selected speaker *(phase 2)* |
| `--threshold COS` | `0.65` | Cosine similarity needed to join an existing speaker *(phase 2)* |
| `--buffer-ms MS` | `64` | Output jitter buffer; raise it if you hear dropouts |
| `--latency low\|high\|SEC` | `low` | PortAudio device buffer size hint; try `high` if the Buds crackle |
| `--allow-speakers` | | Permit loudspeaker output (the app refuses by default because of feedback) |
| `--debug` | | Log timing every second to the console and `~/.focus-ear/focus-ear.log` |

## Debug mode

`--debug` replaces the status line with one log line per second:

```
e2e 110 ms ≈ in-dev 42 + in-queue 0 + proc 0.2 + out-buf 50 + out-dev 35 | stage ms mean/max: total 0.17/0.32 | RTF 0.005 | mic -48 dB | in-ovf 0 in-drop 0ms out-underrun 0 out-underflow 0 out-skip 0ms worker-skip 0ms
```

- **e2e**: measured time from a sample hitting the mic's ADC to it reaching
  the output DAC, based on CoreAudio timestamps. The terms after `≈` show
  where the time goes; they are approximate and don't sum exactly.
- **stage ms**: mean/max processing time per stage per 32 ms block.
- **RTF**: real-time factor, meaning processing time ÷ audio time. It must
  stay well below 1. Above ~0.5 you're at risk of falling behind.
- **Glitch counters** (all per second):
  - `in-ovf`: the mic overflowed.
  - `in-drop`: the worker was too slow to drain the mic.
  - `out-underrun`: our output buffer ran dry, so you hear a dropout.
  - `out-underflow`: CoreAudio's own buffer ran dry.
  - `out-skip`: audio was discarded to keep latency bounded. Expect some
    after startup and reconnects, and a little every ~30 min from clock
    drift between mic and Buds.
  - `worker-skip`: the worker fell >300 ms behind and jumped ahead.

## Troubleshooting

- **Buds sound like a phone call.** Something opened the Buds' microphone,
  which switches Bluetooth to hands-free mode. focus-ear never records from
  the Buds (it refuses to), but another app (Zoom, Discord, dictation) may be
  doing it. Close it, or set the Mac's input to the built-in mic.
- **"looks like loudspeakers"**. That is the feedback guard: mic → speakers in
  the same room howls. Connect the Buds, or pass `--allow-speakers` at very
  low volume.
- **Dropouts or crackle.** Try `--buffer-ms 120`, then `--latency high`.
  `--debug` shows which counter is climbing.
- **Buds never come back after sleeping.** Check `--list-devices` in another
  terminal. If macOS doesn't list them, reconnect them in the Bluetooth menu.

## Design

```
built-in mic ─► input callback ─► in_ring ─► worker thread ─► out_ring ─► output callback ─► Buds
                                             (VAD, embeddings,
                                              clustering, gain)
                        supervisor thread: detects dead streams, probes for devices, reconnects
```

- **Two independent streams, not one duplex stream.** The mic and the Buds
  run on different hardware clocks, and when the Buds sleep only the output
  should die. The mic keeps feeding the analysis while the output waits.
- **The callbacks only copy.** No ML, locks, logging or (steady-state)
  allocation in them. The rings are lock-free single-producer/single-consumer
  buffers.
- **Clock drift and jitter** are absorbed by the output buffer. It holds
  `--buffer-ms` in reserve and trims stale audio (after reconnects, or as the
  clocks drift) so latency can't creep up.
- **Reconnects.** PortAudio only enumerates devices at startup, so a
  throwaway subprocess checks every 2 s whether the Buds are back. Then
  PortAudio is re-initialised and both streams reopen (a ~0.4 s gap in the
  mic).
- **Extension points** (`focus_ear/pipeline.py`), defined but not
  implemented:
  - noise suppression: an `AudioStage` before gain
  - overlapping-speech separation: an `AudioStage` that replaces gating
  - transcription: an `AudioTap` that observes blocks and speaker labels

| Module | Role |
|---|---|
| `audio_io.py` | Ring buffers, device matching, streams, reconnects |
| `pipeline.py` | Worker thread, stage/tap interfaces, metrics |
| `main.py` | CLI, logging, status line |
| `config.py` | Shared settings |
| `vad.py`, `embeddings.py`, `clustering.py`, `gain.py` | Phase 2 (stubs with interfaces) |
| `tui.py`, `profiles.py` | Phase 3 (stubs with interfaces) |

## Roadmap

1. ✅ **Audio I/O + passthrough.** Device selection, reconnects, latency
   metrics.
2. **Identify and gate.**
   - silero-vad (ONNX) on each 512-sample block.
   - ECAPA-TDNN embeddings over ~1 s of speech. MPS vs CPU is picked by
     benchmark; for inputs this small, CPU may win.
   - Online clustering with the cosine threshold.
   - Gain with 50 ms ramps.
   - Speaker identity needs ~1 s of speech, so gain decisions lag each turn.
     An optional lookahead delay trades latency for fewer mis-gated turn
     starts.
3. **TUI and enrollment.**
   - Textual UI: speaker list, level meters, active and selected speaker.
   - Keys: 1–9 select, 0 passthrough, r reset, n name, q quit.
   - Saved profiles in `~/.focus-ear/speakers.json`, auto-selected on
     startup.

## Tests

```sh
python -m unittest discover -s tests
```
