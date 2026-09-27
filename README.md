# Soundxx: focus-ear

Real-time voice isolation for a noisy lecture hall. focus-ear listens through
the MacBook's built-in mic, tells the people in the room apart, and plays the
room to your Galaxy Buds with the speaker you pick at full volume and
everyone else turned down. It also transcribes that speaker to a Markdown
file as the lecture goes.

**Status:** five stages are built, plus audio conditioning and the
hardening needed for 90-minute lectures.

1. Speech detection
2. Noise gate
3. Speaker identification
4. Per-speaker gain, with an interactive terminal UI
5. Transcription of the selected speaker (Whisper on the Apple GPU)

Around them:

- **Input:** an 80 Hz high-pass filter, then DeepFilterNet3 noise
  suppression.
- **Output:** automatic gain control (AGC) and a peak limiter.
- **Long sessions:** clip detection, Bluetooth reconnects, bounded memory,
  and a clean Ctrl-C.

An extension point exists for overlapping-speech separation, but it isn't
implemented.

## Setup

Needs macOS on Apple Silicon and Python 3.11.

```sh
cd Soundxx
python3.11 -m venv venv          # skip if venv/ already exists
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-denoise.txt   # noise suppression (optional)
```

The second line installs DeepFilterNet. It needs `--no-deps`: its package
metadata pins numpy below 2 and would downgrade everything else, although
it works fine with numpy 2 (a test checks this). `pip check` will complain
about those pins; that's expected. Without it, focus-ear runs with
`--denoise off` and says so at startup.

**First run downloads the speaker model** (~80 MB, `speechbrain/spkrec-ecapa-voxceleb`)
into `~/.focus-ear/models/`, and the **speech recognition model** (~480 MB,
`mlx-community/whisper-small-mlx`) into `~/.cache/huggingface/`, and the
**noise suppression model** (~8 MB, DeepFilterNet3) into
`~/Library/Caches/DeepFilterNet/`. So do the first run with internet access. After that both load offline. That matters
in a lecture hall without wifi. Trying a bigger model later
(`--asr-model mlx-community/whisper-medium-mlx`) downloads again, so do that
at home too.

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
python -m focus_ear --file lecture.wav   # transcribe a recording instead (no devices, no UI)
```

```
MacBook Pro Microphone → ● Mewo   latency 612 ms (incl. 250 ms lookahead)   drops: audio 0 · embeddings 0
mic  -40 dB   SPEECH    gate open   focus: Speaker 1, others -20 dB

key  speaker                   level                        spoken
0      Everyone (passthrough)
1    ▶ Speaker 1               ████████████········  -32 dB  ◀ talking  4:12
2      Speaker 2               ██··················               0:07
Transcript → 2026-09-26_0914.md   ● listening   ⋯ transcribing
00:03:58 Speaker 1: Oke, jadi hari ini kita pakai loop untuk iterate array.
00:04:06 Speaker 1: Kalau kita pakai for loop, kompleksitasnya O(n).
 1-9 Select   0 Everyone   n Name speaker   r Reset   t Transcript   d Debug   q Quit
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
   speaker (useful for tuning, below). `t` hides or shows the transcript
   (it keeps being transcribed and saved either way). `q` quits, after
   transcribing what was still being said.

`--no-tui` gives a one-line status instead of the interactive UI; you can't
select a speaker there, but a named speaker is still auto-selected.
`--no-speakers` skips stages 3–4 entirely (no model, no added latency).
`--passthrough` processes nothing but still shows speech and speakers.
`--no-transcribe` skips stage 5 (no Whisper model), `--no-save` transcribes
without writing files. `--denoise off`, `--no-agc` and `--no-limiter` switch
off the conditioning, and `--passthrough` skips every stage of audio
processing.

### The transcript

Each session writes two files to `~/.focus-ear/sessions/`, named for when
it started (`2026-09-26_0914.md`, `.jsonl`). Every utterance is appended and
flushed to disk as soon as it's transcribed, so a crash loses at most the
sentence in progress.

```markdown
# Lecture transcript: 2026-09-26 09:14

- **Date:** Saturday 26 September 2026, 09:14
- **Speakers:** Prof. Lee
- **Source:** live, MacBook Pro Microphone
- **Model:** mlx-community/whisper-small-mlx (MLX, GPU), language id

**[00:03:58] Prof. Lee:** Oke, jadi hari ini kita pakai loop untuk iterate array.
```

The speaker list in the heading is filled in when you quit. The `.jsonl`
has one record per utterance, for evaluation:

| field | meaning |
|---|---|
| `start`, `end` | seconds since the session (or file) started |
| `speaker_id`, `speaker` | cluster id and its label at the time |
| `text` | Whisper's output; `""` when it found no words (those aren't in the `.md`) |
| `language` | the language decoded as |
| `lang_probs` | top-3 language probabilities, when identification ran (`--language auto`, or `--debug`) |
| `avg_logprob` | token-weighted mean log-probability: closer to 0 is more confident |
| `no_speech_prob`, `compression_ratio` | Whisper's own quality signals |
| `garbled` | Whisper looped (compression ratio over 2.4); shown as *(unclear)* in the `.md` |
| `audio_s`, `queue_s`, `processing_s` | audio length, time waiting for Whisper, time in Whisper |
| `model` | the model and backend |

**Who gets transcribed.** The selected speaker only. While nobody is
selected (`0`, "Everyone"), everyone is, labelled by speaker. With
`--no-speakers`, all speech is, unlabelled.

**Offline evaluation.** `--file recording.wav` runs the same chain (VAD,
speaker identification, utterances, Whisper) over a recording and writes the
same two files, named `<date>_<time>_<file name>`. Any format libsndfile
reads works (WAV, FLAC, AIFF…); stereo is mixed down. It runs as fast as
the Mac allows (a 35 s test clip took 7 s once the models had loaded), never drops
an utterance, and embeds every voiceprint inline so repeated runs give the
same result. Saved speakers are recognised and auto-selected, exactly as live.

## What to expect

### Latency

The total is roughly 650 ms in your ears:

- about 100 ms from the mic to macOS's output
- 41 ms of noise suppression
- the 250 ms `--lookahead`
- the limiter's 2 ms
- Bluetooth, about 250 ms for the Buds

You'll hear the lecturer directly and again through the Buds; if the Buds
have ANC, turn it on.

At startup the log shows a latency estimate. It's built from the latencies
the devices report, which for Bluetooth may be less than the real delay, so
compare it with the measured `latency` in the header. If denoising is what
pushes the estimate past `--latency-budget` (500 ms), you get a warning
suggesting `--denoise off`. If the estimate is over budget even without
denoising, an info line says so instead, with denoising's share;
`--lookahead` is the bigger lever then.

### Noise suppression: does it help the models, or just your ears?

DeepFilterNet3 runs on the CPU, streamed 10 ms at a time. That takes
2.7–2.9 ms per 32 ms block with one torch thread, against 5.8 ms with
torch's default four, because the frames are too small to split.
`--denoise-scope` decides who gets the denoised audio:

| scope | VAD, embeddings, Whisper | your ears |
|---|---|---|
| `playback` (default) | raw mic | denoised |
| `all` | denoised | denoised |
| `models` | denoised | raw mic (delayed the same 41 ms, so decisions stay in step) |

The default is `playback` because of this measurement. It ran on the
synthesised lecture below with fan, pink and mains-hum noise added (about
−1 dB SNR), through `--file`:

| what the models heard | Whisper avg logprob | speakers found (2 real) | transcript |
|---|---|---|---|
| raw | −0.90 | 5 clusters, 2 shown | mostly right |
| denoised | −4.05 | 1 (student merged into lecturer) | nonsense |
| 50 % mix | −1.41 | 6 clusters, 4 shown | worse than raw |

DeepFilterNet raised the signal-to-noise ratio by 13 dB. But Whisper
handles noise better than it handles enhancement artefacts, and the
suppression stripped exactly the voice detail that told the two speakers
apart. On clean speech it changed little (voiceprint cosine 0.788 →
0.784). The effect on Whisper isn't an artefact of focus-ear's streaming:
the deepfilternet package's own offline `enhance()` gave the same kind of
nonsense on the same clip.

That was synthetic speech in synthetic noise; your hall is the real test.
Record a lecture, then compare the summaries at the end of

```sh
python -m focus_ear --file lecture.wav --denoise off
python -m focus_ear --file lecture.wav --denoise-scope all
```

Each summary covers VAD speech %, speakers shown and clusters founded, how
closely voiceprints matched their speaker, and the mean Whisper avg
logprob. `--denoise-mix 0.5` blends the original back in if full
suppression sounds processed.

### Output level: AGC and limiter

- **AGC** brings the speech you're listening to towards `--agc-target`
  (−23 dBFS RMS).
  - It rises at most 6 dB/s and falls at most 20 dB/s, within −12…+24 dB.
  - It only adapts during speech that plays at full volume. Attenuated
    students and pauses never move it, so room noise isn't pumped up
    between sentences.
- **The limiter** guarantees no sample goes above `--limiter-ceiling`
  (−6 dBFS).
  - A cough or a dropped book is capped at about 17 dB above the speech
    level, instead of blasting.
  - Below the ceiling it's bit-exact, just delayed 2 ms.
  - Its gain recovers over 80 ms.
- **Clipping** is checked at the mic, after the denoiser and at the output.
  Output clipping or a clipping mic is logged as a warning, at most every
  10 s. With the limiter on, the output can't clip.

### Long sessions

**Memory.** A 45-minute soak pushed noisy lecture audio through the whole
chain: denoise, VAD, embeddings, Whisper, gate, gains, AGC and limiter.

- Total memory settled at about 3.1 GB from minute 15 onwards, and stayed
  there.
- Most of that is the models: Whisper 0.5 GB plus its working set, the
  speaker model's Metal pools about 1 GB.
- Without the cap below, it had reached 4.2 GB and was still climbing. The
  cause was MLX, the Whisper runtime, keeping every freed GPU buffer. It's
  now capped at 256 MB, which made Whisper no slower.

Everything else is bounded by construction:

- **Audio buffers:** the rings are fixed at about 3 s, and the denoiser's
  FIFO is capped.
- **Speakers:** at most `--max-speakers`, and unnamed ones expire.
- **Transcripts:** the transcription queue holds at most 4 utterances, and
  one utterance at most `--max-utterance`. The transcript lives on disk,
  not in memory; the UI keeps the last 1000 lines.
- **Log file:** it rotates at 5 MB, keeping 3 old files.

**Health line.** With `--debug`, the log gets a line every 30 s (also shown
in the `d` panel):

```
health: up 1:02:13 | mem 3121 MB (mlx 756 mps 85) | drops: worker 0 in-ovf 0 underrun 0 embed 0 asr 0 |
        queues: in 0 ms out 64 ms asr 0 | clips: mic 0 denoise 0 output 0 | reconnects 2
```

**If something breaks mid-lecture:**

- A stage that throws an error, for example a model failing, doesn't stop
  the audio. That block plays unprocessed, the first error is logged in
  full, and the header shows a red count.
- Bluetooth dropping out is handled as described under Troubleshooting.

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

### Transcription

Measured on an Apple M3 with `whisper-small`, on a synthesised code-mixed
lecture (macOS's Indonesian voice reading lines like "Hari ini kita pakai
loop untuk iterate array", with an English question in between):

- **Speed:** a 10 s utterance transcribes in ~0.4–0.7 s, 20 s in ~0.6 s.
  Text appears about a second after the speaker pauses (the 800 ms gap plus
  Whisper).
- **The audio path doesn't notice.** In a real-time replay through the live
  worker, with speaker embeddings on the GPU too, the worker's worst block
  took 13.5 ms with transcription and 13.6 ms without (budget 32 ms). There
  were no audio drops and no dropped embeddings either way. MLX releases
  Python's lock while it computes: a thread waking every 2 ms was never late
  by more than 3.3 ms during back-to-back 20 s transcriptions.
- **The initial prompt helps, and its wording matters.** The default is a
  spoken sentence of code-mixed lecture Indonesian. It raised the average
  logprob on all three lecturer clips (−0.32 → −0.25, −0.41 → −0.33,
  −0.27 → −0.23) and fixed "Link List" → "linked list". A keyword-list prompt
  was repeated back verbatim when given noise; this sentence returned
  nothing there, where no prompt at all hallucinated "Terima kasih."
- **Language detection on code-mixed speech.** Whisper put Indonesian at
  0.98–1.00 on every lecturer clip, so `id` is safe for the lecturer.
- **`--language id` garbles English speakers.** The English question,
  forced to Indonesian, came out as nonsense ("Maaf, China mesin
  penutupoh…") or a repetition loop. Detected as English, it was perfect.
  This only matters while nobody is selected, since then students get
  transcribed too: use `--language auto` if you want their questions.

Utterances, not fixed windows: an utterance opens at speech onset with
300 ms of the audio before it (so the first syllable isn't clipped) and
closes after `--utterance-gap` of silence or at `--max-utterance`. At the
cap it's cut at the latest pause in the last 5 s, if there is one, so a
word isn't split. Utterances with less than 400 ms of speech are discarded.
Because identification lags speech by about a second, all speech is
buffered and each utterance is attributed when it closes (to whoever most
of it was identified as). Buffering only audio already identified as the
selected speaker would clip the first second of every turn.

These are synthetic voices in a simulated room. Your hall will differ, so
tune with `d` open:

| symptom | what to change |
|---|---|
| One person keeps turning into new speakers | Lower `--cluster-threshold` (0.40) or raise `--embed-window` |
| Two people share one row | Raise `--cluster-threshold` (0.5–0.55) |
| Rows appear for noises | Raise `--min-sightings` |
| The gate chops words | Raise `--gate-hangover-ms`, or lower `--vad-threshold` |
| The gate opens on rustling | Raise `--vad-threshold` to 0.6–0.7 |
| Sentences split mid-thought | Raise `--utterance-gap` (1000–1500 ms) |
| Text arrives too late | Lower `--max-utterance` (10 s) |
| Technical terms misspelled | Put your course's terms in `--initial-prompt`, as a spoken sentence |
| `drops: … transcripts` climbing | Whisper can't keep up: use a smaller `--asr-model` |

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--list-devices` | | Print devices and which ones would be used |
| `--input-device NAME\|INDEX` | built-in mic | Mic to record from |
| `--output-device NAME\|INDEX` | `Mewo` | Output, by case-insensitive name substring |
| `--passthrough` | | No audio processing: no high-pass, denoise, gate, gains, AGC or limiter (analysis still shown) |
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
| `--highpass HZ` | `80` | High-pass against rumble and handling noise; `0` = off |
| `--denoise on\|off` | `on` | DeepFilterNet3 noise suppression |
| `--denoise-mix 0-1` | `1.0` | Blend of denoised and original |
| `--denoise-scope all\|playback\|models` | `playback` | Who hears the denoised audio (see above) |
| `--latency-budget MS` | `500` | Warn at startup if denoising pushes the latency estimate past this |
| `--no-agc` | | No automatic gain control |
| `--agc-target DBFS` | `-23` | Speech level the AGC aims for |
| `--no-limiter` | | No peak limiter |
| `--limiter-ceiling DBFS` | `-6` | No output sample goes above this |
| `--no-transcribe` | | Skip transcription (no Whisper model) |
| `--asr-model REPO\|PATH` | `mlx-community/whisper-small-mlx` | MLX Whisper model, e.g. `…/whisper-medium-mlx`, `…/whisper-large-v3-turbo` |
| `--language id\|en\|auto` | `id` | Language to transcribe as; `auto` detects it per utterance |
| `--initial-prompt TEXT` | a code-mixed lecture sentence | Context for Whisper; `''` for none |
| `--utterance-gap MS` | `800` | Silence that ends an utterance |
| `--max-utterance SEC` | `20` | Longest utterance before it's cut |
| `--no-save` | | Don't write `~/.focus-ear/sessions/` files |
| `--file PATH` | | Transcribe a recording instead of the mic |
| `--buffer-ms MS` | `64` | Output jitter buffer; raise it if you hear dropouts |
| `--latency low\|high\|SEC` | `low` | PortAudio device buffer hint |
| `--allow-speakers` | | Permit loudspeaker output (refused by default: feedback) |
| `--debug` | | Debug panel on; per-second timing, per-utterance language/logprob/speed, and a health line every 30 s in `~/.focus-ear/focus-ear.log` |

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
- **"DOWNGRADE: transcribing with faster-whisper … on the CPU"** in the
  log. mlx-whisper couldn't load (the reason is on the line before), so
  focus-ear fell back to faster-whisper. That's CPU-only on a Mac, several
  times slower, and competes with the audio for CPU. Usually the model
  isn't downloaded yet and there's no internet.
- **"transcription failed to start"** in the header. Neither backend
  loaded; audio still works. The log has the reason.
- **The Buds went to sleep.** The header shows "waiting for 'Mewo'".
  Playback resumes about 2 s after they reconnect; speakers, selection and
  the transcript carry on. The log says how long they were gone.
- **"noise suppression is off: DeepFilterNet couldn't load"** at startup.
  It isn't installed (see Setup), or it's the first run without internet.
  Audio works without it.
- **"the mic is clipping"** in the log. Something loud is close to the
  mic, or its input level in Audio MIDI Setup is too high. Clipping at the
  mic can't be undone later in the chain.
- **Denoised speech sounds watery or robotic.** `--denoise-mix 0.7` blends
  some of the original back in.
- **Ctrl-C.** The first press quits: the sentence being spoken is
  transcribed and the files are closed. Pressing again while it waits for
  Whisper skips the queued sentences, but still closes the files properly.
  Closing the terminal window or `kill` does the same as the first Ctrl-C.
- **The mic works in other apps but focus-ear reports silence.** Give *this*
  terminal microphone permission (see Setup).

## Design

The full sample-rate flow is documented at the top of `pipeline.py`.

```
                        ┌──────────── worker thread ─────────────────────────────┐
built-in mic ─► in_ring ┤ high-pass 80 Hz → DeepFilterNet3 (48 kHz, 41 ms)         │
 (44.1 kHz)             │ analysis: resample to 16 kHz → 512-sample frames         │
 (44.1 kHz)   (callback │   → VAD (silero, ONNX) ─► speech yes/no                 │
               copies)  │   → every 0.25 s: speech frames of the last 1.5 s ──────┼─► speaker-embed thread
                        │                                                         │   ECAPA (MPS) → cluster
                        │   → utterance segmenter ─ utterance ─────────────────────┼─► asr thread (bounded queue)
                        │                                                         │   Whisper (MLX) → .md + .jsonl
                        │ audio: gate → 250 ms delay → speaker gain → AGC → limiter ─► out_ring ─► Buds
                        └─────────────────────────────────────────────────────────┘
          UI (main thread) ◄── SpeakerTracker snapshot, 10×/s (lock never taken by audio callbacks)
          supervisor thread: detects dead streams, probes for the Buds, reconnects
```

- **The audio callbacks only copy** into and out of lock-free
  single-producer/single-consumer rings. No inference, allocation, logging
  or locks.
- **Native-rate streams.** Playback runs at the mic's own 44.1 kHz. The
  models see a 16 kHz copy made by a streaming soxr resampler.
- **Streaming DeepFilterNet.** The deepfilternet package only enhances whole
  recordings. `denoise.py` runs the same trained model incrementally: every
  causal convolution keeps the frames it needs, and every GRU keeps its
  state. Its output matches the package's offline result to 3e-7, 30 ms
  later.
- **The denoiser's resampling is exact.** The 44.1 ↔ 48 kHz trip uses a
  small polyphase filter, not soxr. soxr's streaming resampler holds back a
  wandering amount of audio: over an hour it drifted between 46 and 465
  samples short. Each shortfall would have clicked and shifted the
  denoiser's delay, and it did in testing, about once every 14 s at first.
  The polyphase filter's output is fixed by its input, so the delay stays
  exactly 1795 samples for as long as it runs (a test checks 3 minutes).
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
- **Transcription on its own thread, with a queue of 4 utterances.** The
  worker only appends frames, and once per utterance appends to the queue.
  If Whisper falls behind, the oldest waiting utterance is dropped and
  counted (`drops: … transcripts`), so a slow model never delays audio.
  The model is loaded on that thread too: MLX binds a model to the thread
  that loaded it. If mlx-whisper won't load, faster-whisper takes over on
  the CPU (int8), and the log says so.
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
| `denoise.py` | Streaming DeepFilterNet3, exact polyphase resampler, fixed-delay wrapper |
| `dsp.py` | High-pass, AGC, limiter, clip meters |
| `health.py` | Memory, uptime, drops and queues for the health line |
| `transcript.py` | Utterance segmenter, ASR thread and queue, Markdown/JSONL writer |
| `asr.py` | mlx-whisper backend, faster-whisper fallback |
| `tui.py` | Textual UI |
| `main.py` | CLI, wiring, plain status mode, `--file` mode |

**Extension point** (in `pipeline.py`, not implemented): overlapping-speech
separation, as an `AudioStage` that replaces the speaker gain.

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
- **Utterances:** gap, pre-roll, tail, the 400 ms minimum, the length cap
  cutting at a pause without losing or repeating audio, selected-speaker
  filtering, splitting at a change of speaker, ignoring one misidentified
  window.
- **Transcriber:** drop-oldest when full without ever blocking, waiting in
  `--file` mode, failures, the Markdown/JSONL files, the faster-whisper
  fallback.
- **Denoiser:**
  - streaming output equal to the package's offline `enhance()`
  - more than 20 dB of noise removed
  - a delay fixed to the sample over 3 minutes, with no FIFO under-runs
  - `--denoise-mix 0` equal to the original
  - the three scopes routing audio where they should
  - the resampler flat, with exact counts
- **Audio quality:**
  - high-pass: hum down at least 15 dB, speech untouched
  - limiter: never over the ceiling, bit-exact below it
  - AGC: slow, holds for silence and for attenuated speakers
  - clip counting
- **Reconnects:** a fake sounddevice puts the Buds to sleep and wakes them
  10 times, once failing to reopen. The mic keeps feeding the analysis, no
  threads pile up, and every stream is closed.
- **Shutdown:** quitting mid-utterance still transcribes it, a second
  Ctrl-C skips the wait but closes the files, and signals are counted, not
  raised.
- **TUI:** keys, naming (digits and `q` type into the name box), exit codes,
  the transcript panel and `t`, driven headless with Textual's pilot.
- **Models:** the real ECAPA test, and a `--file` run through the real
  Whisper model checking one utterance per speech span at the right times,
  run when the models have been downloaded.
