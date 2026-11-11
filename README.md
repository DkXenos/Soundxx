# Soundxx

A lecture companion for a MacBook and a pair of Galaxy Buds. During class it plays the room into
your earbuds, cleaned up, with the person you pick louder; it records, and writes down who says
what as they say it. After class it makes a careful transcript from the recording, lets you fix
names and words while listening, and writes lecture notes.

Everything except the notes runs on the Mac, offline. The notes are written by Claude from the
transcript **text**; the recording never leaves the Mac.

## Setting up (once, online)

1. Install Python 3.11 if you don't have it: `brew install python@3.11`.
2. Double-click **`scripts/setup.command`**. It installs the packages, downloads the models
   (about 3 GB, with progress), runs the self-test, and builds **Soundxx.app** in this folder.
3. Drag Soundxx.app to the Dock.
4. For lecture notes: Claude Code must be installed and signed in (`claude` in Terminal).

The first time you record, macOS asks whether Soundxx may use the microphone: click **Allow**.
If it never asks and the mic stays silent, start Soundxx with `scripts/soundxx.command` instead
(it runs under Terminal's microphone permission).

## In class

Open Soundxx. The **Record** screen shows the microphone level moving, where the sound will play,
and anything that would stop a recording, each with its fix. While you look at it, the models
load, so **Start recording** is ready in about five seconds.

- **Play into**: your Buds. MacBook speakers are listed but can't be chosen (the mic would hear
  them and howl). "No earbuds" records and transcribes without playing anything. If the Buds
  aren't connected yet you can start anyway; they're picked up when they connect.
- **In your ears**:
  - **Listen to everyone**: the whole room, denoised and levelled, with the least delay (about
    0.18 s before Bluetooth).
  - **Focus on a speaker**: the person you select is played at full volume and everyone else
    20 dB down. This costs about 0.25 s more delay.
- **Lecture** and **Course** are optional; they name the session and group it later.

While recording:

- **Speakers** appear on the left after a few seconds of speech, with a level bar and how long
  they've talked; the one talking now is highlighted. Click someone to focus on them (this switches
  to Focus). **Everyone** goes back to the whole room.
- **Name** a speaker (for example the lecturer). With "Remember this voice" ticked, they're
  recognised by name in every later lecture.
- **The transcript** runs on the right, a few seconds behind, labelled with who spoke. It's rough;
  the careful one comes after class.
- **Pause** stops recording, transcribing and playback (for a break); the level meter keeps
  moving. **Stop recording** ends the session.
- The status line says when the Buds are connected or being waited for, when the mic is too loud
  or silent, and when the transcript is falling behind.

Soundxx keeps the Mac from going to sleep while it records. Don't close the lid: with the lid
closed, macOS switches the built-in microphone off.

If anything crashes, the recording up to that moment is kept. The next time you open Soundxx it
offers to **Recover** the session.

## After class

When you stop, Soundxx makes the **careful transcript** in the background. For a 90-minute lecture
this takes about [see below] and needs no internet. It:

1. finds every stretch of speech in the recording;
2. works out who spoke when, listening to the whole lecture at once (much steadier than the live
   labels). People you saved are named; otherwise the person who talks most is **Dosen** and the
   others **Mahasiswa 1, 2, …**;
3. transcribes with a larger speech model (whisper-large-v3-turbo), deciding per turn whether it's
   Indonesian or English, and marks lines the model wasn't sure about;
4. then, with an internet connection, writes **notes**: a summary, key points, chapters, terms,
   tasks and deadlines, and the questions that were asked. Without internet they're queued and
   written the next time Soundxx opens online. Lectures with almost no speech get no automatic notes.

If you start recording while a lecture is still being processed, processing pauses and carries on
afterwards from where it stopped.

## Reviewing a lecture

**Sessions** lists every lecture, newest first, filterable by course. Open one:

- The transcript and the notes sit side by side, with the recording in a player at the bottom.
  Click a time, a chapter or a question to hear that moment; the line being played is highlighted
  and followed. Space plays and pauses, the arrow keys skip 5 s, and the speed button cycles
  1×–2×.
- Lines the speech model wasn't sure about are underlined with a wave.
- **Fixing it**, all of it undoable (the Undo button undoes the last change):
  - a speaker's **Change** menu: rename them; say they're the same person as another speaker
    (merge); or **Remember this voice** so they're named automatically from now on;
  - on a line: **Edit** (or double-click) to correct the words, and **Someone else said this** to
    move it to another or a new speaker.
- **Write the notes again** after correcting the transcript, so the notes use your corrections.
- **Export transcript / Export notes** save Markdown files; **Show in Finder** opens the session's
  folder.
- **Bring in a recording** (on the Sessions screen) imports an audio file you already have (wav,
  m4a, mp3) as a session and processes it.

Everything lives in `~/.focus-ear/sessions/<date>_<time>_<title>/`; the Markdown files there open
in any editor. Deleting a session moves it to the Trash.

## What's measured, and what isn't yet

See `eval/results/` for the tables. In short (November 2026):

- **Live, on this Mac (MacBook mic to a virtual output, no Bluetooth):** about 160–180 ms in
  Listen and 400–450 ms in Focus, before Bluetooth; no dropouts in the runs logged.
  `scripts/measure_latency.py` measures the real figure into your ear, Bluetooth included (hold a
  Bud to the mic; instructions in the script). **Not yet measured with the Buds.**
- **The after-class pass** was developed and tuned on a *synthetic* lecture (macOS voices, simulated
  reverb and room noise), because no real recording existed yet. The numbers in
  `eval/results/` say whether the machinery works and which settings are better there; they don't
  predict a real hall.
- **The single most useful next step is a real lecture**: record one, correct 10–15 minutes of it
  in the review screen, then run
  `venv/bin/python -m focus_ear.eval reference <session> --from <sec> --to <sec>` and
  `venv/bin/python -m focus_ear.eval run eval/cases/<session>`. That re-checks every default on
  your actual hall.

## Privacy

- Audio, voiceprints and transcripts stay in `~/.focus-ear/` on this Mac.
- Notes: the transcript text (line numbers, times, speaker names, words) is sent to Claude through
  your Claude Code login. Nothing else is sent. Turn automatic notes off with
  `"auto_notes": false` in `~/.focus-ear/settings.json`.
- Students are labelled "Mahasiswa N" per session; a voiceprint is saved only when you choose
  "Remember this voice".
- Record people only with their permission.

## Troubleshooting

| What you see | What to do |
|---|---|
| "The mic delivers pure silence" | System Settings > Privacy & Security > Microphone: allow Soundxx (or Terminal), then restart Soundxx. |
| "The lid is closed" | Open it: the built-in mic is off while the lid is closed. |
| "Waiting for Mewo" | Take the Buds out of the case or connect them in the Bluetooth menu. The recording carries on meanwhile. |
| Dropouts in the Buds | Switch to Listen; if it persists, close other heavy apps. The recording itself isn't affected. |
| "Transcript is behind" | Harmless: the live transcript catches up, and the careful one is made from the recording anyway. |
| Processing stopped with an error | Open the session and press Process again; it resumes from the last finished step. The log is `post/post.log` in the session folder. |
| Notes "waiting for an internet connection" | They're written automatically next time Soundxx opens online, or press Write notes. |
| Anything else | `~/.focus-ear/app.log` and the session's `engine.log`. |

## For tinkering

- The engine still runs on its own: `scripts/record.command` (double-click: listen mode in Terminal,
  records into a session folder), or `venv/bin/python -m focus_ear --help`. Its design, options and
  how each default was chosen: [docs/ENGINEERING.md](docs/ENGINEERING.md).
- `python -m focus_ear.post <session>`: the after-class pass by hand (`--from asr` reruns from a
  step, `--cluster-threshold`, `--en-threshold`).
- `python -m focus_ear.notes <session>`: notes by hand.
- `python -m focus_ear.eval run <case>…`: the evaluation; `python -m focus_ear.eval.synth <dir>`
  makes a synthetic test lecture; `python -m focus_ear.eval.separation`: the overlapping-speech
  experiment.
- Tests: `venv/bin/python -m unittest discover -s tests`.
- `STATUS.md` is the audit of the engine before the app (29 September 2026).
