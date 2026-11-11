"""The after-class pass over a session's raw recording.

    python -m focus_ear.post SESSION [--step regions|diarize|asr|merge] [--force]

SESSION is a session id (a folder name in ~/.focus-ear/sessions) or a path
to any folder holding an audio.wav. Without --step, the four steps run one
after another, each in a process of its own so each model's memory is
handed back before the next loads (8 GB):

  1. regions  silero VAD over the whole recording       -> post/regions.json
  2. diarize  ECAPA + offline clustering, names         -> post/diarization.json, speakers.npz
  3. asr      whisper-large-v3-turbo, temperature 0      -> post/asr.json
  4. merge    words + turns -> lines                     -> final.json, transcript.json, transcript.md

A step whose output exists is skipped (--force reruns it and everything
after it). Every output is written atomically, so an interrupted run
resumes where it stopped.
"""
