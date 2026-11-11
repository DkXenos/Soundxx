"""Integration check of the live engine subprocess on this Mac's real mic.

    venv/bin/python scripts/check_headless.py [--output ZoomAudioDevice] [--kill]

Starts focus_ear.headless, records into a session in a temporary folder,
pauses and resumes, then stops it with "stop" (or kill -9 with --kill) and
checks what was left behind: session.json, audio.wav length (pauses
excluded), live transcript, recovery of an interrupted session.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from focus_ear.sessions import SessionStore, wav_seconds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="ZoomAudioDevice")
    ap.add_argument("--kill", action="store_true", help="kill -9 instead of a clean stop, then recover")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="focus-ear-check-")) / "sessions"
    tmp.mkdir()
    os.environ["FOCUS_EAR_SESSIONS"] = str(tmp)  # the engine's settings go there too, not to ~/.focus-ear
    store = SessionStore(tmp)
    cmd = [sys.executable, "-m", "focus_ear.headless", "--mode", "listen"]
    if args.output:
        cmd += ["--output", args.output]
    p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    msgs: list[dict] = []
    ready = threading.Event()

    def reader():
        for raw in p.stdout:
            try:
                m = json.loads(raw)
            except ValueError:
                print("NOT JSON on stdout:", raw[:200])
                continue
            msgs.append(m)
            if m["type"] in ("ready", "fatal"):
                ready.set()
    threading.Thread(target=reader, daemon=True).start()

    def send(**c):
        p.stdin.write(json.dumps(c) + "\n")
        p.stdin.flush()

    t0 = time.monotonic()
    if not ready.wait(120):
        print("FAIL: engine not ready in 120 s")
        p.kill()
        return 1
    if msgs[-1]["type"] == "fatal":
        print("FAIL: fatal:", msgs[-1])
        return 1
    print(f"ready after {time.monotonic() - t0:.1f} s:", {k: msgs[-1][k] for k in ("input", "output", "models")})
    info = store.create(title="headless check", course="test")
    send(cmd="start", session_dir=str(store.path(info["id"])))
    rec_start = time.monotonic()
    time.sleep(args.seconds / 3)
    send(cmd="set_mode", mode="full")
    send(cmd="pause")
    time.sleep(3.0)
    send(cmd="resume")
    send(cmd="set_mode", mode="listen")
    time.sleep(args.seconds * 2 / 3)
    elapsed = time.monotonic() - rec_start
    states = [m for m in msgs if m["type"] == "state"]
    phases = [s["phase"] for s in states]
    print("phases seen:", sorted(set(phases)), "states:", len(states), f"({len(states) / (time.monotonic() - t0):.1f}/s overall)")
    last = states[-1]
    print("last state:", {k: last[k] for k in ("phase", "mode", "output", "output_state", "mic_db", "recording_s",
                                              "latency_ms", "drops", "errors", "mic_silent")})
    events = [m for m in msgs if m["type"] == "event"]
    for e in events:
        print("  event:", e["level"], e["message"][:140])
    if args.kill:
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        data = store.load(info["id"])
        print("after kill -9: status", data["status"], "| interrupted list:", store.interrupted())
        data = store.recover(info["id"])
        print("recovered:", {k: data[k] for k in ("status", "interrupted", "duration_s", "ended")})
    else:
        send(cmd="stop")
        p.wait(60)
        data = store.load(info["id"])
        print("stopped: exit", p.returncode, {k: data[k] for k in ("status", "duration_s", "pauses", "speakers")})
    folder = store.path(info["id"])
    audio = wav_seconds(folder / "audio.wav")
    print(f"files: {sorted(x.name for x in folder.iterdir())}")
    print(f"audio.wav {audio:.1f} s for {elapsed:.1f} s wall clock with a 3 s pause "
          f"(expect about {elapsed - 3:.1f})")
    ok = abs(audio - (elapsed - 3)) < 1.5 and data["status"] == "recorded"
    print("OK" if ok else "FAIL", "| session folder:", folder)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
