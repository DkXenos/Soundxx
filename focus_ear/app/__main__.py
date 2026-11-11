"""Soundxx: the app. python -m focus_ear.app [--browser] [--port N]

Starts the local server on 127.0.0.1 and opens it in a native window
(pywebview, WebKit). --browser opens your web browser instead, for when
the window can't start. Closing the window while a lecture is being
recorded asks first, then stops the recording cleanly.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser

from ..config import DATA_DIR

log = logging.getLogger("focus_ear.app")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("focus_ear")
    root.setLevel(logging.INFO)
    h = logging.handlers.RotatingFileHandler(DATA_DIR / "app.log", maxBytes=2 * 2**20, backupCount=2)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(h)
    s = logging.StreamHandler(sys.stderr)
    s.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(s)


def serve(app, port: int) -> "uvicorn.Server":  # noqa: F821
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio"
                            if _has("websockets") else "auto", lifespan="on")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="server", daemon=True).start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="soundxx")
    p.add_argument("--browser", action="store_true", help="open in the web browser instead of a window")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-open", action="store_true", help="only serve (prints the URL)")
    args = p.parse_args(argv)
    setup_logging()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from .server import create_app
    port = args.port or free_port()
    token = secrets.token_urlsafe(24)
    app = create_app(token=token, port=port)
    server = serve(app, port)
    url = f"http://127.0.0.1:{port}/#token={token}"
    log.info("Soundxx at http://127.0.0.1:%d", port)
    engine = app.state.engine

    def shutdown() -> None:
        engine.stop(60)
        app.state.jobs.shutdown()
        server.should_exit = True

    def on_term(signum, frame) -> None:
        # Logout or kill: close the recording and stop the job properly, then leave.
        threading.Thread(target=lambda: (shutdown(), logging.shutdown(), os._exit(0)), daemon=True).start()

    import signal
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)

    if args.no_open:
        print(url, flush=True)
    if args.browser or args.no_open:
        if args.browser:
            webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        shutdown()
        return 0

    import webview

    class Api:
        """Called from the page as window.pywebview.api.*"""

        def pick_audio(self):
            kinds = ("Audio (*.wav;*.m4a;*.mp3;*.flac;*.ogg;*.aac;*.mp4)", "All files (*.*)")
            picked = window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False, file_types=kinds)
            return picked[0] if picked else None

    window = webview.create_window("Soundxx", url, width=1280, height=820, min_size=(900, 600),
                                   background_color="#EEF2F5", text_select=True, js_api=Api())

    def on_closing():
        if engine.recording:
            ok = window.create_confirmation_dialog(
                "A lecture is being recorded",
                "Stop the recording and quit? Everything recorded so far is saved.")
            if not ok:
                return False
        threading.Thread(target=shutdown, daemon=True).start()
        return True

    window.events.closing += on_closing
    webview.start(private_mode=False, storage_path=str(DATA_DIR / "webview"))
    # The window is gone: make sure a recording was closed properly before the process ends.
    engine.stop(60)
    app.state.jobs.shutdown()
    server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
