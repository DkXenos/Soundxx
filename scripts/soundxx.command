#!/bin/zsh
# Soundxx without the app bundle: the same app, started from Terminal (so it
# uses Terminal's microphone permission). Use it if Soundxx.app can't get the mic.
cd "${0:A:h}/.." || exit 1
exec venv/bin/python -m focus_ear.app "$@"
