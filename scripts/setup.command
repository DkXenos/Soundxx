#!/bin/zsh
# Double-click once, while online. Sets up everything Soundxx needs:
#   1. a Python environment (venv) with the pinned packages
#   2. every model, downloaded with progress (about 3 GB), so class works offline
#   3. a self-test (the unit tests, about 30 s)
#   4. Soundxx.app in this folder (drag it to the Dock)
# Safe to run again: it only adds what's missing.
cd "${0:A:h}/.." || exit 1
REPO="$PWD"
step() { print -P "\n%B$1%b" }
fail() { print -P "\n%F{red}%BSetup stopped:%b $1%f"; read -k1 "?Press any key to close."; exit 1 }

step "1/4  Python environment"
PY=""
for c in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 python3.11; do
  if command -v $c >/dev/null 2>&1; then PY=$c; break; fi
done
[[ -x venv/bin/python ]] || { [[ -n $PY ]] || fail "Python 3.11 isn't installed. Install it with: brew install python@3.11"; $PY -m venv venv || fail "couldn't create the venv"; }
venv/bin/python -m pip install -q --upgrade pip >/dev/null
venv/bin/pip install -q -r requirements.txt || fail "installing requirements.txt failed (internet?)"
venv/bin/pip install -q --no-deps -r requirements-denoise.txt || fail "installing requirements-denoise.txt failed"
venv/bin/pip install -q -r requirements-app.txt || fail "installing requirements-app.txt failed"
echo "✓ packages installed"

step "2/4  Models (downloaded once)"
venv/bin/python -m focus_ear.setup_models || fail "a model couldn't be downloaded"

step "3/4  Self-test"
venv/bin/python -m unittest discover -s tests 2>&1 | tail -3
[[ ${pipestatus[1]} -eq 0 ]] || print -P "%F{yellow}Some tests failed: Soundxx may still work; the output above says which.%f"

step "4/4  Soundxx.app"
scripts/build_app.sh "$REPO" || fail "couldn't build the app"
open -R "$REPO/Soundxx.app"
print -P "\n%BDone.%b Drag Soundxx.app (shown in Finder) to the Dock, then open it."
print "The first time, macOS asks for the microphone: click Allow."
print "If the window never asks and the mic stays silent, use scripts/soundxx.command instead (it runs under Terminal's permission)."
read -k1 "?Press any key to close."
