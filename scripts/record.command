#!/bin/zsh
# Double-click to record a lecture: listen mode (denoise, gate, level into
# the Buds), live transcript, raw mic saved. Works without the Buds too: it
# records and transcribes, and starts playing as soon as they connect.
# Quit with q (or close this window): the recording and transcript are kept
# in ~/.focus-ear/sessions/<date>_<time>/.
cd "${0:A:h}/.." || exit 1
if [[ ! -x venv/bin/python ]]; then
  echo "Not set up yet: double-click scripts/setup.command first (needs the internet once)."
  read -k1 "?Press any key to close."
  exit 1
fi
caffeinate -i -w $$ &  # no idle sleep during the lecture (closing the lid still sleeps: the mic is off then)
venv/bin/python -m focus_ear --mode listen "$@"
code=$?
echo
echo "Saved in ~/.focus-ear/sessions (newest folder). Open the Soundxx app to process and review it."
read -k1 "?Press any key to close this window."
exit $code
