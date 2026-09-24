"""Interactive terminal UI (phase 3, not implemented yet).

Plan: a Textual app (Textual rather than plain rich, because 'n' needs a text
input for the speaker's name). It polls the pipeline's state ~15 times a
second and never touches audio. Shows the speakers with live level meters,
the active speaker and the selected one (highlighted).
Keys: 1-9 select, 0 passthrough, r reset clustering, n name the selected
speaker, q quit.
"""
from __future__ import annotations


def run_tui(engine, pipeline) -> int:
    raise NotImplementedError("the TUI arrives in phase 3")
