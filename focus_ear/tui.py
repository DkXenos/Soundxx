"""Stage 4: the interactive terminal UI (Textual), with the Stage 5 transcript panel.

Runs on the main thread and never touches audio. Ten times a second it reads
a SpeakerTracker snapshot (under the tracker's lock, which no audio callback
takes), drains finished transcript lines, and reads a few plain counters.
Whisper runs on its own thread, so the meters keep moving while it works.
Key presses go back through the tracker's methods.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, RichLog, Static

from .health import format_conditioning, health_line
from .pipeline import MIC_SILENT_HINT
from .transcript import clock

if TYPE_CHECKING:
    from .clustering import TrackerSnapshot
    from .main import Session
    from .pipeline import MetricsSnapshot
    from .transcript import TranscriptLine

log = logging.getLogger(__name__)

LEVEL_FLOOR_DB, LEVEL_TOP_DB = -75.0, -15.0


def level_bar(db: float, width: int = 20) -> Text:
    frac = min(1.0, max(0.0, (db - LEVEL_FLOOR_DB) / (LEVEL_TOP_DB - LEVEL_FLOOR_DB)))
    filled = round(width * frac)
    return Text.assemble(("█" * filled, "green"), ("·" * (width - filled), "dim"))


def duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


class FocusEarApp(App):
    TITLE = "focus-ear"
    AUTO_FOCUS = None  # otherwise the (hidden) name box grabs focus and eats the number keys
    CSS = """
    #header { height: auto; padding: 0 1; background: $panel; }
    #speakers { height: 1fr; padding: 1 1 0 1; }
    Screen.-transcript #speakers { height: auto; max-height: 14; }
    #asr-status { height: 1; padding: 0 1; background: $panel; }
    #transcript { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    #debug { height: auto; padding: 0 1; border-top: solid $accent; }
    #name { margin: 0 1; }
    """
    BINDINGS = [
        Binding("1", "select(1)", "Select", key_display="1-9"),
        *[Binding(str(k), f"select({k})", show=False) for k in range(2, 10)],
        Binding("0", "select(0)", "Everyone"),
        Binding("n", "name", "Name speaker"),
        Binding("r", "reset", "Reset"),
        Binding("t", "toggle_transcript", "Transcript"),
        Binding("d", "toggle_debug", "Debug"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", show=False, priority=True),
        Binding("escape", "cancel_name", show=False),
    ]

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.metrics: MetricsSnapshot | None = None
        self.health = ""
        self._next_clip_notice = 0.0

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Static(id="speakers")
        yield Static(id="asr-status")
        yield RichLog(id="transcript", wrap=True, max_lines=1000, auto_scroll=True)
        yield Static(id="debug")
        yield Input(id="name")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#debug").display = self.session.cfg.debug
        self.query_one("#name").display = False
        self.show_transcript(self.session.transcriber is not None)
        # Closing the terminal or `kill` quits like q, so the transcript gets flushed and closed.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                loop.add_signal_handler(sig, self.exit)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        self.set_interval(0.1, self.refresh_view)
        self.set_interval(1.0, self.refresh_metrics)
        self.refresh_view()

    # Keys ----------------------------------------------------------------
    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # While the name box is open, keys type into it instead of driving the app.
        if self.query_one("#name").display and action not in ("quit", "cancel_name"):
            return False
        return True

    def action_select(self, key: int) -> None:
        tracker = self.session.tracker
        if tracker is None:
            self.notify("Speaker detection is off (--no-speakers)", severity="warning")
            return
        self.notify(tracker.select(key), timeout=2)
        self.refresh_view()

    def action_name(self) -> None:
        tracker = self.session.tracker
        label = tracker.snapshot().selected_label if tracker else None
        if label is None:
            self.notify("Select a speaker (1-9) first, then press n to name them", severity="warning")
            return
        box = self.query_one("#name", Input)
        box.value = ""
        box.placeholder = f"Name for {label} (Enter saves, Esc cancels)"
        box.display = True
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        self.action_cancel_name()
        if not name:
            return
        profiles = self.session.tracker.enroll_selected(name)
        if profiles is None:
            self.notify("Nobody is selected any more", severity="warning")
            return
        try:
            self.session.save_profiles(profiles)
        except OSError as exc:
            self.notify(f"Couldn't save speaker profiles: {exc}", severity="error")
            return
        self.notify(f"Saved {name}: they'll be recognised and selected next time")

    def action_cancel_name(self) -> None:
        box = self.query_one("#name", Input)
        box.display = False
        box.value = ""
        self.set_focus(None)

    def action_reset(self) -> None:
        if self.session.tracker is not None:
            self.session.tracker.reset()
            self.notify("Speakers reset (saved speakers kept)", timeout=2)

    def show_transcript(self, show: bool) -> None:
        self.query_one("#asr-status").display = show
        self.query_one("#transcript").display = show
        self.screen.set_class(show, "-transcript")

    def action_toggle_transcript(self) -> None:
        s = self.session
        if s.transcriber is None:
            why = f"it failed to start: {s.asr_error}" if s.asr_error else "--no-transcribe"
            self.notify(f"Transcription is off ({why})", severity="warning")
            return
        self.show_transcript(not self.query_one("#transcript").display)
        self.refresh_view()

    def action_toggle_debug(self) -> None:
        debug = self.query_one("#debug")
        debug.display = not debug.display
        self.refresh_view()

    # Refresh -------------------------------------------------------------
    def refresh_metrics(self) -> None:
        try:
            self._refresh_metrics()
        except NoMatches:
            pass  # a timer tick during shutdown, after the widgets are gone

    def _refresh_metrics(self) -> None:
        self.metrics = self.session.metrics.snapshot()
        self.session.log_metrics(self.metrics)
        clips = self.metrics.clips
        now = time.monotonic()
        if (clips.get("output") or clips.get("mic")) and now >= self._next_clip_notice:
            self._next_clip_notice = now + 10.0
            where = "the mic is clipping (too loud at the source)" if clips.get("mic") else "the output clipped"
            self.notify(f"⚠ {where}; see the log", severity="warning", timeout=5)
        if self.query_one("#debug").display:
            try:
                self.health = health_line(self.session)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never take the UI down
                self.health = f"unavailable ({type(exc).__name__}: {exc})"

    def refresh_view(self) -> None:
        try:
            self._refresh_view()
        except NoMatches:
            pass  # a timer tick during shutdown, after the widgets are gone

    def _refresh_view(self) -> None:
        s = self.session
        if s.engine.error is not None:
            self.exit(return_code=2, message=f"focus-ear: {s.engine.error}")
            return
        snap = None
        if s.tracker is not None:
            for event in s.tracker.drain_events():
                self.notify(event)
            snap = s.tracker.snapshot()
        self.query_one("#header", Static).update(self.render_header(snap))
        self.query_one("#speakers", Static).update(self.render_speakers(snap))
        if s.transcriber is not None:
            log_widget = self.query_one("#transcript", RichLog)
            for line in s.transcriber.drain_lines():  # also while hidden, so it's current when shown
                log_widget.write(self.render_line(line))
            status = self.query_one("#asr-status", Static)
            if status.display:
                status.update(self.render_asr_status())
        debug = self.query_one("#debug", Static)
        if debug.display:
            debug.update(self.render_debug(snap))

    def render_header(self, snap: TrackerSnapshot | None) -> Text:
        s, cfg, m = self.session, self.session.cfg, self.metrics
        engine, pipe = s.engine, s.pipeline
        if engine.state == "running":
            out = f"[green]● {escape(engine.output_name or '')}[/]"
        else:
            out = f"[yellow]○ waiting for {escape(repr(cfg.output_device))} (asleep or disconnected)[/]"
        latency = f"{m.e2e_ms:.0f} ms" if m is not None and m.e2e_ms is not None else "…"
        if pipe.delay_samples:
            latency += f" (incl. {1e3 * pipe.delay_samples / engine.samplerate:.0f} ms lookahead)"
        embed_drops = s.analyzer.dropped if s.analyzer is not None else 0
        line1 = (f"[b]{escape(engine.input_name or '?')}[/] → {out}   latency {latency}   "
                 f"drops: audio {pipe.drop_events} · embeddings {embed_drops}")
        if s.transcriber is not None:
            line1 += f" · transcripts {s.transcriber.dropped}"
        elif s.asr_error:
            line1 += "   [b red]transcription failed to start[/]"

        if pipe.mic_silent:
            return Text.from_markup(f"{line1}\n[b red]⚠ {escape(MIC_SILENT_HINT)}[/]")
        if getattr(pipe, "errors", 0):
            line1 += f"\n[b red]⚠ processing failed {pipe.errors}× ({escape(pipe.last_error or '')}); playing unprocessed[/]"
        ctx = pipe.ctx
        speech = "[b green]SPEECH [/]" if ctx is not None and ctx.is_speech else "[dim]silence[/]"
        if s.gate is None:
            gate = "gate off"
        else:
            gate = "gate open" if s.gate.gain_db > -0.5 else f"gate {s.gate.gain_db:.0f} dB"
        if cfg.passthrough:
            focus = "[yellow]--passthrough: audio is not processed[/]"
        elif snap is None:
            focus = "speaker detection off"
        elif snap.selected_label is None:
            focus = "focus: [b]everyone[/] (passthrough)"
        else:
            focus = f"focus: [b]{escape(snap.selected_label)}[/], others {cfg.attenuation_db:g} dB"
        cond = []
        if getattr(s, "denoiser", None) is not None:
            cond.append(f"denoise {pipe.denoise_scope}")
        if getattr(s, "agc", None) is not None:
            cond.append(f"agc {s.agc.gain_db:+.0f} dB")
        cond = f"   [dim]{' · '.join(cond)}[/]" if cond else ""
        line2 = f"mic {pipe.in_db:4.0f} dB   {speech}   {gate}   {focus}{cond}"
        return Text.from_markup(f"{line1}\n{line2}")

    def render_speakers(self, snap: TrackerSnapshot | None) -> Table | Text:
        if snap is None:
            return Text("Speaker detection is off (--no-speakers).", style="dim")
        table = Table(box=None, expand=True, pad_edge=False, show_edge=False)
        for col, kw in (("key", {"width": 4}), ("speaker", {"ratio": 2}), ("level", {"ratio": 3}),
                        ("", {"width": 8}), ("", {"width": 10}), ("spoken", {"width": 7, "justify": "right"})):
            table.add_column(col, **kw)
        table.add_row("0", ("▶ " if snap.selected_label is None else "  ") + "Everyone (passthrough)",
                      "", "", "", "", style="reverse" if snap.selected_label is None else "")
        for v in snap.speakers:
            table.add_row(
                str(v.key) if v.key <= 9 else "",
                ("▶ " if v.selected else "  ") + v.label + (" ★" if v.enrolled else ""),
                level_bar(v.level_db),
                f"{v.level_db:.0f} dB" if v.level_db > -99 else "",
                "◀ talking" if v.talking else "",
                duration(v.speaking_s),
                style="reverse bold" if v.selected else "",
            )
        if not snap.speakers:
            table.add_row("", Text("Listening… a speaker appears once they've been heard "
                                   f"{self.session.cfg.min_sightings} times", style="dim"), "", "", "", "")
        if snap.unconfirmed:
            table.caption = f"+{snap.unconfirmed} unconfirmed (heard fewer than {self.session.cfg.min_sightings} times)"
        return table

    @staticmethod
    def render_line(line: TranscriptLine) -> Text:
        return Text.assemble(
            (f"{clock(line.start)} ", "dim"), (f"{line.speaker}: " if line.speaker else "", "bold"),
            (line.display_text, "dim italic" if line.garbled else ""))

    def render_asr_status(self) -> Text:
        s = self.session
        tr, seg = s.transcriber, s.segmenter
        parts: list[tuple[str, str] | str] = [("Transcript", "bold")]
        writer = tr.writer
        parts.append((f" → {writer.md_path.name}" if writer is not None else " (not saved: --no-save)", "dim"))
        if seg is not None and seg.recording_s > 0:
            parts.append(("   ● listening", "red"))
        if tr.pending:
            parts.append((f"   ⋯ transcribing{f' ({tr.pending})' if tr.pending > 1 else ''}", "yellow"))
        return Text.assemble(*parts)

    def render_debug(self, snap: TrackerSnapshot | None) -> Text:
        s, m = self.session, self.metrics
        lines = []
        if m is not None:
            stages = " · ".join(f"{name} {mean:.2f}/{mx:.2f}" for name, (mean, mx) in m.stages.items())
            lines.append(f"worker ms/block mean/max (budget {m.block_ms:.0f}): {stages or '-'}   RTF {m.rtf:.3f}")
            e2e = f"{m.e2e_ms:.0f}" if m.e2e_ms is not None else "?"
            lines.append(
                f"latency {e2e} ms ≈ mic {m.in_device_ms:.0f} + queue {m.in_queue_ms:.0f} + proc {m.proc_ms:.1f}"
                f" + lookahead {m.delay_ms:.0f} + out-buf {m.out_buffer_ms:.0f} + out-dev {m.out_device_ms:.0f}"
                f"   glitches/s: underrun {m.out_underruns} underflow {m.out_underflows}"
                f" overflow {m.in_overflows} worker-drop {m.worker_drop_events}")
            ctx = s.pipeline.ctx
            vad_now = ctx.vad_prob if ctx is not None else 0.0
            gain = f"   speaker gain {s.speaker_gain.gain_db:.0f} dB" if s.speaker_gain is not None else ""
            lines.append(f"VAD p={vad_now:.2f} (1 s mean {m.vad_prob_mean:.2f}, speech {m.speech_pct:.0f}%){gain}")
        a = s.analyzer
        if a is not None and snap is not None:
            lines.append(f"embeddings on {s.embed_device}: last {a.last_ms:.1f} ms · {a.embedded} done · "
                         f"{a.dropped} dropped (worker late) · {a.skipped_silent} skipped (mostly silence)")
            sims = "  ".join(f"{label} {v:.2f}{' ✓' if v >= snap.threshold else ''}"
                             for label, v in snap.last_similarities[:6]) or "-"
            lines.append(f"last embedding → {snap.last_assigned or '-'}   cosine: {sims}   "
                         f"(threshold {snap.threshold:.2f})")
        conditioning = format_conditioning(s)
        if conditioning:
            lines.append(conditioning.removeprefix(" | ").replace(" | ", " · "))
        if self.health:
            lines.append(f"health: {self.health}")
        tr, seg = s.transcriber, s.segmenter
        if tr is not None and seg is not None:
            rtf = tr.last_processing_s / tr.last_audio_s if tr.last_audio_s else 0.0
            lines.append(f"asr {tr.description}: last {tr.last_processing_s:.2f} s for {tr.last_audio_s:.1f} s "
                         f"(RTF {rtf:.2f}) · {tr.done} done · {tr.pending} pending · {tr.dropped} dropped "
                         f"(queue full) · {tr.failed} failed · discarded: {seg.too_short} too short, "
                         f"{seg.other_speaker} other speakers")
        return Text("\n".join(lines) or "collecting…")


def run_tui(session: Session) -> int:
    app = FocusEarApp(session)
    app.run()
    return app.return_code or 0
