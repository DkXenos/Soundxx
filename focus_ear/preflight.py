"""Before a session: everything that could stop it working, in plain words with the fix.

Each check returns a Check: ``ok``, ``warning`` (you can start, but read
this) or ``blocker`` (Start stays disabled). This module is light: it
imports neither torch nor MLX, so the app can run it on every screen
refresh. The mic *signal* check uses the engine's live meter when the
engine is already running (the app's standby engine), so the mic is never
opened twice.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DATA_DIR, SESSIONS_DIR

OK, WARNING, BLOCKER = "ok", "warning", "blocker"

# 16 kHz mono 16-bit: 115 MB an hour. Below MIN_FREE a session could fill the
# disk (macOS misbehaves near zero); below LOW_FREE, warn.
MIN_FREE_GB = 2.0
LOW_FREE_GB = 5.0

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
LIVE_ASR = "mlx-community/whisper-small-mlx"
FINAL_ASR = "mlx-community/whisper-large-v3-turbo"
ECAPA_DIR = DATA_DIR / "models" / "spkrec-ecapa-voxceleb"
DEEPFILTER_DIR = Path.home() / "Library" / "Caches" / "DeepFilterNet" / "DeepFilterNet3"


@dataclass
class Check:
    id: str
    level: str
    title: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def hf_cached(repo: str, hub: Path = HF_HUB) -> bool:
    """True if a snapshot of ``repo`` with weights is in the Hugging Face cache."""
    d = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not d.exists():
        return False
    for snap in d.iterdir():
        for f in snap.iterdir():
            if f.suffix in (".safetensors", ".npz", ".bin") and f.resolve().exists() and f.resolve().stat().st_size > 1e6:
                return True
    return False


def mic_permission() -> str:
    """'authorized', 'denied', 'restricted', 'not_determined' or 'unknown' (no AVFoundation)."""
    try:
        import AVFoundation  # pyobjc
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio)
        return {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(int(status), "unknown")
    except Exception:  # noqa: BLE001 - not installed, or not macOS
        return "unknown"


def is_loudspeaker(name: str) -> bool:
    return "speaker" in name.lower()


def check_devices(devices: list[dict], output: str | None) -> list[Check]:
    from .audio_io import DeviceError, find_device

    out: list[Check] = []
    try:
        idx = find_device(devices, None, "input")
        out.append(Check("mic", OK, "Microphone", f"{devices[idx]['name']}"))
    except DeviceError:
        out.append(Check("mic", BLOCKER, "Microphone", "No built-in microphone was found.",
                         "Check System Settings > Sound > Input. An external mic can be chosen with the CLI's "
                         "--input-device."))
    if output is None:
        out.append(Check("output", WARNING, "Earbuds", "No output chosen: the lecture is recorded and "
                         "transcribed, but nothing plays in your ears.", "Pick your earbuds in the list."))
        return out
    try:
        idx = find_device(devices, output, "output")
    except DeviceError:
        out.append(Check("output", WARNING, "Earbuds", f"{output} isn't connected.",
                         "Take the Buds out of the case, or connect them in the Bluetooth menu. You can start "
                         "anyway: they're picked up as soon as they connect."))
        return out
    name = devices[idx]["name"]
    if is_loudspeaker(name):
        out.append(Check("output", BLOCKER, "Earbuds", f"{name} are loudspeakers: the mic would pick them up "
                         "and howl.", "Choose your earbuds, or no output."))
    else:
        out.append(Check("output", OK, "Earbuds", f"{name} connected"))
    return out


def check_lid(closed: bool | None = None) -> Check:
    from .audio_io import lid_closed
    closed = lid_closed() if closed is None else closed
    if closed:
        return Check("lid", BLOCKER, "Lid", "The lid is closed: macOS switches the built-in mic off.",
                     "Open the lid.")
    return Check("lid", OK, "Lid", "open")


def check_permission(status: str | None = None) -> Check:
    status = status or mic_permission()
    if status == "denied" or status == "restricted":
        return Check("permission", BLOCKER, "Microphone access", "macOS is blocking the microphone for this app.",
                     "System Settings > Privacy & Security > Microphone: switch on Soundxx (or Terminal), "
                     "then restart the app.")
    if status == "not_determined":
        return Check("permission", WARNING, "Microphone access", "macOS will ask for microphone access when the "
                     "recording starts.", "Click Allow when asked.")
    return Check("permission", OK, "Microphone access", "allowed" if status == "authorized" else "not checked")


def check_signal(mic_db: float | None, mic_silent: bool | None) -> Check:
    if mic_silent:
        return Check("signal", BLOCKER, "Microphone signal", "The mic delivers pure silence: macOS is blocking it.",
                     "System Settings > Privacy & Security > Microphone: allow Soundxx (or Terminal), then "
                     "restart the app.")
    if mic_db is None:
        return Check("signal", OK, "Microphone signal", "checked once the engine is ready")
    if mic_db < -85:
        return Check("signal", WARNING, "Microphone signal", f"Very quiet ({mic_db:.0f} dB).",
                     "Check the input level in System Settings > Sound > Input.")
    return Check("signal", OK, "Microphone signal", f"{mic_db:.0f} dB")


def check_models(hub: Path = HF_HUB) -> list[Check]:
    out = []
    live = hf_cached(LIVE_ASR, hub)
    out.append(Check("model_live", OK if live else WARNING, "Live transcription model",
                     "whisper-small is ready" if live else "whisper-small isn't downloaded: it needs the internet "
                     "the first time.", "" if live else "Run scripts/setup.command once while online."))
    final = hf_cached(FINAL_ASR, hub)
    out.append(Check("model_final", OK if final else WARNING, "After-class model",
                     "whisper-large-v3-turbo is ready" if final else "whisper-large-v3-turbo isn't downloaded: "
                     "the after-class transcript needs it.", "" if final else "Run scripts/setup.command once "
                     "while online."))
    ecapa = (ECAPA_DIR / "hyperparams.yaml").exists()
    out.append(Check("model_speaker", OK if ecapa else WARNING, "Speaker model",
                     "ECAPA is ready" if ecapa else "The speaker model isn't downloaded.",
                     "" if ecapa else "Run scripts/setup.command once while online."))
    df = DEEPFILTER_DIR.exists()
    out.append(Check("model_denoise", OK if df else WARNING, "Noise suppression",
                     "DeepFilterNet3 is ready" if df else "DeepFilterNet3 isn't downloaded: no noise suppression.",
                     "" if df else "Run scripts/setup.command once while online."))
    return out


def check_disk(path: Path = SESSIONS_DIR) -> Check:
    p = path
    while not p.exists():
        p = p.parent
    free_gb = shutil.disk_usage(p).free / 1e9
    hours = free_gb * 1e3 / 115
    if free_gb < MIN_FREE_GB:
        return Check("disk", BLOCKER, "Free disk space", f"Only {free_gb:.1f} GB free.",
                     "Free some space (a 90-minute lecture needs about 0.2 GB, but macOS needs room to work).")
    if free_gb < LOW_FREE_GB:
        return Check("disk", WARNING, "Free disk space", f"{free_gb:.1f} GB free.", "Free some space soon.")
    return Check("disk", OK, "Free disk space", f"{free_gb:.0f} GB free (about {min(hours, 999):.0f} hours of "
                 "recording)")


def check_interrupted(sessions: list[str]) -> Check:
    if sessions:
        n = len(sessions)
        return Check("interrupted", WARNING, "Interrupted session", f"{n} session{'s' if n > 1 else ''} stopped "
                     "unexpectedly. The audio up to that moment is saved.", "Open Sessions and click Recover.")
    return Check("interrupted", OK, "Earlier sessions", "all closed properly")


def run_preflight(devices: list[dict] | None, output: str | None, mic_db: float | None = None,
                  mic_silent: bool | None = None, interrupted: list[str] | None = None,
                  permission: str | None = None, lid: bool | None = None) -> list[dict]:
    """Every check, as dicts for the app. ``devices`` None: the device list couldn't be read."""
    checks: list[Check] = []
    if devices is None:
        checks.append(Check("mic", BLOCKER, "Audio devices", "The device list couldn't be read.",
                            "Restart the app. If it persists, restart the Mac (CoreAudio is stuck)."))
    else:
        checks += check_devices(devices, output)
    checks.append(check_lid(lid))
    checks.append(check_permission(permission))
    checks.append(check_signal(mic_db, mic_silent))
    checks += check_models()
    checks.append(check_disk())
    checks.append(check_interrupted(interrupted or []))
    return [c.to_dict() for c in checks]


def can_start(checks: list[dict]) -> bool:
    return not any(c["level"] == BLOCKER for c in checks)
