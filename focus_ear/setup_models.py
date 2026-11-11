"""Download every model once, while online, with visible progress. python -m focus_ear.setup_models

After this the app works with no internet (only lecture notes need it).
"""
from __future__ import annotations

import sys
import time


def step(name: str, fn) -> bool:
    print(f"• {name}…", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {name} failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"  ✓ ready ({time.perf_counter() - t0:.0f} s)", flush=True)
    return True


def main() -> int:
    from huggingface_hub import snapshot_download

    from .preflight import FINAL_ASR, LIVE_ASR

    ok = True
    ok &= step(f"Live speech recognition ({LIVE_ASR}, about 0.5 GB)", lambda: snapshot_download(LIVE_ASR))
    ok &= step(f"After-class speech recognition ({FINAL_ASR}, about 1.6 GB)", lambda: snapshot_download(FINAL_ASR))

    def ecapa():
        from .embeddings import EcapaEmbedder
        EcapaEmbedder("cpu")
    ok &= step("Speaker model (ECAPA-TDNN, 80 MB)", ecapa)

    def deepfilter():
        from .denoise import load_model
        load_model()
    ok &= step("Noise suppression (DeepFilterNet3, 8 MB)", deepfilter)
    print("All models are ready; Soundxx now works offline." if ok else
          "Some models are missing: check the internet connection and run setup again.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
