"""Console helpers for running task scripts on Windows terminals."""

from __future__ import annotations

import sys


def configure_utf8_output() -> None:
    """Prefer UTF-8 for stdout/stderr so Unicode logs do not crash on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
