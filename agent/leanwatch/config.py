"""Settings for the local agent. Every field can be overridden on the command
line (see ``python -m leanwatch --help``) or with a LEANWATCH_* variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"LEANWATCH_{name}", default)


@dataclass
class Config:
    # Claude
    model: str = field(default_factory=lambda: _env("MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("EFFORT", "medium"))
    max_tokens: int = 16000
    max_turns: int = 40
    # Optional API betas: progress notes between tool calls, and server-side
    # fallback to another model if a request is declined by a safety filter.
    progress_updates: bool = True
    fallbacks: bool = True

    # Local server
    host: str = "127.0.0.1"
    port: int = field(default_factory=lambda: int(_env("PORT", "8765")))
    open_browser: bool = True

    # Screen and motion
    screenshot_long_edge: int = 1366
    screenshot_max_pixels: int = 1_100_000
    move_speed: float = 1.0
    takeover_tolerance_px: int = 30

    # Safety
    dry_run: bool = False
    confirm_timeout_s: float = 180.0

    @property
    def web_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "web"

    @property
    def session_file(self) -> Path:
        return Path.home() / ".leanwatch" / "session.json"
