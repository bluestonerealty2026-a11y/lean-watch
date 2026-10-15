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
    # Which brain chooses the actions:
    #   "local"  - an open vision-language model running on this computer (default, no API key)
    #   "claude" - the Claude API (optional; needs ANTHROPIC_API_KEY and `pip install anthropic`)
    brain: str = field(default_factory=lambda: _env("BRAIN", "local"))
    max_turns: int = 40

    # Local open model (Hugging Face id, or a folder from training/train_sft.py)
    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "ByteDance-Seed/UI-TARS-1.5-7B"))
    adapter: str | None = field(default_factory=lambda: _env("ADAPTER", "") or None)   # LoRA folder, optional
    load_in_4bit: bool = field(default_factory=lambda: _env("4BIT", "0") == "1")          # needs bitsandbytes + NVIDIA GPU
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))                  # auto, cuda, mps or cpu
    # How the model writes points: "abs" (pixels of its resized image: Qwen2.5-VL, UI-TARS-1.5), "rel1000"
    # (0-1000: Qwen2-VL, UI-TARS-2B/7B v1, Qwen3-VL), or "auto" = read it from the model's architecture.
    coords: str = field(default_factory=lambda: _env("COORDS", "auto"))
    style: str = field(default_factory=lambda: _env("STYLE", "box"))                     # how points are written: box or point
    local_max_pixels: int | None = None   # cap on image size the model sees; None keeps the model's own setting
    max_new_tokens: int = 512
    preload: bool = False                 # load the model at start-up instead of on the first task
    history_images: int = 3               # screenshots kept in the conversation
    language: str = "English"             # language of the explanations

    # Claude (optional brain)
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("EFFORT", "medium"))
    max_tokens: int = 16000
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
