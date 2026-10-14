"""Pick the brain that decides what to click."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import Config
from .controller import Controller
from .hooks import Hooks


DEFAULT_LOCAL_MODEL = "ByteDance-Seed/UI-TARS-1.5-7B"


def adapter_settings(adapter: str | None) -> dict:
    """What train_sft.py recorded about an adapter: base model, coordinate format, image size cap."""
    if not adapter:
        return {}
    folder = Path(adapter)
    info = {}
    if (folder / "adapter_config.json").exists():
        info["base_model"] = json.loads((folder / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if (folder / "leanwatch.json").exists():
        info.update(json.loads((folder / "leanwatch.json").read_text()))
    return {k: v for k, v in info.items() if v is not None}


def resolve_model_id(model: str | None, adapter: str | None) -> str:
    """The model to load: the one asked for, else the adapter's base, else the default."""
    return model or adapter_settings(adapter).get("base_model") or DEFAULT_LOCAL_MODEL


def load_local_model(cfg: Config):
    from .local_brain import LocalModel

    return LocalModel(cfg.local_model, adapter=cfg.adapter, device=cfg.device, load_in_4bit=cfg.load_in_4bit,
                      max_pixels=cfg.local_max_pixels, max_new_tokens=cfg.max_new_tokens)


def make_brain(cfg: Config, controller: Controller, hooks: Hooks, stop: threading.Event, local_model=None):
    if cfg.brain == "claude":
        from .brain import ClaudeBrain

        return ClaudeBrain(cfg, controller, hooks, stop)
    if cfg.brain != "local":
        raise ValueError(f"Unknown brain {cfg.brain!r}. Use 'local' or 'claude'.")
    from .local_brain import LocalBrain

    return LocalBrain(cfg, controller, hooks, stop, local_model or load_local_model(cfg))


def model_name(cfg: Config) -> str:
    if cfg.brain == "claude":
        return cfg.claude_model
    return cfg.local_model + (f" + {cfg.adapter}" if cfg.adapter else "")
