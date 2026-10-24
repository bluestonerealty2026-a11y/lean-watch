"""Pick the brain that decides what to click."""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

from .config import Config
from .controller import Controller
from .errors import ModelError, explain_load_error
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


def adapter_problem(adapter: str) -> str | None:
    """Why --adapter can't be used, or None. Without its folder LeanWatch can't tell the adapter's base model, and would
    quietly fall back to the default one."""
    folder = Path(adapter).expanduser()
    if not folder.is_dir():
        if adapter.count("/") == 1 and not adapter.startswith((".", "/", "~")):     # looks like a Hugging Face repo
            return (f"--adapter takes a folder, and there is none at {folder}. If {adapter} is a Hugging Face repo, "
                    f"download it first: hf download {adapter} --local-dir ~/lw-runs/{adapter.split('/')[-1]}")
        return (f"There is no adapter folder at {folder}. Point --adapter at the folder that has adapter_config.json "
                "and leanwatch.json in it.")
    if (folder / "adapter_config.json").exists():
        return None
    inner = sorted(folder.glob("*/adapter_config.json"))
    return f"{folder} has no adapter_config.json, so it isn't an adapter folder." + (
        f" The adapter is one folder down: --adapter {inner[0].parent}" if inner else "")


def trained_settings(adapter: str | None, model: str | None = None) -> dict:
    """Settings saved with a fine-tuned model: from the adapter folder, else from a merged model folder."""
    if adapter:
        return adapter_settings(adapter)
    if model and (Path(model) / "leanwatch.json").exists():
        return json.loads((Path(model) / "leanwatch.json").read_text())
    return {}


def resolve_model_id(model: str | None, adapter: str | None) -> str:
    """The model to load: the one asked for, else the adapter's base, else the default."""
    return model or adapter_settings(adapter).get("base_model") or DEFAULT_LOCAL_MODEL


def load_local_model(cfg: Config, status=None):
    """Start the local model.

    By default it loads in its own process (model_process.py) and this returns at once; its wait_ready()
    or first answer() waits for the load. With cfg.model_process off it loads here, before returning.
    status: also told which weight precision was chosen and why (that line is always printed).
    """
    def note(text: str) -> None:
        print(f"  {text}", flush=True)
        if status:
            status(text)

    options = dict(adapter=cfg.adapter, device=cfg.device, load_in_4bit=cfg.load_in_4bit,
                   max_pixels=cfg.local_max_pixels, max_new_tokens=cfg.max_new_tokens, cpu_dtype=cfg.cpu_dtype)
    if cfg.model_process:
        from .model_process import ModelProcess

        return ModelProcess(cfg.local_model, status=note, **options)
    from .local_model import LocalModel

    try:
        return LocalModel(cfg.local_model, status=note, **options)
    except Exception as err:
        traceback.print_exc()
        raise ModelError(explain_load_error(err, cfg.local_model, cfg.cpu_dtype)) from err


def make_brain(cfg: Config, controller: Controller, hooks: Hooks, stop: threading.Event, local_model=None):
    if cfg.brain == "claude":
        try:
            from .brain import ClaudeBrain
        except ModuleNotFoundError as err:
            if err.name != "anthropic":
                raise
            raise RuntimeError("The Claude brain needs the anthropic package: "
                               "pip install -r requirements-claude.txt (or run the default local brain).") from None

        return ClaudeBrain(cfg, controller, hooks, stop)
    if cfg.brain != "local":
        raise ValueError(f"Unknown brain {cfg.brain!r}. Use 'local' or 'claude'.")
    from .local_brain import LocalBrain

    return LocalBrain(cfg, controller, hooks, stop, local_model or load_local_model(cfg))


def model_name(cfg: Config) -> str:
    if cfg.brain == "claude":
        return cfg.claude_model
    return cfg.local_model + (f" + {cfg.adapter}" if cfg.adapter else "")
