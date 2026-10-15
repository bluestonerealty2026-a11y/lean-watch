"""Command line entry point: ``python -m leanwatch``."""

from __future__ import annotations

import argparse
import os
import sys
import threading

from .config import Config
from .native import DEFAULT_HOTKEY

SMALL_MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"


def main(argv: list[str] | None = None) -> None:
    cfg = Config()
    p = argparse.ArgumentParser(
        prog="python -m leanwatch",
        allow_abbrev=False,          # --autostart saves the other options; an abbreviation would slip through
        description="Let an AI model move your mouse and keyboard while you watch and learn. "
                    "Starts a floating bubble that stays on top of every app: click it and type what you need.",
    )
    p.add_argument("--port", type=int, default=cfg.port, help=f"local port (default {cfg.port})")
    p.add_argument("--web", action="store_true", help="also open the web control center in your browser")
    p.add_argument("--no-bubble", action="store_true", help="no floating bubble: use the web control center only")
    p.add_argument("--no-browser", action="store_true", help="with --no-bubble: don't open the browser")
    p.add_argument("--overlay", action="store_true", help=argparse.SUPPRESS)   # old name; the bubble is now the default
    p.add_argument("--hotkey", default=DEFAULT_HOTKEY, help=f"keyboard shortcut that opens the box from anywhere (default {DEFAULT_HOTKEY}; 'none' to turn off)")
    p.add_argument("--autostart", choices=["on", "off"], help="start LeanWatch (with the other options you give) every time you log in, or stop doing that")
    p.add_argument("--open", action="store_true", help="open the box of the LeanWatch that is running (for a desktop keyboard shortcut)")
    p.add_argument("--shortcut", choices=["on", "off"], help="GNOME: add (or remove) a keyboard shortcut in GNOME's settings that opens the box from any app; needed on Wayland")
    p.add_argument("--preload", action="store_true", help="load the model at start-up instead of on the first task")
    p.add_argument("--dry-run", action="store_true", help="plan and narrate, but never move the mouse or press keys")
    p.add_argument("--brain", default=cfg.brain, choices=["local", "claude"], help="local: open model on this computer (default); claude: Claude API, needs a key")
    p.add_argument("--model", help=f"local brain: Hugging Face id or folder of your fine-tuned model (default {cfg.local_model}); claude brain: model id")
    p.add_argument("--adapter", default=cfg.adapter, help="folder of a LoRA adapter from training/train_sft.py")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true", default=cfg.load_in_4bit, help="load the local model in 4-bit (NVIDIA GPU, needs bitsandbytes)")
    p.add_argument("--device", default=cfg.device, choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--coords", default=None, choices=["auto", "abs", "rel1000"], help="how the model writes points: abs (Qwen2.5-VL, UI-TARS-1.5) or rel1000 (Qwen2-VL, UI-TARS 2B, Qwen3-VL). Default: worked out from the model")
    p.add_argument("--small", action="store_true", help=f"use the small model ({SMALL_MODEL}, about 9 GB of memory on a CPU) for laptops without a big GPU")
    p.add_argument("--check", action="store_true", help="check whether this computer is ready (screen, libraries, GPU, memory) and which model fits, then exit")
    p.add_argument("--style", default=None, choices=["box", "point"], help="how points are written in the prompt: box (start_box=..., default) or point (point='<point>x y</point>'). Read from the adapter when you pass one")
    p.add_argument("--max-pixels", type=int, default=cfg.local_max_pixels, help="largest screenshot size, in pixels, the local model sees")
    p.add_argument("--language", default=None, help="language for the explanations (default English, or what your fine-tuned model was trained with)")
    p.add_argument("--effort", default=cfg.effort, choices=["low", "medium", "high", "xhigh", "max"], help="claude brain only: how hard it thinks per step")
    p.add_argument("--speed", type=float, default=cfg.move_speed, help="pointer speed multiplier (default 1.0)")
    p.add_argument("--cli", metavar="TASK", help="run one task in this terminal, without the web page")
    args = p.parse_args(argv)

    cfg.port = args.port
    cfg.dry_run = args.dry_run
    cfg.brain = args.brain
    if args.small and not args.model:
        args.model = SMALL_MODEL
    if args.model:
        if cfg.brain == "claude":
            cfg.claude_model = args.model
        else:
            cfg.local_model = args.model
    cfg.adapter = args.adapter
    if cfg.brain == "local":
        from .brains import resolve_model_id, trained_settings

        # Which model: --model, else LEANWATCH_LOCAL_MODEL, else the adapter's base model, else the default.
        if cfg.adapter and not args.model and not os.environ.get("LEANWATCH_LOCAL_MODEL"):
            cfg.local_model = resolve_model_id(None, cfg.adapter)
        # A model you fine-tuned remembers how its data was written (train_sft.py saves leanwatch.json
        # next to the adapter and the merged model). Use the same settings, unless a flag says otherwise.
        trained = trained_settings(cfg.adapter, cfg.local_model)
        cfg.language = trained.get("language", cfg.language)
        cfg.coords = trained.get("coords", cfg.coords)
        cfg.style = trained.get("style", cfg.style)
        cfg.history_images = trained.get("history_images", cfg.history_images)
        cfg.local_max_pixels = trained.get("max_pixels", cfg.local_max_pixels)
    cfg.load_in_4bit = args.load_in_4bit
    cfg.device = args.device
    if args.coords:
        cfg.coords = args.coords
    if args.style:
        cfg.style = args.style
    if args.max_pixels:
        cfg.local_max_pixels = args.max_pixels
    if args.language:
        cfg.language = args.language
    cfg.effort = args.effort
    cfg.move_speed = args.speed
    cfg.preload = args.preload

    if args.check:
        from .doctor import report

        raise SystemExit(report(cfg))

    if args.open:
        from .open_box import open_box

        raise SystemExit(open_box())

    if args.shortcut:
        from . import gnome_shortcut

        if args.shortcut == "off":
            print(gnome_shortcut.remove())
            return
        combo = args.hotkey if args.hotkey.strip().lower() not in ("", "none", "off") else DEFAULT_HOTKEY
        try:
            print(gnome_shortcut.install(combo))
        except Exception as err:
            raise SystemExit(f"Could not add the shortcut: {err}")
        return

    if args.autostart:
        from .autostart import Autostart

        auto = Autostart()
        if args.autostart == "off":
            print(auto.remove())
        else:
            keep = strip_option(list(argv if argv is not None else sys.argv[1:]), "--autostart", takes_value=True)
            print(auto.install(absolute_paths(keep)))
        return

    if args.cli:
        run_cli(cfg, args.cli)
        return

    if args.no_bubble:
        from .server import serve

        cfg.open_browser = not args.no_browser
        serve(cfg)
        return

    from .desktop import run

    hotkey = None if args.hotkey.strip().lower() in ("", "none", "off") else args.hotkey
    raise SystemExit(run(cfg, web_page=args.web, hotkey=hotkey))


def absolute_paths(argv: list[str]) -> list[str]:
    """At login LeanWatch starts in another folder, so saved folders (--adapter, a local --model) must be absolute."""
    out = list(argv)
    for i, a in enumerate(out):
        for opt in ("--adapter", "--model"):
            if a == opt and i + 1 < len(out) and os.path.exists(out[i + 1]):
                out[i + 1] = os.path.abspath(out[i + 1])
            elif a.startswith(opt + "=") and os.path.exists(a.split("=", 1)[1]):
                out[i] = f"{opt}={os.path.abspath(a.split('=', 1)[1])}"
    return out


def strip_option(argv: list[str], name: str, takes_value: bool = False) -> list[str]:
    """Remove `--name value` / `--name=value` from an argument list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == name:
            skip = takes_value
            continue
        if a.startswith(name + "="):
            continue
        out.append(a)
    return out


def run_cli(cfg: Config, task: str) -> None:
    from .brains import make_brain
    from .hooks import ConsoleHooks
    from .controller import Controller, DryRunBackend, PyAutoGuiBackend
    from .permissions import Permissions
    from .screen import Geometry, ScreenCapture
    from . import wayland

    print("LeanWatch will move your mouse and type on this computer to do:")
    print(f"  {task}")
    print("Stop it any time with Ctrl+C here, " + ("or by stopping the screen sharing from your desktop."
                                                  if wayland.wanted() else "or by pushing the mouse into a screen corner."))
    if input("Allow? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Okay, nothing was touched.")
        return

    stop = threading.Event()
    screen = None
    if wayland.wanted():
        said = []

        def asking():
            if not said:
                said.append(print(wayland.ASK_TEXT))

        if wayland.will_ask():
            asking()
        screen = wayland.WaylandScreen(stop=stop, asking=asking)
        if screen.note:
            print(screen.note)
        real = screen.backend
    else:
        real = PyAutoGuiBackend()
    backend = DryRunBackend(real.size()) if cfg.dry_run else real
    geometry = Geometry.for_screen(*backend.size(), cfg.screenshot_long_edge, cfg.screenshot_max_pixels)
    perms = Permissions()
    perms.grant(mouse=True, keyboard=True)
    hooks = ConsoleHooks()
    controller = Controller(backend, ScreenCapture(geometry, grab=screen.grab if screen else None), geometry, perms,
                            speed=cfg.move_speed, takeover_px=cfg.takeover_tolerance_px,
                            confirm=hooks.confirm, stop=stop)
    try:
        if cfg.brain == "local":
            print(f"Loading {cfg.local_model}…")
        result = make_brain(cfg, controller, hooks, stop).run(task)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
        return
    finally:
        if screen is not None:
            screen.close()
    print(("\n✓ " if result.ok else "\n✕ ") + result.message)


if __name__ == "__main__":
    main()
