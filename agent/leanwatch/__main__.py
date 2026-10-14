"""Command line entry point: ``python -m leanwatch``."""

from __future__ import annotations

import argparse
import threading

from .config import Config


def main(argv: list[str] | None = None) -> None:
    cfg = Config()
    p = argparse.ArgumentParser(
        prog="python -m leanwatch",
        description="Let an AI model move your mouse and keyboard while you watch and learn.",
    )
    p.add_argument("--port", type=int, default=cfg.port, help=f"local port (default {cfg.port})")
    p.add_argument("--no-browser", action="store_true", help="don't open the control center page")
    p.add_argument("--overlay", action="store_true", help="also show the floating command box in the screen corner")
    p.add_argument("--dry-run", action="store_true", help="plan and narrate, but never move the mouse or press keys")
    p.add_argument("--brain", default=cfg.brain, choices=["local", "claude"], help="local: open model on this computer (default); claude: Claude API, needs a key")
    p.add_argument("--model", help=f"local brain: Hugging Face id or folder of your fine-tuned model (default {cfg.local_model}); claude brain: model id")
    p.add_argument("--adapter", default=cfg.adapter, help="folder of a LoRA adapter from training/train_sft.py")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true", default=cfg.load_in_4bit, help="load the local model in 4-bit (NVIDIA GPU, needs bitsandbytes)")
    p.add_argument("--device", default=cfg.device, choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--coords", default=None, choices=["abs", "rel1000"], help="how the model writes points: abs (UI-TARS-1.5, Qwen2.5-VL; default) or rel1000. Read from the adapter when you pass one")
    p.add_argument("--style", default=None, choices=["box", "point"], help="how points are written in the prompt: box (start_box=..., default) or point (point='<point>x y</point>'). Read from the adapter when you pass one")
    p.add_argument("--max-pixels", type=int, default=cfg.local_max_pixels, help="largest screenshot size, in pixels, the local model sees")
    p.add_argument("--language", default=cfg.language, help="language for the explanations (default English)")
    p.add_argument("--effort", default=cfg.effort, choices=["low", "medium", "high", "xhigh", "max"], help="claude brain only: how hard it thinks per step")
    p.add_argument("--speed", type=float, default=cfg.move_speed, help="pointer speed multiplier (default 1.0)")
    p.add_argument("--cli", metavar="TASK", help="run one task in this terminal, without the web page")
    args = p.parse_args(argv)

    cfg.port = args.port
    cfg.open_browser = not args.no_browser
    cfg.dry_run = args.dry_run
    cfg.brain = args.brain
    if args.model:
        if cfg.brain == "claude":
            cfg.claude_model = args.model
        else:
            cfg.local_model = args.model
    cfg.adapter = args.adapter
    if cfg.brain == "local":
        from .brains import resolve_model_id, trained_settings

        # A model you fine-tuned remembers how its data was written (train_sft.py saves leanwatch.json
        # next to the adapter and the merged model). Use the same settings, unless a flag says otherwise.
        trained = trained_settings(cfg.adapter, args.model)
        if cfg.adapter and not args.model:
            cfg.local_model = resolve_model_id(None, cfg.adapter)
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
    cfg.language = args.language
    cfg.effort = args.effort
    cfg.move_speed = args.speed

    if args.cli:
        run_cli(cfg, args.cli)
        return

    from .server import serve

    serve(cfg, overlay=args.overlay)


def run_cli(cfg: Config, task: str) -> None:
    from .brains import make_brain
    from .hooks import ConsoleHooks
    from .controller import Controller, DryRunBackend, PyAutoGuiBackend
    from .permissions import Permissions
    from .screen import Geometry, ScreenCapture

    print("LeanWatch will move your mouse and type on this computer to do:")
    print(f"  {task}")
    print("Stop it any time by pushing the mouse into a screen corner, or with Ctrl+C here.")
    if input("Allow? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Okay, nothing was touched.")
        return

    real = PyAutoGuiBackend()
    backend = DryRunBackend(real.size()) if cfg.dry_run else real
    geometry = Geometry.for_screen(*backend.size(), cfg.screenshot_long_edge, cfg.screenshot_max_pixels)
    perms = Permissions()
    perms.grant(mouse=True, keyboard=True)
    hooks = ConsoleHooks()
    stop = threading.Event()
    controller = Controller(backend, ScreenCapture(geometry), geometry, perms,
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
    print(("\n✓ " if result.ok else "\n✕ ") + result.message)


if __name__ == "__main__":
    main()
