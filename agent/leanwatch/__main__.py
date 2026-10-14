"""Command line entry point: ``python -m leanwatch``."""

from __future__ import annotations

import argparse
import threading

from .config import Config


def main(argv: list[str] | None = None) -> None:
    cfg = Config()
    p = argparse.ArgumentParser(
        prog="python -m leanwatch",
        description="Let Claude move your mouse and keyboard while you watch and learn.",
    )
    p.add_argument("--port", type=int, default=cfg.port, help=f"local port (default {cfg.port})")
    p.add_argument("--no-browser", action="store_true", help="don't open the control center page")
    p.add_argument("--overlay", action="store_true", help="also show the floating command box in the screen corner")
    p.add_argument("--dry-run", action="store_true", help="plan and narrate, but never move the mouse or press keys")
    p.add_argument("--model", default=cfg.model, help=f"Claude model (default {cfg.model})")
    p.add_argument("--effort", default=cfg.effort, choices=["low", "medium", "high", "xhigh", "max"], help="how hard Claude thinks per step")
    p.add_argument("--speed", type=float, default=cfg.move_speed, help="pointer speed multiplier (default 1.0)")
    p.add_argument("--cli", metavar="TASK", help="run one task in this terminal, without the web page")
    args = p.parse_args(argv)

    cfg.port = args.port
    cfg.open_browser = not args.no_browser
    cfg.dry_run = args.dry_run
    cfg.model = args.model
    cfg.effort = args.effort
    cfg.move_speed = args.speed

    if args.cli:
        run_cli(cfg, args.cli)
        return

    from .server import serve

    serve(cfg, overlay=args.overlay)


def run_cli(cfg: Config, task: str) -> None:
    from .brain import Brain, ConsoleHooks
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
        result = Brain(cfg, controller, hooks, stop).run(task)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
        return
    print(("\n✓ " if result.ok else "\n✕ ") + result.message)


if __name__ == "__main__":
    main()
