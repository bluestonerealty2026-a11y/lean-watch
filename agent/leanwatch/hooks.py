"""What every brain shares: how it talks to the person watching, how a run ends,
and how one action is carried out safely (pause, stop, take-over)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .controller import Controller, UserTookOver, describe


class Stopped(Exception):
    """The user pressed Stop."""


class Hooks:
    """How the loop talks to whoever is watching. The server and the console override these."""

    def narrate(self, text: str, kind: str = "say") -> None: ...
    def action(self, name: str, inp: dict, summary: str) -> None: ...
    def preview(self) -> None: ...
    def confirm(self, action: str, reason: str) -> bool:
        return False
    def paused(self, reason: str) -> None: ...
    def wait_if_paused(self) -> None: ...


@dataclass
class RunResult:
    ok: bool
    message: str


def check_stop(stop: threading.Event) -> None:
    if stop.is_set():
        raise Stopped()


def run_action(controller: Controller, hooks: Hooks, stop: threading.Event, name: str, inp: dict):
    """Run one computer action, honouring pause and stop, and hand control back if the person moves the mouse."""
    hooks.wait_if_paused()
    check_stop(stop)
    if name not in ("screenshot", "zoom", "cursor_position"):
        hooks.action(name, inp, describe(name, inp))
    try:
        return controller.execute(name, inp)
    except UserTookOver as err:
        hooks.paused(str(err))
        hooks.wait_if_paused()
        check_stop(stop)
        controller.resync()
        return controller.execute(name, inp)


class ConsoleHooks(Hooks):
    """Plain terminal output, for `python -m leanwatch --cli "task"`."""

    def narrate(self, text, kind="say"):
        print(("   … " if kind == "thinking" else "▶ ") + text, flush=True)

    def action(self, name, inp, summary):
        print(f"     [{summary}]", flush=True)

    def confirm(self, action, reason):
        try:
            return input(f"\n⚠ {action}. {reason}\n  Allow? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def __init__(self):
        self._paused = False

    def paused(self, reason):
        print(f"⏸ {reason}", flush=True)
        self._paused = True

    def wait_if_paused(self):
        if self._paused:
            try:
                input("  Press Enter to let me carry on… ")
            except EOFError:
                pass
            self._paused = False
