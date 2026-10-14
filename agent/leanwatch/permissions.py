"""Consent and safety rules.

Nothing touches the mouse or keyboard until the user grants permission, and
each grant names what is allowed (pointer, keyboard). On top of the model's
own request_confirmation tool, typed text that looks like a destructive
command is held back until the user approves it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

OBSERVE_ACTIONS = {"screenshot", "zoom", "cursor_position", "wait"}
KEYBOARD_ACTIONS = {"type", "key", "hold_key"}
MOUSE_ACTIONS = {
    "left_click", "right_click", "middle_click", "double_click", "triple_click",
    "left_click_drag", "mouse_move", "left_mouse_down", "left_mouse_up", "scroll",
}

# Typed text that deletes, formats, escalates or pipes the internet into a shell.
_RISKY_TEXT = [
    (r"\brm\s+-[a-z]*r", "deletes files and folders permanently"),
    (r"\b(rmdir|rd)\b", "deletes a folder"),
    (r"\bdel\s+\S", "deletes files"),
    (r"\bRemove-Item\b", "deletes files"),
    (r"\bformat\s+[a-z]:", "erases a whole drive"),
    (r"\b(mkfs|diskpart)\b", "changes disks or partitions"),
    (r"\bdd\s+if=", "writes raw data to a disk"),
    (r"\b(shutdown|reboot|halt)\b", "turns the computer off or restarts it"),
    (r"\bsudo\b", "runs a command as administrator"),
    (r"\breg\s+(delete|add)\b", "changes the Windows registry"),
    (r"\b(curl|wget|iwr|Invoke-WebRequest)\b.*\|\s*(sh|bash|zsh|iex|python)", "downloads and runs a script"),
    (r"\b(pip|npm|apt|brew|winget|choco)\s+(install|uninstall|remove)\b", "installs or removes software"),
]

_RISKY_KEYS = {
    "shift+delete": "deletes the selection without the Recycle Bin",
    "shift+del": "deletes the selection without the Recycle Bin",
    "cmd+option+delete": "deletes the selection without the Trash",
}


class PermissionDenied(Exception):
    """The user has not allowed this kind of action."""


def risky_text(text: str) -> str | None:
    """Why typing this text is risky, or None when it looks harmless."""
    for pattern, reason in _RISKY_TEXT:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return None


def risky_key(combo: str) -> str | None:
    return _RISKY_KEYS.get(combo.strip().lower().replace(" ", ""))


@dataclass
class Permissions:
    granted: bool = False
    mouse: bool = False
    keyboard: bool = False
    granted_at: float | None = None

    def grant(self, mouse: bool = True, keyboard: bool = True) -> None:
        self.granted = bool(mouse or keyboard)
        self.mouse = bool(mouse)
        self.keyboard = bool(keyboard)
        self.granted_at = time.time() if self.granted else None

    def revoke(self) -> None:
        self.granted = self.mouse = self.keyboard = False
        self.granted_at = None

    def check(self, action: str) -> None:
        """Raise PermissionDenied unless the user allowed this action."""
        if action in OBSERVE_ACTIONS:
            if not self.granted:
                raise PermissionDenied("The user has not given LeanWatch permission yet.")
            return
        if action in MOUSE_ACTIONS and not self.mouse:
            raise PermissionDenied("The user has not allowed mouse control. Ask them to allow it, or explain the steps instead.")
        if action in KEYBOARD_ACTIONS and not self.keyboard:
            raise PermissionDenied("The user has not allowed keyboard control. Ask them to type it themselves, and tell them exactly what to type.")
        if action not in OBSERVE_ACTIONS | MOUSE_ACTIONS | KEYBOARD_ACTIONS:
            raise PermissionDenied(f"Unknown action {action!r}.")

    def as_dict(self) -> dict:
        return {"granted": self.granted, "mouse": self.mouse, "keyboard": self.keyboard}
