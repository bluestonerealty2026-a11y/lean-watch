"""A GNOME keyboard shortcut that opens LeanWatch's box:  python -m leanwatch --shortcut on

On Wayland an app can't watch the keyboard while other apps are in front, so LeanWatch's own
shortcut only works in X11 sessions. GNOME's custom shortcuts (Settings > Keyboard > View and
Customize Shortcuts > Custom Shortcuts) work everywhere: this adds one there that runs open_box.py.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
ENTRY = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/leanwatch/"
OPEN_SCRIPT = Path(__file__).with_name("open_box.py")


def _gsettings(*args: str) -> str:
    return subprocess.run(["gsettings", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def _text(value: str) -> str:
    """A GVariant string literal: gsettings parses what it is given, so quote it."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _entries() -> list[str]:
    raw = _gsettings("get", SCHEMA, "custom-keybindings")
    raw = raw.removeprefix("@as ").strip()
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else []


def _relocatable() -> str:
    return f"{SCHEMA}.custom-keybinding:{ENTRY}"


def gnome_binding(combo: str) -> str:
    """'ctrl+alt+space' -> '<Control><Alt>space' (GNOME's accelerator syntax)."""
    names = {"ctrl": "<Control>", "control": "<Control>", "alt": "<Alt>", "shift": "<Shift>",
             "super": "<Super>", "cmd": "<Super>", "win": "<Super>", "meta": "<Super>"}
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty shortcut")
    *mods, key = parts
    return "".join(names[m] for m in mods) + key


def command(python: str = sys.executable) -> str:
    return f'"{python}" "{OPEN_SCRIPT}"'


def available() -> bool:
    try:
        _gsettings("list-keys", SCHEMA)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def installed() -> bool:
    try:
        return ENTRY in _entries()
    except (OSError, subprocess.SubprocessError):
        return False


def install(combo: str) -> str:
    if not available():
        raise RuntimeError("GNOME's keyboard settings aren't available here (this works on GNOME, e.g. Ubuntu's desktop).")
    binding = gnome_binding(combo)
    entries = _entries()
    if ENTRY not in entries:
        entries.append(ENTRY)
    rel = _relocatable()
    _gsettings("set", rel, "name", _text("LeanWatch"))
    _gsettings("set", rel, "command", _text(command()))
    _gsettings("set", rel, "binding", _text(binding))
    _gsettings("set", SCHEMA, "custom-keybindings", "[" + ", ".join(_text(e) for e in entries) + "]")
    return (f"Added the shortcut {combo} to GNOME's keyboard settings: it opens LeanWatch's box from any app. "
            "Change or remove it in Settings > Keyboard > View and Customize Shortcuts > Custom Shortcuts.")


def remove() -> str:
    if not available():
        return "GNOME's keyboard settings aren't available here, so there is nothing to remove."
    entries = _entries()
    if ENTRY not in entries:
        return "There was no LeanWatch shortcut in GNOME's keyboard settings."
    _gsettings("set", SCHEMA, "custom-keybindings", "[" + ", ".join(_text(e) for e in entries if e != ENTRY) + "]")
    rel = _relocatable()
    for key in ("name", "command", "binding"):
        try:
            _gsettings("reset", rel, key)
        except subprocess.SubprocessError:
            pass
    return "Removed LeanWatch's shortcut from GNOME's keyboard settings."
