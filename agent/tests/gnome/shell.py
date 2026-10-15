"""Test-only control of a headless GNOME Shell started with --unsafe-mode: clicks, keys, screenshots,
window positions. These use the shell's own D-Bus Eval, which only works in unsafe mode (tests)."""

import ast
import json
import subprocess
import time


def ev(js: str):
    """Run JavaScript in the shell; returns the result (the shell sends it as JSON)."""
    out = subprocess.run(["gdbus", "call", "--session", "--dest", "org.gnome.Shell", "--object-path", "/org/gnome/Shell",
                          "--method", "org.gnome.Shell.Eval", js], capture_output=True, text=True, timeout=20)
    text = out.stdout.strip()
    if not text.startswith("(true,"):
        raise RuntimeError(f"Shell.Eval failed: {text or out.stderr.strip()}")
    payload = ast.literal_eval("(True," + text[len("(true,"):])[1]     # GVariant text: ('json',)
    return json.loads(payload) if payload else None


_SETUP = ("if (!global._lw) { const seat = imports.gi.Clutter.get_default_backend().get_default_seat();"
          " global._lw = {p: seat.create_virtual_device(imports.gi.Clutter.InputDeviceType.POINTER_DEVICE),"
          " k: seat.create_virtual_device(imports.gi.Clutter.InputDeviceType.KEYBOARD_DEVICE)}; } ")
_NOW = "imports.gi.GLib.get_monotonic_time()"     # events dated in the future confuse mutter's focus handling


def click(x: int, y: int, button: int = 1) -> None:
    ev(_SETUP + f"global._lw.p.notify_absolute_motion({_NOW}, {x}, {y}); 1")
    time.sleep(0.05)
    ev(_SETUP + f"global._lw.p.notify_button({_NOW}, {button}, imports.gi.Clutter.ButtonState.PRESSED); 1")
    time.sleep(0.05)
    ev(_SETUP + f"global._lw.p.notify_button({_NOW}, {button}, imports.gi.Clutter.ButtonState.RELEASED); 1")


def key(keyval: int) -> None:
    ev(_SETUP + f"global._lw.k.notify_keyval({_NOW}, {keyval}, imports.gi.Clutter.KeyState.PRESSED); 1")
    time.sleep(0.02)
    ev(_SETUP + f"global._lw.k.notify_keyval({_NOW}, {keyval}, imports.gi.Clutter.KeyState.RELEASED); 1")


def type_text(text: str) -> None:
    for ch in text:
        key(0xFF0D if ch == "\n" else ord(ch))
        time.sleep(0.02)


def screenshot(path: str) -> None:
    out = subprocess.run(["gdbus", "call", "--session", "--dest", "org.gnome.Shell.Screenshot", "--object-path",
                          "/org/gnome/Shell/Screenshot", "--method", "org.gnome.Shell.Screenshot.Screenshot",
                          "false", "false", path], capture_output=True, text=True, timeout=20)
    if "true" not in out.stdout:
        raise RuntimeError(f"screenshot failed: {out.stdout} {out.stderr}")


def windows() -> list[dict]:
    """Every window: title, frame rectangle, has focus."""
    return json.loads(ev(
        "JSON.stringify(global.get_window_actors().map(a => { const w = a.meta_window, r = w.get_frame_rect();"
        " return {title: w.get_title(), x: r.x, y: r.y, w: r.width, h: r.height, focus: w.has_focus(),"
        " wayland: w.get_client_type() === 0}; }))"))


def window(title: str, timeout: float = 15) -> dict | None:
    end = time.time() + timeout
    while time.time() < end:
        for w in windows():
            if w["title"] == title and w["w"] > 0:      # placed and sized
                return w
        time.sleep(0.2)
    return None


def stop_sharing() -> None:
    """What the person does with the screen-sharing indicator: end every remote desktop session."""
    ev("const c = global.backend.get_remote_access_controller(); c.inhibit_remote_access();"
       " c.uninhibit_remote_access(); 1")
