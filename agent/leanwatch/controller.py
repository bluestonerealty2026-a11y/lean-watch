"""Turns computer-use actions into real mouse and keyboard input.

Movements follow the same human-motor model as the web page's pointer:
Fitts's law sets how long a move takes, and a minimum-jerk profile along a
gently curved path sets the speed, so a person can follow it with their eyes.
"""

from __future__ import annotations

import math
import random
import sys
import threading
import time
from typing import Callable, Protocol

from .permissions import Permissions, PermissionDenied, risky_key, risky_text
from .screen import Geometry, ScreenCapture, image_block, png_b64

FITTS_A = 0.20   # s
FITTS_B = 0.12   # s per bit
TARGET_W = 24    # px


class UserTookOver(Exception):
    """The real mouse moved without us: the person wants control back."""


class EmergencyStop(Exception):
    """Fail-safe: the mouse was pushed into a screen corner."""


class Backend(Protocol):
    def size(self) -> tuple[int, int]: ...
    def position(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def mouse_down(self, button: str = "left") -> None: ...
    def mouse_up(self, button: str = "left") -> None: ...
    def click(self, button: str = "left") -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def write(self, text: str) -> None: ...


class PyAutoGuiBackend:
    """Real input through pyautogui. Moving the mouse into a corner aborts (FAILSAFE)."""

    def __init__(self) -> None:
        if sys.platform == "win32":
            # With display scaling on, make pointer coordinates and screenshots use the same real pixels.
            try:
                import ctypes

                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self.pg = pyautogui

    def size(self):
        s = self.pg.size()
        return int(s[0]), int(s[1])

    def position(self):
        p = self.pg.position()
        return int(p[0]), int(p[1])

    def move_to(self, x, y):
        self.pg.moveTo(x, y, _pause=False)

    def mouse_down(self, button="left"):
        self.pg.mouseDown(button=button)

    def mouse_up(self, button="left"):
        self.pg.mouseUp(button=button)

    def click(self, button="left"):
        self.pg.click(button=button)

    def scroll(self, dx, dy):
        if dy:
            self.pg.scroll(dy)
        if dx:
            self.pg.hscroll(dx)

    def key_down(self, key):
        self.pg.keyDown(key)

    def key_up(self, key):
        self.pg.keyUp(key)

    def write(self, text):
        self.pg.write(text, interval=0.03)


class DryRunBackend:
    """Records what would happen without touching the real mouse or keyboard."""

    def __init__(self, size: tuple[int, int] = (1920, 1080)) -> None:
        self._size = size
        self._pos = (size[0] // 2, size[1] // 2)
        self.log: list[tuple] = []

    def size(self):
        return self._size

    def position(self):
        return self._pos

    def move_to(self, x, y):
        self._pos = (x, y)

    def mouse_down(self, button="left"):
        self.log.append(("down", button, self._pos))

    def mouse_up(self, button="left"):
        self.log.append(("up", button, self._pos))

    def click(self, button="left"):
        self.log.append(("click", button, self._pos))

    def scroll(self, dx, dy):
        self.log.append(("scroll", dx, dy, self._pos))

    def key_down(self, key):
        self.log.append(("key_down", key))

    def key_up(self, key):
        self.log.append(("key_up", key))

    def write(self, text):
        self.log.append(("write", text))


# xdotool-style names the model uses -> pyautogui key names
_KEYS = {
    "return": "enter", "enter": "enter", "kp_enter": "enter", "escape": "esc", "esc": "esc",
    "backspace": "backspace", "tab": "tab", "space": "space", "delete": "delete", "del": "delete",
    "insert": "insert", "home": "home", "end": "end",
    "page_up": "pageup", "pageup": "pageup", "prior": "pageup",
    "page_down": "pagedown", "pagedown": "pagedown", "next": "pagedown",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "ctrl": "ctrl", "control": "ctrl", "control_l": "ctrl", "control_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "option": "option",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "caps_lock": "capslock", "capslock": "capslock",
    "print": "printscreen", "printscreen": "printscreen", "menu": "apps",
    "minus": "-", "plus": "+", "equal": "=", "comma": ",", "period": ".", "slash": "/",
    "semicolon": ";", "apostrophe": "'", "bracketleft": "[", "bracketright": "]", "backslash": "\\", "grave": "`",
}


def key_name(name: str, platform: str = sys.platform) -> str:
    n = name.strip()
    low = n.lower()
    if low in ("super", "super_l", "super_r", "win", "windows", "meta", "cmd", "command"):
        return "command" if platform == "darwin" else "win"
    if low in _KEYS:
        return _KEYS[low]
    return low  # letters, digits, punctuation and f1..f24 are already pyautogui names


def parse_combo(text: str, platform: str = sys.platform) -> list[str]:
    """'ctrl+shift+t' -> ['ctrl', 'shift', 't']; 'ctrl++' -> ['ctrl', '+']."""
    text = text.strip()
    parts = text.split("+")
    keys: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "" and i + 1 < len(parts) and parts[i + 1] == "":
            keys.append("+")
            i += 2
            continue
        if p:
            keys.append(key_name(p, platform))
        i += 1
    return keys


def minimum_jerk(t: float) -> float:
    return t * t * t * (10 - 15 * t + 6 * t * t)


def movement_time(dist: float) -> float:
    """Fitts's law: T = a + b * log2(D / W + 1)."""
    return FITTS_A + FITTS_B * math.log2(dist / TARGET_W + 1)


def human_path(x0: float, y0: float, x1: float, y1: float, steps: int) -> list[tuple[int, int]]:
    """Points along a slightly curved Bezier path, spaced by a minimum-jerk profile."""
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        return [(round(x1), round(y1))]
    bend = max(6.0, min(60.0, dist * 0.15)) * random.choice((-1, 1))
    nx, ny = -(y1 - y0) / dist, (x1 - x0) / dist
    c1 = (x0 + (x1 - x0) * 0.3 + nx * bend, y0 + (y1 - y0) * 0.3 + ny * bend)
    c2 = (x0 + (x1 - x0) * 0.75 + nx * bend * 0.4, y0 + (y1 - y0) * 0.75 + ny * bend * 0.4)
    pts = []
    for i in range(1, steps + 1):
        s = minimum_jerk(i / steps)
        u = 1 - s
        x = u**3 * x0 + 3 * u * u * s * c1[0] + 3 * u * s * s * c2[0] + s**3 * x1
        y = u**3 * y0 + 3 * u * u * s * c1[1] + 3 * u * s * s * c2[1] + s**3 * y1
        pts.append((round(x), round(y)))
    pts[-1] = (round(x1), round(y1))
    return pts


def describe(name: str, inp: dict) -> str:
    """Short human-readable label for an action, for the step log."""
    c = inp.get("coordinate")
    at = f" at ({c[0]}, {c[1]})" if c else ""
    mods = f" holding {inp['text']}" if inp.get("text") and name.endswith(("click", "scroll", "drag")) else ""
    if name == "type":
        t = inp.get("text", "")
        return f"type “{t[:60]}{'…' if len(t) > 60 else ''}”"
    if name in ("key", "hold_key"):
        rep = f" ×{inp['repeat']}" if inp.get("repeat", 1) > 1 else ""
        return f"{'hold' if name == 'hold_key' else 'press'} {inp.get('text', '')}{rep}"
    if name == "scroll":
        return f"scroll {inp.get('scroll_direction')} {inp.get('scroll_amount')}{at}{mods}"
    if name == "left_click_drag":
        s = inp.get("start_coordinate") or ["?", "?"]
        return f"drag from ({s[0]}, {s[1]}) to ({c[0]}, {c[1]})" if c else "drag"
    if name == "zoom":
        return f"zoom into {inp.get('region')}"
    if name == "wait":
        return f"wait {inp.get('duration')} s"
    return f"{name.replace('_', ' ')}{at}{mods}"


class Controller:
    """Executes one computer-use action at a time, with safety checks."""

    def __init__(
        self,
        backend: Backend,
        capture: ScreenCapture | None,
        geometry: Geometry,
        permissions: Permissions,
        *,
        speed: float = 1.0,
        takeover_px: int = 30,
        confirm: Callable[[str, str], bool] | None = None,
        stop: threading.Event | None = None,
        platform: str = sys.platform,
        shield=None,
    ) -> None:
        self.backend = backend
        self.capture = capture
        self.g = geometry
        self.perms = permissions
        self.speed = speed
        self.takeover_px = takeover_px
        self.confirm = confirm or (lambda action, reason: False)
        self.stop = stop or threading.Event()
        self.platform = platform
        # The desktop bubble: clear_of(points) moves LeanWatch's windows out of the way of a click.
        self.shield = shield
        self._expected: tuple[int, int] | None = None

    # ---------------- helpers ----------------
    def resync(self) -> None:
        """Accept wherever the mouse is now (after the person hands control back)."""
        self._expected = self.backend.position()

    def _check_takeover(self) -> None:
        pos = self.backend.position()
        if self._expected is not None and math.dist(pos, self._expected) > self.takeover_px:
            self._expected = None
            raise UserTookOver("You moved the mouse, so I paused. Press Resume when you want me to carry on.")
        self._expected = pos

    def _move(self, x: int, y: int) -> None:
        x0, y0 = self.backend.position()
        dist = math.hypot(x - x0, y - y0)
        duration = movement_time(dist) / max(self.speed, 0.1)
        steps = max(2, int(duration / 0.012))
        t0 = time.monotonic()
        for i, (px, py) in enumerate(human_path(x0, y0, x, y, steps), start=1):
            if self.stop.is_set():
                break
            self.backend.move_to(px, py)
            lag = t0 + duration * i / steps - time.monotonic()
            if lag > 0:
                time.sleep(lag)
        self._expected = (x, y)

    def _point(self, coord) -> tuple[int, int]:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValueError("coordinate must be [x, y]")
        return self.g.to_screen(float(coord[0]), float(coord[1]))

    def _with_modifiers(self, text: str | None, fn: Callable[[], None]) -> None:
        mods = parse_combo(text, self.platform) if text else []
        for m in mods:
            self.backend.key_down(m)
        try:
            fn()
        finally:
            for m in reversed(mods):
                self.backend.key_up(m)

    def screenshot_block(self) -> list[dict]:
        if self.capture is None:
            raise RuntimeError("Screen capture is not available on this computer.")
        return [image_block(png_b64(self.capture.screenshot()))]

    def cursor_fraction(self) -> tuple[float, float]:
        return self.g.fraction(*self.backend.position())

    # ---------------- dispatch ----------------
    def execute(self, name: str, inp: dict):
        """Run one action. Returns tool_result content (a string or a list of blocks)."""
        self.perms.check(name)
        try:
            return self._execute(name, inp or {})
        except PermissionDenied:
            raise
        except Exception as err:  # pyautogui raises FailSafeException from a screen corner
            if type(err).__name__ == "FailSafeException":
                raise EmergencyStop("The mouse was pushed into a screen corner, so everything stopped (fail-safe).") from err
            raise

    def _targets(self, name: str, inp: dict) -> list[tuple[int, int]]:
        """Screen points a pointer action will press on (not the path it travels)."""
        pts = []
        for key in ("start_coordinate", "coordinate"):
            if name in ("mouse_move", "zoom", "cursor_position", "screenshot"):
                break
            if inp.get(key) is not None:
                try:
                    pts.append(self._point(inp[key]))
                except ValueError:
                    pass
        if not pts and name in ("left_click", "right_click", "middle_click", "double_click", "triple_click",
                                "scroll", "left_mouse_down", "left_mouse_up"):
            pts.append(self.backend.position())
        return pts

    def _execute(self, name: str, inp: dict):
        points = self._targets(name, inp) if self.shield is not None else []
        if points:
            with self.shield.clear_of(points):
                return self._execute_now(name, inp)
        return self._execute_now(name, inp)

    def _execute_now(self, name: str, inp: dict):
        b = self.backend
        if name == "screenshot":
            return self.screenshot_block()
        if name == "zoom":
            if self.capture is None:
                raise RuntimeError("Screen capture is not available on this computer.")
            return [image_block(png_b64(self.capture.zoom(inp["region"])))]
        if name == "cursor_position":
            x, y = self.g.to_shot(*b.position())
            return f"X={x}, Y={y}"
        if name == "wait":
            self._sleep(float(inp.get("duration", 1)))
            return "OK"

        # If the person moved the mouse since our last step (say, to click into another app while the
        # model was thinking), stop before typing: the keys would go to whatever they clicked.
        self._check_takeover()
        if name in ("type", "key", "hold_key"):
            return self._keyboard(name, inp)

        button = {"right_click": "right", "middle_click": "middle"}.get(name, "left")
        if name in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            if inp.get("coordinate") is not None:
                self._move(*self._point(inp["coordinate"]))
            count = {"double_click": 2, "triple_click": 3}.get(name, 1)

            def press():
                for i in range(count):
                    b.click(button)
                    if i + 1 < count:
                        time.sleep(0.06)

            self._with_modifiers(inp.get("text"), press)
            return "OK"
        if name == "mouse_move":
            self._move(*self._point(inp["coordinate"]))
            return "OK"
        if name == "left_click_drag":
            start = self._point(inp["start_coordinate"])
            end = self._point(inp["coordinate"])

            def drag():
                self._move(*start)
                b.mouse_down("left")
                time.sleep(0.08)
                self._move(*end)
                time.sleep(0.05)
                b.mouse_up("left")

            self._with_modifiers(inp.get("text"), drag)
            return "OK"
        if name == "left_mouse_down":
            b.mouse_down("left")
            return "OK"
        if name == "left_mouse_up":
            b.mouse_up("left")
            return "OK"
        if name == "scroll":
            if inp.get("coordinate") is not None:
                self._move(*self._point(inp["coordinate"]))
            amount = int(inp.get("scroll_amount", 3))
            direction = inp.get("scroll_direction", "down")
            dx = {"left": -amount, "right": amount}.get(direction, 0)
            dy = {"up": amount, "down": -amount}.get(direction, 0)
            self._with_modifiers(inp.get("text"), lambda: b.scroll(dx, dy))
            return "OK"
        raise ValueError(f"Unsupported action {name!r}")

    def _keyboard(self, name: str, inp: dict):
        b = self.backend
        text = str(inp.get("text", ""))
        if name == "type":
            reason = risky_text(text)
            if reason and not self.confirm(f"Type “{text[:80]}”", f"This {reason}."):
                raise PermissionDenied("The user declined typing this text. Do not try it another way; explain what you skipped.")
            for chunk_start in range(0, len(text), 40):
                if self.stop.is_set():
                    break
                b.write(text[chunk_start:chunk_start + 40])
            return "OK"
        reason = risky_key(text)
        if reason and not self.confirm(f"Press {text}", f"This {reason}."):
            raise PermissionDenied("The user declined this key press.")
        keys = parse_combo(text, self.platform)
        if not keys:
            raise ValueError("key text is empty")
        if name == "hold_key":
            for k in keys:
                b.key_down(k)
            try:
                self._sleep(min(float(inp.get("duration", 1)), 300))
            finally:
                for k in reversed(keys):
                    b.key_up(k)
            return "OK"
        for _ in range(max(1, min(int(inp.get("repeat", 1)), 100))):
            for k in keys:
                b.key_down(k)
            for k in reversed(keys):
                b.key_up(k)
            time.sleep(0.03)
        return "OK"

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + min(seconds, 300)
        while time.monotonic() < end and not self.stop.is_set():
            time.sleep(min(0.1, end - time.monotonic()))
