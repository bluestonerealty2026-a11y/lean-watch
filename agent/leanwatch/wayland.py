"""Seeing and controlling a Wayland desktop (GNOME, KDE) through the desktop's own permission dialog.

On Wayland an app can't read the screen or move the pointer by itself (X11 tools only reach older
X11 apps). It asks the desktop "portal" (xdg-desktop-portal) for a remote desktop session; the
desktop shows its own dialog and the person decides. LeanWatch asks to remember the answer
(persist_mode 2), so the dialog only comes back if the person withdraws it.

- Mouse and keyboard: the RemoteDesktop portal's Notify* calls.
- The screen: frames of the same session's screen-cast stream, read from PipeWire by GStreamer in
  the system's own Python (GNOME depends on both, so they are there). No flash, no shutter sound.
  If that isn't available, the Screenshot portal, which flashes the screen and plays a sound each time.

While a task runs the desktop shows that the screen is shared (GNOME: an icon in the top bar).
Stopping the sharing from there stops LeanWatch.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .controller import EmergencyStop

PORTAL_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
RD = "org.freedesktop.portal.RemoteDesktop"
SC = "org.freedesktop.portal.ScreenCast"
SHOT = "org.freedesktop.portal.Screenshot"

KEYBOARD, POINTER = 1, 2
MONITOR = 1
CURSOR_HIDDEN, CURSOR_EMBEDDED = 1, 2
BUTTONS = {"left": 0x110, "right": 0x111, "middle": 0x112}       # Linux input event codes (BTN_*)
STATE_FILE = Path.home() / ".leanwatch" / "wayland.json"


class PortalError(RuntimeError):
    pass


class PortalDenied(PortalError):
    """The person said no in the desktop's dialog (or closed it)."""


class SharingStopped(EmergencyStop):
    """The person stopped the screen sharing from the desktop (e.g. GNOME's top bar): stop everything."""


def wanted() -> bool:
    """Use the portal: a Wayland session, unless LEANWATCH_WAYLAND=0 asks for the X11 tools only."""
    from .native import wayland_session

    return wayland_session() and os.environ.get("LEANWATCH_WAYLAND", "1") != "0"


def _saved_token(state_file: Path) -> str | None:
    try:
        data = json.loads(state_file.read_text())
    except (OSError, ValueError):
        return None
    token = data.get("restore_token") if isinstance(data, dict) else None
    return token if isinstance(token, str) and token else None


def will_ask(state_file: Path | None = None) -> bool:
    """True if the desktop will show its permission dialog (nothing remembered yet)."""
    return _saved_token(state_file or STATE_FILE) is None


ASK_TEXT = ("Your desktop will ask to share the screen. Turn on “Allow Remote Interaction”, then click Share "
            "(it remembers your answer).")


def portal_versions(bus: "PortalBus | None" = None) -> dict:
    """Which portals the desktop offers, e.g. {"RemoteDesktop": 2, "ScreenCast": 5, "Screenshot": 2}."""
    own = bus is None
    bus = bus or PortalBus()
    try:
        found = {}
        for name, iface in (("RemoteDesktop", RD), ("ScreenCast", SC), ("Screenshot", SHOT)):
            try:
                found[name] = int(bus.get_property(iface, "version"))
            except Exception:
                pass
        return found
    finally:
        if own:
            bus.close()


def available() -> bool:
    """The Python side is installed (jeepney talks to D-Bus without compiled code)."""
    try:
        import jeepney  # noqa: F401

        return True
    except ImportError:
        return False


def _unwrap(value):
    """jeepney gives variants as (signature, value); return plain Python values."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str) and _is_signature(value[0]):
        return _unwrap(value[1])
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unwrap(v) for v in value)
    return value


def _is_signature(s: str) -> bool:
    return bool(s) and re.fullmatch(r"[ybnqiuxtdsoghav(){}]+", s) is not None


class PortalBus:
    """A session-bus connection to the portal, with the Request/Response dance most portal calls use."""

    def __init__(self) -> None:
        from jeepney.io.threading import DBusRouter, open_dbus_connection

        conn = open_dbus_connection(bus="SESSION", enable_fds=True)
        self.router = DBusRouter(conn)
        self.sender = conn.unique_name.lstrip(":").replace(".", "_")
        self._tokens = itertools.count(1)

    def close(self) -> None:
        try:
            self.router.close()
            self.router.conn.close()
        except Exception:
            pass

    def token(self) -> str:
        return f"leanwatch{os.getpid()}_{next(self._tokens)}"

    def call(self, interface: str, method: str, signature: str = "", body: tuple = (), *,
             path: str = PORTAL_PATH, dest: str = PORTAL_NAME, timeout: float = 10.0):
        from jeepney import DBusAddress, MessageType, new_method_call

        msg = new_method_call(DBusAddress(path, bus_name=dest, interface=interface), method, signature, body)
        reply = self.router.send_and_get_reply(msg, timeout=timeout)
        if reply.header.message_type == MessageType.error:
            name = reply.header.fields.get(4, "error")       # HeaderFields.error_name
            text = reply.body[0] if reply.body else ""
            raise PortalError(f"{interface}.{method}: {name} {text}".strip())
        return reply.body

    def get_property(self, interface: str, name: str):
        return _unwrap(self.call("org.freedesktop.DBus.Properties", "Get", "ss", (interface, name))[0])

    def match(self, **rule):
        """Subscribe to signals (AddMatch) and return the jeepney rule for router.filter()."""
        from jeepney import MatchRule

        r = MatchRule(type="signal", **rule)
        self.call("org.freedesktop.DBus", "AddMatch", "s", (r.serialise(),), path="/org/freedesktop/DBus",
                  dest="org.freedesktop.DBus")
        return r

    def unmatch(self, rule) -> None:
        try:
            self.call("org.freedesktop.DBus", "RemoveMatch", "s", (rule.serialise(),), path="/org/freedesktop/DBus",
                      dest="org.freedesktop.DBus")
        except Exception:
            pass

    def request(self, interface: str, method: str, signature: str, args: tuple, options: dict,
                stop: threading.Event | None = None, timeout: float = 30.0, waiting=None) -> dict:
        """Call a portal method that answers later with a Request.Response signal; return its results.

        Methods that show a dialog wait for the person, so they get a long timeout, and `stop`
        (LeanWatch's Stop button) cancels the request and closes the dialog. `waiting` is called once
        if there's no answer within a second (the desktop is showing its dialog).
        """
        token = self.token()
        path = f"{PORTAL_PATH}/request/{self.sender}/{token}"
        rule = self.match(interface="org.freedesktop.portal.Request", member="Response", path=path)
        try:
            with self.router.filter(rule) as queue:
                self.call(interface, method, signature, (*args, {**options, "handle_token": ("s", token)}))
                deadline = time.monotonic() + timeout
                nudge_at = time.monotonic() + 1.0 if waiting else None
                while True:
                    if nudge_at is not None and time.monotonic() >= nudge_at:
                        nudge_at = None
                        try:
                            waiting()
                        except Exception:
                            pass
                    if stop is not None and stop.is_set():
                        try:
                            self.call("org.freedesktop.portal.Request", "Close", path=path)
                        except Exception:
                            pass
                        raise PortalDenied("Stopped while waiting for the desktop's permission dialog.")
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise PortalError(f"The desktop didn't answer {method} in time.")
                    try:
                        msg = queue.get(timeout=min(0.25, left))
                        break
                    except Exception:      # queue.Empty
                        continue
        finally:
            self.unmatch(rule)
        code, results = msg.body
        results = _unwrap(results)
        if code == 1:
            raise PortalDenied("You chose not to allow it in the desktop's dialog.")
        if code != 0:
            raise PortalError(f"The desktop refused {method} (code {code}).")
        return results


def _described(answer: dict) -> str:
    """What a Start answer gave, for the terminal (never the token itself)."""
    devices = int(answer.get("devices", 0) or 0)
    return (f"mouse {'yes' if devices & POINTER else 'no'}, keyboard {'yes' if devices & KEYBOARD else 'no'}, "
            f"{len(answer.get('streams') or [])} screen(s), fields {sorted(answer)}")


class RemoteDesktop:
    """One remote desktop session with a screen-cast stream: pointer, keyboard and the screen's frames."""

    def __init__(self, bus: PortalBus | None = None, state_file: Path = STATE_FILE):
        self.bus = bus or PortalBus()
        self.state_file = state_file
        self.session: str | None = None
        self.node: int | None = None
        self.size: tuple[int, int] | None = None        # the stream in logical pixels (the pointer's space)
        self.position: tuple[int, int] = (0, 0)
        self.devices = 0
        self.closed = threading.Event()
        self._closed_rule = None
        self._closed_filter = None

    # ---------------- start / stop ----------------
    def _load_token(self) -> str | None:
        return _saved_token(self.state_file)

    def _save_token(self, token: str | None) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"restore_token": token or ""}))
            self.state_file.chmod(0o600)
        except OSError:
            pass

    def start(self, stop: threading.Event | None = None, ask_timeout: float = 300.0, asking=None) -> None:
        """Open the session. The first time (or after the person withdrew it) the desktop asks; `asking`
        is called if its dialog is up, so LeanWatch can say what to do in it."""
        token = self._load_token()
        r = self._open(token, stop, ask_timeout, asking)
        if token and not (int(r.get("devices", 0)) & POINTER and r.get("streams")):
            # A remembered answer can come back without the screen (or the mouse): forget it and ask again.
            print(f"  Wayland: the desktop's remembered answer gave {_described(r)}; asking again.", file=sys.stderr,
                  flush=True)
            self._save_token(None)
            self.close()
            self.closed = threading.Event()
            r = self._open(None, stop, ask_timeout, asking)
        self.devices = int(r.get("devices", 0))
        if not self.devices & POINTER:
            # Don't keep this answer: restoring it would never give the mouse. Ask again next time.
            self._save_token(None)
            self.close()
            raise PortalDenied("The desktop shared the screen but not the mouse and keyboard. Next time, turn on "
                               "“Allow Remote Interaction” in the desktop's dialog before you click Share.")
        streams = r.get("streams") or []
        if not streams:
            self._save_token(None)       # restoring it would give no screen again
            self.close()
            print(f"  Wayland: the desktop's answer gave {_described(r)}.", file=sys.stderr, flush=True)
            raise PortalError("The desktop shared the mouse but no screen. Try again, and in the desktop's dialog "
                              "choose your screen before you click Share. If it happens again, send Claude the "
                              "terminal output.")
        self._save_token(r.get("restore_token"))   # only an answer that gave both the screen and the mouse
        node, props = streams[0]
        self.node = int(node)
        if "size" in props:
            self.size = (int(props["size"][0]), int(props["size"][1]))
        if "position" in props:
            self.position = (int(props["position"][0]), int(props["position"][1]))
        # GNOME drops the first events of a new session while it sets up its virtual devices. Spend
        # them on things no app acts on (a pointer move of zero pixels, a lone Shift tap), so the
        # agent's first click and first keys arrive.
        self.bus.call(RD, "NotifyPointerMotion", "oa{sv}dd", (self.session, {}, 0.0, 0.0))
        if self.devices & KEYBOARD:
            self.key(0xFFE1, True)
            self.key(0xFFE1, False)
        time.sleep(0.15)

    def _open(self, token: str | None, stop, ask_timeout: float, asking) -> dict:
        """CreateSession, SelectDevices (restoring `token` if given), SelectSources and Start; the Start answer."""
        b = self.bus
        r = b.request(RD, "CreateSession", "a{sv}", (), {"session_handle_token": ("s", b.token())})
        self.session = r["session_handle"]
        self._watch_closed()
        try:
            wanted = KEYBOARD | POINTER
            try:
                wanted &= int(b.get_property(RD, "AvailableDeviceTypes")) or wanted
            except Exception:
                pass
            options = {"types": ("u", wanted), "persist_mode": ("u", 2)}
            if token:
                options["restore_token"] = ("s", token)
            b.request(RD, "SelectDevices", "oa{sv}", (self.session,), options)
            cursor = CURSOR_HIDDEN
            try:
                modes = int(b.get_property(SC, "AvailableCursorModes"))
                cursor = CURSOR_HIDDEN if modes & CURSOR_HIDDEN else CURSOR_EMBEDDED if modes & CURSOR_EMBEDDED else 0
            except Exception:
                pass
            sources = {"types": ("u", MONITOR), "multiple": ("b", False)}
            if cursor:
                sources["cursor_mode"] = ("u", cursor)
            b.request(SC, "SelectSources", "oa{sv}", (self.session,), sources)
            # This is when the desktop shows its dialog (the first time): wait for the person.
            try:
                return b.request(RD, "Start", "osa{sv}", (self.session, ""), {}, stop=stop, timeout=ask_timeout,
                                 waiting=asking)
            except PortalDenied as err:
                if stop is not None and stop.is_set():
                    raise
                raise PortalDenied("I can only help after your desktop lets me see the screen and use the mouse. "
                                   "Try again and click Share, with “Allow Remote Interaction” turned on.") from err
        except BaseException:
            self.close()
            raise

    def _watch_closed(self) -> None:
        """The desktop closes the session when the person stops sharing (or the dialog is cancelled)."""
        session, closed = self.session, self.closed
        self._closed_rule = self.bus.match(interface="org.freedesktop.portal.Session", member="Closed", path=session)
        self._closed_filter = self.bus.router.filter(self._closed_rule)
        queue = self._closed_filter.__enter__()

        def wait():
            while not closed.is_set():
                try:
                    queue.get(timeout=0.5)
                    closed.set()
                except Exception:
                    if self.session != session:      # closed by LeanWatch (or a new session took over)
                        return

        threading.Thread(target=wait, daemon=True, name="leanwatch-portal-closed").start()

    def close(self) -> None:
        session, self.session = self.session, None
        if session:
            try:
                self.bus.call("org.freedesktop.portal.Session", "Close", path=session)
            except Exception:
                pass
        if self._closed_filter is not None:
            try:
                self._closed_filter.__exit__(None, None, None)
            except Exception:
                pass
            self._closed_filter = None
        if self._closed_rule is not None:
            self.bus.unmatch(self._closed_rule)
            self._closed_rule = None

    @property
    def active(self) -> bool:
        return self.session is not None and not self.closed.is_set()

    def _check(self) -> str:
        if self.closed.is_set():
            raise SharingStopped("You stopped the screen sharing, so I stopped.")
        if self.session is None:
            raise PortalError("The remote desktop session isn't running.")
        return self.session

    # ---------------- input ----------------
    def move(self, x: float, y: float) -> None:
        self.bus.call(RD, "NotifyPointerMotionAbsolute", "oa{sv}udd", (self._check(), {}, self.node, float(x), float(y)))

    def button(self, button: str, pressed: bool) -> None:
        self.bus.call(RD, "NotifyPointerButton", "oa{sv}iu", (self._check(), {}, BUTTONS[button], 1 if pressed else 0))

    def scroll_steps(self, axis: int, steps: int) -> None:
        """axis 0 = vertical (positive = down), 1 = horizontal (positive = right)."""
        self.bus.call(RD, "NotifyPointerAxisDiscrete", "oa{sv}ui", (self._check(), {}, axis, int(steps)))

    def key(self, keysym: int, pressed: bool) -> None:
        self.bus.call(RD, "NotifyKeyboardKeysym", "oa{sv}iu", (self._check(), {}, int(keysym), 1 if pressed else 0))

    def pipewire_fd(self) -> int:
        """A PipeWire connection (file descriptor) that can read this session's screen stream."""
        fd = self.bus.call(SC, "OpenPipeWireRemote", "oa{sv}", (self._check(), {}))[0]
        return fd.to_raw_fd() if hasattr(fd, "to_raw_fd") else int(fd)


# X keysyms for pyautogui's key names (the names Controller uses).
_KEYSYMS = {
    "enter": 0xFF0D, "return": 0xFF0D, "esc": 0xFF1B, "escape": 0xFF1B, "backspace": 0xFF08, "tab": 0xFF09,
    "space": 0x20, "delete": 0xFFFF, "del": 0xFFFF, "insert": 0xFF63, "home": 0xFF50, "end": 0xFF57,
    "pageup": 0xFF55, "pagedown": 0xFF56, "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53,
    "ctrl": 0xFFE3, "ctrlleft": 0xFFE3, "ctrlright": 0xFFE4, "alt": 0xFFE9, "altleft": 0xFFE9, "altright": 0xFFEA,
    "option": 0xFFE9, "shift": 0xFFE1, "shiftleft": 0xFFE1, "shiftright": 0xFFE2,
    "win": 0xFFEB, "winleft": 0xFFEB, "winright": 0xFFEC, "command": 0xFFEB, "super": 0xFFEB,
    "capslock": 0xFFE5, "numlock": 0xFF7F, "scrolllock": 0xFF14, "printscreen": 0xFF61, "prtsc": 0xFF61,
    "apps": 0xFF67, "pause": 0xFF13, "volumeup": 0x1008FF13, "volumedown": 0x1008FF11, "volumemute": 0x1008FF12,
}


def keysym_for(key: str) -> int:
    """pyautogui key name or one character -> X keysym."""
    k = key if len(key) == 1 else key.lower()
    if k in _KEYSYMS:
        return _KEYSYMS[k]
    if re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", k):
        return 0xFFBE + int(k[1:]) - 1
    if len(key) == 1:
        return char_keysym(key)
    raise ValueError(f"Unknown key {key!r}")


def char_keysym(ch: str) -> int:
    if ch == "\n":
        return 0xFF0D
    if ch == "\t":
        return 0xFF09
    code = ord(ch)
    if 0x20 <= code <= 0x7E or 0xA0 <= code <= 0xFF:
        return code                     # Latin-1 keysyms equal their code points
    return 0x01000000 | code            # Unicode keysym


class PortalBackend:
    """Controller backend for Wayland: real input through the RemoteDesktop portal.

    There is no way for an app to read where the real pointer is on Wayland, so position() is
    where LeanWatch last put it. That also means LeanWatch can't notice the person grabbing the
    mouse: Stop, Esc or the desktop's "stop sharing" button stop it.
    """

    def __init__(self, remote: RemoteDesktop, key_interval: float = 0.03):
        self.remote = remote
        self.key_interval = key_interval
        w, h = remote.size or (1920, 1080)
        self._size = (w, h)
        self._pos = (w // 2, h // 2)

    def size(self):
        return self._size

    def position(self):
        return self._pos

    def move_to(self, x, y):
        x = min(max(int(x), 0), self._size[0] - 1)
        y = min(max(int(y), 0), self._size[1] - 1)
        self.remote.move(x, y)
        self._pos = (x, y)

    def mouse_down(self, button="left"):
        self.remote.button(button, True)

    def mouse_up(self, button="left"):
        self.remote.button(button, False)

    def click(self, button="left"):
        self.remote.button(button, True)
        time.sleep(0.02)
        self.remote.button(button, False)

    def scroll(self, dx, dy):
        # pyautogui's sign: dy > 0 scrolls up. The portal's: positive steps scroll down / right.
        if dy:
            self.remote.scroll_steps(0, -int(dy))
        if dx:
            self.remote.scroll_steps(1, int(dx))

    def key_down(self, key):
        self.remote.key(keysym_for(key), True)

    def key_up(self, key):
        self.remote.key(keysym_for(key), False)

    def nudge(self):
        """Move the pointer one pixel and back: makes the desktop send a fresh screen frame."""
        x, y = self._pos
        self.remote.move(x + 1 if x + 1 < self._size[0] else x - 1, y)
        time.sleep(0.05)
        self.remote.move(x, y)

    def write(self, text):
        for ch in text:
            sym = char_keysym(ch)
            self.remote.key(sym, True)
            self.remote.key(sym, False)
            time.sleep(self.key_interval)


# ---------------- the screen ----------------
_HELPER = Path(__file__).with_name("pw_frames.py")
_PROBE = ("import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); "
          "import sys; sys.exit(0 if Gst.ElementFactory.find('pipewiresrc') else 3)")
_gst_python: list = []      # cached result of find_gst_python()


def _clean_env() -> dict:
    """The environment for the system's Python: without this Python's (e.g. Anaconda's) settings."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    prefix = sys.prefix
    if env.get("LD_LIBRARY_PATH"):
        keep = [p for p in env["LD_LIBRARY_PATH"].split(":") if p and not p.startswith(prefix)]
        env["LD_LIBRARY_PATH"] = ":".join(keep)
    return env


def find_gst_python() -> str | None:
    """A Python with GStreamer's bindings and the PipeWire plugin: the system's own (GNOME needs both)."""
    if _gst_python:
        return _gst_python[0]
    candidates = ["/usr/bin/python3"]
    # The bindings are built for one Python version (e.g. _gi.cpython-312-...so): try that one too.
    for so in sorted(glob.glob("/usr/lib/python3/dist-packages/gi/_gi.cpython-3*.so"), reverse=True):
        m = re.search(r"cpython-3(\d+)", so)
        if m:
            candidates.append(f"/usr/bin/python3.{m.group(1)}")
    found = None
    for exe in dict.fromkeys(candidates):
        if not os.path.exists(exe):
            continue
        try:
            if subprocess.run([exe, "-s", "-c", _PROBE], env=_clean_env(), capture_output=True, timeout=20).returncode == 0:
                found = exe
                break
        except (OSError, subprocess.TimeoutExpired):
            continue
    _gst_python.append(found)
    return found


class PipeWireFrames:
    """Reads the session's screen stream through a small GStreamer helper in the system's Python."""

    def __init__(self, remote: RemoteDesktop, python: str):
        fd = remote.pipewire_fd()
        try:
            self.proc = subprocess.Popen(
                [python, "-s", str(_HELPER), str(fd), str(remote.node)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, pass_fds=(fd,), env=_clean_env(),
                bufsize=0)     # unbuffered: select() must see every byte that hasn't been read yet
        finally:
            os.close(fd)      # the helper has its own copy
        self._lock = threading.Lock()
        header = self._read_line(timeout=15)
        if header != "ready":
            self.close()
            raise PortalError(f"The screen reader didn't start ({header or 'no answer'}).")

    def _read(self, n: int, timeout: float, line: bool = False) -> bytes:
        """Read n bytes (or one line) from the helper, giving up after `timeout` seconds."""
        import select

        fd = self.proc.stdout.fileno()
        end = time.monotonic() + timeout
        buf = bytearray()
        while (not buf.endswith(b"\n")) if line else len(buf) < n:
            left = end - time.monotonic()
            if left <= 0:
                break
            r, _, _ = select.select([fd], [], [], min(left, 0.5))
            if r:
                chunk = os.read(fd, 1 if line else n - len(buf))
                if not chunk:
                    break         # the helper exited
                buf += chunk
        return bytes(buf)

    def _read_line(self, timeout: float) -> str:
        data = self._read(0, timeout, line=True)
        return data.decode(errors="replace").strip() if data.endswith(b"\n") else ""

    def grab(self, since: float | None = None, wait: float = 0.5, nudge=None):
        """The newest frame. With `since` (a time.monotonic() value, the same clock in every process),
        wait up to `wait` seconds for a frame taken after it: after LeanWatch hides its windows, the
        desktop sends a new frame without them.

        The desktop only sends a frame when something on the screen changes, so right after the
        stream starts there may be none yet: then `nudge` (moves the pointer by a pixel) makes one.
        """
        from PIL import Image

        with self._lock:
            line = self._ask(wait if since is not None else 0.0, since)
            if line.startswith("none") and nudge is not None:
                nudge()
                line = self._ask(3.0, time.monotonic() - 0.2)
            parts = line.split()
            if len(parts) != 4 or parts[0] != "frame":
                if not line:
                    # No answer in time: a late one would be taken for the next picture. This reader is done.
                    self.close()
                raise PortalError(f"The screen reader sent no picture ({line or 'no answer'}).")
            w, h, stride = int(parts[1]), int(parts[2]), int(parts[3])
            data = self._read(stride * h, timeout=10)
            if len(data) != stride * h:
                self.close()
                raise PortalError("The screen reader sent a broken picture.")
        return Image.frombuffer("RGB", (w, h), data, "raw", "BGRX", stride, 1).copy()

    def _ask(self, wait: float, since: float | None = None) -> str:
        if self.proc.poll() is not None:
            raise PortalError("The screen reader stopped.")
        self.proc.stdin.write(f"grab {wait:.3f} {since if since is not None else -1:.4f}\n".encode())
        self.proc.stdin.flush()
        return self._read_line(timeout=wait + 10)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b"quit\n")
                self.proc.stdin.flush()
                self.proc.wait(2)
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()


def portal_screenshot(bus: PortalBus | None = None, stop: threading.Event | None = None):
    """One screenshot through the Screenshot portal (GNOME flashes the screen and plays a sound).
    The first time, the desktop asks the person to allow screenshots."""
    from urllib.parse import unquote, urlparse

    from PIL import Image

    own = bus is None
    bus = bus or PortalBus()
    try:
        r = bus.request(SHOT, "Screenshot", "sa{sv}", ("",), {"interactive": ("b", False)}, stop=stop, timeout=120)
    finally:
        if own:
            bus.close()
    path = unquote(urlparse(r["uri"]).path)
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    finally:
        try:
            os.remove(path)          # GNOME saves it in ~/Pictures; it was only for LeanWatch
        except OSError:
            pass


def crop_to_stream(img, position, size):
    """The Screenshot portal pictures every monitor; the session shares one, and the pointer's coordinates
    are that monitor's. Cut it out (its place in the picture, trying the usual scales: 1:1, then the
    monitor filling the picture's height or width)."""
    x, y = position
    w, h = size
    if (x, y) == (0, 0) and abs(img.width / img.height - w / h) < 0.01:
        return img
    for s in (1.0, img.height / (y + h), img.width / (x + w)):
        box = (round(x * s), round(y * s), round((x + w) * s), round((y + h) * s))
        if box[2] <= img.width and box[3] <= img.height and box[2] > box[0] and box[3] > box[1]:
            return img.crop(box)
    return img


class WaylandScreen:
    """The remote desktop session for one task: backend (mouse and keyboard) and grab() for ScreenCapture."""

    def __init__(self, stop: threading.Event | None = None, asking=None):
        self.remote = RemoteDesktop()
        self.remote.start(stop=stop, asking=asking)
        self.backend = PortalBackend(self.remote)
        self.note = ""
        self.frames = self._start_frames()

    def _start_frames(self):
        python = find_gst_python()
        if not python:
            self.note = ("GStreamer's PipeWire plugin wasn't found (sudo apt install gstreamer1.0-pipewire python3-gi), "
                         "so LeanWatch takes screenshots instead: the screen flashes each time.")
            return None
        try:
            return PipeWireFrames(self.remote, python)
        except Exception as err:
            self.note = f"Reading the screen stream failed ({err}); using screenshots instead."
            return None

    def grab(self, since: float | None = None):
        """The whole screen (in the screen's own pixels). See PipeWireFrames.grab for `since`."""
        if self.remote.closed.is_set():
            raise SharingStopped("You stopped the screen sharing, so I stopped.")
        if self.frames is not None:
            try:
                return self.frames.grab(since=since, nudge=self.backend.nudge)
            except PortalError:
                # A stuck or crashed reader: start a new one once, else use screenshots for the rest of the task.
                self.frames.close()
                self.frames = self._start_frames()
                if self.frames is not None:
                    return self.frames.grab(since=since, nudge=self.backend.nudge)
        return crop_to_stream(portal_screenshot(self.remote.bus), self.remote.position, self.backend.size())

    def close(self) -> None:
        if self.frames is not None:
            self.frames.close()
            self.frames = None
        self.remote.close()
