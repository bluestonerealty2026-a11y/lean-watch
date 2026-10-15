"""Run LeanWatch as a desktop helper: the floating bubble plus the agent, in one process.

The bubble (Qt) owns the main thread, as macOS requires. The agent's local server
(HTTP API and the optional web control center) runs on a background thread.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys
import threading
import webbrowser
from pathlib import Path

from .config import Config
from .native import DEFAULT_HOTKEY


class ServerThread(threading.Thread):
    """aiohttp's server on its own event loop, so the GUI can keep the main thread."""

    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True, name="leanwatch-server")
        self.app, self.host, self.port = app, host, port
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        from aiohttp import web

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        runner = web.AppRunner(self.app)
        try:
            self.loop.run_until_complete(runner.setup())
            self.loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
        except Exception as err:   # usually: the port is taken, because LeanWatch is already running
            self.error = err
            self.ready.set()
            self.loop.run_until_complete(runner.cleanup())
            return
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(runner.cleanup())
            self.loop.close()

    def stop(self) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.join(5)


def _xcb_cursor_available() -> bool:
    """Qt 6.5+ can't start its X11 plugin without libxcb-cursor, and then aborts without a useful message."""
    if os.environ.get("QT_QPA_PLATFORM", "xcb").split(":")[0] not in ("xcb", ""):
        return True
    import ctypes
    import ctypes.util

    try:
        ctypes.CDLL("libxcb-cursor.so.0")      # what Qt itself loads (also finds it in LD_LIBRARY_PATH)
        return True
    except OSError:
        return ctypes.util.find_library("xcb-cursor") is not None


def gnome_shortcut_installed() -> bool:
    try:
        from .gnome_shortcut import installed

        return installed()
    except Exception:
        return False


def run(cfg: Config, web_page: bool = False, hotkey: str | None = DEFAULT_HOTKEY) -> int:
    from .native import enable_dpi_awareness

    enable_dpi_awareness()        # Windows: before Qt, pyautogui or mss choose a DPI mode
    if sys.platform == "darwin":
        # Before Qt starts: QWidget.raise_() must not activate the app (it would take the keyboard from the
        # app LeanWatch is working in), and Qt must not turn LeanWatch into a Dock app at launch.
        os.environ.setdefault("QT_MAC_SET_RAISE_PROCESS", "0")
        os.environ.setdefault("QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM", "1")
    if sys.platform.startswith("linux") and not _xcb_cursor_available():
        print("Qt needs the libxcb-cursor library to show the bubble. Install it, e.g. on Ubuntu/Debian:")
        print("  sudo apt install libxcb-cursor0        (Fedora: sudo dnf install xcb-util-cursor)")
        return 1
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        # Qt can't place always-on-top windows on Wayland, and pyautogui needs X anyway: use XWayland.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The floating bubble needs PySide6:  pip install PySide6-Essentials")
        print("Opening the web control center instead.")
        from .server import serve

        cfg.open_browser = True
        serve(cfg)
        return 0

    from .bubble import BubbleApp
    from .brains import model_name
    from .native import for_platform
    from .server import SESSION, make_app, write_session_file

    qt = QApplication.instance() or QApplication(sys.argv[:1])
    qt.setApplicationName("LeanWatch")
    qt.setQuitOnLastWindowClosed(False)     # closing the box only shrinks it back to the bubble
    native = for_platform()
    if hasattr(native, "setup_app"):
        native.setup_app()

    token = secrets.token_urlsafe(24)
    app = make_app(cfg, token)
    session = app[SESSION]
    server = ServerThread(app, cfg.host, cfg.port)
    server.start()
    server.ready.wait(10)
    if server.error is not None:
        print(f"LeanWatch could not start on 127.0.0.1:{cfg.port} ({server.error}).")
        print("It is probably running already: look for the teal bubble. Or pick another port with --port.")
        return 0      # not an error: a start-at-login service must not keep retrying
    session_path = write_session_file(cfg, token)
    url = f"http://127.0.0.1:{cfg.port}/?token={token}"

    bubble = BubbleApp(session, native, settings_path=Path.home() / ".leanwatch" / "bubble.json")
    stop_hotkey = None
    if hotkey:
        try:
            stop_hotkey = native.register_hotkey(hotkey, bubble.hotkey_pressed.emit)
        except Exception as err:
            print(f"  Shortcut {hotkey} is not available: {err}")
    # On Wayland our key grab only sees keys while an X11 window is in front; a shortcut in GNOME's
    # own settings (python -m leanwatch --shortcut on) works in every app.
    from . import wayland

    on_wayland = wayland.wanted()
    shortcut_works = gnome_shortcut_installed() if on_wayland else stop_hotkey is not None
    bubble.shortcut_ok = shortcut_works            # decides whether "Hide the bubble" is safe to offer
    bubble.show()

    pretty = "+".join(p.capitalize() for p in hotkey.split("+")) if hotkey else ""
    print("LeanWatch is ready.")
    print("  Click the teal bubble" + (f" or press {pretty}" if hotkey and shortcut_works else "")
          + " and tell me what you need.")
    if on_wayland and not shortcut_works:
        print("  For a keyboard shortcut that works in every app on Wayland, run once:  python -m leanwatch --shortcut on")
    print(f"  Brain: {'local open model' if cfg.brain == 'local' else 'Claude API'}: {model_name(cfg)}"
          + ("   DRY RUN: the mouse will not move" if cfg.dry_run else ""))
    if hotkey and not stop_hotkey and not on_wayland:
        print(f"  (The shortcut {pretty} could not be set up on this computer; the bubble still works.)")
    for note in native.notes():
        print(f"  Note: {note}")
    if session.screen_error:
        print(f"  Warning: {session.screen_error}")
    print(f"  Web control center (optional): {url}")
    print("  Quit: right-click the bubble > Quit, or Ctrl+C here.")
    if web_page:
        webbrowser.open(url)
    if cfg.preload and cfg.brain == "local":
        from .server import preload_model

        preload_model(session)

    # Let Ctrl+C in the terminal quit: Python only handles signals while it runs Python code.
    signal.signal(signal.SIGINT, lambda *_: qt.quit())
    ticker = QTimer()
    ticker.timeout.connect(lambda: None)
    ticker.start(300)
    try:
        return qt.exec()
    finally:
        session.shutdown()          # also ends the model's process
        if stop_hotkey:
            try:
                stop_hotkey()
            except Exception:
                pass
        server.stop()
        try:
            session_path.unlink()
        except OSError:
            pass
