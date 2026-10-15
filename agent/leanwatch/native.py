"""Operating-system details for the floating bubble.

Everything here is best effort: if a call is not available (old OS version, missing
optional package), the bubble still works, just with one convenience fewer.

- prepare_window / after_show: stay on top, on every virtual desktop / Space
- exclude_from_capture: keep our windows out of screenshots, so the model never sees them
  (when the OS can't, bubble.Shield hides them for the moment of the screenshot instead)
- focus_tracker: remember the app the person was using and give it the keyboard back
- activate_own: bring the command box to the front with the keyboard in it
- register_hotkey: a keyboard shortcut that opens the box from anywhere
- supports_translucency: round see-through windows need a compositor on Linux
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable

DEFAULT_HOTKEY = "ctrl+alt+space"


def parse_hotkey(text: str) -> tuple[set[str], str]:
    """'Ctrl+Alt+Space' -> ({'ctrl', 'alt'}, 'space'). Modifiers: ctrl, alt (option), shift, cmd (win, super)."""
    names = {"control": "ctrl", "ctl": "ctrl", "option": "alt", "opt": "alt", "win": "cmd", "super": "cmd",
             "meta": "cmd", "command": "cmd"}
    parts = [names.get(p.strip().lower(), p.strip().lower()) for p in text.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty shortcut")
    *mods, key = parts
    bad = [m for m in mods if m not in ("ctrl", "alt", "shift", "cmd")]
    if bad or key in ("ctrl", "alt", "shift", "cmd"):
        raise ValueError(f"not a shortcut: {text!r}")
    if not mods:
        raise ValueError("a shortcut needs at least one of ctrl, alt, shift, cmd")
    return set(mods), key


class FocusTracker:
    """Remembers the last app/window, other than LeanWatch, that had the keyboard."""

    def __init__(self, current: Callable[[], object], is_ours: Callable[[object], bool], activate: Callable[[object], None]):
        self._current = current
        self._is_ours = is_ours
        self._activate = activate
        self.last = None

    def poll(self) -> None:
        try:
            h = self._current()
            if h and not self._is_ours(h):
                self.last = h
        except Exception:
            pass

    def restore(self) -> None:
        if self.last is None:
            return
        try:
            self._activate(self.last)
        except Exception:
            pass


class Native:
    """Fallback for any platform: plain Qt behaviour."""

    name = "generic"

    def supports_translucency(self) -> bool:
        return True

    def prepare_window(self, widget) -> None:
        pass

    def after_show(self, widget) -> None:
        pass

    def exclude_from_capture(self, widget) -> bool:
        return False

    def activate_own(self, widget) -> None:
        widget.raise_()

    def focus_tracker(self) -> FocusTracker:
        return FocusTracker(lambda: None, lambda h: True, lambda h: None)

    def register_hotkey(self, combo: str, callback: Callable[[], None]):
        """Returns a function that unregisters it, or None if shortcuts aren't available here."""
        return _pynput_hotkey(combo, callback)

    def notes(self) -> list[str]:
        return []


def _pynput_hotkey(combo: str, callback):
    try:
        from pynput import keyboard
    except Exception:
        return None
    mods, key = parse_hotkey(combo)
    spec = "+".join([f"<{m}>" for m in sorted(mods)] + [f"<{key}>" if len(key) > 1 else key])
    listener = keyboard.GlobalHotKeys({spec: callback})
    listener.daemon = True
    listener.start()
    return listener.stop


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
class WindowsNative(Native):
    name = "windows"
    WDA_EXCLUDEFROMCAPTURE = 0x11   # Windows 10 version 2004 and later
    VK = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
          **{f"f{i}": 0x6F + i for i in range(1, 25)}}
    MODS = {"alt": 0x1, "ctrl": 0x2, "shift": 0x4, "cmd": 0x8}
    MOD_NOREPEAT = 0x4000

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ct, self.wt = ctypes, wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u = self.user32
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.SetForegroundWindow.restype = wintypes.BOOL
        u.BringWindowToTop.argtypes = [wintypes.HWND]
        u.IsWindow.argtypes = [wintypes.HWND]
        u.IsWindow.restype = wintypes.BOOL
        u.IsIconic.argtypes = [wintypes.HWND]
        u.IsIconic.restype = wintypes.BOOL
        u.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.SetWindowDisplayAffinity.restype = wintypes.BOOL
        u.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        u.AttachThreadInput.restype = wintypes.BOOL
        u.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        u.RegisterHotKey.restype = wintypes.BOOL
        u.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        u.GetMessageW.restype = wintypes.BOOL
        u.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def _pid_of(self, hwnd) -> int:
        pid = self.wt.DWORD(0)
        self.user32.GetWindowThreadProcessId(hwnd, self.ct.byref(pid))
        return pid.value

    def exclude_from_capture(self, widget) -> bool:
        return bool(self.user32.SetWindowDisplayAffinity(int(widget.winId()), self.WDA_EXCLUDEFROMCAPTURE))

    def _force_foreground(self, hwnd) -> None:
        """SetForegroundWindow, with the usual AttachThreadInput step so Windows lets us do it."""
        u = self.user32
        fg = u.GetForegroundWindow()
        fg_thread = u.GetWindowThreadProcessId(fg, None) if fg else 0
        me = self.kernel32.GetCurrentThreadId()
        attached = bool(fg_thread and fg_thread != me and u.AttachThreadInput(fg_thread, me, True))
        try:
            u.BringWindowToTop(hwnd)
            u.SetForegroundWindow(hwnd)
        finally:
            if attached:
                u.AttachThreadInput(fg_thread, me, False)

    def activate_own(self, widget) -> None:
        widget.raise_()
        self._force_foreground(int(widget.winId()))

    def focus_tracker(self) -> FocusTracker:
        me = os.getpid()

        def activate(hwnd):
            if self.user32.IsWindow(hwnd) and not self.user32.IsIconic(hwnd):
                self._force_foreground(hwnd)

        return FocusTracker(self.user32.GetForegroundWindow, lambda h: self._pid_of(h) == me, activate)

    def register_hotkey(self, combo: str, callback):
        mods, key = parse_hotkey(combo)
        vk = self.VK.get(key) or (ord(key.upper()) if len(key) == 1 and key.isalnum() else None)
        if vk is None:
            return None
        flags = self.MOD_NOREPEAT
        for m in mods:
            flags |= self.MODS[m]
        ready = threading.Event()
        state = {"thread": 0, "ok": False}
        WM_HOTKEY, WM_QUIT = 0x0312, 0x0012

        def loop():
            u = self.user32
            state["thread"] = self.kernel32.GetCurrentThreadId()
            state["ok"] = bool(u.RegisterHotKey(None, 1, flags, vk))   # the hotkey belongs to this thread
            ready.set()
            if not state["ok"]:
                return
            msg = self.wt.MSG()
            while u.GetMessageW(self.ct.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    callback()
            u.UnregisterHotKey(None, 1)

        threading.Thread(target=loop, daemon=True, name="leanwatch-hotkey").start()
        ready.wait(2)
        if not state["ok"]:
            return None   # taken by another app
        return lambda: self.user32.PostThreadMessageW(state["thread"], WM_QUIT, 0, 0)


# ---------------------------------------------------------------------------
# macOS (needs pyobjc-framework-Cocoa for the native parts)
# ---------------------------------------------------------------------------
class MacNative(Native):
    name = "macos"
    # NSWindowCollectionBehavior: CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary
    BEHAVIOR = (1 << 0) | (1 << 4) | (1 << 6) | (1 << 8)
    STATUS_LEVEL = 25          # NSStatusWindowLevel: above normal and floating windows

    def __init__(self):
        try:
            import AppKit  # noqa: F401
            import objc  # noqa: F401

            self.appkit = AppKit
            self.objc = objc
        except Exception:
            self.appkit = None

    def _nswindow(self, widget):
        view = self.objc.objc_object(c_void_p=int(widget.winId()))
        return view.window()

    def setup_app(self) -> None:
        """No Dock icon (an accessory app), which also lets our windows float over full-screen apps."""
        if self.appkit:
            self.appkit.NSApplication.sharedApplication().setActivationPolicy_(1)   # Accessory

    def prepare_window(self, widget) -> None:
        if not self.appkit:
            return
        win = self._nswindow(widget)
        if win is None:
            return
        win.setCollectionBehavior_(self.BEHAVIOR)
        win.setLevel_(self.STATUS_LEVEL)
        win.setHidesOnDeactivate_(False)

    def after_show(self, widget) -> None:
        self.prepare_window(widget)   # Qt may reset the level when it shows a window again

    def exclude_from_capture(self, widget) -> bool:
        # NSWindowSharingNone keeps the window out of other apps' captures on older macOS, but newer
        # versions ignore it for some capture paths, so LeanWatch still hides the window to be sure.
        if self.appkit:
            try:
                self._nswindow(widget).setSharingType_(0)
            except Exception:
                pass
        return False

    def activate_own(self, widget) -> None:
        if self.appkit:
            app = self.appkit.NSApplication.sharedApplication()
            try:
                app.activate()                          # macOS 14+
            except AttributeError:
                app.activateIgnoringOtherApps_(True)
        widget.raise_()

    def focus_tracker(self) -> FocusTracker:
        if not self.appkit:
            return super().focus_tracker()
        ws = self.appkit.NSWorkspace.sharedWorkspace()
        me = os.getpid()

        def activate(app):
            if not app.isTerminated():
                app.activateWithOptions_(1 << 1)        # NSApplicationActivateIgnoringOtherApps (ignored on 14+, harmless)

        return FocusTracker(ws.frontmostApplication, lambda a: a.processIdentifier() == me, activate)

    def register_hotkey(self, combo: str, callback):
        if not self.appkit:
            return None
        mods, key = parse_hotkey(combo)
        A = self.appkit
        want = 0
        for m, bit in (("shift", 1 << 17), ("ctrl", 1 << 18), ("alt", 1 << 19), ("cmd", 1 << 20)):
            if m in mods:
                want |= bit
        mask_all = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20)
        char = " " if key == "space" else key

        def matches(event) -> bool:
            if (event.modifierFlags() & mask_all) != want:
                return False
            if key == "space":
                return event.keyCode() == 49
            return (event.charactersIgnoringModifiers() or "").lower() == char

        def on_global(event):
            if matches(event):
                callback()

        def on_local(event):
            if matches(event):
                callback()
                return None        # swallow it inside LeanWatch
            return event

        key_down = 1 << 10          # NSEventMaskKeyDown
        g = A.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(key_down, on_global)   # needs Accessibility
        local = A.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(key_down, on_local)

        def stop():
            for monitor in (g, local):
                if monitor is not None:
                    A.NSEvent.removeMonitor_(monitor)

        return stop

    def notes(self) -> list[str]:
        if not self.appkit:
            return ["Install pyobjc-framework-Cocoa so the bubble floats over full-screen apps and every Space: "
                    "pip install pyobjc-framework-Cocoa"]
        return []


# ---------------------------------------------------------------------------
# Linux (X11; under Wayland LeanWatch runs through XWayland)
# ---------------------------------------------------------------------------
class X11Native(Native):
    name = "x11"

    def __init__(self):
        try:
            from Xlib import X, display, protocol

            self.X, self.protocol = X, protocol
            self.dpy = display.Display()
            self.root = self.dpy.screen().root
            self.atom = lambda name: self.dpy.intern_atom(name)
            self._lock = threading.Lock()
        except Exception:
            self.dpy = None

    def supports_translucency(self) -> bool:
        if self.dpy is None:
            return True
        with self._lock:
            owner = self.dpy.get_selection_owner(self.atom(f"_NET_WM_CM_S{self.dpy.get_default_screen()}"))
        return owner not in (0, self.X.NONE)

    def _send(self, window_id: int, message: str, data: list[int]) -> None:
        win = self.dpy.create_resource_object("window", window_id)
        ev = self.protocol.event.ClientMessage(window=win, client_type=self.atom(message),
                                               data=(32, (data + [0] * 5)[:5]))
        self.root.send_event(ev, event_mask=self.X.SubstructureRedirectMask | self.X.SubstructureNotifyMask)
        self.dpy.flush()

    def after_show(self, widget) -> None:
        """On every virtual desktop and above other windows. Window managers drop this when a window hides."""
        if self.dpy is None:
            return
        with self._lock:
            wid = int(widget.winId())
            self._send(wid, "_NET_WM_STATE", [1, self.atom("_NET_WM_STATE_STICKY"), self.atom("_NET_WM_STATE_ABOVE"), 1])
            self._send(wid, "_NET_WM_DESKTOP", [0xFFFFFFFF, 1])

    def focus_tracker(self) -> FocusTracker:
        if self.dpy is None:
            return super().focus_tracker()
        me = os.getpid()

        def current():
            with self._lock:
                prop = self.root.get_full_property(self.atom("_NET_ACTIVE_WINDOW"), self.X.AnyPropertyType)
            return int(prop.value[0]) if prop and len(prop.value) and prop.value[0] else None

        def is_ours(wid):
            with self._lock:
                win = self.dpy.create_resource_object("window", wid)
                prop = win.get_full_property(self.atom("_NET_WM_PID"), self.X.AnyPropertyType)
            return bool(prop and len(prop.value) and int(prop.value[0]) == me)

        def activate(wid):
            with self._lock:
                self._send(wid, "_NET_ACTIVE_WINDOW", [2, self.X.CurrentTime, 0])   # 2 = asked for by a pager/user tool

        return FocusTracker(current, is_ours, activate)

    def notes(self) -> list[str]:
        if os.environ.get("WAYLAND_DISPLAY"):
            return ["Wayland session: LeanWatch runs through XWayland. Some desktops keep Wayland apps above it; "
                    "for full control log in with an X11 (Xorg) session."]
        return []


def for_platform(platform: str = sys.platform) -> Native:
    try:
        if platform == "win32":
            return WindowsNative()
        if platform == "darwin":
            return MacNative()
        if platform.startswith("linux") and os.environ.get("DISPLAY"):
            return X11Native()
    except Exception:
        pass
    return Native()
