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
import time
from typing import Callable

# Ctrl+Option+Space switches the input language on a Mac, so macOS gets its own default.
DEFAULT_HOTKEY = "ctrl+shift+space" if sys.platform == "darwin" else "ctrl+alt+space"


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

    def restore(self, target=None) -> None:
        """Give the keyboard to `target` (e.g. the app the agent was working in), else to the last app used."""
        target = target if target is not None else self.last
        if target is None:
            return
        try:
            self._activate(target)
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

    def keep_on_top(self, widget) -> None:
        """Called every couple of seconds: other always-on-top windows may have covered ours."""
        widget.raise_()

    def override_redirect_bubble(self) -> bool:
        """Linux/X11: make the round bubble a window the window manager doesn't manage (always on top)."""
        return False

    def window_rect_px(self, widget):
        """The window's rectangle in mouse pixels (pyautogui/mss), or None to let the caller compute it."""
        return None

    def activate_own(self, widget) -> None:
        widget.raise_()

    def focus_tracker(self) -> FocusTracker:
        return FocusTracker(lambda: None, lambda h: True, lambda h: None)

    def register_hotkey(self, combo: str, callback: Callable[[], None]):
        """Returns a function that unregisters it, or None if shortcuts aren't available here."""
        return _pynput_hotkey(combo, callback)

    def missing_permissions(self) -> list[tuple[str, str]]:
        """Operating-system permissions LeanWatch still needs: (what, settings link)."""
        return []

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
def enable_dpi_awareness(platform: str = sys.platform) -> None:
    """Windows: per-monitor DPI awareness (v2), set before Qt, pyautogui or mss can pick a worse mode.

    The first call in a process wins. pyautogui asks for "system aware" when it is imported, which
    makes Windows stretch our windows and shift coordinates on a second monitor with another scale.
    """
    if platform != "win32":
        return
    import ctypes

    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):   # PER_MONITOR_AWARE_V2
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


class WindowsNative(Native):
    name = "windows"
    WDA_EXCLUDEFROMCAPTURE = 0x11   # Windows 10 version 2004 (build 19041) and later
    HWND_TOPMOST = -1
    SWP_FLAGS = 0x0001 | 0x0002 | 0x0010 | 0x0200   # NOSIZE | NOMOVE | NOACTIVATE | NOOWNERZORDER
    # Shell surfaces that are not "the app the person was using".
    SHELL_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow", "TopLevelWindowForOverflowXamlIsland",
                     "XamlExplorerHostIslandWindow", "Windows.UI.Core.CoreWindow", "ForegroundStaging", "MultitaskingViewFrame"}
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
        u.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowDisplayAffinity.restype = wintypes.BOOL
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        u.SetWindowPos.restype = wintypes.BOOL
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        u.GetAncestor.restype = wintypes.HWND
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
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
        # Older builds treat this like WDA_MONITOR and paint a black shape into screenshots instead.
        if sys.getwindowsversion().build < 19041:
            return False
        hwnd = int(widget.winId())
        if not self.user32.SetWindowDisplayAffinity(hwnd, self.WDA_EXCLUDEFROMCAPTURE):
            return False      # e.g. refused for some layered (see-through) windows; the bubble hides instead
        got = self.wt.DWORD(0)
        return bool(self.user32.GetWindowDisplayAffinity(hwnd, self.ct.byref(got))) and got.value == self.WDA_EXCLUDEFROMCAPTURE

    def keep_on_top(self, widget) -> None:
        # All always-on-top windows share one layer and the last one raised wins (the taskbar after a
        # click, video players, meeting bars), so pin ours back on top, without taking focus.
        self.user32.SetWindowPos(int(widget.winId()), self.HWND_TOPMOST, 0, 0, 0, 0, self.SWP_FLAGS)

    def after_show(self, widget) -> None:
        self.keep_on_top(widget)

    def window_rect_px(self, widget):
        r = self.wt.RECT()
        if not self.user32.GetWindowRect(int(widget.winId()), self.ct.byref(r)):
            return None
        return r.left, r.top, r.right, r.bottom      # physical pixels: the same space as pyautogui and mss

    def _class_name(self, hwnd) -> str:
        buf = self.ct.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

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

        def current():
            h = self.user32.GetForegroundWindow()
            if not h:
                return None
            root = self.user32.GetAncestor(h, 2) or h          # GA_ROOT: the top-level window
            return None if self._class_name(root) in self.SHELL_CLASSES else root

        def activate(hwnd):
            if not self.user32.IsWindow(hwnd):
                return
            if self.user32.IsIconic(hwnd):
                self.user32.ShowWindow(hwnd, 9)                 # SW_RESTORE
            self._force_foreground(hwnd)

        return FocusTracker(current, lambda h: self._pid_of(h) == me, activate)

    def register_hotkey(self, combo: str, callback):
        # RegisterHotKey needs no permissions or keyboard hook, and a hotkey press gives LeanWatch the
        # right to bring its box to the front. It fails cleanly if another app already uses the combination.
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
    """macOS, through pyobjc. Details from Qt's cocoa plugin source and Apple's AppKit documentation:

    - Qt.Tool windows are NSPanels; with level 25 (NSStatusWindowLevel), an accessory app (no Dock icon)
      and the collection behaviours below they float over full-screen apps and on every Space.
    - QWidget.raise_() would activate the whole app (stealing the keyboard) unless QT_MAC_SET_RAISE_PROCESS=0,
      which desktop.py sets; to stay in front we use orderFrontRegardless, which never activates.
    - Screenshots: NSWindowSharingNone is a legacy setting (at best a black box), so the bubble steps aside.
    """

    name = "macos"
    # CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary. Assigned, never OR-ed:
    # Qt's default includes MoveToActiveSpace, which conflicts with CanJoinAllSpaces.
    BEHAVIOR = (1 << 0) | (1 << 4) | (1 << 6) | (1 << 8)
    JOIN_ALL_APPS = 1 << 18      # CanJoinAllApplications (macOS 13+): a floating overlay, outside Stage Manager
    STATUS_LEVEL = 25            # above the Dock (20) and the menu bar (24), below pop-up menus (101)
    KEYS_CTRL, KEYS_SHIFT, KEYS_ALT, KEYS_CMD = 0x1000, 0x200, 0x800, 0x100   # Carbon modifier masks
    SETTINGS = "x-apple.systempreferences:com.apple.preference.security?"

    def __init__(self):
        try:
            import AppKit  # noqa: F401
            import objc  # noqa: F401

            self.appkit = AppKit
            self.objc = objc
        except Exception:
            self.appkit = None
        self._observer = None

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
        try:
            win.setCollectionBehavior_(self.BEHAVIOR | self.JOIN_ALL_APPS)
        except Exception:
            win.setCollectionBehavior_(self.BEHAVIOR)
        win.setLevel_(self.STATUS_LEVEL)
        win.setHidesOnDeactivate_(False)
        win.setCanHide_(False)          # survive Cmd-H and "Hide Others"
        if widget.isVisible():
            win.orderFrontRegardless()

    def after_show(self, widget) -> None:
        self.prepare_window(widget)     # Qt resets the level whenever it (re)creates or reshows the window

    def keep_on_top(self, widget) -> None:
        if self.appkit:
            try:
                self._nswindow(widget).orderFrontRegardless()   # front of its level, without activating LeanWatch
            except Exception:
                pass

    def exclude_from_capture(self, widget) -> bool:
        return False                    # see the class notes: the bubble steps aside for each screenshot

    def activate_own(self, widget) -> None:
        if self.appkit:
            app = self.appkit.NSApplication.sharedApplication()
            if app.respondsToSelector_("activate"):
                app.activate()                          # macOS 14+
            else:
                app.activateIgnoringOtherApps_(True)
        widget.raise_()

    def focus_tracker(self) -> FocusTracker:
        if not self.appkit:
            return super().focus_tracker()
        A = self.appkit
        ws = A.NSWorkspace.sharedWorkspace()
        me = os.getpid()

        def activate(app):
            if app.isTerminated():
                return
            NSApp = A.NSApplication.sharedApplication()
            if NSApp.respondsToSelector_("yieldActivationToApplication:"):        # macOS 14+: cooperative activation
                NSApp.yieldActivationToApplication_(app)
                if not app.activateFromApplication_options_(A.NSRunningApplication.currentApplication(), 0):
                    app.activateWithOptions_(0)
            else:
                app.activateWithOptions_(1 << 1)                                  # macOS 13: IgnoringOtherApps
            # Activation is asynchronous: wait (briefly) until it is really in front, so typing goes there.
            from PySide6.QtCore import QCoreApplication, QEventLoop

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                front = ws.frontmostApplication()
                if front is not None and front.processIdentifier() == app.processIdentifier():
                    return
                QCoreApplication.processEvents(QEventLoop.AllEvents, 20)
                time.sleep(0.02)

        tracker = FocusTracker(ws.frontmostApplication, lambda a: a.processIdentifier() == me, activate)
        # Also follow app switches as they happen, not only on the poll.
        def on_switch(note):
            app = note.userInfo()[A.NSWorkspaceApplicationKey]
            if app is not None and app.processIdentifier() != me:
                tracker.last = app

        try:
            self._observer = ws.notificationCenter().addObserverForName_object_queue_usingBlock_(
                A.NSWorkspaceDidActivateApplicationNotification, None, A.NSOperationQueue.mainQueue(), on_switch)
        except Exception:
            pass
        return tracker

    def register_hotkey(self, combo: str, callback):
        """A Carbon hot key (via quickmachotkey): no permission needed, and the key press is not passed on."""
        mods, key = parse_hotkey(combo)
        try:
            from quickmachotkey import constants as K
            from quickmachotkey import quickHotKey
        except Exception:
            return self._monitor_hotkey(mods, key, callback)
        if key == "space":
            vkey = K.kVK_Space
        else:
            vkey = getattr(K, f"kVK_ANSI_{key.upper()}", None)   # kVK_ANSI_A is 0, so test for None
            if vkey is None:
                vkey = getattr(K, f"kVK_{key.upper()}", None)
        if vkey is None:
            return None
        mask = 0
        for m, bit in (("ctrl", self.KEYS_CTRL), ("shift", self.KEYS_SHIFT), ("alt", self.KEYS_ALT), ("cmd", self.KEYS_CMD)):
            if m in mods:
                mask |= bit

        @quickHotKey(virtualKey=vkey, modifierMask=mask)
        def leanwatch_hotkey() -> None:
            callback()

        return leanwatch_hotkey.unregister

    def _monitor_hotkey(self, mods, key, callback):
        """Fallback without quickmachotkey: an NSEvent monitor (needs Accessibility; the app sees the keys too)."""
        if not self.appkit:
            return None
        A = self.appkit
        want = 0
        for m, bit in (("shift", 1 << 17), ("ctrl", 1 << 18), ("alt", 1 << 19), ("cmd", 1 << 20)):
            if m in mods:
                want |= bit
        mask_all = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20)

        def matches(event) -> bool:
            if (event.modifierFlags() & mask_all) != want:
                return False
            if key == "space":
                return event.keyCode() == 49
            return (event.charactersIgnoringModifiers() or "").lower() == key

        def on_global(event):
            if matches(event):
                callback()

        def on_local(event):
            if matches(event):
                callback()
                return None
            return event

        key_down = 1 << 10
        g = A.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(key_down, on_global)
        local = A.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(key_down, on_local)

        def stop():
            for monitor in (g, local):
                if monitor is not None:
                    A.NSEvent.removeMonitor_(monitor)

        return stop

    def missing_permissions(self) -> list[tuple[str, str]]:
        """(what, System Settings link) for each permission macOS hasn't granted yet."""
        import ctypes

        missing = []
        try:
            ax = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
            ax.AXIsProcessTrusted.restype = ctypes.c_bool
            if not ax.AXIsProcessTrusted():
                missing.append(("Accessibility (to move the mouse and type)", self.SETTINGS + "Privacy_Accessibility"))
        except OSError:
            pass
        try:
            cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
            if not cg.CGPreflightScreenCaptureAccess():
                missing.append(("Screen Recording (to see the screen)", self.SETTINGS + "Privacy_ScreenCapture"))
        except (OSError, AttributeError):
            pass
        return missing

    def notes(self) -> list[str]:
        out = []
        if not self.appkit:
            out.append("Install pyobjc-framework-Cocoa so the bubble floats over full-screen apps and every Space: "
                       "pip install pyobjc-framework-Cocoa")
        for what, _url in self.missing_permissions():
            out.append(f"macOS hasn't allowed {what} yet: System Settings > Privacy & Security. "
                       "Allow the app you started LeanWatch from (Terminal, or Python at login), then restart LeanWatch.")
        return out


# ---------------------------------------------------------------------------
# Linux (X11; under Wayland LeanWatch runs through XWayland)
# ---------------------------------------------------------------------------
def ensure_xauthority() -> None:
    """pyautogui pulls in an old X11 library (python3-Xlib 0.15) that refuses to connect when there is no
    ~/.Xauthority file, even to a display that needs none. Point it at an empty file in that case."""
    if not sys.platform.startswith("linux") or os.environ.get("XAUTHORITY"):
        return
    from pathlib import Path

    if (Path.home() / ".Xauthority").exists():
        return
    import tempfile

    fd, path = tempfile.mkstemp(prefix="leanwatch-xauth-")
    os.close(fd)
    os.environ["XAUTHORITY"] = path


def wayland_session() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY")) or os.environ.get("XDG_SESSION_TYPE") == "wayland"


class X11Native(Native):
    """Linux. A real X11 session works fully; under Wayland LeanWatch runs through XWayland.

    - In an X11 session the round bubble is an override-redirect window (no window manager involved):
      GNOME and KDE both draw those above everything, including full-screen apps (KDE puts ordinary
      keep-above windows *under* a focused full-screen app), on every virtual desktop.
    - The command box stays a managed window (it needs the keyboard), kept above and on all desktops.
    - The shortcut is an X key grab: it works without permissions and the key doesn't reach the app.
    """

    name = "x11"

    def __init__(self):
        try:
            ensure_xauthority()
            from Xlib import X, display, protocol

            self.X, self.protocol = X, protocol
            self.dpy = display.Display()
            self.root = self.dpy.screen().root
            self.atom = lambda name: self.dpy.intern_atom(name)
            self._lock = threading.Lock()   # python-xlib connections aren't thread-safe: this one is GUI-thread only
        except Exception:
            self.dpy = None

    def override_redirect_bubble(self) -> bool:
        return self.dpy is not None and not wayland_session()

    def supports_translucency(self) -> bool:
        if self.dpy is None:
            return False      # can't tell: a shaped (masked) bubble looks right with or without a compositor
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
        if self.dpy is None or widget.windowFlags() & _bypass_flag():
            return          # override-redirect windows aren't managed: nothing to tell the window manager
        with self._lock:
            wid = int(widget.winId())
            self._send(wid, "_NET_WM_STATE", [1, self.atom("_NET_WM_STATE_STICKY"), self.atom("_NET_WM_STATE_ABOVE"), 2])
            self._send(wid, "_NET_WM_DESKTOP", [0xFFFFFFFF, 2])

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
                # source 2 = a pager or tool acting for the user, timestamp 0 = now: GNOME and KDE both honour it.
                self._send(wid, "_NET_ACTIVE_WINDOW", [2, self.X.CurrentTime, 0])
            deadline = time.monotonic() + 0.5          # wait until it is really active, before the agent types
            while time.monotonic() < deadline and current() != wid:
                time.sleep(0.02)

        return FocusTracker(current, is_ours, activate)

    def register_hotkey(self, combo: str, callback):
        """Grab the key combination on the root window, in its own thread and X connection."""
        if self.dpy is None:
            return None
        import select

        from Xlib import X, XK, display, error

        mods, key = parse_hotkey(combo)
        bits = {"ctrl": X.ControlMask, "alt": X.Mod1Mask, "shift": X.ShiftMask, "cmd": X.Mod4Mask}
        mask = 0
        for m in mods:
            mask |= bits[m]
        name = key.upper() if (key.startswith("f") and key[1:].isdigit()) else key
        keysym = XK.string_to_keysym(name)
        state = {"ok": False}
        ready, stop = threading.Event(), threading.Event()
        variants = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)   # also with Caps Lock / Num Lock on

        def run():
            d = display.Display()
            root = d.screen().root
            code = d.keysym_to_keycode(keysym) if keysym else 0
            if not code:
                ready.set()
                d.close()
                return
            catch = error.CatchError(error.BadAccess)
            for extra in variants:
                root.grab_key(code, mask | extra, True, X.GrabModeAsync, X.GrabModeAsync, onerror=catch)
            d.sync()
            if catch.get_error():         # another program already uses this combination
                ready.set()
                d.close()
                return
            state["ok"] = True
            ready.set()
            while not stop.is_set():
                select.select([d.fileno()], [], [], 0.25)
                while d.pending_events():
                    ev = d.next_event()
                    if ev.type == X.KeyPress and ev.detail == code:
                        callback()
            for extra in variants:
                root.ungrab_key(code, mask | extra)
            d.close()

        threading.Thread(target=run, daemon=True, name="leanwatch-hotkey").start()
        ready.wait(3)
        return stop.set if state["ok"] else None

    def notes(self) -> list[str]:
        if wayland_session():
            return ["Wayland session: LeanWatch sees and controls the screen through your desktop's screen sharing. "
                    "The first task asks for permission: turn on “Allow Remote Interaction”, then click Share."]
        return []


def _bypass_flag():
    from PySide6.QtCore import Qt

    return Qt.X11BypassWindowManagerHint


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
