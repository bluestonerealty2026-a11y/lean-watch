"""The macOS and Windows code paths, run against stand-in system libraries.

These can't prove the real OS behaves as documented (that needs a Mac and a PC), but they
catch wrong method names, wrong arguments and wrong logic in code this machine can't run.
"""

import sys
import types

import pytest

from leanwatch import native


class Widget:
    def __init__(self, wid=1234, visible=True):
        self.wid = wid
        self.visible = visible
        self.raised = 0

    def winId(self):
        return self.wid

    def isVisible(self):
        return self.visible

    def raise_(self):
        self.raised += 1


# --------------------------------------------------------------------------- macOS
class FakeNSWindow:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def record(*args):
            self.calls[name] = args
        return record


class FakeApp:
    def __init__(self, pid):
        self.pid = pid
        self.activated = []

    def processIdentifier(self):
        return self.pid

    def isTerminated(self):
        return False

    def activateFromApplication_options_(self, me, opts):
        self.activated.append(("from", opts))
        Workspace.front = self
        return True

    def activateWithOptions_(self, opts):
        self.activated.append(("with", opts))
        Workspace.front = self
        return True


class Workspace:
    front = None
    observer = None

    @classmethod
    def sharedWorkspace(cls):
        return cls

    @classmethod
    def frontmostApplication(cls):
        return cls.front

    @classmethod
    def notificationCenter(cls):
        return cls

    @classmethod
    def addObserverForName_object_queue_usingBlock_(cls, name, obj, queue, block):
        cls.observer = block
        return "token"


def fake_appkit(modern=True):
    window = FakeNSWindow()
    nsapp = types.SimpleNamespace(policy=None, yielded=[])
    nsapp.setActivationPolicy_ = lambda p: setattr(nsapp, "policy", p)
    nsapp.respondsToSelector_ = lambda sel: modern
    nsapp.yieldActivationToApplication_ = lambda app: nsapp.yielded.append(app)
    nsapp.activate = lambda: None
    appkit = types.SimpleNamespace(
        NSApplication=types.SimpleNamespace(sharedApplication=lambda: nsapp),
        NSWorkspace=Workspace,
        NSRunningApplication=types.SimpleNamespace(currentApplication=lambda: FakeApp(0)),
        NSOperationQueue=types.SimpleNamespace(mainQueue=lambda: "main"),
        NSWorkspaceDidActivateApplicationNotification="didActivate",
        NSWorkspaceApplicationKey="app",
    )
    objc = types.SimpleNamespace(objc_object=lambda c_void_p: types.SimpleNamespace(window=lambda: window))
    return appkit, objc, window, nsapp


def mac(monkeypatch, modern=True):
    appkit, objc, window, nsapp = fake_appkit(modern)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "objc", objc)
    m = native.MacNative()
    assert m.appkit is appkit
    return m, window, nsapp


def test_mac_window_floats_over_everything(monkeypatch):
    m, window, nsapp = mac(monkeypatch)
    m.setup_app()
    assert nsapp.policy == 1                                   # accessory: no Dock icon
    m.prepare_window(Widget())
    behaviour = window.calls["setCollectionBehavior_"][0]
    assert behaviour & 1 and behaviour & (1 << 8) and behaviour & (1 << 18) and not behaviour & (1 << 1)
    assert window.calls["setLevel_"] == (25,)
    assert window.calls["setHidesOnDeactivate_"] == (False,) and window.calls["setCanHide_"] == (False,)
    assert "orderFrontRegardless" in window.calls


def test_mac_keep_on_top_never_activates(monkeypatch):
    m, window, _ = mac(monkeypatch)
    w = Widget()
    m.keep_on_top(w)
    assert "orderFrontRegardless" in window.calls and w.raised == 0
    assert m.exclude_from_capture(w) is False


@pytest.mark.parametrize("modern", [True, False])
def test_mac_gives_the_keyboard_back(monkeypatch, qapp_if_available, modern):
    m, _, nsapp = mac(monkeypatch, modern)
    editor = FakeApp(4242)
    Workspace.front = editor
    tracker = m.focus_tracker()
    tracker.poll()
    Workspace.front = FakeApp(native.os.getpid())              # LeanWatch is in front now
    tracker.poll()
    assert tracker.last is editor
    tracker.restore()
    assert Workspace.front is editor
    assert editor.activated == ([("from", 0)] if modern else [("with", 2)])
    assert nsapp.yielded == ([editor] if modern else [])
    other = FakeApp(5151)
    Workspace.observer(types.SimpleNamespace(userInfo=lambda: {"app": other}))   # an app switch notification
    assert tracker.last is other


def test_mac_hotkey_uses_carbon(monkeypatch):
    m, *_ = mac(monkeypatch)
    seen = {}
    constants = types.SimpleNamespace(kVK_Space=0x31, kVK_ANSI_L=0x25)

    def quickHotKey(virtualKey, modifierMask):
        seen.update(key=virtualKey, mask=modifierMask)

        def wrap(fn):
            fn.unregister = lambda: seen.setdefault("unregistered", True)
            seen["fn"] = fn
            return fn
        return wrap

    monkeypatch.setitem(sys.modules, "quickmachotkey", types.SimpleNamespace(quickHotKey=quickHotKey, constants=constants))
    monkeypatch.setitem(sys.modules, "quickmachotkey.constants", constants)
    fired = []
    stop = m.register_hotkey("ctrl+shift+space", lambda: fired.append(1))
    assert seen["key"] == 0x31 and seen["mask"] == 0x1000 | 0x200
    seen["fn"]()
    assert fired == [1]
    stop()
    assert seen["unregistered"]
    m.register_hotkey("cmd+alt+l", lambda: None)
    assert seen["key"] == 0x25 and seen["mask"] == 0x100 | 0x800


def test_permissions_check_is_quiet_off_macos(monkeypatch):
    m, *_ = mac(monkeypatch)
    assert m.missing_permissions() == []                      # the frameworks don't exist here: nothing to report


@pytest.fixture
def qapp_if_available():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- Windows
class Fn:
    def __init__(self, name, impl):
        self.name, self.impl, self.argtypes, self.restype = name, impl, None, None

    def __call__(self, *args):
        return self.impl(*args)


class FakeDLL:
    def __init__(self, impls):
        self._impls = impls
        self.calls = []

    def __getattr__(self, name):
        impl = self._impls.get(name, lambda *a: 0)

        def logged(*args):
            self.calls.append((name, args))
            return impl(*args)
        fn = Fn(name, logged)
        setattr(self, name, fn)
        return fn


def windows(monkeypatch, build=22631, user32=None):
    import ctypes

    user = FakeDLL(user32 or {})
    kernel = FakeDLL({"GetCurrentThreadId": lambda: 7})
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: user if name == "user32" else kernel, raising=False)
    monkeypatch.setattr(sys, "getwindowsversion", lambda: types.SimpleNamespace(build=build), raising=False)
    return native.WindowsNative(), user


def test_windows_capture_exclusion_needs_build_19041_and_is_read_back(monkeypatch):
    def get_affinity(hwnd, out):
        out._obj.value = 0x11
        return 1

    w, user = windows(monkeypatch, user32={"SetWindowDisplayAffinity": lambda h, a: 1, "GetWindowDisplayAffinity": get_affinity})
    assert w.exclude_from_capture(Widget()) is True
    assert ("SetWindowDisplayAffinity", (1234, 0x11)) in user.calls
    old, user_old = windows(monkeypatch, build=18363)
    assert old.exclude_from_capture(Widget()) is False      # older Windows would paint a black disc instead
    assert not any(c[0] == "SetWindowDisplayAffinity" for c in user_old.calls)


def test_windows_keep_on_top_pins_without_activating(monkeypatch):
    w, user = windows(monkeypatch)
    w.keep_on_top(Widget())
    name, args = user.calls[-1]
    assert name == "SetWindowPos" and args[0] == 1234 and args[1] == -1
    assert args[-1] & 0x0010                                 # SWP_NOACTIVATE


def test_windows_focus_tracker_skips_the_taskbar(monkeypatch):
    state = {"fg": 111}
    classes = {111: "Notepad", 222: "Shell_TrayWnd"}

    def class_name(h, buf, n):
        buf.value = classes[h]
        return len(classes[h])

    w, user = windows(monkeypatch, user32={
        "GetForegroundWindow": lambda: state["fg"], "GetAncestor": lambda h, flag: h,
        "GetClassNameW": class_name, "GetWindowThreadProcessId": lambda h, pid: 0,
        "IsWindow": lambda h: 1, "IsIconic": lambda h: 0, "SetForegroundWindow": lambda h: 1,
        "AttachThreadInput": lambda a, b, c: 1, "BringWindowToTop": lambda h: 1,
    })
    t = w.focus_tracker()
    t.poll()
    state["fg"] = 222                                          # the person clicked the taskbar
    t.poll()
    assert t.last == 111
    t.restore()
    assert ("SetForegroundWindow", (111,)) in user.calls


def test_dpi_awareness_is_windows_only():
    native.enable_dpi_awareness("linux")                      # no-op, no error


def test_mac_hotkey_with_the_letter_a(monkeypatch):
    m, *_ = mac(monkeypatch)
    seen = {}
    constants = types.SimpleNamespace(kVK_Space=0x31, kVK_ANSI_A=0)          # A really is key code 0

    def quickHotKey(virtualKey, modifierMask):
        seen["key"] = virtualKey

        def wrap(fn):
            fn.unregister = lambda: None
            return fn
        return wrap

    monkeypatch.setitem(sys.modules, "quickmachotkey", types.SimpleNamespace(quickHotKey=quickHotKey, constants=constants))
    monkeypatch.setitem(sys.modules, "quickmachotkey.constants", constants)
    assert m.register_hotkey("cmd+shift+a", lambda: None) is not None and seen["key"] == 0
