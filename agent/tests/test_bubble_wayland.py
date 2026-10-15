"""The bubble on Wayland (Qt's offscreen platform): the "open the box" message, giving the keyboard back
by stepping aside, the screen size learnt from the portal, the consent wording, and desktop.run's choice
of whether the keyboard shortcut works (shortcut_ok)."""

import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from leanwatch import bubble as bubble_mod  # noqa: E402
from leanwatch import gnome_shortcut, wayland  # noqa: E402
from leanwatch.bubble import BubbleApp  # noqa: E402
from leanwatch.native import FocusTracker, Native  # noqa: E402
from leanwatch.permissions import Permissions  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/nonexistent/leanwatch-unit-tests")

    def no_gsettings(*args):
        raise AssertionError("a unit test tried to run the real gsettings")

    monkeypatch.setattr(gnome_shortcut, "_gsettings", no_gsettings)
    return home


class FakeSession:
    def __init__(self, granted=False):
        self.permissions = Permissions()
        if granted:
            self.permissions.grant(mouse=True, keyboard=True)
        self.listeners = []
        self.shield = None
        self.screen_size = None
        self.sent = []

    def handle(self, msg):
        self.sent.append(msg)
        if msg["type"] == "grant":
            self.permissions.grant(mouse=True, keyboard=True)
        if msg["type"] == "command":
            self.emit({"type": "run_started", "text": msg["text"]})
        return None

    def emit(self, msg):
        for listener in self.listeners:
            listener(msg)


class FakeNative(Native):
    def __init__(self):
        self.restored = 0
        self.hotkeys = []
        self.hotkey_error = None
        self.unregistered = 0

    def focus_tracker(self):
        native = self

        def activate(_h):
            native.restored += 1

        return FocusTracker(lambda: "editor", lambda h: False, activate)

    def register_hotkey(self, combo, callback):
        if self.hotkey_error:
            raise self.hotkey_error
        self.hotkeys.append(combo)

        def stop():
            self.unregistered += 1

        return stop


def make(qapp, monkeypatch, *, wayland_on, granted=False, screen_px_size=None):
    monkeypatch.setattr(wayland, "wanted", lambda: wayland_on)
    session, native = FakeSession(granted), FakeNative()
    app = BubbleApp(session, native, screen_px_size=screen_px_size)
    app.shortcut_ok = True
    app.show()
    qapp.processEvents()
    return app, session, native


def pump(qapp, seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def run_in_worker(qapp, fn):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("value", fn()))
    t.start()
    while t.is_alive():
        qapp.processEvents()
        time.sleep(0.002)
    pump(qapp)
    return out.get("value")


def close(app):
    for w in app.windows():
        w.removeEventFilter(app)
        w.hide()
        w.deleteLater()
    app.session.listeners.clear()
    app._keep_on_top.stop()
    app._focus_poll.stop()


@pytest.fixture
def fast_clock(monkeypatch):
    """bubble.py's time module without real waiting (the step-aside pause is 0.25 s)."""
    clock = SimpleNamespace(t=0.0)

    def monotonic():
        clock.t += 0.05
        return clock.t

    monkeypatch.setattr(bubble_mod, "time", SimpleNamespace(monotonic=monotonic, sleep=lambda s: None))
    return clock


def watch_panel(app):
    """Record the panel's hide() and show() calls, and whether a show would take the keyboard."""
    seen = []
    panel = app.panel
    hide, show = panel.hide, panel.show
    panel.hide = lambda: (seen.append("hide"), hide())[1]
    panel.show = lambda: (seen.append(("show", panel.testAttribute(Qt.WA_ShowWithoutActivating))), show())[1]
    return seen


# --------------------------------------------------------------------------- knowing it is on Wayland
def test_on_wayland_follows_wayland_wanted(qapp, monkeypatch):
    app, _, _ = make(qapp, monkeypatch, wayland_on=True)
    assert app.on_wayland is True
    close(app)
    app, _, _ = make(qapp, monkeypatch, wayland_on=False)
    assert app.on_wayland is False
    close(app)


def test_a_broken_wayland_check_means_x11(qapp, monkeypatch):
    def broken():
        raise RuntimeError("no")

    monkeypatch.setattr(wayland, "wanted", broken)
    assert bubble_mod._on_wayland() is False


# --------------------------------------------------------------------------- the "open the box" message
def test_open_box_message_opens_and_closes_the_box(qapp, monkeypatch):
    app, session, _ = make(qapp, monkeypatch, wayland_on=True)
    assert not app.panel.isVisible()
    # Sent by the server's thread when GNOME's shortcut runs `python -m leanwatch --open`.
    t = threading.Thread(target=lambda: session.emit({"type": "open_box"}))
    t.start()
    t.join()
    pump(qapp)
    assert app.expanded and app.panel.isVisible() and not app.bubble.isVisible()
    assert QApplication.focusWidget() is app.panel.input
    app._on_event({"type": "open_box"})          # pressed again while idle: closes it
    assert not app.expanded and not app.panel.isVisible() and app.bubble.isVisible()
    close(app)


def test_open_box_message_brings_back_a_hidden_bubble(qapp, monkeypatch):
    app, _, _ = make(qapp, monkeypatch, wayland_on=True)
    app.set_hidden(True)
    assert not app.bubble.isVisible()
    app._on_event({"type": "open_box"})
    assert app.panel.isVisible() and not app.hidden_by_user
    close(app)


def test_open_box_message_during_a_task_keeps_the_box_open(qapp, monkeypatch):
    app, _, _ = make(qapp, monkeypatch, wayland_on=True, granted=True)
    app.expand()
    app.submit("open the settings")
    assert app.running
    app._on_event({"type": "open_box"})
    assert app.expanded and app.panel.isVisible()
    close(app)


# --------------------------------------------------------------------------- keyboard hand-back on Wayland
def test_before_keys_on_wayland_steps_aside_without_taking_the_keyboard_back(qapp, monkeypatch):
    app, session, native = make(qapp, monkeypatch, wayland_on=True, granted=True)
    app.expand()
    app.submit("open settings")
    before = native.restored
    app.panel.isActiveWindow = lambda: True      # the person clicked into the box during the task
    seen = watch_panel(app)
    started = time.monotonic()
    run_in_worker(qapp, app.shield.before_keys)  # the agent is about to type (from its worker thread)
    assert seen == ["hide", ("show", True)]      # out of the way, then back without activating
    assert time.monotonic() - started >= 0.2     # the desktop got time to move the keyboard
    assert not app.panel.testAttribute(Qt.WA_ShowWithoutActivating)   # normal again for the next open
    assert app.panel.isVisible() and app.expanded
    assert native.restored == before             # no X11-style hand-off (it can't work on Wayland)
    close(app)


def test_before_keys_on_wayland_does_nothing_when_the_box_has_no_keyboard(qapp, monkeypatch):
    app, _, native = make(qapp, monkeypatch, wayland_on=True, granted=True)
    app.expand()
    app.submit("x")
    app.panel.isActiveWindow = lambda: False
    seen = watch_panel(app)
    before = native.restored
    run_in_worker(qapp, app.shield.before_keys)
    assert seen == [] and native.restored == before and app.panel.isVisible()
    close(app)


def test_before_keys_on_x11_hands_the_keyboard_over(qapp, monkeypatch):
    app, _, native = make(qapp, monkeypatch, wayland_on=False, granted=True)
    app.expand()
    app.submit("x")
    app.panel.isActiveWindow = lambda: True
    seen = watch_panel(app)
    before = native.restored
    run_in_worker(qapp, app.shield.before_keys)
    assert seen == [] and native.restored == before + 1
    close(app)


def test_allow_on_wayland_steps_aside_before_the_agent_continues(qapp, monkeypatch, fast_clock):
    app, session, _ = make(qapp, monkeypatch, wayland_on=True, granted=True)
    app.expand()
    app.submit("send the email")
    app._on_event({"type": "confirm_request", "id": "q1", "action": "Send the email?", "reason": "send"})
    app.panel.isActiveWindow = lambda: True
    order = []
    hide = app.panel.hide
    app.panel.hide = lambda: (order.append("hide"), hide())[1]
    handle = session.handle
    session.handle = lambda m: (order.append(m["type"]), handle(m))[1]
    app._answer(True)
    assert order == ["hide", "confirm"]
    assert app.panel.isVisible()
    close(app)


def test_a_box_closed_meanwhile_is_not_shown_again(qapp, monkeypatch, fast_clock):
    app, _, _ = make(qapp, monkeypatch, wayland_on=True, granted=True)
    app.expand()
    app.panel.isActiveWindow = lambda: True
    app.expanded = False                          # e.g. collapsed while the desktop moved the keyboard
    seen = watch_panel(app)
    app._release_keyboard()
    assert seen == ["hide"] and not app.panel.isVisible()
    close(app)


# --------------------------------------------------------------------------- screen size from the session
def test_screen_px_size_is_read_from_the_session_each_time(qapp, monkeypatch):
    app, session, _ = make(qapp, monkeypatch, wayland_on=True)
    g = QApplication.primaryScreen().geometry()
    r = QRect(10, 20, 30, 40)
    assert app.screen_px_size is None
    assert app._to_screen_px(r) == (10, 20, 40, 60)
    session.screen_size = (g.width() * 2, g.height() * 2)      # the desktop shared the screen (its pixels)
    assert app.screen_px_size == (g.width() * 2, g.height() * 2)
    assert app._to_screen_px(r) == (20, 40, 80, 120)
    app.screen_px_size = (g.width(), g.height())               # set explicitly: wins
    assert app._to_screen_px(r) == (10, 20, 40, 60)
    close(app)


def test_screen_px_size_given_at_start_wins(qapp, monkeypatch):
    g = QApplication.primaryScreen().geometry()
    app, session, _ = make(qapp, monkeypatch, wayland_on=True, screen_px_size=(g.width() * 3, g.height() * 3))
    session.screen_size = (g.width(), g.height())
    assert app.screen_px_size == (g.width() * 3, g.height() * 3)
    assert app._to_screen_px(QRect(10, 20, 30, 40)) == (30, 60, 120, 180)
    close(app)


# --------------------------------------------------------------------------- consent wording
def consent_text(qapp, monkeypatch, wayland_on):
    app, _, _ = make(qapp, monkeypatch, wayland_on=wayland_on)
    app.expand()
    app.submit("turn on dark mode")
    text = app.panel.consent.detail.text()
    visible = app.panel.consent.isVisible()
    app._consent(False)
    close(app)
    return visible, text


def test_consent_on_wayland_mentions_the_desktops_dialog(qapp, monkeypatch):
    visible, text = consent_text(qapp, monkeypatch, True)
    assert visible
    assert "stop sharing" in text and "Allow Remote Interaction" in text
    assert "screen corner" not in text and "moving the mouse yourself" not in text


def test_consent_on_x11_mentions_the_screen_corner(qapp, monkeypatch):
    visible, text = consent_text(qapp, monkeypatch, False)
    assert visible
    assert "screen corner" in text and "moving the mouse yourself" in text
    assert "Allow Remote Interaction" not in text


# --------------------------------------------------------------------------- desktop.run: shortcut_ok
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def run_desktop(qapp, monkeypatch, capsys):
    from leanwatch import desktop, native as native_mod
    from leanwatch.config import Config

    made = []

    class RecordingBubbleApp(BubbleApp):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    def run(*, wayland_on, gnome_installed, hotkey_error=None, hotkey="ctrl+alt+space"):
        nat = FakeNative()
        nat.hotkey_error = hotkey_error
        gnome_calls = []
        monkeypatch.setattr(native_mod, "for_platform", lambda platform=None: nat)
        monkeypatch.setattr(bubble_mod, "BubbleApp", RecordingBubbleApp)
        monkeypatch.setattr(wayland, "wanted", lambda: wayland_on)
        monkeypatch.setattr(desktop, "gnome_shortcut_installed", lambda: gnome_calls.append(1) or gnome_installed)
        monkeypatch.setattr(desktop, "signal", SimpleNamespace(signal=lambda *a: None, SIGINT=2))
        monkeypatch.setattr(QApplication, "exec", lambda *a, **k: 0)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        cfg = Config()
        cfg.port = free_port()
        cfg.dry_run = True
        assert desktop.run(cfg, hotkey=hotkey) == 0
        app = made[-1]
        out = capsys.readouterr().out
        result = SimpleNamespace(shortcut_ok=app.shortcut_ok, out=out, native=nat, gnome_calls=gnome_calls)
        close(app)
        return result

    yield run
    for app in made:
        try:
            close(app)
        except RuntimeError:
            pass


def test_wayland_with_the_gnome_shortcut(run_desktop):
    r = run_desktop(wayland_on=True, gnome_installed=True)
    assert r.shortcut_ok is True
    assert "or press Ctrl+Alt+Space" in r.out and "--shortcut on" not in r.out
    assert r.native.unregistered == 1                     # the X grab is still cleaned up on quit


def test_wayland_without_the_gnome_shortcut(run_desktop):
    r = run_desktop(wayland_on=True, gnome_installed=False)
    assert r.shortcut_ok is False                         # the X grab alone doesn't count on Wayland
    assert "or press" not in r.out
    assert "python -m leanwatch --shortcut on" in r.out
    assert "could not be set up" not in r.out


def test_wayland_with_a_failed_x_grab_says_nothing_about_it(run_desktop):
    r = run_desktop(wayland_on=True, gnome_installed=True, hotkey_error=RuntimeError("grab failed"))
    assert r.shortcut_ok is True and "could not be set up" not in r.out


def test_x11_with_a_working_grab(run_desktop):
    r = run_desktop(wayland_on=False, gnome_installed=False)
    assert r.shortcut_ok is True and "or press Ctrl+Alt+Space" in r.out
    assert r.gnome_calls == []                            # GNOME's settings aren't asked on X11
    assert "--shortcut on" not in r.out


def test_x11_with_a_failed_grab(run_desktop):
    r = run_desktop(wayland_on=False, gnome_installed=True, hotkey_error=RuntimeError("taken"))
    assert r.shortcut_ok is False
    assert "could not be set up" in r.out and "or press" not in r.out


def test_no_hotkey_at_all(run_desktop):
    r = run_desktop(wayland_on=True, gnome_installed=True, hotkey=None)
    assert r.shortcut_ok is True and "or press" not in r.out and r.native.hotkeys == []
