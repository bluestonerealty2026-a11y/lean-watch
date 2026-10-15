"""The floating bubble, without a real screen (Qt's offscreen platform)."""

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from leanwatch.bubble import BubbleApp, Shield, clamp_position, default_position, panel_geometry  # noqa: E402
from leanwatch.native import FocusTracker, Native, parse_hotkey  # noqa: E402
from leanwatch.permissions import Permissions  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


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
    def __init__(self, excluded=False):
        self.excluded = excluded
        self.restored = 0
        self.prepared = []

    def prepare_window(self, widget):
        self.prepared.append(widget.objectName())

    def exclude_from_capture(self, widget):
        return self.excluded

    def focus_tracker(self):
        native = self

        def activate(_h):
            native.restored += 1

        tracker = FocusTracker(lambda: "editor", lambda h: False, activate)
        return tracker


def make(qapp, tmp_path=None, granted=False, excluded=False, shortcut=True):
    session = FakeSession(granted)
    native = FakeNative(excluded)
    app = BubbleApp(session, native, settings_path=(tmp_path / "bubble.json") if tmp_path else None)
    app.shortcut_ok = shortcut          # desktop.run sets this once the keyboard shortcut is registered
    app.show()
    qapp.processEvents()
    return app, session, native


def pump(qapp, seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def mouse(widget, kind, local, button=Qt.LeftButton):
    local = QPointF(local)
    glob = QPointF(widget.mapToGlobal(local.toPoint()))
    buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    ev = QMouseEvent(kind, local, glob, button if kind != QEvent.MouseMove else Qt.NoButton, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


def close(app):
    for w in app.windows():
        w.removeEventFilter(app)
        w.hide()
        w.deleteLater()
    app.session.listeners.clear()


# --------------------------------------------------------------------------- geometry
def test_default_position_is_bottom_right_inside_the_usable_area():
    area = QRect(0, 0, 1920, 1040)                       # taskbar takes the bottom 40 px
    pos = default_position(area, QSize(76, 76))
    assert QRect(pos, QSize(76, 76)).bottomRight() == QPoint(1920 - 16 - 1, 1040 - 16 - 1)


def test_clamp_keeps_the_bubble_on_screen():
    area = QRect(0, 0, 800, 600)
    assert clamp_position(QPoint(-50, 900), QSize(76, 76), area) == QPoint(0, 600 - 76)


@pytest.mark.parametrize("where,expect_bottom", [((1800, 980), True), ((40, 980), True), ((1800, 40), False), ((40, 40), False)])
def test_box_opens_toward_the_middle_of_the_screen(where, expect_bottom):
    area = QRect(0, 0, 1920, 1080)
    bubble = QRect(QPoint(*where), QSize(76, 76))
    rect, bottom = panel_geometry(bubble, QSize(400, 260), area)
    assert bottom is expect_bottom
    assert area.contains(rect)
    # It covers the bubble's spot, so the text field appears where the person clicked.
    assert rect.intersects(bubble)
    if where[0] > 960:
        assert rect.right() == bubble.right()
    else:
        assert rect.left() == bubble.left()


# --------------------------------------------------------------------------- clicking and dragging
def test_click_opens_the_box_and_focuses_the_text_field(qapp):
    app, session, native = make(qapp)
    assert app.bubble.isVisible() and not app.panel.isVisible()
    assert set(native.prepared) == {"leanwatch-bubble", "leanwatch-panel"}
    QTest.mouseClick(app.bubble, Qt.LeftButton, pos=QPoint(38, 38))
    qapp.processEvents()
    assert app.expanded and app.panel.isVisible() and not app.bubble.isVisible()
    assert app.panel.input.placeholderText() == "e.g. turn on dark mode"
    assert app.panel.go.text() == "Go"
    close(app)


def test_dragging_moves_the_bubble_without_opening_it_and_remembers_the_spot(qapp, tmp_path):
    app, _, _ = make(qapp, tmp_path)
    start = app.bubble.pos()
    mouse(app.bubble, QEvent.MouseButtonPress, QPoint(38, 38))
    mouse(app.bubble, QEvent.MouseMove, QPoint(38 - 120, 38 - 90))
    mouse(app.bubble, QEvent.MouseButtonRelease, QPoint(38 - 120, 38 - 90))
    qapp.processEvents()
    assert not app.expanded
    assert app.bubble.pos() != start
    saved = (tmp_path / "bubble.json").read_text()
    assert f'"x": {app.bubble.pos().x()}' in saved
    close(app)

    again, _, _ = make(qapp, tmp_path)           # next start: same place
    assert again.bubble.pos() == app.bubble.pos()
    close(again)


def test_saved_position_on_a_missing_monitor_falls_back_to_the_corner(qapp, tmp_path):
    (tmp_path / "bubble.json").write_text('{"x": 99999, "y": 99999}')
    app, _, _ = make(qapp, tmp_path)
    area = QApplication.primaryScreen().availableGeometry()
    assert area.contains(app.bubble.frameGeometry())
    close(app)


def test_bubble_cannot_be_hidden_when_nothing_could_bring_it_back(qapp, tmp_path):
    app, _, _ = make(qapp, tmp_path, shortcut=False)      # and the offscreen platform has no tray
    app.set_hidden(True)
    assert app.bubble.isVisible() and not app.hidden_by_user
    close(app)
    (tmp_path / "bubble.json").write_text('{"x": 10, "y": 10, "hidden": true}')
    again, _, _ = make(qapp, tmp_path, shortcut=False)  # a saved "hidden" is ignored too
    assert again.bubble.isVisible()
    close(again)


def test_hiding_the_bubble_is_remembered(qapp, tmp_path):
    app, _, _ = make(qapp, tmp_path)
    app.set_hidden(True)
    assert not app.bubble.isVisible()
    close(app)
    again, _, _ = make(qapp, tmp_path)
    assert not again.bubble.isVisible()
    again.open_from_hotkey()                      # the shortcut brings it back
    assert again.panel.isVisible() and not again.hidden_by_user
    close(again)


# --------------------------------------------------------------------------- commands
def test_first_command_asks_permission_then_runs(qapp):
    app, session, native = make(qapp)
    app.expand()
    app.panel.input.setText("turn on dark mode")
    QTest.keyClick(app.panel.input, Qt.Key_Return)
    qapp.processEvents()
    assert app.panel.consent.isVisible()
    assert not any(m["type"] == "command" for m in session.sent)
    app.collapse()                                # can't hide the question by accident
    assert app.panel.isVisible()
    QTest.mouseClick(app.panel.consent.yes, Qt.LeftButton)
    qapp.processEvents()
    kinds = [m["type"] for m in session.sent]
    assert kinds == ["grant", "command"]
    assert session.sent[1]["text"] == "turn on dark mode"
    assert app.running and app.panel.go.text() == "Stop" and not app.panel.input.isEnabled()
    assert native.restored == 1                   # the keyboard went back to the person's app
    close(app)


def test_saying_no_to_permission_runs_nothing(qapp):
    app, session, _ = make(qapp)
    app.expand()
    app.submit("delete my photos")
    QTest.mouseClick(app.panel.consent.no, Qt.LeftButton)
    assert [m["type"] for m in session.sent] == []
    assert "Nothing will move" in app.panel.notice.text()
    close(app)


def test_error_from_the_agent_is_shown_in_the_box(qapp):
    app, session, _ = make(qapp, granted=True)
    session.handle = lambda msg: {"type": "error", "message": "I'm already working on something. Press Stop first."}
    app.expand()
    app.submit("open paint")
    assert "already working" in app.panel.notice.text()
    close(app)


def test_steps_are_shown_as_plain_text_numbered(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    app.submit("rename the file")
    app._on_event({"type": "narration", "text": "I click <b>File</b> & then Rename.", "kind": "say"})
    app._on_event({"type": "narration", "text": "(1.2 s to decide)", "kind": "thinking"})
    html = app.panel.steps.text()
    assert "&lt;b&gt;File&lt;/b&gt; &amp; then" in html and "1." in html
    assert app.panel.activity.text() == "(1.2 s to decide)"
    for i in range(6):
        app._on_event({"type": "narration", "text": f"step {i + 2}", "kind": "say"})
    assert "7." in app.panel.steps.text() and "1." not in app.panel.steps.text().split("</span>")[0]
    close(app)


def test_risky_step_question_opens_the_box_without_stealing_focus(qapp):
    app, session, native = make(qapp, granted=True)
    app.expand()
    app.submit("empty the recycle bin")
    app.collapse()
    assert not app.panel.isVisible()
    app._on_event({"type": "confirm_request", "id": "abc", "action": "Empty the Recycle Bin", "reason": "This can't be undone."})
    qapp.processEvents()
    assert app.panel.isVisible() and app.panel.confirm.isVisible() and app.bubble.attention
    assert QApplication.focusWidget() is not app.panel.input
    QTest.mouseClick(app.panel.confirm.no, Qt.LeftButton)
    assert session.sent[-1] == {"type": "confirm", "id": "abc", "allow": False}
    assert not app.panel.confirm.isVisible() and not app.bubble.attention
    close(app)


def test_escape_stops_a_running_task_and_closes_an_idle_box(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    app.submit("open the calculator")
    QTest.keyClick(app.panel, Qt.Key_Escape)
    assert session.sent[-1] == {"type": "stop"}
    app._on_event({"type": "run_finished", "ok": False, "message": "Stopped. You are in control.", "result": "Stopped. You are in control."})
    assert not app.running and app.panel.go.text() == "Go" and app.panel.input.isEnabled()
    assert app.panel.notice.text().startswith("✕ Stopped")
    QTest.keyClick(app.panel, Qt.Key_Escape)
    assert not app.panel.isVisible() and app.bubble.isVisible()
    close(app)


def test_pause_and_resume(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    app.submit("open settings")
    app._on_event({"type": "paused", "reason": "You moved the mouse, so I paused."})
    assert app.panel.paused.isVisible() and app.panel.state.text() == "Paused"
    QTest.mouseClick(app.panel.paused.yes, Qt.LeftButton)
    assert session.sent[-1] == {"type": "resume"}
    close(app)


def test_finished_task_shows_the_summary_and_is_ready_again(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    app.submit("turn on dark mode")
    app.collapse()
    app._on_event({"type": "run_finished", "ok": True, "message": "", "result": "Dark mode is on."})
    assert app.panel.isVisible() and app.panel.notice.text() == "✓ Dark mode is on."
    assert not app.bubble.busy and app.panel.go.text() == "Go"
    close(app)


def test_events_from_another_thread_reach_the_box(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    t = threading.Thread(target=lambda: session.emit({"type": "narration", "text": "From the worker.", "kind": "say"}))
    t.start()
    t.join()
    pump(qapp)
    assert "From the worker." in app.panel.steps.text()
    close(app)


# --------------------------------------------------------------------------- shield
def run_in_worker(qapp, fn):
    out = {}

    def work():
        out["value"] = fn()

    t = threading.Thread(target=work)
    t.start()
    while t.is_alive():
        qapp.processEvents()
        time.sleep(0.002)
    pump(qapp)
    return out.get("value")


def test_screenshots_are_taken_without_our_windows(qapp):
    app, session, _ = make(qapp)
    app.shield.settle_s = 0.01
    assert session.shield is app.shield

    def capture():
        with app.shield.cloak():
            return app.bubble.isVisible()

    assert run_in_worker(qapp, capture) is False
    assert app.bubble.isVisible()                 # back afterwards
    close(app)


def test_no_hiding_when_the_os_already_keeps_us_out_of_screenshots(qapp):
    app, _, _ = make(qapp, excluded=True)
    assert app.shield.excluded_from_capture

    def capture():
        with app.shield.cloak():
            return app.bubble.isVisible()

    assert run_in_worker(qapp, capture) is True
    close(app)


def test_clicks_on_a_spot_under_the_bubble_hide_it_first(qapp):
    app, _, _ = make(qapp, excluded=True)         # even then: a click must not land on the bubble
    app.shield.settle_s = 0.01
    app.shield.refresh()
    centre = app.bubble.frameGeometry().center()
    elsewhere = (5, 5)

    def click_at(point):
        with app.shield.clear_of([point]):
            return app.bubble.isVisible()

    assert run_in_worker(qapp, lambda: click_at((centre.x(), centre.y()))) is False
    assert run_in_worker(qapp, lambda: click_at(elsewhere)) is True
    assert app.bubble.isVisible()
    close(app)


def test_back_to_back_cloaks_bring_the_windows_back(qapp):
    """The worker often starts the next screenshot before the GUI has run the previous show()."""
    app, _, _ = make(qapp)
    app.shield.settle_s = 0.0

    def two_quick_captures():
        with app.shield.cloak():
            pass
        with app.shield.cloak():            # the queued show() from the first one hasn't run yet
            pass
        return True

    t = threading.Thread(target=two_quick_captures)
    t.start()
    while t.is_alive():
        qapp.processEvents()
    pump(qapp, 0.1)
    assert app.bubble.isVisible()
    close(app)


def test_rects_scale_to_mouse_pixels(qapp):
    app, session, _ = make(qapp)
    app.screen_px_size = (QApplication.primaryScreen().geometry().width() * 2,
                          QApplication.primaryScreen().geometry().height() * 2)   # e.g. 200 % scaling
    r = app._to_screen_px(QRect(10, 20, 30, 40))
    assert r == (20, 40, 80, 120)
    assert app._to_screen_px(QRect(-5000, -5000, 10, 10)) is None
    close(app)


def test_nested_cloaks_show_the_windows_once_at_the_end(qapp):
    calls = []

    class W:
        def __init__(self):
            self.visible = True

        def isVisible(self):
            return self.visible

        def hide(self):
            calls.append("hide")
            self.visible = False

        def show(self):
            calls.append("show")
            self.visible = True

        def frameGeometry(self):
            return QRect(0, 0, 10, 10)

    w = W()
    shield = Shield(lambda: [w], lambda widget: (0, 0, 10, 10), settle_s=0)

    def nested():
        with shield.cloak():
            with shield.clear_of([(5, 5)]):
                pass
        return True

    shield.refresh()
    run_in_worker(qapp, nested)
    assert calls == ["hide", "show"]


# --------------------------------------------------------------------------- helpers
def test_parse_hotkey():
    assert parse_hotkey("Ctrl+Alt+Space") == ({"ctrl", "alt"}, "space")
    assert parse_hotkey("control+option+l") == ({"ctrl", "alt"}, "l")
    assert parse_hotkey("cmd+shift+k") == ({"cmd", "shift"}, "k")
    for bad in ("space", "ctrl+alt", "hyper+x", ""):
        with pytest.raises(ValueError):
            parse_hotkey(bad)


def test_focus_tracker_skips_our_own_windows():
    seen = iter(["editor", "leanwatch", "browser", "leanwatch"])
    activated = []
    t = FocusTracker(lambda: next(seen), lambda h: h == "leanwatch", activated.append)
    for _ in range(4):
        t.poll()
    t.restore()
    assert activated == ["browser"]


def test_exclusion_is_double_checked_with_a_real_grab(qapp):
    app, _, _ = make(qapp, excluded=True)
    assert app.shield.excluded_from_capture
    frames = iter([b"same", b"same"])
    assert app.verify_exclusion(grab=lambda rect: next(frames)) is True
    frames = iter([b"bubble visible", b"desktop only"])        # the OS ignored the request
    assert app.verify_exclusion(grab=lambda rect: next(frames)) is False
    assert not app.shield.excluded_from_capture and app.bubble.isVisible()
    close(app)


def test_keys_go_back_to_the_app_before_the_agent_types(qapp):
    """The agent asks for the keyboard right before typing; if the person clicked into the box, it moves first."""
    app, session, native = make(qapp, granted=True)
    app.expand()
    app.submit("open settings")
    before = native.restored
    app.panel.isActiveWindow = lambda: True       # the person clicked into the box during the task
    pump(qapp, 0.3)
    assert native.restored == before              # no timer takes the keyboard away from the person
    assert run_in_worker(qapp, lambda: app.shield.before_keys() or True)
    assert native.restored == before + 1          # ...but it moves back right before the agent types
    app.panel.isActiveWindow = lambda: False
    run_in_worker(qapp, lambda: app.shield.before_keys())
    assert native.restored == before + 1          # nothing to do when the box doesn't have it
    close(app)


def test_allow_gives_the_keyboard_back_before_the_agent_continues(qapp):
    app, session, native = make(qapp, granted=True)
    order = []
    app.focus._activate = lambda h: order.append(("focus", h))
    session_handle = session.handle
    session.handle = lambda m: (order.append(("send", m["type"])), session_handle(m))[1]
    app.expand()
    app.submit("send the email")
    app.focus._current = lambda: "mail app"       # the agent is working in the mail app
    app._on_event({"type": "confirm_request", "id": "q1", "action": "Send the email?", "reason": "send"})
    app.focus._current = lambda: "chat app"       # the person checks something in another app meanwhile
    app.focus.poll()
    app.panel.isActiveWindow = lambda: True       # ...then clicks Allow in the box
    order.clear()
    app._answer(True)
    assert order == [("focus", "mail app"), ("send", "confirm")]   # the agent's app, and first
    close(app)


def test_expired_question_is_cleared(qapp):
    app, session, _ = make(qapp, granted=True)
    app.expand()
    app.submit("x")
    app._on_event({"type": "confirm_request", "id": "q9", "action": "Delete?", "reason": "delete"})
    assert app.panel.confirm.isVisible() and app.bubble.attention
    app._on_event({"type": "confirm_closed", "id": "q9", "allow": False, "timed_out": True})
    assert not app.panel.confirm.isVisible() and not app.bubble.attention and app.confirm_id is None
    assert "skipped" in app.panel.activity.text()
    app.collapse()
    assert not app.panel.isVisible()              # no stale question blocking the box any more
    close(app)


def test_opening_the_box_during_a_screenshot_waits_until_it_is_taken(qapp):
    app, session, _ = make(qapp, granted=True)
    app.shield.settle_s = 0.0
    seen = {}

    def capture():
        with app.shield.cloak():
            time.sleep(0.15)                       # the person presses the shortcut in the middle of it
            seen["panel"] = app.panel.isVisible()
        return True

    t = threading.Thread(target=capture)
    t.start()
    end = time.monotonic() + 0.05
    while time.monotonic() < end:
        qapp.processEvents()
    app.open_from_hotkey()
    while t.is_alive():
        qapp.processEvents()
    pump(qapp, 0.1)
    assert seen["panel"] is False                  # not in the model's screenshot
    assert app.panel.isVisible() and not app.bubble.isVisible()   # shown afterwards, without the bubble on top
    close(app)


def test_missing_os_permissions_are_explained_in_the_box(qapp):
    app, _, native = make(qapp)
    native.missing_permissions = lambda: [("Screen Recording (to see the screen)", "x-apple.systempreferences:...")]
    app.expand()
    assert app.panel.permits.isVisible() and "Screen Recording" in app.panel.permits.detail.text()
    QTest.mouseClick(app.panel.permits.no, Qt.LeftButton)
    assert not app.panel.permits.isVisible()
    close(app)
