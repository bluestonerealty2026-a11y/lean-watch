"""The LeanWatch bubble: a small round cursor button that floats above every app.

Click it and it opens into a command box. Type what you need ("turn on dark mode"),
press Go, and watch: the box shows each step in plain words while LeanWatch moves
the mouse, with Stop, Pause and Allow / Don't allow for risky steps.

    collapsed  ->  click (or the shortcut)  ->  expanded: command box
    expanded   ->  Go                       ->  running: steps, Stop
    running    ->  done                     ->  result, ready for the next command

It lives in the same process as the agent (desktop.py starts both), listens to the
Session's events, and gives the Session a Shield so its own windows never appear
in the model's screenshots or under the model's clicks.
"""

from __future__ import annotations

import html
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QAction, QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap, QRadialGradient, QRegion,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QSizePolicy, QSystemTrayIcon,
    QToolButton, QVBoxLayout, QWidget,
)

# Colours from the design: teal button, near-black box, dark field with a teal outline.
TEAL = "#3BBFB3"
TEAL_HOVER = "#55D2C6"
INK = "#0E181C"        # the cursor icon and text on teal
BG = "#0E191D"         # the box
FIELD = "#152328"      # the text field
LINE = "#23383E"
TEXT = "#E6F1F0"
MUTED = "#8FA6A8"
RED = "#E5484D"
AMBER = "#FFC928"

DIAMETER = 56          # the round button
PAD = 10               # room around it for the shadow and the activity ring
PANEL_WIDTH = 400
EDGE = 16              # distance from the screen edge for the default position
SETTLE_S = 0.15        # how long a hidden window takes to leave the screen image

STYLE = f"""
#card {{ background: {BG}; border: 1px solid {LINE}; border-radius: 16px; }}
QLabel {{ color: {TEXT}; font-size: 14px; }}
QLabel#brand {{ font-weight: 700; font-size: 14px; }}
QLabel#state, QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#step {{ font-size: 15px; }}
QLineEdit#cmd {{
    background: {FIELD}; color: {TEXT}; border: 2px solid {TEAL}; border-radius: 12px;
    padding: 9px 12px; font-size: 15px; selection-background-color: {TEAL}; selection-color: {INK};
}}
QLineEdit#cmd:disabled {{ border-color: #2C4A4C; color: {MUTED}; }}
QPushButton {{ border-radius: 12px; font-size: 14px; font-weight: 700; padding: 9px 16px; }}
QPushButton#go {{ background: {TEAL}; color: {INK}; border: none; min-width: 58px; font-size: 16px; }}
QPushButton#go:hover {{ background: {TEAL_HOVER}; }}
QPushButton#go[stop="true"] {{ background: {RED}; color: white; }}
QPushButton#primary {{ background: {TEAL}; color: {INK}; border: none; }}
QPushButton#primary:hover {{ background: {TEAL_HOVER}; }}
QPushButton#secondary {{ background: transparent; color: {TEXT}; border: 1px solid #3B5A5E; }}
QPushButton#secondary:hover {{ border-color: {TEAL}; }}
QPushButton#danger {{ background: {RED}; color: white; border: none; }}
QFrame#ask {{ background: {FIELD}; border: 1px solid #2B4A4F; border-radius: 12px; }}
QFrame#ask[tone="warn"] {{ border-color: {AMBER}; }}
QToolButton#close {{ color: {MUTED}; border: none; font-size: 18px; padding: 0 6px; }}
QToolButton#close:hover {{ color: {TEXT}; }}
"""


# ---------------------------------------------------------------------------
# Geometry helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------
def default_position(area: QRect, size: QSize) -> QPoint:
    """Bottom-right corner of the usable screen area (clear of the taskbar or dock)."""
    return QPoint(area.right() - size.width() - EDGE + 1, area.bottom() - size.height() - EDGE + 1)


def clamp_position(pos: QPoint, size: QSize, area: QRect) -> QPoint:
    x = min(max(pos.x(), area.left()), area.right() - size.width() + 1)
    y = min(max(pos.y(), area.top()), area.bottom() - size.height() + 1)
    return QPoint(x, y)


def panel_geometry(bubble: QRect, size: QSize, area: QRect) -> tuple[QRect, bool]:
    """Where the opened box goes: over the bubble, growing toward the middle of the screen.

    Returns (rect, input_at_bottom). A bubble in the lower half opens upward with the text
    field at the bottom, next to where the bubble was; in the upper half it opens downward.
    """
    right_side = bubble.center().x() >= area.center().x()
    lower_half = bubble.center().y() >= area.center().y()
    x = bubble.right() - size.width() + 1 if right_side else bubble.left()
    y = bubble.bottom() - size.height() + 1 if lower_half else bubble.top()
    pos = clamp_position(QPoint(x, y), size, area)
    return QRect(pos, size), lower_half


def cursor_path(size: float) -> QPainterPath:
    """The classic arrow pointer, `size` px tall, tip at (0, 0)."""
    s = size / 20.0
    pts = [(0, 0), (0, 16.5), (4.4, 12.6), (7.4, 19.4), (10.2, 18.2), (7.3, 11.6), (13, 11.6)]
    path = QPainterPath(QPointF(pts[0][0] * s, pts[0][1] * s))
    for x, y in pts[1:]:
        path.lineTo(x * s, y * s)
    path.closeSubpath()
    return path


def draw_logo(p: QPainter, rect: QRectF, fill: QColor) -> None:
    """Teal disc with a dark arrow pointer, as in the design."""
    p.setPen(Qt.NoPen)
    p.setBrush(fill)
    p.drawEllipse(rect)
    arrow = cursor_path(rect.height() * 0.42)
    box = arrow.boundingRect()
    # Centre the arrow's box, nudged slightly right so it looks optically centred.
    arrow.translate(rect.center().x() - box.width() / 2 + rect.width() * 0.03, rect.center().y() - box.height() / 2)
    p.setBrush(QColor(INK))
    p.drawPath(arrow)


def logo_icon(px: int = 64) -> QIcon:
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    draw_logo(p, QRectF(1, 1, px - 2, px - 2), QColor(TEAL))
    p.end()
    return QIcon(pm)


def on_gui_thread() -> bool:
    app = QApplication.instance()
    return app is not None and QThread.currentThread() == app.thread()


def translucency_ok(native) -> bool:
    """Round, see-through windows need a compositor. Without one, fall back to a shape mask."""
    try:
        return bool(native.supports_translucency())
    except Exception:
        return True


# ---------------------------------------------------------------------------
# The round button
# ---------------------------------------------------------------------------
class BubbleButton(QWidget):
    clicked = Signal()
    dropped = Signal(QPoint)          # after the person drags it somewhere
    menu_requested = Signal(QPoint)

    def __init__(self, translucent: bool = True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setObjectName("leanwatch-bubble")
        self.setWindowTitle("LeanWatch")
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)   # macOS hides tool windows of inactive apps otherwise
        self.translucent = translucent
        if translucent:
            self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(DIAMETER + 2 * PAD, DIAMETER + 2 * PAD)
        if not translucent:
            self.setMask(QRegion(QRect(PAD, PAD, DIAMETER, DIAMETER), QRegion.Ellipse))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("LeanWatch: click and tell me what you need help with")
        self.setAccessibleName("LeanWatch assistant")
        self.hover = False
        self.busy = False
        self.attention = False
        self.angle = 0.0
        self._press: QPoint | None = None
        self._origin = QPoint()
        self._dragging = False
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._tick)

    # state shown on the button
    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._animate()

    def set_attention(self, attention: bool) -> None:
        self.attention = attention
        self._animate()

    def _animate(self) -> None:
        if self.busy or self.attention:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self.angle = (self.angle + 4.5) % 360
        self.update()

    # painting
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        disc = QRectF(PAD, PAD, DIAMETER, DIAMETER)
        if self.translucent:
            r = DIAMETER / 2 + PAD
            shadow = QRadialGradient(disc.center() + QPointF(0, 3), r)
            shadow.setColorAt(DIAMETER / 2 / r - 0.04, QColor(0, 0, 0, 110))
            shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(shadow)
            p.drawEllipse(disc.center() + QPointF(0, 3), r, r)
        draw_logo(p, disc, QColor(TEAL_HOVER if self.hover else TEAL))
        if self.busy or self.attention:
            # A ring that runs round the edge while LeanWatch works (amber when it needs you).
            colour = QColor(AMBER) if self.attention else QColor(INK)
            colour.setAlpha(230 if self.attention else 170)
            pen = QPen(colour, 3.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            ring = disc.adjusted(4, 4, -4, -4)
            if self.attention:
                p.drawEllipse(ring)
            else:
                p.drawArc(ring, int(-self.angle * 16), 100 * 16)
        p.end()

    # mouse: click opens, drag moves, right click shows the menu
    def enterEvent(self, _event) -> None:
        self.hover = True
        self.update()

    def leaveEvent(self, _event) -> None:
        self.hover = False
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._press = e.globalPosition().toPoint()
            self._origin = self.pos()
            self._dragging = False
        elif e.button() == Qt.RightButton:
            self.menu_requested.emit(e.globalPosition().toPoint())

    def mouseMoveEvent(self, e) -> None:
        if self._press is None:
            return
        delta = e.globalPosition().toPoint() - self._press
        if not self._dragging and delta.manhattanLength() > 5:
            self._dragging = True
            self.setCursor(Qt.ClosedHandCursor)
        if self._dragging:
            self.move(self._origin + delta)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.LeftButton or self._press is None:
            return
        self._press = None
        self.setCursor(Qt.PointingHandCursor)
        if self._dragging:
            self._dragging = False
            self.dropped.emit(self.pos())
        else:
            self.clicked.emit()


# ---------------------------------------------------------------------------
# The opened box
# ---------------------------------------------------------------------------
class AskCard(QFrame):
    """A question with two buttons: permission, a risky step, or a paused run."""

    def __init__(self, yes: str, no: str, yes_style: str = "primary"):
        super().__init__()
        self.setObjectName("ask")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.PlainText)
        self.detail = QLabel()
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.PlainText)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.yes = QPushButton(yes)
        self.yes.setObjectName(yes_style)
        self.no = QPushButton(no)
        self.no.setObjectName("secondary")
        for b in (self.yes, self.no):
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)   # clicking must not pull keyboard focus from the app being worked on
        row.addWidget(self.yes)
        row.addWidget(self.no)
        row.addStretch(1)
        lay.addWidget(self.title)
        lay.addWidget(self.detail)
        lay.addLayout(row)
        self.hide()

    def ask(self, title: str, detail: str = "", warn: bool = False) -> None:
        self.title.setText(title)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.setProperty("tone", "warn" if warn else "")
        self.style().unpolish(self)
        self.style().polish(self)
        self.show()


class Panel(QWidget):
    submitted = Signal(str)
    stop_clicked = Signal()
    collapse_clicked = Signal()

    def __init__(self, translucent: bool = True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setObjectName("leanwatch-panel")
        self.setWindowTitle("LeanWatch")
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.translucent = translucent
        if translucent:
            self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)
        self.setFixedWidth(PANEL_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.card = QFrame()
        self.card.setObjectName("card")
        outer.addWidget(self.card)
        self.body = QVBoxLayout(self.card)
        self.body.setContentsMargins(14, 12, 14, 14)
        self.body.setSpacing(10)

        # header
        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        icon = QLabel()
        icon.setPixmap(logo_icon(40).pixmap(20, 20))
        self.brand = QLabel("LeanWatch")
        self.brand.setObjectName("brand")
        self.state = QLabel("Ready")
        self.state.setObjectName("state")
        self.close_btn = QToolButton()
        self.close_btn.setObjectName("close")
        self.close_btn.setText("–")
        self.close_btn.setToolTip("Shrink back to the bubble (Esc)")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setFocusPolicy(Qt.NoFocus)
        self.close_btn.clicked.connect(self.collapse_clicked)
        head.addWidget(icon)
        head.addWidget(self.brand)
        head.addWidget(self.state)
        head.addStretch(1)
        head.addWidget(self.close_btn)

        # what is happening
        self.notice = QLabel("What do you need help with? I'll do it step by step and explain as I go.")
        self.notice.setObjectName("muted")
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.PlainText)
        self.steps = QLabel()
        self.steps.setObjectName("step")
        self.steps.setWordWrap(True)
        self.steps.setTextFormat(Qt.RichText)
        self.steps.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.steps.hide()
        self.activity = QLabel()
        self.activity.setObjectName("muted")
        self.activity.setWordWrap(True)
        self.activity.setTextFormat(Qt.PlainText)
        self.activity.hide()
        self.consent = AskCard("Allow for this session", "Not now")
        self.confirm = AskCard("Allow", "Don't allow")
        self.paused = AskCard("Resume", "Stop", yes_style="primary")
        self.paused.no.setObjectName("danger")

        # the command line, as in the design: dark field with a teal outline, teal Go button
        self.input_row = QWidget()
        row = QHBoxLayout(self.input_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.input = QLineEdit()
        self.input.setObjectName("cmd")
        self.input.setPlaceholderText("e.g. turn on dark mode")
        self.input.setAccessibleName("What should LeanWatch do?")
        self.input.setMaxLength(2000)
        pal = self.input.palette()
        pal.setColor(QPalette.PlaceholderText, QColor(MUTED))
        self.input.setPalette(pal)
        self.input.returnPressed.connect(self._go)
        self.go = QPushButton("Go")
        self.go.setObjectName("go")
        self.go.setCursor(Qt.PointingHandCursor)
        self.go.setFocusPolicy(Qt.NoFocus)
        self.go.clicked.connect(self._go)
        self.go.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        row.addWidget(self.input, 1)
        row.addWidget(self.go)

        self.feed = [self.notice, self.steps, self.activity, self.consent, self.confirm, self.paused]
        self.input_at_bottom = True
        self._layout_rows()
        self.running = False
        self.anchor: tuple[QRect, QRect] | None = None   # (bubble rect, screen area) the panel is placed against

    # layout ------------------------------------------------------------
    def _layout_rows(self) -> None:
        """Header first; the text field at the bottom (box opened upward) or right under the header."""
        while self.body.count():
            self.body.takeAt(0)
        rows = self.feed + [self.input_row] if self.input_at_bottom else [self.input_row] + self.feed
        for w in [self.header] + rows:
            self.body.addWidget(w)

    def set_input_at_bottom(self, bottom: bool) -> None:
        if bottom != self.input_at_bottom:
            self.input_at_bottom = bottom
            self._layout_rows()

    def place(self, bubble: QRect, area: QRect) -> None:
        """Size to the content and sit over the bubble, growing toward the middle of the screen."""
        self.anchor = (bubble, area)
        self.card.layout().activate()
        self.adjustSize()
        rect, bottom = panel_geometry(bubble, QSize(PANEL_WIDTH, self.sizeHint().height()), area)
        if bottom != self.input_at_bottom:
            self.set_input_at_bottom(bottom)
            self.card.layout().activate()
            self.adjustSize()
            rect, _ = panel_geometry(bubble, QSize(PANEL_WIDTH, self.sizeHint().height()), area)
        self.setGeometry(rect)
        if not self.translucent:
            self.setMask(QRegion(self.rect(), QRegion.Rectangle))

    def refit(self) -> None:
        """After the content changes, keep the same corner next to the bubble."""
        if self.anchor and self.isVisible():
            self.place(*self.anchor)

    # content -------------------------------------------------------------
    def set_state(self, text: str) -> None:
        self.state.setText(text)

    def set_running(self, running: bool) -> None:
        self.running = running
        self.input.setEnabled(not running)
        self.go.setText("Stop" if running else "Go")
        self.go.setProperty("stop", "true" if running else "false")
        self.go.setToolTip("Stop now (Esc)" if running else "Start")
        self.go.style().unpolish(self.go)
        self.go.style().polish(self.go)

    def show_steps(self, steps: list[str]) -> None:
        if not steps:
            self.steps.hide()
            return
        shown = steps[-4:]
        first = len(steps) - len(shown) + 1
        rows = []
        for i, text in enumerate(shown):
            n = first + i
            latest = i == len(shown) - 1
            colour = TEXT if latest else MUTED
            weight = "600" if latest else "400"
            rows.append(f"<div style='color:{colour}; font-weight:{weight}; margin-bottom:6px'>"
                        f"<span style='color:{TEAL}'>{n}.</span> {html.escape(text)}</div>")
        self.steps.setText("".join(rows))
        self.steps.show()

    def show_activity(self, text: str) -> None:
        self.activity.setText(text)
        self.activity.setVisible(bool(text))

    def _go(self) -> None:
        if self.running:
            self.stop_clicked.emit()
            return
        text = self.input.text().strip()
        if text:
            self.submitted.emit(text)

    def keyPressEvent(self, e) -> None:
        # The text field passes Esc on to here.
        if e.key() == Qt.Key_Escape:
            if self.running:
                self.stop_clicked.emit()
            else:
                self.collapse_clicked.emit()
            return
        super().keyPressEvent(e)


# ---------------------------------------------------------------------------
# Keeping LeanWatch's own windows out of the model's way
# ---------------------------------------------------------------------------
class Shield(QObject):
    """Used by the agent's worker thread (see ScreenCapture and Controller).

    cloak():          our windows leave the screen while a screenshot is taken, unless the
                      operating system already keeps them out of screenshots.
    clear_of(points): our windows leave the screen while the model clicks a spot they cover.
    """

    _hide = Signal()
    _show = Signal()

    def __init__(self, windows, to_screen_px, settle_s: float = SETTLE_S):
        super().__init__()
        self.windows = windows                 # callable -> list of our top-level widgets
        self.to_screen_px = to_screen_px       # callable: QRect (Qt logical) -> (x0, y0, x1, y1) mouse pixels
        self.settle_s = settle_s
        self.excluded_from_capture = False     # set when the OS hides our windows from screenshots
        self._lock = threading.Lock()
        self._depth = 0
        self._hidden: list[QWidget] = []
        self._rects: list[tuple[int, int, int, int]] = []
        self._hide.connect(self._do_hide, Qt.BlockingQueuedConnection)
        self._show.connect(self._do_show, Qt.QueuedConnection)

    # called on the GUI thread whenever a window moves, resizes, shows or hides
    def refresh(self) -> None:
        rects = []
        for w in self.windows():
            try:
                if w.isVisible():
                    rects.append(self.to_screen_px(w.frameGeometry()))
            except RuntimeError:   # the window is being destroyed (LeanWatch is quitting)
                continue
        with self._lock:
            self._rects = [r for r in rects if r is not None]

    def covers(self, x: int, y: int, margin: int = 12) -> bool:
        with self._lock:
            return any(x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin for x0, y0, x1, y1 in self._rects)

    @contextmanager
    def cloak(self):
        if self.excluded_from_capture or on_gui_thread():
            yield
            return
        self._enter()
        try:
            yield
        finally:
            self._leave()

    @contextmanager
    def clear_of(self, points):
        if on_gui_thread() or not any(self.covers(x, y) for x, y in points):
            yield
            return
        self._enter()
        try:
            yield
        finally:
            self._leave()

    def _enter(self) -> None:
        with self._lock:
            self._depth += 1
            first = self._depth == 1
        if first:
            self._hide.emit()          # waits until the GUI thread has hidden the windows
            time.sleep(self.settle_s)  # and until the compositor has redrawn without them

    def _leave(self) -> None:
        with self._lock:
            self._depth -= 1
            last = self._depth == 0
        if last:
            self._show.emit()

    @Slot()
    def _do_hide(self) -> None:
        self._hidden = [w for w in self.windows() if w.isVisible()]
        for w in self._hidden:
            w.hide()

    @Slot()
    def _do_show(self) -> None:
        with self._lock:
            if self._depth:
                return  # another cloak started in the meantime
        for w in self._hidden:
            w.show()
        self._hidden = []
        self.refresh()


# ---------------------------------------------------------------------------
# Wiring the widgets to the agent
# ---------------------------------------------------------------------------
class BubbleApp(QObject):
    """Connects the bubble and the box to a Session (server.Session) on this computer."""

    session_event = Signal(dict)
    hotkey_pressed = Signal()

    def __init__(self, session, native, settings_path: Path | None = None, screen_px_size: tuple[int, int] | None = None):
        super().__init__()
        self.session = session
        self.native = native
        self.settings_path = settings_path
        self.screen_px_size = screen_px_size or getattr(session, "screen_size", None)
        translucent = translucency_ok(native)
        self.bubble = BubbleButton(translucent)
        self.panel = Panel(translucent)
        self.expanded = False
        self.running = False
        self.pending_text: str | None = None     # command waiting for permission
        self.confirm_id: str | None = None
        self.steps: list[str] = []
        self.hidden_by_user = False
        self.focus = native.focus_tracker()

        self.shield = Shield(self.windows, self._to_screen_px)
        self.session.shield = self.shield

        self.bubble.clicked.connect(self.toggle)
        self.bubble.dropped.connect(self._dropped)
        self.bubble.menu_requested.connect(self._menu)
        self.panel.submitted.connect(self.submit)
        self.panel.stop_clicked.connect(self.stop)
        self.panel.collapse_clicked.connect(self.collapse)
        self.panel.consent.yes.clicked.connect(lambda: self._consent(True))
        self.panel.consent.no.clicked.connect(lambda: self._consent(False))
        self.panel.confirm.yes.clicked.connect(lambda: self._answer(True))
        self.panel.confirm.no.clicked.connect(lambda: self._answer(False))
        self.panel.paused.yes.clicked.connect(self.resume)
        self.panel.paused.no.clicked.connect(self.stop)
        self.session_event.connect(self._on_event)        # queued: events arrive from the worker thread
        self.hotkey_pressed.connect(self.open_from_hotkey)
        self.session.listeners.append(self.session_event.emit)

        self.tray = self._make_tray()
        self._keep_on_top = QTimer(self, interval=2500)
        self._keep_on_top.timeout.connect(self._raise_windows)
        self._keep_on_top.start()
        # Keep track of the app the person is using (not LeanWatch), to hand the keyboard back to it.
        self._focus_poll = QTimer(self, interval=400)
        self._focus_poll.timeout.connect(self.focus.poll)
        self._focus_poll.start()
        self._place_bubble()
        for w in self.windows():
            w.installEventFilter(self)

    # ---------------- windows ----------------
    def windows(self) -> list[QWidget]:
        return [self.bubble, self.panel]

    def show(self) -> None:
        for w in self.windows():
            w.winId()                     # create the native window so the OS tweaks can apply
            self.native.prepare_window(w)
        self.shield.excluded_from_capture = all(self.native.exclude_from_capture(w) for w in self.windows())
        if not self.hidden_by_user:
            self.bubble.show()
        self.shield.refresh()

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Move, QEvent.Resize, QEvent.Show, QEvent.Hide):
            self.shield.refresh()
        if event.type() == QEvent.Show:
            # Window managers forget "on every desktop / above others" when a window hides; set it again.
            QTimer.singleShot(0, lambda w=obj: self._after_show(w))
        return False

    def _after_show(self, w) -> None:
        try:
            if w.isVisible():
                self.native.after_show(w)
        except Exception:
            pass

    def _area(self, rect: QRect | None = None) -> QRect:
        screen = QGuiApplication.screenAt((rect or self.bubble.frameGeometry()).center()) or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _place_bubble(self) -> None:
        saved = self._load_settings()
        size = self.bubble.size()
        pos = None
        if "x" in saved and "y" in saved:
            pos = QPoint(int(saved["x"]), int(saved["y"]))
            if QGuiApplication.screenAt(QRect(pos, size).center()) is None:
                pos = None                # that monitor is gone
        area = self._area(QRect(pos, size)) if pos else QGuiApplication.primaryScreen().availableGeometry()
        self.bubble.move(clamp_position(pos, size, area) if pos else default_position(area, size))
        self.hidden_by_user = bool(saved.get("hidden"))

    def _dropped(self, pos: QPoint) -> None:
        area = self._area()
        self.bubble.move(clamp_position(pos, self.bubble.size(), area))
        self._save_settings()

    def _to_screen_px(self, r: QRect):
        """Qt's logical coordinates -> the mouse pixels pyautogui uses, on the main monitor."""
        primary = QGuiApplication.primaryScreen()
        g = primary.geometry()
        if not r.intersects(g):
            return None                  # the agent only works on the main monitor
        sx = sy = 1.0
        if self.screen_px_size:
            sx, sy = self.screen_px_size[0] / g.width(), self.screen_px_size[1] / g.height()
        return (round((r.left() - g.left()) * sx), round((r.top() - g.top()) * sy),
                round((r.right() + 1 - g.left()) * sx), round((r.bottom() + 1 - g.top()) * sy))

    def _raise_windows(self) -> None:
        """Some apps and the taskbar push other top-most windows down; quietly come back on top."""
        for w in self.windows():
            if w.isVisible():
                w.raise_()

    # ---------------- open and close ----------------
    def toggle(self) -> None:
        if self.expanded:
            self.collapse()
        else:
            self.expand(focus=True)

    def expand(self, focus: bool = True) -> None:
        self.focus.poll()                     # the app the person was using, to give it back the keyboard
        self.panel.place(self.bubble.frameGeometry(), self._area())
        self.bubble.hide()
        self.panel.show()
        self.panel.raise_()
        self.expanded = True
        if focus and not self.running:
            self.native.activate_own(self.panel)
            self.panel.activateWindow()
            self.panel.input.setFocus(Qt.OtherFocusReason)
        self.shield.refresh()

    def collapse(self) -> None:
        if self.confirm_id or self.pending_text:
            return                            # a question is waiting; keep it on screen
        self.panel.hide()
        self.expanded = False
        if not self.hidden_by_user:
            self.bubble.show()
        if not self.running:
            self.focus.restore()
        self.shield.refresh()

    def open_from_hotkey(self) -> None:
        if self.hidden_by_user:
            self.set_hidden(False)
        if self.expanded and not self.running:
            self.collapse()
        else:
            self.expand(focus=True)

    def set_hidden(self, hidden: bool) -> None:
        self.hidden_by_user = hidden
        if hidden:
            self.panel.hide()
            self.bubble.hide()
            self.expanded = False
        else:
            self.bubble.show()
        self._save_settings()
        self.shield.refresh()

    # ---------------- commands ----------------
    def submit(self, text: str) -> None:
        if self.running:
            return
        if not self.session.permissions.granted:
            self.pending_text = text
            self.panel.consent.ask(
                "May I use your mouse and keyboard?",
                "I'll move the pointer, click and type on this computer to show you how it's done. "
                "Stop me any time with Stop or Esc, by moving the mouse yourself, or by pushing the mouse into a screen corner.")
            self.panel.refit()
            return
        self._start(text)

    def _consent(self, allow: bool) -> None:
        text, self.pending_text = self.pending_text, None
        self.panel.consent.hide()
        if allow:
            self.session.handle({"type": "grant", "scopes": {"mouse": True, "keyboard": True}})
            if text:
                self._start(text)
        else:
            self.panel.notice.setText("Okay. Nothing will move until you allow it.")
        self.panel.refit()

    def _start(self, text: str) -> None:
        reply = self.session.handle({"type": "command", "text": text})
        if reply and reply.get("type") == "error":
            self.panel.notice.setText(reply.get("message", "Something went wrong."))
            self.panel.refit()
            return
        self.panel.input.clear()
        self.panel.input.setPlaceholderText(text[:80])
        # Give the keyboard back to the app the person was using: LeanWatch will type into it.
        self.focus.restore()

    def stop(self) -> None:
        self.session.handle({"type": "stop"})
        if self.confirm_id:
            self._answer(False)

    def resume(self) -> None:
        self.panel.paused.hide()
        self.session.handle({"type": "resume"})
        self.panel.refit()

    def _answer(self, allow: bool) -> None:
        cid, self.confirm_id = self.confirm_id, None
        self.panel.confirm.hide()
        self.bubble.set_attention(False)
        if cid:
            self.session.handle({"type": "confirm", "id": cid, "allow": allow})
        self.panel.refit()

    # ---------------- events from the agent ----------------
    @Slot(dict)
    def _on_event(self, msg: dict) -> None:
        kind = msg.get("type")
        p = self.panel
        if kind == "run_started":
            self.running = True
            self.steps = []
            p.notice.setText(f"“{msg.get('text', '')}”")
            p.show_steps([])
            p.show_activity("Getting ready…")
            p.set_running(True)
            p.set_state("Working")
            self.bubble.set_busy(True)
        elif kind == "narration":
            text = str(msg.get("text", "")).strip()
            if not text:
                return
            if msg.get("kind") == "thinking":
                p.show_activity(text)
            elif msg.get("kind") == "final":
                p.show_activity("")
                p.notice.setText("✓ " + text)
            else:
                self.steps.append(text)
                p.show_steps(self.steps)
                p.show_activity("")
        elif kind == "action":
            p.show_activity(str(msg.get("summary", "")))
        elif kind == "confirm_request":
            self.confirm_id = str(msg.get("id"))
            p.confirm.ask(str(msg.get("action", "May I do this?")), str(msg.get("reason", "")), warn=True)
            self.bubble.set_attention(True)
            if not self.expanded:
                self.expand(focus=False)
        elif kind == "paused":
            p.paused.ask("Paused", str(msg.get("reason", "")))
            p.set_state("Paused")
            if not self.expanded:
                self.expand(focus=False)
        elif kind == "resumed":
            p.paused.hide()
            p.set_state("Working")
        elif kind == "run_finished":
            self.running = False
            self.confirm_id = None
            p.confirm.hide()
            p.paused.hide()
            p.set_running(False)
            p.set_state("Ready")
            p.show_activity("")
            p.input.setPlaceholderText("e.g. turn on dark mode")
            self.bubble.set_busy(False)
            self.bubble.set_attention(False)
            if msg.get("ok"):
                p.notice.setText("✓ " + (msg.get("result") or "Done."))
            else:
                p.notice.setText("✕ " + (msg.get("message") or msg.get("result") or "Stopped."))
            if not self.expanded:
                self.expand(focus=False)
        else:
            return
        p.refit()

    # ---------------- menu, tray and settings ----------------
    def _menu_actions(self, menu: QMenu) -> None:
        menu.addAction("Open the command box", lambda: self.expand(focus=True))
        menu.addAction("Show the bubble" if self.hidden_by_user else "Hide the bubble",
                       lambda: self.set_hidden(not self.hidden_by_user))
        menu.addAction("Move to the bottom-right corner", self._reset_position)
        menu.addSeparator()
        if self.running:
            menu.addAction("Stop the current task", self.stop)
        quit_action = QAction("Quit LeanWatch", menu)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

    def _menu(self, at: QPoint) -> None:
        menu = QMenu()
        self._menu_actions(menu)
        menu.exec(at)

    def _make_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(logo_icon(), self)
        tray.setToolTip("LeanWatch")
        menu = QMenu()
        menu.aboutToShow.connect(lambda: (menu.clear(), self._menu_actions(menu)))
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: self.expand(focus=True) if reason == QSystemTrayIcon.Trigger else None)
        tray.show()
        return tray

    def _reset_position(self) -> None:
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.bubble.move(default_position(area, self.bubble.size()))
        self.set_hidden(False)
        self._save_settings()

    def _load_settings(self) -> dict:
        if not self.settings_path:
            return {}
        try:
            data = json.loads(self.settings_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self) -> None:
        if not self.settings_path:
            return
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            pos = self.bubble.pos()
            self.settings_path.write_text(json.dumps({"x": pos.x(), "y": pos.y(), "hidden": self.hidden_by_user}))
        except OSError:
            pass
