"""Wayland support (leanwatch/wayland.py) with fakes instead of D-Bus, PipeWire and GStreamer.

The real thing was checked end to end in a headless GNOME Shell (tests/gnome/); these tests pin the
behaviour quickly. No test here may reach the real session bus or the real ~/.leanwatch.
"""

import contextlib
import itertools
import json
import os
import queue
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from PIL import Image

from leanwatch import wayland
from leanwatch.controller import Controller, EmergencyStop
from leanwatch.permissions import Permissions
from leanwatch.screen import Geometry, ScreenCapture

RD, SC = wayland.RD, wayland.SC
SESSION = "/org/freedesktop/portal/desktop/session/1_42/leanwatch_s1"
FAKE_HELPER = Path(__file__).with_name("fake_pw_frames.py")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private HOME and state file, no session bus, no GStreamer probing."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/nonexistent/leanwatch-unit-tests")
    state = home / ".leanwatch" / "wayland.json"
    monkeypatch.setattr(wayland, "STATE_FILE", state)
    # RemoteDesktop's default argument was bound at import time: point it at the private file too.
    monkeypatch.setattr(wayland.RemoteDesktop.__init__, "__defaults__", (None, state))

    def no_bus(self):
        raise AssertionError("a unit test tried to open the real session bus")

    monkeypatch.setattr(wayland.PortalBus, "__init__", no_bus)
    monkeypatch.setattr(wayland, "_gst_python", [None])        # "no GStreamer" unless a test says otherwise
    return state


class FakeTime:
    """Stands in for the time module inside wayland.py: no real sleeping, a clock that tests can move."""

    def __init__(self, step=0.0):
        self.t = 1000.0
        self.step = step
        self.slept = []

    def monotonic(self):
        self.t += self.step
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


@pytest.fixture
def fake_time(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(wayland, "time", clock)
    return clock


# --------------------------------------------------------------------------- key names -> X keysyms
@pytest.mark.parametrize("key,sym", [
    ("a", 0x61), ("A", 0x41), ("1", 0x31), (" ", 0x20), ("~", 0x7E),
    ("é", 0xE9), ("ÿ", 0xFF), (" ", 0xA0),                       # Latin-1: the keysym is the code point
    ("€", 0x010020AC), ("ő", 0x01000151), ("😀", 0x0101F600),         # beyond Latin-1: Unicode keysyms
    ("\n", 0xFF0D), ("\t", 0xFF09),
])
def test_char_keysym_latin1_and_unicode(key, sym):
    assert wayland.char_keysym(key) == sym
    assert wayland.keysym_for(key) == sym


def test_control_characters_are_unicode_keysyms_not_latin1():
    assert wayland.char_keysym("\x7f") == 0x0100007F
    assert wayland.char_keysym("\x85") == 0x01000085      # between the two Latin-1 ranges


@pytest.mark.parametrize("key,sym", [
    ("enter", 0xFF0D), ("Return", 0xFF0D), ("ESC", 0xFF1B), ("backspace", 0xFF08), ("tab", 0xFF09),
    ("space", 0x20), ("delete", 0xFFFF), ("del", 0xFFFF), ("home", 0xFF50), ("end", 0xFF57),
    ("pageup", 0xFF55), ("pagedown", 0xFF56), ("up", 0xFF52), ("down", 0xFF54), ("left", 0xFF51), ("right", 0xFF53),
    ("printscreen", 0xFF61), ("apps", 0xFF67), ("volumeup", 0x1008FF13), ("volumemute", 0x1008FF12),
])
def test_named_keys(key, sym):
    assert wayland.keysym_for(key) == sym


@pytest.mark.parametrize("key,sym", [
    ("ctrl", 0xFFE3), ("ctrlright", 0xFFE4), ("alt", 0xFFE9), ("option", 0xFFE9), ("altright", 0xFFEA),
    ("shift", 0xFFE1), ("shiftright", 0xFFE2), ("win", 0xFFEB), ("command", 0xFFEB), ("super", 0xFFEB),
    ("winright", 0xFFEC), ("capslock", 0xFFE5),
])
def test_modifiers(key, sym):
    assert wayland.keysym_for(key) == sym


@pytest.mark.parametrize("key,sym", [("f1", 0xFFBE), ("F5", 0xFFC2), ("f10", 0xFFC7), ("f12", 0xFFC9), ("F24", 0xFFD5)])
def test_function_keys(key, sym):
    assert wayland.keysym_for(key) == sym


def test_single_letters_f_and_F_are_letters_not_function_keys():
    assert wayland.keysym_for("f") == 0x66
    assert wayland.keysym_for("F") == 0x46


@pytest.mark.parametrize("key", ["f0", "f25", "hyper", "kp_add", "ctrl+c", "", "pgupp"])
def test_unknown_key_raises(key):
    with pytest.raises(ValueError):
        wayland.keysym_for(key)


def test_unknown_key_raises_value_error_with_its_name():
    with pytest.raises(ValueError, match="kp_add"):
        wayland.keysym_for("kp_add")


# --------------------------------------------------------------------------- PortalBackend
class RecordingRemote:
    """A RemoteDesktop that records the input instead of sending it to the desktop."""

    def __init__(self, size=(1280, 720)):
        self.size = size
        self.events = []

    def move(self, x, y):
        self.events.append(("move", x, y))

    def button(self, button, pressed):
        self.events.append(("button", button, pressed))

    def scroll_steps(self, axis, steps):
        self.events.append(("scroll", axis, steps))

    def key(self, keysym, pressed):
        self.events.append(("key", keysym, pressed))

    def held(self):
        down = set()
        for e in self.events:
            if e[0] == "key":
                (down.add if e[2] else down.discard)(e[1])
        return down


def backend(size=(1280, 720)):
    remote = RecordingRemote(size)
    return wayland.PortalBackend(remote, key_interval=0), remote


def test_backend_size_and_start_position(fake_time):
    b, _ = backend((1280, 720))
    assert b.size() == (1280, 720) and b.position() == (640, 360)
    b, _ = backend(None)                               # the desktop didn't say: a common size
    assert b.size() == (1920, 1080) and b.position() == (960, 540)


@pytest.mark.parametrize("dx,dy,expected", [
    (0, 3, [("scroll", 0, -3)]),          # pyautogui: dy > 0 is up; the portal: negative is up
    (0, -2, [("scroll", 0, 2)]),          # down
    (4, 0, [("scroll", 1, 4)]),           # right: positive in both
    (-1, 0, [("scroll", 1, -1)]),         # left
    (2, -5, [("scroll", 0, 5), ("scroll", 1, 2)]),
    (0, 0, []),
])
def test_scroll_signs(dx, dy, expected, fake_time):
    b, remote = backend()
    b.scroll(dx, dy)
    assert remote.events == expected


def test_move_to_clamps_to_the_shared_screen(fake_time):
    b, remote = backend((1280, 720))
    b.move_to(-15, 5000)
    b.move_to(100.7, 200.2)
    b.move_to(99999, -1)
    assert remote.events == [("move", 0, 719), ("move", 100, 200), ("move", 1279, 0)]
    assert b.position() == (1279, 0)


@pytest.mark.parametrize("start,first", [((640, 360), (641, 360)), ((1279, 10), (1278, 10)), ((0, 719), (1, 719))])
def test_nudge_moves_one_pixel_and_back_inside_the_screen(start, first, fake_time):
    b, remote = backend((1280, 720))
    b.move_to(*start)
    remote.events.clear()
    b.nudge()
    assert remote.events == [("move", *first), ("move", *start)]
    assert b.position() == start


def test_buttons_and_click(fake_time):
    b, remote = backend()
    b.click("right")
    b.mouse_down()
    b.mouse_up("left")
    assert remote.events == [("button", "right", True), ("button", "right", False),
                             ("button", "left", True), ("button", "left", False)]


def test_keys_and_typing(fake_time):
    b, remote = backend()
    b.key_down("ctrl")
    b.key_down("a")
    b.key_up("a")
    b.key_up("ctrl")
    assert remote.events == [("key", 0xFFE3, True), ("key", 0x61, True), ("key", 0x61, False), ("key", 0xFFE3, False)]
    remote.events.clear()
    b.write("Hé€\n")
    syms = [0x48, 0xE9, 0x010020AC, 0xFF0D]
    assert remote.events == [e for s in syms for e in (("key", s, True), ("key", s, False))]
    assert remote.held() == set()


def make_controller(b, capture=None):
    perms = Permissions()
    perms.grant(mouse=True, keyboard=True)
    return Controller(b, capture, Geometry.for_screen(*b.size()), perms)


@pytest.mark.parametrize("direction,amount,expected", [
    ("down", 2, ("scroll", 0, 2)), ("up", 3, ("scroll", 0, -3)), ("right", 1, ("scroll", 1, 1)), ("left", 4, ("scroll", 1, -4)),
])
def test_controller_scroll_directions_reach_the_portal_with_its_signs(direction, amount, expected, fake_time):
    b, remote = backend()
    make_controller(b).execute("scroll", {"scroll_direction": direction, "scroll_amount": amount})
    assert remote.events == [expected]


def test_controller_key_combo_through_the_portal(fake_time):
    b, remote = backend()
    make_controller(b).execute("key", {"text": "ctrl+shift+t"})
    assert [e for e in remote.events] == [("key", 0xFFE3, True), ("key", 0xFFE1, True), ("key", 0x74, True),
                                          ("key", 0x74, False), ("key", 0xFFE1, False), ("key", 0xFFE3, False)]


def test_an_unknown_key_in_a_combo_leaves_no_key_held_down(fake_time):
    b, remote = backend()
    with pytest.raises(ValueError):
        make_controller(b).execute("key", {"text": "ctrl+kp_add"})
    assert remote.held() == set()


# --------------------------------------------------------------------------- remembering the answer
def test_will_ask(isolated):
    assert wayland.will_ask() is True                            # nothing saved yet
    isolated.parent.mkdir(parents=True)
    isolated.write_text('{"restore_token": "abc"}')
    assert wayland.will_ask() is False
    isolated.write_text('{"restore_token": ""}')
    assert wayland.will_ask() is True
    isolated.write_text("{not json")
    assert wayland.will_ask() is True
    other = isolated.parent / "other.json"
    other.write_text('{"restore_token": "x"}')
    assert wayland.will_ask(other) is False


@pytest.mark.parametrize("content", ["null", "[]", "3"])
def test_a_state_file_that_is_not_an_object_means_ask_again(isolated, content):
    isolated.parent.mkdir(parents=True)
    isolated.write_text(content)
    assert wayland.will_ask() is True
    assert wayland.RemoteDesktop(bus=object())._load_token() is None


# --------------------------------------------------------------------------- variants from jeepney
def test_unwrap_plain_values_from_variants():
    results = {
        "session_handle": ("o", "/org/freedesktop/portal/desktop/session/1_42/x"),
        "devices": ("u", 3),
        "restore_token": ("s", "3f6c-token"),
        "streams": ("a(ua{sv})", [(44, {"size": ("(ii)", (1280, 720)), "position": ("(ii)", (0, 0)),
                                        "source_type": ("u", 1)})]),
    }
    assert wayland._unwrap(results) == {
        "session_handle": "/org/freedesktop/portal/desktop/session/1_42/x",
        "devices": 3,
        "restore_token": "3f6c-token",
        "streams": [(44, {"size": (1280, 720), "position": (0, 0), "source_type": 1})],
    }
    assert wayland._unwrap(("v", ("u", 7))) == 7                  # a variant inside a variant
    assert wayland._unwrap((1, 2)) == (1, 2)
    assert wayland._unwrap(("not a signature!", 2)) == ("not a signature!", 2)
    assert wayland._unwrap(("", 2)) == ("", 2)


# --------------------------------------------------------------------------- PortalBus.request
class ScriptedQueue:
    """The queue router.filter() gives: answers from a list, or queue.Empty at once while the clock moves on."""

    def __init__(self, clock):
        self.clock = clock
        self.items = []
        self.gets = 0

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        self.gets += 1
        if self.items:
            return self.items.pop(0)
        self.clock.t += timeout or 0
        raise queue.Empty


class Handle:
    def __init__(self, router, rule, q):
        self.router, self.rule, self.q = router, rule, q

    def __enter__(self):
        self.router.entered.append(self.rule)
        return self.q

    def __exit__(self, *exc):
        self.router.exited.append(self.rule)
        return False


def request_bus(clock, answer=None):
    """A PortalBus without a D-Bus connection (bypasses __init__). answer(bus, method) runs when the portal
    method is called, and usually puts the Response into bus.q."""
    bus = wayland.PortalBus.__new__(wayland.PortalBus)
    bus.sender = "1_42"
    bus._tokens = itertools.count(1)
    bus.q = ScriptedQueue(clock)
    bus.router = SimpleNamespace(entered=[], exited=[])
    bus.router.filter = lambda rule: Handle(bus.router, rule, bus.q)
    bus.calls, bus.matches, bus.unmatched = [], [], []

    def match(**rule):
        bus.matches.append(rule)
        return rule

    def call(interface, method, signature="", body=(), **kw):
        bus.calls.append((interface, method, signature, body, kw))
        if interface != "org.freedesktop.portal.Request" and answer is not None:
            answer(bus, method)
        return ()

    bus.match = match
    bus.unmatch = bus.unmatched.append
    bus.call = call
    return bus


def response(code, results=None):
    return SimpleNamespace(body=(code, results or {}))


def test_request_returns_plain_results_and_cleans_up(fake_time):
    bus = request_bus(fake_time, lambda bus, m: bus.q.put(response(0, {"session_handle": ("o", "/s/1"), "n": ("u", 2)})))
    options = {"session_handle_token": ("s", "abc")}
    out = bus.request(RD, "CreateSession", "a{sv}", (), options)
    assert out == {"session_handle": "/s/1", "n": 2}
    token = f"leanwatch{os.getpid()}_1"
    path = f"/org/freedesktop/portal/desktop/request/1_42/{token}"
    rule = {"interface": "org.freedesktop.portal.Request", "member": "Response", "path": path}
    assert bus.matches == [rule]                                    # subscribed before the call
    assert bus.calls == [(RD, "CreateSession", "a{sv}", ({"session_handle_token": ("s", "abc"),
                                                          "handle_token": ("s", token)},), {})]
    assert options == {"session_handle_token": ("s", "abc")}       # the caller's dict is left alone
    assert bus.unmatched == [rule] and bus.router.exited == [rule]


def test_request_keeps_the_arguments_before_the_options(fake_time):
    bus = request_bus(fake_time, lambda bus, m: bus.q.put(response(0)))
    bus.request(RD, "Start", "osa{sv}", ("/s/1", ""), {})
    assert bus.calls[0][3] == ("/s/1", "", {"handle_token": ("s", f"leanwatch{os.getpid()}_1")})


def test_tokens_are_unique_per_call(fake_time):
    bus = request_bus(fake_time)
    assert bus.token() != bus.token()


def test_request_code_1_means_the_person_said_no(fake_time):
    bus = request_bus(fake_time, lambda bus, m: bus.q.put(response(1)))
    with pytest.raises(wayland.PortalDenied, match="chose not to allow"):
        bus.request(RD, "Start", "osa{sv}", ("/s", ""), {})
    assert len(bus.unmatched) == 1


def test_request_code_2_is_an_error_not_a_no(fake_time):
    bus = request_bus(fake_time, lambda bus, m: bus.q.put(response(2)))
    with pytest.raises(wayland.PortalError, match=r"refused Start \(code 2\)") as err:
        bus.request(RD, "Start", "osa{sv}", ("/s", ""), {})
    assert not isinstance(err.value, wayland.PortalDenied)
    assert len(bus.unmatched) == 1


def test_stop_closes_the_dialog_and_says_so(fake_time):
    stop = threading.Event()
    bus = request_bus(fake_time, lambda bus, m: stop.set())       # Stop pressed while the dialog is up
    with pytest.raises(wayland.PortalDenied, match="Stopped while waiting"):
        bus.request(RD, "Start", "osa{sv}", ("/s", ""), {}, stop=stop, timeout=300)
    path = bus.matches[0]["path"]
    assert bus.calls[-1] == ("org.freedesktop.portal.Request", "Close", "", (), {"path": path})
    assert bus.unmatched == [bus.matches[0]] and bus.router.exited == [bus.matches[0]]


def test_stop_wins_even_if_closing_the_dialog_fails(fake_time):
    stop = threading.Event()
    stop.set()
    bus = request_bus(fake_time)
    real_call = bus.call

    def call(interface, method, *a, **kw):
        if method == "Close":
            raise wayland.PortalError("no such request")
        return real_call(interface, method, *a, **kw)

    bus.call = call
    with pytest.raises(wayland.PortalDenied):
        bus.request(RD, "Start", "osa{sv}", ("/s", ""), {}, stop=stop)


def test_request_times_out(fake_time):
    bus = request_bus(fake_time)                                    # the desktop never answers
    started = time.monotonic()
    with pytest.raises(wayland.PortalError, match="didn't answer SelectDevices in time") as err:
        bus.request(RD, "SelectDevices", "oa{sv}", ("/s",), {}, timeout=5)
    assert not isinstance(err.value, wayland.PortalDenied)
    assert time.monotonic() - started < 1                          # (simulated time)
    assert 5 <= bus.q.gets * 0.25 <= 5.5                            # it kept waiting in short slices until then
    assert len(bus.unmatched) == 1


def test_waiting_is_called_once_when_the_dialog_takes_more_than_a_second(fake_time):
    seen = []

    def waiting():
        seen.append(fake_time.t)

    bus = request_bus(fake_time)
    real_get = bus.q.get

    def get(timeout=None):
        if bus.q.gets == 12:                                        # about 3 s later: the person clicks Share
            bus.q.put(response(0, {"devices": ("u", 3)}))
        return real_get(timeout)

    bus.q.get = get
    t0 = fake_time.t
    assert bus.request(RD, "Start", "osa{sv}", ("/s", ""), {}, waiting=waiting, timeout=300) == {"devices": 3}
    assert len(seen) == 1 and seen[0] - t0 >= 1.0


def test_waiting_is_not_called_for_a_quick_answer(fake_time):
    seen = []
    bus = request_bus(fake_time, lambda bus, m: bus.q.put(response(0)))
    bus.request(RD, "Start", "osa{sv}", ("/s", ""), {}, waiting=lambda: seen.append(1))
    assert seen == []


def test_a_failing_waiting_callback_does_not_break_the_request(fake_time):
    bus = request_bus(fake_time)
    real_get = bus.q.get

    def get(timeout=None):
        if bus.q.gets == 8:
            bus.q.put(response(0))
        return real_get(timeout)

    bus.q.get = get

    def waiting():
        raise RuntimeError("narration failed")

    assert bus.request(RD, "Start", "osa{sv}", ("/s", ""), {}, waiting=waiting) == {}


def test_a_failing_call_still_removes_the_subscription(fake_time):
    bus = request_bus(fake_time)

    def call(*a, **kw):
        raise wayland.PortalError("org.freedesktop.DBus.Error.ServiceUnknown")

    bus.call = call
    with pytest.raises(wayland.PortalError, match="ServiceUnknown"):
        bus.request(RD, "CreateSession", "a{sv}", (), {})
    assert len(bus.unmatched) == 1 and len(bus.router.exited) == 1


def test_request_with_a_real_queue_and_clock():
    """The same dance with the real time module and a real queue filled from another thread."""
    bus = request_bus(None)
    bus.q = queue.Queue()

    def answer_later():
        time.sleep(0.05)
        bus.q.put(response(0, {"uri": ("s", "file:///x.png")}))

    threading.Thread(target=answer_later, daemon=True).start()
    assert bus.request(wayland.SHOT, "Screenshot", "sa{sv}", ("",), {}, timeout=5) == {"uri": "file:///x.png"}


def test_call_get_property_and_match_with_jeepney_messages():
    pytest.importorskip("jeepney")
    from jeepney import HeaderFields, MessageType

    sent = []

    def reply(kind, fields=None, body=()):
        return SimpleNamespace(header=SimpleNamespace(message_type=kind, fields=fields or {}), body=body)

    def send_and_get_reply(msg, timeout=None):
        sent.append(msg)
        member = msg.header.fields[HeaderFields.member]
        if member == "Get":
            return reply(MessageType.method_return, body=(("u", 2),))
        if member in ("AddMatch", "RemoveMatch"):
            return reply(MessageType.method_return)
        return reply(MessageType.error, {4: "org.freedesktop.DBus.Error.UnknownMethod"}, ("No such method 'Nope'",))

    bus = wayland.PortalBus.__new__(wayland.PortalBus)
    bus.router = SimpleNamespace(send_and_get_reply=send_and_get_reply)
    assert bus.get_property(RD, "version") == 2
    f = sent[0].header.fields
    assert f[HeaderFields.destination] == wayland.PORTAL_NAME
    assert f[HeaderFields.path] == wayland.PORTAL_PATH
    assert f[HeaderFields.interface] == "org.freedesktop.DBus.Properties"
    assert sent[0].body == (RD, "version")
    with pytest.raises(wayland.PortalError,
                       match="RemoteDesktop.Nope: org.freedesktop.DBus.Error.UnknownMethod No such method"):
        bus.call(RD, "Nope")

    rule = bus.match(interface="org.freedesktop.portal.Request", member="Response", path="/r/1")
    add = sent[-1]
    assert add.header.fields[HeaderFields.destination] == "org.freedesktop.DBus"
    assert add.header.fields[HeaderFields.path] == "/org/freedesktop/DBus"
    assert add.header.fields[HeaderFields.member] == "AddMatch"
    text = add.body[0]
    assert "type='signal'" in text and "member='Response'" in text and "path='/r/1'" in text
    assert rule.serialise() == text
    bus.unmatch(rule)
    assert sent[-1].header.fields[HeaderFields.member] == "RemoveMatch" and sent[-1].body == (text,)

    def broken(msg, timeout=None):
        raise ConnectionResetError("the bus went away")

    bus.router.send_and_get_reply = broken
    bus.unmatch(rule)                                # quietly: it's only clean-up


def test_portal_versions(monkeypatch):
    props = {(RD, "version"): 2, (SC, "version"): 5}

    class Bus:
        closed = False

        def get_property(self, iface, name):
            return props[(iface, name)]          # KeyError for the missing Screenshot portal

        def close(self):
            Bus.closed = True

    assert wayland.portal_versions(Bus()) == {"RemoteDesktop": 2, "ScreenCast": 5}
    assert Bus.closed is False                    # a bus that was passed in stays open
    monkeypatch.setattr(wayland, "PortalBus", Bus)
    assert wayland.portal_versions() == {"RemoteDesktop": 2, "ScreenCast": 5}
    assert Bus.closed is True                     # its own bus is closed


# --------------------------------------------------------------------------- RemoteDesktop.start
class FakePortal:
    """What RemoteDesktop needs from PortalBus, answering like GNOME's portal does."""

    def __init__(self, props=None, start=None):
        self.router = SimpleNamespace(entered=[], exited=[], queues={})

        def filter(rule):
            q = queue.Queue()
            self.router.queues[rule["member"]] = q
            return Handle(self.router, rule, q)

        self.router.filter = filter
        self.props = {(RD, "AvailableDeviceTypes"): 7, (SC, "AvailableCursorModes"): 7} if props is None else props
        self.start_result = start if start is not None else {
            "devices": 3, "restore_token": "new-token",
            "streams": [(44, {"size": (1280, 720), "position": (0, 0), "source_type": 1})]}
        self.fail = {}
        self.requests, self.calls, self.matches, self.unmatched = [], [], [], []
        self.fd = 7
        self._n = itertools.count(1)

    def token(self):
        return f"tok{next(self._n)}"

    def request(self, interface, method, signature, args, options, stop=None, timeout=30.0, waiting=None):
        self.requests.append(SimpleNamespace(interface=interface, method=method, signature=signature, args=args,
                                             options=options, stop=stop, timeout=timeout, waiting=waiting))
        if method in self.fail:
            raise self.fail[method]
        if method == "CreateSession":
            return {"session_handle": SESSION}
        if method == "Start":
            return self.start_result
        return {}

    def req(self, method):
        return next(r for r in self.requests if r.method == method)

    def get_property(self, interface, name):
        if (interface, name) not in self.props:
            raise wayland.PortalError("org.freedesktop.DBus.Error.InvalidArgs")
        return self.props[(interface, name)]

    def call(self, interface, method, signature="", body=(), **kw):
        self.calls.append((interface, method, signature, body, kw))
        if method == "OpenPipeWireRemote":
            return [self.fd]
        return ()

    def match(self, **rule):
        self.matches.append(rule)
        return rule

    def unmatch(self, rule):
        self.unmatched.append(rule)

    def input_calls(self):
        return [c[:4] for c in self.calls if c[1].startswith("Notify")]

    def closed_session(self):
        return ("org.freedesktop.portal.Session", "Close", "", (), {"path": SESSION}) in self.calls


def started(isolated, fake_time, **portal_kw):
    portal = FakePortal(**portal_kw)
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    rd.start()
    return rd, portal


def test_start_asks_for_both_devices_to_be_remembered(isolated, fake_time):
    portal = FakePortal()
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    stop, asking = threading.Event(), (lambda: None)
    rd.start(stop=stop, ask_timeout=123.0, asking=asking)
    assert [r.method for r in portal.requests] == ["CreateSession", "SelectDevices", "SelectSources", "Start"]
    assert portal.req("CreateSession").options == {"session_handle_token": ("s", "tok1")}
    sel = portal.req("SelectDevices")
    assert (sel.interface, sel.signature, sel.args) == (RD, "oa{sv}", (SESSION,))
    assert sel.options == {"types": ("u", 3), "persist_mode": ("u", 2)}           # no restore_token yet
    src = portal.req("SelectSources")
    assert (src.interface, src.args) == (SC, (SESSION,))
    assert src.options == {"types": ("u", 1), "multiple": ("b", False), "cursor_mode": ("u", 1)}   # cursor hidden
    st = portal.req("Start")
    assert (st.interface, st.signature, st.args, st.options) == (RD, "osa{sv}", (SESSION, ""), {})
    assert st.stop is stop and st.timeout == 123.0 and st.waiting is asking
    assert rd.active and rd.session == SESSION
    rd.close()


def test_start_keeps_the_new_token_privately(isolated, fake_time):
    rd, _ = started(isolated, fake_time)
    assert json.loads(isolated.read_text()) == {"restore_token": "new-token"}
    assert stat.S_IMODE(isolated.stat().st_mode) == 0o600
    assert wayland.will_ask(isolated) is False
    rd.close()


def test_start_restores_a_saved_answer(isolated, fake_time):
    isolated.parent.mkdir(parents=True)
    isolated.write_text('{"restore_token": "old-token"}')
    rd, portal = started(isolated, fake_time)
    assert portal.req("SelectDevices").options == {"types": ("u", 3), "persist_mode": ("u", 2),
                                                   "restore_token": ("s", "old-token")}
    assert json.loads(isolated.read_text()) == {"restore_token": "new-token"}     # tokens are single use
    rd.close()


def test_an_empty_saved_token_is_not_sent(isolated, fake_time):
    isolated.parent.mkdir(parents=True)
    isolated.write_text('{"restore_token": ""}')
    rd, portal = started(isolated, fake_time)
    assert "restore_token" not in portal.req("SelectDevices").options
    rd.close()


def test_no_token_in_the_answer_means_ask_next_time(isolated, fake_time):
    isolated.parent.mkdir(parents=True)
    isolated.write_text('{"restore_token": "old-token"}')
    rd, _ = started(isolated, fake_time, start={"devices": 3, "streams": [(44, {"size": (800, 600)})]})
    assert json.loads(isolated.read_text()) == {"restore_token": ""}
    assert wayland.will_ask(isolated) is True
    rd.close()


@pytest.mark.parametrize("available,types", [(7, 3), (3, 3), (2, 2), (0, 3), (None, 3)])
def test_device_types_follow_what_the_desktop_offers(available, types, isolated, fake_time):
    props = {(SC, "AvailableCursorModes"): 7}
    if available is not None:
        props[(RD, "AvailableDeviceTypes")] = available
    rd, portal = started(isolated, fake_time, props=props)
    assert portal.req("SelectDevices").options["types"] == ("u", types)
    rd.close()


@pytest.mark.parametrize("modes,cursor", [(7, 1), (1, 1), (2, 2), (6, 2), (4, None), (0, None), ("missing", 1)])
def test_cursor_mode_hidden_when_available(modes, cursor, isolated, fake_time):
    props = {(RD, "AvailableDeviceTypes"): 7}
    if modes != "missing":
        props[(SC, "AvailableCursorModes")] = modes
    rd, portal = started(isolated, fake_time, props=props)
    options = portal.req("SelectSources").options
    if cursor is None:
        assert "cursor_mode" not in options
    else:
        assert options["cursor_mode"] == ("u", cursor)
    rd.close()


def test_size_position_and_node_come_from_the_first_stream(isolated, fake_time):
    rd, _ = started(isolated, fake_time, start={
        "devices": 3, "restore_token": "t",
        "streams": [(51, {"size": (1440, 900), "position": (1920, 0)}), (52, {"size": (800, 600)})]})
    assert (rd.node, rd.size, rd.position, rd.devices) == (51, (1440, 900), (1920, 0), 3)
    assert wayland.PortalBackend(rd).size() == (1440, 900)
    rd.close()


def test_a_stream_without_size_keeps_the_defaults(isolated, fake_time):
    rd, _ = started(isolated, fake_time, start={"devices": 3, "streams": [(51, {})]})
    assert (rd.node, rd.size, rd.position) == (51, None, (0, 0))
    rd.close()


def test_warm_up_events_after_start(isolated, fake_time):
    rd, portal = started(isolated, fake_time)
    assert portal.input_calls() == [
        (RD, "NotifyPointerMotion", "oa{sv}dd", (SESSION, {}, 0.0, 0.0)),
        (RD, "NotifyKeyboardKeysym", "oa{sv}iu", (SESSION, {}, 0xFFE1, 1)),       # a lone Shift tap
        (RD, "NotifyKeyboardKeysym", "oa{sv}iu", (SESSION, {}, 0xFFE1, 0)),
    ]
    assert fake_time.slept and fake_time.slept[-1] > 0                          # then a short pause
    rd.close()


def test_warm_up_without_a_keyboard_presses_no_key(isolated, fake_time):
    rd, portal = started(isolated, fake_time, start={"devices": 2, "restore_token": "t", "streams": [(44, {})]})
    assert [c[1] for c in portal.input_calls()] == ["NotifyPointerMotion"]
    rd.close()


def test_no_pointer_clears_the_token_closes_and_explains(isolated, fake_time):
    isolated.parent.mkdir(parents=True)
    isolated.write_text('{"restore_token": "old-token"}')
    portal = FakePortal(start={"devices": 1, "restore_token": "screen-only", "streams": [(44, {})]})
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(wayland.PortalDenied, match="Allow Remote Interaction"):
        rd.start()
    assert json.loads(isolated.read_text()) == {"restore_token": ""}            # don't restore a screen-only answer
    assert wayland.will_ask(isolated) is True
    assert portal.closed_session() and rd.session is None and not rd.active
    assert portal.unmatched == [portal.matches[0]]                              # the Session.Closed subscription
    assert portal.input_calls() == []                                           # no warm-up


def test_no_stream_is_an_error(isolated, fake_time):
    portal = FakePortal(start={"devices": 3, "restore_token": "t", "streams": []})
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(wayland.PortalError, match="shared no screen") as err:
        rd.start()
    assert not isinstance(err.value, wayland.PortalDenied)
    assert portal.closed_session() and rd.session is None


def test_saying_no_in_the_dialog_gets_a_helpful_message(isolated, fake_time):
    portal = FakePortal()
    original = wayland.PortalDenied("You chose not to allow it in the desktop's dialog.")
    portal.fail["Start"] = original
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(wayland.PortalDenied, match="I can only help after your desktop lets me see the screen") as err:
        rd.start(stop=threading.Event())
    assert "Allow Remote Interaction" in str(err.value) and err.value.__cause__ is original
    assert portal.closed_session() and portal.unmatched and portal.router.exited
    assert not isolated.exists()                                                # nothing saved


def test_stop_during_the_dialog_keeps_the_stop_message(isolated, fake_time):
    portal = FakePortal()
    stop = threading.Event()
    stop.set()
    portal.fail["Start"] = wayland.PortalDenied("Stopped while waiting for the desktop's permission dialog.")
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(wayland.PortalDenied, match="^Stopped while waiting"):
        rd.start(stop=stop)
    assert portal.closed_session()


@pytest.mark.parametrize("err", [wayland.PortalError("The desktop refused Start (code 2)."), KeyboardInterrupt()])
def test_other_failures_close_the_session_and_pass_through(err, isolated, fake_time):
    portal = FakePortal()
    portal.fail["Start"] = err
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(type(err)) as raised:
        rd.start()
    assert raised.value is err
    assert portal.closed_session() and rd.session is None


def test_failing_before_the_session_exists_closes_nothing(isolated, fake_time):
    portal = FakePortal()
    portal.fail["CreateSession"] = wayland.PortalError("The desktop didn't answer CreateSession in time.")
    rd = wayland.RemoteDesktop(bus=portal, state_file=isolated)
    with pytest.raises(wayland.PortalError):
        rd.start()
    assert portal.calls == [] and portal.matches == []


def test_input_calls_after_start(isolated, fake_time):
    rd, portal = started(isolated, fake_time)
    portal.calls.clear()
    rd.move(10.6, 20)
    rd.button("middle", True)
    rd.button("right", False)
    rd.scroll_steps(1, -2)
    rd.key(0x1008FF13, False)
    assert portal.input_calls() == [
        (RD, "NotifyPointerMotionAbsolute", "oa{sv}udd", (SESSION, {}, 44, 10.6, 20.0)),
        (RD, "NotifyPointerButton", "oa{sv}iu", (SESSION, {}, 0x112, 1)),
        (RD, "NotifyPointerButton", "oa{sv}iu", (SESSION, {}, 0x111, 0)),
        (RD, "NotifyPointerAxisDiscrete", "oa{sv}ui", (SESSION, {}, 1, -2)),
        (RD, "NotifyKeyboardKeysym", "oa{sv}iu", (SESSION, {}, 0x1008FF13, 0)),
    ]
    rd.close()


def test_pipewire_fd(isolated, fake_time):
    rd, portal = started(isolated, fake_time)
    portal.fd = SimpleNamespace(to_raw_fd=lambda: 11)          # jeepney's FileDescriptor
    assert rd.pipewire_fd() == 11
    assert portal.calls[-1][:4] == (SC, "OpenPipeWireRemote", "oa{sv}", (SESSION, {}))
    portal.fd = 12
    assert rd.pipewire_fd() == 12
    rd.close()


def test_stopping_the_sharing_on_the_desktop_stops_leanwatch(isolated, fake_time):
    rd, portal = started(isolated, fake_time)
    assert portal.matches[0] == {"interface": "org.freedesktop.portal.Session", "member": "Closed", "path": SESSION}
    portal.router.queues["Closed"].put(SimpleNamespace(body=({},)))       # GNOME's top bar: "Stop sharing"
    assert rd.closed.wait(3)
    assert not rd.active
    with pytest.raises(wayland.SharingStopped, match="stopped the screen sharing") as err:
        rd.move(1, 1)
    assert isinstance(err.value, EmergencyStop)                           # the brains stop on EmergencyStop
    rd.close()


def test_after_close_input_is_refused_and_close_is_idempotent(isolated, fake_time):
    rd, portal = started(isolated, fake_time)
    rd.close()
    rd.close()
    assert sum(1 for c in portal.calls if c[1] == "Close") == 1
    assert portal.router.exited == [portal.matches[0]] and portal.unmatched == [portal.matches[0]]
    with pytest.raises(wayland.PortalError, match="isn't running") as err:
        rd.key(0x61, True)
    assert not isinstance(err.value, wayland.SharingStopped)


# --------------------------------------------------------------------------- PipeWireFrames (fake helper)
class PipeRemote:
    """What PipeWireFrames needs from RemoteDesktop. `node` carries the fake helper's mode."""

    def __init__(self, mode):
        self.node = mode
        self.fd, self.write_end = os.pipe()

    def pipewire_fd(self):
        return self.fd


@pytest.fixture
def helper(monkeypatch, tmp_path):
    log = tmp_path / "helper.log"
    monkeypatch.setenv("LEANWATCH_FAKE_LOG", str(log))
    monkeypatch.setattr(wayland, "_HELPER", FAKE_HELPER)
    made = []

    def start(mode):
        remote = PipeRemote(mode)
        try:
            frames = wayland.PipeWireFrames(remote, sys.executable)
        finally:
            os.close(remote.write_end)
        made.append(frames)
        return frames, remote

    start.log = lambda: log.read_text().splitlines() if log.exists() else []
    yield start
    for frames in made:
        frames.close()


def expected_frame():
    img = Image.new("RGB", (3, 2))
    img.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (128, 128, 128)])
    return img


def test_frames_with_padded_rows(helper):
    frames, remote = helper("frames")
    with pytest.raises(OSError):
        os.fstat(remote.fd)                               # LeanWatch closed its copy; the helper has its own
    img = frames.grab()
    assert img.mode == "RGB" and img.size == (3, 2)
    assert img.tobytes() == expected_frame().tobytes()      # the padding bytes are not pixels
    img2 = frames.grab(since=12.5)
    assert img2.tobytes() == img.tobytes()
    frames.grab(since=99.25, wait=0.25)
    frames.close()
    assert frames.proc.returncode == 0
    assert helper.log() == ["fd-ok", "grab 0.000 -1.0000", "grab 0.500 12.5000", "grab 0.250 99.2500", "quit", "bye"]


def test_no_frame_yet_nudges_once_and_asks_again(helper):
    frames, _ = helper("none-once")
    nudges = []
    before = time.monotonic()
    img = frames.grab(nudge=lambda: nudges.append(time.monotonic()))
    assert len(nudges) == 1 and img.size == (3, 2)
    second = helper.log()[2].split()
    assert second[:2] == ["grab", "3.000"]                    # wait for the frame the nudge causes
    assert before - 0.3 <= float(second[2]) <= time.monotonic()
    frames.grab(nudge=lambda: nudges.append(1))               # a frame is there now: no more nudging
    assert len(nudges) == 1


def test_still_no_frame_after_the_nudge_is_an_error(helper):
    frames, _ = helper("none")
    nudges = []
    with pytest.raises(wayland.PortalError, match=r"sent no picture \(none no frame yet\)"):
        frames.grab(nudge=lambda: nudges.append(1))
    assert nudges == [1]


def test_no_frame_and_no_nudge_is_an_error_at_once(helper):
    frames, _ = helper("none")
    with pytest.raises(wayland.PortalError, match="no picture"):
        frames.grab()
    assert [line for line in helper.log() if line.startswith("grab")] == ["grab 0.000 -1.0000"]


def test_helper_that_exited(helper):
    frames, _ = helper("exit")
    frames.proc.wait(5)
    with pytest.raises(wayland.PortalError, match="screen reader stopped"):
        frames.grab()


def test_helper_exiting_without_an_answer(helper):
    frames, _ = helper("silent-exit")
    started = time.monotonic()
    with pytest.raises(wayland.PortalError, match=r"no picture \(no answer\)"):
        frames.grab()
    assert time.monotonic() - started < 5                     # end of file, not the 10 s timeout


def test_helper_dying_in_the_middle_of_a_frame(helper):
    frames, _ = helper("broken")
    with pytest.raises(wayland.PortalError, match="broken picture"):
        frames.grab()


def test_a_late_answer_is_not_taken_for_the_next_picture(helper):
    frames, _ = helper("slow-first")
    read_line = frames._read_line
    frames._read_line = lambda timeout: read_line(min(timeout, 0.05))  # the helper stalls longer than LeanWatch waits
    with pytest.raises(wayland.PortalError, match="no answer"):
        frames.grab(since=1.0)
    frames._read_line = read_line
    try:
        img = frames.grab(since=2.0)
    except wayland.PortalError:
        return                                                          # refusing is fine too
    assert img.getpixel((0, 0))[0] == 2, "got the late picture asked for by the previous grab"


def test_helper_that_never_gets_ready(helper):
    with pytest.raises(wayland.PortalError, match=r"didn't start \(error no stream\)"):
        helper("not-ready")


def test_close_after_the_helper_is_gone_is_quiet(helper):
    frames, _ = helper("exit")
    frames.proc.wait(5)
    frames.close()
    frames.close()


def test_helper_environment_drops_this_pythons_settings(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("PYTHONHOME", "/elsewhere")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{sys.prefix}/lib:/opt/leanwatch-test/lib::{sys.prefix}/lib/extra")
    env = wayland._clean_env()
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["LD_LIBRARY_PATH"] == "/opt/leanwatch-test/lib"
    assert env["HOME"] == os.environ["HOME"]


# --------------------------------------------------------------------------- finding GStreamer
class OsProxy:
    def __init__(self, exists):
        self.path = SimpleNamespace(exists=exists)

    def __getattr__(self, name):
        return getattr(os, name)


def test_find_gst_python_tries_the_version_the_bindings_were_built_for(monkeypatch):
    monkeypatch.setattr(wayland, "_gst_python", [])
    monkeypatch.setattr(wayland, "glob", SimpleNamespace(glob=lambda pattern: [
        "/usr/lib/python3/dist-packages/gi/_gi.cpython-312-x86_64-linux-gnu.so"]))
    monkeypatch.setattr(wayland, "os", OsProxy(lambda p: p in ("/usr/bin/python3", "/usr/bin/python3.12")))
    monkeypatch.setenv("PYTHONPATH", "/anaconda/lib")
    probes = []

    def run(cmd, env=None, capture_output=False, timeout=None):
        probes.append(cmd[0])
        assert cmd[1:3] == ["-s", "-c"] and "pipewiresrc" in cmd[3]
        assert "PYTHONPATH" not in env
        return SimpleNamespace(returncode=0 if cmd[0] == "/usr/bin/python3.12" else 3)

    monkeypatch.setattr(wayland, "subprocess", SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    assert wayland.find_gst_python() == "/usr/bin/python3.12"
    assert probes == ["/usr/bin/python3", "/usr/bin/python3.12"]
    assert wayland.find_gst_python() == "/usr/bin/python3.12"        # remembered: no second probe
    assert len(probes) == 2


def test_find_gst_python_none_found_is_remembered(monkeypatch):
    monkeypatch.setattr(wayland, "_gst_python", [])
    monkeypatch.setattr(wayland, "glob", SimpleNamespace(glob=lambda pattern: []))
    monkeypatch.setattr(wayland, "os", OsProxy(lambda p: True))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[0])
        raise subprocess.TimeoutExpired(cmd, 20)

    monkeypatch.setattr(wayland, "subprocess", SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    assert wayland.find_gst_python() is None
    assert wayland.find_gst_python() is None
    assert calls == ["/usr/bin/python3"]


# --------------------------------------------------------------------------- Screenshot portal
def test_portal_screenshot_reads_and_deletes_the_file(tmp_path):
    shot = tmp_path / "Pictures" / "Screenshot from now.png"
    shot.parent.mkdir()
    Image.new("RGBA", (40, 30), (10, 20, 30, 255)).save(shot)
    asked = []

    class Bus:
        def request(self, interface, method, signature, args, options, stop=None, timeout=30.0, waiting=None):
            asked.append((interface, method, signature, args, options, stop, timeout))
            return {"uri": "file://" + quote(str(shot))}

    stop = threading.Event()
    img = wayland.portal_screenshot(Bus(), stop=stop)
    assert img.mode == "RGB" and img.size == (40, 30) and img.getpixel((0, 0)) == (10, 20, 30)
    assert not shot.exists()                                  # it was only for LeanWatch
    assert asked == [(wayland.SHOT, "Screenshot", "sa{sv}", ("",), {"interactive": ("b", False)}, stop, 120)]


# --------------------------------------------------------------------------- WaylandScreen
@pytest.fixture
def fake_remote(monkeypatch):
    made = []

    class FakeRemoteDesktop:
        fail = None

        def __init__(self):
            self.closed = threading.Event()
            self.size = (1280, 720)
            self.position = (0, 0)
            self.bus = "the session's bus"
            self.started = None
            self.close_calls = 0
            self.moves = []
            made.append(self)

        def start(self, stop=None, asking=None):
            self.started = (stop, asking)
            if FakeRemoteDesktop.fail:
                raise FakeRemoteDesktop.fail

        def move(self, x, y):
            self.moves.append((x, y))

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(wayland, "RemoteDesktop", FakeRemoteDesktop)
    FakeRemoteDesktop.made = made
    return FakeRemoteDesktop


def test_screen_without_gstreamer_uses_the_screenshot_portal(fake_remote, monkeypatch):
    shots = []
    monkeypatch.setattr(wayland, "find_gst_python", lambda: None)
    # one monitor: the picture has the shared screen's shape (16:9 like the fake 1280x720 stream)
    monkeypatch.setattr(wayland, "portal_screenshot", lambda bus=None, stop=None: shots.append(bus) or Image.new("RGB", (16, 9)))
    stop, asking = threading.Event(), (lambda: None)
    screen = wayland.WaylandScreen(stop=stop, asking=asking)
    remote = fake_remote.made[0]
    assert remote.started == (stop, asking)
    assert screen.frames is None and "GStreamer" in screen.note and "flashes" in screen.note
    assert screen.backend.size() == (1280, 720)
    assert screen.grab(since=5.0).size == (16, 9)
    assert shots == ["the session's bus"]
    screen.close()
    assert remote.close_calls == 1


def test_screenshot_fallback_shows_only_the_shared_monitor(fake_remote, monkeypatch):
    monkeypatch.setattr(wayland, "find_gst_python", lambda: None)
    whole = Image.new("RGB", (1920 + 1280, 1080), (90, 90, 90))      # a 1920x1080 monitor, then 1280x720
    whole.paste((255, 0, 0), (1920, 0, 1920 + 1280, 720))             # the shared one, at x = 1920
    monkeypatch.setattr(wayland, "portal_screenshot", lambda bus=None, stop=None: whole.copy())
    screen = wayland.WaylandScreen()
    screen.remote.position = (1920, 0)
    img = screen.grab()
    assert abs(img.width / img.height - 1280 / 720) < 0.01
    assert img.getpixel((0, 0)) == (255, 0, 0) and img.getpixel((img.width - 1, img.height - 1)) == (255, 0, 0)


def test_screen_with_a_failing_stream_reader_falls_back(fake_remote, monkeypatch):
    monkeypatch.setattr(wayland, "find_gst_python", lambda: "/usr/bin/python3")

    def broken(remote, python):
        raise wayland.PortalError("The screen reader didn't start (no answer).")

    monkeypatch.setattr(wayland, "PipeWireFrames", broken)
    screen = wayland.WaylandScreen()
    assert screen.frames is None
    assert screen.note == ("Reading the screen stream failed (The screen reader didn't start (no answer).); "
                           "using screenshots instead.")


def test_screen_reads_frames_and_nudges_through_the_backend(fake_remote, monkeypatch, fake_time):
    monkeypatch.setattr(wayland, "find_gst_python", lambda: "/usr/bin/python3.12")
    made = []

    class Frames:
        def __init__(self, remote, python):
            self.remote, self.python, self.closed = remote, python, 0
            self.asked = []
            made.append(self)

        def grab(self, since=None, wait=0.5, nudge=None):
            self.asked.append(since)
            nudge()
            return Image.new("RGB", (4, 4))

        def close(self):
            self.closed += 1

    monkeypatch.setattr(wayland, "PipeWireFrames", Frames)
    monkeypatch.setattr(wayland, "portal_screenshot", lambda *a, **k: pytest.fail("no screenshots with a stream"))
    screen = wayland.WaylandScreen()
    frames = made[0]
    assert frames.remote is screen.remote and frames.python == "/usr/bin/python3.12" and screen.note == ""
    screen.grab(since=42.0)
    assert frames.asked == [42.0]
    assert screen.remote.moves == [(641, 360), (640, 360)]       # the backend's nudge
    screen.close()
    assert frames.closed == 1 and screen.frames is None and screen.remote.close_calls == 1


def test_grab_after_the_sharing_stopped_raises_sharing_stopped(fake_remote, monkeypatch):
    monkeypatch.setattr(wayland, "find_gst_python", lambda: None)
    monkeypatch.setattr(wayland, "portal_screenshot", lambda *a, **k: pytest.fail("must not take a screenshot"))
    screen = wayland.WaylandScreen()
    screen.remote.closed.set()
    with pytest.raises(wayland.SharingStopped) as err:
        screen.grab()
    assert isinstance(err.value, EmergencyStop)


def test_screen_that_is_refused_does_not_look_for_gstreamer(fake_remote, monkeypatch):
    fake_remote.fail = wayland.PortalDenied("no")
    monkeypatch.setattr(wayland, "find_gst_python", lambda: pytest.fail("not needed"))
    with pytest.raises(wayland.PortalDenied):
        wayland.WaylandScreen()


# --------------------------------------------------------------------------- ScreenCapture(grab=...)
class FakeShield:
    def __init__(self):
        self.inside = False
        self.entered = []

    @contextlib.contextmanager
    def cloak(self):
        self.entered.append(time.monotonic())
        self.inside = True
        try:
            yield
        finally:
            self.inside = False


def test_capture_uses_grab_instead_of_mss(monkeypatch):
    monkeypatch.setitem(sys.modules, "mss", None)          # importing mss would fail
    g = Geometry.for_screen(640, 360)
    calls = []
    cap = ScreenCapture(g, grab=lambda since: calls.append(since) or Image.new("RGB", (1280, 720), "red"))
    shot = cap.screenshot()
    assert calls == [None] and shot.size == (g.shot_w, g.shot_h) and cap.last_image is shot


def test_capture_grabs_inside_the_cloak_with_the_time_it_started(monkeypatch):
    monkeypatch.setitem(sys.modules, "mss", None)
    shield = FakeShield()
    seen = []

    def grab(since):
        seen.append((since, shield.inside))
        return Image.new("RGB", (1280, 720), "blue")

    before = time.monotonic()
    cap = ScreenCapture(Geometry.for_screen(1280, 720), shield=shield, grab=grab)
    cap.screenshot()
    (since, inside), = seen
    assert inside is True                                   # LeanWatch's windows are hidden meanwhile
    assert before <= since <= shield.entered[0]             # taken right before hiding them


def test_zoom_also_uses_grab(monkeypatch):
    monkeypatch.setitem(sys.modules, "mss", None)
    raw = Image.new("RGB", (2560, 1440), "black")
    raw.paste((255, 0, 0), (0, 0, 1280, 720))
    shield = FakeShield()
    cap = ScreenCapture(Geometry.for_screen(1280, 720), shield=shield, grab=lambda since: raw)
    g = cap.geometry
    z = cap.zoom([0, 0, g.shot_w / 4, g.shot_h / 4])
    assert z.getpixel((z.width // 2, z.height // 2)) == (255, 0, 0)
    assert len(shield.entered) == 1


def test_grab_errors_reach_the_caller(monkeypatch):
    monkeypatch.setitem(sys.modules, "mss", None)

    def grab(since):
        raise wayland.SharingStopped("You stopped the screen sharing, so I stopped.")

    cap = ScreenCapture(Geometry.for_screen(1280, 720), shield=FakeShield(), grab=grab)
    with pytest.raises(wayland.SharingStopped):
        cap.screenshot()
