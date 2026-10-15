"""The desktop's permission dialog on Wayland: LeanWatch says what to do while it is up, Stop closes
it, and a "no" there gets a plain explanation. Fake D-Bus: nothing here talks to a real desktop."""

import contextlib
import itertools
import queue
import threading
import types

import pytest

from leanwatch import wayland as wl


class FakeBus(wl.PortalBus):
    """PortalBus without a connection: every request is answered after `delay` seconds."""

    def __init__(self, delay=0.0, code=0, results=None):
        self.sender = "1_1"
        self._tokens = itertools.count(1)
        self.delay, self.code, self.results = delay, code, results or {}
        self.calls = []
        self.q = queue.Queue()
        self.router = types.SimpleNamespace(filter=lambda rule: contextlib.nullcontext(self.q))

    def match(self, **rule):
        return rule

    def unmatch(self, rule):
        pass

    def call(self, interface, method, signature="", body=(), **kw):
        self.calls.append(method)
        if method != "Close":
            reply = types.SimpleNamespace(body=(self.code, self.results))
            threading.Timer(self.delay, self.q.put, args=(reply,)).start()
        return ()


def test_waiting_is_called_once_when_the_dialog_stays_up():
    bus, seen = FakeBus(delay=1.4, results={"devices": ("u", 3)}), []
    assert bus.request(wl.RD, "Start", "osa{sv}", ("/s", ""), {}, waiting=lambda: seen.append(1)) == {"devices": 3}
    assert seen == [1]


def test_no_waiting_message_when_the_desktop_answers_at_once():
    seen = []
    FakeBus(delay=0.05).request(wl.RD, "Start", "osa{sv}", ("/s", ""), {}, waiting=lambda: seen.append(1))
    assert seen == []


def test_stop_closes_the_dialog():
    bus, stop = FakeBus(delay=30), threading.Event()
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(wl.PortalDenied):
        bus.request(wl.RD, "Start", "osa{sv}", ("/s", ""), {}, stop=stop)
    assert "Close" in bus.calls


def test_cancel_in_the_dialog_is_explained(tmp_path, monkeypatch):
    class Bus(FakeBus):
        def request(self, interface, method, *a, **k):
            if method == "CreateSession":
                return {"session_handle": "/org/freedesktop/portal/desktop/session/1_1/x"}
            if method == "Start":
                raise wl.PortalDenied("You chose not to allow it in the desktop's dialog.")
            return {}

        def get_property(self, interface, name):
            return 3 if name == "AvailableDeviceTypes" else 7

    rd = wl.RemoteDesktop(bus=Bus(), state_file=tmp_path / "wayland.json")
    monkeypatch.setattr(rd, "_watch_closed", lambda: None)
    with pytest.raises(wl.PortalDenied, match="Allow Remote Interaction"):
        rd.start()
    assert rd.session is None          # the half-open session was closed
