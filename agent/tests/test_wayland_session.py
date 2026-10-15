"""The Wayland paths outside wayland.py: the server's Session, the command line, --check and the
"open the box" message. WaylandScreen is a fake here; no D-Bus, no real ~/.leanwatch."""

import asyncio
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import leanwatch.server as server
from leanwatch import brains, doctor, gnome_shortcut, wayland
from leanwatch.config import Config
from leanwatch.controller import DryRunBackend
from leanwatch.hooks import RunResult
from leanwatch.native import DEFAULT_HOTKEY


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/nonexistent/leanwatch-unit-tests")
    monkeypatch.setattr(wayland, "STATE_FILE", home / ".leanwatch" / "wayland.json")

    def no_bus(self):
        raise AssertionError("a unit test tried to open the real session bus")

    def no_gsettings(*args):
        raise AssertionError("a unit test tried to run the real gsettings")

    monkeypatch.setattr(wayland.PortalBus, "__init__", no_bus)
    monkeypatch.setattr(gnome_shortcut, "_gsettings", no_gsettings)
    return home


class Screen:
    """Stands in for wayland.WaylandScreen. `script(screen)` runs while it "opens" (the dialog)."""

    def __init__(self, script=None, note="", size=(1280, 720)):
        self.script, self.note_text, self.size = script, note, size
        self.made = []

    def __call__(self, stop=None, asking=None):
        outer = self

        class Made:
            def __init__(self):
                self.stop, self.asking = stop, asking
                self.backend = DryRunBackend(outer.size)       # stands in for PortalBackend; records input
                self.note = outer.note_text
                self.grabs = []
                self.closed = 0

            def grab(self, since=None):
                self.grabs.append(since)
                return Image.new("RGB", outer.size, "white")

            def close(self):
                self.closed += 1

        made = Made()
        self.made.append(made)
        if self.script:
            self.script(made)
        return made


class Brain:
    """A brain that takes one screenshot (through the controller, like the real ones) and finishes."""

    def __init__(self, result=None, error=None):
        self.result = result or RunResult(True, "Done it.")
        self.error = error
        self.controllers = []

    def make(self, cfg, controller, hooks, stop, local_model=None):
        self.controllers.append(controller)
        brain = self

        class Run:
            def run(self, text):
                controller.capture.screenshot()
                if brain.error:
                    raise brain.error
                return brain.result

        return Run()


def session(monkeypatch, screen, brain, *, will_ask=False, dry_run=False, available=True):
    monkeypatch.setattr(wayland, "wanted", lambda: True)
    monkeypatch.setattr(wayland, "available", lambda: available)
    monkeypatch.setattr(wayland, "will_ask", lambda state_file=None: will_ask)
    monkeypatch.setattr(wayland, "WaylandScreen", screen)
    monkeypatch.setattr(server, "make_brain", brain.make)
    cfg = Config()
    cfg.dry_run = dry_run
    s = server.Session(cfg, "tok")
    s.local_model = object()          # don't load a model
    events = []
    s.listeners.append(events.append)
    return s, events


def narrations(events, kind=None):
    return [e["text"] for e in events if e["type"] == "narration" and (kind is None or e["kind"] == kind)]


def finished(events):
    (done,) = [e for e in events if e["type"] == "run_finished"]
    return done


# --------------------------------------------------------------------------- Session._worker on Wayland
def test_first_time_the_person_is_told_what_to_click(monkeypatch):
    screen, brain = Screen(), Brain()
    s, events = session(monkeypatch, screen, brain, will_ask=True)
    s._worker("open the files app")
    assert narrations(events, "say")[0] == wayland.ASK_TEXT
    assert finished(events)["ok"] is True and finished(events)["result"] == "Done it."
    made = screen.made[0]
    assert made.stop is s.stop_event and callable(made.asking)
    assert made.closed == 1                                     # the sharing ends with the task
    assert not s.running


def test_the_ask_text_is_said_once_even_when_the_dialog_also_shows(monkeypatch):
    screen = Screen(script=lambda made: (made.asking(), made.asking()))   # the portal's "still waiting" callback
    s, events = session(monkeypatch, screen, Brain(), will_ask=True)
    s._worker("x")
    assert narrations(events).count(wayland.ASK_TEXT) == 1


def test_a_remembered_answer_says_nothing(monkeypatch):
    screen = Screen()
    s, events = session(monkeypatch, screen, Brain(), will_ask=False)
    s._worker("x")
    assert wayland.ASK_TEXT not in narrations(events)


def test_a_withdrawn_permission_still_explains_the_dialog(monkeypatch):
    """Nothing to say up front (a token was saved), but the desktop shows its dialog anyway."""
    screen = Screen(script=lambda made: made.asking())
    s, events = session(monkeypatch, screen, Brain(), will_ask=False)
    s._worker("x")
    assert narrations(events, "say").count(wayland.ASK_TEXT) == 1


def test_the_screen_and_its_size_come_from_the_portal(monkeypatch):
    screen, brain = Screen(note="GStreamer's PipeWire plugin wasn't found ...", size=(1440, 900)), Brain()
    s, events = session(monkeypatch, screen, brain)
    s._worker("x")
    made = screen.made[0]
    assert "GStreamer's PipeWire plugin wasn't found ..." in narrations(events, "thinking")
    assert s.screen_size == (1440, 900)
    assert any(e["type"] == "status" and e["screen"] == {"width": 1440, "height": 900} for e in events)
    c = brain.controllers[0]
    assert c.backend is made.backend                            # input goes through the portal
    assert c.capture.grab == made.grab and made.grabs == [None]   # pictures too (no shield: no `since`)
    assert (c.g.screen_w, c.g.screen_h) == (1440, 900)


def test_dry_run_on_wayland_moves_nothing_but_still_sees(monkeypatch):
    screen, brain = Screen(), Brain()
    s, events = session(monkeypatch, screen, brain, dry_run=True)
    s._worker("x")
    c = brain.controllers[0]
    assert isinstance(c.backend, DryRunBackend) and c.backend is not screen.made[0].backend
    assert c.backend.size() == (1280, 720)
    assert c.capture.grab == screen.made[0].grab


def test_stop_during_the_dialog(monkeypatch):
    s_holder = {}

    def press_stop_then_refuse(made):
        s_holder["s"].stop()                                    # Stop in LeanWatch while the dialog is up
        raise wayland.PortalDenied("Stopped while waiting for the desktop's permission dialog.")

    screen, brain = Screen(script=press_stop_then_refuse), Brain()
    s, events = session(monkeypatch, screen, brain, will_ask=True)
    s_holder["s"] = s
    s._worker("x")
    done = finished(events)
    assert done["ok"] is False and done["message"] == "Stopped. You are in control."
    assert brain.controllers == [] and not s.running


def test_a_refusal_reaches_the_person(monkeypatch):
    msg = ("I can only help after your desktop lets me see the screen and use the mouse. "
           "Try again and click Share, with “Allow Remote Interaction” turned on.")

    def refuse(made):
        raise wayland.PortalDenied(msg)

    s, events = session(monkeypatch, Screen(script=refuse), Brain())
    s._worker("x")
    done = finished(events)
    assert done["ok"] is False and done["message"] == f"Could not run the task: {msg}"


def test_sharing_stopped_from_the_desktop_reaches_the_person(monkeypatch):
    brain = Brain(error=wayland.SharingStopped("You stopped the screen sharing, so I stopped."))
    screen = Screen()
    s, events = session(monkeypatch, screen, brain)
    s._worker("x")
    assert finished(events)["message"] == "Could not run the task: You stopped the screen sharing, so I stopped."
    assert screen.made[0].closed == 1


def test_the_screen_is_closed_when_the_brain_fails(monkeypatch):
    screen = Screen()
    s, events = session(monkeypatch, screen, Brain(error=RuntimeError("model crashed")))
    s._worker("x")
    assert screen.made[0].closed == 1
    assert finished(events)["message"] == "Could not run the task: model crashed"


def test_without_jeepney_the_fix_is_explained(monkeypatch):
    screen = Screen()
    s, events = session(monkeypatch, screen, Brain(), available=False)
    s._worker("x")
    assert screen.made == []
    assert "pip install jeepney" in finished(events)["message"]


def test_x11_sessions_never_open_the_portal(monkeypatch):
    screen, brain = Screen(), Brain()
    s, events = session(monkeypatch, screen, brain, dry_run=True)
    monkeypatch.setattr(wayland, "wanted", lambda: False)
    brain_runs = []
    monkeypatch.setattr(server, "make_brain", lambda cfg, controller, hooks, stop, local_model=None: type(
        "B", (), {"run": lambda self, text: brain_runs.append(controller) or RunResult(True, "ok")})())
    s._worker("x")
    assert screen.made == [] and brain_runs[0].capture.grab is None


def test_through_start_on_the_worker_thread(monkeypatch):
    screen = Screen()
    s, events = session(monkeypatch, screen, Brain(), will_ask=True)
    s.permissions.grant(mouse=True, keyboard=True)
    assert s.start("open the files app") is None
    s.thread.join(5)
    assert finished(events)["ok"] is True and screen.made[0].closed == 1


# --------------------------------------------------------------------------- "open the box"
def test_open_message_asks_the_bubble_to_open_the_box():
    s = server.Session(Config(), "tok")
    events = []
    s.listeners.append(events.append)
    assert s.handle({"type": "open"}) is None
    assert events == [{"type": "open_box"}]


def test_open_api_needs_the_token():
    async def main():
        cfg = Config()
        cfg.port = 8765
        app = server.make_app(cfg, "tok-open")
        events = []
        app[server.SESSION].listeners.append(events.append)
        host = {"Host": "127.0.0.1:8765"}
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/api/open", headers=host)
            assert r.status == 403 and events == []
            r = await client.post("/api/open", data=b"{}", headers={**host, "X-LeanWatch-Token": "tok-open",
                                                                  "Content-Type": "application/json"})
            assert r.status == 200 and await r.json() == {"ok": True}
            r = await client.post("/api/open", headers={**host, "X-LeanWatch-Token": "tok-open"})   # no body
            assert r.status == 200
        assert events == [{"type": "open_box"}, {"type": "open_box"}]

    asyncio.run(main())


# --------------------------------------------------------------------------- python -m leanwatch
def cli(monkeypatch, screen, brain, *, will_ask=True, answer="y", wanted=True):
    from leanwatch import __main__ as main_mod

    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    monkeypatch.setattr(wayland, "wanted", lambda: wanted)
    monkeypatch.setattr(wayland, "will_ask", lambda state_file=None: will_ask)
    monkeypatch.setattr(wayland, "WaylandScreen", screen)
    monkeypatch.setattr(brains, "make_brain", lambda cfg, controller, hooks, stop: brain.make(cfg, controller, hooks, stop))
    return main_mod


def test_cli_on_wayland(monkeypatch, capsys):
    screen, brain = Screen(note="Reading the screen stream failed (x); using screenshots instead."), Brain()
    main_mod = cli(monkeypatch, screen, brain)
    cfg = Config()
    cfg.brain = "claude"
    main_mod.run_cli(cfg, "open the files app")
    out = capsys.readouterr().out
    assert "or by stopping the screen sharing from your desktop" in out and "screen corner" not in out
    assert out.count(wayland.ASK_TEXT) == 1
    assert "Reading the screen stream failed (x)" in out
    assert "✓ Done it." in out
    made = screen.made[0]
    assert made.closed == 1 and isinstance(made.stop, threading.Event)
    c = brain.controllers[0]
    assert c.backend is made.backend and c.capture.grab == made.grab and c.stop is made.stop


def test_cli_says_the_ask_text_once_when_the_dialog_shows(monkeypatch, capsys):
    screen = Screen(script=lambda made: (made.asking(), made.asking()))
    main_mod = cli(monkeypatch, screen, Brain(), will_ask=False)
    main_mod.run_cli(Config(), "x")
    assert capsys.readouterr().out.count(wayland.ASK_TEXT) == 1


def test_cli_dry_run_on_wayland(monkeypatch, capsys):
    screen, brain = Screen(size=(1024, 768)), Brain()
    main_mod = cli(monkeypatch, screen, brain)
    cfg = Config()
    cfg.dry_run = True
    main_mod.run_cli(cfg, "x")
    c = brain.controllers[0]
    assert isinstance(c.backend, DryRunBackend) and c.backend is not screen.made[0].backend
    assert c.backend.size() == (1024, 768)


def test_cli_ctrl_c_still_ends_the_sharing(monkeypatch, capsys):
    screen = Screen()
    main_mod = cli(monkeypatch, screen, Brain(error=KeyboardInterrupt()))
    main_mod.run_cli(Config(), "x")
    assert "Stopped." in capsys.readouterr().out
    made = screen.made[0]
    assert made.closed == 1 and made.stop.is_set()


def test_cli_declined_asks_the_desktop_nothing(monkeypatch, capsys):
    screen = Screen()
    main_mod = cli(monkeypatch, screen, Brain(), answer="n")
    main_mod.run_cli(Config(), "x")
    assert "nothing was touched" in capsys.readouterr().out
    assert screen.made == []


def test_cli_on_x11_mentions_the_screen_corner(monkeypatch, capsys):
    main_mod = cli(monkeypatch, Screen(), Brain(), answer="n", wanted=False)
    main_mod.run_cli(Config(), "x")
    out = capsys.readouterr().out
    assert "screen corner" in out and "screen sharing" not in out


def test_open_flag_runs_open_box(monkeypatch):
    from leanwatch import __main__ as main_mod
    from leanwatch import open_box

    monkeypatch.setattr(open_box, "open_box", lambda session_file=None: 3)
    with pytest.raises(SystemExit) as err:
        main_mod.main(["--open"])
    assert err.value.code == 3


@pytest.mark.parametrize("argv,combo", [
    (["--shortcut", "on"], DEFAULT_HOTKEY),
    (["--shortcut", "on", "--hotkey", "ctrl+shift+k"], "ctrl+shift+k"),
    (["--shortcut", "on", "--hotkey", "none"], DEFAULT_HOTKEY),      # 'none' turns off our own grab, not this
])
def test_shortcut_on(argv, combo, monkeypatch, capsys):
    from leanwatch import __main__ as main_mod

    seen = []
    monkeypatch.setattr(gnome_shortcut, "install", lambda c: seen.append(c) or "Added the shortcut.")
    monkeypatch.setattr(gnome_shortcut, "remove", lambda: pytest.fail("not asked to remove"))
    main_mod.main(argv)
    assert seen == [combo] and "Added the shortcut." in capsys.readouterr().out


def test_shortcut_on_failure_exits_with_the_reason(monkeypatch):
    from leanwatch import __main__ as main_mod

    def install(combo):
        raise RuntimeError("GNOME's keyboard settings aren't available here")

    monkeypatch.setattr(gnome_shortcut, "install", install)
    with pytest.raises(SystemExit) as err:
        main_mod.main(["--shortcut", "on"])
    assert err.value.code == "Could not add the shortcut: GNOME's keyboard settings aren't available here"


def test_shortcut_off(monkeypatch, capsys):
    from leanwatch import __main__ as main_mod

    monkeypatch.setattr(gnome_shortcut, "remove", lambda: "Removed LeanWatch's shortcut.")
    monkeypatch.setattr(gnome_shortcut, "install", lambda c: pytest.fail("not asked to install"))
    main_mod.main(["--shortcut", "off"])
    assert "Removed LeanWatch's shortcut." in capsys.readouterr().out


# --------------------------------------------------------------------------- python -m leanwatch --check
def lines_for(monkeypatch, *, wanted=True, available=True, versions=None, python="/usr/bin/python3",
              gnome=True, installed=False):
    monkeypatch.setattr(wayland, "wanted", lambda: wanted)
    monkeypatch.setattr(wayland, "available", lambda: available)

    def portal_versions(bus=None):
        if isinstance(versions, Exception):
            raise versions
        return {"RemoteDesktop": 2, "ScreenCast": 5, "Screenshot": 2} if versions is None else versions

    monkeypatch.setattr(wayland, "portal_versions", portal_versions)
    monkeypatch.setattr(wayland, "find_gst_python", lambda: python)
    monkeypatch.setattr(gnome_shortcut, "available", lambda: gnome)
    monkeypatch.setattr(gnome_shortcut, "installed", lambda: installed)
    out = []
    ok = doctor._wayland_lines(lambda good, text: out.append((good, text)))
    return ok, out


def test_check_all_good(monkeypatch):
    ok, out = lines_for(monkeypatch, installed=True)
    assert ok is True
    assert [good for good, _ in out] == [True, True, True]
    assert "Allow Remote Interaction" in out[0][1]
    assert out[1][1] == "Screen pictures through PipeWire (GStreamer in /usr/bin/python3)"
    assert out[2][1] == "Keyboard shortcut set up in GNOME's settings"


def test_check_without_the_gnome_shortcut_suggests_it(monkeypatch):
    ok, out = lines_for(monkeypatch, installed=False)
    assert ok is True and out[-1] == (None, "For a keyboard shortcut that works in every app: python -m leanwatch --shortcut on")


def test_check_outside_gnome_says_nothing_about_shortcuts(monkeypatch):
    ok, out = lines_for(monkeypatch, gnome=False)
    assert ok is True and len(out) == 2 and not any("shortcut" in text for _, text in out)


def test_check_without_gstreamer_is_a_note_not_a_failure(monkeypatch):
    ok, out = lines_for(monkeypatch, python=None)
    assert ok is True
    assert out[0][0] is True
    assert out[1][0] is None and "gstreamer1.0-pipewire" in out[1][1] and "flashes" in out[1][1]


def test_check_without_jeepney(monkeypatch):
    ok, out = lines_for(monkeypatch, available=False)
    assert ok is False and out == [(False, "Wayland needs the jeepney package to ask your desktop for access: pip install jeepney")]


def test_check_without_a_portal(monkeypatch):
    ok, out = lines_for(monkeypatch, versions=RuntimeError("org.freedesktop.DBus.Error.ServiceUnknown"))
    assert ok is False and len(out) == 1
    assert out[0][0] is False and "xdg-desktop-portal" in out[0][1] and "ServiceUnknown" in out[0][1]


@pytest.mark.parametrize("versions", [{"ScreenCast": 5, "Screenshot": 2}, {"RemoteDesktop": 2}, {}])
def test_check_without_the_remote_desktop_portal(versions, monkeypatch):
    ok, out = lines_for(monkeypatch, versions=versions)
    assert ok is False and len(out) == 1
    assert out[0][0] is False and "xdg-desktop-portal-gnome" in out[0][1]


def test_check_with_wayland_turned_off(monkeypatch):
    ok, out = lines_for(monkeypatch, wanted=False, available=False)
    assert ok is True and out == [(None, "Wayland session, but LEANWATCH_WAYLAND=0: LeanWatch only reaches older X11 apps.")]


def test_wanted_follows_the_session_and_the_switch(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("LEANWATCH_WAYLAND", raising=False)
    assert wayland.wanted() is False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert wayland.wanted() is True
    monkeypatch.setenv("LEANWATCH_WAYLAND", "0")
    assert wayland.wanted() is False
    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("LEANWATCH_WAYLAND", "1")
    assert wayland.wanted() is True


# --------------------------------------------------------------------------- desktop helper
def test_gnome_shortcut_installed_never_raises(monkeypatch):
    from leanwatch import desktop

    monkeypatch.setattr(gnome_shortcut, "installed", lambda: True)
    assert desktop.gnome_shortcut_installed() is True
    monkeypatch.setattr(gnome_shortcut, "installed", lambda: False)
    assert desktop.gnome_shortcut_installed() is False

    def broken():
        raise RuntimeError("dconf is broken")

    monkeypatch.setattr(gnome_shortcut, "installed", broken)
    assert desktop.gnome_shortcut_installed() is False
