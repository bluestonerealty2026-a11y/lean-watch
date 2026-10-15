import plistlib
import sys
import threading
from contextlib import contextmanager

import pytest

from leanwatch.__main__ import main, strip_option
from leanwatch.autostart import RUN_KEY, Autostart, desktop_quote, gui_python, launcher_source
from leanwatch.config import Config
from leanwatch.controller import Controller, DryRunBackend
from leanwatch.permissions import Permissions
from leanwatch.screen import Geometry


def test_linux_desktop_entry(tmp_path):
    a = Autostart(home=tmp_path, platform="linux", executable="/opt/py env/bin/python3")
    msg = a.install(["--4bit", "--adapter", "/runs/my model"])
    entry = tmp_path / ".config" / "autostart" / "leanwatch.desktop"
    text = entry.read_text()
    assert 'Exec="/opt/py env/bin/python3" "' + str(tmp_path / ".leanwatch" / "start_leanwatch.py") + '"' in text
    assert "Terminal=false" in text and str(entry) in msg
    launcher = (tmp_path / ".leanwatch" / "start_leanwatch.py").read_text()
    assert "['leanwatch'] + ['--4bit', '--adapter', '/runs/my model']" in launcher
    compile(launcher, "launcher", "exec")
    assert a.is_installed()
    assert "no longer" in a.remove()
    assert not entry.exists() and not a.is_installed()
    assert "was not set" in a.remove()


def test_macos_launch_agent(tmp_path):
    a = Autostart(home=tmp_path, platform="darwin", executable="/usr/local/bin/python3")
    a.install([])
    plist = plistlib.loads((tmp_path / "Library" / "LaunchAgents" / "io.leanwatch.bubble.plist").read_bytes())
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == ["/usr/local/bin/python3", str(tmp_path / ".leanwatch" / "start_leanwatch.py")]
    assert plist["StandardErrorPath"].endswith("leanwatch.log")


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    REG_SZ = 1
    KEY_SET_VALUE = 2

    def __init__(self):
        self.values = {}

    @contextmanager
    def CreateKey(self, root, path):
        yield (root, path)

    @contextmanager
    def OpenKey(self, root, path, reserved=0, access=0):
        yield (root, path)

    def SetValueEx(self, key, name, reserved, kind, value):
        self.values[(key, name)] = value

    def QueryValueEx(self, key, name):
        if (key, name) not in self.values:
            raise FileNotFoundError(name)
        return self.values[(key, name)], 1

    def DeleteValue(self, key, name):
        if (key, name) not in self.values:
            raise FileNotFoundError(name)
        del self.values[(key, name)]


def test_windows_run_key_uses_pythonw(tmp_path):
    exe = tmp_path / "Python313" / "python.exe"
    exe.parent.mkdir()
    exe.write_text("")
    (exe.parent / "pythonw.exe").write_text("")
    reg = FakeWinreg()
    a = Autostart(home=tmp_path, platform="win32", executable=str(exe), winreg=reg)
    a.install(["--preload"])
    value = reg.values[(("HKCU", RUN_KEY), "LeanWatch")]
    assert value == f'"{exe.parent / "pythonw.exe"}" "{tmp_path / ".leanwatch" / "start_leanwatch.pyw"}"'
    assert a.is_installed()
    a.remove()
    assert not a.is_installed() and not (tmp_path / ".leanwatch" / "start_leanwatch.pyw").exists()


def test_helpers():
    assert desktop_quote('a "b" $c `d` \\e') == '"a \\"b\\" \\$c \\`d\\` \\\\e"'
    assert gui_python("/usr/bin/python3", "linux") == "/usr/bin/python3"
    assert strip_option(["--autostart", "on", "--4bit", "--autostart=on", "--web"], "--autostart", takes_value=True) == ["--4bit", "--web"]
    src = launcher_source(["--hotkey", "ctrl+alt+l"])
    assert "sys.stdout = sys.stderr = open(log" in src   # pythonw has no console to print to


def test_cli_autostart_keeps_the_other_options(tmp_path, monkeypatch):
    import leanwatch.autostart as auto

    monkeypatch.setattr(auto.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(auto, "Autostart", lambda: Autostart(home=tmp_path, platform="linux"))
    main(["--autostart", "on", "--4bit", "--hotkey", "ctrl+alt+l"])
    launcher = (tmp_path / ".leanwatch" / "start_leanwatch.py").read_text()
    assert "['--4bit', '--hotkey', 'ctrl+alt+l']" in launcher
    main(["--autostart", "off"])
    assert not (tmp_path / ".config" / "autostart" / "leanwatch.desktop").exists()


class RecordingShield:
    def __init__(self, covered):
        self.covered = covered
        self.calls = []

    @contextmanager
    def clear_of(self, points):
        self.calls.append(list(points))
        yield

    @contextmanager
    def cloak(self):
        yield


def test_controller_moves_the_bubble_out_of_the_way_of_clicks_only():
    shield = RecordingShield(covered=True)
    backend = DryRunBackend((1000, 800))
    perms = Permissions()
    perms.grant(mouse=True, keyboard=True)
    c = Controller(backend, None, Geometry.for_screen(1000, 800, 1000, 10**7), perms, speed=50, shield=shield,
                   stop=threading.Event())
    c.execute("left_click", {"coordinate": [900, 700]})
    c.execute("mouse_move", {"coordinate": [100, 100]})
    c.execute("left_click_drag", {"start_coordinate": [10, 10], "coordinate": [20, 20]})
    c.execute("key", {"text": "ctrl+s"})
    assert shield.calls == [[(900, 700)], [(10, 10), (20, 20)]]


def test_launcher_is_utf8_and_paths_are_absolute(tmp_path, monkeypatch):
    import leanwatch.autostart as auto

    home = tmp_path / "José Ñúñez"
    run = home / "runs" / "my-model"
    run.mkdir(parents=True)
    monkeypatch.chdir(home)
    monkeypatch.setattr(auto, "Autostart", lambda: Autostart(home=home, platform="linux"))
    main(["--autostart", "on", "--adapter", "runs/my-model", "--language", "Español"])
    launcher = (home / ".leanwatch" / "start_leanwatch.py").read_bytes().decode("utf-8")
    assert str(run) in launcher and "Español" in launcher
    compile(launcher, "launcher", "exec")


def test_abbreviated_autostart_option_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(["--auto", "on"])


def test_small_switch_and_check(monkeypatch):
    import leanwatch.doctor as doctor
    import leanwatch.server as server

    seen = []
    monkeypatch.setattr(server, "serve", lambda cfg: seen.append(cfg))
    main(["--small", "--no-bubble", "--no-browser"])
    assert seen[-1].local_model == "ByteDance-Seed/UI-TARS-2B-SFT"
    main(["--small", "--model", "my/model", "--no-bubble", "--no-browser"])
    assert seen[-1].local_model == "my/model"                    # an explicit --model wins
    monkeypatch.setattr(doctor, "report", lambda cfg: 7)
    with pytest.raises(SystemExit) as stop:
        main(["--check"])
    assert stop.value.code == 7


def test_check_report_runs_here(capsys):
    from leanwatch.doctor import report
    from leanwatch.config import Config

    code = report(Config())
    out = capsys.readouterr().out
    assert code in (0, 1) and "Start LeanWatch with" in out
