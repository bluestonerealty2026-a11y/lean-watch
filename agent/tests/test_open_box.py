"""open_box.py (python -m leanwatch --open): asks the running LeanWatch, over its local API, to open the box.

Against a real local server (the same ServerThread and make_app the desktop helper uses) on a free port.
"""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from leanwatch import open_box as open_box_mod
from leanwatch.config import Config
from leanwatch.desktop import ServerThread
from leanwatch.server import SESSION, make_app, write_session_file

TOKEN = "open-box-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def running(home):
    cfg = Config()
    cfg.port = free_port()
    cfg.dry_run = True
    app = make_app(cfg, TOKEN)
    events = []
    app[SESSION].listeners.append(events.append)
    server = ServerThread(app, "127.0.0.1", cfg.port)
    server.start()
    assert server.ready.wait(10) and server.error is None
    yield cfg, events
    server.stop()


def session_file(path, **info):
    path.write_text(json.dumps(info))
    return path


def wait_for(events, n=1, timeout=3.0):
    end = time.monotonic() + timeout
    while len(events) < n and time.monotonic() < end:
        time.sleep(0.01)
    return events


def test_opens_the_box_of_the_running_leanwatch(running, tmp_path, capsys):
    cfg, events = running
    path = session_file(tmp_path / "session.json", port=cfg.port, token=TOKEN, pid=os.getpid())
    assert open_box_mod.open_box(path) == 0
    assert wait_for(events) == [{"type": "open_box"}]
    assert capsys.readouterr().err == ""


def test_reads_the_session_file_the_server_writes(running, home):
    cfg, events = running
    written = write_session_file(cfg, TOKEN)
    assert written == home / ".leanwatch" / "session.json"
    assert open_box_mod.open_box() == 0                    # the default place
    assert wait_for(events) == [{"type": "open_box"}]


def test_never_goes_through_a_proxy(running, tmp_path, monkeypatch):
    cfg, events = running
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")     # nothing listens there
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    path = session_file(tmp_path / "session.json", port=cfg.port, token=TOKEN)
    assert open_box_mod.open_box(path) == 0
    assert wait_for(events) == [{"type": "open_box"}]


def test_a_wrong_token_is_refused(running, tmp_path, capsys):
    cfg, events = running
    path = session_file(tmp_path / "session.json", port=cfg.port, token="stale-token")
    assert open_box_mod.open_box(path) == 1
    assert "403" in capsys.readouterr().err
    time.sleep(0.05)
    assert events == []


def test_nobody_listening(home, tmp_path, capsys):
    path = session_file(tmp_path / "session.json", port=free_port(), token=TOKEN)
    started = time.monotonic()
    assert open_box_mod.open_box(path) == 1
    assert "didn't answer" in capsys.readouterr().err
    assert time.monotonic() - started < 5


def test_missing_session_file(home, capsys):
    assert open_box_mod.open_box() == 1                    # ~/.leanwatch/session.json doesn't exist
    assert "isn't running" in capsys.readouterr().err


@pytest.mark.parametrize("content", [
    "", "not json", "{", "null", "[]", "42", '"text"',
    '{"token": "t"}', '{"port": 8765}', '{"port": "eighty", "token": "t"}', '{"port": null, "token": "t"}',
    '{"port": [1], "token": "t"}',
])
def test_garbled_session_file(content, tmp_path, capsys):
    path = tmp_path / "session.json"
    path.write_text(content)
    assert open_box_mod.open_box(path) == 1
    assert "isn't running" in capsys.readouterr().err


def test_runs_as_a_standalone_script(running, home):
    """GNOME's shortcut runs the file itself, from any folder, with nothing but the standard library."""
    cfg, events = running
    write_session_file(cfg, TOKEN)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["HOME"] = str(home)
    r = subprocess.run([sys.executable, "-I", str(open_box_mod.__file__)], env=env, cwd="/",
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert wait_for(events) == [{"type": "open_box"}]
