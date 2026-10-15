"""Opt-in: the real bubble on a virtual X display, driven with real mouse and keyboard events.

    LEANWATCH_X11_TESTS=1 python -m pytest tests/x11
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = ("Xvfb", "openbox", "xdotool", "xterm")
pytestmark = pytest.mark.skipif(
    os.environ.get("LEANWATCH_X11_TESTS") != "1" or any(shutil.which(t) is None for t in TOOLS),
    reason="set LEANWATCH_X11_TESTS=1 and install Xvfb, openbox, xdotool and xterm",
)


@pytest.mark.parametrize("mode", ["plain", "comp"])
def test_bubble_with_real_input(tmp_path, mode):
    if mode == "comp" and shutil.which("xcompmgr") is None:
        pytest.skip("xcompmgr not installed")
    script = Path(__file__).with_name("run_bubble_x11.py")
    r = subprocess.run([sys.executable, "-W", "ignore", str(script), str(tmp_path), mode],
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def test_shortcut_taken_by_another_program_is_reported():
    """A second grab of the same combination fails cleanly (BadAccess) instead of silently doing nothing."""
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb not installed")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from leanwatch.native import X11Native\n"
        "first = X11Native().register_hotkey('ctrl+alt+space', lambda: None)\n"
        "second = X11Native().register_hotkey('ctrl+alt+space', lambda: None)\n"
        "other = X11Native().register_hotkey('ctrl+alt+l', lambda: None)\n"
        "print(first is not None, second is None, other is not None)\n"
    ) % str(Path(__file__).resolve().parents[2])
    xvfb = subprocess.Popen(["Xvfb", ":93", "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time

        time.sleep(1)
        r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "DISPLAY": ":93"},
                           capture_output=True, text=True, timeout=30)
        assert r.stdout.split() == ["True", "True", "True"], r.stdout + r.stderr
    finally:
        xvfb.terminate()
