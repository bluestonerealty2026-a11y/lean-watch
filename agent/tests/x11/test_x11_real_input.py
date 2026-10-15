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
