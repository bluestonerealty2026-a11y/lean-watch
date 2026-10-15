"""Opt-in: the real bubble in a headless GNOME Shell (Wayland), using the desktop's remote desktop portal.

    LEANWATCH_GNOME_TESTS=1 python -m pytest tests/gnome        (as root, e.g. in a container)

Needs gnome-shell, xdg-desktop-portal-gnome, pipewire, wireplumber, xwayland, gstreamer1.0-pipewire,
python3-gi with GTK 4, and jeepney. See run_gnome.py for what is checked.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = ("gnome-shell", "pipewire", "wireplumber", "unshare", "gdbus", "dbus-run-session", "pw-dump", "Xwayland")
pytestmark = pytest.mark.skipif(
    os.environ.get("LEANWATCH_GNOME_TESTS") != "1" or any(shutil.which(t) is None for t in TOOLS)
    or not hasattr(os, "geteuid") or os.geteuid() != 0 or importlib.util.find_spec("jeepney") is None,
    reason="set LEANWATCH_GNOME_TESTS=1, run as root and install GNOME Shell, the portals, PipeWire and jeepney",
)


def test_bubble_on_gnome_wayland(tmp_path):
    script = Path(__file__).with_name("run_gnome.py")
    r = subprocess.run([sys.executable, "-W", "ignore", str(script), str(tmp_path / "run")],
                       capture_output=True, text=True, timeout=400)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
