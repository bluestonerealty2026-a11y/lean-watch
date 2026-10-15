"""`python -m leanwatch --check`: is this computer ready, and which model fits?"""

from __future__ import annotations

import os
import platform
import sys

from .config import Config

# Approximate memory for the weights alone (activations add 1-3 GB).
MODELS = {
    "ByteDance-Seed/UI-TARS-1.5-7B": 8.3e9,
    "ByteDance-Seed/UI-TARS-2B-SFT": 2.2e9,
}


def _ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def _gpu():
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return "cuda", p.name, p.total_memory / 1e9
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple GPU", None
    except Exception:
        pass
    return None, None, None


def _wayland_lines(line) -> bool:
    """Wayland: the screen, mouse and keyboard go through the desktop's portal (see wayland.py)."""
    from . import wayland

    if not wayland.wanted():
        line(None, "Wayland session, but LEANWATCH_WAYLAND=0: LeanWatch only reaches older X11 apps.")
        return True
    if not wayland.available():
        line(False, "Wayland needs the jeepney package to ask your desktop for access: pip install jeepney")
        return False
    try:
        versions = wayland.portal_versions()
    except Exception as err:
        line(False, f"Can't reach your desktop's portal (xdg-desktop-portal): {err}")
        return False
    if "RemoteDesktop" not in versions or "ScreenCast" not in versions:
        line(False, "Your desktop has no remote desktop portal. On GNOME: sudo apt install xdg-desktop-portal-gnome "
                    "(KDE: xdg-desktop-portal-kde), then log out and in again.")
        return False
    line(True, f"Wayland: screen, mouse and keyboard through your desktop's screen sharing "
               f"(it asks once; turn on “Allow Remote Interaction”)")
    python = wayland.find_gst_python()
    if python:
        line(True, f"Screen pictures through PipeWire (GStreamer in {python})")
    else:
        line(None, "GStreamer's PipeWire plugin isn't available, so LeanWatch takes screenshots instead and the screen "
                   "flashes each time: sudo apt install gstreamer1.0-pipewire python3-gi gir1.2-gstreamer-1.0")
    from . import gnome_shortcut

    if gnome_shortcut.available():
        if gnome_shortcut.installed():
            line(True, "Keyboard shortcut set up in GNOME's settings")
        else:
            line(None, "For a keyboard shortcut that works in every app: python -m leanwatch --shortcut on")
    return True


def report(cfg: Config) -> int:
    ok = True

    def line(good: bool | None, text: str) -> None:
        mark = {True: "OK  ", False: "FIX ", None: "NOTE"}[good]
        print(f"  {mark} {text}")

    print(f"LeanWatch check — Python {platform.python_version()} on {platform.system()} {platform.release()}")

    # 1. The bubble (Qt)
    try:
        import PySide6

        line(True, f"PySide6 {PySide6.__version__} (the bubble)")
    except ImportError:
        ok = False
        line(False, "PySide6 is missing: pip install PySide6-Essentials")
    if sys.platform.startswith("linux"):
        from .desktop import _xcb_cursor_available

        if _xcb_cursor_available():
            line(True, "libxcb-cursor (Qt's X11 plugin needs it)")
        else:
            ok = False
            line(False, "libxcb-cursor is missing: sudo apt install libxcb-cursor0   (Fedora: sudo dnf install xcb-util-cursor)")
        from .native import wayland_session

        if wayland_session():
            ok = _wayland_lines(line) and ok
        elif os.environ.get("DISPLAY"):
            line(True, "X11 session")
        else:
            ok = False
            line(False, "No display found (DISPLAY is not set). Run LeanWatch from your desktop session.")

    # 2. Mouse, keyboard and screen
    try:
        from .controller import PyAutoGuiBackend

        w, h = PyAutoGuiBackend().size()
        line(True, f"Screen {w} x {h}, mouse and keyboard reachable")
    except Exception as err:
        ok = False
        line(False, f"Can't reach the screen: {err}")

    # 3. The model
    ram = _ram_gb()
    kind, name, vram = _gpu()
    print()
    if kind == "cuda":
        line(True, f"NVIDIA GPU: {name}, {vram:.0f} GB")
    elif kind == "mps":
        line(True, "Apple GPU (Metal)")
    else:
        line(None, "No NVIDIA/Apple GPU: the model runs on the CPU (slower: several seconds to a minute per step)")
    if ram:
        line(None, f"Memory: {ram:.0f} GB")
    big, small = MODELS["ByteDance-Seed/UI-TARS-1.5-7B"], MODELS["ByteDance-Seed/UI-TARS-2B-SFT"]
    if kind == "cuda" and vram:
        if vram >= big * 2 / 1e9 + 2:
            advice = "python -m leanwatch            (UI-TARS-1.5-7B in 16-bit fits your GPU)"
        elif vram >= 7:
            advice = "python -m leanwatch --4bit     (pip install bitsandbytes first; UI-TARS-1.5-7B in about 6 GB)"
        else:
            advice = "python -m leanwatch --small    (your GPU is small: the 2B model)"
    elif kind == "mps":
        advice = "python -m leanwatch" if (ram or 0) >= 24 else "python -m leanwatch --small"
    else:
        need_big, need_small = big * 4 / 1e9 + 2, small * 4 / 1e9 + 1.5     # 32-bit weights on a CPU
        if ram and ram >= need_big:
            advice = "python -m leanwatch            (UI-TARS-1.5-7B on the CPU: works, but slowly)"
        elif ram is None or ram >= need_small:
            advice = (f"python -m leanwatch --small    (UI-TARS-1.5-7B needs about {need_big:.0f} GB on a CPU; "
                      f"the 2B model needs about {need_small:.0f} GB)")
        else:
            ok = False
            advice = f"This computer has less memory than even the small model needs (about {need_small:.0f} GB)."
    print(f"\n  Start LeanWatch with:\n    {advice}")
    print("\nEverything looks ready." if ok else "\nFix the items marked FIX, then run this check again.")
    return 0 if ok else 1
