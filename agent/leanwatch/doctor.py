"""`python -m leanwatch --check`: is this computer ready, and which model fits?"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .config import Config
from .memory import MODELS, SMALL_MODEL, available_gb as _available_gb, need_gb, ram_gb as _ram_gb

BIG_MODEL = "ByteDance-Seed/UI-TARS-1.5-7B"
AI_CHECK_TIMEOUT_S = 300          # the first import of torch on a slow disk can take minutes
AI_CHECK_MARK = "LEANWATCH-CHECK "
# Run with this computer's Python in a fresh process, like the model's own process.
AI_CHECK = r'''
import json, sys, traceback

def root(e):
    seen = [e]
    while True:
        n = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        if n is None or any(n is s for s in seen):
            return e
        seen.append(n)
        e = n

def check_torch():
    import torch
    return torch.__version__

def check_torchvision():
    import torch, torchvision
    torch.ops.torchvision.nms     # its compiled operators: missing when torch and torchvision don't match
    return torchvision.__version__

def check_accelerate():
    import accelerate
    return accelerate.__version__

def check_transformers():
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor   # what LeanWatch loads
    return transformers.__version__

out = {}
for name, check in (("torch", check_torch), ("torchvision", check_torchvision),
                    ("accelerate", check_accelerate), ("transformers", check_transformers)):
    before = set(sys.modules)
    try:
        out[name] = {"ok": True, "version": check()}
    except BaseException as e:
        r = root(e)
        tb = "".join(traceback.format_exception(type(r), r, r.__traceback__)).splitlines()
        out[name] = {"ok": False, "type": type(e).__name__, "message": str(e)[:1000],
                     "root_type": type(r).__name__, "root_message": str(r)[:1000], "traceback": tb[-8:]}
        for m in set(sys.modules) - before:      # a half-imported package must not spoil the next check
            sys.modules.pop(m, None)
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = {"kind": "cuda", "name": p.name, "memory_gb": p.total_memory / 1e9}
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out["gpu"] = {"kind": "mps", "name": "Apple GPU"}
except BaseException:
    pass
print("LEANWATCH-CHECK " + json.dumps(out), flush=True)
'''


def check_ai_libraries(timeout: float = AI_CHECK_TIMEOUT_S, no_user_site: bool = False) -> dict:
    """Import torch, torchvision, accelerate and transformers in a fresh Python (this one), as the model does.
    no_user_site: without the packages in ~/.local (python -s).

    Returns {"torch": {"ok": True, "version": "2.5.1"}, ..., "gpu": {...}}, or {"error": ..., "traceback": [...]}
    when that Python crashed or hung.
    """
    import json
    import signal
    import subprocess

    try:
        done = subprocess.run([sys.executable, *(["-s"] if no_user_site else []), "-c", AI_CHECK],
                              capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"Importing the AI libraries took more than {timeout / 60:.0f} minutes, so I gave up."}
    except OSError as err:
        return {"error": f"Could not start {sys.executable}: {err}"}
    for text in reversed(done.stdout.splitlines()):
        if text.startswith(AI_CHECK_MARK):
            return json.loads(text[len(AI_CHECK_MARK):])
    code = done.returncode
    try:
        how = f"crashed ({signal.Signals(-code).name})" if code < 0 else f"stopped (exit code {code})"
    except ValueError:
        how = f"crashed (signal {-code})"
    return {"error": f"Python {how} while importing the AI libraries.",
            "traceback": done.stderr.strip().splitlines()[-8:]}


def ai_libraries_ok(result: dict) -> bool:
    return "error" not in result and all(r.get("ok") for name, r in result.items() if name != "gpu")


def user_site_packages() -> str | None:
    """~/.local/lib/pythonX.Y/site-packages, if this Python mixes packages from there in (pip install --user)."""
    import site

    path = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    try:
        return path if path and any(Path(path).iterdir()) else None
    except OSError:
        return None


def ai_library_lines(result: dict) -> list[tuple[bool | None, str, list[str]]]:
    """--check lines for the AI libraries: (OK / FIX / NOTE, text, last lines of the traceback)."""
    from .errors import clean_env_command, known_fix, one_line

    if "error" in result:
        return [(False, result["error"], result.get("traceback", [])),
                (None, f"A clean environment fixes most install problems: {clean_env_command()}", [])]
    libraries = {k: v for k, v in result.items() if k != "gpu"}
    fine = [f"{name} {r['version']}" for name, r in libraries.items() if r.get("ok")]
    lines: list = []
    reasons: dict = {}
    for name, r in libraries.items():
        if r.get("ok"):
            continue
        reason = (r["root_type"], r["root_message"])
        if reason in reasons:
            lines.append((False, f"{name} doesn't import either, for the same reason as {reasons[reason]}.", []))
            continue
        reasons[reason] = name
        fix = known_fix(r["root_type"], r["root_message"])[0] or known_fix(r["type"], r["message"])[0]
        text = f"{name} doesn't import: {r['root_type']}: {one_line(r['root_message'])}" + (f"\n{fix}" if fix else "")
        lines.append((False, text, r.get("traceback", [])))
    if not lines:
        return [(True, "AI libraries: " + ", ".join(fine), [])]
    if fine:
        lines.insert(0, (True, "Imported fine: " + ", ".join(fine), []))
    lines.append((None, f"A clean environment fixes most install problems: {clean_env_command()}", []))
    return lines


def memory_line(ram: float, free: float | None) -> str:
    text = f"Memory: {ram:.0f} GB"
    if free is not None and free < ram * 0.7:
        text += f", but only {free:.1f} GB free right now: close big apps such as your web browser before you start"
    return text


def model_advice(kind: str | None, vram: float | None, ram: float | None) -> tuple[str | None, str | None]:
    """(how to start LeanWatch on this computer, a FIX line when not even the small model fits)."""
    big, small = MODELS[BIG_MODEL], MODELS[SMALL_MODEL]
    if kind == "cuda" and vram:
        if vram >= big * 2 / 1e9 + 2:
            return "python -m leanwatch            (UI-TARS-1.5-7B in 16-bit fits your GPU)", None
        if vram >= 7:
            return "python -m leanwatch --4bit     (pip install bitsandbytes first; UI-TARS-1.5-7B in about 6 GB)", None
        return "python -m leanwatch --small    (your GPU is small: the 2B model)", None
    if kind == "mps":
        return ("python -m leanwatch" if (ram or 0) >= 24 else "python -m leanwatch --small"), None
    big32, small32, small16 = need_gb(big, "float32"), need_gb(small, "float32"), need_gb(small, "bfloat16")
    if ram is not None and ram >= big32:
        return "python -m leanwatch            (UI-TARS-1.5-7B on the CPU: works, but slowly)", None
    if ram is None or ram >= small32:
        return (f"python -m leanwatch --small    (UI-TARS-1.5-7B needs about {big32:.0f} GB on a CPU; "
                f"the 2B model needs about {small32:.0f} GB)"), None
    if ram >= small16:
        return (f"python -m leanwatch --small    (it loads in 16-bit, about {small16:.0f} GB: "
                "close your web browser first)"), None
    return None, (f"This computer has {ram:.0f} GB of memory, less than even the small model needs "
                  f"(about {small16:.0f} GB in 16-bit). Use a computer with 8 GB or more, or the Claude brain "
                  "(--brain claude, needs an API key).")


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
        nonlocal ok
        ok = ok and good is not False     # "Fix the items marked FIX" only when one was printed
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
    print()
    print("  Checking the AI libraries (this can take a minute the first time)...", flush=True)
    libraries = check_ai_libraries()
    for good, text, details in ai_library_lines(libraries):
        line(good, text.replace("\n", "\n" + " " * 7))
        for d in details:
            print(f"         {d}")
    user_site = user_site_packages()
    if not ai_libraries_ok(libraries) and user_site:
        # Packages installed with "pip install --user" are mixed into every Python of this version and are a
        # common cause: try once without them.
        print(f"  Trying again without the packages in {user_site}...", flush=True)
        if ai_libraries_ok(check_ai_libraries(no_user_site=True)):
            line(False, f"Without the packages in {user_site} the AI libraries import fine, so one of those breaks "
                        "them. Start LeanWatch like this:\n"
                        "         PYTHONNOUSERSITE=1 python -m leanwatch\n"
                        "       or find and remove the package that clashes (pip list --user shows them).")
        else:
            line(None, f"It fails without the packages in {user_site} too, so they are not the cause.")
    gpu = libraries.get("gpu") or {}
    kind, vram = gpu.get("kind"), gpu.get("memory_gb")
    ram, free = _ram_gb(), _available_gb()
    if kind == "cuda":
        line(True, f"NVIDIA GPU: {gpu.get('name')}, {vram:.0f} GB")
    elif kind == "mps":
        line(True, "Apple GPU (Metal)")
    else:
        line(None, "No NVIDIA/Apple GPU: the model runs on the CPU (slower: from several seconds to a few minutes per step)")
    if ram:
        line(None, memory_line(ram, free))
    advice, fix = model_advice(kind, vram, ram)
    if fix:
        line(False, fix)
    if advice:
        print(f"\n  Start LeanWatch with:\n    {advice}")
    print("\nEverything looks ready." if ok else "\nFix the items marked FIX, then run this check again.")
    return 0 if ok else 1
