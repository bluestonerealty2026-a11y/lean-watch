"""Start LeanWatch when you log in, so the bubble is always there when you need help.

    python -m leanwatch --autostart on     # plus any options you want it to start with
    python -m leanwatch --autostart off

Per-user, no administrator rights:
- Windows: a value under HKEY_CURRENT_USER\\...\\CurrentVersion\\Run, started with pythonw (no console window)
- macOS:   ~/Library/LaunchAgents/io.leanwatch.bubble.plist
- Linux:   ~/.config/autostart/leanwatch.desktop (freedesktop autostart, used by GNOME, KDE, Xfce, ...)
All three start a small launcher script in ~/.leanwatch that writes output to ~/.leanwatch/leanwatch.log.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path

LABEL = "io.leanwatch.bubble"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
AGENT_DIR = Path(__file__).resolve().parents[1]


def gui_python(executable: str = sys.executable, platform: str = sys.platform) -> str:
    """On Windows, pythonw.exe runs without opening a console window."""
    if platform == "win32":
        p = Path(executable)
        w = p.with_name(p.name.lower().replace("python.exe", "pythonw.exe"))
        if w.exists():
            return str(w)
    return executable


def launcher_source(args: list[str], agent_dir: Path = AGENT_DIR) -> str:
    return (
        "# -*- coding: utf-8 -*-\n"
        "# Written by `python -m leanwatch --autostart on`. Starts LeanWatch at login.\n"
        "import sys\n"
        "from pathlib import Path\n"
        "log = Path.home() / '.leanwatch' / 'leanwatch.log'\n"
        "log.parent.mkdir(parents=True, exist_ok=True)\n"
        "sys.stdout = sys.stderr = open(log, 'a', buffering=1, encoding='utf-8')\n"
        f"sys.path.insert(0, {str(agent_dir)!r})\n"
        f"sys.argv = ['leanwatch'] + {list(args)!r}\n"
        "from leanwatch.__main__ import main\n"
        "if __name__ == '__main__':   # the model's process re-runs this file and must not start LeanWatch again\n"
        "    main()\n"
    )


def desktop_quote(arg: str) -> str:
    """Quote one argument for the Exec= line of a .desktop file."""
    out = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{out}"'


class Autostart:
    def __init__(self, home: Path | None = None, platform: str = sys.platform, executable: str = sys.executable,
                 winreg=None):
        self.home = home or Path.home()
        self.platform = platform
        self.executable = executable
        self._winreg = winreg

    # where things go
    @property
    def launcher(self) -> Path:
        return self.home / ".leanwatch" / ("start_leanwatch.pyw" if self.platform == "win32" else "start_leanwatch.py")

    @property
    def entry(self) -> Path | None:
        if self.platform == "darwin":
            return self.home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        if self.platform.startswith("linux"):
            config = os.environ.get("XDG_CONFIG_HOME") if self.home == Path.home() else None
            return Path(config or self.home / ".config") / "autostart" / "leanwatch.desktop"
        return None

    def winreg(self):
        if self._winreg is None:
            import winreg

            self._winreg = winreg
        return self._winreg

    # install / remove
    def install(self, args: list[str]) -> str:
        self.launcher.parent.mkdir(parents=True, exist_ok=True)
        self.launcher.write_text(launcher_source(args), encoding="utf-8")   # Python reads source as UTF-8
        python = gui_python(self.executable, self.platform)
        if self.platform == "win32":
            command = f'"{python}" "{self.launcher}"'
            if len(command) > 260:
                raise RuntimeError("Windows limits start-up commands to 260 characters, and this one is longer: "
                                   "install Python (or LeanWatch's virtual environment) in a shorter folder.")
            reg = self.winreg()
            with reg.CreateKey(reg.HKEY_CURRENT_USER, RUN_KEY) as key:
                reg.SetValueEx(key, "LeanWatch", 0, reg.REG_SZ, command)
            note = (" Note: this looks like the Microsoft Store Python, which may keep the setting in a private copy "
                    "of the registry; the python.org installer is more reliable for this.") if "WindowsApps" in python else ""
            return f"LeanWatch will start when you sign in to Windows (HKCU\\{RUN_KEY}\\LeanWatch).{note}"
        entry = self.entry
        if entry is None:
            raise RuntimeError(f"Starting at login is not supported on {self.platform}.")
        entry.parent.mkdir(parents=True, exist_ok=True)
        if self.platform == "darwin":
            log = str(self.home / ".leanwatch" / "leanwatch.log")
            entry.write_bytes(plistlib.dumps({
                "Label": LABEL,
                "ProgramArguments": [python, str(self.launcher)],
                "WorkingDirectory": str(self.launcher.parent),
                "RunAtLoad": True,
                "KeepAlive": {"SuccessfulExit": False},      # restart after a crash, not after Quit
                "ProcessType": "Interactive",
                "LimitLoadToSessionType": "Aqua",            # only in the graphical login session
                "StandardOutPath": log,
                "StandardErrorPath": log,
            }))
            entry.chmod(0o644)
            return (f"LeanWatch will start when you log in to your Mac ({entry}). macOS asks for permissions again for "
                    "the Python it starts at login: allow 'Python' under System Settings > Privacy & Security > "
                    "Accessibility and Screen Recording.")
        entry.write_text(encoding="utf-8", data=
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=LeanWatch\n"
            "Comment=Floating helper that shows you how to do things on your computer\n"
            f"Exec={desktop_quote(python)} {desktop_quote(str(self.launcher))}\n"
            "Terminal=false\n"
            "X-GNOME-Autostart-enabled=true\n"
            "X-GNOME-Autostart-Delay=5\n"
        )
        return (f"LeanWatch will start when you log in ({entry}). If your window manager doesn't run autostart "
                f"entries (i3, sway, bspwm, ...), add this to its config instead: {python} {self.launcher}")

    def remove(self) -> str:
        removed = False
        if self.platform == "win32":
            reg = self.winreg()
            try:
                with reg.OpenKey(reg.HKEY_CURRENT_USER, RUN_KEY, 0, reg.KEY_SET_VALUE) as key:
                    reg.DeleteValue(key, "LeanWatch")
                    removed = True
            except OSError:
                pass
        elif self.entry is not None and self.entry.exists():
            self.entry.unlink()
            removed = True
        if self.launcher.exists():
            self.launcher.unlink()
        return "LeanWatch will no longer start at login." if removed else "LeanWatch was not set to start at login."

    def is_installed(self) -> bool:
        if self.platform == "win32":
            reg = self.winreg()
            try:
                with reg.OpenKey(reg.HKEY_CURRENT_USER, RUN_KEY) as key:
                    reg.QueryValueEx(key, "LeanWatch")
                return True
            except OSError:
                return False
        return self.entry is not None and self.entry.exists()
