"""The floating command box: a small always-on-top window in the bottom-right
corner of every screen. Type a task, press Go, then lean back and watch.

Run it with ``python -m leanwatch --overlay`` (starts the agent too) or on its
own with ``python -m leanwatch.overlay`` while the agent is running.
It talks to the agent over its local HTTP API using the session token from
~/.leanwatch/session.json.
"""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import messagebox

from .config import Config

WIDTH, HEIGHT = 380, 128
MARGIN_X, MARGIN_Y = 20, 64   # keep clear of the taskbar / dock

INK, PAPER, TEAL, MARKER, SOFT = "#11242b", "#ffffff", "#0b6e68", "#ffc928", "#4a5d63"


class Api:
    def __init__(self, cfg: Config):
        info = json.loads(cfg.session_file.read_text())
        self.base = f"http://127.0.0.1:{info['port']}"
        self.token = info["token"]

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body or {}).encode() if method == "POST" else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-LeanWatch-Token": self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as err:
            try:
                return json.loads(err.read())
            except ValueError:
                return {"type": "error", "message": f"HTTP {err.code}"}

    def status(self) -> dict:
        return self._call("GET", "/api/status")

    def post(self, kind: str, body: dict | None = None) -> dict:
        return self._call("POST", f"/api/{kind}", body)


class Overlay:
    def __init__(self, api: Api):
        self.api = api
        self.root = tk.Tk()
        self.root.title("LeanWatch")
        self.root.overrideredirect(True)          # no title bar: it floats like a widget
        self.root.attributes("-topmost", True)    # stays above every window
        self.root.configure(bg=PAPER, highlightthickness=2, highlightbackground=TEAL)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{sw - WIDTH - MARGIN_X}+{sh - HEIGHT - MARGIN_Y}")

        head = tk.Frame(self.root, bg=PAPER)
        head.pack(fill="x", padx=10, pady=(8, 2))
        title = tk.Label(head, text="▲ LeanWatch", fg=INK, bg=PAPER, font=("Segoe UI", 11, "bold"), cursor="fleur")
        title.pack(side="left")
        tk.Button(head, text="✕", command=self.root.destroy, relief="flat", bg=PAPER, fg=SOFT, bd=0, cursor="hand2").pack(side="right")
        title.bind("<ButtonPress-1>", self._drag_start)
        title.bind("<B1-Motion>", self._drag_move)

        row = tk.Frame(self.root, bg=PAPER)
        row.pack(fill="x", padx=10, pady=4)
        self.entry = tk.Entry(row, font=("Segoe UI", 12), relief="solid", bd=1, fg=INK)
        self.entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.entry.bind("<Return>", lambda e: self.go())
        self.entry.bind("<Escape>", lambda e: self.stop())
        self.go_btn = tk.Button(row, text="Go", command=self.go, bg=TEAL, fg="white", relief="flat", padx=14, cursor="hand2",
                                font=("Segoe UI", 11, "bold"), activebackground=INK, activeforeground="white")
        self.go_btn.pack(side="left", padx=(6, 0))

        self.status = tk.Label(self.root, text="What should I do?", fg=SOFT, bg=PAPER, anchor="w", justify="left",
                               wraplength=WIDTH - 24, font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=10, pady=(2, 8))
        self.running = False
        self._note_until = 0.0
        self.entry.focus_set()
        self._poll()

    # dragging the box around
    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _bg(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def go(self):
        if self.running:
            return self.stop()
        text = self.entry.get().strip()
        if not text:
            return
        st = self.api.status()
        if not st.get("permission", {}).get("granted"):
            ok = messagebox.askyesno(
                "Allow LeanWatch?",
                "LeanWatch will move your real mouse pointer, click and type to show you how it's done.\n\n"
                "Stop it any time with Stop, or by pushing the mouse into a screen corner.\n\n"
                "Allow for this session?",
                parent=self.root,
            )
            if not ok:
                self._note("Okay. Nothing will move until you allow it.")
                return
            self.api.post("grant", {"mouse": True, "keyboard": True})
        self.entry.delete(0, "end")
        reply = self.api.post("command", {"text": text})
        if reply.get("type") == "error":
            self._note(reply.get("message", "Something went wrong."))

    def _note(self, text: str, seconds: float = 6) -> None:
        self.status.config(text=text, fg=INK)
        self._note_until = time.monotonic() + seconds

    def stop(self):
        self._bg(self.api.post, "stop")

    def _poll(self):
        def work():
            try:
                st = self.api.status()
            except (OSError, ValueError):
                st = {"error": True}
            self.root.after(0, self._show, st)
        self._bg(work)
        self.root.after(900, self._poll)

    def _show(self, st: dict):
        if st.get("error"):
            self.status.config(text="Can't reach the agent. Is `python -m leanwatch` running?")
            return
        self.running = bool(st.get("running"))
        self.go_btn.config(text="Stop" if self.running else "Go", bg="#b3261e" if self.running else TEAL)
        if st.get("paused"):
            self.status.config(text="Paused. Press Resume in the LeanWatch page.", fg=INK)
        elif self.running:
            self.status.config(text=st.get("last") or "Working…", fg=INK)
        elif time.monotonic() > self._note_until:
            self.status.config(text=st.get("last") or "What should I do?", fg=SOFT)

    def run(self):
        self.root.mainloop()


def main() -> None:
    cfg = Config()
    try:
        api = Api(cfg)
    except (OSError, ValueError, KeyError):
        print("The LeanWatch agent is not running. Start it with: python -m leanwatch --overlay")
        raise SystemExit(1)
    Overlay(api).run()


if __name__ == "__main__":
    main()
