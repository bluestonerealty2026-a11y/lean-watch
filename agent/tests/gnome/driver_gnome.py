"""Runs the real LeanWatch desktop app in a headless GNOME (Wayland) session with a scripted model.

Started by run_gnome.py. The bubble is a Qt window on XWayland; the screen, mouse and keyboard go
through the desktop's remote desktop portal (leanwatch/wayland.py). The scripted model reads its
answers from <out>/script.json (rewritten by the runner for each task); an answer may name a file
in "wait_for" that the runner creates when it is ready for that step. Every screenshot the model
is given is saved, so the test can check LeanWatch's own windows are never in it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # agent/

OUT = Path(sys.argv[1])

import leanwatch.bubble as bubble_mod  # noqa: E402
import leanwatch.server as server  # noqa: E402
from leanwatch.config import Config  # noqa: E402
from leanwatch.local_brain import ModelAnswer  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402

LOG = open(OUT / "events.log", "a", buffering=1)


def log(*parts):
    LOG.write(f"{time.monotonic():.3f} " + " ".join(map(str, parts)) + "\n")


class RecordingApp(bubble_mod.BubbleApp):
    """The real BubbleApp, plus a JSON dump of its widgets and focus, for the runner to read."""

    app = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        RecordingApp.app = self
        self._dump_timer = QTimer(self, interval=150)
        self._dump_timer.timeout.connect(self.dump)
        self._dump_timer.start()
        hide, show = self.shield._do_hide, self.shield._do_show

        def logged_hide():
            hide()
            log("shield hide", len(self.shield._hidden))

        def logged_show():
            log("shield show")
            show()

        self.shield._do_hide, self.shield._do_show = logged_hide, logged_show
        step_aside = self._step_aside

        def logged_step_aside():
            log("step aside")
            step_aside()

        self._step_aside = logged_step_aside
        self.session.listeners.append(self._event)

    def _event(self, m):
        if m.get("type") == "run_started":
            ScriptedModel.n = 0
        if m.get("type") != "screenshot":
            log("event", json.dumps(m)[:400])

    def dump(self):
        def geo(w):
            r = w.frameGeometry()
            return [r.x(), r.y(), r.width(), r.height(), w.isVisible()]

        def centre(w):
            c = w.mapToGlobal(w.rect().center())
            return [c.x(), c.y(), w.isVisible()]

        p = self.panel
        data = {"bubble": geo(self.bubble), "panel": geo(p), "consent_yes": centre(p.consent.yes),
                "input": centre(p.input), "input_text": p.input.text(), "go": centre(p.go), "running": self.running, "expanded": self.expanded,
                "panel_active": p.isActiveWindow(), "notice": p.notice.text(), "steps": p.steps.text(),
                "activity": p.activity.text() if hasattr(p, "activity") else "", "wayland": self.on_wayland,
                "shortcut_ok": self.shortcut_ok}
        tmp = OUT / "state.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(OUT / "state.json")


class ScriptedModel:
    n = 0

    def answer(self, messages, images, stop=None):
        img = images[-1]
        i = ScriptedModel.n
        ScriptedModel.n += 1
        script = json.loads((OUT / "script.json").read_text())
        step = script[min(i, len(script) - 1)]
        img.save(OUT / f"model_saw_{script_name()}_{i}.png")
        if step.get("wait_for"):
            end = time.time() + 60
            while not (OUT / step["wait_for"]).exists() and time.time() < end and not (stop and stop.is_set()):
                time.sleep(0.1)
        time.sleep(0.3)
        log("model answer", i, step["text"].splitlines()[-1])
        return ModelAnswer(step["text"], img.width, img.height)


def script_name():
    try:
        return json.loads((OUT / "script.json").read_text())[0].get("task", "task")
    except Exception:
        return "task"


bubble_mod.BubbleApp = RecordingApp
server.load_local_model = lambda cfg: ScriptedModel()

cfg = Config()
cfg.port = 8797
cfg.local_model = "scripted-test-model"
cfg.move_speed = 2.0
from leanwatch.desktop import run  # noqa: E402

sys.exit(run(cfg, web_page=False, hotkey="ctrl+alt+space"))
