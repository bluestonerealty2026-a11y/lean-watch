"""Runs the real LeanWatch desktop app with a scripted model and reports what is on screen.

Started by run_bubble_x11.py on a virtual X display. The scripted model stands in for the
16 GB vision model: it clicks into a terminal, types a command, clicks the spot the LeanWatch
box covers (to check the box steps aside), and finishes. It saves every screenshot it is given.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # agent/

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

import leanwatch.bubble as bubble_mod  # noqa: E402
import leanwatch.server as server  # noqa: E402
from leanwatch.config import Config  # noqa: E402
from leanwatch.local_brain import ModelAnswer  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402

LOG = open(OUT / "events.log", "a", buffering=1)


def log(*parts):
    LOG.write(f"{time.monotonic():.3f} " + " ".join(map(str, parts)) + "\n")


class RecordingApp(bubble_mod.BubbleApp):
    """The real BubbleApp, plus a JSON dump of where its widgets are, for the test to read."""

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
            log("shield hide", [w.objectName() for w in self.shield._hidden])

        def logged_show():
            log("shield show")
            show()

        self.shield._do_hide, self.shield._do_show = logged_hide, logged_show
        self.session.listeners.append(lambda m: log("event", json.dumps(m)[:300]) if m.get("type") != "screenshot" else None)

    def dump(self):
        def geo(w):
            r = w.frameGeometry()
            return [r.x(), r.y(), r.width(), r.height(), w.isVisible()]

        def centre(w):
            c = w.mapToGlobal(w.rect().center())
            return [c.x(), c.y(), w.isVisible()]

        p = self.panel
        data = {"bubble": geo(self.bubble), "panel": geo(p), "consent_yes": centre(p.consent.yes),
                "confirm_yes": centre(p.confirm.yes), "go": centre(p.go), "input": centre(p.input),
                "running": self.running, "expanded": self.expanded, "notice": p.notice.text(),
                "steps": p.steps.text(), "state": p.state.text(), "translucent": self.bubble.translucent,
                "excluded": self.shield.excluded_from_capture}
        tmp = OUT / "state.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(OUT / "state.json")


class ScriptedModel:
    def __init__(self):
        self.n = 0

    def answer(self, messages, images, stop=None):
        img = images[-1]
        img.save(OUT / f"model_saw_{self.n}.png")
        tx, ty = json.loads(os.environ["XTERM_CENTRE"])
        pg = RecordingApp.app.panel.frameGeometry()
        px, py = pg.center().x(), pg.center().y()
        answers = [
            f"Thought: I click in the terminal window so my typing goes there.\nAction: click(start_box='<|box_start|>({tx},{ty})<|box_end|>')",
            "Thought: I type a short command and press Enter, so the terminal greets you.\nAction: type(content='echo hello from LeanWatch\\n')",
            f"Thought: I click on the spot the LeanWatch box covers, to show it steps aside.\nAction: click(start_box='<|box_start|>({px},{py})<|box_end|>')",
            "Thought: The terminal printed the greeting.\nAction: finished(content='Done: the terminal said hello. You can type commands there yourself the same way.')",
        ]
        time.sleep(float(os.environ.get("THINK_S", "0.6")))
        text = answers[min(self.n, len(answers) - 1)]
        self.n += 1
        log("model answer", self.n, text.splitlines()[-1])
        return ModelAnswer(text, img.width, img.height)


bubble_mod.BubbleApp = RecordingApp
server.load_local_model = lambda cfg: ScriptedModel()

cfg = Config()
cfg.port = int(os.environ.get("PORT", "8799"))
cfg.local_model = "scripted-test-model"
cfg.move_speed = 1.5
from leanwatch.desktop import run  # noqa: E402

sys.exit(run(cfg, web_page=False, hotkey=os.environ.get("HOTKEY", "ctrl+alt+space")))
