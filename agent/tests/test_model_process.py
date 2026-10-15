"""The model in its own process. The fake model (fake_model.py) loads in milliseconds; one test loads a real tiny one."""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil
import pytest
from PIL import Image

from leanwatch.config import Config
from leanwatch.errors import ModelError, Stopped
from leanwatch.hooks import RunResult
from leanwatch.local_model import ModelAnswer
from leanwatch.model_process import ModelProcess

HERE = Path(__file__).resolve().parent
FAKE = "fake_model:FakeModel"
posix_only = pytest.mark.skipif(os.name == "nt", reason="SIGKILL and the out-of-memory killer are POSIX")


def ask(text="dark mode"):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def shot(w=64, h=48):
    return [Image.new("RGB", (w, h), "gray")]


def gone(pid: int, within: float = 10.0) -> bool:
    """True once the process has ended (a zombie waiting to be reaped counts as ended)."""
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def models():
    made = []

    def make(**options):
        m = ModelProcess("org/fake-2B", loader=FAKE, **options)
        made.append(m)
        return m

    yield make
    for m in made:
        m.close()


def test_loads_reports_and_answers(models):
    notes = []
    m = models(status=notes.append)
    m.wait_ready()
    assert m.coords == "rel1000" and m.info["dtype"] == "bfloat16" and m.info["pid"] == m.pid != os.getpid()
    assert notes == ["Loading org/fake-2B (fake) in 16-bit."]
    later = []
    m.listen(later.append)                    # a listener that comes late still hears the loading notes
    assert later == notes
    ans = m.answer(ask("turn on dark mode"), shot(1366, 768) + shot(320, 200))
    assert isinstance(ans, ModelAnswer) and (ans.image_w, ans.image_h) == (1000, 800)
    assert "task='turn on dark mode'" in ans.text and "(1366, 768), (320, 200)" in ans.text
    assert not m.failed


def test_options_reach_the_loader(models):
    m = models(cpu_dtype="bfloat16", device="cpu")
    m.wait_ready()
    assert m.info["options"] == ["cpu_dtype", "device"]


def test_the_child_stays_light(models):
    """No Qt, pyautogui or controller in the model's process: only the model and its error messages."""
    m = models()
    m.wait_ready()
    assert set(m.info["modules"]) <= {"leanwatch", "leanwatch.model_process", "leanwatch.errors", "leanwatch.memory",
                                      "leanwatch.local_model"}


def test_stop_ends_a_long_answer_early(models):
    m = models(steps=200, step_s=0.01)        # a 2-second answer
    m.wait_ready()
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    t = time.monotonic()
    ans = m.answer(ask(), shot(), stop=stop)
    assert time.monotonic() - t < 1.5
    assert "tokens=" in ans.text and int(ans.text.split("tokens=")[1].split()[0]) < 100   # the model saw the flag
    full = m.answer(ask("again"), shot(), stop=threading.Event())   # the flag is cleared for the next answer
    assert "task='again'" in full.text and "tokens=200" in full.text


def test_stop_does_not_wait_for_a_model_that_cannot_stop(models):
    m = models(steps=60, step_s=0.01, ignore_stop=True)    # e.g. still reading the screenshot
    m.stop_grace = 0.1
    m.wait_ready()
    stop = threading.Event()
    stop.set()
    with pytest.raises(Stopped):
        m.answer(ask(), shot(), stop=stop)
    stop2 = threading.Event()
    threading.Timer(0.05, stop2.set).start()
    t = time.monotonic()
    with pytest.raises(Stopped):
        m.answer(ask("cut short"), shot(), stop=stop2)
    assert time.monotonic() - t < 1.5
    # The late answers are thrown away: the next question gets its own answer.
    ans = m.answer(ask("next one"), shot())
    assert "task='next one'" in ans.text


def test_stop_while_loading_keeps_loading_for_the_next_task(models):
    m = models(load_s=1.5)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    t = time.monotonic()
    with pytest.raises(Stopped):
        m.wait_ready(stop)
    assert time.monotonic() - t < 1.0
    assert not m.failed
    m.wait_ready(threading.Event())           # the next task: same process, now loaded
    assert "task='later'" in m.answer(ask("later"), shot()).text


def test_load_error_shows_the_real_reason(models):
    m = models(fail="torchvision")
    with pytest.raises(ModelError) as err:
        m.wait_ready()
    text = str(err.value)
    assert text.startswith("Could not load the model: RuntimeError: operator torchvision::nms does not exist.")
    assert "torchvision" in text.splitlines()[1] and "pip install" in text
    assert m.failed
    with pytest.raises(ModelError):           # answer() reports the same, instead of hanging
        m.answer(ask(), shot())


@posix_only
def test_killed_while_answering_says_out_of_memory(models, monkeypatch):
    import leanwatch.memory as memory

    monkeypatch.setattr(memory, "ram_gb", lambda: 8.2)
    import leanwatch.model_process as mp

    monkeypatch.setattr(mp, "ram_gb", lambda: 8.2)
    m = models(die_on_answer=True)
    m.wait_ready()
    with pytest.raises(ModelError) as err:
        m.answer(ask(), shot())
    text = str(err.value)
    assert "ran out of memory (this computer has 8 GB)" in text
    assert "web browser" in text and "The next task starts the model again." in text
    assert m.failed and gone(m.pid)


@posix_only
def test_killed_while_loading(models):
    m = models(die_on_load=True)
    with pytest.raises(ModelError, match="ran out of memory"):
        m.wait_ready()


def test_close_ends_the_child(models):
    m = models()
    m.wait_ready()
    pid = m.pid
    m.close()
    assert gone(pid, 5)
    with pytest.raises(ModelError, match="shut down"):
        m.answer(ask(), shot())


# --------------------------------------------------------------------------- the whole agent
def fake_session(monkeypatch, **first_options):
    """A Session whose model is a fake one in its own process; the first one made gets first_options."""
    import leanwatch.server as server

    made = []

    def load(cfg):
        m = ModelProcess("org/fake-2B", loader=FAKE, **(first_options if not made else {}))
        made.append(m)
        return m

    class Brain:
        def __init__(self, model, stop):
            self.model, self.stop = model, stop

        def run(self, task):
            return RunResult(True, self.model.answer(ask(task), shot(), stop=self.stop).text)

    monkeypatch.setattr(server, "load_local_model", load)
    monkeypatch.setattr(server, "make_brain", lambda cfg, controller, hooks, stop, local_model=None: Brain(local_model, stop))
    cfg = Config()
    cfg.dry_run = True
    s = server.Session(cfg, "tok")
    s.permissions.grant()
    finished = []
    s.listeners.append(lambda e: e["type"] == "run_finished" and finished.append(e))

    def run(task, stop_after=None):
        assert s.start(task) is None
        if stop_after is not None:
            time.sleep(stop_after)
            s.stop()
        s.thread.join(30)
        return finished[-1]

    return s, made, run


@posix_only
def test_after_a_crash_the_next_task_starts_a_new_model(monkeypatch):
    s, made, run = fake_session(monkeypatch, die_on_answer=True)
    try:
        first = run("first")
        assert not first["ok"] and "ran out of memory" in first["message"]
        second = run("second")
        assert second["ok"] and "task='second'" in second["result"]
        assert len(made) == 2 and made[0].pid != made[1].pid and gone(made[0].pid)
    finally:
        s.shutdown()
    assert gone(made[1].pid, 5)


def test_stop_while_the_model_loads(monkeypatch):
    s, made, run = fake_session(monkeypatch, load_s=1.5)
    try:
        t = time.monotonic()
        first = run("first", stop_after=0.3)
        assert time.monotonic() - t < 1.3
        assert not first["ok"] and "keeps loading" in first["message"]
        second = run("second")
        assert second["ok"] and len(made) == 1          # the same model, loaded in the meantime
    finally:
        s.shutdown()


def test_a_failed_load_is_tried_again_with_a_fresh_process(monkeypatch):
    s, made, run = fake_session(monkeypatch, fail="torchvision")
    try:
        first = run("first")
        assert not first["ok"] and first["message"].startswith("Could not load the model: RuntimeError")
        assert run("second")["ok"] and len(made) == 2
    finally:
        s.shutdown()


# --------------------------------------------------------------------------- the child on its own
def run_script(tmp_path, body: str, **kw) -> subprocess.Popen:
    script = tmp_path / "start.py"
    script.write_text(f"import sys\nsys.path[:0] = [{str(HERE.parent)!r}, {str(HERE)!r}]\n" + textwrap.dedent(body))
    return subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, cwd=tmp_path, **kw)


@posix_only
def test_child_exits_when_leanwatch_is_killed(tmp_path):
    parent = run_script(tmp_path, """
        import time
        from leanwatch.model_process import ModelProcess
        if __name__ == "__main__":
            m = ModelProcess("x", loader="fake_model:FakeModel")
            m.wait_ready()
            print(m.pid, flush=True)
            time.sleep(60)
    """)
    pid = int(parent.stdout.readline())
    parent.kill()                              # no goodbye: like a crash, or the OOM killer picking LeanWatch
    parent.wait(10)
    assert gone(pid, 10)


def test_child_exits_with_leanwatch_and_writes_to_its_log(tmp_path):
    log = tmp_path / "leanwatch.log"
    parent = run_script(tmp_path, f"""
        sys.stdout = sys.stderr = open({str(log)!r}, "a", buffering=1, encoding="utf-8")   # like the login launcher
        from leanwatch.model_process import ModelProcess
        if __name__ == "__main__":
            m = ModelProcess("x", loader="fake_model:FakeModel", say="hello from the model's process")
            m.wait_ready()
            sys.__stdout__.write(f"{{m.pid}}\\n")
            sys.__stdout__.flush()
    """)                                       # then it simply ends, without close(): atexit ends the child
    pid = int(parent.stdout.readline())
    assert parent.wait(20) == 0
    assert gone(pid, 5)
    assert "hello from the model's process" in log.read_text()


def test_an_old_start_script_is_not_run_twice(tmp_path):
    """spawn re-runs the script that started LeanWatch in the model's process; main() must start nothing there."""
    parent = run_script(tmp_path, """
        import leanwatch.doctor as doctor
        doctor.report = lambda cfg: print("LEANWATCH STARTED", flush=True) or 0
        from leanwatch.__main__ import main
        try:
            main(["--check"])                  # an older login launcher calls main() with no __main__ check
        except SystemExit:
            pass
        from leanwatch.model_process import ModelProcess
        if __name__ == "__main__":
            m = ModelProcess("x", loader="fake_model:FakeModel")
            m.wait_ready()
            print("MODEL READY", flush=True)
            m.close()
    """)
    out, _ = parent.communicate(timeout=30)
    assert out.count("LEANWATCH STARTED") == 1 and out.count("MODEL READY") == 1


# --------------------------------------------------------------------------- choosing where the model runs
def test_load_local_model_picks_the_process_or_this_one(monkeypatch, capsys):
    import leanwatch.local_model as local_model
    import leanwatch.model_process as model_process
    from leanwatch.brains import load_local_model

    seen = {}

    class Recorder:
        def __init__(self, model_id, status=None, **options):
            seen.update(model_id=model_id, options=options)
            status("Loading in 16-bit (bfloat16) to fit.")

    monkeypatch.setattr(model_process, "ModelProcess", Recorder)
    cfg = Config()
    cfg.local_model, cfg.cpu_dtype = "org/fake-2B", "bfloat16"
    assert isinstance(load_local_model(cfg), Recorder)
    assert seen["model_id"] == "org/fake-2B" and seen["options"]["cpu_dtype"] == "bfloat16"
    assert "  Loading in 16-bit (bfloat16) to fit." in capsys.readouterr().out   # one line in the terminal

    def broken(model_id, **options):
        try:
            raise RuntimeError("operator torchvision::nms does not exist")
        except RuntimeError as err:
            raise ModuleNotFoundError("Could not import module 'AutoModelForImageTextToText'.") from err

    monkeypatch.setattr(local_model, "LocalModel", broken)
    cfg.model_process = False
    with pytest.raises(ModelError, match="RuntimeError: operator torchvision::nms does not exist"):
        load_local_model(cfg)
    assert "Traceback" in capsys.readouterr().err       # the full error, for the terminal or the log


def test_real_tiny_qwen2_vl_answers_from_its_own_process_in_16_bit(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(HERE.parents[1] / "training" / "tests"))
    from tiny_model import build

    from leanwatch.brains import load_local_model
    from leanwatch.local_prompts import build_messages, computer_prompt

    cfg = Config()
    cfg.local_model = str(build(tmp_path / "tiny", family="qwen2_vl"))
    cfg.device, cfg.cpu_dtype, cfg.max_new_tokens = "cpu", "auto", 8
    notes = []
    model = load_local_model(cfg, status=notes.append)
    try:
        assert isinstance(model, ModelProcess)
        messages, _ = build_messages(computer_prompt("Turn on dark mode"), [], 3)
        ans = model.answer(messages, [Image.new("RGB", (1366, 768), "gray")])
        assert isinstance(ans, ModelAnswer) and isinstance(ans.text, str)
        assert ans.image_w % 28 == 0 and ans.image_h % 28 == 0 and ans.image_w * ans.image_h <= 28 * 28 * 64
        assert model.coords == "rel1000"                       # Qwen2-VL, like UI-TARS-2B-SFT: a 0-1000 grid
        assert model.info["dtype"] == "bfloat16" and model.info["device"] == "cpu"
        assert len(notes) == 1 and "16-bit (bfloat16)" in notes[0]
    finally:
        model.close()
