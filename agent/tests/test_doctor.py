import json
import subprocess

import pytest

import leanwatch.doctor as doctor
from leanwatch.config import Config

ALL_OK = {"torch": {"ok": True, "version": "2.5.1"}, "torchvision": {"ok": True, "version": "0.20.1"},
          "accelerate": {"ok": True, "version": "1.15.0"}, "transformers": {"ok": True, "version": "5.11.0"}}


def broken(root_type, root_message, top_type="ModuleNotFoundError",
           top_message="Could not import module 'AutoModelForImageTextToText'. Are this object's requirements defined correctly?"):
    return {"ok": False, "type": top_type, "message": top_message, "root_type": root_type, "root_message": root_message,
            "traceback": ['  File ".../torchvision/_meta_registrations.py", line 164, in <module>',
                          f"{root_type}: {root_message}"]}


def fake_probe(monkeypatch, result=None, returncode=0, stderr=""):
    """Replace the subprocess that imports the AI libraries."""
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        out = "some library chatter\n" + (doctor.AI_CHECK_MARK + json.dumps(result) + "\n" if result is not None else "")
        return subprocess.CompletedProcess(cmd, returncode, out, stderr)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_the_check_runs_in_a_fresh_python(monkeypatch):
    import sys

    calls = fake_probe(monkeypatch, ALL_OK)
    assert doctor.check_ai_libraries() == ALL_OK
    cmd, kw = calls[0]
    assert cmd[0] == sys.executable and cmd[1] == "-c" and "torch.ops.torchvision.nms" in cmd[2]
    assert "AutoModelForImageTextToText" in cmd[2] and kw["timeout"] >= 120


def test_all_libraries_fine():
    assert doctor.ai_library_lines(ALL_OK) == [
        (True, "AI libraries: torch 2.5.1, torchvision 0.20.1, accelerate 1.15.0, transformers 5.11.0", [])]


def test_mismatched_torchvision_is_explained(monkeypatch):
    import leanwatch.errors as errors

    monkeypatch.setattr(errors, "installed", {"torch": "2.5.1", "torchvision": "0.15.2"}.get)
    monkeypatch.setattr(errors.sys, "platform", "linux")
    result = dict(ALL_OK, torchvision=broken("RuntimeError", "operator torchvision::nms does not exist", "RuntimeError",
                                             "operator torchvision::nms does not exist"),
                  transformers=broken("RuntimeError", "operator torchvision::nms does not exist"))
    lines = doctor.ai_library_lines(result)
    assert lines[0] == (True, "Imported fine: torch 2.5.1, accelerate 1.15.0", [])
    good, text, details = lines[1]
    assert good is False and text.startswith("torchvision doesn't import: RuntimeError: operator torchvision::nms does not exist")
    assert text.endswith("pip install torchvision==0.20.1") and details[-1].startswith("RuntimeError")
    assert lines[2][:2] == (False, "transformers doesn't import either, for the same reason as torchvision.")
    assert lines[3][0] is None and "clean environment" in lines[3][1]


def test_a_crash_while_importing(monkeypatch):
    fake_probe(monkeypatch, None, returncode=-11, stderr="Fatal Python error: Segmentation fault\n  File torch/__init__.py")
    result = doctor.check_ai_libraries()
    assert result["error"] == "Python crashed (SIGSEGV) while importing the AI libraries."
    good, text, details = doctor.ai_library_lines(result)[0]
    assert good is False and details[-1].endswith("torch/__init__.py")


def test_a_hang(monkeypatch):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    assert "took more than 5 minutes" in doctor.check_ai_libraries(timeout=300)["error"]


def report(monkeypatch, capsys, ram, free=None, libraries=ALL_OK):
    monkeypatch.setattr(doctor, "check_ai_libraries", lambda timeout=None, no_user_site=False: libraries)
    monkeypatch.setattr(doctor, "user_site_packages", lambda: None)
    monkeypatch.setattr(doctor, "_ram_gb", lambda: ram)
    monkeypatch.setattr(doctor, "_available_gb", lambda: free)
    code = doctor.report(Config())
    out = capsys.readouterr().out
    model_part = out.split("Checking the AI libraries")[1]
    # Whatever the screen checks above said: the summary must match the lines printed.
    assert ("Fix the items marked FIX" in out) == ("  FIX " in out) == (code == 1)
    return model_part, out


def test_8_gb_laptop_without_gpu(monkeypatch, capsys):
    part, out = report(monkeypatch, capsys, ram=8.2, free=3.1)
    assert out.index("Checking the AI libraries (this can take a minute the first time)...") < out.index("AI libraries: torch")
    assert "  FIX " not in part
    assert "Memory: 8 GB, but only 3.1 GB free right now: close big apps such as your web browser" in part
    assert "python -m leanwatch --small    (it loads in 16-bit, about 6 GB: close your web browser first)" in part


def test_too_little_memory_is_a_fix_line(monkeypatch, capsys):
    part, out = report(monkeypatch, capsys, ram=4.0, free=3.5)
    assert "  FIX  This computer has 4 GB of memory, less than even the small model needs (about 6 GB in 16-bit)" in part
    assert "Start LeanWatch with" not in part and "Fix the items marked FIX" in out


def test_big_and_medium_computers(monkeypatch, capsys):
    part, _ = report(monkeypatch, capsys, ram=16.7, free=14.0)
    assert "free right now" not in part and "--small    (UI-TARS-1.5-7B needs about 37 GB on a CPU" in part
    part, _ = report(monkeypatch, capsys, ram=64.0)
    assert "python -m leanwatch            (UI-TARS-1.5-7B on the CPU" in part


def test_gpu_comes_from_the_fresh_python(monkeypatch, capsys):
    libs = dict(ALL_OK, gpu={"kind": "cuda", "name": "RTX 4060", "memory_gb": 8.6})
    part, _ = report(monkeypatch, capsys, ram=16.7, libraries=libs)
    assert "OK   NVIDIA GPU: RTX 4060, 9 GB" in part and "--4bit" in part


def test_broken_libraries_make_the_check_fail(monkeypatch, capsys):
    libs = dict(ALL_OK, accelerate=broken("ModuleNotFoundError", "No module named 'yaml'", "ModuleNotFoundError",
                                          "No module named 'yaml'"))
    part, out = report(monkeypatch, capsys, ram=16.7, libraries=libs)
    assert "  FIX  accelerate doesn't import: ModuleNotFoundError: No module named 'yaml'\n" in part
    assert "pip install -r " in part and "Fix the items marked FIX" in out


@pytest.mark.parametrize("dist", ["torch", "torchvision", "accelerate", "transformers"])
def test_requirements_name_what_the_check_needs(dist):
    from leanwatch.errors import REQUIREMENTS

    assert any(line.startswith(dist) for line in REQUIREMENTS.read_text().splitlines())


def test_check_retries_without_the_user_site_packages(monkeypatch, capsys, tmp_path):
    """A stray package in ~/.local (pip install --user) breaks accelerate: --check says so and how to start."""
    import leanwatch.doctor as doctor
    from leanwatch.config import Config

    calls = []

    def check(timeout=None, no_user_site=False):
        calls.append(no_user_site)
        if no_user_site:
            return dict(ALL_OK)
        return dict(ALL_OK, accelerate=broken("ImportError", "cannot import name 'dispatch_model' from partially "
                                              "initialized module 'accelerate.big_modeling'"))

    (tmp_path / "stray").mkdir()
    monkeypatch.setattr(doctor, "check_ai_libraries", check)
    monkeypatch.setattr(doctor, "user_site_packages", lambda: str(tmp_path))
    doctor.report(Config())
    out = capsys.readouterr().out
    assert calls == [False, True]
    assert f"FIX  Without the packages in {tmp_path} the AI libraries import fine" in out
    assert "PYTHONNOUSERSITE=1 python -m leanwatch" in out


def test_no_retry_when_the_libraries_import_or_there_is_no_user_site(monkeypatch, capsys):
    import leanwatch.doctor as doctor
    from leanwatch.config import Config

    calls = []
    monkeypatch.setattr(doctor, "check_ai_libraries", lambda timeout=None, no_user_site=False: calls.append(no_user_site) or dict(ALL_OK))
    monkeypatch.setattr(doctor, "user_site_packages", lambda: "/home/x/.local/lib/python3.11/site-packages")
    doctor.report(Config())
    assert calls == [False]


def test_a_library_whose_dependency_failed_is_not_checked():
    from leanwatch.doctor import ai_library_lines

    libs = dict(ALL_OK, accelerate=broken("AttributeError", "module 'lib' has no attribute 'GEN_EMAIL'"),
                transformers={"ok": False, "skipped": "accelerate"})
    lines = ai_library_lines(libs)
    texts = [t for _, t, _ in lines]
    assert any("pip install --upgrade pyopenssl" in t for t in texts)
    assert (None, "transformers not checked: it needs accelerate, which doesn't import.", []) in lines


def test_the_check_script_skips_dependents_for_real(tmp_path):
    """Run the real check script with a fake accelerate that fails like the one on that laptop."""
    import json
    import subprocess
    import sys

    from leanwatch.doctor import AI_CHECK, AI_CHECK_MARK

    fake = tmp_path / "accelerate"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise AttributeError(\"module 'lib' has no attribute 'GEN_EMAIL'\")\n")
    out = subprocess.run([sys.executable, "-c", AI_CHECK], capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)})
    line = [s for s in out.stdout.splitlines() if s.startswith(AI_CHECK_MARK)][-1]
    result = json.loads(line[len(AI_CHECK_MARK):])
    assert result["accelerate"]["root_message"] == "module 'lib' has no attribute 'GEN_EMAIL'"
    assert result["transformers"] == {"ok": False, "skipped": "accelerate"}
