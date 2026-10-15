import sys

import pytest

import leanwatch.errors as errors
from leanwatch.errors import TORCHVISION_FOR_TORCH, explain_load_error, known_fix, root_cause


def chained(root: BaseException) -> BaseException:
    """What transformers' lazy import raises when importing the model class fails part-way."""
    try:
        try:
            raise root
        except BaseException as err:
            raise ModuleNotFoundError("Could not import module 'AutoModelForImageTextToText'. "
                                      "Are this object's requirements defined correctly?") from err
    except ModuleNotFoundError as outer:
        return outer


@pytest.fixture
def versions(monkeypatch):
    found = {"torch": "2.5.1", "torchvision": "0.15.2", "numpy": "1.26.4"}
    monkeypatch.setattr(errors, "installed", lambda dist: found.get(dist))
    monkeypatch.setattr(errors.sys, "platform", "linux")
    return found


def test_root_cause_follows_cause_and_context():
    try:
        try:
            raise KeyError("deep")
        except KeyError:
            raise ValueError("middle")         # __context__
    except ValueError as err:
        outer = RuntimeError("top")
        outer.__cause__ = err
    assert isinstance(root_cause(outer), KeyError)
    assert root_cause(KeyError("alone")).args == ("alone",)


def test_root_cause_survives_a_cycle():
    a, b = ValueError("a"), TypeError("b")
    a.__context__, b.__context__ = b, a
    assert root_cause(a) is b
    a.__cause__ = a
    assert root_cause(a) is a


def test_root_cause_respects_from_none():
    try:
        try:
            raise KeyError("hidden on purpose")
        except KeyError:
            raise RuntimeError("the real message") from None
    except RuntimeError as err:
        assert root_cause(err) is err


def test_torchvision_mismatch_names_the_matching_version(versions):
    msg = explain_load_error(chained(RuntimeError("operator torchvision::nms does not exist")))
    lines = msg.splitlines()
    assert lines[0] == "Could not load the model: RuntimeError: operator torchvision::nms does not exist."
    assert "torch 2.5.1 and torchvision 0.15.2 don't match" in lines[1]
    assert lines[1].endswith("pip install torchvision==0.20.1")        # no full stop after a command
    assert "clean environment" in lines[2]
    assert lines[-1] == "The full error is printed in the terminal."


def test_cpu_only_torch_gets_the_cpu_build_of_torchvision(versions):
    versions["torch"] = "2.5.1+cpu"
    fix, install = known_fix("RuntimeError", "operator torchvision::nms does not exist")
    assert fix.endswith("pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu") and install
    versions["torchvision"] = "0.20.1+cpu"         # right version, broken install
    assert "--force-reinstall --no-deps torchvision==0.20.1" in known_fix("RuntimeError", "Couldn't load custom C++ ops")[0]
    # torchvision 0.15.2 still imports with torch 2.5.1, but its operators are missing:
    assert "torchvision==0.20.1" in known_fix("AttributeError", "'_OpNamespace' 'torchvision' object has no attribute 'nms'")[0]
    versions["torch"] = "9.9.9"
    assert "previous-versions" in known_fix("RuntimeError", "operator torchvision::nms does not exist")[0]


def test_version_pairs_move_together():
    """Each torch minor release has its own torchvision minor release, in order."""
    pairs = sorted((tuple(map(int, t.split("."))), tuple(map(int, v.split(".")))) for t, v in TORCHVISION_FOR_TORCH.items())
    assert TORCHVISION_FOR_TORCH["2.5.1"] == "0.20.1"
    for (t, v) in pairs:
        assert v[1] == t[1] + 15 and v[0] == 0


def test_half_imported_package(versions):
    err = ImportError("cannot import name 'dispatch_model' from partially initialized module 'accelerate.big_modeling' "
                      "(most likely due to a circular import)")
    msg = explain_load_error(err)
    assert "ImportError: cannot import name 'dispatch_model'" in msg
    assert "half-loaded by an earlier failed try" in msg and "python -m leanwatch --check" in msg
    assert "clean environment" in msg


@pytest.mark.parametrize("name,fix", [
    ("cv2", "pip install opencv-python"), ("yaml", "pip install -r "), ("flash_attn", "pip install flash-attn"),
    ("torch", "pip install -r "), ("some_new_thing", "pip install some-new-thing"),
    ("PIL._not_there", "PIL is installed incompletely (PIL._not_there is missing). Fix: pip install --force-reinstall pillow"),
])
def test_missing_module(name, fix):
    err = ModuleNotFoundError(f"No module named '{name}'", name=name)
    got, install = known_fix(type(err).__name__, str(err))
    assert fix in got and install
    if "pip install -r" in fix:
        assert got.endswith("requirements.txt")


def test_numpy_built_for_another_version(versions):
    err = chained(AttributeError("_ARRAY_API not found"))
    assert "pip install --upgrade numpy" in explain_load_error(err)
    versions["numpy"] = "2.1.3"
    msg = explain_load_error(ImportError("A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.3"))
    assert "NumPy 2.1.3 is installed" in msg and 'pip install "numpy<2"' in msg
    assert "numpy" in known_fix("ValueError", "numpy.dtype size changed, may indicate binary incompatibility")[0].lower()


def test_out_of_memory_advice(monkeypatch):
    import leanwatch.memory as memory

    monkeypatch.setattr(memory, "ram_gb", lambda: 8.2)
    msg = explain_load_error(RuntimeError("[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: can't allocate "
                                          "memory: you tried to allocate 1048576 bytes."), "ByteDance-Seed/UI-TARS-1.5-7B")
    assert "ran out of memory (it has 8 GB)" in msg
    assert "web browser" in msg and "--small" in msg
    assert "clean environment" not in msg                  # not an install problem
    small = explain_load_error(MemoryError(), "ByteDance-Seed/UI-TARS-2B-SFT")
    assert "--small" not in small and "UI-TARS-2B-SFT in 16-bit needs about 6 GB" in small
    forced = explain_load_error(MemoryError(), "ByteDance-Seed/UI-TARS-2B-SFT", cpu_dtype="float32")
    assert "--cpu-dtype bfloat16" in forced
    assert "graphics card" in known_fix("OutOfMemoryError", "CUDA out of memory. Tried to allocate 2 GiB")[0]


def test_other_first_start_problems():
    assert "internet connection" in known_fix("OSError", "We couldn't connect to 'https://huggingface.co' "
                                                          "(Max retries exceeded)")[0]
    assert "disk is full" in known_fix("OSError", "[Errno 28] No space left on device")[0]
    assert "--model" in known_fix("OSError", "foo/bar is not a local folder and is not a valid model identifier")[0]
    assert known_fix("ValueError", "something nobody has seen") == (None, False)


def test_unknown_error_still_shows_type_message_and_where_details_are():
    msg = explain_load_error(ValueError("first line\nsecond line"))
    assert msg.splitlines() == ["Could not load the model: ValueError: first line.", "The full error is printed in the terminal."]


def test_details_go_to_the_log_file_when_started_at_login(tmp_path, monkeypatch):
    log = tmp_path / "leanwatch.log"
    with open(log, "a") as f:
        monkeypatch.setattr(sys, "stderr", f)
        assert errors.details_hint() == f"The full error is in {log}."


def test_clean_environment_command(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/home/me/anaconda3")
    cmd = errors.clean_env_command()
    assert cmd.startswith("conda create -n leanwatch python=3.11 -y && conda activate leanwatch && pip install -r ")
    assert cmd.endswith("requirements.txt")
    monkeypatch.delenv("CONDA_PREFIX")
    monkeypatch.setattr(errors, "in_conda", lambda: False)
    monkeypatch.setattr(errors.os, "name", "posix")
    assert "-m venv" in errors.clean_env_command()
