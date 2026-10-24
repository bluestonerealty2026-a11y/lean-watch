"""--adapter must point at the adapter's folder: without it LeanWatch can't tell the base model and would quietly
load the default 7B one (seen on a real laptop, with the adapter unzipped somewhere else)."""

import json

import pytest

from leanwatch.__main__ import main
from leanwatch.brains import adapter_problem


def adapter(folder):
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "ByteDance-Seed/UI-TARS-2B-SFT"}))
    (folder / "leanwatch.json").write_text(json.dumps({"base_model": "ByteDance-Seed/UI-TARS-2B-SFT"}))
    return folder


def test_a_real_adapter_folder_is_fine(tmp_path):
    assert adapter_problem(str(adapter(tmp_path / "run"))) is None


def test_a_missing_folder_is_refused(tmp_path):
    assert "There is no adapter folder at" in adapter_problem(str(tmp_path / "lw-runs" / "ui-tars-2b-sft-leanwatch"))


def test_the_unzipped_parent_folder_points_one_level_down(tmp_path):
    inner = adapter(tmp_path / "ui-tars-2b-sft-leanwatch-20261024T112118Z-1-001" / "ui-tars-2b-sft-leanwatch")
    problem = adapter_problem(str(inner.parent))
    assert "no adapter_config.json" in problem and f"--adapter {inner}" in problem


def test_a_hub_repo_id_says_to_download_it():
    assert "hf download someone/uitars2b-leanwatch --local-dir ~/lw-runs/uitars2b-leanwatch" in \
        adapter_problem("someone/uitars2b-leanwatch")


def test_the_command_line_stops_before_starting_anything(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--adapter", str(tmp_path / "nowhere"), "--no-bubble", "--no-browser"])
    assert stopped.value.code == 2 and "There is no adapter folder at" in capsys.readouterr().err


def test_a_quoted_tilde_still_finds_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    adapter(tmp_path / "lw-runs" / "run")
    assert adapter_problem("~/lw-runs/run") is None
