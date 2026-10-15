"""Regression tests for the training-side review findings."""

import json

import pytest

import evaluate


def example(target, action="click", bbox=(0, 0, 100, 100)):
    return {"task": "actions", "meta": {"action": action, "bbox": list(bbox), "width": 1000, "height": 625},
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": target}]}]}


def test_scoring_compares_with_the_trained_answer():
    right = example("Thought: menu\nAction: right_single(start_box='<|box_start|>(50,50)<|box_end|>')")
    assert evaluate.score(right, "Action: right_single(start_box='(50,50)')", 1000, 625, "abs")["type_ok"]
    assert not evaluate.score(right, "Action: click(start_box='(50,50)')", 1000, 625, "abs")["type_ok"]
    double = example("Action: left_double(start_box='<|box_start|>(50,50)<|box_end|>')")
    assert not evaluate.score(double, "Action: click(start_box='(50,50)')", 1000, 625, "abs")["type_ok"]
    single = example("Action: click(start_box='<|box_start|>(50,50)<|box_end|>')")
    assert evaluate.score(single, "Action: left_single(start_box='(50,50)')", 1000, 625, "abs")["type_ok"]


def test_rows_with_different_meta_load_together(tmp_path):
    pytest.importorskip("datasets")
    import train_sft
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    prompt = [{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image"}]}]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": "Action: wait()"}]}]
    rows = [
        {"images": ["a.png"], "prompt": prompt, "completion": completion, "task": "grounding", "meta": {"bbox": [1, 2, 3, 4]}},
        {"images": ["a.png"], "prompt": prompt, "completion": completion, "task": "grounding", "meta": {"point": [0.5, 0.5], "x": "y"}},
        {"images": ["a.png"], "prompt": prompt, "completion": completion, "task": "actions", "meta": {"step": 3}},
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = train_sft.load_split(tmp_path / "train.jsonl", tmp_path)
    assert len(ds) == 3 and set(ds.column_names) == {"images", "prompt", "completion"}
    only = train_sft.load_split(tmp_path / "train.jsonl", tmp_path, {"actions"})
    assert len(only) == 1


def test_env_model_folder_settings_are_used(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "leanwatch.json").write_text(json.dumps({"coords": "rel1000", "style": "point", "max_pixels": 602112,
                                                       "history_images": 1, "language": "Spanish"}))
    monkeypatch.setenv("LEANWATCH_LOCAL_MODEL", str(merged))
    import leanwatch.__main__ as entry
    import leanwatch.server as server

    seen = []
    monkeypatch.setattr(server, "serve", lambda cfg: seen.append(cfg))
    entry.main(["--no-bubble", "--no-browser"])
    cfg = seen[-1]
    assert (cfg.local_model, cfg.coords, cfg.style, cfg.local_max_pixels, cfg.history_images, cfg.language) == \
        (str(merged), "rel1000", "point", 602112, 1, "Spanish")
    entry.main(["--no-bubble", "--no-browser", "--language", "French"])
    assert seen[-1].language == "French"            # a flag still wins
