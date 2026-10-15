"""The whole training pipeline on a CPU, with a tiny Qwen2.5-VL instead of a 7B one.

practice recordings -> build_dataset -> train_sft (LoRA) -> evaluate -> the agent loads the adapter.
It checks that the parts fit together, not that the tiny model learns anything useful.
"""

import json

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")

import build_dataset  # noqa: E402
import evaluate  # noqa: E402
import train_sft  # noqa: E402
from tiny_model import build as build_tiny  # noqa: E402

from leanwatch.actions import CoordSpace, parse_response  # noqa: E402

W, H = 1000, 625
SAVE = (700, 500, 820, 540)       # bbox of a "Save" button
DARK = (100, 120, 260, 150)       # bbox of a "Dark mode" switch


def screen(path, pressed=False):
    img = Image.new("RGB", (W, H), "#20242c" if pressed else "#e8ebf0")
    d = ImageDraw.Draw(img)
    d.rectangle(SAVE, fill="#3a7bd5")
    d.rectangle(DARK, fill="#888888")
    img.save(path)


def centre(b):
    return {"x": (b[0] + b[2]) / 2, "y": (b[1] + b[3]) / 2}


def make_practice(root, episodes=6):
    (root / "images").mkdir(parents=True)
    with open(root / "episodes.jsonl", "w") as eps, open(root / "grounding.jsonl", "w") as gr:
        for e in range(episodes):
            names = [f"e{e:05d}_s{k:02d}.png" for k in range(4)]
            for k, name in enumerate(names):
                screen(root / "images" / name, pressed=k > 0)
            steps = [
                {"image": f"images/{names[0]}", "thought": "Dark mode is under Display. I click its switch.",
                 "action": {"type": "click", "clicks": 1, **centre(DARK)}, "bbox": list(DARK), "label": "Dark mode"},
                {"image": f"images/{names[1]}", "thought": "Now I name the file.",
                 "action": {"type": "type", "text": "notes 'v2'\n"}},
                {"image": f"images/{names[2]}", "thought": "Save it with Ctrl+S.",
                 "action": {"type": "hotkey", "keys": ["Ctrl", "S"]}},
                {"image": f"images/{names[3]}", "thought": "Dark mode is on and the file is saved.",
                 "action": {"type": "finished", "text": "Done. Next time: Display, then Dark mode."}},
            ]
            eps.write(json.dumps({"id": f"e{e:05d}", "instruction": "Turn on dark mode and save my notes",
                                  "skills": ["dark"], "summary": "", "width": W, "height": H, "steps": steps}) + "\n")
            for label, role, b in (("Save", "button", SAVE), ("Dark mode", "switch", DARK)):
                gr.write(json.dumps({"image": f"images/{names[0]}", "label": label, "role": role, "target": label,
                                     "bbox": list(b), "width": W, "height": H}) + "\n")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_tiny(tmp_path_factory.mktemp("tiny"))


@pytest.mark.parametrize("style", ["box", "point"])
def test_pipeline_end_to_end(tmp_path, tiny, style):
    practice, data, run = tmp_path / "practice", tmp_path / "sft", tmp_path / "run"
    make_practice(practice)
    max_pixels = 28 * 28 * 48   # smaller than the tiny processor's own cap, so the override must win

    build_dataset.main(["--practice", str(practice), "--out", str(data), "--style", style,
                        "--max-pixels", str(max_pixels), "--val-fraction", "0.34", "--seed", "1"])
    meta = json.loads((data / "meta.json").read_text())
    assert meta["style"] == style and meta["max_pixels"] == max_pixels
    rows = [json.loads(line) for line in open(data / "train.jsonl")] + [json.loads(line) for line in open(data / "val.jsonl")]
    assert {r["task"] for r in rows} == {"actions", "grounding"}
    # Prompt and answer use the same way of writing points.
    marker = "<point>" if style == "point" else "<|box_start|>"
    for r in rows:
        assert marker in r["prompt"][0]["content"][0]["text"]
        answer = r["completion"][0]["content"][0]["text"]
        parse_response(answer)
        if r["meta"].get("bbox"):
            assert marker in answer
        assert sum(c["type"] == "image" for m in r["prompt"] for c in m["content"]) == len(r["images"])

    train_sft.main(["--data", str(data), "--base", str(tiny), "--out", str(run), "--device", "cpu",
                    "--max-steps", "2", "--batch", "1", "--grad-accum", "1", "--lora-r", "4", "--lora-alpha", "8"])
    saved = json.loads((run / "leanwatch.json").read_text())
    assert saved["style"] == style and saved["coords"] == "abs" and saved["max_pixels"] == max_pixels
    assert saved["base_model"] == str(tiny)
    assert (run / "adapter_config.json").exists() and (run / "adapter_model.safetensors").exists()

    results = evaluate.main(["--data", str(data), "--split", "val", "--adapter", str(run), "--device", "cpu",
                             "--max-new-tokens", "6"])
    assert results and all(r["task"] in ("actions", "grounding") for r in results)

    # The agent picks up the trained settings from the adapter folder...
    from leanwatch.__main__ import main as cli  # noqa: F401  (import check)
    from leanwatch.brains import resolve_model_id, trained_settings
    from leanwatch.local_brain import LocalModel
    from leanwatch.local_prompts import build_messages, computer_prompt

    settings = trained_settings(str(run))
    assert resolve_model_id(None, str(run)) == str(tiny)
    model = LocalModel(resolve_model_id(None, str(run)), adapter=str(run), device="cpu",
                       max_pixels=settings["max_pixels"], max_new_tokens=4)
    # ...and sees screenshots at exactly the size the training coordinates were written for.
    messages, turns = build_messages(computer_prompt("Turn on dark mode", "English", settings["style"]), [], 3)
    ans = model.answer(messages, [Image.open(practice / "images" / "e00000_s00.png").convert("RGB")])
    space = CoordSpace("abs", W, H, min_pixels=meta["min_pixels"], max_pixels=max_pixels)
    assert (ans.image_w, ans.image_h) == space.model_size
    # The processor saved with the adapter already carries the cap, even without an override.
    plain = LocalModel(str(tiny), adapter=str(run), device="cpu", max_new_tokens=1)
    again = plain.answer(messages, [Image.new("RGB", (W, H))])
    assert (again.image_w, again.image_h) == space.model_size


def test_cli_reads_settings_from_adapter(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "some/base"}))
    (run / "leanwatch.json").write_text(json.dumps({"base_model": "some/base", "coords": "rel1000", "style": "point",
                                                    "max_pixels": 12345, "history_images": 2}))
    seen = []
    import leanwatch.__main__ as entry
    import leanwatch.server as server

    monkeypatch.setattr(server, "serve", lambda cfg: seen.append(cfg))
    entry.main(["--adapter", str(run), "--no-bubble", "--no-browser"])
    cfg = seen[-1]
    assert (cfg.local_model, cfg.coords, cfg.style, cfg.local_max_pixels, cfg.history_images) == \
        ("some/base", "rel1000", "point", 12345, 2)

    # Flags still win over the saved settings.
    entry.main(["--adapter", str(run), "--no-bubble", "--no-browser", "--style", "box", "--max-pixels", "999999", "--model", "other/model"])
    cfg = seen[-1]
    assert (cfg.local_model, cfg.style, cfg.local_max_pixels, cfg.coords) == ("other/model", "box", 999999, "rel1000")

    # A merged model folder carries the same file.
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "leanwatch.json").write_text(json.dumps({"coords": "abs", "style": "point", "max_pixels": 777}))
    entry.main(["--model", str(merged), "--no-bubble", "--no-browser"])
    cfg = seen[-1]
    assert (cfg.local_model, cfg.style, cfg.local_max_pixels) == (str(merged), "point", 777)
