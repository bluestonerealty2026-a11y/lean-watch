"""Turn recorded episodes into training examples for a vision-language GUI model.

Input: folders written by record_practice.mjs (episodes.jsonl, grounding.jsonl,
images/). Output: train.jsonl and val.jsonl in TRL's conversational
prompt-completion format, plus meta.json:

    {"images": ["/abs/path/a.png", ...],
     "prompt": [messages, with {"type": "image"} placeholders in order],
     "completion": [{"role": "assistant", "content": [{"type": "text", "text": "Thought: ...\\nAction: ..."}]}],
     "task": "actions" | "grounding",
     "meta": {"episode": ..., "bbox": [x0, y0, x1, y1], "width": W, "height": H, ...}}

Two tasks, as in the GUI-agent literature (e.g. Aguvis, UI-TARS):
- grounding: "click the Dark mode switch" -> one click. Teaches the model where things are.
- actions:   instruction + recent screenshots + its earlier answers -> next Thought and Action.
             Teaches it to plan, act and explain like a tutor.

The conversation is built with the same function the agent uses at run time
(leanwatch.local_prompts.build_messages), so training and inference match.
Coordinates are written in the model's own space; see --coords and --max-pixels.

    python training/build_dataset.py --practice data/practice --out data/sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import leanwatch_paths  # noqa: F401  (adds agent/ to sys.path)
from leanwatch.actions import MAX_PIXELS, MIN_PIXELS, CoordSpace, format_action, format_response
from leanwatch.local_prompts import APPROVED_NOTE, build_messages, computer_prompt, grounding_prompt

DEFAULT_MAX_PIXELS = 1280 * 28 * 28   # about 1,280 image tokens per screenshot: fits LoRA on a 24 GB GPU

ROLE_PHRASES = {
    "switch": "the “{}” switch", "slider": "the “{}” slider", "menuitem": "the “{}” menu item",
    "option": "“{}”", "input": "the “{}” box", "button": "the “{}” button",
}


def grounding_instruction(label: str, role: str, rng: random.Random) -> str:
    target = ROLE_PHRASES.get(role, "“{}”").format(label)
    return rng.choice(["Click {}.", "Find {} and click it.", "Where is {}? Click it."]).format(target)


def group_of(g: dict) -> str:
    """Grounding samples from one recorded episode (e00012_s03.png) stay together; imported ones group by screenshot."""
    if g.get("episode"):
        return str(g["episode"])
    stem = Path(g["image"]).stem
    m = re.match(r"(e\d+)_s\d+$", stem)
    return m.group(1) if m else stem


def split_of(key: str, val_fraction: float) -> str:
    """Stable split by episode, so screens from one episode never land in both sets."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def build(practice_dirs: list[Path], out: Path, coords: str, max_pixels: int, min_pixels: int,
          history_images: int, val_fraction: float, tasks: set[str], style: str, language: str, seed: int) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": open(out / "train.jsonl", "w"), "val": open(out / "val.jsonl", "w")}
    counts = {"train": {"actions": 0, "grounding": 0}, "val": {"actions": 0, "grounding": 0}}

    def write(split, record):
        files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split][record["task"]] += 1

    for root in practice_dirs:
        root = root.resolve()
        tag = root.name
        if "actions" in tasks:
            for line in open(root / "episodes.jsonl"):
                ep = json.loads(line)
                split = split_of(f"{tag}/{ep['id']}", val_fraction)
                space = CoordSpace(coords, ep["width"], ep["height"], min_pixels=min_pixels, max_pixels=max_pixels)
                prompt_text = computer_prompt(ep["instruction"], language, style)
                past: list[dict] = []
                for k, step in enumerate(ep["steps"]):
                    target = format_response(step.get("thought", ""), format_action(step["action"], space, style))
                    messages, image_turns = build_messages(prompt_text, past, history_images)
                    images = [str(root / ep["steps"][t]["image"]) for t in image_turns]
                    meta = {"source": tag, "episode": ep["id"], "step": k, "action": step["action"]["type"],
                            "width": ep["width"], "height": ep["height"], "instruction": ep["instruction"]}
                    if step.get("bbox"):
                        meta["bbox"] = step["bbox"]
                    write(split, {"images": images, "prompt": messages,
                                  "completion": [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
                                  "task": "actions", "meta": meta})
                    note = APPROVED_NOTE if step["action"]["type"] == "confirm" else None
                    past.append({"response": target, "note": note})
        if "grounding" in tasks and (root / "grounding.jsonl").exists():
            for line in open(root / "grounding.jsonl"):
                g = json.loads(line)
                episode = group_of(g)
                split = split_of(f"{tag}/{episode}", val_fraction)
                space = CoordSpace(coords, g["width"], g["height"], min_pixels=min_pixels, max_pixels=max_pixels)
                if g.get("bbox"):
                    x0, y0, x1, y1 = g["bbox"]
                    px, py = (x0 + x1) / 2, (y0 + y1) / 2
                else:  # point-only sources (e.g. AGUVIS)
                    px, py = g["point"]
                click = {"type": "click", "x": px, "y": py, "clicks": 1}
                target = format_response("", format_action(click, space, style))
                # Sources that already give a full instruction keep it; bare element names get a short sentence.
                instruction = g.get("instruction") or grounding_instruction(g["label"], g.get("role", ""), rng)
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": grounding_prompt(instruction, style)},
                    {"type": "image"}]}]
                write(split, {"images": [str(root / g["image"])], "prompt": messages,
                              "completion": [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
                              "task": "grounding",
                              "meta": {"source": tag, "episode": episode, "label": g["label"], "width": g["width"],
                                       "height": g["height"], **({"bbox": g["bbox"]} if g.get("bbox") else {"point": g["point"]})}})
    for f in files.values():
        f.close()
    meta = {"coords": coords, "max_pixels": max_pixels, "min_pixels": min_pixels, "history_images": history_images,
            "style": style, "language": language, "counts": counts, "sources": [str(p) for p in practice_dirs]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--practice", nargs="+", type=Path, required=True, help="folders from record_practice.mjs")
    p.add_argument("--out", type=Path, default=Path("data/sft"))
    p.add_argument("--tasks", default="actions,grounding", help="comma list: actions, grounding")
    p.add_argument("--coords", default="abs", choices=["abs", "rel1000"],
                   help="abs: pixels of the resized image (UI-TARS-1.5, Qwen2.5-VL). rel1000: 0-1000 (Qwen3-VL style)")
    p.add_argument("--style", default="box", choices=["box", "point"],
                   help="box: start_box='<|box_start|>(x,y)<|box_end|>' (UI-TARS-1.5). point: point='<point>x y</point>'")
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                   help=f"image size cap used by the model's processor (default {DEFAULT_MAX_PIXELS}); training and the agent must use the same")
    p.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    p.add_argument("--history-images", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.max_pixels > MAX_PIXELS:
        p.error(f"--max-pixels must be at most {MAX_PIXELS}")
    meta = build(a.practice, a.out, a.coords, a.max_pixels, a.min_pixels, a.history_images, a.val_fraction,
                 set(a.tasks.split(",")), a.style, a.language, a.seed)
    c = meta["counts"]
    print(f"train: {c['train']['actions']} action steps, {c['train']['grounding']} grounding  |  "
          f"val: {c['val']['actions']} action steps, {c['val']['grounding']} grounding  ->  {a.out}")


if __name__ == "__main__":
    main()
