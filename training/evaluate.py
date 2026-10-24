"""Measure how well a model clicks the right thing, before and after fine-tuning.

For every example in a split from build_dataset.py the model answers once, and we check:
- grounding examples: is the click inside the target element's box?
- action examples:    is the action type right, and for clicks/drags is the point inside the box?
Reports accuracy per task, plus answers that could not be parsed.

    python training/evaluate.py --data data/sft --split val --model ByteDance-Seed/UI-TARS-1.5-7B
    python training/evaluate.py --data data/sft --split val --adapter runs/uitars-leanwatch
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import leanwatch_paths  # noqa: F401
from leanwatch.actions import ParseError, parse_response

# Names that mean the same action.
SAME = {"left_single": "click", "press": "hotkey", "keydown": "hotkey", "call_user": "ask_user", "move_to": "mouse_move"}


def canonical(name: str) -> str:
    return SAME.get(name, name)


def score(example: dict, answer_text: str, model_w: int, model_h: int, coords: str) -> dict:
    meta = example["meta"]
    out = {"task": example["task"], "parsed": False, "type_ok": False, "point_ok": None}
    try:
        parsed = parse_response(answer_text)
    except ParseError:
        return out
    out["parsed"] = True
    a = parsed.actions[0]
    # Compare with the exact answer the model was trained on, so a right click (right_single) or a
    # double click (left_double) only counts when the model chose that same kind of click.
    target = parse_response(example["completion"][0]["content"][0]["text"]).actions[0].name
    out["type_ok"] = canonical(a.name) == canonical(target)
    if "bbox" in meta and a.point is not None:
        sx, sy = (meta["width"] / 1000, meta["height"] / 1000) if coords == "rel1000" else (meta["width"] / model_w, meta["height"] / model_h)
        x, y = a.point[0] * sx, a.point[1] * sy
        x0, y0, x1, y1 = meta["bbox"]
        out["point_ok"] = bool(x0 <= x <= x1 and y0 <= y <= y1)
    elif "bbox" in meta:
        out["point_ok"] = False
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--model", default=None, help="Hugging Face id or folder (default: the adapter's base, else UI-TARS-1.5-7B)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tasks", default="actions,grounding")
    p.add_argument("--limit", type=int, default=0, help="evaluate at most this many examples, spread over the split (0 = all)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--4bit", dest="four_bit", action="store_true")
    p.add_argument("--gpu-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                   help="weights on an NVIDIA GPU. auto: bf16 where native, else fp16. float32 if answers come out as '!!!!'")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--save", type=Path, help="write per-example results as JSONL")
    a = p.parse_args(argv)

    from PIL import Image

    from leanwatch.brains import resolve_model_id
    from leanwatch.local_brain import LocalModel

    meta = json.loads((a.data / "meta.json").read_text())
    model_id = resolve_model_id(a.model, a.adapter)
    model = LocalModel(model_id, adapter=a.adapter, device=a.device, load_in_4bit=a.four_bit,
                       max_pixels=meta["max_pixels"], max_new_tokens=a.max_new_tokens, gpu_dtype=a.gpu_dtype)
    tasks = set(a.tasks.split(","))
    rows = [json.loads(line) for line in open(a.data / f"{a.split}.jsonl")]
    rows = [r for r in rows if r["task"] in tasks]
    if a.limit and len(rows) > a.limit:
        # Spread the sample over the whole split: the file lists every action example before the grounding ones.
        rows = [rows[int(i * len(rows) / a.limit)] for i in range(a.limit)]

    results, save = [], open(a.save, "w") if a.save else None
    for i, ex in enumerate(rows, 1):
        images = [Image.open(path).convert("RGB") for path in ex["images"]]
        t0 = time.monotonic()
        ans = model.answer(ex["prompt"], images)
        r = score(ex, ans.text, ans.image_w, ans.image_h, meta["coords"])
        r["seconds"] = round(time.monotonic() - t0, 2)
        results.append(r)
        if save:
            save.write(json.dumps({**r, "answer": ans.text, "target": ex["completion"][0]["content"][0]["text"], "meta": ex["meta"],
                                   "image": ex["images"][-1], "image_size": [ans.image_w, ans.image_h]}) + "\n")
        if i % 20 == 0:
            print(f"{i}/{len(rows)}")
    if save:
        save.close()

    print(f"\nModel: {model_id}{' + ' + a.adapter if a.adapter else ''}   split: {a.split}   examples: {len(results)}")
    for task in sorted(tasks):
        rs = [r for r in results if r["task"] == task]
        if not rs:
            continue
        pts = [r["point_ok"] for r in rs if r["point_ok"] is not None]
        print(f"  {task:10s} parsed {sum(r['parsed'] for r in rs) / len(rs):6.1%}   right action {sum(r['type_ok'] for r in rs) / len(rs):6.1%}"
              + (f"   click inside target {sum(pts) / len(pts):6.1%} (of {len(pts)})" if pts else ""))
    by_action = Counter(ex["meta"].get("action", "click") for ex in rows)
    print(f"  action mix: {dict(by_action)}")
    if results:
        info = getattr(model, "info", {})
        print(f"  speed: {sum(r['seconds'] for r in results) / len(results):.1f} s per answer"
              + (f" on {info['device']} ({info['dtype']})" if info else ""))
    return results


if __name__ == "__main__":
    main()
