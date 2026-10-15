"""Import public GUI datasets into LeanWatch's raw format, so build_dataset.py can mix
them with your practice recordings.

Every source is converted to the same files record_practice.mjs writes:
    <out>/images/...                 screenshots
    <out>/episodes.jsonl             tasks: {"id", "instruction", "width", "height", "steps": [
                                        {"image", "thought", "action": {...in image pixels...}, "bbox"?}]}
    <out>/grounding.jsonl            {"image", "label", "role", "bbox"?: [x0,y0,x1,y1], "point"?: [x,y], "width", "height"}
Coordinates are always pixels of the saved image, whatever the source used.

Sources (column layouts from the authors' loaders; run with --limit 20 first and look at
the printed sample to confirm the files you downloaded match):

  aguvis      smolagents/aguvis-stage-1 or -2 (load_dataset; normalized [0,1] actions)
  agentnet    xlangai/AgentNet trajectories (normalized pyautogui code). Desktop: Windows, macOS, Ubuntu. MIT.
  seeclick    OS-Atlas desktop / SeeClick-style JSON: img_filename + bbox [x1,y1,x2,y2] normalized
  showui      showlab/ShowUI-desktop metadata/hf_train.json (element list, normalized)
  groundcua   ServiceNow/GroundCUA instruction files (bbox [x1,y1,x2,y2] pixels). MIT.
  screenspot  ScreenSpot-v2 ([x,y,w,h] pixels) or ScreenSpot-Pro ([x1,y1,x2,y2] pixels) - for evaluation

Examples
    python training/import_hf.py aguvis --dataset smolagents/aguvis-stage-1 --config omniact --out data/aguvis-omniact --limit 5000
    hf download xlangai/AgentNet --repo-type dataset --local-dir raw/agentnet
    python training/import_hf.py agentnet --path raw/agentnet --out data/agentnet --limit 2000
    python training/import_hf.py screenspot --path raw/ScreenSpot-Pro --out data/screenspot-pro --limit 0
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path
from typing import Iterator

# ----------------------------------------------------------------------------
# Action code parsing (pyautogui strings and the AGUVIS action language)
# ----------------------------------------------------------------------------
PYAUTOGUI_KEYS = {"enter": "enter", "return": "enter", "esc": "esc", "escape": "esc", "command": "cmd", "win": "win",
                  "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "option": "alt", "shift": "shift"}


def _calls(code: str) -> list[tuple[str, list, dict]]:
    """Every function call in a snippet: (name, positional args, keyword args). Unknown values become None."""
    out = []
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name is None:
                continue
            try:
                args = [ast.literal_eval(a) for a in node.args]
                kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
            except (ValueError, SyntaxError):
                continue
            out.append((name, args, kwargs))
    return out  # ast.walk visits statements first, so calls come out in source order


def _xy(args, kwargs, w, h, normalized=True):
    x = kwargs.get("x", args[0] if len(args) > 0 else None)
    y = kwargs.get("y", args[1] if len(args) > 1 else None)
    if x is None or y is None:
        return None
    if normalized and 0 <= x <= 1 and 0 <= y <= 1:
        return round(x * w, 1), round(y * h, 1)
    return float(x), float(y)


def pyautogui_to_actions(code: str, w: int, h: int, last_point=None) -> list[dict] | None:
    """Convert one step of (normalized) pyautogui code to LeanWatch actions. None if unsupported."""
    calls = _calls(code)
    if not calls:
        return None
    actions, pending_move = [], None
    for name, args, kw in calls:
        n = name.lower()
        if n in ("click", "leftclick", "doubleclick", "tripleclick", "rightclick"):
            p = _xy(args, kw, w, h) or pending_move or last_point
            if p is None:
                return None
            clicks = {"doubleclick": 2, "tripleclick": 3}.get(n, int(kw.get("clicks", 1) or 1))
            a = {"type": "click", "x": p[0], "y": p[1], "clicks": clicks, "button": "right" if n == "rightclick" or kw.get("button") == "right" else "left"}
            actions.append(a)
            pending_move = None
        elif n == "moveto":
            pending_move = _xy(args, kw, w, h)
        elif n == "dragto":
            end = _xy(args, kw, w, h)
            start = pending_move or last_point
            if not (start and end):
                return None
            actions.append({"type": "drag", "x1": start[0], "y1": start[1], "x2": end[0], "y2": end[1]})
            pending_move = None
        elif n == "scroll" or n == "hscroll":
            amount = args[0] if args else kw.get("clicks", 0)
            p = _xy(args[1:], kw, w, h) or pending_move or last_point or (w / 2, h / 2)
            direction = ("up" if amount > 0 else "down") if n == "scroll" else ("right" if amount > 0 else "left")
            actions.append({"type": "scroll", "x": p[0], "y": p[1], "direction": direction})
        elif n in ("write", "typewrite"):
            text = kw.get("message", args[0] if args else "")
            if isinstance(text, list):
                text = "".join(text)
            actions.append({"type": "type", "text": str(text)})
        elif n == "press":
            keys = kw.get("keys", args[0] if args else "")
            keys = keys if isinstance(keys, list) else [keys]
            actions.append({"type": "hotkey", "keys": [PYAUTOGUI_KEYS.get(str(k).lower(), str(k).lower()) for k in keys]})
        elif n == "hotkey":
            keys = [PYAUTOGUI_KEYS.get(str(k).lower(), str(k).lower()) for k in args]
            actions.append({"type": "hotkey", "keys": keys})
        elif n == "terminate":
            actions.append({"type": "finished", "text": str(kw.get("status", args[0] if args else "success"))})
        elif n in ("wait", "sleep"):
            actions.append({"type": "wait"})
        else:
            return None  # e.g. mouseDown, keyDown, open_app: not in LeanWatch's action space
    return actions or None


def aguvis_to_actions(code: str, w: int, h: int) -> list[dict] | None:
    """AGUVIS action language: click(x=0.1, y=0.2), drag(from_coord=[..], to_coord=[..]), type(text=..), press(keys=[..]), scroll(direction=.., amount=..)."""
    m = re.search(r"<code>(.*?)</code>", code, re.DOTALL)
    calls = _calls(m.group(1) if m else code)
    if not calls:
        return None
    out = []
    for name, args, kw in calls:
        n = name.lower()
        if n in ("click", "double_click", "right_click"):
            p = _xy(args, kw, w, h)
            if not p:
                return None
            out.append({"type": "click", "x": p[0], "y": p[1], "clicks": 2 if n == "double_click" else 1, "button": "right" if n == "right_click" else "left"})
        elif n == "drag":
            a, b = kw.get("from_coord"), kw.get("to_coord")
            if not (a and b):
                return None
            out.append({"type": "drag", "x1": a[0] * w, "y1": a[1] * h, "x2": b[0] * w, "y2": b[1] * h})
        elif n == "type":
            out.append({"type": "type", "text": str(kw.get("text", args[0] if args else ""))})
        elif n == "press":
            keys = kw.get("keys", args[0] if args else [])
            keys = keys if isinstance(keys, list) else [keys]
            out.append({"type": "hotkey", "keys": [PYAUTOGUI_KEYS.get(str(k).lower(), str(k).lower()) for k in keys]})
        elif n == "scroll":
            out.append({"type": "scroll", "x": w / 2, "y": h / 2, "direction": str(kw.get("direction", "down"))})
        elif n in ("final_answer", "terminate"):
            out.append({"type": "finished", "text": str(kw.get("answer", args[0] if args else ""))})
        elif n == "wait":
            out.append({"type": "wait"})
        else:
            return None  # long_press, swipe, open_app, navigate_home: mobile only
    return out or None


def think_of(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ----------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------
class Writer:
    def __init__(self, out: Path, image_format: str):
        self.out = out
        (out / "images").mkdir(parents=True, exist_ok=True)
        self.ep = open(out / "episodes.jsonl", "w")
        self.gr = open(out / "grounding.jsonl", "w")
        self.fmt = image_format
        self.n_images = self.n_episodes = self.n_steps = self.n_grounding = 0
        self.skipped: dict[str, int] = {}

    def skip(self, why: str):
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def image(self, src, name: str) -> tuple[str, int, int]:
        """Save a PIL image or copy a file; returns (relative path, width, height)."""
        from PIL import Image

        if isinstance(src, (str, Path)):
            src = Path(src)
            rel = f"images/{name}{src.suffix.lower()}"
            shutil.copyfile(src, self.out / rel)
            with Image.open(src) as im:
                w, h = im.size
        else:
            rel = f"images/{name}.{self.fmt}"
            src.convert("RGB").save(self.out / rel, quality=95) if self.fmt == "jpg" else src.save(self.out / rel)
            w, h = src.size
        self.n_images += 1
        return rel, w, h

    def episode(self, rec: dict):
        self.ep.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_episodes += 1
        self.n_steps += len(rec["steps"])

    def grounding(self, rec: dict):
        self.gr.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_grounding += 1

    def close(self, source: str, args: dict):
        self.ep.close()
        self.gr.close()
        meta = {"source": source, "args": args, "images": self.n_images, "episodes": self.n_episodes,
                "steps": self.n_steps, "grounding": self.n_grounding, "skipped": self.skipped}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def in_range(box, w, h) -> bool:
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= w + 1 and 0 <= y0 < y1 <= h + 1


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
def import_aguvis(a, wr: Writer):
    from datasets import load_dataset

    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True)
    for i, row in enumerate(ds):
        if a.limit and wr.n_images >= a.limit:
            break
        images = row.get("images") or []
        texts = row.get("texts") or []
        if not images or not texts:
            wr.skip("no image or text")
            continue
        rel, w, h = wr.image(images[0], f"aguvis_{i:07d}")
        stage2 = "<code>" in json.dumps(texts)
        if not stage2:  # grounding: each turn is (instruction -> click)
            for t in texts:
                acts = aguvis_to_actions(t.get("assistant", ""), w, h)
                if not acts or acts[0]["type"] != "click":
                    wr.skip("grounding answer is not a click")
                    continue
                wr.grounding({"image": rel, "label": t.get("user", "").strip(), "role": "", "point": [acts[0]["x"], acts[0]["y"]],
                              "width": w, "height": h, "instruction": t.get("user", "").strip()})
        else:  # one step of a trajectory
            t = texts[-1]
            acts = aguvis_to_actions(t.get("assistant", ""), w, h)
            if not acts:
                wr.skip("unsupported action")
                continue
            instruction = re.sub(r"\s+", " ", t.get("user", "")).strip()
            steps = [{"image": rel, "thought": think_of(t.get("assistant", "")), "action": acts[0]}]
            wr.episode({"id": f"aguvis_{i:07d}", "instruction": instruction, "width": w, "height": h, "steps": steps,
                        "source": f"{a.dataset}/{a.config}"})


def import_agentnet(a, wr: Writer):
    root = Path(a.path)
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    files = [f for f in files if "meta_data" not in f.name] or files
    base = root if root.is_dir() else root.parent
    images = Path(a.images) if a.images else base

    def find_image(task_id: str, name: str) -> Path | None:
        # Screenshot names (0.png, 1.png, ...) repeat across tasks, so look in the task's own folder first.
        name = Path(name).name
        for c in (images / task_id / name, images / "images" / task_id / name, base / task_id / name,
                  images / f"{task_id}_{name}", images / name):
            if c.exists():
                return c
        return None

    for f in files:
        for line in open(f):
            if a.limit and wr.n_episodes >= a.limit:
                return
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if "traj" not in t:
                continue
            steps, last = [], None
            ok = True
            for st in t["traj"]:
                v = st.get("value", {})
                img = find_image(str(t["task_id"]), st.get("image", ""))
                if img is None:
                    ok = False
                    wr.skip("image file not found")
                    break
                rel, w, h = wr.image(img, f"{t['task_id']}_{st.get('index', len(steps)):03d}")
                acts = pyautogui_to_actions(v.get("code", ""), w, h, last)
                if not acts:
                    ok = False
                    wr.skip("unsupported action")
                    break
                thought = (v.get("thought") or v.get("action") or "").strip()
                for j, act in enumerate(acts):
                    steps.append({"image": rel, "thought": thought if j == 0 else "", "action": act})
                    if "x" in act:
                        last = (act["x"], act["y"])
            if not ok or not steps:
                continue
            if steps[-1]["action"]["type"] != "finished":
                steps.append({"image": steps[-1]["image"], "thought": "The task is complete.", "action": {"type": "finished", "text": "Done."}})
            instruction = t.get("natural_language_task") or t.get("instruction") or t.get("actual_task") or ""
            wr.episode({"id": str(t["task_id"]), "instruction": instruction, "width": w, "height": h, "steps": steps, "source": "xlangai/AgentNet"})


def _seeclick_rows(path: Path) -> Iterator[dict]:
    data = json.loads(path.read_text())
    for row in data if isinstance(data, list) else data.get("data", []):
        if "elements" in row:
            for el in row["elements"]:
                yield {"img_filename": row["img_filename"], **el}
        else:
            yield row


def import_seeclick(a, wr: Writer):
    root = Path(a.path)
    images_dir = Path(a.images) if a.images else root.parent
    copied: dict[str, tuple[str, int, int]] = {}
    for ann in ([root] if root.is_file() else sorted(root.glob("*.json"))):
        for row in _seeclick_rows(ann):
            if a.limit and wr.n_grounding >= a.limit:
                return
            src = images_dir / row["img_filename"]
            if not src.exists():
                wr.skip("image file not found")
                continue
            if row["img_filename"] not in copied:
                copied[row["img_filename"]] = wr.image(src, f"seeclick_{len(copied):07d}")
            rel, w, h = copied[row["img_filename"]]
            x0, y0, x1, y1 = row["bbox"]
            box = [x0 * w, y0 * h, x1 * w, y1 * h] if max(row["bbox"]) <= 1 else [x0, y0, x1, y1]
            if not in_range(box, w, h):
                wr.skip("box outside image")
                continue
            wr.grounding({"image": rel, "label": row.get("instruction", "").strip(), "role": row.get("data_type", ""),
                          "bbox": box, "width": w, "height": h})


def import_showui(a, wr: Writer):
    root = Path(a.path)
    meta = json.loads((root / "metadata" / "hf_train.json").read_text()) if root.is_dir() else json.loads(root.read_text())
    images_dir = Path(a.images) if a.images else (root / "images")
    for k, row in enumerate(meta):
        if a.limit and wr.n_grounding >= a.limit:
            return
        src = images_dir / Path(row["img_url"]).name
        if not src.exists():
            wr.skip("image file not found")
            continue
        rel, w, h = wr.image(src, f"showui_{k:07d}")
        for el in row.get("element", []):
            x0, y0, x1, y1 = el["bbox"]
            box = [x0 * w, y0 * h, x1 * w, y1 * h]
            if not in_range(box, w, h):
                wr.skip("box outside image")
                continue
            wr.grounding({"image": rel, "label": el.get("instruction", "").strip(), "role": el.get("data_type", ""),
                          "bbox": box, "width": w, "height": h})


def import_groundcua(a, wr: Writer):
    root = Path(a.path)
    images_dir = Path(a.images) if a.images else root / "images"
    files = [root] if root.is_file() else sorted(root.rglob("*instructions*.json"))
    copied: dict[str, tuple[str, int, int]] = {}
    for f in files:
        rows = json.loads(f.read_text())
        for row in rows if isinstance(rows, list) else rows.get("data", []):
            if a.limit and wr.n_grounding >= a.limit:
                return
            if "bbox" not in row:
                wr.skip("no bbox (point-only file)")
                continue
            candidates = [images_dir / row.get("platform", "") / f"{row['id']}.png", images_dir / f"{row['id']}.png"]
            src = next((c for c in candidates if c.exists()), None)
            if src is None:
                wr.skip("image file not found")
                continue
            key = str(src)
            if key not in copied:
                copied[key] = wr.image(src, f"groundcua_{len(copied):07d}")
            rel, w, h = copied[key]
            box = [float(v) for v in row["bbox"]]
            if not in_range(box, w, h):
                wr.skip("box outside image")
                continue
            instruction = (row.get("instruction") or "").strip()
            wr.grounding({"image": rel, "label": (row.get("text") or instruction).strip(), "instruction": instruction or None,
                          "role": (row.get("category") or "").lower(), "bbox": box, "width": w, "height": h,
                          "platform": row.get("platform", "")})


def import_screenspot(a, wr: Writer):
    root = Path(a.path)
    ann_files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    copied: dict[str, tuple[str, int, int]] = {}
    for f in ann_files:
        rows = json.loads(f.read_text())
        if not isinstance(rows, list):
            continue
        for row in rows:
            if a.limit and wr.n_grounding >= a.limit:
                return
            name = row.get("img_filename") or row.get("image")
            if not name:
                continue
            candidates = ([Path(a.images) / name] if a.images else []) + [
                f.parent / name, f.parent.parent / "images" / name, root / "images" / name, root / name]
            src = next((c for c in candidates if c.exists()), None)
            if src is None:
                wr.skip("image file not found")
                continue
            if str(src) not in copied:
                copied[str(src)] = wr.image(src, f"screenspot_{len(copied):07d}")
            rel, w, h = copied[str(src)]
            b = [float(v) for v in row["bbox"]]
            # ScreenSpot-Pro gives img_size and [x1,y1,x2,y2]; ScreenSpot(-v2) gives [x,y,w,h].
            box = b if "img_size" in row else [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            if max(box) <= 1:
                box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
            if not in_range(box, w, h):
                wr.skip("box outside image")
                continue
            instruction = row.get("instruction", "").strip()
            wr.grounding({"image": rel, "label": instruction, "instruction": instruction or None,
                          "role": row.get("data_type") or row.get("ui_type", ""),
                          "bbox": box, "width": w, "height": h, "platform": row.get("platform") or row.get("application", "")})


SOURCES = {"aguvis": import_aguvis, "agentnet": import_agentnet, "seeclick": import_seeclick,
           "showui": import_showui, "groundcua": import_groundcua, "screenspot": import_screenspot}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--path", help="downloaded files (agentnet, seeclick, showui, groundcua, screenspot)")
    p.add_argument("--images", help="folder with the screenshots, if not next to the annotations")
    p.add_argument("--dataset", default="smolagents/aguvis-stage-1", help="aguvis: Hub dataset id")
    p.add_argument("--config", default=None, help="aguvis: subset, e.g. omniact, ui_refexp")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=1000, help="stop after this many images/episodes/samples (0 = all)")
    p.add_argument("--image-format", default="png", choices=["png", "jpg"])
    a = p.parse_args(argv)
    if a.source != "aguvis" and not a.path:
        p.error("--path is required for this source")

    wr = Writer(a.out, a.image_format)
    SOURCES[a.source](a, wr)
    meta = wr.close(a.source, {k: str(v) for k, v in vars(a).items()})
    print(json.dumps(meta, indent=2))
    sample = a.out / ("episodes.jsonl" if meta["episodes"] else "grounding.jsonl")
    if sample.exists() and sample.stat().st_size:
        print("\nFirst record (check that the coordinates land on the right element):")
        print(open(sample).readline()[:800])
    if meta["episodes"] == 0 and meta["grounding"] == 0:
        print("\nNothing was imported. Check --path/--images and that the files match this source's layout (see --help).")


if __name__ == "__main__":
    main()
