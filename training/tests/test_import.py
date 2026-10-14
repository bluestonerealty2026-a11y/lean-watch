import json
from types import SimpleNamespace as NS

import pytest
from PIL import Image

import import_hf
from import_hf import aguvis_to_actions, pyautogui_to_actions, think_of


def img(path, size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def args(**kw):
    base = dict(path=None, images=None, dataset="smolagents/aguvis-stage-1", config=None, split="train", limit=0, image_format="png")
    base.update(kw)
    return NS(**base)


def run(source, out, **kw):
    wr = import_hf.Writer(out, "png")
    import_hf.SOURCES[source](args(**kw), wr)
    return wr.close(source, {})


def lines(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


@pytest.mark.parametrize("code,expect", [
    ("pyautogui.click(x=0.5, y=0.25)", {"type": "click", "x": 100, "y": 25, "clicks": 1, "button": "left"}),
    ("pyautogui.doubleClick(x=0.1, y=0.1)", {"type": "click", "clicks": 2}),
    ("pyautogui.rightClick(x=0.1, y=0.1)", {"type": "click", "button": "right"}),
    ("pyautogui.moveTo(x=0.1, y=0.2)\npyautogui.dragTo(x=0.9, y=0.2, button='left')", {"type": "drag", "x1": 20, "x2": 180}),
    ("pyautogui.scroll(-3)", {"type": "scroll", "direction": "down"}),
    ("pyautogui.write(message='hello')", {"type": "type", "text": "hello"}),
    ("pyautogui.press('enter')", {"type": "hotkey", "keys": ["enter"]}),
    ("pyautogui.hotkey('ctrl', 'c')", {"type": "hotkey", "keys": ["ctrl", "c"]}),
    ("computer.terminate(status='success')", {"type": "finished", "text": "success"}),
])
def test_pyautogui(code, expect):
    a = pyautogui_to_actions(code, 200, 100)[0]
    for k, v in expect.items():
        assert a[k] == pytest.approx(v) if isinstance(v, (int, float)) else a[k] == v


def test_pyautogui_unsupported():
    assert pyautogui_to_actions("pyautogui.mouseDown()", 200, 100) is None
    assert pyautogui_to_actions("not code (", 200, 100) is None


def test_aguvis_language():
    assert aguvis_to_actions("click(x=0.5, y=0.25)", 200, 100)[0] == {"type": "click", "x": 100, "y": 25, "clicks": 1, "button": "left"}
    d = aguvis_to_actions("drag(from_coord=[0.1, 0.1], to_coord=[0.5, 0.5])", 200, 100)[0]
    assert d["type"] == "drag" and d["x2"] == pytest.approx(100)
    s2 = "<think>\nOpen the File menu.\n</think>\n<code>\nclick(x=0.41, y=0.178)\n</code>"
    assert think_of(s2) == "Open the File menu." and aguvis_to_actions(s2, 200, 100)[0]["type"] == "click"
    assert aguvis_to_actions("press(keys=['ctrl', 's'])", 200, 100)[0]["keys"] == ["ctrl", "s"]
    assert aguvis_to_actions("long_press(x=0.1, y=0.1)", 200, 100) is None


def test_agentnet_with_repeated_image_names(tmp_path):
    raw = tmp_path / "raw"
    for task in ("t1", "t2"):
        for n in range(3):
            img(raw / task / f"{n}.png")
    trajs = [
        {"task_id": "t1", "natural_language_task": "Open the File menu and save", "traj": [
            {"index": 0, "image": "0.png", "value": {"code": "pyautogui.click(x=0.5, y=0.5)", "thought": "I open File."}},
            {"index": 1, "image": "1.png", "value": {"code": "pyautogui.hotkey('ctrl', 's')", "thought": "Save."}},
            {"index": 2, "image": "2.png", "value": {"code": "computer.terminate(status='success')", "thought": "Done."}}]},
        {"task_id": "t2", "instruction": "Unsupported", "traj": [
            {"index": 0, "image": "0.png", "value": {"code": "pyautogui.mouseDown()", "thought": "?"}}]},
    ]
    (raw / "data.jsonl").write_text("\n".join(json.dumps(t) for t in trajs))
    meta = run("agentnet", tmp_path / "out", path=str(raw))
    eps = lines(tmp_path / "out" / "episodes.jsonl")
    assert meta["episodes"] == 1 and meta["skipped"] == {"unsupported action": 1}
    s = eps[0]["steps"]
    assert eps[0]["instruction"] == "Open the File menu and save"
    assert s[0]["action"] == {"type": "click", "x": 100, "y": 50, "clicks": 1, "button": "left"}
    assert s[0]["thought"] == "I open File." and s[-1]["action"]["type"] == "finished"


def test_seeclick_flat_and_elements(tmp_path):
    raw = tmp_path / "raw"
    img(raw / "a.png")
    (raw / "ann.json").write_text(json.dumps([
        {"img_filename": "a.png", "bbox": [0.1, 0.2, 0.3, 0.4], "instruction": "Start"},
        {"img_filename": "a.png", "elements": [{"bbox": [0.5, 0.5, 0.6, 0.6], "instruction": "Close", "data_type": "icon"}]},
        {"img_filename": "missing.png", "bbox": [0.1, 0.1, 0.2, 0.2], "instruction": "x"},
    ]))
    meta = run("seeclick", tmp_path / "out", path=str(raw / "ann.json"))
    g = lines(tmp_path / "out" / "grounding.jsonl")
    assert meta["grounding"] == 2 and meta["images"] == 1
    assert g[0]["bbox"] == pytest.approx([20, 20, 60, 40]) and g[1]["label"] == "Close"


def test_showui(tmp_path):
    raw = tmp_path / "raw"
    img(raw / "images" / "s1.png")
    (raw / "metadata").mkdir()
    (raw / "metadata" / "hf_train.json").write_text(json.dumps([
        {"img_url": "s1.png", "img_size": [200, 100], "element": [{"instruction": "Save", "bbox": [0.0, 0.0, 0.5, 0.5], "data_type": "text", "point": [0.25, 0.25]}]}]))
    run("showui", tmp_path / "out", path=str(raw))
    assert lines(tmp_path / "out" / "grounding.jsonl")[0]["bbox"] == pytest.approx([0, 0, 100, 50])


def test_groundcua(tmp_path):
    raw = tmp_path / "raw"
    img(raw / "images" / "Blender" / "abc.png")
    (raw / "direct_instructions.json").write_text(json.dumps([
        {"id": "abc", "platform": "Blender", "bbox": [10, 10, 50, 30], "text": "Render", "category": "Button", "instruction": "Click Render"},
        {"id": "abc", "platform": "Blender", "coordinate": [5, 5], "instruction": "point only"},
    ]))
    meta = run("groundcua", tmp_path / "out", path=str(raw))
    g = lines(tmp_path / "out" / "grounding.jsonl")
    assert meta["grounding"] == 1 and g[0]["bbox"] == [10, 10, 50, 30] and g[0]["role"] == "button"


def test_screenspot_v2_and_pro(tmp_path):
    raw = tmp_path / "raw"
    img(raw / "images" / "v2.png")
    img(raw / "images" / "pro.png", (400, 200))
    (raw / "annotations").mkdir()
    (raw / "annotations" / "v2.json").write_text(json.dumps([{"img_filename": "v2.png", "bbox": [10, 20, 30, 40], "instruction": "OK"}]))
    (raw / "annotations" / "pro.json").write_text(json.dumps([{"img_filename": "pro.png", "bbox": [100, 50, 140, 70], "img_size": [400, 200], "instruction": "Layers"}]))
    run("screenspot", tmp_path / "out", path=str(raw))
    g = {r["label"]: r for r in lines(tmp_path / "out" / "grounding.jsonl")}
    assert g["OK"]["bbox"] == [10, 20, 40, 60]          # [x, y, w, h] -> corners
    assert g["Layers"]["bbox"] == [100, 50, 140, 70]    # already corners


def test_aguvis_from_hub(tmp_path, monkeypatch):
    rows = [
        {"images": [Image.new("RGB", (200, 100))], "texts": [{"user": "Click the search box", "assistant": "click(x=0.5, y=0.1)"}], "source": "omniact"},
        {"images": [Image.new("RGB", (200, 100))], "texts": [{"system": "s", "user": "Book a flight", "assistant": "<think>Open dates.</think>\n<code>\nclick(x=0.2, y=0.3)\n</code>"}], "source": "x"},
    ]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: iter(rows))
    meta = run("aguvis", tmp_path / "out")
    g = lines(tmp_path / "out" / "grounding.jsonl")
    e = lines(tmp_path / "out" / "episodes.jsonl")
    assert meta["grounding"] == 1 and g[0]["point"] == [100, 10] and g[0]["instruction"] == "Click the search box"
    assert meta["episodes"] == 1 and e[0]["steps"][0]["thought"] == "Open dates."
