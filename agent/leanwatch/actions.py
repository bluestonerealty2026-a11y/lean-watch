"""The action language spoken by open GUI models (UI-TARS style).

A model answers with a short explanation and one or more actions::

    Thought: Dark mode lives in the Display section, so I click Display.
    Action: click(start_box='<|box_start|>(312,145)<|box_end|>')

This module is shared by the agent (to understand the model) and by the
training scripts (to write targets the model learns from), so both always
agree on the format.

Coordinates
-----------
Different model families write points in different spaces:

- ``abs``: absolute pixels of the image *after* the Qwen2.5-VL processor's
  ``smart_resize`` (multiples of 28, capped by max_pixels). UI-TARS-1.5 and
  Qwen2.5-VL do this.
- ``rel1000``: 0..1000 relative to the image width and height (UI-TARS 1.0,
  Seed/Doubao ``<point>x y</point>`` output, Qwen3-VL grounding).

``CoordSpace`` converts between a model's space and the screenshot's pixels.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# Official UI-TARS action space (desktop), plus one LeanWatch addition:
# ask_user(content=...) asks the person before a risky step.
ACTION_NAMES = {
    "click", "left_single", "left_double", "right_single", "drag", "hotkey", "press", "keydown",
    "type", "scroll", "wait", "finished", "call_user", "ask_user", "move_to", "mouse_move",
}


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2.5-VL's resize rule (same as the official UI-TARS code): returns (height, width)."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"aspect ratio must be below {MAX_RATIO}")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


@dataclass(frozen=True)
class CoordSpace:
    """Maps points between a model's coordinate space and image pixels."""

    mode: str                      # "abs" or "rel1000"
    image_w: int
    image_h: int
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS

    @property
    def model_size(self) -> tuple[int, int]:
        """(width, height) of the space the model writes coordinates in."""
        if self.mode == "abs":
            h, w = smart_resize(self.image_h, self.image_w, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            return w, h
        if self.mode == "rel1000":
            return 1000, 1000
        raise ValueError(f"unknown coordinate mode {self.mode!r}")

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        mw, mh = self.model_size
        return x * self.image_w / mw, y * self.image_h / mh

    def to_model(self, x: float, y: float) -> tuple[int, int]:
        mw, mh = self.model_size
        return round(x * mw / self.image_w), round(y * mh / self.image_h)


@dataclass
class Action:
    """One parsed action. Points are in the model's coordinate space."""

    name: str
    point: tuple[float, float] | None = None
    end_point: tuple[float, float] | None = None
    text: str = ""            # type/finished/ask_user content, or hotkey keys
    direction: str = ""       # scroll
    raw: str = ""


@dataclass
class Parsed:
    thought: str
    actions: list[Action] = field(default_factory=list)
    raw: str = ""


class ParseError(ValueError):
    """The model's answer could not be understood."""


_NUM = r"-?\d+(?:\.\d+)?"


def _parse_point(value: str) -> tuple[float, float]:
    """Accepts (x,y), [x,y], <point>x y</point>, <|box_start|>(x,y)<|box_end|> and boxes (centre is used)."""
    nums = [float(n) for n in re.findall(_NUM, value)]
    if len(nums) == 2:
        return nums[0], nums[1]
    if len(nums) == 4:
        return (nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2
    raise ParseError(f"cannot read a point from {value!r}")


def _split_calls(action_text: str) -> list[str]:
    """Split 'a(...)\\n\\nb(...)' into calls without breaking quoted text."""
    calls, depth, quote, start, i = [], 0, None, None, 0
    while i < len(action_text):
        ch = action_text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            if depth == 0:
                j = i - 1
                while j >= 0 and (action_text[j].isalnum() or action_text[j] == "_"):
                    j -= 1
                start = j + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                calls.append(action_text[start:i + 1])
                start = None
        i += 1
    return calls


def _call_kwargs(call: str) -> tuple[str, dict]:
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        # Models sometimes leave quotes inside type(content='...') unescaped; recover the content.
        m = re.match(r"\s*(\w+)\(\s*content\s*=\s*'(.*)'\s*\)\s*$", call, re.DOTALL)
        if m:
            return m.group(1), {"content": m.group(2).replace("\\'", "'").replace("\\n", "\n")}
        raise ParseError(f"cannot parse {call!r}") from None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ParseError(f"not an action call: {call!r}")
    kwargs = {}
    for kw in node.keywords:
        if not isinstance(kw.value, ast.Constant):
            raise ParseError(f"argument {kw.arg!r} must be a literal in {call!r}")
        kwargs[kw.arg] = kw.value.value
    return node.func.id, kwargs


def parse_response(text: str) -> Parsed:
    """Parse 'Thought: ... Action: ...' into a thought and a list of actions."""
    raw = text
    text = text.replace("[EOS]", "").strip()
    thought = ""
    m = re.search(r"(?:Thought|Action_Summary|Reflection):\s*(.*?)(?=\n\s*Action:|\Z)", text, re.DOTALL)
    if m:
        thought = m.group(1).strip()
    if "Action:" not in text:
        raise ParseError("the answer has no 'Action:' line")
    action_text = text.split("Action:")[-1].strip()
    calls = _split_calls(action_text)
    if not calls and re.match(r"^\w+\(.*\)$", action_text, re.DOTALL):
        calls = [action_text]  # one call whose text has an unescaped quote
    if not calls:
        raise ParseError(f"no action found in {action_text!r}")
    actions = []
    for call in calls:
        name, kw = _call_kwargs(call)
        if name not in ACTION_NAMES:
            raise ParseError(f"unknown action {name!r}")
        a = Action(name=name, raw=call)
        start = kw.get("start_box", kw.get("point", kw.get("start_point")))
        end = kw.get("end_box", kw.get("end_point"))
        if start is not None:
            a.point = _parse_point(str(start))
        if end is not None:
            a.end_point = _parse_point(str(end))
        a.text = str(kw.get("content", kw.get("key", kw.get("hotkey", ""))))
        a.direction = str(kw.get("direction", ""))
        actions.append(a)
    return Parsed(thought=thought, actions=actions, raw=raw)


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _box(x: float, y: float, style: str) -> str:
    if style == "point":
        return f"<point>{round(x)} {round(y)}</point>"
    return f"<|box_start|>({round(x)},{round(y)})<|box_end|>"


def format_action(action: dict, space: CoordSpace, style: str = "box") -> str:
    """Write a LeanWatch/recorder action dict (image pixels) as a model action string.

    style "box" gives UI-TARS-1.5 syntax (start_box='<|box_start|>(x,y)<|box_end|>'),
    style "point" gives the newer point='<point>x y</point>' syntax.
    """
    kind = action["type"]
    key = "point" if style == "point" else "start_box"
    if kind == "click":
        x, y = space.to_model(action["x"], action["y"])
        # The action space has no triple click. The recorder only triple-clicks to select a
        # number in a box, and a double click selects that number just as well.
        name = "click" if action.get("clicks", 1) == 1 else "left_double"
        if action.get("button") == "right":
            name = "right_single"
        return f"{name}({key}='{_box(x, y, style)}')"
    if kind == "drag":
        x1, y1 = space.to_model(action["x1"], action["y1"])
        x2, y2 = space.to_model(action["x2"], action["y2"])
        end = "end_point" if style == "point" else "end_box"
        return f"drag({key}='{_box(x1, y1, style)}', {end}='{_box(x2, y2, style)}')"
    if kind == "type":
        return f"type(content='{_quote(action['text'])}')"
    if kind == "hotkey":
        return f"hotkey(key='{' '.join(k.lower() for k in action['keys'])}')"
    if kind == "scroll":
        x, y = space.to_model(action["x"], action["y"])
        return f"scroll({key}='{_box(x, y, style)}', direction='{action['direction']}')"
    if kind == "wait":
        return "wait()"
    if kind == "confirm":
        return f"ask_user(content='{_quote(action['text'])}')"
    if kind == "finished":
        return f"finished(content='{_quote(action.get('text', ''))}')"
    raise ValueError(f"unknown action type {kind!r}")


def format_response(thought: str, action: str) -> str:
    return f"Thought: {thought.strip()}\nAction: {action}" if thought else f"Action: {action}"


def hotkey_to_combo(keys: str) -> str:
    """UI-TARS 'ctrl c' -> controller 'ctrl+c'. Also accepts 'ctrl+c'."""
    parts = [p for p in re.split(r"[\s+]+", keys.strip().lower()) if p]
    names = {"arrowleft": "Left", "arrowright": "Right", "arrowup": "Up", "arrowdown": "Down",
             "enter": "Return", "esc": "Escape", "escape": "Escape", "space": "space", "backspace": "BackSpace",
             "pagedown": "Page_Down", "pageup": "Page_Up", "delete": "Delete", "tab": "Tab", "home": "Home", "end": "End"}
    return "+".join(names.get(p, p) for p in parts)
