import pytest

from leanwatch.actions import (
    CoordSpace, ParseError, format_action, format_response, hotkey_to_combo, parse_response, smart_resize,
)


def test_smart_resize_matches_qwen_rule():
    assert smart_resize(1080, 1920) == (1092, 1932)
    h, w = smart_resize(1080, 1920, max_pixels=1280 * 28 * 28)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 1280 * 28 * 28


@pytest.mark.parametrize("text,point", [
    ("Thought: Click the button\nAction: click(start_box='(100,200)')", (100, 200)),
    ("Thought: x\nAction: click(start_box='<|box_start|>(312,145)<|box_end|>')", (312, 145)),
    ("Action: click(point='<point>200 300</point>')", (200, 300)),
    ("Action: click(start_box='[10,20,30,40]')", (20, 30)),
])
def test_click_variants(text, point):
    p = parse_response(text)
    assert p.actions[0].name == "click" and p.actions[0].point == point


def test_thought_drag_type_hotkey_scroll_finished():
    p = parse_response(
        "Thought: Open the menu, then type.\n"
        "Action: drag(start_box='(1,2)', end_box='(3,4)')\n\n"
        "type(content='it\\'s done\\n')\n\n"
        "hotkey(key='ctrl c')\n\n"
        "scroll(start_box='(5,6)', direction='down')\n\n"
        "finished(content='All set')")
    assert p.thought == "Open the menu, then type."
    names = [a.name for a in p.actions]
    assert names == ["drag", "type", "hotkey", "scroll", "finished"]
    assert p.actions[0].point == (1, 2) and p.actions[0].end_point == (3, 4)
    assert p.actions[1].text == "it's done\n"
    assert p.actions[2].text == "ctrl c" and hotkey_to_combo(p.actions[2].text) == "ctrl+c"
    assert p.actions[3].direction == "down"
    assert p.actions[4].text == "All set"


def test_unescaped_quote_in_type_is_recovered():
    p = parse_response("Action: type(content='don't stop')")
    assert p.actions[0].text == "don't stop"


@pytest.mark.parametrize("bad", ["Thought: no action here", "Action: explode(start_box='(1,2)')", "Action: click"])
def test_bad_answers_raise(bad):
    with pytest.raises(ParseError):
        parse_response(bad)


@pytest.mark.parametrize("mode", ["abs", "rel1000"])
@pytest.mark.parametrize("style", ["box", "point"])
def test_round_trip_through_model_space(mode, style):
    space = CoordSpace(mode, 1500, 938, max_pixels=1280 * 28 * 28)
    for action in [
        {"type": "click", "x": 412, "y": 604, "clicks": 1},
        {"type": "click", "x": 1360, "y": 640, "clicks": 3},
        {"type": "drag", "x1": 100, "y1": 200, "x2": 900, "y2": 210},
    ]:
        text = format_response("I click it.", format_action(action, space, style))
        a = parse_response(text).actions[0]
        x, y = space.to_image(*a.point)
        ox, oy = (action["x"], action["y"]) if "x" in action else (action["x1"], action["y1"])
        tol = 1500 / space.model_size[0] + 1
        assert abs(x - ox) <= tol and abs(y - oy) <= tol
    assert parse_response(format_response("t", format_action({"type": "click", "x": 5, "y": 5, "clicks": 3}, space, style))).actions[0].name == "left_double"


def test_other_actions_format():
    space = CoordSpace("abs", 1000, 625)
    assert format_action({"type": "type", "text": "it's"}, space) == "type(content='it\\'s')"
    assert format_action({"type": "hotkey", "keys": ["ctrl", "z"]}, space) == "hotkey(key='ctrl z')"
    assert format_action({"type": "confirm", "text": "Delete it?"}, space) == "ask_user(content='Delete it?')"
    assert parse_response("Action: " + format_action({"type": "finished", "text": "Done.\nBye"}, space)).actions[0].text == "Done.\nBye"
