import pytest

from leanwatch.controller import (
    Controller, DryRunBackend, EmergencyStop, UserTookOver, describe, human_path, movement_time, parse_combo,
)
from leanwatch.permissions import PermissionDenied, Permissions
from leanwatch.screen import Geometry


def make(confirm=None, keyboard=True, size=(1920, 1080)):
    backend = DryRunBackend(size)
    g = Geometry.for_screen(*size)
    perms = Permissions()
    perms.grant(mouse=True, keyboard=keyboard)
    c = Controller(backend, None, g, perms, speed=50, confirm=confirm, platform="linux")
    return c, backend, g


@pytest.mark.parametrize("combo,keys", [
    ("ctrl+s", ["ctrl", "s"]),
    ("Return", ["enter"]),
    ("alt+Tab", ["alt", "tab"]),
    ("ctrl+shift+Escape", ["ctrl", "shift", "esc"]),
    ("ctrl++", ["ctrl", "+"]),
    ("Page_Down", ["pagedown"]),
    ("F5", ["f5"]),
    ("super", ["win"]),
])
def test_parse_combo(combo, keys):
    assert parse_combo(combo, "linux") == keys


def test_super_is_command_on_mac():
    assert parse_combo("super+space", "darwin") == ["command", "space"]
    assert parse_combo("cmd+c", "darwin") == ["command", "c"]


def test_motion_model():
    assert movement_time(800) > movement_time(100) > movement_time(5)
    pts = human_path(0, 0, 400, 300, 40)
    assert pts[-1] == (400, 300)
    assert len(pts) == 40


def test_click_maps_model_coordinates_to_the_screen():
    c, b, g = make()
    c.execute("left_click", {"coordinate": [g.shot_w // 2, g.shot_h // 2]})
    kind, button, pos = b.log[-1]
    assert (kind, button) == ("click", "left")
    assert abs(pos[0] - 960) <= 2 and abs(pos[1] - 540) <= 2


def test_triple_click_and_modifiers():
    c, b, _ = make()
    c.execute("triple_click", {"coordinate": [10, 10]})
    assert [e[0] for e in b.log].count("click") == 3
    b.log.clear()
    c.execute("left_click", {"coordinate": [20, 20], "text": "ctrl+shift"})
    assert [e[0] for e in b.log] == ["key_down", "key_down", "click", "key_up", "key_up"]
    assert b.log[0][1] == "ctrl" and b.log[-1][1] == "ctrl"


def test_drag_presses_moves_and_releases():
    c, b, g = make()
    c.execute("left_click_drag", {"start_coordinate": [10, 10], "coordinate": [200, 100]})
    kinds = [e[0] for e in b.log]
    assert kinds == ["down", "up"]
    assert b.log[1][2] == g.to_screen(200, 100)


def test_scroll_direction():
    c, b, _ = make()
    c.execute("scroll", {"scroll_direction": "down", "scroll_amount": 5})
    assert b.log[-1][:3] == ("scroll", 0, -5)
    c.execute("scroll", {"scroll_direction": "right", "scroll_amount": 2})
    assert b.log[-1][:3] == ("scroll", 2, 0)


def test_key_repeat_and_typing():
    c, b, _ = make()
    c.execute("key", {"text": "Down", "repeat": 3})
    assert [e for e in b.log if e[0] == "key_down"] == [("key_down", "down")] * 3
    c.execute("type", {"text": "hello"})
    assert b.log[-1] == ("write", "hello")


def test_risky_typing_needs_confirmation():
    asked = []
    c, b, _ = make(confirm=lambda action, reason: asked.append(action) or False)
    with pytest.raises(PermissionDenied):
        c.execute("type", {"text": "rm -rf ~/Pictures"})
    assert asked and not any(e[0] == "write" for e in b.log)

    c2, b2, _ = make(confirm=lambda action, reason: True)
    c2.execute("type", {"text": "rm -rf build"})
    assert b2.log[-1] == ("write", "rm -rf build")


def test_keyboard_scope_is_enforced():
    c, b, _ = make(keyboard=False)
    with pytest.raises(PermissionDenied):
        c.execute("type", {"text": "hi"})
    assert b.log == []


def test_user_moving_the_mouse_pauses_the_agent():
    c, b, _ = make()
    c.execute("mouse_move", {"coordinate": [100, 100]})
    b.move_to(1500, 900)  # the person grabs the mouse
    with pytest.raises(UserTookOver):
        c.execute("left_click", {"coordinate": [100, 100]})
    c.resync()
    c.execute("left_click", {"coordinate": [100, 100]})  # fine after handing back


def test_failsafe_becomes_emergency_stop():
    class FailSafeException(Exception):
        pass

    c, b, _ = make()

    def boom(button="left"):
        raise FailSafeException("corner")

    b.click = boom
    with pytest.raises(EmergencyStop):
        c.execute("left_click", {})


def test_cursor_position_is_in_screenshot_space():
    c, b, g = make()
    b.move_to(1919, 1079)
    x, y = (int(v.split("=")[1]) for v in c.execute("cursor_position", {}).split(", "))
    assert abs(x - g.shot_w) <= 2 and abs(y - g.shot_h) <= 2


def test_screenshot_without_capture_is_a_clear_error():
    c, _, _ = make()
    with pytest.raises(RuntimeError, match="Screen capture"):
        c.execute("screenshot", {})


def test_describe_is_readable():
    assert describe("left_click", {"coordinate": [5, 6]}) == "left click at (5, 6)"
    assert describe("key", {"text": "ctrl+s"}) == "press ctrl+s"
    assert describe("type", {"text": "hi"}) == "type “hi”"
