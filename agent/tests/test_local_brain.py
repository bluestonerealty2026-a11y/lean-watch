import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from leanwatch.config import Config
from leanwatch.controller import Controller, DryRunBackend
from leanwatch.hooks import Hooks
from leanwatch.local_brain import LocalBrain, ModelAnswer, risky_intent
from leanwatch.permissions import Permissions
from leanwatch.screen import Geometry, ScreenCapture


class ScriptedModel:
    """Plays back answers and records what the brain sent."""

    def __init__(self, answers, size=(1372, 756)):
        self.answers = list(answers)
        self.size = size
        self.calls = []

    def answer(self, messages, images, stop=None):
        self.calls.append((messages, images))
        return ModelAnswer(self.answers.pop(0), *self.size)


class Recorder(Hooks):
    def __init__(self, allow=True):
        self.said, self.asked, self.allow = [], [], allow

    def narrate(self, text, kind="say"):
        self.said.append((kind, text))

    def confirm(self, action, reason):
        self.asked.append(action)
        return self.allow


def make(answers, allow=True, size=(1372, 756), **cfg_kw):
    backend = DryRunBackend((1920, 1080))
    g = Geometry.for_screen(1920, 1080)
    cap = ScreenCapture(g)
    cap._grab_raw = lambda: Image.new("RGB", (1920, 1080), "white")
    perms = Permissions()
    perms.grant()
    stop = threading.Event()
    hooks = Recorder(allow)
    controller = Controller(backend, cap, g, perms, speed=50, confirm=hooks.confirm, stop=stop, platform="linux")
    cfg = Config()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    model = ScriptedModel(answers, size)
    return LocalBrain(cfg, controller, hooks, stop, model), backend, hooks, model, g, stop


def test_click_maps_model_pixels_to_the_screen():
    brain, backend, hooks, model, g, _ = make([
        "Thought: I open Settings from the taskbar.\nAction: click(start_box='<|box_start|>(686,378)<|box_end|>')",
        "Thought: All done.\nAction: finished(content='Settings is open.')",
    ])
    result = brain.run("open settings")
    assert result.ok and result.message == "Settings is open."
    click = [e for e in backend.log if e[0] == "click"][0]
    # (686, 378) in a 1372x756 model image is the centre of the screen.
    assert abs(click[2][0] - 960) <= 3 and abs(click[2][1] - 540) <= 3
    assert ("say", "I open Settings from the taskbar.") in hooks.said
    # Second turn carries the first answer as history.
    second_messages = model.calls[1][0]
    assert any(m["role"] == "assistant" for m in second_messages)


def test_rel1000_coordinates():
    brain, backend, *_ = make([
        "Action: click(point='<point>500 500</point>')",
        "Action: finished(content='ok')",
    ], coords="rel1000")
    assert brain.run("x").ok
    click = [e for e in backend.log if e[0] == "click"][0]
    assert abs(click[2][0] - 960) <= 3 and abs(click[2][1] - 540) <= 3


def test_type_with_newline_submits_and_hotkeys_map():
    brain, backend, *_ = make([
        "Thought: I type the command.\nAction: type(content='ipconfig\\n')",
        "Thought: Undo.\nAction: hotkey(key='ctrl z')",
        "Action: finished(content='ok')",
    ])
    assert brain.run("x").ok
    assert ("write", "ipconfig") in backend.log
    downs = [e[1] for e in backend.log if e[0] == "key_down"]
    assert downs == ["enter", "ctrl", "z"]


def test_risky_thought_needs_approval():
    brain, backend, hooks, *_ = make([
        "Thought: I click Delete to remove the folder.\nAction: click(start_box='(100,100)')",
    ], allow=False)
    result = brain.run("delete Old Photos")
    assert not result.ok and hooks.asked
    assert not any(e[0] == "click" for e in backend.log)


def test_ask_user_round_trip():
    brain, backend, hooks, model, *_ = make([
        "Thought: This deletes it for good.\nAction: ask_user(content='Delete Old Photos?')",
        "Thought: Approved, I click Delete.\nAction: click(start_box='(100,100)')",
        "Action: finished(content='Deleted.')",
    ])
    assert brain.run("delete Old Photos").ok
    assert hooks.asked[0] == "Delete Old Photos?"
    # The approval is passed to the model on the next turn.
    texts = [c.get("text") for m in model.calls[1][0] for c in m["content"] if c["type"] == "text"]
    assert any("approved" in (t or "") for t in texts)
    assert any(e[0] == "click" for e in backend.log)


def test_unreadable_answers_stop_after_three():
    brain, backend, *_ = make(["hello", "still no action", "nope"])
    result = brain.run("x")
    assert not result.ok and "could not understand" in result.message


def test_stop_ends_the_run():
    brain, backend, hooks, model, g, stop = make(["Action: click(start_box='(1,1)')"] * 5)
    stop.set()
    assert "Stopped" in brain.run("x").message


def test_only_recent_screenshots_are_sent():
    answers = [f"Thought: step {i}\nAction: wait()" for i in range(5)] + ["Action: finished(content='ok')"]
    brain, _, _, model, *_ = make(answers, history_images=2)
    assert brain.run("x").ok
    assert all(len(images) <= 2 for _, images in model.calls)


@pytest.mark.parametrize("thought,risky", [
    ("I click Delete to remove the file.", True), ("Now I press Send.", True), ("I click Install.", True),
    ("I open the Display settings.", False), ("I click the Image menu.", False),
])
def test_risky_intent(thought, risky):
    assert bool(risky_intent(thought)) is risky


def test_real_tiny_qwen_model_answers(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "tests"))
    from tiny_model import build

    from leanwatch.local_brain import LocalModel
    from leanwatch.local_prompts import build_messages, computer_prompt

    path = build(tmp_path / "tiny")
    model = LocalModel(str(path), device="cpu", max_new_tokens=8)
    messages, turns = build_messages(computer_prompt("Turn on dark mode"), [], 3)
    ans = model.answer(messages, [Image.new("RGB", (1366, 768), "gray")])
    assert isinstance(ans.text, str)
    # The tiny processor caps images at 64 patches of 28px; the brain must learn the real size from the grid.
    assert ans.image_w % 28 == 0 and ans.image_h % 28 == 0 and ans.image_w * ans.image_h <= 28 * 28 * 64


@pytest.mark.parametrize("style,mark,other", [("box", "<|box_start|>(x1,y1)<|box_end|>", "<point>"),
                                              ("point", "<point>x1 y1</point>", "box_start")])
def test_prompt_writes_points_in_the_trained_style(style, mark, other):
    from leanwatch.local_prompts import computer_prompt, grounding_prompt

    text = computer_prompt("Rename {P1} to {END}", "Spanish", style)
    assert mark in text and other not in text and "{START}" not in text
    assert "Rename {P1} to {END}" in text and "Use Spanish in `Thought` part." in text
    assert mark in grounding_prompt("Click OK", style)
    with pytest.raises(ValueError):
        computer_prompt("x", style="xml")


def test_claude_brain_without_its_package_says_how_to_install(monkeypatch):
    import sys as _sys

    from leanwatch.brains import make_brain

    monkeypatch.setitem(_sys.modules, "anthropic", None)
    monkeypatch.delitem(_sys.modules, "leanwatch.brain", raising=False)
    cfg = Config()
    cfg.brain = "claude"
    with pytest.raises(RuntimeError, match="requirements-claude.txt"):
        make_brain(cfg, None, None, threading.Event())


# --------------------------------------------------------------------------- review fixes
@pytest.mark.parametrize("thought", [
    "I click Empty Trash to free up space.", "I click Empty Recycle Bin.", "I click Replace to overwrite the existing file.",
    "I turn off the firewall.", "I click Place order.", "I press Delete.",
    "Hago clic en Vaciar papelera para borrar todo.", "Je clique sur Supprimer.", "Ich klicke auf Löschen.",
    "Clico em Excluir.", "Clicco su Elimina.", "J'envoie le message.",
])
def test_risky_words_in_several_languages(thought):
    assert risky_intent(thought)


@pytest.mark.parametrize("thought", [
    "I compress the image to make it smaller.", "I open the Public folder.", "I invite you to look at the menu.",
    "I click the Display page.", "I format the text as bold.", "I open the Command Prompt.",
])
def test_ordinary_steps_are_not_flagged(thought):
    assert risky_intent(thought) is None


def test_keydown_is_checked_too():
    brain, backend, hooks, *_ = make(["Thought: I press Enter to send the email.\nAction: keydown(key='enter')"], allow=False)
    assert not brain.run("send it").ok
    assert hooks.asked and not [e for e in backend.log if e[0] == "key_down"]


def test_unknown_language_asks_before_every_acting_step():
    brain, backend, hooks, *_ = make([
        "Thought: Ninjiye muri Settings.\nAction: click(start_box='<|box_start|>(686,378)<|box_end|>')",
        "Thought: Ndategereza.\nAction: wait()",
        "Action: finished(content='Byarangiye.')",
    ], language="Kinyarwanda")
    assert brain.run("fungura settings").ok
    assert hooks.asked == ["Ninjiye muri Settings."]      # the click was asked about; wait and finished were not


def test_approval_covers_the_steps_that_carry_it_out():
    brain, backend, hooks, *_ = make([
        "Thought: Deleting can't be undone, so I ask first.\nAction: ask_user(content='Delete the folder Old Photos?')",
        "Thought: You approved, so I click Delete.\nAction: click(start_box='<|box_start|>(686,378)<|box_end|>')",
        "Thought: Files asks me to confirm. I click Delete.\nAction: click(start_box='<|box_start|>(700,400)<|box_end|>')",
        "Thought: Now I delete the Music folder too.\nAction: click(start_box='<|box_start|>(10,10)<|box_end|>')",
        "Action: finished(content='Done.')",
    ])
    assert brain.run("delete old photos").ok
    # One question for the approved deletion; the later, unrelated delete is asked about again.
    assert hooks.asked == ["Delete the folder Old Photos?", "Now I delete the Music folder too."]


def test_stop_while_paused_does_not_start_another_step():
    brain, backend, hooks, model, g, stop = make(["Action: finished(content='x')"])

    def paused_then_stopped():
        stop.set()

    hooks.wait_if_paused = paused_then_stopped
    assert not brain.run("x").ok
    assert model.calls == []


def test_answers_keep_box_tokens_for_the_history():
    from leanwatch.local_brain import clean_reply

    raw = "Thought: ok\nAction: click(start_box='<|box_start|>(12,34)<|box_end|>')<|im_end|><|endoftext|><|endoftext|>"
    assert clean_reply(raw) == "Thought: ok\nAction: click(start_box='<|box_start|>(12,34)<|box_end|>')"
    assert clean_reply("Action: wait()</s>junk", ("</s>",)) == "Action: wait()"


def test_plus_key_survives():
    from leanwatch.actions import hotkey_to_combo
    from leanwatch.controller import parse_combo

    assert parse_combo(hotkey_to_combo("ctrl +"), "linux") == ["ctrl", "+"]
    assert parse_combo(hotkey_to_combo("ctrl++"), "linux") == ["ctrl", "+"]
    assert parse_combo(hotkey_to_combo("ctrl shift +"), "linux") == ["ctrl", "shift", "+"]
    assert parse_combo(hotkey_to_combo("+"), "linux") == ["+"]
    assert hotkey_to_combo("ctrl c") == "ctrl+c" and hotkey_to_combo("ctrl+c") == "ctrl+c"


def test_typing_waits_if_the_person_moved_the_mouse_since_the_last_step():
    from leanwatch.controller import UserTookOver

    brain, backend, *_ = make([])
    c = brain.controller
    c.execute("left_click", {"coordinate": [100, 100]})
    backend._pos = (900, 700)                     # the person clicked into another app meanwhile
    with pytest.raises(UserTookOver):
        c.execute("type", {"text": "see you\n"})
    assert not [e for e in backend.log if e[0] == "write"]


def test_point_style_drag_uses_start_point():
    from leanwatch.actions import CoordSpace, format_action

    space = CoordSpace("abs", 1000, 625)
    text = format_action({"type": "drag", "x1": 10, "y1": 10, "x2": 100, "y2": 10}, space, "point")
    assert text.startswith("drag(start_point='<point>") and "end_point='<point>" in text


# --------------------------------------------------------------------------- second review round
@pytest.mark.parametrize("thought", [
    "I click Sell to sell 10 shares.", "I click Don’t Save.", "I close it without saving.",
    "Ich klicke auf Absenden.", "Ich leere den Papierkorb.", "Je clique sur Ne pas enregistrer.",
    "Haz clic en Restablecer este equipo.", "Clico em Não salvar.", "Ich setze den PC zurück: Zurücksetzen.",
])
def test_more_risky_phrasings(thought):
    assert risky_intent(thought)


@pytest.mark.parametrize("thought", [
    "I'm formatting the paragraph as a list.", "Premo Invio per cercare.", "J'ouvre l'invite de commande.",
    "Ich sortiere die Zahlen in der Tabelle.", "Formatto il testo in grassetto.", "I open the Display page.",
])
def test_more_ordinary_steps_stay_quiet(thought):
    assert risky_intent(thought) is None


def test_an_approval_covers_only_what_was_approved():
    brain, backend, hooks, *_ = make([
        "Thought: Moving files to the Recycle Bin, so I ask first.\nAction: ask_user(content='Delete these three files?')",
        "Thought: You approved, so I click Delete.\nAction: click(start_box='<|box_start|>(686,378)<|box_end|>')",
        "Thought: Now I empty the Recycle Bin to free space.\nAction: click(start_box='<|box_start|>(700,400)<|box_end|>')",
        "Action: finished(content='Done.')",
    ])
    assert brain.run("delete these files").ok
    assert hooks.asked == ["Delete these three files?",
                           "Now I empty the Recycle Bin to free space."]   # a different, worse step: asked again


def test_a_typed_step_is_asked_about_once_with_the_exact_text():
    brain, backend, hooks, *_ = make([
        "Thought: I delete the old logs.\nAction: type(content='rm -rf ./old-logs\\n')",
        "Action: finished(content='Done.')",
    ])
    assert brain.run("clean up").ok
    assert len(hooks.asked) == 1                                 # not again by the controller's own check
    assert "rm -rf ./old-logs" in hooks.asked[0] and "press Enter" in hooks.asked[0]
    assert ("write", "rm -rf ./old-logs") in backend.log
    assert brain.controller.preapproved is False                 # only for that one action


def test_controller_still_asks_for_steps_nobody_approved():
    brain, backend, hooks, *_ = make([
        "Thought: I type the cleanup command.\nAction: type(content='rm -rf /tmp/x')",   # thought has no risky word
        "Action: finished(content='Done.')",
    ], allow=False)
    brain.run("tidy")
    assert hooks.asked and hooks.asked[0].startswith("Type")     # the controller's typed-text check caught it
    assert not [e for e in backend.log if e[0] == "write"]
