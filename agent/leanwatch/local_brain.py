"""The default brain: an open vision-language GUI model running on your own computer.

No API key and no screenshots leave the machine. Works with UI-TARS-1.5 out of the box
and with any model you fine-tune with training/train_sft.py (pass its folder as --model,
or the LoRA folder as --adapter).

Each step: take a screenshot -> build the conversation (training/inference share
local_prompts.build_messages) -> the model answers "Thought: ... Action: ..." ->
show the thought to the person -> safety checks -> move the real mouse/keyboard.
"""

from __future__ import annotations

import threading
import time

from .actions import Parsed, ParseError, hotkey_to_combo, parse_response
from .config import Config
from .controller import Controller, EmergencyStop
from .hooks import Hooks, RunResult, Stopped, check_stop, run_action
from .local_model import END_MARKERS, LocalModel, ModelAnswer, clean_reply, coords_for, set_image_size  # noqa: F401  (re-exported)
from .local_prompts import APPROVED_NOTE, build_messages, computer_prompt
from .permissions import PermissionDenied
from .risk import language_checked, risky_intent  # noqa: F401  (risky_intent is re-exported for tests)

# Actions that change nothing on the screen: no approval needed for these.
PASSIVE_ACTIONS = {"wait", "move_to", "mouse_move", "finished", "ask_user", "call_user"}
# After the person approves an ask_user step, the next turns carry it out without asking again
# (e.g. "click Delete", then "the app asks to confirm, click Delete").
APPROVAL_TURNS = 2


def describe_step(thought: str, action) -> str:
    """The question shown before a risky step: the model's explanation plus exactly what will be typed or pressed."""
    detail = ""
    if action.name == "type" and action.text:
        shown = action.text.rstrip("\n")
        detail = f"Type “{shown[:120]}{'…' if len(shown) > 120 else ''}”" + (" and press Enter" if action.text.endswith("\n") else "")
    elif action.name in ("hotkey", "press", "keydown") and action.text:
        detail = f"Press {action.text}"
    if thought and detail:
        return f"{thought}\n{detail}"
    return thought or detail or action.name


class LocalBrain:
    """Runs one task with a LocalModel. Same Hooks and safety rules as the Claude brain."""

    def __init__(self, cfg: Config, controller: Controller, hooks: Hooks, stop: threading.Event, model):
        self.cfg = cfg
        self.controller = controller
        self.hooks = hooks
        self.stop = stop
        self.model = model

    # ------------------------------------------------------------------
    def run(self, task: str) -> RunResult:
        prompt = computer_prompt(task, self.cfg.language, self.cfg.style)
        screenshots: list = []
        past: list[dict] = []
        bad_answers = 0
        self.approved_turns = 0
        self.approved_text = ""
        self.ask_every_step = not language_checked(self.cfg.language)
        try:
            for _ in range(self.cfg.max_turns):
                check_stop(self.stop)
                self.hooks.wait_if_paused()
                check_stop(self.stop)          # Stop pressed while paused: don't start another step
                screenshots.append(self.controller.capture.screenshot())
                messages, image_turns = build_messages(prompt, past, self.cfg.history_images)
                t0 = time.monotonic()
                answer = self.model.answer(messages, [screenshots[i] for i in image_turns], stop=self.stop)
                check_stop(self.stop)
                try:
                    parsed = parse_response(answer.text)
                except ParseError as err:
                    bad_answers += 1
                    self.hooks.narrate(f"(The model's answer could not be read: {err}. Trying again.)", "thinking")
                    if bad_answers >= 3:
                        return RunResult(False, "The model kept giving answers I could not understand, so I stopped.")
                    past.append({"response": answer.text, "note": "Your last answer could not be parsed. Answer with Thought: and one Action: from the action space."})
                    continue
                bad_answers = 0
                if parsed.thought:
                    self.hooks.narrate(parsed.thought, "say")
                self.hooks.narrate(f"({time.monotonic() - t0:.1f} s to decide)", "thinking")
                outcome, note = self._act(parsed, answer)
                past.append({"response": answer.text, "note": note})
                if outcome is not None:
                    return outcome
            return RunResult(False, f"Stopped after {self.cfg.max_turns} steps without finishing. Try a smaller task.")
        except Stopped:
            return RunResult(False, "Stopped. You are in control.")
        except EmergencyStop as err:
            return RunResult(False, str(err))

    # ------------------------------------------------------------------
    def _act(self, parsed: Parsed, answer: ModelAnswer) -> tuple[RunResult | None, str | None]:
        """Carry out this turn's actions. Returns (result if the task ended, note for the next turn)."""
        g = self.controller.g
        sx, sy = g.shot_w / answer.image_w, g.shot_h / answer.image_h
        coords = self.cfg.coords if self.cfg.coords != "auto" else getattr(self.model, "coords", "abs")
        if coords == "rel1000":
            sx, sy = g.shot_w / 1000, g.shot_h / 1000

        def at(p):
            return [round(p[0] * sx), round(p[1] * sy)]

        asked = False
        approved_now = False
        turn_has_approval = self.approved_turns > 0
        self.approved_turns = max(0, self.approved_turns - 1)
        for a in parsed.actions:
            check_stop(self.stop)
            if a.name in ("finished",):
                return RunResult(True, a.text or parsed.thought or "Done."), None
            if a.name in ("ask_user", "call_user"):
                question = a.text or parsed.thought or "May I continue?"
                if not self.hooks.confirm(question, "The assistant wants your OK before this step."):
                    return RunResult(False, f"You said no to: {question} Nothing more was done."), None
                # The approval carries over to the steps that carry it out (e.g. "click Delete", then the
                # app's own "Delete?" dialog), but only for the same kind of thing the person agreed to.
                self.approved_turns = 1 if self.ask_every_step else APPROVAL_TURNS
                self.approved_text = (question + " " + (parsed.thought or "")).lower()
                return None, APPROVED_NOTE
            if a.name not in PASSIVE_ACTIONS and not asked:
                reason = risky_intent(parsed.thought)
                covered = turn_has_approval and (reason is None and self.ask_every_step or
                                                 reason is not None and reason in self.approved_text)
                if covered:
                    approved_now = True
                elif reason or self.ask_every_step:
                    asked = True
                    why = (f"This step mentions “{reason}”, which can be hard to undo." if reason else
                           f"LeanWatch can't check {self.cfg.language} explanations for risky steps, so it asks before each one.")
                    if not self.hooks.confirm(describe_step(parsed.thought, a), why):
                        return RunResult(False, "You said no, so I stopped before that step."), None
                    approved_now = True
            try:
                # The person has just seen and allowed this exact step: the controller needn't ask again.
                self.controller.preapproved = approved_now and a.name in ("type", "hotkey", "press", "keydown")
                self._do(a, at)
            except PermissionDenied as err:
                return None, f"That action was refused: {err}"
            except (Stopped, EmergencyStop):
                raise
            except Exception as err:  # bad coordinates, unknown key, ...
                return None, f"That action failed: {err}. Try a different way."
            finally:
                self.controller.preapproved = False
        self.hooks.preview()
        return None, None

    def _do(self, a, at) -> None:
        def go(name, inp):
            return run_action(self.controller, self.hooks, self.stop, name, inp)

        if a.name in ("click", "left_single"):
            go("left_click", {"coordinate": at(a.point)})
        elif a.name == "left_double":
            go("double_click", {"coordinate": at(a.point)})
        elif a.name == "right_single":
            go("right_click", {"coordinate": at(a.point)})
        elif a.name in ("move_to", "mouse_move"):
            go("mouse_move", {"coordinate": at(a.point)})
        elif a.name == "drag":
            go("left_click_drag", {"start_coordinate": at(a.point), "coordinate": at(a.end_point)})
        elif a.name in ("hotkey", "press", "keydown"):
            go("key", {"text": hotkey_to_combo(a.text)})
        elif a.name == "type":
            text = a.text
            submit = text.endswith("\n")
            if text.rstrip("\n"):
                go("type", {"text": text.rstrip("\n")})
            if submit:
                go("key", {"text": "Return"})
        elif a.name == "scroll":
            inp = {"scroll_direction": a.direction or "down", "scroll_amount": 5}
            if a.point:
                inp["coordinate"] = at(a.point)
            go("scroll", inp)
        elif a.name == "wait":
            go("wait", {"duration": 2})
        else:
            raise ValueError(f"unsupported action {a.name}")
