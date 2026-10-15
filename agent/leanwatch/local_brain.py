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
from dataclasses import dataclass

from .actions import Parsed, ParseError, hotkey_to_combo, parse_response
from .config import Config
from .controller import Controller, EmergencyStop
from .hooks import Hooks, RunResult, Stopped, check_stop, run_action
from .local_prompts import APPROVED_NOTE, build_messages, computer_prompt
from .permissions import PermissionDenied
from .risk import language_checked, risky_intent  # noqa: F401  (risky_intent is re-exported for tests)

# Actions that change nothing on the screen: no approval needed for these.
PASSIVE_ACTIONS = {"wait", "move_to", "mouse_move", "finished", "ask_user", "call_user"}
# After the person approves an ask_user step, the next turns carry it out without asking again
# (e.g. "click Delete", then "the app asks to confirm, click Delete").
APPROVAL_TURNS = 2
END_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_text|>", "<|eot_id|>")


def coords_for(model_type: str) -> str:
    """How a model family writes points. Qwen2.5-VL (UI-TARS-1.5) uses pixels of the image it saw;
    Qwen2-VL (UI-TARS 2B/7B v1) and Qwen3-VL use a 0-1000 grid (see UI-TARS' own action_parser.py)."""
    return "abs" if model_type in ("qwen2_5_vl", "qwen2_5_vl_text") else "rel1000"


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


def clean_reply(text: str, extra_markers=()) -> str:
    """Cut a raw decoded answer at the end-of-turn marker, keeping other special tokens.

    The answer is decoded *with* special tokens because UI-TARS writes points as
    <|box_start|>(x,y)<|box_end|>; the agent stores the answer as history, and history must look
    exactly like the training data, which keeps those tokens.
    """
    markers = [m for m in (*END_MARKERS, *extra_markers) if m]
    cut = min((i for i in (text.find(m) for m in markers) if i >= 0), default=len(text))
    text = text[:cut]
    for m in markers:
        text = text.replace(m, "")
    return text.strip()


def set_image_size(processor, max_pixels: int | None = None, min_pixels: int | None = None) -> None:
    """Cap the screenshot size a Qwen-style processor resizes to (coordinates depend on it).

    transformers 5 keeps the cap in image_processor.size (shortest_edge/longest_edge, in pixels);
    the old min_pixels/max_pixels attributes still work but are deprecated, so set the size.
    """
    ip = processor.image_processor
    size = getattr(ip, "size", None)
    if size is not None and "longest_edge" in dict(size):
        new = {k: v for k, v in dict(size).items() if v is not None}
        if max_pixels:
            new["longest_edge"] = max_pixels
        if min_pixels:
            new["shortest_edge"] = min_pixels
        ip.size = type(size)(**new) if not isinstance(size, dict) else new
    for name, value in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
        if value and getattr(ip, name, None) is not None:   # older processors that keep the attributes
            setattr(ip, name, value)


@dataclass
class ModelAnswer:
    text: str
    image_w: int   # size of the image the model actually saw (after its own resize)
    image_h: int


class LocalModel:
    """Loads a Hugging Face vision-language model once and answers one step at a time."""

    def __init__(self, model_id: str, adapter: str | None = None, device: str = "auto", load_in_4bit: bool = False,
                 max_pixels: int | None = None, max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        kwargs: dict = {}
        if device == "cpu" or not (cuda or mps):
            kwargs["dtype"] = torch.float32
        elif cuda:
            # bf16 only where the GPU supports it natively (Ampere or newer); a T4 would emulate it slowly.
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = torch.float16
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=kwargs.get("dtype", torch.float16))
            kwargs["device_map"] = "auto"

        # An adapter folder from train_sft.py carries the processor it was trained with (same image size cap).
        from pathlib import Path

        has_processor = adapter and any((Path(adapter) / f).exists() for f in ("processor_config.json", "preprocessor_config.json"))
        self.processor = AutoProcessor.from_pretrained(adapter if has_processor else model_id)
        if max_pixels:
            set_image_size(self.processor, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        if "device_map" not in kwargs:
            self.model.to("mps" if (mps and device != "cpu") else "cpu")
        self.model.eval()
        self.patch = getattr(self.processor.image_processor, "patch_size", 14)
        self.coords = coords_for(getattr(self.model.config, "model_type", ""))

    def answer(self, messages: list[dict], images: list, stop: threading.Event | None = None) -> ModelAnswer:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        if "image_grid_thw" not in inputs:
            raise RuntimeError(f"{self.model_id} does not report image sizes (image_grid_thw). LeanWatch works with "
                               "Qwen2-VL, Qwen2.5-VL (UI-TARS-1.5) and Qwen3-VL family models.")

        class StopWhenAsked(StoppingCriteria):
            """Lets Stop end a long generation early instead of waiting for the full answer."""

            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), bool(stop and stop.is_set()), dtype=torch.bool, device=input_ids.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      stopping_criteria=StoppingCriteriaList([StopWhenAsked()]))
        tok = getattr(self.processor, "tokenizer", self.processor)
        raw = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        reply = clean_reply(raw, (getattr(tok, "eos_token", None), getattr(tok, "pad_token", None)))
        # The last image is the current screenshot; its grid tells us the size the model saw.
        _, gh, gw = inputs["image_grid_thw"][-1].tolist()
        return ModelAnswer(reply, int(gw * self.patch), int(gh * self.patch))


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
