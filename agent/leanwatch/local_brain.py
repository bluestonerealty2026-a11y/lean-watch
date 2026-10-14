"""The default brain: an open vision-language GUI model running on your own computer.

No API key and no screenshots leave the machine. Works with UI-TARS-1.5 out of the box
and with any model you fine-tune with training/train_sft.py (pass its folder as --model,
or the LoRA folder as --adapter).

Each step: take a screenshot -> build the conversation (training/inference share
local_prompts.build_messages) -> the model answers "Thought: ... Action: ..." ->
show the thought to the person -> safety checks -> move the real mouse/keyboard.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from .actions import Parsed, ParseError, hotkey_to_combo, parse_response
from .config import Config
from .controller import Controller, EmergencyStop
from .hooks import Hooks, RunResult, Stopped, check_stop, run_action
from .local_prompts import APPROVED_NOTE, build_messages, computer_prompt
from .permissions import PermissionDenied

# Thoughts that announce something hard to undo. The person approves these first,
# even if the model forgot to call ask_user.
RISKY_INTENT = re.compile(
    r"\b(delet\w*|remov\w*|eras\w*|uninstall\w*|install\w*|format\w*|wip\w*|"
    r"purchas\w*|buy\w*|pay|paying|payment|checkout|send\w*|submit\w*|post(?:ing)?|publish\w*|"
    r"empty(?:ing)? the (?:trash|recycle bin)|shut(?:ting)? down|restart\w*|sign(?:ing)? out|log(?:ging)? out|"
    r"password|credit card|transfer\w*)\b",
    re.IGNORECASE,
)
POINTER_ACTIONS = {"click", "left_single", "left_double", "right_single", "drag", "scroll", "type", "hotkey", "press"}


def risky_intent(thought: str) -> str | None:
    m = RISKY_INTENT.search(thought or "")
    return m.group(0).lower() if m else None


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
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
            ip = self.processor.image_processor
            ip.max_pixels = max_pixels
            if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
                ip.size["longest_edge"] = max_pixels
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        if "device_map" not in kwargs:
            self.model.to("mps" if (mps and device != "cpu") else "cpu")
        self.model.eval()
        self.patch = getattr(self.processor.image_processor, "patch_size", 14)

    def answer(self, messages: list[dict], images: list) -> ModelAnswer:
        import torch

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        # The last image is the current screenshot; its grid tells us the size the model saw.
        _, gh, gw = inputs["image_grid_thw"][-1].tolist()
        return ModelAnswer(reply.strip(), int(gw * self.patch), int(gh * self.patch))


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
        prompt = computer_prompt(task, self.cfg.language)
        screenshots: list = []
        past: list[dict] = []
        bad_answers = 0
        try:
            for _ in range(self.cfg.max_turns):
                check_stop(self.stop)
                self.hooks.wait_if_paused()
                screenshots.append(self.controller.capture.screenshot())
                messages, image_turns = build_messages(prompt, past, self.cfg.history_images)
                t0 = time.monotonic()
                answer = self.model.answer(messages, [screenshots[i] for i in image_turns])
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
        if self.cfg.coords == "rel1000":
            sx, sy = g.shot_w / 1000, g.shot_h / 1000

        def at(p):
            return [round(p[0] * sx), round(p[1] * sy)]

        asked = False
        for a in parsed.actions:
            check_stop(self.stop)
            if a.name in ("finished",):
                return RunResult(True, a.text or parsed.thought or "Done."), None
            if a.name in ("ask_user", "call_user"):
                question = a.text or parsed.thought or "May I continue?"
                if not self.hooks.confirm(question, "The assistant wants your OK before this step."):
                    return RunResult(False, f"You said no to: {question} Nothing more was done."), None
                return None, APPROVED_NOTE
            if a.name in POINTER_ACTIONS and not asked:
                reason = risky_intent(parsed.thought)
                if reason:
                    asked = True
                    if not self.hooks.confirm(parsed.thought, f"This step mentions “{reason}”, which can be hard to undo."):
                        return RunResult(False, "You said no, so I stopped before that step."), None
            try:
                self._do(a, at)
            except PermissionDenied as err:
                return None, f"That action was refused: {err}"
            except (Stopped, EmergencyStop):
                raise
            except Exception as err:  # bad coordinates, unknown key, ...
                return None, f"That action failed: {err}. Try a different way."
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
