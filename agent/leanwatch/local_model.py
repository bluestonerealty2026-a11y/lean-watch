"""The open vision-language model itself: load it once, then one answer per step.

Kept light on purpose: nothing here imports Qt, pyautogui or the controller, so the model can run in
its own process (model_process.py). torch and transformers are imported only when a model loads.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .memory import choose_cpu_dtype

END_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_text|>", "<|eot_id|>")
# float32 copy of the bfloat16 weights, made one block at a time: 16M numbers = 64 MB.
FLOAT32_BLOCK = 1 << 24


def coords_for(model_type: str) -> str:
    """How a model family writes points. Qwen2.5-VL (UI-TARS-1.5) uses pixels of the image it saw;
    Qwen2-VL (UI-TARS 2B/7B v1) and Qwen3-VL use a 0-1000 grid (see UI-TARS' own action_parser.py)."""
    return "abs" if model_type in ("qwen2_5_vl", "qwen2_5_vl_text") else "rel1000"


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


def float32_arithmetic(model) -> None:
    """Keep a CPU model's weights in bfloat16 (half the memory) but do its arithmetic in float32.

    Most CPUs have no fast bfloat16 arithmetic: without AVX-512, PyTorch 2.5 multiplies a screenshot's worth
    of bfloat16 numbers about 100 times slower than float32 ones. Instead every Linear layer copies its weight
    into one shared float32 buffer, a block at a time, just for its own multiplication; the token embedding
    hands on float32 numbers; the small layers (norms, biases, the patch embedding) become float32.
    For UI-TARS-2B on 4 AVX2 cores, one step then takes about as long as in float32 (125 s against 113 s),
    and the process peaks at 5.5 GB of memory instead of 13 GB.
    """
    import torch
    import torch.nn.functional as F

    shared = {}

    def buffer(rows: int, cols: int):
        buf = shared.get("buf")
        if buf is None or buf.numel() < rows * cols:
            buf = shared["buf"] = torch.empty(rows * cols, dtype=torch.float32)
        return buf[: rows * cols].view(rows, cols)

    def linear(module, x):
        weight, bias = module.weight, module.bias
        x = x.float()
        if weight.dtype == torch.float32:
            return F.linear(x, weight, bias)
        n, k = weight.shape
        step = max(1, min(n, FLOAT32_BLOCK // k))
        if step == n:
            return F.linear(x, buffer(n, k).copy_(weight), bias)
        out = x.new_empty(*x.shape[:-1], n)
        for i in range(0, n, step):
            j = min(n, i + step)
            out[..., i:j] = F.linear(x, buffer(j - i, k).copy_(weight[i:j]), None if bias is None else bias[i:j])
        return out

    def float_output(forward):
        return lambda *args, **kwargs: forward(*args, **kwargs).float()

    for module in model.modules():
        kept = None
        if type(module) is torch.nn.Linear:
            kept = module.weight
            module.forward = linear.__get__(module)
        elif type(module) is torch.nn.Embedding:
            kept = module.weight
            module.forward = float_output(module.forward)
        for name, p in module.named_parameters(recurse=False):
            if p is not kept and p.dtype == torch.bfloat16:
                p.data = p.data.float()
        for name, b in module.named_buffers(recurse=False):
            if b.dtype == torch.bfloat16:
                setattr(module, name, b.float())


class LocalModel:
    """Loads a Hugging Face vision-language model once and answers one step at a time.

    cpu_dtype: weight precision on a CPU, "float32", "bfloat16" or "auto" (from the memory, see memory.py).
    status: called with one line saying which precision was chosen and why.
    """

    def __init__(self, model_id: str, adapter: str | None = None, device: str = "auto", load_in_4bit: bool = False,
                 max_pixels: int | None = None, max_new_tokens: int = 512, cpu_dtype: str = "float32", status=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        on_cpu = device == "cpu" or not (cuda or mps)
        kwargs: dict = {}
        if on_cpu:
            dtype, why = choose_cpu_dtype(model_id, cpu_dtype)
            kwargs["dtype"] = getattr(torch, dtype)
            if status:
                status(why)
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
        if on_cpu and kwargs["dtype"] == torch.bfloat16 and not load_in_4bit:
            float32_arithmetic(self.model)
        self.model.eval()
        self.patch = getattr(self.processor.image_processor, "patch_size", 14)
        self.coords = coords_for(getattr(self.model.config, "model_type", ""))
        self.info = {"device": str(self.model.device), "dtype": str(kwargs["dtype"]).replace("torch.", "")}

    def answer(self, messages: list[dict], images: list, stop: threading.Event | None = None) -> ModelAnswer:
        """stop: anything with is_set(), such as a threading.Event or a multiprocessing Event."""
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
