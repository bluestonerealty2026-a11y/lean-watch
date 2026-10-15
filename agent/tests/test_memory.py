import sys
from pathlib import Path

import pytest

from leanwatch.memory import choose_cpu_dtype, memory_advice, need_gb, param_count

BIG, SMALL = "ByteDance-Seed/UI-TARS-1.5-7B", "ByteDance-Seed/UI-TARS-2B-SFT"


@pytest.mark.parametrize("model_id,params", [
    (BIG, 8.3e9), (SMALL, 2.2e9), ("ByteDance-Seed/UI-TARS-7B-DPO", 7e9), ("Qwen/Qwen2.5-VL-3B-Instruct", 3e9),
    ("someone/tiny-3.5B-vl", 3.5e9), ("Qwen/Qwen3-VL-30B-A3B-Instruct", 30e9), ("runs/my-model", None),
    ("org/model-bf16-4bit", None), ("/home/me/models/72b", 72e9),
])
def test_parameter_count_from_table_or_name(model_id, params):
    assert param_count(model_id) == params


def test_8_gb_laptop_loads_the_small_model_in_16_bit():
    dtype, why = choose_cpu_dtype(SMALL, "auto", ram=8.2)
    assert dtype == "bfloat16"
    assert why == ("Loading UI-TARS-2B-SFT in 16-bit (bfloat16) to fit in this computer's 8 GB of memory "
                   "(about 6 GB; 32-bit would need about 12 GB).")


@pytest.mark.parametrize("model_id,ram,dtype", [
    (SMALL, 16.7, "float32"), (SMALL, 12.0, "bfloat16"), (BIG, 16.7, "bfloat16"), (BIG, 64.0, "float32"),
    ("runs/my-model", 64.0, "bfloat16"),       # size unknown: the safe choice
])
def test_float32_only_when_it_fits_with_room_to_spare(model_id, ram, dtype):
    assert choose_cpu_dtype(model_id, "auto", ram=ram)[0] == dtype
    assert dtype != "float32" or ram >= param_count(model_id) * 4 / 1e9 + 3.5


def test_explicit_precision_wins_and_bad_values_are_explained():
    assert choose_cpu_dtype(BIG, "float32", ram=8)[0] == "float32"
    assert choose_cpu_dtype(SMALL, "bfloat16", ram=512)[0] == "bfloat16"
    with pytest.raises(ValueError, match="auto, float32 or bfloat16"):
        choose_cpu_dtype(SMALL, "fp8")


def test_needs():
    assert round(need_gb(2.2e9, "bfloat16")) == 6 and round(need_gb(2.2e9, "float32")) == 12
    assert round(need_gb(8.3e9, "float32")) == 37


def test_memory_advice_fits_the_situation(monkeypatch):
    import leanwatch.memory as memory

    monkeypatch.setattr(memory, "ram_gb", lambda: 8.2)
    assert memory_advice(BIG) == ("Close big apps such as your web browser or use the small model (--small). "
                                  "UI-TARS-1.5-7B in 16-bit needs about 18 GB on a CPU.")
    assert memory_advice(SMALL, "float32").startswith(
        "Close big apps such as your web browser or load it in 16-bit (--cpu-dtype bfloat16).")


def test_config_and_command_line(monkeypatch):
    from leanwatch.config import Config

    monkeypatch.setenv("LEANWATCH_CPU_DTYPE", "bfloat16")
    monkeypatch.setenv("LEANWATCH_MODEL_PROCESS", "0")
    cfg = Config()
    assert cfg.cpu_dtype == "bfloat16" and cfg.model_process is False
    monkeypatch.delenv("LEANWATCH_CPU_DTYPE")
    monkeypatch.delenv("LEANWATCH_MODEL_PROCESS")
    cfg = Config()
    assert cfg.cpu_dtype == "auto" and cfg.model_process is True

    import leanwatch.server as server
    from leanwatch.__main__ import main

    seen = []
    monkeypatch.setattr(server, "serve", seen.append)
    main(["--small", "--cpu-dtype", "float32", "--no-bubble", "--no-browser"])
    assert seen[-1].cpu_dtype == "float32"


def test_bfloat16_weights_with_float32_arithmetic_match_float32(tmp_path):
    """LeanWatch's 16-bit CPU mode: weights stored in bfloat16, every multiplication done in float32.

    It must give exactly the answer of a float32 model holding the same (bfloat16-rounded) weights, and
    keep the big matrices in bfloat16. Plain bfloat16 arithmetic is ~100x slower on CPUs without AVX-512.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "tests"))
    from PIL import Image
    from tiny_model import build
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from leanwatch.local_model import FLOAT32_BLOCK, float32_arithmetic
    from leanwatch.local_prompts import build_messages, computer_prompt

    path = build(tmp_path / "tiny", family="qwen2_vl")
    processor = AutoProcessor.from_pretrained(path)
    messages, _ = build_messages(computer_prompt("Turn on dark mode"), [], 3)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[Image.new("RGB", (640, 400), "gray")], return_tensors="pt")

    ours = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
    float32_arithmetic(ours)
    reference = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).float().eval()
    linear = [m for m in ours.modules() if type(m) is torch.nn.Linear]
    assert linear and all(m.weight.dtype == torch.bfloat16 for m in linear)
    assert ours.get_input_embeddings().weight.dtype == torch.bfloat16      # tied with the output layer, kept 16-bit
    with torch.inference_mode():
        a = ours(**inputs).logits
        b = reference(**inputs).logits
        assert a.dtype == torch.float32 and torch.allclose(a, b, atol=1e-5)
        out = ours.generate(**inputs, max_new_tokens=6, do_sample=False)
        ref = reference.generate(**inputs, max_new_tokens=6, do_sample=False)
    assert out.tolist() == ref.tolist()

    # Big matrices are copied a block at a time (the output layer of a real model is 150k x 1.5k).
    import leanwatch.local_model as local_model

    layer = torch.nn.Linear(64, 1000).to(torch.bfloat16)
    expected = torch.nn.functional.linear(torch.ones(3, 64), layer.weight.float(), layer.bias.float())
    old = FLOAT32_BLOCK
    try:
        local_model.FLOAT32_BLOCK = 64 * 300           # 4 blocks: 300 + 300 + 300 + 100 rows
        float32_arithmetic(layer)
        assert torch.allclose(layer(torch.ones(3, 64)), expected, atol=1e-6)
    finally:
        local_model.FLOAT32_BLOCK = old
