"""Build a tiny, randomly initialised Qwen2.5-VL (the architecture UI-TARS-1.5 uses), or a Qwen2-VL
(UI-TARS-2B-SFT, the --small model) with family="qwen2_vl".

It has the real model class, processor, chat template and special tokens, but a
few thousand parameters instead of 7 billion, so the training and inference code
can be tested end to end on a CPU without downloading anything.

    python training/tests/tiny_model.py /tmp/tiny-qwen
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/, for leanwatch_paths

SPECIAL = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|image_pad|>", "<|video_pad|>", "<|box_start|>", "<|box_end|>",
]

# The official Qwen2.5-VL chat template, trimmed of video handling.
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _corpus():
    from leanwatch_paths import agent_path  # noqa: F401  (adds agent/ to sys.path)
    from leanwatch.local_prompts import computer_prompt, grounding_prompt

    lines = [f(style) for style in ("box", "point")
             for f in (lambda s: computer_prompt("Turn on dark mode", "English", s), lambda s: grounding_prompt("Click OK.", s))]
    lines += [f"Thought: I click the button.\nAction: click(start_box='<|box_start|>({i},{i * 7 % 997})<|box_end|>')" for i in range(1000)]
    lines += ["type(content='hello') hotkey(key='ctrl z') drag(start_box=, end_box=) scroll(direction='down') finished(content='Done.')"] * 20
    return lines


def build(out_dir: str | Path, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 64, family: str = "qwen2_5_vl") -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        PreTrainedTokenizerFast, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
        Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor,
    )

    qwen2 = family == "qwen2_vl"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(_corpus(), trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[3:], padding_side="left",
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    image_processor = Qwen2VLImageProcessor(min_pixels=min_pixels, max_pixels=max_pixels)
    processor = (Qwen2VLProcessor if qwen2 else Qwen2_5_VLProcessor)(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor(),
        chat_template=CHAT_TEMPLATE)

    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
    vision = (dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2) if qwen2 else
              dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                   fullatt_block_indexes=[1], window_size=112))
    config = (Qwen2VLConfig if qwen2 else Qwen2_5_VLConfig)(
        text_config=dict(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0, "mrope_section": [2, 3, 3]},
            eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"], bos_token_id=None,
        ),
        vision_config=vision,
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"], bos_token_id=None,
        **({"tie_word_embeddings": True} if qwen2 else {}),     # as in UI-TARS-2B-SFT
    )
    model = (Qwen2VLForConditionalGeneration if qwen2 else Qwen2_5_VLForConditionalGeneration)(config)
    model.generation_config.eos_token_id = ids["<|im_end|>"]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-qwen"))
