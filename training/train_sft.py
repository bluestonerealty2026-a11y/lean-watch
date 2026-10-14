"""Fine-tune an open vision-language model to drive LeanWatch (LoRA / QLoRA SFT with TRL).

    # 1. record data on the practice computer, 2. build the dataset, 3. train
    node training/record_practice.mjs --episodes 2000 --out data/practice --scale 1.5
    python training/build_dataset.py --practice data/practice --out data/sft
    python training/train_sft.py --data data/sft --base ByteDance-Seed/UI-TARS-1.5-7B --out runs/uitars-leanwatch --4bit

    # 4. use it
    cd agent && python -m leanwatch --adapter ../runs/uitars-leanwatch

What it does
- Loads the base model and its processor, and sets the processor's image size cap
  to the one the dataset was built with (coordinates depend on it).
- Adds LoRA adapters to the language model's attention and MLP layers only; the
  vision encoder stays frozen. --4bit loads the base in 4-bit first (QLoRA).
- Trains with TRL's SFTTrainer on conversational prompt-completion data, so the loss
  is computed only on the answer ("Thought: ...\\nAction: ..."), never on the prompt.
- Saves the adapter, the processor and leanwatch.json (how the data was built) to --out.
  With --merge it also writes a standalone merged model to --out/merged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import leanwatch_paths  # noqa: F401

LANGUAGE_LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_split(path: Path, data_dir: Path):
    from datasets import Image as HFImage, Sequence, load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")

    def absolute(example):
        example["images"] = [p if Path(p).is_absolute() else str((data_dir / p).resolve()) for p in example["images"]]
        return example

    ds = ds.map(absolute)
    # Decode image paths into PIL images when a batch is built (TRL's vision collator expects images).
    ds = ds.cast_column("images", Sequence(HFImage()))
    keep = {"images", "prompt", "completion"}
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def portable_trainer_class():
    """SFTTrainer whose loss also works without an NVIDIA GPU.

    TRL computes the loss with a fused Triton kernel, which needs CUDA. On a Mac (MPS) or a
    CPU we fall back to the model's own cross-entropy loss, which gives the same value.
    """
    import torch
    from trl import SFTTrainer

    class PortableSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if torch.cuda.is_available():
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    return PortableSFTTrainer


def pick_precision(device: str):
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32, {"use_cpu": device == "cpu" or not torch.cuda.is_available(), "bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, {"bf16": True, "fp16": False}
    return torch.float16, {"bf16": False, "fp16": True}   # e.g. T4 / Colab free tier


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="folder from build_dataset.py")
    p.add_argument("--base", default="ByteDance-Seed/UI-TARS-1.5-7B", help="Hugging Face id or folder of the base model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", default="actions,grounding", help="which example types to train on")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (for quick tests)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1, help="examples per device per step")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA: load the base in 4-bit (needs an NVIDIA GPU and bitsandbytes)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--merge", action="store_true", help="also save base+adapter merged into --out/merged (not with --4bit)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none", help="e.g. trackio, wandb, tensorboard")
    a = p.parse_args(argv)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig

    meta = json.loads((a.data / "meta.json").read_text())
    tasks = set(a.tasks.split(","))

    processor = AutoProcessor.from_pretrained(a.base)
    ip = processor.image_processor
    # Coordinates in the dataset were written for this exact resize rule, so the processor must use it too.
    ip.min_pixels, ip.max_pixels = meta["min_pixels"], meta["max_pixels"]
    if isinstance(getattr(ip, "size", None), dict):
        ip.size = {"shortest_edge": meta["min_pixels"], "longest_edge": meta["max_pixels"]}

    dtype, precision_args = pick_precision(a.device)
    model_kwargs = {"dtype": dtype}
    if a.four_bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available() and a.device != "cpu":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(a.base, **model_kwargs)
    if a.four_bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    def only_tasks(ds_path):
        ds = load_split(ds_path, a.data)
        if tasks == {"actions", "grounding"}:
            return ds
        raw = [json.loads(line)["task"] for line in open(ds_path)]
        return ds.select([i for i, t in enumerate(raw) if t in tasks])

    train = only_tasks(a.data / "train.jsonl")
    val_path = a.data / "val.jsonl"
    val = only_tasks(val_path) if val_path.exists() and val_path.stat().st_size else None
    print(f"training on {len(train)} examples ({a.tasks}); validating on {len(val) if val else 0}")

    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
        target_modules=LANGUAGE_LORA_TARGETS, task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(a.out),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * max(1, a.max_steps if a.max_steps > 0 else len(train) // max(1, a.batch * a.grad_accum)))),
        logging_steps=5,
        save_strategy="epoch" if a.max_steps < 0 else "no",
        eval_strategy="epoch" if (val is not None and a.max_steps < 0) else "no",
        # A screenshot alone is about 1,000-2,000 tokens; TRL's default limit (1024) would cut image tokens.
        max_length=None,
        completion_only_loss=True,   # learn only the answer, never the prompt or the image tokens
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=a.report_to,
        seed=a.seed,
        dataloader_num_workers=0,
        **precision_args,
    )
    trainer = portable_trainer_class()(model=model, args=args, train_dataset=train, eval_dataset=val,
                         processing_class=processor, peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    trainer.save_model(str(a.out))
    processor.save_pretrained(str(a.out))
    (a.out / "leanwatch.json").write_text(json.dumps({
        "base_model": a.base, "coords": meta["coords"], "style": meta["style"], "max_pixels": meta["max_pixels"],
        "min_pixels": meta["min_pixels"], "history_images": meta["history_images"], "tasks": sorted(tasks),
        "train_examples": len(train), "train_loss": result.training_loss,
    }, indent=2))
    print(f"saved adapter to {a.out}  (final train loss {result.training_loss:.4f})")

    if a.merge:
        if a.four_bit:
            print("--merge skipped: merge into a 16-bit copy of the base instead of the 4-bit one.")
        else:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(a.out / "merged"))
            processor.save_pretrained(str(a.out / "merged"))
            print(f"saved merged model to {a.out / 'merged'}")


if __name__ == "__main__":
    main()
