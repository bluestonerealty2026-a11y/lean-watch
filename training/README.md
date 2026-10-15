# Fine-tuning an open model for LeanWatch

LeanWatch's brain is an open vision-language model from Hugging Face, running on your own computer.
No API key, and no screenshots leave the machine. Out of the box it uses
[UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) (Apache-2.0), a GUI agent
built on Qwen2.5-VL-7B. This folder has everything needed to make it better at LeanWatch's job:
clicking the right thing, and explaining each step like a patient tutor.

```
record_practice.mjs   data engine: drives the practice computer in a headless browser and records
                      screenshot + instruction + thought + action for every step
import_hf.py          converts public GUI datasets (GroundCUA, OS-Atlas, Aguvis, AgentNet, ...) to the same format
build_dataset.py      turns recordings into chat-format training examples (grounding + multi-step actions)
train_sft.py          LoRA / QLoRA supervised fine-tuning with TRL; loss on the answer only
evaluate.py           click-inside-target accuracy, before and after training
tests/                end-to-end test of the whole pipeline with a tiny Qwen2.5-VL on a CPU
```

## About "pretraining"

Pretraining a vision-language model from scratch takes billions of image-text pairs and thousands of
GPU-days. Nobody needs to do that here. GUI agents in the literature (UI-TARS, Aguvis, OS-Atlas,
OpenCUA) all start from a pretrained VLM and train it further in stages, and this pipeline does the same:

1. **Grounding** on large public GUI datasets: "click the Save button" → one point. This is the
   "pretraining-like" stage. The model learns where things are on many different screens.
2. **Actions** on multi-step demonstrations: the instruction plus the last few screenshots → the next
   `Thought:` and `Action:`. LeanWatch's own recordings teach the tutoring style, the `ask_user`
   action for risky steps, and the practice apps.
3. Optional and later: **reinforcement learning** (e.g. GRPO), with the practice computer as the
   environment and a reward when the task's check passes.

Mix 1 and 2 in one run (`build_dataset.py --practice a b c`) or train them one after the other.

## Quick start

```bash
pip install -r training/requirements.txt        # torch, transformers, peft, trl, datasets
pip install bitsandbytes                        # NVIDIA only: needed for --4bit (QLoRA)
npm install                                     # Playwright, for the data engine

# 1. Record demonstrations on the practice computer
node training/record_practice.mjs --episodes 2000 --out data/practice --scale 1.5

# 2. Optional: add public desktop data (see the table below)
python training/import_hf.py aguvis --dataset smolagents/aguvis-stage-1 --config omniact --out data/aguvis-omniact --limit 5000

# 3. Build the training set
python training/build_dataset.py --practice data/practice data/aguvis-omniact --out data/sft

# 4. Measure the base model first, so you know if training helped
python training/evaluate.py --data data/sft --split val --limit 200 --4bit

# 5. Fine-tune (QLoRA: 4-bit base + LoRA adapters)
python training/train_sft.py --data data/sft --out runs/uitars-leanwatch --4bit

# 6. Measure again, then use it
python training/evaluate.py --data data/sft --split val --limit 200 --adapter runs/uitars-leanwatch --4bit
cd agent && python -m leanwatch --adapter ../runs/uitars-leanwatch --4bit
```

The adapter folder holds `leanwatch.json` (base model, coordinate format, point syntax, image size cap)
and the processor it was trained with. The agent reads both, so its prompts and screenshots match the
training data exactly. With `--merge`, `train_sft.py` also writes a standalone model to
`runs/.../merged`, which you can share or load with `--model`. After a `--4bit` run the merge loads a
16-bit copy of the base on the CPU (about 17 GB of RAM for a 7B model), because 4-bit weights can't hold
the merged result exactly.

## Which model, for which computer

| Your hardware | Run LeanWatch with | Fine-tune with |
| --- | --- | --- |
| NVIDIA, 8–12 GB | UI-TARS-1.5-7B `--4bit` (about 6 GB) | Rent a 24 GB GPU, or train Qwen2.5-VL-3B |
| NVIDIA, 16 GB (e.g. Colab T4) | UI-TARS-1.5-7B `--4bit` | QLoRA with `--max-pixels 602112 --history-images 1` in `build_dataset.py` |
| NVIDIA, 24 GB (RTX 3090/4090, L4, A10G) | UI-TARS-1.5-7B in bf16 (about 17 GB) | QLoRA `--4bit` with the defaults |
| NVIDIA, 40–80 GB (A100, H100) | anything above | LoRA in bf16 (drop `--4bit`), larger batches |
| Apple Silicon, 32 GB+ | UI-TARS-1.5-7B in fp16 on `mps` (slower than a GPU) | `train_sft.py` falls back to the CPU here; use a cloud GPU |
| CPU only | Works, but takes minutes per step | Not practical beyond the tests |

These are estimates: memory grows with the image size cap (`--max-pixels`) and the number of screenshots
kept in the conversation (`--history-images`). If training runs out of memory, lower those first.

Other bases. The pipeline works with the Qwen vision-language family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL and
models built on them), whose processors report each image's size (`image_grid_thw`), which LeanWatch uses
to map points back to your screen. Other families (LLaVA, SmolVLM, ...) are not supported yet.
- **Qwen2.5-VL-3B/7B-Instruct**, the base UI-TARS-1.5 was built on: same coordinate format (`--coords abs`),
  but it does not speak the UI-TARS action language yet, so give it plenty of data.
- **Qwen3-VL** models write points on a 0–1000 grid: build the dataset with `--coords rel1000`. Their image
  patches are 16 px instead of 14, which only `rel1000` handles correctly. Not tested here yet.
- Newer Qwen3.5-based GUI agents (UI-Mate, UI-Venus-2, Holo) are stronger, but they answer in their own
  tool-call formats. Supporting them is on the roadmap.

## Datasets

Grounding and desktop trajectories, all for **desktop** screens (LeanWatch controls a computer, not a phone).
Check each dataset card for the current license before using it, especially for commercial use.

| Dataset | What it has | Importer |
| --- | --- | --- |
| [ServiceNow/GroundCUA](https://huggingface.co/datasets/ServiceNow/GroundCUA) | Dense desktop grounding: elements with boxes, pixel coordinates. MIT | `groundcua` |
| [OS-Copilot/OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) (desktop part) | Large cross-platform grounding corpus, SeeClick-style JSON | `seeclick` |
| [showlab/ShowUI-desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop) | Desktop element lists | `showui` |
| [smolagents/aguvis-stage-1](https://huggingface.co/datasets/smolagents/aguvis-stage-1), [-stage-2](https://huggingface.co/datasets/smolagents/aguvis-stage-2) | Grounding (stage 1) and multi-step reasoning trajectories (stage 2), many sources in one action language | `aguvis` |
| [xlangai/AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) | 22.5K desktop tasks on Windows, macOS and Ubuntu, with thoughts. MIT | `agentnet` |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2), [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | Grounding **benchmarks** (Pro: professional apps like Photoshop and Blender) | `screenspot`. Evaluate on these, never train on them |

Some popular GUI datasets (for example UGround-V1-Data and GUI-Actor-Data) are non-commercial only.

Always run an importer with `--limit 20` first and look at the sample it prints, to check the files you
downloaded have the layout the importer expects.

## What the training examples look like

`build_dataset.py` uses the same function the agent uses at run time (`leanwatch.local_prompts.build_messages`),
so training and inference see the same conversation:

```text
user:       <UI-TARS-1.5 computer-use prompt with LeanWatch's tutoring notes> + "## User Instruction\nTurn on dark mode"
user:       (earlier screenshot)          <- older turns keep their answers but drop the image
assistant:  Thought: Dark mode lives in Settings, so I open it from the taskbar first.
            Action: click(start_box='<|box_start|>(41,697)<|box_end|>')
user:       <screenshot>
assistant:  Thought: Settings is open on Display. The Dark mode switch is on the right; I click it.
            Action: click(start_box='<|box_start|>(822,231)<|box_end|>')       <- the loss is computed on this answer only
```

Points are pixels of the screenshot **after** the processor's resize (Qwen2.5-VL's `smart_resize`), which is
why the image size cap is fixed at dataset-build time and saved with the model. `--style point` writes
`point='<point>x y</point>'` instead of `start_box=...`, as in the newer UI-TARS prompt template. The agent's
parser reads both styles.

## What to try next

- Hold out whole apps, not just episodes, to see whether the model generalises to software it has never seen.
- Compare grounding-only, actions-only and mixed training on the same validation set.
- Evaluate on ScreenSpot-Pro. Import it, turn it into an evaluation set with the **same** image size cap as
  your training data, then score it:
  ```bash
  python training/import_hf.py screenspot --path raw/ScreenSpot-Pro --out data/screenspot-pro --limit 0
  python training/build_dataset.py --practice data/screenspot-pro --out data/ss-eval --tasks grounding --val-fraction 1
  python training/evaluate.py --data data/ss-eval --split val --tasks grounding --adapter runs/uitars-leanwatch --4bit
  ```
- Reinforcement learning with GRPO: sample several answers per screen, reward the ones whose click lands
  inside the target box (grounding) or whose episode passes its check (practice computer).
