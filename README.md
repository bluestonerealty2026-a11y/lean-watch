# LeanWatch

Type what you want to do. LeanWatch moves the cursor, clicks and types for you while you lean back and watch, and it explains every step in plain words. When it finishes, press **Now you try**: the screen rewinds and LeanWatch outlines each spot so you can do it with your own hands.

It is for people who find video tutorials hard to follow: editing a photo, rotating a model in a 3D app, using the command prompt, or just finding a setting.

The project has two parts:

| Part | What it does | Needs |
| --- | --- | --- |
| `web/` control center | Command box in the bottom-right corner, permission screen, step-by-step notes, a **practice computer** with five apps, build-progress tracker | Just a browser: https://bluestonerealty2026-a11y.github.io/lean-watch/ or `web/index.html` |
| `agent/` local agent | Controls your **real** mouse and keyboard with an open vision-language model running on your computer, serves the control center, and adds an always-on-top command box to every screen | Python 3.10+, about 6 GB of GPU memory (or more RAM and patience). No API key |
| `training/` | Records training data on the practice computer, imports public GUI datasets, and fine-tunes the open model with LoRA / QLoRA | A GPU for real runs; see [training/README.md](training/README.md) |

## Try it in your browser

Live: **https://bluestonerealty2026-a11y.github.io/lean-watch/**. Or open `web/index.html` (double-click it). Nothing to install.

1. Type a task in the box at the bottom right, or pick an example: *Turn on dark mode*, *Make my photo black and white*, *Rotate the cube 45 degrees*, *Create a folder called Homework using the terminal*.
2. Click **Allow** on the permission screen.
3. Watch. Each step is explained beside the screen, with how to do the same thing in Windows, macOS, Photoshop or Blender.
4. Press **Now you try** to practise it yourself.

Press <kbd>/</kbd> to jump to the command box and <kbd>Esc</kbd> to stop. The practice computer has Settings, PhotoLab (a Photoshop-style editor), Blocks3D (a Blender-style 3D editor), Command Prompt and Files.

## Run it on your real computer

```bash
cd agent
pip install -r requirements.txt         # with an NVIDIA GPU, install the CUDA build of torch first
python -m leanwatch                     # opens the control center, already connected
```

In the command box choose **My computer**, give permission, and type a task. The first task downloads
[UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) (Apache-2.0, about 16 GB) from
Hugging Face. Everything then runs on your computer: no API key, and no screenshots leave the machine.
On an NVIDIA GPU with less than 24 GB, add `--4bit` (`pip install bitsandbytes` first) to fit it in about
6 GB. On an Apple Silicon Mac it runs on the GPU (`mps`) as is.

| Option | What it does |
| --- | --- |
| `--overlay` | Also shows the floating command box in the bottom-right corner of every screen |
| `--dry-run` | Plans and narrates on your real screen but never moves the mouse or presses keys. Start here. |
| `--cli "task"` | Runs one task in the terminal, no browser |
| `--speed 0.6` | Slower pointer, easier to follow |
| `--adapter runs/my-model` | Use a model you fine-tuned with `training/train_sft.py` (it remembers its own settings) |
| `--model` | Another Hugging Face model id or a local folder (default `ByteDance-Seed/UI-TARS-1.5-7B`) |
| `--device`, `--max-pixels` | Where the model runs (`auto`, `cuda`, `mps`, `cpu`) and the largest screenshot it sees |
| `--language Spanish` | Language of the explanations |
| `--brain claude` | Optional: use the Claude API instead (`pip install -r requirements-claude.txt`, needs `ANTHROPIC_API_KEY`) |

### Platform notes

- **macOS**: give your terminal app permission under System Settings → Privacy & Security → **Accessibility** (to move the mouse) and **Screen Recording** (to take screenshots), then restart the terminal. Retina screens are handled: screenshots are scaled from the logical screen size.
- **Windows**: the agent turns on DPI awareness so the pointer and screenshots use the same pixels when display scaling is on. Run the terminal as your normal user, not as administrator.
- **Linux**: needs an X11 session. pyautogui cannot control the mouse under Wayland. The overlay needs `python3-tk`.

## Safety

- Nothing moves until you grant permission, and you choose pointer and keyboard separately.
- Stop at any time: <kbd>Esc</kbd> in the page, the Stop button, or push the mouse into a screen corner (pyautogui's fail-safe).
- If you move the mouse yourself, the agent pauses and waits for **Resume**.
- Deleting, sending, buying, installing and changing security settings need your **Allow**. Typed text that looks destructive (`rm -rf`, `format`, `sudo`, …) is held back for approval even if the model didn't ask.
- Text on web pages and in documents is treated as information, never as instructions, to resist prompt injection.
- The agent listens only on `127.0.0.1`, rejects other Host headers and other websites' origins, and needs a random session token that only the page it opens (and `~/.leanwatch/session.json`, mode 600) knows.

## How it works

Both modes run the same loop a robot does: **look** at the screen, **decide** the next move, **act**, then **check** the result.

```
web/
  index.html           control center page
  css/styles.css
  js/simos.js          practice computer: windows, taskbar, shared files, settings
  js/apps/*.js         Settings, PhotoLab, Blocks3D, Command Prompt, Files
  js/cursor.js         pointer motion: Fitts's law timing + minimum-jerk Bezier path
  js/planner.js        TF-IDF nearest-neighbour intent classifier, slot filling, 29 skills
  js/agent.js          runs plans by moving, clicking, typing and dragging; "Now you try" coaching
  js/bridge.js         WebSocket link to the local agent
  js/progress.js       build tracker and live checks
  js/main.js           wiring
  tests/smoke.mjs      Playwright end-to-end test

agent/leanwatch/
  local_brain.py       default brain: open GUI model on this computer (screenshot -> Thought + Action)
  actions.py           UI-TARS action language: parse answers, write training targets, coordinate spaces
  local_prompts.py     the prompt and conversation, shared by training and the agent
  brains.py            picks the brain; reads a fine-tuned model's saved settings
  brain.py             optional Claude computer-use brain
  controller.py        executes the 17 computer-use actions with pyautogui
  screen.py            screenshots, scaling and coordinate mapping
  permissions.py       consent scopes and risky-action rules
  server.py            aiohttp server: page, WebSocket bridge, API for the overlay
  overlay.py           always-on-top command box (tkinter)
  __main__.py          command line

training/              data engine, dataset importers, LoRA/QLoRA fine-tuning, evaluation
```

- **Practice screen**: the planner matches your sentence to the closest example sentence of each skill (cosine similarity of TF-IDF vectors), pulls out details like numbers, names and colours, and builds a list of steps. The executor only acts through the pointer: it moves to a point and clicks whatever is under it, as a person would.
- **Your computer**: `local_brain.py` gives the model the task and the last three screenshots. The model answers in the UI-TARS action language, for example `Thought: The Display page is open, so I click the Dark mode switch.` then `Action: click(start_box='<|box_start|>(822,231)<|box_end|>')`. LeanWatch shows the thought to you, maps the point from the model's resized image back to your screen, and `controller.py` moves the real pointer along the same human-like path. A step whose thought mentions deleting, sending, buying, installing and the like needs your Allow even if the model forgot to ask (`ask_user`).
- **Training your own**: `training/record_practice.mjs` drives the practice computer in a headless browser and records every step (screenshot, instruction, explanation, action). `build_dataset.py` turns recordings and public datasets into chat examples, `train_sft.py` fine-tunes UI-TARS-1.5 with LoRA or QLoRA (TRL), and `evaluate.py` measures click-inside-target accuracy before and after. The full guide, with model and dataset choices per GPU, is in [training/README.md](training/README.md).

## Tests

```bash
cd agent && pip install -r requirements-dev.txt && python -m pytest   # no API key, display or model download needed
cd training && python -m pytest                                       # whole training pipeline with a tiny model on CPU
npm install && npm test                                               # 26 browser checks on the practice screen
```

## Publishing

`.github/workflows/static.yml` publishes the `web/` folder to GitHub Pages on every push to `claude/bold-bell-tq6s9b` (the default branch; add others to `on.push.branches`). It first serves the exact files under `/lean-watch/` and runs the browser smoke test against them, so a broken page is never deployed. In the repository's Settings → Pages, **Source** must be **GitHub Actions**.

## Roadmap

Open the **Build progress** tab for the live list. Next: the first live run on real Windows, macOS and Linux desktops, the first full fine-tune of UI-TARS-1.5-7B on a GPU, support for newer Qwen3.5-based GUI models, reinforcement learning with the practice computer as the environment, and benchmarks (ScreenSpot-Pro, OSWorld-G).
