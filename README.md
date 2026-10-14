# LeanWatch

Type what you want to do. LeanWatch moves the cursor, clicks and types for you while you lean back and watch, and it explains every step in plain words. When it finishes, press **Now you try**: the screen rewinds and LeanWatch outlines each spot so you can do it with your own hands.

It is for people who find video tutorials hard to follow: editing a photo, rotating a model in a 3D app, using the command prompt, or just finding a setting.

The project has two parts:

| Part | What it does | Needs |
| --- | --- | --- |
| `web/` control center | Command box in the bottom-right corner, permission screen, step-by-step notes, a **practice computer** with five apps, build-progress tracker | Just a browser: https://bluestonerealty2026-a11y.github.io/lean-watch/ or `web/index.html` |
| `agent/` local agent | Controls your **real** mouse and keyboard with Claude's computer-use toolset, serves the control center, and adds an always-on-top command box to every screen | Python 3.10+, a Claude API key |

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
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # or: ant auth login
python -m leanwatch                     # opens the control center, already connected
```

In the command box choose **My computer**, give permission, and type a task.

| Option | What it does |
| --- | --- |
| `--overlay` | Also shows the floating command box in the bottom-right corner of every screen |
| `--dry-run` | Plans and narrates on your real screen but never moves the mouse or presses keys. Start here. |
| `--cli "task"` | Runs one task in the terminal, no browser |
| `--speed 0.6` | Slower pointer, easier to follow |
| `--model`, `--effort` | Which Claude model to use (default `claude-opus-5-5`) and how hard it thinks (default `medium`) |

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
  brain.py             Claude computer-use loop with narrate and request_confirmation tools
  controller.py        executes the 17 computer-use actions with pyautogui
  screen.py            screenshots, scaling and coordinate mapping
  permissions.py       consent scopes and risky-action rules
  server.py            aiohttp server: page, WebSocket bridge, API for the overlay
  overlay.py           always-on-top command box (tkinter)
  __main__.py          command line
```

- **Practice screen**: the planner matches your sentence to the closest example sentence of each skill (cosine similarity of TF-IDF vectors), pulls out details like numbers, names and colours, and builds a list of steps. The executor only acts through the pointer: it moves to a point and clicks whatever is under it, as a person would.
- **Your computer**: `brain.py` sends a scaled screenshot to Claude with the `computer_toolset_20260801` toolset. Claude answers with actions (`left_click`, `type`, `scroll`, …) plus a `narrate` call explaining each one; `controller.py` maps the coordinates back to your screen and moves the real pointer along the same human-like path. The history is append-only so prompt caching and thinking blocks stay valid. Two optional API betas are on by default and switch themselves off if your account lacks them: progress notes between tool calls (`thinking.display: "updates"`), and server-side fallback (`fallbacks: "default"`), which re-runs a request on another Claude model if a safety filter declines it.

## Tests

```bash
cd agent && pip install -r requirements-dev.txt && python -m pytest   # 63 tests, no API key or display needed
npm install && npm test                                               # 26 browser checks on the practice screen
```

## Publishing

`.github/workflows/static.yml` publishes the `web/` folder to GitHub Pages on every push to `claude/bold-bell-tq6s9b` (the default branch; add others to `on.push.branches`). It first serves the exact files under `/lean-watch/` and runs the browser smoke test against them, so a broken page is never deployed. In the repository's Settings → Pages, **Source** must be **GitHub Actions**.

## Roadmap

Open the **Build progress** tab for the live list. The next milestones are the first live run on real Windows, macOS and Linux desktops, then the machine-learning track: record demonstrations on the practice screen, train a UI-element detector and a behaviour-cloning policy in PyTorch, fine-tune with reinforcement learning using the practice screen as the environment, and benchmark the result against Claude.
