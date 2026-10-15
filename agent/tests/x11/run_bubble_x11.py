"""Drive the real LeanWatch bubble on a virtual X display with real mouse clicks and typing.

Needs: Xvfb, openbox, xdotool, xterm (and xcompmgr for the "comp" mode), plus the agent's
requirements and pyautogui. Run from the repository root:

    python agent/tests/x11/run_bubble_x11.py /tmp/lw-x11 plain   # no compositor: shaped windows
    python agent/tests/x11/run_bubble_x11.py /tmp/lw-x11 comp    # with a compositor: see-through windows

Screenshots and logs go to the folder given; exit code 0 means every check passed.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

S = Path(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "plain"   # plain (no compositor) or comp
HERE = Path(__file__).resolve().parent
DISP = ":98" if mode == "comp" else ":99"
OUT = S / f"run-{mode}"; OUT.mkdir(parents=True, exist_ok=True)
import shutil
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
os.environ["DISPLAY"] = DISP
env = {**os.environ, "HOME": str(OUT / "home"), "PYTHONUNBUFFERED": "1"}
procs = []
def start(cmd, **kw):
    p = subprocess.Popen([str(c) for c in cmd], env=kw.pop("env", env), stdout=kw.pop("stdout", subprocess.DEVNULL), stderr=subprocess.STDOUT, **kw)
    procs.append(p); return p
def xdo(*a):
    return subprocess.run(["xdotool", *map(str, a)], env=env, capture_output=True, text=True).stdout.strip()
results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

try:
    start(["Xvfb", DISP, "-screen", "0", "1366x768x24", "-nolisten", "tcp"]); time.sleep(1.2)
    start(["openbox"]); time.sleep(0.8)
    if mode == "comp":
        start(["xcompmgr", "-n"]); time.sleep(0.8)
    start(["xterm", "-geometry", "70x18+60+60", "-bg", "white", "-fg", "black", "-fa", "DejaVu Sans Mono", "-fs", "12", "-T", "Terminal"])
    time.sleep(1.5)
    wid = xdo("search", "--name", "^Terminal$").split()[0]
    xdo("windowactivate", "--sync", wid)
    geo = xdo("getwindowgeometry", wid)
    (x, y), (w, h) = map(int, re.search(r"Position: (\d+),(\d+)", geo).groups()), map(int, re.search(r"Geometry: (\d+)x(\d+)", geo).groups())
    pos = re.search(r"Position: (\d+),(\d+)", geo).groups(); size = re.search(r"Geometry: (\d+)x(\d+)", geo).groups()
    tx, ty = int(pos[0]) + int(size[0]) // 2, int(pos[1]) + int(size[1]) // 2
    import mss
    from PIL import Image
    sct = mss.mss()
    def shot(name):
        raw = sct.grab(sct.monitors[1]); img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"); img.save(OUT / f"{name}.png"); return img
    def state():
        try: return json.loads((OUT / "state.json").read_text())
        except Exception: return {}
    def wait_for(fn, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            st = state()
            if st and fn(st): return st
            time.sleep(0.1)
        return state()

    log = open(OUT / "driver.out", "w")
    start([sys.executable, HERE / "driver.py", OUT], env={**env, "XTERM_CENTRE": json.dumps([tx, ty]), "PORT": "8799" if mode == "plain" else "8798"}, stdout=log)
    st = wait_for(lambda s: s["bubble"][4])
    check("bubble appears", st.get("bubble", [0] * 5)[4], f"at {st.get('bubble')}; translucent={st.get('translucent')}")
    time.sleep(0.6)
    img = shot("01-desktop-with-bubble")
    bx, by, bw, bh, _ = st["bubble"]
    px = img.getpixel((bx + bw // 2 - 12, by + bh // 2 + 8))
    check("bubble is teal on screen, bottom-right", abs(px[0] - 0x3B) < 30 and abs(px[1] - 0xBF) < 30 and bx > 1366 - 160 and by > 768 - 200, f"pixel {px}")
    active_before = xdo("getactivewindow", "getwindowname")
    check("the person's app keeps focus while the bubble floats", active_before == "Terminal", active_before)

    # 1. click the bubble
    xdo("mousemove", "--sync", bx + bw // 2, by + bh // 2); time.sleep(0.2); xdo("click", "1")
    st = wait_for(lambda s: s["expanded"] and s["panel"][4], 5)
    time.sleep(0.5)
    shot("02-box-open")
    check("click opens the command box", st.get("expanded") and st["panel"][4] and not st["bubble"][4])
    active = xdo("getactivewindow", "getwindowname")
    check("the box has the keyboard", active == "LeanWatch", active)

    # 2. type a command and press Enter
    xdo("type", "--delay", "40", "say hello in the terminal"); xdo("key", "Return")
    st = wait_for(lambda s: s["consent_yes"][2], 5); time.sleep(0.4)
    shot("03-permission")
    check("first command asks permission", st["consent_yes"][2])

    # 3. allow
    cx, cy, _ = st["consent_yes"]; xdo("mousemove", "--sync", cx, cy); time.sleep(0.2); xdo("click", "1")
    st = wait_for(lambda s: s["running"], 5)
    check("task starts after Allow", st.get("running"))
    time.sleep(0.4)
    active = xdo("getactivewindow", "getwindowname")
    check("keyboard handed back to the person's app", active == "Terminal", active)

    n = 0
    end = time.time() + 60
    while time.time() < end:
        st = state()
        if not st.get("running") and "Done" in st.get("notice", ""):
            break
        if st.get("steps"):
            shot(f"04-running-{n:02d}"); n += 1
        time.sleep(0.8)
    time.sleep(0.6)
    shot("05-finished")
    st = state()
    check("task finishes with the summary in the box", "Done: the terminal said hello" in st.get("notice", ""), st.get("notice"))
    check("steps were shown", st.get("steps", "").count("</div>") >= 3)

    # 4. what the model saw: never the bubble or the box
    teal_counts = []
    for f in sorted(OUT.glob("model_saw_*.png")):
        im = Image.open(f).convert("RGB")
        teal = sum(1 for p in im.getdata() if abs(p[0] - 0x3B) < 12 and abs(p[1] - 0xBF) < 12 and abs(p[2] - 0xB3) < 12)
        teal_counts.append((f.name, teal))
    check("model screenshots never show LeanWatch", teal_counts and all(t < 20 for _, t in teal_counts), str(teal_counts))
    events = (OUT / "events.log").read_text()
    hides = events.count("shield hide")
    check("click under the box: the box stepped aside", "shield hide ['leanwatch-panel']" in events, f"{hides} hides")

    # 5. Esc closes the box and the bubble comes back
    xdo("mousemove", "--sync", *st["input"][:2]); xdo("click", "1"); time.sleep(0.3)
    xdo("key", "Escape")
    st = wait_for(lambda s: not s["expanded"], 5)
    check("Esc shrinks the box back to the bubble", not st.get("expanded") and st["bubble"][4])

    # 6. drag the bubble to the top-left and check the box opens downward there
    bx, by, bw, bh, _ = st["bubble"]
    xdo("mousemove", "--sync", bx + bw // 2, by + bh // 2); xdo("mousedown", "1")
    for k in range(1, 21):
        xdo("mousemove", "--sync", bx + bw // 2 - k * 55, by + bh // 2 - k * 30); time.sleep(0.02)
    xdo("mouseup", "1")
    saved_file = OUT / "home" / ".leanwatch" / "bubble.json"
    end = time.time() + 5
    while time.time() < end and not saved_file.exists():
        time.sleep(0.1)
    time.sleep(0.4)
    st = state()
    check("drag moves the bubble without opening it", st["bubble"][0] < 400 and not st["expanded"], str(st["bubble"]))
    saved = json.loads((OUT / "home" / ".leanwatch" / "bubble.json").read_text())
    check("new spot is remembered", abs(saved["x"] - st["bubble"][0]) <= 2, str(saved))
    bx, by, bw, bh, _ = st["bubble"]
    xdo("mousemove", "--sync", bx + bw // 2, by + bh // 2); xdo("click", "1")
    st = wait_for(lambda s: s["expanded"], 5); time.sleep(0.4)
    shot("06-box-top-left")
    check("box opens toward the middle of the screen", st["panel"][1] <= by + 2 and st["panel"][0] <= bx + 2, str(st["panel"]))
finally:
    for p in reversed(procs):
        p.terminate()
    time.sleep(0.5)
    for p in procs:
        if p.poll() is None: p.kill()
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks passed")
sys.exit(0 if results and all(ok for _, ok, _ in results) else 1)
