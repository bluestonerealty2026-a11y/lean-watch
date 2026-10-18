"""Drive the real LeanWatch bubble in a headless GNOME Shell (Wayland), with real input through the
desktop's remote desktop portal and screen frames through PipeWire.

Needs root in a container (it hides /run/systemd in a private mount namespace and may start the
system D-Bus), plus: gnome-shell, xdg-desktop-portal(-gnome), pipewire, wireplumber, xwayland,
gstreamer1.0-pipewire, python3-gi with GTK 4, and LeanWatch's requirements with jeepney.

    python agent/tests/gnome/run_gnome.py /tmp/lw-gnome

Screenshots and logs go to that folder; exit code 0 means every check passed.
"""

import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shell  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "home").mkdir(parents=True)
results = []
procs = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def system_bus():
    """Containers often have no system bus; GNOME Shell needs one to start."""
    try:
        s = socket.socket(socket.AF_UNIX)
        s.connect("/run/dbus/system_bus_socket")
        s.close()
        return
    except OSError:
        pass
    for f in ("/run/dbus/system_bus_socket", "/run/dbus/pid"):
        try:
            os.remove(f)
        except OSError:
            pass
    os.makedirs("/run/dbus", exist_ok=True)
    subprocess.run(["dbus-daemon", "--system", "--fork"], check=True)


def gtk_python():
    for exe in ["/usr/bin/python3"] + sorted(glob.glob("/usr/bin/python3.[0-9]*"), reverse=True):
        if re.search(r"-config$", exe) or not os.access(exe, os.X_OK):
            continue
        probe = "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk"
        if subprocess.run([exe, "-s", "-c", probe], capture_output=True).returncode == 0:
            return exe
    raise SystemExit("No Python with GTK 4 bindings (sudo apt install python3-gi gir1.2-gtk-4.0)")


def state():
    try:
        return json.loads((OUT / "state.json").read_text())
    except Exception:
        return {}


def wait_for(fn, timeout=20.0, what=None):
    end = time.time() + timeout
    while time.time() < end:
        try:
            v = fn()
            if v:
                return v
        except Exception:
            pass
        time.sleep(0.15)
    return None


def app_log():
    try:
        return [json.loads(line) for line in (OUT / "testapp.jsonl").read_text().splitlines() if line.strip()]
    except Exception:
        return []


def events():
    try:
        return (OUT / "events.log").read_text()
    except OSError:
        return ""


def script(task, steps):
    for s in steps:
        s.setdefault("task", task)
    (OUT / "script.json").write_text(json.dumps(steps))


def submit(text):
    """Type a request into the LeanWatch box (opening it first) like a person would."""
    for _ in range(3):          # under XWayland the first click on a new window is sometimes lost
        st = state()
        if st.get("expanded") and st["panel"][4]:
            break
        bx, by, bw, bh, _ = st["bubble"]
        shell.click(bx + bw // 2, by + bh // 2)
        wait_for(lambda: state().get("expanded") and state()["panel"][4], 3)
    for _ in range(3):
        ix, iy, _ = state()["input"]
        shell.click(ix, iy)
        wait_for(lambda: state().get("panel_active"), 3)
        shell.type_text(text)
        if wait_for(lambda: state().get("input_text") == text, 3):
            break
        shell.key(0xFF57)                         # End, then clear what got there, and try again
        for _ in range(len(state().get("input_text", "")) + 2):
            shell.key(0xFF08)
    shell.type_text("\n")


def sharing_streams():
    out = subprocess.run(["pw-dump"], capture_output=True, text=True, env=os.environ).stdout or "[]"
    nodes = [o for o in json.loads(out) if o.get("type", "").endswith("Node")]
    return [n for n in nodes if ((n.get("info") or {}).get("props") or {}).get("media.class") == "Stream/Output/Video"]


def dialog_buttons(d):
    """GNOME's Remote Desktop dialog (xdg-desktop-portal-gnome): the switch and the Share button."""
    return (d["x"] + d["w"] - 77, d["y"] + 157), (d["x"] + d["w"] - 45, d["y"] + 23)


session = None
try:
    if os.geteuid() != 0:
        raise SystemExit("run_gnome.py needs root (a private mount namespace for the headless session)")
    system_bus()
    runtime = tempfile.mkdtemp(dir=f"/run/user/{os.getuid()}" if os.path.isdir(f"/run/user/{os.getuid()}") else "/tmp",
                               prefix="lw")
    os.chmod(runtime, 0o700)
    env = {**os.environ, "XDG_RUNTIME_DIR": runtime, "HOME": str(OUT / "home"), "OUT": str(OUT)}
    for k in ("WAYLAND_DISPLAY", "DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "XAUTHORITY", "LEANWATCH_WAYLAND"):
        env.pop(k, None)
    session = subprocess.Popen(
        ["unshare", "-m", "--propagation", "private", "bash", "-c",
         f"mount -t tmpfs none /run/systemd && exec dbus-run-session -- bash {HERE / 'session.sh'}"],
        env=env, stdout=open(OUT / "session.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    ok = wait_for(lambda: (OUT / "session.env").exists() and (OUT / "session.env").read_text().strip(), 90)
    senv = dict(line.split("=", 1) for line in (OUT / "session.env").read_text().splitlines() if "=" in line)
    if not ok or "DBUS_SESSION_BUS_ADDRESS" not in senv:
        check("GNOME Shell starts headless", False, (OUT / "shell.log").read_text()[-1500:] if (OUT / "shell.log").exists() else "")
        raise SystemExit(1)
    env.update(senv)
    os.environ.update({k: env[k] for k in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "WAYLAND_DISPLAY", "DISPLAY",
                                           "XAUTHORITY", "HOME") if env.get(k)})
    check("GNOME Shell starts headless", True, f"Wayland {senv['WAYLAND_DISPLAY']}, XWayland {senv['DISPLAY']}")
    # PipeWire's session manager links the screen stream to whoever reads it.
    procs.append(subprocess.Popen(["wireplumber"], env=env, stdout=open(OUT / "wireplumber.log", "w"), stderr=subprocess.STDOUT))
    shell.ev("Main.overview.hide(); 1")

    procs.append(subprocess.Popen([gtk_python(), "-s", str(HERE / "testapp.py"), str(OUT / "testapp.jsonl")],
                                  env={**env, "GDK_BACKEND": "wayland"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    win = shell.window("LeanWatch Wayland test")
    check("a native Wayland app is open", win and win["wayland"], str(win))
    entry = (win["x"] + win["w"] // 2, win["y"] + 54)
    button = (win["x"] + win["w"] // 2, win["y"] + 96)
    area = (win["x"] + 200, win["y"] + 300)

    denv = {**env, "QT_QPA_PLATFORM": "xcb", "XDG_SESSION_TYPE": "wayland", "PYTHONUNBUFFERED": "1"}
    procs.append(subprocess.Popen([sys.executable, "-W", "ignore", str(HERE / "driver_gnome.py"), str(OUT)], env=denv,
                                  stdout=open(OUT / "driver.out", "w"), stderr=subprocess.STDOUT))
    st = wait_for(lambda: state().get("bubble", [0] * 5)[4] and state(), 30)
    check("the bubble appears on the Wayland desktop", st, (OUT / "driver.out").read_text()[-800:] if not st else "")
    if not st:
        raise SystemExit(1)
    # GNOME opens its overview at start-up (it hides ordinary windows): make sure it is closed now.
    wait_for(lambda: shell.ev("Main.overview.hide(); Main.overview.visible") is False, 15)
    time.sleep(0.5)
    check("LeanWatch knows it is on Wayland", st["wayland"])
    check("no keyboard shortcut is promised on Wayland without GNOME's own", st["shortcut_ok"] is False)
    time.sleep(0.8)
    shell.screenshot(str(OUT / "01-desktop.png"))
    from PIL import Image

    img = Image.open(OUT / "01-desktop.png").convert("RGB")
    bx, by, bw, bh, _ = st["bubble"]
    px = img.getpixel((bx + bw // 2 - 12, by + bh // 2 + 8))
    check("the bubble is teal, bottom-right", abs(px[0] - 0x3B) < 40 and abs(px[1] - 0xBF) < 40 and bx > 1280 - 160, f"pixel {px}")

    # ---- task 1: first time, the desktop asks; then click, hand the keyboard back, type ----
    script("t1", [
        {"text": f"Thought: I click the text box.\nAction: click(start_box='<|box_start|>({entry[0]},{entry[1]})<|box_end|>')"},
        {"text": "Thought: I type a greeting and press Enter.\nAction: type(content='hello wayland\\n')", "wait_for": "go1"},
        {"text": "Thought: Done.\nAction: finished(content='I typed the greeting.')"},
    ])
    submit("say hello in the text box")
    st = wait_for(lambda: state().get("consent_yes", [0, 0, False])[2] and state(), 8)
    if st:
        shell.click(*st["consent_yes"][:2])
    dialog = shell.window("Remote Desktop", 20)
    shell.screenshot(str(OUT / "02-desktop-asks.png"))
    check("the desktop asks for permission the first time", dialog)
    check("LeanWatch says to turn on “Allow Remote Interaction”", "Allow Remote Interaction" in events())
    if dialog:
        switch, share = dialog_buttons(dialog)
        shell.click(*switch)
        time.sleep(0.5)
        shell.click(*share)
    check("the first click lands in the native app",
          wait_for(lambda: "model answer 1" in events() or "model answer 0" in events(), 30) is not None)
    time.sleep(1.5)
    # The person clicks into LeanWatch's box while the model thinks: its typing must still go to the app.
    ix, iy, _ = state()["input"]
    shell.click(ix, iy)
    check("the box took the keyboard", wait_for(lambda: state().get("panel_active"), 5))
    (OUT / "go1").write_text("")
    got = wait_for(lambda: [e for e in app_log() if e.get("ev") == "enter"], 30)
    check("typing reaches the native Wayland app", got and got[-1]["text"] == "hello wayland", str(got))
    check("the box stepped aside to give the keyboard back", "step aside" in events())
    done = wait_for(lambda: "run_finished" in events() and not state().get("running"), 30)
    check("task 1 finishes", done and '"ok": true' in events().split("run_finished")[1][:40], events()[-600:])
    shell.screenshot(str(OUT / "03-after-task-1.png"))
    saw = sorted(OUT.glob("model_saw_t1_*.png"))
    pg = state()["panel"]
    if saw:
        img = Image.open(saw[0]).convert("RGB")
        cx, cy = pg[0] + pg[2] // 2, pg[1] + pg[3] // 2
        p_panel = img.getpixel((min(cx, img.width - 1), min(cy, img.height - 1)))
        p_app = img.getpixel((min(area[0], img.width - 1), min(area[1], img.height - 1)))
        check("the model's screenshot shows the app (not a black screen)", sum(p_app) > 600, f"pixel {p_app}")
        check("the model's screenshot never shows LeanWatch's box", not (p_panel[0] < 40 and p_panel[1] < 50 and p_panel[2] < 55),
              f"pixel at the box {p_panel}")
    else:
        check("the model got screenshots", False)
    check("the screen sharing ends with the task", wait_for(lambda: not sharing_streams(), 10) is not None)

    # ---- task 2: remembered permission, no dialog ----
    script("t2", [
        {"text": f"Thought: I press the button.\nAction: click(start_box='<|box_start|>({button[0]},{button[1]})<|box_end|>')"},
        {"text": "Thought: Done.\nAction: finished(content='Pressed it.')"},
    ])
    before = len(events())
    submit("press the button")
    check("the second task needs no dialog", shell.window("Remote Desktop", 3) is None)
    check("the click reaches the native app", wait_for(lambda: [e for e in app_log() if e.get("ev") == "button"], 30))
    wait_for(lambda: "run_finished" in events()[before:] and not state().get("running"), 30)

    # ---- task 3: the person stops the sharing from the desktop ----
    script("t3", [{"text": "Thought: I wait for the page.\nAction: wait()"}])
    before = len(events())
    submit("wait for something")
    wait_for(lambda: "model answer 1" in events()[before:], 30)
    shell.stop_sharing()
    stopped = wait_for(lambda: "run_finished" in events()[before:] and not state().get("running"), 20)
    tail = events()[before:]
    check("stopping the sharing from the desktop stops LeanWatch",
          stopped and "stopped the screen sharing" in tail, tail[-500:])

    # ---- the open command (for a GNOME keyboard shortcut) ----
    was = state().get("expanded")
    r = subprocess.run([sys.executable, str(HERE.parents[1] / "leanwatch" / "open_box.py")], env=env, capture_output=True, text=True)
    check("python -m leanwatch --open toggles the box", r.returncode == 0 and wait_for(lambda: state().get("expanded") != was, 5),
          r.stderr)
    shell.screenshot(str(OUT / "04-end.png"))
finally:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(5)
        except Exception:
            p.kill()
    if session is not None:
        try:
            os.killpg(session.pid, signal.SIGTERM)
            session.wait(10)
        except Exception:
            try:
                os.killpg(session.pid, signal.SIGKILL)
            except Exception:
                pass

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""))
sys.exit(1 if failed or not results else 0)
