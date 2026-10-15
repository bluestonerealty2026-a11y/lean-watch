"""Local web server: serves the control center page and bridges it to the agent.

Security, because this process can move your mouse:
- listens on 127.0.0.1 only;
- rejects any Host header other than 127.0.0.1/localhost (blocks DNS rebinding);
- every WebSocket and API call needs the random session token, which is only
  in the URL this program opens and in ~/.leanwatch/session.json (mode 600);
- WebSocket connections must come from this page's own origin.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import threading
import uuid
import webbrowser
from pathlib import Path

from aiohttp import WSMsgType, web

from .brains import load_local_model, make_brain, model_name
from .hooks import Hooks, RunResult
from .config import Config
from .controller import Controller, DryRunBackend, PyAutoGuiBackend
from .permissions import Permissions
from .screen import Geometry, ScreenCapture, jpeg_data_url

MAX_TASK_CHARS = 2000


def api_hint(cfg: Config | None = None) -> tuple[bool, str]:
    """Whether the chosen brain can run. The local brain needs no key."""
    if cfg is None or cfg.brain != "claude":
        return True, ""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, ""
    if (Path.home() / ".config" / "anthropic").exists():
        return True, ""
    return False, "Set ANTHROPIC_API_KEY (or run `ant auth login`) and restart the agent."


class Session(Hooks):
    """One agent on this computer, shared by every connected page and the overlay."""

    def __init__(self, cfg: Config, token: str):
        self.cfg = cfg
        self.token = token
        self.permissions = Permissions()
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.paused_flag = False
        self.running = False
        self.thread: threading.Thread | None = None
        self.pending: dict[str, dict] = {}
        self.last_narration = ""
        self.controller: Controller | None = None
        self.local_model = None  # loaded on the first task (or by --preload), then kept
        self._model_lock = threading.Lock()
        # In-process listeners (the desktop bubble). Called from any thread with every event.
        self.listeners: list = []
        # Set by the desktop bubble so it never shows up in the model's screenshots or under its clicks.
        self.shield = None
        self.screen_size, self.screen_error = self._probe_screen()

    # ---------------- setup ----------------
    def _make_backend(self):
        if self.cfg.dry_run:
            size = self.screen_size if getattr(self, "screen_size", None) else (1920, 1080)
            return DryRunBackend(size)
        return PyAutoGuiBackend()

    def _probe_screen(self):
        try:
            return PyAutoGuiBackend().size(), None
        except Exception as err:  # no display, or pyautogui missing
            return None, f"Cannot reach the screen: {err}"

    # ---------------- messaging ----------------
    def status(self) -> dict:
        ready, hint = api_hint(self.cfg)
        w_h = self.screen_size
        return {
            "type": "status",
            "brain": self.cfg.brain,
            "model": model_name(self.cfg),
            "effort": self.cfg.effort,
            "dry_run": self.cfg.dry_run,
            "permission": self.permissions.as_dict(),
            "running": self.running,
            "paused": self.paused_flag,
            "screen": {"width": w_h[0], "height": w_h[1]} if w_h else None,
            "screen_error": self.screen_error,
            "api_ready": ready,
            "api_hint": hint,
            "last": self.last_narration,
        }

    def emit(self, msg: dict) -> None:
        """Send to every page and in-process listener. Safe to call from any thread."""
        for listener in list(self.listeners):
            try:
                listener(msg)
            except Exception:
                pass  # a broken listener must never stop a run
        if self.loop is None or self.loop.is_closed() or not self.clients:
            return
        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._broadcast(msg)))
        except RuntimeError:  # the server loop closed between the check and the call
            pass

    async def _broadcast(self, msg: dict) -> None:
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    def handle(self, msg: dict) -> dict | None:
        """Apply a message from a page or the overlay. Returns an immediate reply, if any."""
        kind = msg.get("type")
        if kind == "hello":
            return self.status()
        if kind == "grant":
            scopes = msg.get("scopes") or {}
            self.permissions.grant(mouse=bool(scopes.get("mouse", True)), keyboard=bool(scopes.get("keyboard", True)))
            self.emit(self.status())
            return None
        if kind == "revoke":
            self.permissions.revoke()
            self.stop()
            self.emit(self.status())
            return None
        if kind == "command":
            return self.start(str(msg.get("text", "")))
        if kind == "stop":
            self.stop()
            return None
        if kind == "pause":
            if self.running and not self.paused_flag:
                self.paused("Paused. Press Resume to continue.")
            return None
        if kind == "resume":
            self.resume()
            return None
        if kind == "confirm":
            p = self.pending.get(str(msg.get("id")))
            if p:
                p["allow"] = bool(msg.get("allow"))
                p["event"].set()
            return None
        return {"type": "error", "message": f"Unknown message type {kind!r}."}

    # ---------------- runs ----------------
    def start(self, text: str) -> dict | None:
        text = text.strip()
        if not text:
            return {"type": "error", "message": "Type what you would like me to do."}
        if len(text) > MAX_TASK_CHARS:
            return {"type": "error", "message": "That request is too long. Keep it under 2000 characters."}
        if not self.permissions.granted:
            return {"type": "error", "message": "I need your permission first. Click Allow in LeanWatch."}
        if self.running:
            return {"type": "error", "message": "I'm already working on something. Press Stop first."}
        self.stop_event.clear()
        self.resume_event.set()
        self.paused_flag = False
        self.running = True
        self.thread = threading.Thread(target=self._worker, args=(text,), daemon=True, name="leanwatch-run")
        self.thread.start()
        self.emit({"type": "run_started", "text": text})
        self.emit(self.status())
        return None

    def _worker(self, text: str) -> None:
        try:
            backend = self._make_backend()
            w, h = backend.size()
            geometry = Geometry.for_screen(w, h, self.cfg.screenshot_long_edge, self.cfg.screenshot_max_pixels)
            # Created in this thread: mss handles are per-thread on Windows.
            capture = ScreenCapture(geometry, shield=self.shield)
            self.controller = Controller(
                backend, capture, geometry, self.permissions,
                speed=self.cfg.move_speed, takeover_px=self.cfg.takeover_tolerance_px,
                confirm=self.confirm, stop=self.stop_event, shield=self.shield,
            )
            self.preview()
            if self.cfg.brain == "local":
                self.ensure_model()
            brain = make_brain(self.cfg, self.controller, self, self.stop_event, local_model=self.local_model)
            result = brain.run(text)
        except Exception as err:
            result = RunResult(False, f"Could not run the task: {err}")
        finally:
            self.running = False
            self.paused_flag = False
            for p in list(self.pending.values()):
                p["event"].set()
        if result.ok and result.message:
            self.narrate(result.message, "final")
        self.emit({"type": "run_finished", "ok": result.ok, "message": result.message if not result.ok else "",
                   "result": result.message})
        self.emit(self.status())

    def ensure_model(self):
        """Load the local model once, even if a task and --preload ask at the same time."""
        with self._model_lock:
            if self.local_model is None:
                self.narrate(f"Loading {self.cfg.local_model}. The first time this downloads several GB and can take a few minutes.", "thinking")
                self.local_model = load_local_model(self.cfg)
            return self.local_model

    def stop(self) -> None:
        self.stop_event.set()
        self.resume_event.set()
        for p in list(self.pending.values()):
            p["event"].set()

    def resume(self) -> None:
        if self.paused_flag:
            self.paused_flag = False
            self.resume_event.set()
            self.emit({"type": "resumed"})

    # ---------------- Hooks used by the brain ----------------
    def narrate(self, text: str, kind: str = "say") -> None:
        if kind != "thinking":
            self.last_narration = text
        self.emit({"type": "narration", "text": text, "kind": kind})

    def action(self, name: str, inp: dict, summary: str) -> None:
        self.emit({"type": "action", "name": name, "summary": summary})

    def preview(self) -> None:
        c = self.controller
        if c is None or c.capture is None or not self.clients:
            return  # previews are only for the web page; skip the extra screenshot when no page is open
        try:
            img = c.capture.screenshot()
            fx, fy = c.cursor_fraction()
            self.emit({"type": "screenshot", "image": jpeg_data_url(img), "cursor": {"fx": fx, "fy": fy}})
        except Exception:
            pass  # previews are a convenience; never fail a run over one

    def confirm(self, action: str, reason: str) -> bool:
        cid = uuid.uuid4().hex[:12]
        event = threading.Event()
        self.pending[cid] = {"event": event, "allow": False}
        self.emit({"type": "confirm_request", "id": cid, "action": action, "reason": reason})
        event.wait(self.cfg.confirm_timeout_s)
        allow = self.pending.pop(cid, {}).get("allow", False) and not self.stop_event.is_set()
        self.narrate(f"You {'allowed' if allow else 'declined'}: {action}.", "thinking")
        if self.controller is not None:
            # The person moved the mouse to answer; that is not taking over, so don't pause for it.
            self.controller.resync()
        return allow

    def paused(self, reason: str) -> None:
        self.paused_flag = True
        self.resume_event.clear()
        self.emit({"type": "paused", "reason": reason})

    def wait_if_paused(self) -> None:
        while not self.resume_event.wait(0.2):
            if self.stop_event.is_set():
                return


SESSION = web.AppKey("session", Session)


# ---------------------------------------------------------------------------
def make_app(cfg: Config, token: str) -> web.Application:
    session = Session(cfg, token)
    hosts = {f"127.0.0.1:{cfg.port}", f"localhost:{cfg.port}"}
    origins = {f"http://{h}" for h in hosts}

    @web.middleware
    async def guard(request: web.Request, handler):
        if request.host not in hosts:
            return web.Response(status=403, text="LeanWatch only answers on 127.0.0.1.")
        if request.path.startswith(("/api/", "/ws")):
            given = request.headers.get("X-LeanWatch-Token") or request.query.get("token") or ""
            if not hmac.compare_digest(given.encode(), token.encode()):
                return web.Response(status=403, text="Missing or wrong session token.")
            origin = request.headers.get("Origin")
            if origin is not None and origin not in origins:
                return web.Response(status=403, text="Requests from other websites are not allowed.")
        response = await handler(request)
        if not response.prepared:
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    async def index(_request):
        return web.FileResponse(cfg.web_dir / "index.html")

    async def ws_handler(request: web.Request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=64 * 1024)
        await ws.prepare(request)
        session.clients.add(ws)
        await ws.send_json(session.status())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                reply = session.handle(data) if isinstance(data, dict) else None
                if reply:
                    await ws.send_json(reply)
        finally:
            session.clients.discard(ws)
        return ws

    async def api_status(_request):
        return web.json_response(session.status())

    async def api_post(request: web.Request):
        kind = request.match_info["kind"]
        if kind not in ("command", "stop", "pause", "resume", "grant", "revoke"):
            raise web.HTTPNotFound()
        try:
            body = await request.json() if request.can_read_body else {}
        except ValueError:
            body = {}
        if kind == "grant":
            body = {"scopes": body}
        reply = session.handle({**(body if isinstance(body, dict) else {}), "type": kind})
        if reply and reply.get("type") == "error":
            return web.json_response(reply, status=409)
        return web.json_response({"ok": True})

    async def on_startup(app):
        session.loop = asyncio.get_running_loop()

    async def on_shutdown(app):
        session.stop()
        for ws in list(session.clients):
            await ws.close()

    app = web.Application(middlewares=[guard])
    app[SESSION] = session
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/{kind}", api_post)
    app.router.add_static("/css/", cfg.web_dir / "css")
    app.router.add_static("/js/", cfg.web_dir / "js")
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def write_session_file(cfg: Config, token: str) -> Path:
    path = cfg.session_file
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"port": cfg.port, "token": token, "pid": os.getpid()}, f)
    return path


def serve(cfg: Config) -> None:
    """Web-only mode (--no-bubble): the control center page in your browser."""
    token = secrets.token_urlsafe(24)
    session_path = write_session_file(cfg, token)
    url = f"http://127.0.0.1:{cfg.port}/?token={token}"
    app = make_app(cfg, token)
    session = app[SESSION]

    print("LeanWatch local agent")
    print(f"  Control center: {url}")
    brain = "local open model" if cfg.brain == "local" else f"Claude API (effort {cfg.effort})"
    print(f"  Brain: {brain}: {model_name(cfg)}{'   DRY RUN: the mouse will not move' if cfg.dry_run else ''}")
    if session.screen_error:
        print(f"  Warning: {session.screen_error}")
    ready, hint = api_hint(cfg)
    if not ready:
        print(f"  Warning: {hint}")
    print("  Stop a task: Esc in the page, the Stop button, or push the mouse into a screen corner.")
    print("  Quit: Ctrl+C")

    async def after_start(_app):
        if cfg.open_browser:
            webbrowser.open(url)

    async def cleanup(_app):
        try:
            session_path.unlink()
        except OSError:
            pass

    app.on_startup.append(after_start)
    app.on_cleanup.append(cleanup)
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)
