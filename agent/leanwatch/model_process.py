"""The local model in its own process.

Why a separate process: a failed import of the AI libraries can't leave half-imported modules behind
for the next try; when the model runs out of memory, the system's out-of-memory killer ends the model's
process (the biggest one) instead of the bubble; and the bubble stays responsive while the model loads.

The child is started with "spawn", never fork (the parent runs Qt and several threads). It loads the model
with a loader given as "module:function", then answers requests over a Pipe. It imports nothing from the
bubble or the controller.
"""

from __future__ import annotations

import importlib
import multiprocessing
import multiprocessing.util
import os
import signal
import sys
import threading
import time
import traceback
import weakref

from .errors import ModelError, Stopped, details_hint, explain_load_error
from .memory import memory_advice, ram_gb

DEFAULT_LOADER = "leanwatch.local_model:LocalModel"
STOP_GRACE_S = 2.0       # after Stop, how long to wait for the model to end its answer early

_running: weakref.WeakSet = weakref.WeakSet()


class ModelProcess:
    """A model loaded in a child process, with the interface of LocalModel: model_id, coords and answer().

    Creating one starts the load and returns at once; wait_ready() (or the first answer()) waits for it.
    options go to the loader, e.g. LocalModel's adapter, device, cpu_dtype.
    """

    def __init__(self, model_id: str, loader: str = DEFAULT_LOADER, status=None, **options):
        self.model_id = model_id
        self.options = options
        self.coords = "abs"
        self.info: dict = {}
        self.stop_grace = STOP_GRACE_S
        self._listeners = [status] if status else []
        self._notes: list[str] = []
        self._changed = threading.Condition()
        self._ready = False
        self._error: str | None = None
        self._replies: dict[int, tuple] = {}
        self._asking = threading.Lock()        # one answer at a time
        self._sending = threading.Lock()
        self._next_id = 0
        self._unfinished: int | None = None    # a request whose answer we stopped waiting for
        self._closing = False
        ctx = multiprocessing.get_context("spawn")
        self._stop_flag = ctx.Event()
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=child_main, name="leanwatch-model",
                                 args=(child_conn, self._stop_flag, loader, model_id, options, _log_file()))
        self._proc.start()
        child_conn.close()                     # so that our end reports EOF when the child dies
        own_scope(self._proc.pid)
        _running.add(self)
        self._reader = threading.Thread(target=self._read, daemon=True, name="leanwatch-model-reader")
        self._reader.start()

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    @property
    def busy(self) -> bool:
        """Still working on a step whose answer nobody waits for any more (Stop came too late to cut it)."""
        return self._unfinished is not None

    def wait_idle(self, stop: threading.Event | None = None) -> None:
        """Wait until that step is done, so the next screenshot is taken after it, not before."""
        with self._asking:
            self._collect_unfinished(stop)

    @property
    def failed(self) -> bool:
        """True after a load error or a crash: the next task needs a new one."""
        return self._error is not None

    def listen(self, status) -> None:
        """Also tell `status` the loading notes (which precision and why), including those already sent."""
        with self._changed:
            if status in self._listeners:
                return
            self._listeners.append(status)
            notes = list(self._notes)
        for text in notes:
            status(text)

    def wait_ready(self, stop: threading.Event | None = None, status=None) -> None:
        """Wait until the model has loaded.

        Stop ends the wait at once with Stopped, but the model keeps loading (the download may be most of
        the work), so the next task starts sooner. Quitting LeanWatch ends it.
        """
        if status is not None:
            self.listen(status)
        with self._changed:
            while True:
                if self._error is not None:
                    raise ModelError(self._error)
                if self._ready:
                    return
                if stop is not None and stop.is_set():
                    raise Stopped()
                self._changed.wait(0.1)

    def answer(self, messages: list[dict], images: list, stop: threading.Event | None = None):
        self.wait_ready(stop)
        with self._asking:
            self._collect_unfinished(stop)
            if stop is not None and stop.is_set():
                raise Stopped()
            self._stop_flag.clear()
            with self._changed:
                self._next_id += 1
                rid = self._next_id
            try:
                with self._sending:
                    self._conn.send(("answer", rid, messages, images))
            except (OSError, ValueError):
                pass                           # the child is gone: the reader says why
            return self._wait_for(rid, stop)

    def close(self) -> None:
        """End the child process. (It also ends by itself when this process goes away.)"""
        self._closing = True
        _running.discard(self)
        try:
            with self._sending:
                self._conn.send(("quit",))
        except (OSError, ValueError):
            pass
        self._proc.join(2)
        if self._proc.is_alive():              # busy loading or answering: it can't read "quit" now
            self._proc.terminate()
            self._proc.join(3)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(3)
        self._reader.join(3)
        self._conn.close()

    # ------------------------------------------------------------------
    def _wait_for(self, rid: int, stop) -> object:
        deadline = None
        with self._changed:
            while rid not in self._replies:
                if self._error is not None:
                    raise ModelError(self._error)
                if stop is not None and stop.is_set():
                    if deadline is None:
                        self._stop_flag.set()  # the model's StoppingCriteria ends the answer at the next word
                        deadline = time.monotonic() + self.stop_grace
                    elif time.monotonic() >= deadline:
                        self._unfinished = rid  # still reading the screenshot, say: collect it next time
                        raise Stopped()
                self._changed.wait(0.1)
            kind, _, payload = self._replies.pop(rid)
        if kind == "failed":
            raise ModelError(payload)
        return payload

    def _collect_unfinished(self, stop) -> None:
        """Wait for the answer Stop cut short (the stop flag is still set, so it ends soon), then drop it."""
        rid = self._unfinished
        if rid is None:
            return
        with self._changed:
            while rid not in self._replies:
                if self._error is not None:
                    raise ModelError(self._error)
                if stop is not None and stop.is_set():
                    raise Stopped()
                self._changed.wait(0.1)
            self._replies.pop(rid)
        self._unfinished = None

    def _read(self) -> None:
        """Receives everything the child sends, and notices when it dies."""
        while True:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                break
            except Exception as err:           # something this process can't unpickle
                traceback.print_exc()
                with self._changed:
                    self._error = (f"LeanWatch could not read what the model sent ({type(err).__name__}: {err}).\n"
                                   f"{details_hint()}")
                    self._changed.notify_all()
                self._proc.kill()
                continue                       # the next recv reports EOF
            if msg[0] == "status":
                self._note(msg[1])
                continue
            with self._changed:
                if msg[0] == "ready":
                    self.info = msg[1]
                    self.coords = msg[1].get("coords", self.coords)
                    self._ready = True
                elif msg[0] == "error":
                    self._error = msg[1]
                else:                          # "answer" or "failed", with the request's id
                    self._replies[msg[1]] = msg
                self._changed.notify_all()
        self._proc.join(10)
        with self._changed:
            try:
                if self._error is None:
                    if self._closing:
                        self._error = "The model was shut down."
                    else:
                        self._error = crash_message(self._proc.exitcode, self.model_id, self.options.get("cpu_dtype", "auto"))
                        print(f"LeanWatch: {self._error}", file=sys.stderr, flush=True)
            except Exception:
                traceback.print_exc()
                self._error = f"The model's process ended unexpectedly (exit code {self._proc.exitcode})."
            finally:
                self._changed.notify_all()     # whatever happened, nobody may keep waiting for this process

    def _note(self, text: str) -> None:
        with self._changed:
            self._notes.append(text)
            listeners = list(self._listeners)
        for status in listeners:
            try:
                status(text)
            except Exception:
                pass                           # a broken listener must not stop the reader


def own_scope(pid: int | None) -> None:
    """Linux with systemd (Ubuntu): put the model's process in its own cgroup ("scope").

    Ubuntu's systemd-oomd ends a whole cgroup when memory runs out, and a child starts in its parent's:
    LeanWatch's (the app's, or the terminal's, shell included). In its own scope only the model is ended,
    and the bubble can say why. Best effort: without systemd or D-Bus nothing changes.
    """
    if not pid or not sys.platform.startswith("linux") or os.environ.get("LEANWATCH_MODEL_SCOPE", "1") == "0":
        return
    try:
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection

        conn = open_dbus_connection(bus="SESSION")
    except Exception:
        return
    try:
        manager = DBusAddress("/org/freedesktop/systemd1", bus_name="org.freedesktop.systemd1",
                              interface="org.freedesktop.systemd1.Manager")
        props = [("PIDs", ("au", [pid])), ("Description", ("s", "LeanWatch's model")),
                 ("CollectMode", ("s", "inactive-or-failed"))]
        conn.send_and_get_reply(new_method_call(manager, "StartTransientUnit", "ssa(sv)a(sa(sv))",
                                                (f"leanwatch-model-{pid}.scope", "fail", props, [])), timeout=3)
    except Exception:
        pass
    finally:
        conn.close()


def crash_message(exitcode: int | None, model_id: str, cpu_dtype: str = "auto") -> str:
    """Why the model's process ended without a word, in plain words."""
    again = "The next task starts the model again."
    if exitcode == -9 and os.name != "nt":
        # SIGKILL, which nothing in LeanWatch sends: almost always the kernel's out-of-memory killer.
        ram = ram_gb()
        has = f" (this computer has {ram:.0f} GB)" if ram else ""
        return (f"The system stopped the model, almost certainly because the computer ran out of memory{has}.\n"
                f"{memory_advice(model_id, cpu_dtype)}\n{again}")
    if exitcode is not None and exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        return (f"The model's process crashed ({name}). This usually means a broken install of torch or another "
                f"AI library. Check with: python -m leanwatch --check\n{details_hint()} {again}")
    return f"The model's process ended unexpectedly (exit code {exitcode}).\n{details_hint()} {again}"


# --------------------------------------------------------------------------- the child process
def child_main(conn, stop_flag, loader: str, model_id: str, options: dict, log_file: str | None = None) -> None:
    """Runs in the child: load the model, then answer until told to quit or the parent goes away."""
    if log_file:                               # started at login: write where LeanWatch itself writes
        try:
            sys.stdout = sys.stderr = open(log_file, "a", buffering=1, encoding="utf-8")
        except OSError:
            pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C in the terminal is for LeanWatch, which then ends us
    try:
        # If memory runs out, the kernel should end the model (the big one), not the bubble.
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except OSError:
        pass
    threading.Thread(target=_exit_with_parent, daemon=True, name="leanwatch-parent-watch").start()
    cpu_dtype = options.get("cpu_dtype", "auto")
    try:
        module, _, name = loader.partition(":")
        factory = getattr(importlib.import_module(module), name)
        model = factory(model_id, status=lambda text: conn.send(("status", text)), **options)
    except BaseException as err:               # SystemExit too: a broken package may raise anything
        traceback.print_exc()
        conn.send(("error", explain_load_error(err, model_id, cpu_dtype)))
        return
    conn.send(("ready", {"coords": getattr(model, "coords", "abs"), **getattr(model, "info", {})}))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] != "answer":
            return
        _, rid, messages, images = msg
        try:
            reply = ("answer", rid, model.answer(messages, images, stop=stop_flag))
        except Exception as err:
            traceback.print_exc()
            reply = ("failed", rid, explain_load_error(err, model_id, cpu_dtype, doing="get the model's answer"))
        conn.send(reply)


def _exit_with_parent() -> None:
    """Ends the child when LeanWatch goes away, even if LeanWatch was killed and could not say so."""
    parent = multiprocessing.parent_process()
    ppid = os.getppid()
    while True:
        if parent is not None:
            parent.join(1.0)
            if not parent.is_alive():
                os._exit(0)
        else:
            time.sleep(1.0)
        if os.name != "nt" and os.getppid() != ppid:
            os._exit(0)


def _log_file() -> str | None:
    """The file this process writes to instead of a terminal (~/.leanwatch/leanwatch.log when started at login)."""
    name = getattr(sys.stderr, "name", None)
    if isinstance(name, str) and not name.startswith("<") and os.path.isfile(name):
        return name
    return None


def _close_all() -> None:
    for model in list(_running):
        model.close()


# At exit multiprocessing waits for its child processes; this runs just before that wait (an atexit
# handler could run after it, and then the wait would never end).
multiprocessing.util.Finalize(None, _close_all, exitpriority=10)
