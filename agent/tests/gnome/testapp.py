"""A native Wayland window (GTK 4) that logs what it receives: the app LeanWatch works in during the
GNOME tests. Run with the system's Python (it has GTK's bindings):  python3 testapp.py <log.jsonl>"""

import json
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

LOG = open(sys.argv[1], "a", buffering=1)


def log(**kw):
    kw["t"] = round(time.monotonic(), 3)
    LOG.write(json.dumps(kw) + "\n")


def activate(app):
    win = Gtk.ApplicationWindow(application=app, title="LeanWatch Wayland test")
    win.set_default_size(760, 460)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    entry = Gtk.Entry()
    entry.connect("changed", lambda e: log(ev="text", text=e.get_text()))
    entry.connect("activate", lambda e: log(ev="enter", text=e.get_text()))
    button = Gtk.Button(label="Press me")
    button.connect("clicked", lambda b: log(ev="button"))
    area = Gtk.DrawingArea()
    area.set_vexpand(True)
    click = Gtk.GestureClick(button=0)
    click.connect("pressed", lambda g, n, x, y: log(ev="press", button=g.get_current_button(), x=round(x), y=round(y)))
    area.add_controller(click)
    scroll = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.BOTH_AXES | Gtk.EventControllerScrollFlags.DISCRETE)
    scroll.connect("scroll", lambda c, dx, dy: (log(ev="scroll", dx=dx, dy=dy), True)[1])
    area.add_controller(scroll)
    keys = Gtk.EventControllerKey()
    keys.connect("key-pressed", lambda c, kv, code, state: (log(ev="key", name=Gdk.keyval_name(kv)), False)[1])
    win.add_controller(keys)
    for w in (entry, button, area):
        box.append(w)
    win.set_child(box)
    win.present()
    log(ev="ready")


app = Gtk.Application(application_id="io.leanwatch.WaylandTest")
app.connect("activate", activate)
app.run([])
