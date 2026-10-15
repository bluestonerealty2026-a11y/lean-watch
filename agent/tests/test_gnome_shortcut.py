"""gnome_shortcut.py (python -m leanwatch --shortcut on|off) against a fake gsettings, never the real one."""

import ast
import glob
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from leanwatch import gnome_shortcut as gs

CUSTOM0 = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/"
CUSTOM1 = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom1/"
REL = f"{gs.SCHEMA}.custom-keybinding:{gs.ENTRY}"


def gvariant_print_strings(values):
    """How `gsettings get` prints an `as` value (GLib's g_variant_print)."""
    if not values:
        return "@as []"
    out = []
    for v in values:
        if "'" in v and '"' not in v:
            out.append('"' + v.replace("\\", "\\\\") + '"')
        else:
            out.append("'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'")
    return "[" + ", ".join(out) + "]"


class FakeGSettings:
    """The few gsettings commands gnome_shortcut uses. Values are parsed like gsettings parses them
    (GVariant text; its string escapes match Python's for \\\\ and \\'), so bad quoting fails here too."""

    def __init__(self, bindings=None, available=True, missing=False):
        self.bindings = list(bindings or [])
        self.keys = {}                    # (relocatable path, key) -> parsed value
        self.available = available
        self.missing = missing            # no gsettings program at all
        self.log = []
        self.fail_reset = False

    def __call__(self, *args):
        self.log.append(args)
        if self.missing:
            raise FileNotFoundError(2, "No such file or directory: 'gsettings'")
        if not self.available:
            raise subprocess.CalledProcessError(1, ["gsettings", *args], stderr="No such schema")
        cmd = args[0]
        if cmd == "list-keys":
            assert args[1] == gs.SCHEMA
            return "custom-keybindings\nscreensaver\nvolume-up"
        if cmd == "get" and args[1:] == (gs.SCHEMA, "custom-keybindings"):
            return gvariant_print_strings(self.bindings)
        if cmd == "set" and args[1:3] == (gs.SCHEMA, "custom-keybindings"):
            value = ast.literal_eval(args[3])
            assert isinstance(value, list) and all(isinstance(v, str) for v in value)
            self.bindings = value
            return ""
        if cmd == "set":
            value = ast.literal_eval(args[3])
            assert isinstance(value, str), f"not a GVariant string literal: {args[3]!r}"
            self.keys[(args[1], args[2])] = value
            return ""
        if cmd == "reset":
            if self.fail_reset:
                raise subprocess.CalledProcessError(1, ["gsettings", *args])
            self.keys.pop((args[1], args[2]), None)
            return ""
        raise AssertionError(f"unexpected gsettings call {args}")


@pytest.fixture
def store(monkeypatch):
    fake = FakeGSettings()
    monkeypatch.setattr(gs, "_gsettings", fake)
    return fake


# --------------------------------------------------------------------------- accelerators and quoting
@pytest.mark.parametrize("combo,binding", [
    ("ctrl+alt+space", "<Control><Alt>space"),
    ("Ctrl+Shift+L", "<Control><Shift>l"),
    ("control + alt + f5", "<Control><Alt>f5"),
    ("super+x", "<Super>x"), ("win+e", "<Super>e"), ("cmd+k", "<Super>k"), ("meta+period", "<Super>period"),
])
def test_gnome_binding(combo, binding):
    assert gs.gnome_binding(combo) == binding


def test_gnome_binding_empty():
    with pytest.raises(ValueError):
        gs.gnome_binding(" + ")


@pytest.mark.parametrize("combo,binding", [
    ("ctrl+option+space", "<Control><Alt>space"), ("ctl+alt+k", "<Control><Alt>k"),
    ("command+shift+k", "<Super><Shift>k"), ("opt+ctrl+x", "<Alt><Control>x"),
])
def test_gnome_binding_accepts_the_names_the_hotkey_option_accepts(combo, binding):
    from leanwatch.native import parse_hotkey

    parse_hotkey(combo)                       # LeanWatch's own shortcut takes it
    assert gs.gnome_binding(combo) == binding


@pytest.mark.parametrize("combo", ["space", "a", "ctrl+alt", "hyper+x"])
def test_gnome_binding_rejects_what_the_hotkey_option_rejects(combo):
    from leanwatch.native import parse_hotkey

    with pytest.raises(ValueError):
        parse_hotkey(combo)
    with pytest.raises(ValueError):
        gs.gnome_binding(combo)


@pytest.mark.parametrize("value", ["LeanWatch", "it's", "back\\slash", "\\'", "both ' and \\ and \"", "Écran"])
def test_text_is_a_gvariant_string_literal(value):
    literal = gs._text(value)
    assert literal.startswith("'") and literal.endswith("'")
    assert ast.literal_eval(literal) == value


def _python_with_glib():
    candidates = []
    for so in sorted(glob.glob("/usr/lib/python3/dist-packages/gi/_gi.cpython-3*.so"), reverse=True):
        m = re.search(r"cpython-3(\d+)", so)
        if m:
            candidates.append(f"/usr/bin/python3.{m.group(1)}")
    for exe in candidates + ["/usr/bin/python3"]:
        if Path(exe).exists():
            return exe
    return None


def test_text_with_glibs_own_parser():
    """The same literals through GLib's GVariant parser (what gsettings really uses), when it is installed."""
    exe = _python_with_glib()
    if exe is None:
        pytest.skip("no system Python with GLib's bindings")
    values = ["it's", "back\\slash", "\\'", "both ' and \\ and \"", "Écran", '"/usr/bin/python3" "/x/open_box.py"']
    script = ("import json, sys\n"
              "from gi.repository import GLib\n"
              "items = json.load(sys.stdin)\n"
              "out = [GLib.Variant.parse(GLib.VariantType('s'), t, None, None).get_string() for t in items['s']]\n"
              "arr = GLib.Variant.parse(GLib.VariantType('as'), items['as'], None, None).unpack()\n"
              "json.dump({'s': out, 'as': arr}, sys.stdout)\n")
    payload = {"s": [gs._text(v) for v in values], "as": "[" + ", ".join(gs._text(v) for v in values) + "]"}
    r = subprocess.run([exe, "-s", "-c", script], input=json.dumps(payload), capture_output=True, text=True,
                       timeout=30, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    if r.returncode != 0 and ("No module named" in r.stderr or "cannot import" in r.stderr):
        pytest.skip(f"GLib's bindings don't load in {exe}")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"s": values, "as": values}


def test_command_runs_open_box_with_this_python():
    assert gs.OPEN_SCRIPT.name == "open_box.py" and gs.OPEN_SCRIPT.exists()
    assert gs.command("/usr/bin/python3") == f'"/usr/bin/python3" "{gs.OPEN_SCRIPT}"'
    assert gs.command() == f'"{sys.executable}" "{gs.OPEN_SCRIPT}"'


# --------------------------------------------------------------------------- install / remove
def test_install_into_an_empty_list(store):
    msg = gs.install("ctrl+alt+space")
    assert "Ctrl" in msg or "ctrl" in msg
    assert store.bindings == [gs.ENTRY]
    assert store.keys == {(REL, "name"): "LeanWatch", (REL, "command"): gs.command(),
                          (REL, "binding"): "<Control><Alt>space"}
    assert gs.installed() is True


def test_install_keeps_the_other_custom_shortcuts(store):
    store.bindings = [CUSTOM0, CUSTOM1]
    gs.install("ctrl+alt+space")
    assert store.bindings == [CUSTOM0, CUSTOM1, gs.ENTRY]


def test_installing_twice_adds_one_entry_and_updates_the_keys(store):
    store.bindings = [CUSTOM0]
    gs.install("ctrl+alt+space")
    gs.install("super+l")
    assert store.bindings == [CUSTOM0, gs.ENTRY]
    assert store.keys[(REL, "binding")] == "<Super>l"


def test_install_quotes_a_path_with_quotes_and_backslashes(store, monkeypatch):
    odd = Path("/home/o'neil/my \\ tools/leanwatch/open_box.py")
    monkeypatch.setattr(gs, "OPEN_SCRIPT", odd)
    gs.install("ctrl+alt+space")
    assert store.keys[(REL, "command")] == gs.command() and str(odd) in gs.command()


def test_install_reads_entries_printed_with_double_quotes(store):
    odd = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/it's/"
    store.bindings = [odd]
    assert gvariant_print_strings(store.bindings).startswith('["')          # how GLib prints it
    gs.install("ctrl+alt+space")
    assert store.bindings == [odd, gs.ENTRY]


def test_install_where_gnome_isnt(store):
    store.available = False
    with pytest.raises(RuntimeError, match="aren't available"):
        gs.install("ctrl+alt+space")
    assert all(call[0] == "list-keys" for call in store.log)              # nothing was changed


def test_install_without_gsettings_at_all(store):
    store.missing = True
    assert gs.available() is False and gs.installed() is False
    with pytest.raises(RuntimeError):
        gs.install("ctrl+alt+space")


def test_installed(store):
    assert gs.installed() is False
    store.bindings = [CUSTOM0]
    assert gs.installed() is False
    store.bindings = [CUSTOM0, gs.ENTRY]
    assert gs.installed() is True
    store.available = False
    assert gs.installed() is False


def test_unreadable_list_counts_as_empty(store, monkeypatch):
    monkeypatch.setattr(gs, "_gsettings", lambda *a: "something gsettings never prints")
    assert gs._entries() == [] and gs.installed() is False
    monkeypatch.setattr(gs, "_gsettings", lambda *a: "'just a string'")
    assert gs._entries() == []


def test_remove_keeps_the_others_and_resets_ours(store):
    store.bindings = [CUSTOM0]
    gs.install("ctrl+alt+space")
    store.bindings = [CUSTOM0, gs.ENTRY, CUSTOM1]
    assert gs.remove() == "Removed LeanWatch's shortcut from GNOME's keyboard settings."
    assert store.bindings == [CUSTOM0, CUSTOM1]
    assert store.keys == {}
    assert gs.installed() is False


def test_remove_the_only_one(store):
    gs.install("ctrl+alt+space")
    gs.remove()
    assert store.bindings == []
    assert ("set", gs.SCHEMA, "custom-keybindings", "[]") in store.log


def test_remove_when_not_installed(store):
    store.bindings = [CUSTOM0]
    assert "There was no LeanWatch shortcut" in gs.remove()
    assert store.bindings == [CUSTOM0]
    assert not any(call[0] in ("set", "reset") for call in store.log)


def test_remove_where_gnome_isnt(store):
    store.available = False
    assert "nothing to remove" in gs.remove()


def test_remove_survives_a_failing_reset(store):
    gs.install("ctrl+alt+space")
    store.fail_reset = True
    assert gs.remove().startswith("Removed")
    assert store.bindings == []


def test_gsettings_runs_the_program_with_a_timeout(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return subprocess.CompletedProcess(cmd, 0, stdout="@as []\n", stderr="")

    monkeypatch.setattr(gs, "subprocess", SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError))
    assert gs._gsettings("get", gs.SCHEMA, "custom-keybindings") == "@as []"
    assert seen["cmd"] == ["gsettings", "get", gs.SCHEMA, "custom-keybindings"]
    assert seen["kw"]["check"] is True and seen["kw"]["timeout"] > 0 and seen["kw"]["text"] is True
