#!/bin/bash
# A headless GNOME Shell (Wayland) for LeanWatch's tests, run by run_gnome.py inside
# "unshare -m" with /run/systemd hidden (containers have no logind, and GNOME Shell then
# uses its no-logind code path) and inside dbus-run-session (its own session bus).
# Writes the bus address, WAYLAND_DISPLAY and DISPLAY to $OUT/session.env when ready.
# WirePlumber is started by run_gnome.py outside the namespace: its logind plugin needs /run/systemd.
set -u
export XDG_CURRENT_DESKTOP=GNOME XDG_SESSION_TYPE=wayland XDG_SESSION_DESKTOP=gnome
pipewire > "$OUT/pipewire.log" 2>&1 &
for i in $(seq 50); do [ -S "$XDG_RUNTIME_DIR/pipewire-0" ] && break; sleep 0.1; done
gnome-shell --headless --wayland --unsafe-mode --virtual-monitor 1280x800 > "$OUT/shell.log" 2>&1 &
SHELL_PID=$!
up=""
for i in $(seq 120); do
  if gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell \
       --method org.gnome.Shell.Eval "1" 2>/dev/null | grep -q true; then up=1; break; fi
  kill -0 $SHELL_PID 2>/dev/null || break
  sleep 0.5
done
if [ -z "$up" ]; then echo "failed" > "$OUT/session.env"; exit 1; fi
getenv() {
  gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell --method org.gnome.Shell.Eval \
    "imports.gi.GLib.getenv('$1')" | sed -E "s/^\(true, '\"(.*)\"'\)$/\1/"
}
WD=$(getenv WAYLAND_DISPLAY); XD=$(getenv DISPLAY); XA=$(getenv XAUTHORITY)
# The portal's dialogs are GTK apps started by the bus: they need to know where the screen is.
dbus-update-activation-environment WAYLAND_DISPLAY="$WD" DISPLAY="$XD" XAUTHORITY="$XA" XDG_CURRENT_DESKTOP=GNOME \
  XDG_SESSION_TYPE=wayland XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" HOME="$HOME"
printf 'DBUS_SESSION_BUS_ADDRESS=%s\nWAYLAND_DISPLAY=%s\nDISPLAY=%s\nXAUTHORITY=%s\n' "$DBUS_SESSION_BUS_ADDRESS" "$WD" "$XD" "$XA" > "$OUT/session.env"
wait $SHELL_PID
