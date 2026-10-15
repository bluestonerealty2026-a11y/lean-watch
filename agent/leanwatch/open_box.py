"""Open LeanWatch's command box from anywhere:  python -m leanwatch --open

For a keyboard shortcut where apps can't watch the keyboard themselves (Wayland): the desktop's
own shortcut runs this file, which asks the running LeanWatch, over its local token-protected API,
to open the box. Standalone (standard library only) so it starts quickly from any folder.
"""

import json
import sys
import urllib.request
from pathlib import Path


def open_box(session_file: Path | None = None) -> int:
    path = session_file or Path.home() / ".leanwatch" / "session.json"
    try:
        info = json.loads(path.read_text())
        port, token = int(info["port"]), str(info["token"])
    except (OSError, ValueError, KeyError, TypeError):
        print("LeanWatch isn't running. Start it with: python -m leanwatch", file=sys.stderr)
        return 1
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/open", data=b"{}", method="POST",
        headers={"X-LeanWatch-Token": token, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never send this through a proxy
    try:
        opener.open(request, timeout=3).read()
        return 0
    except Exception as err:
        print(f"LeanWatch didn't answer ({err}). Is it running?", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(open_box())
