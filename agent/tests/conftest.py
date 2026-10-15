import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Unit tests must never ask the real desktop for screen sharing: when they run inside a Wayland
# session, keep LeanWatch on its X11/offscreen paths (tests that cover Wayland use fakes).
os.environ["LEANWATCH_WAYLAND"] = "0"
