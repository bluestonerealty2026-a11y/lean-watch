import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # training/
sys.path.insert(0, str(HERE))          # tiny_model
import leanwatch_paths  # noqa: E402,F401  (agent/)
