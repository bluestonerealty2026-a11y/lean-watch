"""Lets the training scripts import the agent's shared modules (action format, prompts)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
agent_path = REPO / "agent"
if str(agent_path) not in sys.path:
    sys.path.insert(0, str(agent_path))
