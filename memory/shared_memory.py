"""
Shared Memory helpers — checkpoint save/load and state pretty-printer.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from config.settings import OUTPUTS_DIR

CHECKPOINT_DIR = Path(OUTPUTS_DIR) / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, label: str = "") -> str:
    """Persist state to JSON. Returns the file path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"checkpoint_{ts}_{label}.json" if label else f"checkpoint_{ts}.json"
    path = CHECKPOINT_DIR / name

    safe = {}
    for k, v in state.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)

    with open(path, "w") as f:
        json.dump(safe, f, indent=2)
    return str(path)


def load_checkpoint(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pretty_state(state: dict) -> str:
    lines = [
        f"{'='*60}",
        f"  PIPELINE STATE  (iteration {state.get('iteration', 0)})",
        f"{'='*60}",
        f"  goal            : {state.get('goal', '')[:80]}",
        f"  next_agent      : {state.get('next_agent', '-')}",
        f"  evaluation      : {state.get('evaluation', '-')}",
        f"  plan steps      : {len(state.get('plan', []))}",
        f"  messages        : {len(state.get('messages', []))}",
        f"  tool_results    : {len(state.get('tool_results', []))}",
        f"  unique plots    : {len(list(dict.fromkeys(state.get('plots', []))))}",
        f"  report chars    : {len(state.get('final_report', ''))}",
        f"{'='*60}",
    ]
    return "\n".join(lines)
