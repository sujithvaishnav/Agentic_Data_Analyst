"""
evaluation/history.py — Persists and loads evaluation history across runs.

Stores each EvalReport as one JSON line in outputs/eval_history.jsonl.
Run `python main.py --history` to see a comparison table of all past runs.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict
from config.settings import OUTPUTS_DIR

HISTORY_FILE = Path(OUTPUTS_DIR) / "eval_history.jsonl"


def save_run(report) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id":        report.run_id,
        "query":         report.query,
        "overall_score": report.overall_score,
        "overall_grade": report.overall_grade,
        "goal_met":      report.raw_metrics.goal_met,
        "dimensions":    {d.name: d.score for d in report.dimensions},
        "metrics":       asdict(report.raw_metrics),
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists(): return []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_history_summary() -> None:
    history = load_history()
    if not history:
        print("No evaluation history yet. Run the pipeline first.")
        return

    print("\n" + "=" * 72)
    print("  EVALUATION HISTORY")
    print("=" * 72)
    print(f"  {'Run ID':<18} {'Score':>6} {'Grade':>6} {'Goal':>5} "
          f"{'Iters':>6} {'ExecRate':>9} {'Rule%':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6}")

    for r in history:
        m     = r.get("metrics", {})
        goal  = "✓" if r.get("goal_met") else "✗"
        iters = f"{m.get('total_iterations',0)}/{m.get('max_iterations_limit',8)}"
        exec_rate = f"{m.get('execution_success_rate', 0):.0f}%"
        events = m.get("routing_events", [])
        total_r = len(events)
        rule_r  = sum(1 for e in events if e.get("method") == "rule_based")
        rb_pct  = f"{rule_r/total_r*100:.0f}%" if total_r else "N/A"
        print(f"  {r['run_id']:<18} {r['overall_score']:>6.1f} "
              f"{r['overall_grade']:>6} {goal:>5} {iters:>6} "
              f"{exec_rate:>9} {rb_pct:>6}")

    print("=" * 72)
    print(f"  Total runs: {len(history)}")
    if len(history) > 1:
        scores = [r["overall_score"] for r in history]
        trend  = "↑ improving" if scores[-1] > scores[0] else "↓ declining"
        print(f"  Score trend: {scores[0]:.1f} → {scores[-1]:.1f}  ({trend})")
    print()
