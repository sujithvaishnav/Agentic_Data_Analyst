"""
Goal Interpreter node — first stop after the user query.

FIX: Never use **state in return dict.
     Return ONLY the fields this node changes.
"""

from __future__ import annotations
from state import PipelineState
from llm_client import chat

SYSTEM = """
You are a goal interpretation specialist. Your job is to:
1. Read the user's raw query carefully.
2. Identify the core analytical or reporting goal.
3. Restate it as a single, precise, actionable goal sentence.

Rules:
- Focus on DATA ANALYSIS and REPORTING objectives.
- Ignore vague filler words; keep only intent.
- Output ONLY the goal sentence — no preamble, no explanation.

Example input:  "can you look at my sales data and maybe find some trends?"
Example output: "Identify sales trends and anomalies in the provided dataset and produce a summary report."
"""


def goal_interpreter(state: PipelineState) -> dict:
    print("\n[GoalInterpreter] Interpreting user query...")

    goal = chat(
        system=SYSTEM,
        user=f"User query: {state['user_query']}",
    )

    print(f"[GoalInterpreter] Goal → {goal}")

    # Return ONLY changed fields — no **state spread
    return {
        "goal":            goal,
        "iteration":       0,
        "plan_step_index": 0,
        "error":           "",
        "messages":        [{"role": "goal_interpreter", "content": goal}],
    }
