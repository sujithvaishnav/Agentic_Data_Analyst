"""
Planner node — breaks the goal into an ordered step plan.

FIX: Never use **state in return dict.
"""

from __future__ import annotations
import json
from state import PipelineState
from llm_client import chat_json
from config.settings import VALID_NODES

SYSTEM = f"""
You are a strategic planner for a data-analysis multi-agent system.

Available agents:
  - analyst    : loads data, runs Python/Pandas code, generates statistics and plots
  - researcher : validates findings, cross-checks logic, evaluates methodology
  - writer     : composes final markdown reports and explanations

Your task:
Given a goal, produce a JSON plan — an ordered list of steps.
Each step must specify which agent should handle it.

Output format (strict JSON, no extra text):
{{
  "plan": [
    {{"step": 1, "task": "<concise task description>", "agent": "<agent_name>"}},
    {{"step": 2, "task": "<concise task description>", "agent": "<agent_name>"}}
  ]
}}

Rules:
- Use ONLY these agent names: {VALID_NODES}
- Keep each task description under 20 words
- A typical analysis plan has 3-6 steps
- Always end with a writer step to produce the final report
- analyst before researcher; researcher before writer (usually)
"""


def planner(state: PipelineState) -> dict:
    print("\n[Planner] Building execution plan...")

    result = chat_json(
        system=SYSTEM,
        user=f"Goal: {state['goal']}",
    )

    plan: list[dict] = result.get("plan", [])

    print(f"[Planner] Plan ({len(plan)} steps):")
    for step in plan:
        print(f"  Step {step['step']}: [{step['agent']}] {step['task']}")

    # Return ONLY changed fields — no **state spread
    return {
        "plan":            plan,
        "plan_step_index": 0,
        "messages":        [{"role": "planner", "content": json.dumps(plan)}],
    }
