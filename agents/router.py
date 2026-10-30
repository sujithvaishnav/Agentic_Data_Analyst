"""
Hybrid Router node — LLM-controlled workflow routing.

Strategy:
  1. Rule-based: derive next agent deterministically from plan_step_index
  2. LLM fallback: for ambiguous states (critic loop, plan exhausted)

Fixes applied:
  - Uses plan_step_index (int) — handles multiple analyst steps correctly
  - Rule 2b: retries analyst when last execution failed
  - Compact LLM context — avoids 413 token errors
  - No **state spread in return dict
"""

from __future__ import annotations
from state import PipelineState
from llm_client import chat_json
from config.settings import VALID_NODES, MAX_ITERATIONS

SYSTEM = f"""
You are a routing controller for a multi-agent data-analysis pipeline.

Available agents: {VALID_NODES}
Special values: "writer" (force report), "END" (pipeline complete)

Given the compact pipeline state, decide which agent runs NEXT.

Output strict JSON only:
{{
  "next_agent": "<agent_name_or_END>",
  "reason": "<one sentence>"
}}

Rules:
- Output "END" only when final_report exists AND evaluation is "goal_met".
- If approaching max_iterations, route to "writer".
- Always pick from the valid list above.
"""


def _rule_based_route(state: PipelineState) -> tuple[str, str, int] | None:
    """
    Returns (agent, reason, new_plan_step_index) or None to defer to LLM.
    Uses plan_step_index so multiple steps with the same agent name all execute.
    """
    iteration    = state.get("iteration", 0)
    plan         = state.get("plan", [])
    step_idx     = state.get("plan_step_index", 0)
    evaluation   = state.get("evaluation", "")
    final_report = state.get("final_report", "")

    # Rule 1 — done
    if final_report and evaluation == "goal_met":
        return "END", "Final report produced and goal is met.", step_idx

    # Rule 2 — approaching ceiling
    if iteration >= MAX_ITERATIONS - 1:
        return "writer", f"Approaching iteration limit ({MAX_ITERATIONS}); forcing report.", step_idx

    # Rule 2b — last execution failed → retry analyst
    tool_results = state.get("tool_results", [])
    last_failed  = tool_results and not tool_results[-1].get("success", True)
    if last_failed and step_idx > 0 and step_idx <= len(plan) and plan[step_idx - 1]["agent"] == "analyst":
        return "analyst", "Last code execution failed; retrying analyst.", max(0, step_idx - 1)

    # Rule 3 — follow plan by index
    if step_idx < len(plan):
        step  = plan[step_idx]
        agent = step["agent"]
        return agent, f"Plan step {step['step']}: {step['task']}", step_idx + 1

    # Rule 4 — critic needs_work after plan exhausted → LLM decides
    if evaluation == "needs_work":
        return None

    # Rule 5 — plan done, no report → writer
    if plan and not final_report:
        return "writer", "All plan steps executed; generating report.", step_idx

    return None


def router(state: PipelineState) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"\n[Router] Iteration {iteration}/{MAX_ITERATIONS}")

    rule_result = _rule_based_route(state)

    if rule_result is not None:
        next_agent, reason, new_step_idx = rule_result
        print(f"[Router] Rule-based → {next_agent}  ({reason})")
    else:
        print("[Router] Rule-based inconclusive → asking LLM router...")

        # Compact context — avoids token bloat
        context = {
            "goal":             state.get("goal", "")[:200],
            "plan":             state.get("plan", []),
            "plan_step_index":  state.get("plan_step_index", 0),
            "iteration":        iteration,
            "max_iterations":   MAX_ITERATIONS,
            "evaluation":       state.get("evaluation", ""),
            "critic_feedback":  state.get("critic_feedback", "")[:200],
            "has_final_report": bool(state.get("final_report", "")),
            "plots_count":      len(state.get("plots", [])),
            "analysis_count":   len(state.get("analysis_results", [])),
        }

        result     = chat_json(system=SYSTEM, user=f"State:\n{context}")
        next_agent = result.get("next_agent", "writer")
        reason     = result.get("reason", "LLM routing decision")
        new_step_idx = state.get("plan_step_index", 0)

        if next_agent not in VALID_NODES + ["END"]:
            print(f"[Router] Invalid agent '{next_agent}' → defaulting to writer")
            next_agent = "writer"
            reason     = "LLM suggested invalid node; defaulted to writer."

        print(f"[Router] LLM → {next_agent}  ({reason})")

    # Return ONLY changed fields — no **state spread
    return {
        "next_agent":      next_agent,
        "routing_reason":  reason,
        "iteration":       iteration,
        "plan_step_index": new_step_idx,
        "messages":        [{"role": "router", "content": f"→ {next_agent}: {reason}"}],
    }
