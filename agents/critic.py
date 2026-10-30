"""
Evaluation / Critic node — checks whether results satisfy the goal.

FIX: No **state spread in return dict.
"""

from __future__ import annotations
from state import PipelineState
from llm_client import chat_json

SYSTEM = """
You are a quality-control critic for a data-analysis pipeline.

Evaluate whether the current outputs satisfy the stated goal.

Scoring criteria:
  1. Is there a substantive analysis result (not just "code was run")?
  2. Are key statistics or trends identified?
  3. Is there a final report draft?
  4. Are findings logically consistent with the goal?

Output strict JSON:
{
  "evaluation": "goal_met" | "needs_work",
  "score": 0-10,
  "feedback": "<specific instructions if needs_work, else empty string>"
}

Be strict: "goal_met" requires score >= 7 AND a non-empty final_report.
"""


def critic(state: PipelineState) -> dict:
    print("\n[Critic] Evaluating pipeline outputs...")

    goal             = state.get("goal", "")
    analysis_results = state.get("analysis_results", [])
    final_report     = state.get("final_report", "")
    plots            = state.get("plots", [])
    iteration        = state.get("iteration", 0)

    user_prompt = f"""Goal: {goal}

Analysis results count: {len(analysis_results)}
Analysis results preview: {" | ".join(r[:150] for r in analysis_results[-3:])}
Final report exists: {bool(final_report)}
Final report length: {len(final_report)} chars
Unique plots produced: {len(list(dict.fromkeys(plots)))}
Current iteration: {iteration}
"""

    result = chat_json(system=SYSTEM, user=user_prompt)

    evaluation = result.get("evaluation", "needs_work")
    score      = result.get("score", 0)
    feedback   = result.get("feedback", "")

    print(f"[Critic] evaluation={evaluation}  score={score}/10")
    if feedback:
        print(f"[Critic] feedback: {feedback[:120]}")

    # Return ONLY changed fields — no **state spread
    return {
        "evaluation":     evaluation,
        "critic_feedback": feedback,
        "messages":       [{"role": "critic",
                            "content": f"evaluation={evaluation} score={score}/10 | {feedback[:100]}"}],
    }
