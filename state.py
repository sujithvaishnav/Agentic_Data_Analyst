"""
Shared pipeline state — passed through every node in the LangGraph graph.
All agents read from and write to this TypedDict.

CRITICAL LangGraph rule for Annotated[list, operator.add] fields:
  Return ONLY the NEW items to append — never the full accumulated list.
  LangGraph merges them automatically. Returning the full list doubles
  every entry on each iteration.
"""

from __future__ import annotations
from typing import TypedDict, Annotated
import operator


class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    user_query:   str
    dataset_path: str                           # actual file path → injected as DATASET_PATH in sandbox

    # ── Planner outputs ────────────────────────────────────────────────────────
    goal:         str
    plan:         list[dict]                    # [{"step":1,"task":"...","agent":"..."}]

    # ── Router ─────────────────────────────────────────────────────────────────
    next_agent:      str
    routing_reason:  str

    # ── Agent messages (Annotated → LangGraph appends delta only) ─────────────
    messages:        Annotated[list[dict], operator.add]

    # ── Tool execution ─────────────────────────────────────────────────────────
    code_to_run:     str
    tool_results:    Annotated[list[dict], operator.add]

    # ── Shared memory / artefacts ─────────────────────────────────────────────
    dataset_summary:  str
    plots:            Annotated[list[str],  operator.add]
    analysis_results: Annotated[list[str],  operator.add]

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluation:      str                        # "goal_met" | "needs_work"
    critic_feedback: str

    # ── Final output ──────────────────────────────────────────────────────────
    final_report:    str

    # ── Control ───────────────────────────────────────────────────────────────
    iteration:        int
    plan_step_index:  int                       # index of next unexecuted plan step
    error:            str
