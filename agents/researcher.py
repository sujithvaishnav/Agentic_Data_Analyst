"""
Researcher agent — validate & cross-check findings.

FIX: No **state spread; truncated context; returns only new items for Annotated fields.
"""

from __future__ import annotations
from state import PipelineState
from llm_client import chat

SYSTEM = """
You are a senior data science researcher reviewing analytical work.

Your job:
1. Read the analysis results and execution stdout.
2. Check for statistical validity, completeness, data quality issues, logical consistency.
3. Produce a validated findings summary with corrections or caveats.
4. List significant gaps as "Gaps: ..."

Output format:
VALIDATED FINDINGS:
<bullet list>

CAVEATS:
<bullet list>

GAPS (if any):
<bullet list or "None">
"""


def researcher(state: PipelineState) -> dict:
    print("\n[Researcher] Validating findings...")

    goal             = state.get("goal", "")
    analysis_results = state.get("analysis_results", [])
    tool_results     = state.get("tool_results", [])
    plots            = state.get("plots", [])

    # Truncate to stay under TPM
    all_stdout   = "\n---\n".join(
        tr.get("stdout", "")[:400] for tr in tool_results[-3:] if tr.get("success")
    )
    all_findings = "\n\n".join(r[:300] for r in analysis_results[-3:])

    user_prompt = f"""Goal: {goal}

Analyst findings (truncated):
{all_findings or "None yet."}

Execution stdout (truncated):
{all_stdout or "No output yet."}

Figures produced: {[p.split("\\")[-1].split("/")[-1] for p in plots]}

Validate findings and identify gaps.
"""

    validation = chat(system=SYSTEM, user=user_prompt, temperature=0.2)
    print(f"[Researcher] Validation complete ({len(validation)} chars)")

    # Return ONLY changed fields — no **state spread
    # analysis_results is Annotated → return [single new item] only
    return {
        "analysis_results": [validation],
        "messages":         [{"role": "researcher", "content": validation[:200]}],
    }
