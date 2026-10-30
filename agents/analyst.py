"""
Analyst agent — data work.

Fixes applied:
  - Prompt instructs LLM to use DATASET_PATH variable (never hardcode paths)
  - Pandas 3.x frequency aliases listed explicitly
  - Truncated context to avoid 413 token errors
  - No **state spread in return dict
"""

from __future__ import annotations
from state import PipelineState
from llm_client import chat
from config.settings import OUTPUTS_DIR

CODE_GEN_SYSTEM = f"""
You are a senior data analyst writing Python code for a data-analysis pipeline.

Runtime: Python 3.12, Pandas 3.x, NumPy 2.x, Matplotlib 3.x

Available in the sandbox (pre-loaded — do NOT import or redefine):
  pd            → pandas
  np            → numpy
  plt           → matplotlib.pyplot
  Path          → pathlib.Path
  save_fig(name)→ saves current figure to {OUTPUTS_DIR}/<name>.png
  DATASET_PATH  → string path to the user's CSV file (empty string if no file)

CRITICAL — loading a dataset:
  If DATASET_PATH is not empty, ALWAYS load data like this:
    df = pd.read_csv(DATASET_PATH, encoding='latin-1')
  Never hardcode any file path. Always use the DATASET_PATH variable.
  If DATASET_PATH is empty, generate realistic synthetic data instead.

CRITICAL — Pandas 3.x frequency aliases (old ones raise ValueError):
  Use 'ME'  not 'M'       (month end)
  Use 'QE'  not 'Q'       (quarter end)
  Use 'YE'  not 'A'/'Y'   (year end)
  Use 'h'   not 'H'       (hours)
  Use 'min' not 'T'       (minutes)
  Use 's'   not 'S'       (seconds)

Rules:
- Output ONLY raw Python — no markdown fences, no explanations
- Always call save_fig() for every chart (never plt.show())
- Print key statistics to stdout — they will be captured
- Handle NaN/missing data with dropna() or fillna()
- Do NOT import any libraries (all pre-loaded)
"""


def analyst(state: PipelineState) -> dict:
    print("\n[Analyst] Starting data analysis...")

    goal            = state.get("goal", "")
    plan            = state.get("plan", [])
    tool_results    = state.get("tool_results", [])
    dataset_summary = state.get("dataset_summary", "")
    dataset_path    = state.get("dataset_path", "")
    existing_plots  = state.get("plots", [])

    # Pick current task by counting prior analyst invocations
    analyst_steps  = [s for s in plan if s["agent"] == "analyst"]
    executed_count = sum(1 for m in state.get("messages", []) if m["role"] == "analyst")
    current_task   = (
        analyst_steps[executed_count]["task"]
        if executed_count < len(analyst_steps)
        else "Perform exploratory data analysis"
    )
    print(f"[Analyst] Task: {current_task}")

    # Truncated context — avoids token bloat
    prev_stdout = "\n".join(
        tr.get("stdout", "")[:300] for tr in tool_results[-2:] if tr.get("success")
    )
    plot_names = [p.split("\\")[-1].split("/")[-1] for p in existing_plots]

    dataset_note = (
        "A real dataset IS available — use pd.read_csv(DATASET_PATH, encoding='latin-1')"
        if dataset_path
        else "No dataset file — generate realistic synthetic data."
    )

    user_prompt = f"""Goal: {goal}
Current task: {current_task}
Dataset status: {dataset_note}
Dataset context: {dataset_summary}
Previous stdout (truncated): {prev_stdout or "None"}
Existing plots: {plot_names}

Write the Python code. Use DATASET_PATH variable — never hardcode any path.
"""

    code = chat(system=CODE_GEN_SYSTEM, user=user_prompt, temperature=0.1)

    # Strip accidental markdown fences
    if "```" in code:
        lines = code.strip().split("\n")
        code  = "\n".join(l for l in lines if not l.strip().startswith("```"))

    print(f"[Analyst] Generated {len(code)} chars of code")

    # Return ONLY changed fields — no **state spread
    return {
        "code_to_run":     code,
        "dataset_summary": dataset_summary,
        "messages":        [{"role": "analyst",
                             "content": f"Task: {current_task} | {len(code)} chars queued"}],
    }
