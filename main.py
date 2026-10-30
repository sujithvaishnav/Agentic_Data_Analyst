"""
Agentic Analysis Pipeline — main entry point.

Graph flow:
  GoalInterpreter → Planner → Router → [Analyst → ToolExecutor | Researcher | Writer → Critic]
                                ↑_______________|critic needs_work loop|_____________________|

Evaluation framework wired in via instrumented node wrappers.
Every run produces: final_report.md, evaluation_report.md, eval_history.jsonl
"""

from __future__ import annotations
import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END

from state import PipelineState
from agents.goal_interpreter import goal_interpreter
from agents.planner          import planner
from agents.router           import router
from agents.analyst          import analyst
from agents.researcher       import researcher
from agents.writer           import writer
from agents.critic           import critic
from tools.executor          import tool_executor
from memory.shared_memory    import pretty_state, save_checkpoint
from config.settings         import OUTPUTS_DIR, MAX_ITERATIONS
from pathlib import Path

# ── Evaluation framework ───────────────────────────────────────────────────────
from evaluation.metrics   import RunMetrics
from evaluation.collector import (
    collect_routing, collect_execution, collect_critic,
    collect_final, start_timer, stop_timer,
)
from evaluation.scorer    import evaluate
from evaluation.reporter  import print_report, save_markdown_report
from evaluation.history   import save_run, print_history_summary

# Global metrics instance — reset at the start of each run()
_metrics: RunMetrics = RunMetrics()


# ── Instrumented node wrappers (collect metrics without modifying agents) ─────

def _router(state: PipelineState) -> dict:
    start_timer("router")
    result = router(state)
    merged = {**state, **result}
    stop_timer("router", merged.get("iteration", 0), _metrics)
    collect_routing(merged, _metrics)
    return result


def _analyst(state: PipelineState) -> dict:
    start_timer("analyst")
    result = analyst(state)
    stop_timer("analyst", state.get("iteration", 0), _metrics)
    _metrics.total_llm_calls += 1
    return result


def _tool_executor(state: PipelineState) -> dict:
    start_timer("tool_executor")
    result = tool_executor(state)
    stop_timer("tool_executor", state.get("iteration", 0), _metrics)
    merged = {**state, **result}
    task   = ""
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "analyst":
            task = m.get("content", "")
            break
    collect_execution(merged, _metrics, task=task)
    return result


def _researcher(state: PipelineState) -> dict:
    start_timer("researcher")
    result = researcher(state)
    stop_timer("researcher", state.get("iteration", 0), _metrics)
    _metrics.total_llm_calls += 1
    return result


def _writer(state: PipelineState) -> dict:
    start_timer("writer")
    result = writer(state)
    stop_timer("writer", state.get("iteration", 0), _metrics)
    _metrics.total_llm_calls += 1
    return result


def _critic(state: PipelineState) -> dict:
    start_timer("critic")
    result = critic(state)
    stop_timer("critic", state.get("iteration", 0), _metrics)
    merged = {**state, **result}
    collect_critic(merged, _metrics)
    return result


# ── Conditional edges ──────────────────────────────────────────────────────────

def route_to_agent(state: PipelineState) -> str:
    # Hard ceiling enforced here as a belt-and-suspenders safety net
    if state.get("iteration", 0) > MAX_ITERATIONS:
        return END
    next_agent = state.get("next_agent", "writer")
    if next_agent == "END":
        return END
    if next_agent in ("analyst", "researcher", "writer"):
        return next_agent
    return "writer"


def critic_edge(state: PipelineState) -> str:
    iteration = state.get("iteration", 0)
    # Hard ceiling — never loop past MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS:
        print(f"[Pipeline] Hard ceiling reached ({iteration}/{MAX_ITERATIONS}) → forcing END")
        return END
    if state.get("evaluation", "") == "goal_met":
        return END
    return "router"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("goal_interpreter", goal_interpreter)
    g.add_node("planner",          planner)
    g.add_node("router",           _router)
    g.add_node("analyst",          _analyst)
    g.add_node("tool_executor",    _tool_executor)
    g.add_node("researcher",       _researcher)
    g.add_node("writer",           _writer)
    g.add_node("critic",           _critic)

    g.set_entry_point("goal_interpreter")

    g.add_edge("goal_interpreter", "planner")
    g.add_edge("planner",          "router")

    g.add_conditional_edges("router", route_to_agent, {
        "analyst":    "analyst",
        "researcher": "researcher",
        "writer":     "writer",
        END:          END,
    })

    g.add_edge("analyst",      "tool_executor")
    g.add_edge("tool_executor", "router")
    g.add_edge("researcher",   "router")
    g.add_edge("writer",       "critic")

    g.add_conditional_edges("critic", critic_edge, {
        "router": "router",
        END:      END,
    })

    return g.compile()


# ── Run ────────────────────────────────────────────────────────────────────────

def run(query: str, dataset_path: str = "") -> dict:
    global _metrics

    _metrics = RunMetrics(
        query=query,
        dataset_path=dataset_path,
        max_iterations_limit=MAX_ITERATIONS,
    )

    print("\n" + "=" * 60)
    print("  AGENTIC ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"  Query   : {query}")
    print(f"  Dataset : {dataset_path or 'none (synthetic)'}")
    print("=" * 60 + "\n")

    has_dataset     = bool(dataset_path)
    dataset_summary = (
        "A CSV dataset is available. Load with: pd.read_csv(DATASET_PATH)\n"
        "DATASET_PATH is pre-loaded in sandbox — do NOT hardcode any path."
        if has_dataset
        else "No dataset provided — generate a realistic synthetic sales/revenue dataset."
    )

    initial_state: PipelineState = {
        "user_query":       query,
        "dataset_path":     dataset_path,
        "goal":             "",
        "plan":             [],
        "next_agent":       "",
        "routing_reason":   "",
        "messages":         [],
        "code_to_run":      "",
        "tool_results":     [],
        "dataset_summary":  dataset_summary,
        "plots":            [],
        "analysis_results": [],
        "evaluation":       "",
        "critic_feedback":  "",
        "final_report":     "",
        "iteration":        0,
        "plan_step_index":  0,
        "error":            "",
    }

    graph = build_graph()
    final = graph.invoke(initial_state)

    # Collect final metrics and evaluate
    collect_final(final, _metrics)
    report = evaluate(_metrics)

    print("\n" + pretty_state(final))
    print_report(report)

    # Save all outputs
    ckpt      = save_checkpoint(final, label="final")
    eval_path = save_markdown_report(report)
    save_run(report)

    print(f"\n[Pipeline] State checkpoint  → {ckpt}")
    print(f"[Pipeline] Eval report       → {eval_path}")

    if final.get("final_report"):
        report_path = Path(OUTPUTS_DIR) / "final_report.md"
        report_path.write_text(final["final_report"])
        print(f"[Pipeline] Final report      → {report_path}")

    return final


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Analysis Pipeline")
    parser.add_argument(
        "--query",
        default="Analyse quarterly sales data and identify trends, anomalies, and actionable insights.",
        help="Analysis query",
    )
    parser.add_argument("--dataset", default="", help="Path to a CSV dataset")
    parser.add_argument("--history", action="store_true",
                        help="Print evaluation history table and exit")
    args = parser.parse_args()

    if args.history:
        print_history_summary()
        sys.exit(0)

    result = run(query=args.query, dataset_path=args.dataset)

    print("\n" + "=" * 60)
    print("  FINAL REPORT PREVIEW (first 800 chars)")
    print("=" * 60)
    print(result.get("final_report", "No report generated.")[:800])
    print("...")
