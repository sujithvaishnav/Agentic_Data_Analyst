"""
evaluation/collector.py — Hooks that collect metrics from pipeline state.
Call the appropriate collect_*() function after each node runs.
"""

from __future__ import annotations
import time
from evaluation.metrics import (
    RunMetrics, RoutingEvent, ExecutionEvent, CriticEvent, AgentTimingEvent
)

_agent_start_times: dict[str, float] = {}


def start_timer(agent: str) -> None:
    _agent_start_times[agent] = time.perf_counter()


def stop_timer(agent: str, iteration: int, metrics: RunMetrics) -> None:
    start = _agent_start_times.pop(agent, None)
    if start is None: return
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    metrics.agent_timings.append(AgentTimingEvent(
        agent=agent, iteration=iteration, duration_ms=duration_ms
    ))


def collect_routing(state: dict, metrics: RunMetrics) -> None:
    iteration  = state.get("iteration", 0)
    next_agent = state.get("next_agent", "")
    reason     = state.get("routing_reason", "")
    method     = "llm_fallback" if "LLM" in reason or "llm" in reason.lower() else "rule_based"

    from config.settings import VALID_NODES
    was_valid = next_agent in VALID_NODES + ["END"]

    metrics.routing_events.append(RoutingEvent(
        iteration=iteration, method=method,
        agent_chosen=next_agent, reason=reason[:200], was_valid=was_valid,
    ))
    metrics.total_iterations = iteration
    metrics.total_llm_calls += (1 if method == "llm_fallback" else 0)


def collect_execution(state: dict, metrics: RunMetrics, task: str = "") -> None:
    tool_results = state.get("tool_results", [])
    if not tool_results: return

    latest    = tool_results[-1]
    iteration = state.get("iteration", 0)

    event = ExecutionEvent(
        iteration=iteration, task=task,
        code_chars=len(latest.get("code", "")),
        success=latest.get("success", False),
        stdout_chars=len(latest.get("stdout", "")),
        figures_saved=len(latest.get("figures", [])),
        stderr_preview=latest.get("stderr", "")[:200],
    )
    metrics.execution_events.append(event)
    metrics.total_code_chars_generated += event.code_chars

    if not event.success:
        metrics.execution_failures += 1
    else:
        prev_failures = [
            e for e in metrics.execution_events[:-1]
            if not e.success and e.iteration == iteration
        ]
        if prev_failures:
            metrics.execution_retries += 1

    metrics.total_llm_calls += 1


def collect_critic(state: dict, metrics: RunMetrics) -> None:
    critic_msgs = [m for m in state.get("messages", []) if m.get("role") == "critic"]
    if not critic_msgs: return

    content    = critic_msgs[-1].get("content", "")
    evaluation = "needs_work"
    score      = 0
    try:
        for part in content.split():
            if part.startswith("evaluation="):
                evaluation = part.split("=")[1]
            if part.startswith("score="):
                score = int(part.split("=")[1].split("/")[0])
    except Exception:
        pass

    metrics.critic_events.append(CriticEvent(
        iteration=state.get("iteration", 0),
        score=score, evaluation=evaluation,
        feedback=state.get("critic_feedback", "")[:300],
    ))
    metrics.total_llm_calls += 1
    if evaluation == "goal_met":
        metrics.goal_met = True


def collect_final(state: dict, metrics: RunMetrics) -> None:
    import datetime
    metrics.finished_at        = datetime.datetime.now().isoformat()
    metrics.final_report_chars = len(state.get("final_report", ""))
    metrics.unique_plots       = len(list(dict.fromkeys(state.get("plots", []))))
    metrics.total_findings     = len(state.get("analysis_results", []))
    metrics.hit_ceiling        = state.get("iteration", 0) >= metrics.max_iterations_limit
    metrics.plan_steps_total   = len(state.get("plan", []))
    metrics.plan_steps_completed = state.get("plan_step_index", 0)
    metrics.total_llm_calls   += 2  # goal_interpreter + planner
