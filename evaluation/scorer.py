"""
evaluation/scorer.py — Scores RunMetrics into a structured EvalReport.

5 dimensions with weights:
  Output Quality        0.30
  Pipeline Performance  0.25
  Routing Efficiency    0.20
  Execution Reliability 0.15
  Resource Efficiency   0.10
"""

from __future__ import annotations
from dataclasses import dataclass
from evaluation.metrics import RunMetrics


@dataclass
class DimensionScore:
    name:     str
    score:    float
    grade:    str
    signals:  list[str]    # what contributed positively
    warnings: list[str]    # what pulled it down
    upgrades: list[str]    # actionable improvement suggestions


@dataclass
class EvalReport:
    run_id:        str
    query:         str
    overall_score: float
    overall_grade: str
    dimensions:    list[DimensionScore]
    summary:       str
    top_upgrades:  list[str]
    raw_metrics:   RunMetrics


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def score_pipeline_performance(m: RunMetrics) -> DimensionScore:
    signals, warnings, upgrades = [], [], []
    score = 100.0

    if m.hit_ceiling:
        score -= 40
        warnings.append(f"Hit MAX_ITERATIONS ceiling ({m.max_iterations_limit}) — didn't finish naturally")
        upgrades.append("Increase MAX_ITERATIONS or make Critic less strict")
    else:
        signals.append(f"Finished naturally at iteration {m.total_iterations}/{m.max_iterations_limit}")

    eff = m.iterations_efficiency
    if eff <= 70:
        signals.append(f"Used only {eff}% of iteration budget — efficient")
    elif eff <= 90:
        score -= 10
        warnings.append(f"Used {eff}% of iteration budget — close to ceiling")
        upgrades.append("Tune Critic threshold or restructure plan to reduce late-loop pressure")
    else:
        score -= 20
        warnings.append(f"Used {eff}% of iteration budget — very tight")

    pct = m.plan_completion_rate
    if pct >= 100:
        signals.append("All plan steps completed")
    elif pct >= 80:
        score -= 10
        warnings.append(f"Only {pct}% of plan steps completed")
        upgrades.append("Increase MAX_ITERATIONS to allow full plan execution")
    else:
        score -= 25
        warnings.append(f"Only {pct}% of plan steps completed — major incompleteness")
        upgrades.append("Simplify the plan or increase iteration ceiling")

    if m.goal_met:
        signals.append("Goal marked as met by Critic")
    else:
        score -= 15
        warnings.append("Goal was NOT marked as met")
        upgrades.append("Improve analyst code generation prompts for richer findings")

    return DimensionScore("Pipeline Performance", max(0, score), _grade(max(0, score)),
                          signals, warnings, upgrades)


def score_output_quality(m: RunMetrics) -> DimensionScore:
    signals, warnings, upgrades = [], [], []
    score = 100.0

    final_score = m.final_critic_score
    if final_score is None:
        score -= 30
        warnings.append("No critic evaluation recorded")
        upgrades.append("Ensure writer → critic edge is wired correctly in the graph")
    elif final_score >= 8:
        signals.append(f"Final critic score: {final_score}/10 — high quality")
    elif final_score >= 6:
        score -= 15
        warnings.append(f"Final critic score: {final_score}/10 — acceptable but improvable")
        upgrades.append("Enrich analyst findings with more statistical depth")
    else:
        score -= 35
        warnings.append(f"Final critic score: {final_score}/10 — low quality")
        upgrades.append("Major prompt improvement needed in analyst and writer agents")
        upgrades.append("Consider adding a dedicated statistics step to the plan")

    chars = m.final_report_chars
    if chars >= 2000:
        signals.append(f"Report is {chars} chars — comprehensive")
    elif chars >= 1000:
        score -= 10
        warnings.append(f"Report is only {chars} chars — somewhat thin")
        upgrades.append("Increase writer max_tokens and provide richer findings as input")
    else:
        score -= 25
        warnings.append(f"Report is only {chars} chars — very thin")
        upgrades.append("Check findings are actually reaching the writer agent")

    if m.unique_plots >= 3:
        signals.append(f"{m.unique_plots} unique plots produced — good visual coverage")
    elif m.unique_plots >= 1:
        score -= 10
        warnings.append(f"Only {m.unique_plots} plot(s) produced")
        upgrades.append("Add explicit visualisation tasks in analyst plan steps")
    else:
        score -= 20
        warnings.append("No plots produced — purely textual report")
        upgrades.append("Ensure analyst code always calls save_fig()")

    if m.total_findings >= 3:
        signals.append(f"{m.total_findings} analysis findings accumulated")
    elif m.total_findings >= 1:
        score -= 10
        warnings.append(f"Only {m.total_findings} finding(s) recorded")
    else:
        score -= 20
        warnings.append("No findings in analysis_results")

    return DimensionScore("Output Quality", max(0, score), _grade(max(0, score)),
                          signals, warnings, upgrades)


def score_routing_efficiency(m: RunMetrics) -> DimensionScore:
    signals, warnings, upgrades = [], [], []
    score = 100.0

    rb_pct = m.rule_based_routing_pct
    if rb_pct >= 75:
        signals.append(f"{rb_pct}% rule-based routing — efficient hybrid router")
    elif rb_pct >= 50:
        score -= 10
        warnings.append(f"Only {rb_pct}% rule-based routing — LLM called more than expected")
        upgrades.append("Add deterministic rules for common post-plan scenarios")
    else:
        score -= 25
        warnings.append(f"Only {rb_pct}% rule-based — router heavily LLM-dependent")
        upgrades.append("Review plan_step_index logic — falling back to LLM too often")

    invalid = sum(1 for e in m.routing_events if not e.was_valid)
    if invalid == 0:
        signals.append("All routing decisions were valid")
    else:
        score -= invalid * 10
        warnings.append(f"LLM router suggested {invalid} invalid agent(s)")
        upgrades.append("Strengthen router system prompt with explicit agent name list")

    agents_used = set(e.agent_chosen for e in m.routing_events if e.agent_chosen != "END")
    if len(agents_used) >= 3:
        signals.append(f"All {len(agents_used)} agent types used — good diversity")
    elif len(agents_used) == 2:
        score -= 10
        warnings.append("Only 2 agent types used — researcher or analyst may have been skipped")
    else:
        score -= 20
        warnings.append(f"Only {len(agents_used)} agent type(s) used")
        upgrades.append("Check plan generation — Planner may not be creating diverse plans")

    return DimensionScore("Routing Efficiency", max(0, score), _grade(max(0, score)),
                          signals, warnings, upgrades)


def score_execution_reliability(m: RunMetrics) -> DimensionScore:
    signals, warnings, upgrades = [], [], []
    score = 100.0

    rate = m.execution_success_rate
    if rate >= 90:
        signals.append(f"{rate}% code execution success rate — reliable sandbox")
    elif rate >= 70:
        score -= 15
        warnings.append(f"{rate}% execution success rate — some failures")
        upgrades.append("Review frequency alias patching; ensure LLM uses DATASET_PATH variable")
    else:
        score -= 35
        warnings.append(f"Only {rate}% execution success — frequent failures")
        upgrades.append("Add few-shot code examples to CODE_GEN_SYSTEM prompt")
        upgrades.append("Pass stderr back to analyst for self-correction on failure")

    if m.execution_failures > 0:
        warnings.append(f"{m.execution_failures} execution failure(s) recorded")
        if m.execution_retries > 0:
            signals.append(f"Retry mechanism recovered {m.execution_retries} time(s)")
        else:
            score -= 10
            upgrades.append("Implement error feedback loop: pass stderr back to analyst")

    if m.execution_failures == 0:
        signals.append("Zero execution failures — all generated code ran successfully")

    return DimensionScore("Execution Reliability", max(0, score), _grade(max(0, score)),
                          signals, warnings, upgrades)


def score_resource_efficiency(m: RunMetrics) -> DimensionScore:
    signals, warnings, upgrades = [], [], []
    score = 100.0

    if m.total_llm_calls <= 10:
        signals.append(f"Only {m.total_llm_calls} LLM calls — token-efficient")
    elif m.total_llm_calls <= 15:
        score -= 10
        warnings.append(f"{m.total_llm_calls} LLM calls — moderate usage")
    else:
        score -= 25
        warnings.append(f"{m.total_llm_calls} LLM calls — high, risk of rate limiting")
        upgrades.append("Cache goal interpretation and planning across similar queries")
        upgrades.append("Reduce critic retries by improving initial analysis quality")

    timings = m.avg_agent_duration_ms
    slow    = {a: t for a, t in timings.items() if t > 5000}
    if slow:
        score -= len(slow) * 5
        for agent, ms in slow.items():
            warnings.append(f"{agent} avg latency: {ms:.0f}ms — slow")
            upgrades.append(f"Consider faster Groq model for {agent} or cache its output")
    elif timings:
        signals.append("All agents within acceptable latency bounds")

    return DimensionScore("Resource Efficiency", max(0, score), _grade(max(0, score)),
                          signals, warnings, upgrades)


WEIGHTS = {
    "Pipeline Performance":  0.25,
    "Output Quality":        0.30,
    "Routing Efficiency":    0.20,
    "Execution Reliability": 0.15,
    "Resource Efficiency":   0.10,
}


def evaluate(metrics: RunMetrics) -> EvalReport:
    dimensions = [
        score_pipeline_performance(metrics),
        score_output_quality(metrics),
        score_routing_efficiency(metrics),
        score_execution_reliability(metrics),
        score_resource_efficiency(metrics),
    ]

    overall = round(sum(d.score * WEIGHTS[d.name] for d in dimensions), 1)

    # Top 3 upgrades from worst-scoring dimensions first
    sorted_dims  = sorted(dimensions, key=lambda d: d.score)
    all_upgrades = []
    for d in sorted_dims:
        for u in d.upgrades:
            if u not in all_upgrades:
                all_upgrades.append(u)
    top_upgrades = all_upgrades[:3]

    grade    = _grade(overall)
    goal_str = "Goal was met." if metrics.goal_met else "Goal was NOT met."
    summary  = (
        f"Run '{metrics.run_id}' scored {overall}/100 (Grade {grade}). "
        f"{goal_str} "
        f"Used {metrics.total_iterations}/{metrics.max_iterations_limit} iterations, "
        f"{metrics.execution_success_rate}% execution success, "
        f"{metrics.rule_based_routing_pct}% rule-based routing."
    )

    return EvalReport(
        run_id=metrics.run_id, query=metrics.query,
        overall_score=overall, overall_grade=grade,
        dimensions=dimensions, summary=summary,
        top_upgrades=top_upgrades, raw_metrics=metrics,
    )
