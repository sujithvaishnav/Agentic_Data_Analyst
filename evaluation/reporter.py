"""
evaluation/reporter.py — Renders EvalReport to terminal and Markdown file.
"""

from __future__ import annotations
from pathlib import Path
from evaluation.scorer import EvalReport
from config.settings import OUTPUTS_DIR

GRADE_EMOJI = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
BAR_FILL    = "█"
BAR_EMPTY   = "░"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return BAR_FILL * filled + BAR_EMPTY * (width - filled)


def print_report(report: EvalReport) -> None:
    m = report.raw_metrics
    print("\n" + "=" * 62)
    print("  PIPELINE EVALUATION REPORT")
    print("=" * 62)
    print(f"  Run ID   : {report.run_id}")
    print(f"  Query    : {report.query[:60]}")
    print(f"  Goal met : {'YES ✓' if m.goal_met else 'NO ✗'}")
    print()

    emoji = GRADE_EMOJI.get(report.overall_grade, "⚪")
    print(f"  OVERALL  {report.overall_score:5.1f}/100  [{_bar(report.overall_score)}]  Grade {report.overall_grade} {emoji}")
    print()

    print(f"  {'Dimension':<24} {'Score':>6}  {'Grade':>5}  Bar")
    print(f"  {'-'*24} {'-'*6}  {'-'*5}  {'-'*20}")
    for d in report.dimensions:
        emoji = GRADE_EMOJI.get(d.grade, "⚪")
        print(f"  {d.name:<24} {d.score:>5.1f}    {d.grade} {emoji}  {_bar(d.score)}")

    print()
    print("  ── Raw Metrics ─────────────────────────────────────────")
    print(f"  Iterations used       : {m.total_iterations} / {m.max_iterations_limit}")
    print(f"  Iteration budget used : {m.iterations_efficiency}%")
    print(f"  Plan steps            : {m.plan_steps_completed} / {m.plan_steps_total} completed")
    print(f"  Code execution rate   : {m.execution_success_rate}%")
    print(f"  Execution failures    : {m.execution_failures}")
    print(f"  Execution retries     : {m.execution_retries}")
    print(f"  Rule-based routing    : {m.rule_based_routing_pct}%")
    print(f"  LLM fallback routing  : {m.llm_fallback_routing_pct}%")
    print(f"  Total LLM calls       : {m.total_llm_calls}")
    print(f"  Unique plots          : {m.unique_plots}")
    print(f"  Final report chars    : {m.final_report_chars}")
    print(f"  Avg critic score      : {m.avg_critic_score}/10")
    print(f"  Final critic score    : {m.final_critic_score}/10")
    print(f"  Hit ceiling           : {'YES ⚠' if m.hit_ceiling else 'No ✓'}")
    print()

    if m.critic_events:
        print("  ── Critic Score Progression ────────────────────────────")
        for e in m.critic_events:
            print(f"  Iter {e.iteration:2d}: {e.score:2d}/10  [{_bar(e.score * 10)}]  {e.evaluation}")
        print()

    print("  ── Dimension Details ───────────────────────────────────")
    for d in report.dimensions:
        emoji = GRADE_EMOJI.get(d.grade, "⚪")
        print(f"\n  {d.name} — {d.score:.0f}/100  {emoji}")
        for s in d.signals:
            print(f"    ✓ {s}")
        for w in d.warnings:
            print(f"    ✗ {w}")
    print()

    print("  ── Top Upgrade Recommendations ─────────────────────────")
    for i, upgrade in enumerate(report.top_upgrades, 1):
        print(f"  {i}. {upgrade}")
    print()
    print(f"  {report.summary}")
    print("=" * 62)


def save_markdown_report(report: EvalReport) -> str:
    m = report.raw_metrics
    lines = [
        "# Pipeline Evaluation Report", "",
        f"**Run ID:** `{report.run_id}`  ",
        f"**Query:** {report.query}  ",
        f"**Finished:** {m.finished_at}  ", "",
        "---", "", "## Overall Score", "",
        "| Score | Grade | Goal Met |",
        "|-------|-------|----------|",
        f"| **{report.overall_score}/100** | **{report.overall_grade}** | {'✅ Yes' if m.goal_met else '❌ No'} |",
        "", "---", "", "## Dimension Scores", "",
        "| Dimension | Score | Grade |",
        "|-----------|-------|-------|",
    ]
    for d in report.dimensions:
        lines.append(f"| {d.name} | {d.score:.1f}/100 | {d.grade} |")

    lines += [
        "", "---", "", "## Raw Metrics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Iterations used | {m.total_iterations} / {m.max_iterations_limit} |",
        f"| Iteration budget used | {m.iterations_efficiency}% |",
        f"| Plan completion | {m.plan_completion_rate}% ({m.plan_steps_completed}/{m.plan_steps_total} steps) |",
        f"| Code execution success rate | {m.execution_success_rate}% |",
        f"| Execution failures | {m.execution_failures} |",
        f"| Execution retries (recovered) | {m.execution_retries} |",
        f"| Rule-based routing | {m.rule_based_routing_pct}% |",
        f"| LLM fallback routing | {m.llm_fallback_routing_pct}% |",
        f"| Total LLM calls | {m.total_llm_calls} |",
        f"| Unique plots produced | {m.unique_plots} |",
        f"| Final report length | {m.final_report_chars} chars |",
        f"| Avg critic score | {m.avg_critic_score}/10 |",
        f"| Final critic score | {m.final_critic_score}/10 |",
        f"| Hit iteration ceiling | {'Yes ⚠️' if m.hit_ceiling else 'No ✅'} |",
        "", "---", "", "## Critic Score Progression", "",
        "| Iteration | Score | Evaluation |",
        "|-----------|-------|------------|",
    ]
    for e in m.critic_events:
        lines.append(f"| {e.iteration} | {e.score}/10 | {e.evaluation} |")

    lines += ["", "---", "", "## Dimension Details", ""]
    for d in report.dimensions:
        lines.append(f"### {d.name} — {d.score:.0f}/100 (Grade {d.grade})")
        lines.append("")
        if d.signals:
            lines.append("**Strengths:**")
            for s in d.signals: lines.append(f"- ✅ {s}")
        if d.warnings:
            lines.append(""); lines.append("**Weaknesses:**")
            for w in d.warnings: lines.append(f"- ❌ {w}")
        if d.upgrades:
            lines.append(""); lines.append("**Upgrade Suggestions:**")
            for u in d.upgrades: lines.append(f"- 🔧 {u}")
        lines.append("")

    lines += [
        "---", "", "## Top 3 Upgrade Priorities", "",
    ]
    for i, u in enumerate(report.top_upgrades, 1):
        lines.append(f"{i}. {u}")

    lines += ["", "---", "", f"*{report.summary}*", ""]

    if m.avg_agent_duration_ms:
        lines += ["## Agent Timing", "", "| Agent | Avg Duration (ms) |", "|-------|------------------|"]
        for agent, ms in m.avg_agent_duration_ms.items():
            lines.append(f"| {agent} | {ms} ms |")
        lines.append("")

    content = "\n".join(lines)
    path    = Path(OUTPUTS_DIR) / "evaluation_report.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
