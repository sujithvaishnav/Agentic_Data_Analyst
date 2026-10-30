"""
evaluation/metrics.py — Core metric definitions and event dataclasses.

RunMetrics is populated incrementally as the pipeline runs,
then passed to the Evaluator for scoring.

Metric categories:
  A. Pipeline Performance  — iterations, routing accuracy, execution rate
  B. Output Quality        — report completeness, critic scores, findings depth
  C. Efficiency            — LLM calls, time per agent
  D. Robustness            — error recovery rate, retry success, ceiling hits
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RoutingEvent:
    iteration:    int
    method:       str          # "rule_based" | "llm_fallback"
    agent_chosen: str
    reason:       str
    was_valid:    bool         # was the chosen agent in VALID_NODES?


@dataclass
class ExecutionEvent:
    iteration:      int
    task:           str
    code_chars:     int
    success:        bool
    stdout_chars:   int
    figures_saved:  int
    stderr_preview: str = ""


@dataclass
class CriticEvent:
    iteration:  int
    score:      int            # 0-10
    evaluation: str            # "goal_met" | "needs_work"
    feedback:   str


@dataclass
class AgentTimingEvent:
    agent:       str
    iteration:   int
    duration_ms: float


@dataclass
class RunMetrics:
    # Identity
    run_id:       str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    query:        str = ""
    dataset_path: str = ""
    started_at:   str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at:  str = ""

    # A. Pipeline performance
    total_iterations:     int  = 0
    max_iterations_limit: int  = 8
    hit_ceiling:          bool = False

    routing_events:   list[RoutingEvent]    = field(default_factory=list)
    execution_events: list[ExecutionEvent]  = field(default_factory=list)

    # B. Output quality
    critic_events:      list[CriticEvent]      = field(default_factory=list)
    final_report_chars: int  = 0
    unique_plots:       int  = 0
    total_findings:     int  = 0
    goal_met:           bool = False

    # C. Efficiency
    agent_timings:            list[AgentTimingEvent] = field(default_factory=list)
    total_llm_calls:          int = 0
    total_code_chars_generated: int = 0

    # D. Robustness
    execution_failures:    int = 0
    execution_retries:     int = 0
    plan_steps_completed:  int = 0
    plan_steps_total:      int = 0

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def execution_success_rate(self) -> float:
        total = len(self.execution_events)
        if total == 0: return 0.0
        return round(sum(1 for e in self.execution_events if e.success) / total * 100, 1)

    @property
    def rule_based_routing_pct(self) -> float:
        total = len(self.routing_events)
        if total == 0: return 0.0
        rule = sum(1 for e in self.routing_events if e.method == "rule_based")
        return round(rule / total * 100, 1)

    @property
    def llm_fallback_routing_pct(self) -> float:
        return round(100 - self.rule_based_routing_pct, 1)

    @property
    def avg_critic_score(self) -> Optional[float]:
        if not self.critic_events: return None
        return round(sum(e.score for e in self.critic_events) / len(self.critic_events), 1)

    @property
    def final_critic_score(self) -> Optional[int]:
        if not self.critic_events: return None
        return self.critic_events[-1].score

    @property
    def plan_completion_rate(self) -> float:
        if self.plan_steps_total == 0: return 0.0
        return round(self.plan_steps_completed / self.plan_steps_total * 100, 1)

    @property
    def avg_agent_duration_ms(self) -> dict[str, float]:
        from collections import defaultdict
        totals: dict[str, list[float]] = defaultdict(list)
        for t in self.agent_timings:
            totals[t.agent].append(t.duration_ms)
        return {a: round(sum(v)/len(v), 1) for a, v in totals.items()}

    @property
    def iterations_efficiency(self) -> float:
        if self.max_iterations_limit == 0: return 0.0
        return round(self.total_iterations / self.max_iterations_limit * 100, 1)
