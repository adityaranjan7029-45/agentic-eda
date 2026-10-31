"""The analysis graph (LangGraph).

    plan -> analyse -> narrate -> critic --passed--> finalize
                         ^          |
                         +-revise---+   (while revisions and tokens remain)
                                    |
                                    +--> repair -> finalize   (otherwise)

Only two nodes use the LLM: `plan` (optional, picks preprocessing steps from a
fixed list) and `narrate` (writes the report from the facts sheet). The
statistics, charts and the Critic are deterministic. LangGraph is used for what
it's good at: a loop with conditional routing and shared state. A plain
pipeline would do for the rest.

Every failure path still ends in a verified report: no LLM / all providers down
/ budget spent -> template narrative; Critic still unhappy after revisions ->
failing sentences removed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from eda_engine.agents.charts import ChartSpec, build_charts
from eda_engine.agents.critic import Verdict, critique, repair
from eda_engine.agents.facts import Facts, build_facts
from eda_engine.agents.narrator import (
    narrator_messages,
    normalize_narrative,
    planner_messages,
    template_narrative,
    validate_plan,
)
from eda_engine.findings import FindingsConfig, discover_findings
from eda_engine.llm import BudgetExceeded, LLMClient, LLMError, estimate_messages
from eda_engine.packs import run_fintech_pack
from eda_engine.preprocess import recommend_plan, run_preprocessing
from eda_engine.profile import build_compact_profile
from eda_engine.schema import infer_schema


@dataclass(frozen=True)
class AgentConfig:
    llm_planner: bool = True
    planner_profile_tokens: int = 900
    planner_max_tokens: int = 400
    narrator_max_tokens: int = 1500
    max_revisions: int = 2
    findings: FindingsConfig = field(default_factory=FindingsConfig)


class RunState(TypedDict, total=False):
    data: pd.DataFrame
    target: str | None
    llm: LLMClient | None
    config: AgentConfig
    emit: Callable[[str, str], None]
    schema: Any
    plan: list[str]
    plan_source: str
    plan_rejected: list[str]
    prep: Any
    pack: Any
    facts: Facts
    charts: list[ChartSpec]
    narrative: dict
    narrative_raw: str
    narrative_source: str
    verdict: Verdict
    history: list[dict]
    revisions: int
    status: str
    notes: list[str]


@dataclass
class AnalysisReport:
    target: str | None
    n_rows: int
    plan: list[str]
    plan_source: str
    preprocessing: list[dict]
    excluded: dict[str, str]
    findings: dict
    fintech: dict
    charts: list[ChartSpec]
    narrative: dict
    narrative_source: str          # "llm" | "template"
    verification: dict             # status, revisions, issues per round
    tokens: dict
    notes: list[str]
    seconds: float

    def to_dict(self) -> dict:
        return {
            "target": self.target, "n_rows": self.n_rows, "plan": self.plan, "plan_source": self.plan_source,
            "preprocessing": self.preprocessing, "excluded": self.excluded, "findings": self.findings,
            "fintech": self.fintech, "charts": [c.to_dict() for c in self.charts], "narrative": self.narrative,
            "narrative_source": self.narrative_source, "verification": self.verification,
            "tokens": self.tokens, "notes": self.notes, "seconds": round(self.seconds, 2),
        }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _plan(state: RunState) -> RunState:
    cfg, llm, df = state["config"], state.get("llm"), state["data"]
    schema = infer_schema(df, target=state.get("target"))
    default = recommend_plan(schema)
    plan, source, rejected, notes = default, "default", [], list(state.get("notes", []))
    if llm is not None and cfg.llm_planner:
        profile = build_compact_profile(df, schema, max_tokens=cfg.planner_profile_tokens)
        try:
            raw, _ = llm.complete_json(planner_messages(profile.text, default), tier="small",
                                       max_tokens=cfg.planner_max_tokens, purpose="plan")
            plan, rejected = validate_plan(raw)
            source = "llm"
        except (LLMError, BudgetExceeded) as exc:
            notes.append(f"planner unavailable, used default plan ({type(exc).__name__})")
    state["emit"]("plan", f"{source}: {', '.join(plan) or 'no optional steps'}")
    return {"schema": schema, "plan": plan, "plan_source": source, "plan_rejected": rejected, "notes": notes}


def _analyse(state: RunState) -> RunState:
    cfg = state["config"]
    prep = run_preprocessing(state["data"], state["schema"], state["plan"])
    pack = run_fintech_pack(prep, discover_findings(prep, cfg.findings))
    facts = build_facts(prep, pack)
    charts = build_charts(facts, pack)
    state["emit"]("analyse", f"{len(pack.findings.findings)} findings, {len(charts)} charts")
    return {"prep": prep, "pack": pack, "facts": facts, "charts": charts}


def _narrate(state: RunState) -> RunState:
    cfg, llm, facts, charts = state["config"], state.get("llm"), state["facts"], state["charts"]
    notes = list(state.get("notes", []))
    revision = state.get("revisions", 0)
    if llm is not None:
        verdict = state.get("verdict")
        messages = narrator_messages(facts, charts, feedback=verdict.feedback() if verdict else None,
                                     previous=state.get("narrative_raw"))
        try:
            raw, response = llm.complete_json(messages, max_tokens=cfg.narrator_max_tokens,
                                              purpose="revise" if verdict else "narrate")
            state["emit"]("narrate", f"draft {revision + 1} by {response.usage.provider}")
            return {"narrative": normalize_narrative(raw, facts, charts), "narrative_raw": response.text,
                    "narrative_source": "llm", "notes": notes}
        except (LLMError, BudgetExceeded) as exc:
            notes.append(f"narrator unavailable, used template report ({type(exc).__name__}: {exc})"[:300])
    state["emit"]("narrate", "template report")
    return {"narrative": template_narrative(facts, charts), "narrative_raw": "", "narrative_source": "template",
            "notes": notes}


def _critic(state: RunState) -> RunState:
    verdict = critique(state["narrative"], state["facts"])
    history = [*state.get("history", []),
               {"round": len(state.get("history", [])) + 1, "source": state["narrative_source"],
                "issues": [i.to_dict() for i in verdict.issues]}]
    state["emit"]("critic", "passed" if verdict.passed else f"{len(verdict.issues)} issue(s)")
    return {"verdict": verdict, "history": history}


def _route_after_critic(state: RunState) -> str:
    if state["verdict"].passed:
        return "finalize"
    llm, cfg = state.get("llm"), state["config"]
    if state["narrative_source"] != "llm" or llm is None or state.get("revisions", 0) >= cfg.max_revisions:
        return "repair"
    revision = narrator_messages(state["facts"], state["charts"], state["verdict"].feedback(),
                                 state.get("narrative_raw"))
    if estimate_messages(revision) + cfg.narrator_max_tokens > llm.budget.remaining:
        return "repair"
    return "revise"


def _revise(state: RunState) -> RunState:
    return {"revisions": state.get("revisions", 0) + 1}


def _repair(state: RunState) -> RunState:
    fixed = repair(state["narrative"], state["verdict"], state["facts"])
    verdict = critique(fixed, state["facts"])
    state["emit"]("repair", f"removed failing sentences; {len(verdict.issues)} issue(s) left")
    return {"narrative": fixed, "verdict": verdict, "status": "repaired" if verdict.passed else "unverified"}


def _finalize(state: RunState) -> RunState:
    status = state.get("status") or ("passed" if state.get("revisions", 0) == 0 else "passed_after_revision")
    state["emit"]("finalize", status)
    return {"status": status}


def build_graph():
    graph = StateGraph(RunState)
    for name, fn in [("plan", _plan), ("analyse", _analyse), ("narrate", _narrate), ("critic", _critic),
                     ("revise", _revise), ("repair", _repair), ("finalize", _finalize)]:
        graph.add_node(name, fn)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "analyse")
    graph.add_edge("analyse", "narrate")
    graph.add_edge("narrate", "critic")
    graph.add_conditional_edges("critic", _route_after_critic,
                                {"finalize": "finalize", "revise": "revise", "repair": "repair"})
    graph.add_edge("revise", "narrate")
    graph.add_edge("repair", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


_GRAPH = None


def run_analysis(df: pd.DataFrame, target: str | None = None, llm: LLMClient | None = None,
                 config: AgentConfig | None = None,
                 on_event: Callable[[str, str], None] | None = None) -> AnalysisReport:
    """Run the full analysis. `llm=None` gives a deterministic, template-narrated report."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    started = time.perf_counter()
    state = _GRAPH.invoke({
        "data": df, "target": target, "llm": llm, "config": config or AgentConfig(),
        "emit": on_event or (lambda stage, detail: None), "revisions": 0, "history": [], "notes": [],
    })
    prep, pack = state["prep"], state["pack"]
    return AnalysisReport(
        target=state["schema"].target, n_rows=len(prep.data), plan=state["plan"],
        plan_source=state["plan_source"],
        preprocessing=[{"name": s.name, "kind": s.kind, "detail": s.detail} for s in prep.steps],
        excluded=dict(prep.excluded), findings=pack.findings.to_dict(), fintech=pack.to_dict(),
        charts=state["charts"], narrative=state["narrative"], narrative_source=state["narrative_source"],
        verification={"status": state["status"], "revisions": state.get("revisions", 0),
                      "rounds": state["history"],
                      "remaining_issues": [i.to_dict() for i in state["verdict"].issues]},
        tokens=llm.budget.summary() if llm is not None else {"limit": 0, "used": 0, "calls": 0},
        notes=state.get("notes", []) + ([f"planner proposed unknown steps: {state['plan_rejected']}"]
                                        if state.get("plan_rejected") else []),
        seconds=time.perf_counter() - started,
    )
