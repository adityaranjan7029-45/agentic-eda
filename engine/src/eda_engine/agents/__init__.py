"""Agents: an LLM planner and narrator, a deterministic Critic, wired with LangGraph."""

from eda_engine.agents.charts import ChartSpec, build_charts
from eda_engine.agents.critic import Issue, Verdict, critique, extract_numbers, repair
from eda_engine.agents.facts import Facts, build_facts
from eda_engine.agents.graph import AgentConfig, AnalysisReport, build_graph, run_analysis
from eda_engine.agents.narrator import template_narrative, validate_plan

__all__ = [
    "AgentConfig", "AnalysisReport", "ChartSpec", "Facts", "Issue", "Verdict",
    "build_charts", "build_facts", "build_graph", "critique", "extract_numbers", "repair",
    "run_analysis", "template_narrative", "validate_plan",
]
