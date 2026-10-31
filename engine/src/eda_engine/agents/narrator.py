"""Narrator and planner prompts, plus the no-LLM template report.

The narrator turns the facts sheet into a short report a credit analyst would
read. It never sees raw data and is told to use only numbers from the sheet;
the Critic enforces that. If the LLM is unavailable or the budget is spent,
`template_narrative()` writes a plainer report from the same facts, so every
run produces a verified report.
"""

from __future__ import annotations

from eda_engine.agents.charts import ChartSpec
from eda_engine.agents.facts import Facts, pct
from eda_engine.preprocess import OPTIONAL_STEPS

NARRATOR_SYSTEM = """You are a senior credit-risk analyst writing a short EDA report.
Rules:
- Use ONLY the facts sheet. Every number you write must appear on it (you may round, e.g. 23.7% -> 24%).
- Cite findings by ID in each section's "findings" list.
- Say "is associated with", never "causes" or "leads to".
- If a section uses a finding marked tentative, say "tentative" in that section.
- Interest rates, grades and approved amounts are lender decisions: describe them as reflecting the lender's view.
- Mention leakage and fair-lending items if the sheet lists them.
Reply with JSON only:
{"headline": "...", "sections": [{"title": "...", "text": "2-4 sentences", "findings": ["F2"], "charts": ["C1"]}],
 "caveats": ["..."]}
Write 3-6 sections."""

PLANNER_SYSTEM = """You choose optional preprocessing steps for an EDA run.
Allowed steps (choose any subset, by exact name):
- missing_indicators: add a 0/1 column for each column with >= 5% missing (missingness can be informative)
- group_rare_categories: merge categories under 1% into "other"
- cap_outliers: cap very skewed numeric columns at the 1st/99th percentile
- date_features: add month and day-of-week from date columns
- impute: fill missing values (off by default: statistics here handle missing values)
Reply with JSON only: {"steps": ["..."], "reason": "one sentence"}"""


def narrator_messages(facts: Facts, charts: list[ChartSpec], feedback: str | None = None,
                      previous: str | None = None) -> list[dict]:
    chart_lines = "\n".join(f"{c.id}: {c.title}" + (f" (for {c.finding_id})" if c.finding_id else "")
                            for c in charts)
    messages = [
        {"role": "system", "content": NARRATOR_SYSTEM},
        {"role": "user", "content": f"FACTS SHEET\n{facts.text}\n\nCHARTS\n{chart_lines}"},
    ]
    if feedback:
        messages += [
            {"role": "assistant", "content": previous or ""},
            {"role": "user", "content": "A checker found these problems. Fix only these and return the "
                                        f"full JSON again:\n{feedback}"},
        ]
    return messages


def planner_messages(profile_text: str, default_plan: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Dataset profile:\n{profile_text}\n\nDefault plan: {default_plan}"},
    ]


def validate_plan(raw: dict) -> tuple[list[str], list[str]]:
    """Keep only known optional steps, in canonical order. Returns (plan, rejected)."""
    steps = raw.get("steps", []) if isinstance(raw, dict) else []
    steps = [s for s in steps if isinstance(s, str)]
    plan = [s for s in OPTIONAL_STEPS if s in steps]
    rejected = sorted(set(steps) - set(OPTIONAL_STEPS))
    return plan, rejected


def normalize_narrative(raw: dict, facts: Facts, charts: list[ChartSpec]) -> dict:
    """Coerce model output into the expected shape; drop unknown chart IDs."""
    chart_ids = {c.id for c in charts}
    sections = []
    for s in raw.get("sections", []) if isinstance(raw.get("sections"), list) else []:
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            continue
        sections.append({
            "title": str(s.get("title", "")).strip(),
            "text": str(s.get("text", "")).strip(),
            "findings": [str(f) for f in s.get("findings", []) if isinstance(f, str | int)],
            "charts": [str(c) for c in s.get("charts", []) if str(c) in chart_ids],
        })
    caveats = [str(c).strip() for c in raw.get("caveats", []) if str(c).strip()] \
        if isinstance(raw.get("caveats"), list) else []
    return {"headline": str(raw.get("headline", "")).strip(), "sections": sections, "caveats": caveats}


def template_narrative(facts: Facts, charts: list[ChartSpec]) -> dict:
    """A plain report from the facts alone (no LLM). Built only from facts-sheet numbers."""
    chart_for = {c.finding_id: c.id for c in charts if c.finding_id}
    drivers = [(fid, f) for fid, f in facts.findings.items() if f.kind == "driver"]
    sections = []
    for fid, f in drivers[:5]:
        ev = f.evidence
        if "top_quintile_rate" in ev:
            text = (f"The {f.column} top 20% has a rate of {pct(ev['top_quintile_rate'])}, "
                    f"compared with {pct(ev['bottom_quintile_rate'])} in the bottom 20%.")
        else:
            rates = sorted(ev["rates"].items(), key=lambda kv: -kv[1])
            text = (f"The rate is {pct(rates[0][1])} for {f.column} = {rates[0][0]} and "
                    f"{pct(rates[-1][1])} for {rates[-1][0]}.")
        if f.confidence == "tentative":
            text += " This finding is tentative."
        sections.append({"title": f"{f.column} is associated with the outcome", "text": text,
                         "findings": [fid], "charts": [chart_for[fid]] if fid in chart_for else []})
    for fid in sorted(facts.lender_decisions):
        f = facts.findings[fid]
        sections.append({"title": f"{f.column} reflects the lender's view",
                         "text": f"{f.column} is set by the lender from its own risk assessment, so it "
                                 "describes the lender's view of the applicant rather than the applicant.",
                         "findings": [fid], "charts": []})
    top = drivers[0][1].column if drivers else None
    headline = (f"{facts.n_drivers} independent drivers are associated with the outcome; the strongest is {top}."
                if top else "No variable passed the significance and effect-size gates.")
    return {"headline": headline, "sections": sections, "caveats": list(facts.caveats)}
