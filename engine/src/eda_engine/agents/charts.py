"""Chart specs built from evidence: data, not code.

v1 asked the LLM to write matplotlib code and executed it in a sandbox, which
is slow, fragile and a security risk. v2 emits Vega-Lite specs (a JSON chart
grammar the frontend renders) straight from the verified evidence. The LLM
never touches chart data, so a chart can't disagree with the numbers in the
report, and charts cost zero tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from eda_engine.agents.facts import Facts
from eda_engine.packs.fintech import FintechReport

VEGA_LITE = "https://vega.github.io/schema/vega-lite/v5.json"


@dataclass(frozen=True)
class ChartSpec:
    id: str                  # "C1"
    finding_id: str | None   # "F2", or None for dataset-level charts
    title: str
    kind: str                # "rate_by_group" | "woe_profile" | "variable_strength"
    spec: dict               # Vega-Lite

    def to_dict(self) -> dict:
        return {"id": self.id, "finding_id": self.finding_id, "title": self.title,
                "kind": self.kind, "spec": self.spec}


def _bar(title: str, rows: list[dict], x: str, y: str, y_title: str, sort=None, color=None,
         average: float | None = None) -> dict:
    encoding = {
        "x": {"field": x, "type": "nominal", "sort": sort, "axis": {"labelAngle": -30}, "title": None},
        "y": {"field": y, "type": "quantitative", "title": y_title,
              **({"axis": {"format": ".0%"}} if y == "rate" else {})},
        "tooltip": [{"field": x, "type": "nominal"},
                    {"field": y, "type": "quantitative", **({"format": ".1%"} if y == "rate" else {})}],
    }
    if color:
        encoding["color"] = color
    spec = {"$schema": VEGA_LITE, "title": title, "data": {"values": rows}}
    if average is None:
        return {**spec, "mark": "bar", "encoding": encoding}
    rule = {"mark": {"type": "rule", "strokeDash": [4, 4]},
            "encoding": {"y": {"datum": average}, "tooltip": {"value": f"overall {100 * average:.1f}%"}}}
    return {**spec, "layer": [{"mark": "bar", "encoding": encoding}, rule]}


def build_charts(facts: Facts, pack: FintechReport, max_charts: int = 8) -> list[ChartSpec]:
    target = pack.findings.target
    overall = pack.findings.target_rate
    strength = {v.column: v for v in pack.variable_strength}
    charts: list[ChartSpec] = []

    def add(finding_id, title, kind, spec):
        charts.append(ChartSpec(f"C{len(charts) + 1}", finding_id, title, kind, spec))

    for fid, f in facts.findings.items():
        if len(charts) >= max_charts:
            break
        if f.kind == "correlation":
            continue
        woe = strength.get(f.column).woe if f.column in strength else None
        if woe and len(woe) > 2:
            rows = [{"bin": w["bin"], "rate": w["bad_rate"], "count": w["count"]} for w in woe]
            title = f"{target} rate by {f.column}"
            add(fid, title, "woe_profile",
                _bar(title, rows, "bin", "rate", f"{target} rate", sort=None, average=overall))
        elif "rates" in f.evidence:
            rows = [{"group": k, "rate": v, "count": f.evidence.get("counts", {}).get(k)}
                    for k, v in sorted(f.evidence["rates"].items(), key=lambda kv: -kv[1])]
            title = f"{target} rate by {f.column}"
            add(fid, title, "rate_by_group",
                _bar(title, rows, "group", "rate", f"{target} rate", sort="-y", average=overall))

    if pack.variable_strength:
        top = pack.variable_strength[:12]
        rows = [{"variable": v.column, "iv": round(v.iv, 4), "band": v.band} for v in top]
        title = "Predictive strength (Information Value)"
        add(None, title, "variable_strength",
            _bar(title, rows, "variable", "iv", "Information Value", sort="-y",
                 color={"field": "band", "type": "nominal"}))
    return charts
