"""Chart specs built from evidence: data, not code.

v1 asked the LLM to write matplotlib code and executed it in a sandbox, which
is slow, fragile and a security risk. v2 emits Vega-Lite specs (a JSON chart
grammar the frontend renders) straight from the verified evidence. The LLM
never touches chart data, so a chart can't disagree with the numbers in the
report, and charts cost zero tokens.

Chart kinds:
  rate_trend         line: outcome rate across ordered bands of a numeric variable
  rate_by_group      bars: outcome rate per category
  ks_curve           lines: cumulative share of each outcome class (gap = KS statistic)
  distribution       overlaid histograms: the variable's distribution for each outcome class
  correlation_matrix heatmap: Spearman correlations between the key numeric columns
  variable_strength  bars: Information Value ranking
Every number in them is an aggregate (rates, shares, correlations), never a raw row.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from eda_engine.agents.facts import Facts
from eda_engine.packs.fintech import FintechReport

VEGA_LITE = "https://vega.github.io/schema/vega-lite/v5.json"
DETAILED_DRIVERS = 2          # top numeric drivers that also get a KS curve and a distribution chart
MAX_HEATMAP_COLUMNS = 10
MAX_FINDING_COLUMNS = 7     # the rest is reserved for near-duplicate pairs


@dataclass(frozen=True)
class ChartSpec:
    id: str                  # "C1"
    finding_id: str | None   # "F2", or None for dataset-level charts
    title: str
    kind: str
    spec: dict               # Vega-Lite

    def to_dict(self) -> dict:
        return {"id": self.id, "finding_id": self.finding_id, "title": self.title,
                "kind": self.kind, "spec": self.spec}


def _base(rows: list[dict]) -> dict:
    return {"$schema": VEGA_LITE, "data": {"values": rows}}


def _average_rule(average: float) -> dict:
    return {"mark": {"type": "rule", "strokeDash": [4, 4], "opacity": 0.7},
            "encoding": {"y": {"datum": average}, "tooltip": {"value": f"overall {100 * average:.1f}%"}}}


def _rate_by_group(rows: list[dict], target: str, average: float | None) -> dict:
    encoding = {
        "x": {"field": "group", "type": "nominal", "sort": "-y", "axis": {"labelAngle": 0}, "title": None},
        "y": {"field": "rate", "type": "quantitative", "title": f"{target} rate", "axis": {"format": ".0%"}},
        "tooltip": [{"field": "group", "type": "nominal"}, {"field": "rate", "type": "quantitative", "format": ".1%"},
                    {"field": "count", "type": "quantitative", "format": ","}],
    }
    layers = [{"mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4}, "encoding": encoding}]
    if average is not None:
        layers.append(_average_rule(average))
    return {**_base(rows), "layer": layers}


def _rate_trend(rows: list[dict], target: str, average: float | None) -> dict:
    """Outcome rate across ordered bands: a line shows the trend; 'missing' is a separate point."""
    order = [r["bin"] for r in rows]
    x = {"field": "bin", "type": "ordinal", "sort": order, "axis": {"labelAngle": -35}, "title": None}
    y = {"field": "rate", "type": "quantitative", "title": f"{target} rate", "axis": {"format": ".0%"}}
    tooltip = [{"field": "bin", "type": "nominal", "title": "band"},
               {"field": "rate", "type": "quantitative", "format": ".1%"},
               {"field": "count", "type": "quantitative", "format": ","}]
    layers = [
        {"transform": [{"filter": "datum.bin !== 'missing'"}],
         "mark": {"type": "line", "strokeWidth": 2.5, "interpolate": "monotone"}, "encoding": {"x": x, "y": y}},
        {"mark": {"type": "point", "filled": True, "size": 60}, "encoding": {"x": x, "y": y, "tooltip": tooltip}},
    ]
    if average is not None:
        layers.append(_average_rule(average))
    return {**_base(rows), "layer": layers}


def _ks(curve: list[dict], ks: float, variable: str, target: str) -> dict:
    rows = []
    for point in curve:
        rows.append({"value": point["value"], "group": f"{target} = 1", "share": point["bad"]})
        rows.append({"value": point["value"], "group": f"{target} = 0", "share": point["good"]})
    widest = max(curve, key=lambda p: abs(p["bad"] - p["good"]))
    gap = {"value": widest["value"], "low": min(widest["bad"], widest["good"]),
           "high": max(widest["bad"], widest["good"]), "label": f"KS {ks:.2f}"}
    x = {"field": "value", "type": "quantitative", "title": variable, "scale": {"zero": False}}
    return {
        "$schema": VEGA_LITE,
        "layer": [
            {"data": {"values": rows}, "mark": {"type": "line", "strokeWidth": 2.5, "interpolate": "step-after"},
             "encoding": {"x": x, "y": {"field": "share", "type": "quantitative", "title": "cumulative share",
                                        "axis": {"format": ".0%"}},
                          "color": {"field": "group", "type": "nominal", "title": None, "legend": {"orient": "bottom"}},
                          "tooltip": [{"field": "group"}, {"field": "value", "format": ",.4~g"},
                                      {"field": "share", "format": ".0%"}]}},
            {"data": {"values": [gap]}, "mark": {"type": "rule", "strokeWidth": 2, "strokeDash": [3, 3]},
             "encoding": {"x": {"field": "value", "type": "quantitative"}, "y": {"field": "low", "type": "quantitative"},
                          "y2": {"field": "high"}}},
            {"data": {"values": [gap]}, "mark": {"type": "text", "dx": 6, "align": "left", "fontWeight": "bold"},
             "encoding": {"x": {"field": "value", "type": "quantitative"}, "y": {"field": "high", "type": "quantitative"},
                          "text": {"field": "label"}}},
        ],
    }


def _distribution(bins: list[dict], variable: str, target: str) -> dict:
    rows = []
    for b in bins:
        rows.append({"start": b["start"], "end": b["end"], "group": f"{target} = 1", "share": b["bad"]})
        rows.append({"start": b["start"], "end": b["end"], "group": f"{target} = 0", "share": b["good"]})
    return {
        **_base(rows),
        "mark": {"type": "bar", "opacity": 0.55, "binSpacing": 0},
        "encoding": {
            "x": {"field": "start", "type": "quantitative", "bin": {"binned": True}, "title": variable},
            "x2": {"field": "end"},
            "y": {"field": "share", "type": "quantitative", "stack": None, "title": "share of group",
                  "axis": {"format": ".0%"}},
            "color": {"field": "group", "type": "nominal", "title": None, "legend": {"orient": "bottom"}},
            "tooltip": [{"field": "group"}, {"field": "start", "format": ",.4~g", "title": "from"},
                        {"field": "end", "format": ",.4~g", "title": "to"}, {"field": "share", "format": ".1%"}],
        },
    }


def _heatmap(columns: list[str], matrix: pd.DataFrame) -> dict:
    rows = [{"a": a, "b": b, "r": round(float(matrix.loc[a, b]), 2)} for a in columns for b in columns]
    x = {"field": "a", "type": "nominal", "sort": columns, "title": None, "axis": {"labelAngle": -40}}
    y = {"field": "b", "type": "nominal", "sort": columns, "title": None}
    return {
        **_base(rows),
        "layer": [
            {"mark": "rect", "encoding": {
                "x": x, "y": y,
                "color": {"field": "r", "type": "quantitative", "title": "Spearman",
                          "scale": {"scheme": "redblue", "domain": [-1, 1], "reverse": True}},
                "tooltip": [{"field": "a", "title": "column"}, {"field": "b", "title": "with"},
                            {"field": "r", "title": "Spearman", "format": ".2f"}]}},
            {"mark": {"type": "text", "fontSize": 10}, "encoding": {
                "x": x, "y": y, "text": {"field": "r", "format": ".2f"},
                "color": {"condition": {"test": "abs(datum.r) > 0.6", "value": "white"}, "value": "black"}}},
        ],
    }


def _strength(rows: list[dict]) -> dict:
    return {
        **_base(rows),
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "y": {"field": "variable", "type": "nominal", "sort": "-x", "title": None},
            "x": {"field": "iv", "type": "quantitative", "title": "Information Value"},
            "color": {"field": "band", "type": "nominal", "title": "band",
                      "legend": {"labelLimit": 240,
                                 "labelExpr": "datum.label == 'suspicious' ? 'very strong (check leakage)' : datum.label"},
                      "scale": {"domain": ["useless", "weak", "medium", "strong", "suspicious"],
                                "range": ["#cbd5e1", "#93c5fd", "#3b82f6", "#1d4ed8", "#e11d48"]}},
            "tooltip": [{"field": "variable"}, {"field": "iv", "format": ".3f"}, {"field": "band"}],
        },
    }


def _heatmap_columns(facts: Facts, pack: FintechReport, data: pd.DataFrame) -> list[str]:
    """Key numeric columns: the strongest findings, plus near-duplicate pairs (reserved slots,
    so redundancy is always visible even when there are many findings)."""
    numeric = {c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 2}
    numeric.discard(pack.findings.target)
    chosen: list[str] = []

    def take(names, limit):
        for c in names:
            if c in numeric and c not in chosen and len(chosen) < limit:
                chosen.append(c)

    for f in facts.findings.values():
        take(f.evidence.get("columns", [f.column]) if f.kind == "correlation" else [f.column], MAX_FINDING_COLUMNS)
    for pair in pack.findings.redundant_pairs:
        if all(c in numeric for c in pair["columns"]) and len(chosen) + 2 - sum(c in chosen for c in pair["columns"]) <= MAX_HEATMAP_COLUMNS:
            take(pair["columns"], MAX_HEATMAP_COLUMNS)
    return chosen


def build_charts(facts: Facts, pack: FintechReport, data: pd.DataFrame | None = None,
                 max_charts: int = 14) -> list[ChartSpec]:
    target = pack.findings.target or "outcome"
    overall = pack.findings.target_rate
    strength = {v.column: v for v in pack.variable_strength}
    charts: list[ChartSpec] = []

    def add(finding_id, title, kind, spec):
        if len(charts) < max_charts:
            charts.append(ChartSpec(f"C{len(charts) + 1}", finding_id, title, kind, spec))

    # 1. One rate chart per finding: a trend line for ordered numeric bands, bars for categories.
    for fid, f in list(facts.findings.items())[:6]:
        if f.kind == "correlation":
            continue
        woe = strength[f.column].woe if f.column in strength else None
        if woe and len(woe) > 2 and strength[f.column].ks is not None:
            rows = [{"bin": w["bin"], "rate": w["bad_rate"], "count": w["count"]} for w in woe]
            add(fid, f"{target} rate across {f.column}", "rate_trend", _rate_trend(rows, target, overall))
        elif "rates" in f.evidence:
            rows = [{"group": k, "rate": v, "count": f.evidence.get("counts", {}).get(k)}
                    for k, v in f.evidence["rates"].items()]
            add(fid, f"{target} rate by {f.column}", "rate_by_group", _rate_by_group(rows, target, overall))
        elif woe:
            rows = [{"group": w["bin"], "rate": w["bad_rate"], "count": w["count"]} for w in woe]
            add(fid, f"{target} rate by {f.column}", "rate_by_group", _rate_by_group(rows, target, overall))

    # 2. For the strongest numeric drivers: how the two outcome groups differ (KS curve, distributions).
    detailed = 0
    for fid, f in facts.findings.items():
        v = strength.get(f.column)
        if detailed >= DETAILED_DRIVERS or f.kind != "driver" or v is None or not v.ks_curve:
            continue
        add(fid, f"KS curve: {f.column}", "ks_curve", _ks(v.ks_curve, v.ks, f.column, target))
        if v.distribution:
            add(fid, f"{f.column} by outcome", "distribution", _distribution(v.distribution, f.column, target))
        detailed += 1

    # 3. How the key numeric columns relate to each other (also shows near-duplicates).
    if data is not None:
        columns = _heatmap_columns(facts, pack, data)
        if len(columns) >= 3:
            matrix = data[columns].rank().corr()
            add(None, "How the key columns relate (Spearman)", "correlation_matrix", _heatmap(columns, matrix))

    # 4. Predictive strength ranking.
    if pack.variable_strength:
        rows = [{"variable": v.column, "iv": round(v.iv, 4), "band": v.band} for v in pack.variable_strength[:12]]
        add(None, "Predictive strength (Information Value)", "variable_strength", _strength(rows))
    return charts
