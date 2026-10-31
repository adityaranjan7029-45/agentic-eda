"""The facts sheet: everything the narrator is allowed to say, in compact text.

The narrator LLM sees only this sheet, never the raw data. Each finding gets
an ID (F1, F2, ...) to cite, and every number is written exactly as the
narrator may repeat it. The Critic then checks the narrative against the
numbers in this same text: if a number isn't on the sheet (allowing for
rounding), it was invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eda_engine.findings import Finding
from eda_engine.packs.fintech import FintechReport
from eda_engine.preprocess import PreprocessResult
from eda_engine.tokens import estimate_tokens

SMALL_DATA_ROWS = 2500     # below this, the step 8 benchmark shows weak effects get missed


@dataclass(frozen=True)
class Facts:
    text: str
    findings: dict[str, Finding]             # "F1" -> finding
    tentative: set[str]                       # finding IDs with confidence "tentative"
    lender_decisions: set[str]                # finding IDs of lender-decision variables
    caveats: list[str] = field(default_factory=list)
    n_drivers: int = 0                        # independent drivers among all findings
    scoped: dict[str, str] = field(default_factory=dict)   # finding ID -> its own lines (finding + IV)
    header: str = ""                          # the DATASET line (overall numbers any sentence may use)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ratio(a: float, b: float) -> str | None:
    if min(a, b) <= 0:
        return None
    return f"{max(a, b) / min(a, b):.1f}x"


def _finding_line(fid: str, f: Finding) -> str:
    ev = f.evidence
    parts = [f"{fid} [{f.kind}, {f.confidence} confidence, gini {f.gini:.2f}] {f.column}:"]
    if "top_quintile_rate" in ev:
        top, bottom = ev["top_quintile_rate"], ev["bottom_quintile_rate"]
        parts.append(f"rate {pct(top)} in top 20% (>= {ev['top_quintile_cutoff']:,.4g}) vs "
                     f"{pct(bottom)} in bottom 20% (<= {ev['bottom_quintile_cutoff']:,.4g})")
        r = ratio(top, bottom)
        if r:
            parts.append(f"({r})")
    elif "rates" in ev:
        rates = sorted(ev["rates"].items(), key=lambda kv: -kv[1])
        parts.append("rate by group: " + ", ".join(f"{k} {pct(v)}" for k, v in rates[:6]))
        r = ratio(rates[0][1], rates[-1][1])
        if r:
            parts.append(f"(highest/lowest {r})")
    elif "spearman" in ev:
        parts.append(f"Spearman correlation {ev['spearman']:.2f}")
    if f.related_to:
        parts.append(f"; restates {f.related_to} (correlation {f.correlation:.2f})")
    if f.kind == "lender_decision":
        parts.append("; set by the lender, not an applicant trait")
    return " ".join(parts)


def build_facts(prep: PreprocessResult, pack: FintechReport, max_findings: int = 15) -> Facts:
    report = pack.findings
    n_drivers = sum(f.kind == "driver" for f in report.findings)
    findings = {f"F{i}": f for i, f in enumerate(report.findings[:max_findings], 1)}
    finding_lines = {fid: _finding_line(fid, f) for fid, f in findings.items()}
    lines = [
        f"DATASET: {report.n_rows:,} rows; target {report.target}; overall rate {pct(report.target_rate)}; "
        f"{report.n_tested} columns tested; {len(report.findings)} findings passed the false-discovery "
        f"and effect-size gates ({n_drivers} independent drivers).",
        "",
        "FINDINGS (cite by ID):",
        *finding_lines.values(),
    ]
    if len(report.findings) > max_findings:
        lines.append(f"(+{len(report.findings) - max_findings} weaker findings not listed)")

    strength = {v.column: v for v in pack.variable_strength}

    def iv_line(c: str) -> str:
        v = strength[c]
        return f"{c}: IV {v.iv:.2f} ({v.band})" + (f", KS {v.ks:.2f}" if v.ks is not None else "")

    iv_lines = [iv_line(c) for c in (f.column for f in findings.values()) if c in strength]
    if iv_lines:
        lines += ["", "INFORMATION VALUE:", *iv_lines[:8]]

    critical = [n for n in report.quality_notes if n["severity"] == "critical"]
    if critical:
        lines += ["", "DATA QUALITY (critical):", *(f"{n['column']}: {n['message']}" for n in critical)]
    if report.redundant_pairs:
        lines += ["", "NEAR-DUPLICATE COLUMNS:",
                  *(f"{a} ~ {b}" for a, b in (p["columns"] for p in report.redundant_pairs[:5]))]
    if pack.proxies or pack.protected_attributes:
        lines += ["", "FAIR LENDING:"]
        if pack.protected_attributes:
            lines.append("protected attributes present: " + ", ".join(sorted(pack.protected_attributes)))
        lines += [f"{p['feature']} may be a proxy for {p['protected']} (association {p['strength']:.2f})"
                  for p in pack.proxies]

    caveats = ["Findings are associations, not proof of cause."]
    if report.n_rows < SMALL_DATA_ROWS:
        caveats.append(f"With only {report.n_rows:,} rows, weak effects may be missed and tentative "
                       f"findings may be chance.")
    lines += ["", "CAVEATS:", *caveats]

    return Facts(
        text="\n".join(lines),
        findings=findings,
        tentative={fid for fid, f in findings.items() if f.confidence == "tentative"},
        lender_decisions={fid for fid, f in findings.items() if f.kind == "lender_decision"},
        caveats=caveats,
        n_drivers=n_drivers,
        scoped={fid: finding_lines[fid] + ("\n" + iv_line(f.column) if f.column in strength else "")
                for fid, f in findings.items()},
        header=lines[0],
    )
