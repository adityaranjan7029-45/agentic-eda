"""Findings engine: statistics find and verify the patterns; the LLM only explains them.

v1 handed an LLM a statistical summary and asked it to "find insights". It
reported things that weren't true (two one-hot columns of the same variable
called a "data-encoding conflict") and the Critic approved them. Here every
finding is a hypothesis test with a measured effect, and only findings that
survive the following gates are reported:

  1. Significance with false-discovery control. Testing 120 columns means some
     pure-noise columns look "significant" by luck; Benjamini-Hochberg keeps
     the expected share of false findings below `fdr` (5%).
  2. A minimum effect size (Gini >= 0.10), so tiny-but-significant effects on
     large data aren't reported as insights.
  3. Redundancy: a driver that's mostly a restatement of a stronger driver
     (|Spearman| >= 0.7, e.g. interest_rate priced from credit_score) is kept
     as *secondary*, linked to the stronger one, not presented as a new cause.
  4. Same-source pairs (x vs x__missing, a date vs its month) are never
     compared with each other.

Effect size is reported as Gini (= 2 x |AUC - 0.5|), the standard measure of
a single feature's discriminatory power in credit scoring, so numeric, binary
and categorical findings sit on one scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from eda_engine.preprocess import PreprocessResult
from eda_engine.profile import fmt_number
from eda_engine.schema import ColumnRole
from eda_engine.stats import binary_auc

_TESTABLE = (ColumnRole.NUMERIC, ColumnRole.BINARY, ColumnRole.CATEGORICAL)


@dataclass(frozen=True)
class FindingsConfig:
    fdr: float = 0.05                 # Benjamini-Hochberg false discovery rate
    # Smallest effect worth reporting. In credit scoring a single variable with
    # Gini below ~10% is generally too weak to act on. On the synthetic
    # benchmark (10 seeds) this keeps all 10 planted drivers and drops every
    # noise column; at 0.05, roughly 1 noise column in 3 datasets slipped through.
    min_gini: float = 0.10
    redundancy_threshold: float = 0.7  # |Spearman| at which a weaker driver is "secondary"
    duplicate_threshold: float = 0.98  # |Spearman| at which two columns are near-duplicates
    min_class_size: int = 30           # each target class needs this many non-missing rows
    category_smoothing: float = 20.0
    max_correlations: int = 10         # no-target mode: how many column pairs to report
    # Findings with q above this are labelled "tentative". On the benchmark
    # (step 8) every noise column that passed the FDR gate had q >= 0.008, while
    # every planted driver at 5,000+ rows had q < 0.0001.
    high_confidence_q: float = 0.001


@dataclass(frozen=True)
class Finding:
    kind: str                     # "driver" | "secondary_driver" | "correlation"
    column: str
    direction: str                # "+" (higher -> more positive outcome), "-", or "mixed" (categorical)
    gini: float
    p_value: float
    q_value: float                # Benjamini-Hochberg adjusted p-value
    n: int
    summary: str                  # one plain-English line with the real numbers
    evidence: dict = field(default_factory=dict)
    related_to: str | None = None  # secondary drivers: the stronger driver they restate
    correlation: float | None = None
    confidence: str = "high"       # "high" | "tentative" (significant, but close to the noise level)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "column": self.column, "direction": self.direction,
            "gini": round(self.gini, 4), "p_value": self.p_value, "q_value": self.q_value,
            "n": self.n, "summary": self.summary, "evidence": self.evidence,
            "related_to": self.related_to,
            "correlation": None if self.correlation is None else round(self.correlation, 3),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FindingsReport:
    target: str | None
    target_rate: float | None
    n_rows: int
    n_tested: int
    findings: list[Finding]
    redundant_pairs: list[dict]   # near-duplicate columns: data structure, not insight
    quality_notes: list[dict]     # from preprocessing: exclusions, duplicates, sparse columns
    not_reported: dict[str, int]  # why tested columns didn't become findings

    def drivers(self, include_secondary: bool = True) -> list[Finding]:
        kinds = ("driver", "secondary_driver") if include_secondary else ("driver",)
        return [f for f in self.findings if f.kind in kinds]

    def to_dict(self) -> dict:
        return {
            "target": self.target, "target_rate": self.target_rate, "n_rows": self.n_rows,
            "n_tested": self.n_tested, "findings": [f.to_dict() for f in self.findings],
            "redundant_pairs": self.redundant_pairs, "quality_notes": self.quality_notes,
            "not_reported": self.not_reported,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(column: str) -> str:
    """'income__missing' and 'income' share a source; so do 'date__month' and 'date'."""
    return column.split("__", 1)[0]


def _confidence(q: float, cfg: FindingsConfig) -> str:
    return "high" if q <= cfg.high_confidence_q else "tentative"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjusted p-values (q-values) controlling the false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(adjusted, 0, 1)
    return q.tolist()


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Per-column tests against a binary target
# ---------------------------------------------------------------------------

@dataclass
class _Test:
    column: str
    direction: str
    gini: float
    p_value: float
    n: int
    summary: str
    evidence: dict


def _test_numeric(col: str, x: pd.Series, y: pd.Series, target: str) -> _Test | None:
    auc = binary_auc(x, y)
    if np.isnan(auc):
        return None
    pos, neg = x[y == 1], x[y == 0]
    p = sps.mannwhitneyu(pos, neg, alternative="two-sided", method="asymptotic").pvalue
    # Business-readable evidence: outcome rate in the top vs bottom fifth.
    q20, q80 = (float(v) for v in x.quantile([0.2, 0.8]))
    # Round once; the summary is written from the same numbers stored as
    # evidence, so a checker can verify every figure in the text.
    low_rate, high_rate = round(float(y[x <= q20].mean()), 4), round(float(y[x >= q80].mean()), 4)
    direction = "+" if auc >= 0.5 else "-"
    summary = (
        f"{target} rate is {_pct(high_rate)} in the top 20% of {col} (>= {fmt_number(q80)}) "
        f"vs {_pct(low_rate)} in the bottom 20% (<= {fmt_number(q20)})"
    )
    evidence = {"auc": round(auc, 4), "top_quintile_rate": high_rate, "bottom_quintile_rate": low_rate,
                "top_quintile_cutoff": q80, "bottom_quintile_cutoff": q20}
    return _Test(col, direction, 2 * abs(auc - 0.5), float(p), len(x), summary, evidence)


def _test_categorical(col: str, x: pd.Series, y: pd.Series, target: str, smoothing: float) -> _Test | None:
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    p = sps.chi2_contingency(table, correction=False).pvalue
    counts = table.sum(axis=1)
    rates = (table[1] / counts).astype(float).round(4)  # same rounding in summary and evidence
    prior = round(float(y.mean()), 4)
    smoothed = (table[1] + prior * smoothing) / (counts + smoothing)
    auc = binary_auc(x.map(smoothed).astype(float), y)
    hi, lo = rates.idxmax(), rates.idxmin()
    is_binary = table.shape[0] == 2
    if is_binary:
        other = [v for v in rates.index if v != hi][0]
        summary = f"{target} rate is {_pct(rates[hi])} when {col} = {hi} vs {_pct(rates[other])} when {col} = {other}"
        direction = "+" if str(hi) == str(max(rates.index, key=str)) else "-"
    else:
        summary = (f"{target} rate ranges from {_pct(rates[hi])} ({col} = {hi}, n={counts[hi]}) "
                   f"to {_pct(rates[lo])} ({col} = {lo}, n={counts[lo]}); overall {_pct(prior)}")
        direction = "mixed"
    evidence = {"auc": round(auc, 4),
                "rates": {str(k): float(v) for k, v in rates.sort_values(ascending=False).items()},
                "overall_rate": prior,
                "counts": {str(k): int(v) for k, v in counts.items()}}
    return _Test(col, direction, 2 * abs(auc - 0.5), float(p), len(x), summary, evidence)


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------

def _spearman(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.rank().corr()


def _redundant_pairs(df: pd.DataFrame, cols: list[str], threshold: float) -> list[dict]:
    if len(cols) < 2:
        return []
    corr = _spearman(df, cols)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold and _source(a) != _source(b):
                pairs.append({"columns": [a, b], "spearman": round(float(r), 3),
                              "note": f"{a} and {b} carry nearly the same information"})
    return pairs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def quality_notes(prep: PreprocessResult) -> list[dict]:
    notes = []
    for col, reason in prep.excluded.items():
        severity = "critical" if "leakage" in reason else "info"
        notes.append({"severity": severity, "column": col, "message": f"excluded: {reason}"})
    for step in prep.steps:
        if step.name == "deduplicate" and not step.detail.startswith("removed 0"):
            notes.append({"severity": "warning", "column": None, "message": step.detail})
    for info in prep.schema.columns:
        if info.missing_fraction >= 0.6 and not info.name.endswith("__missing"):
            notes.append({"severity": "warning", "column": info.name,
                          "message": f"{info.missing_fraction:.0%} missing"})
    if prep.rejected_plan_steps:
        notes.append({"severity": "warning", "column": None,
                      "message": f"rejected unknown plan steps: {prep.rejected_plan_steps}"})
    return notes


def discover_findings(prep: PreprocessResult, config: FindingsConfig | None = None) -> FindingsReport:
    """Test every usable column and return the findings that survive all gates."""
    cfg = config or FindingsConfig()
    df, schema, target = prep.data, prep.schema, prep.schema.target
    candidates = [c.name for c in schema.columns if c.role in _TESTABLE and c.name != target]
    numeric_like = [c.name for c in schema.columns
                    if c.role in (ColumnRole.NUMERIC, ColumnRole.BINARY) and c.name != target]
    redundant = _redundant_pairs(df, numeric_like, cfg.duplicate_threshold)
    notes = quality_notes(prep)

    binary_target = target is not None and set(df[target].dropna().unique()) <= {0, 1}
    if not binary_target:
        return _correlation_report(df, schema, target, candidates, numeric_like, redundant, notes, cfg)

    y_all = df[target]
    tests: list[_Test] = []
    skipped = 0
    for col in candidates:
        mask = df[col].notna() & y_all.notna()
        x, y = df.loc[mask, col], y_all[mask].astype(int)
        if min((y == 1).sum(), (y == 0).sum()) < cfg.min_class_size:
            skipped += 1
            continue
        role = schema[col].role
        if role == ColumnRole.NUMERIC:
            result = _test_numeric(col, x.astype(float), y, target)
        else:
            result = _test_categorical(col, x.astype(str) if role == ColumnRole.CATEGORICAL else x, y,
                                       target, cfg.category_smoothing)
        if result is None:
            skipped += 1
        else:
            tests.append(result)

    q_values = benjamini_hochberg([t.p_value for t in tests])
    significant = [(t, q) for t, q in zip(tests, q_values) if q <= cfg.fdr and t.gini >= cfg.min_gini]
    not_significant = sum(q > cfg.fdr for q in q_values)
    too_small = sum(q <= cfg.fdr and t.gini < cfg.min_gini for t, q in zip(tests, q_values))
    significant.sort(key=lambda tq: (-tq[0].gini, tq[0].column))

    # Redundancy among significant numeric/binary drivers, strongest first.
    driver_cols = [t.column for t, _ in significant if t.column in numeric_like]
    corr = _spearman(df, driver_cols) if len(driver_cols) > 1 else None
    findings: list[Finding] = []
    primary: list[str] = []
    for t, q in significant:
        related, r_value = None, None
        if corr is not None and t.column in corr.index:
            for p in primary:
                if p in corr.columns and _source(p) != _source(t.column):
                    r = corr.loc[t.column, p]
                    if pd.notna(r) and abs(r) >= cfg.redundancy_threshold:
                        related, r_value = p, float(r)
                        break
        kind = "secondary_driver" if related else "driver"
        if kind == "driver":
            primary.append(t.column)
        findings.append(Finding(kind, t.column, t.direction, t.gini, t.p_value, q, t.n,
                                t.summary, t.evidence, related, r_value, _confidence(q, cfg)))

    return FindingsReport(
        target=target, target_rate=round(float(y_all.mean()), 4), n_rows=len(df),
        n_tested=len(tests), findings=findings, redundant_pairs=redundant, quality_notes=notes,
        not_reported={"not_significant": not_significant, "effect_too_small": too_small,
                      "untestable": skipped},
    )


def _correlation_report(df, schema, target, candidates, numeric_like, redundant, notes, cfg) -> FindingsReport:
    """No binary target: report the strongest real relationships between columns,
    excluding near-duplicates and same-source pairs (the v1 artifacts)."""
    cols = [c for c in numeric_like if not c.endswith("__missing")]
    tests, pairs = [], []
    if len(cols) > 1:
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric.rank().corr()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.isna(r) or abs(r) >= cfg.duplicate_threshold or _source(a) == _source(b):
                    continue
                n = int((numeric[a].notna() & numeric[b].notna()).sum())
                if n < 10:
                    continue
                t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
                p = 2 * sps.t.sf(abs(t_stat), n - 2)
                tests.append((a, b, float(r), float(p), n))
    q_values = benjamini_hochberg([t[3] for t in tests])
    kept = [(t, q) for t, q in zip(tests, q_values) if q <= cfg.fdr and abs(t[2]) >= 0.3]
    kept.sort(key=lambda tq: -abs(tq[0][2]))
    findings = [
        Finding("correlation", f"{a} ~ {b}", "+" if r > 0 else "-", abs(r), p, q, n,
                f"{a} and {b} move {'together' if r > 0 else 'in opposite directions'} (Spearman {r:.2f})",
                {"spearman": round(r, 4), "columns": [a, b]}, confidence=_confidence(q, cfg))
        for (a, b, r, p, n), q in kept[: cfg.max_correlations]
    ]
    return FindingsReport(
        target=target, target_rate=None, n_rows=len(df), n_tested=len(tests), findings=findings,
        redundant_pairs=redundant, quality_notes=notes,
        not_reported={"not_significant_or_weak": len(tests) - len(kept),
                      "beyond_top_n": max(0, len(kept) - cfg.max_correlations)},
    )
