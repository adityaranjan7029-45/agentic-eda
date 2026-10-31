"""Fintech / credit-risk industry pack.

Adds what a credit-risk team would actually look at, on top of the general
findings engine:

  - Information Value (IV) and Weight of Evidence (WoE): the industry's
    standard way to rate a variable's predictive power, with the usual bands.
    IV > 0.5 is "suspiciously predictive", a second leakage alarm.
  - KS statistic: the maximum separation between the score distributions of
    defaulters and non-defaulters, the other standard scorecard metric.
  - Lender-decision rule: variables the lender sets from its own risk
    assessment (interest rate, assigned grade, approved limit) describe the
    lender's view of the applicant, not the applicant. They're demoted, so the
    applicant variable behind them (e.g. credit score) is reported as the driver.
    Statistics alone can't tell which of two overlapping variables is the cause;
    this is domain knowledge doing that job.
  - Fair lending: detects protected attributes (age, gender, marital status...)
    and flags features that act as proxies for them. Credit decisions must not
    rely on protected attributes, directly or through proxies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats as sps

from eda_engine.findings import Finding, FindingsReport
from eda_engine.preprocess import PreprocessResult
from eda_engine.schema import ColumnRole

IV_BANDS = ((0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong"), (float("inf"), "suspicious"))

# Lender decisions: set by the lender after assessing risk.
LENDER_DECISION_PATTERNS = (
    r"interest_?rate", r"\bapr\b", r"(^|_)grade($|_)", r"sub_?grade", r"pricing",
    r"approved", r"approval", r"offer(ed)?_", r"assigned", r"decision",
)

# Attributes that fair-lending rules (e.g. US ECOA, EU anti-discrimination law)
# protect from being used in credit decisions.
PROTECTED_PATTERNS = {
    "age": r"(^|_)(age|date_of_birth|dob|birth_?year)($|_)",
    "gender": r"(^|_)(gender|sex)($|_)",
    "marital_status": r"(^|_)marital",
    "race_ethnicity": r"(^|_)(race|ethnicity|ethnic)",
    "religion": r"(^|_)religion",
    "national_origin": r"(^|_)(nationality|national_origin|citizenship)",
    "disability": r"(^|_)disab",
}


@dataclass(frozen=True)
class FintechConfig:
    iv_bins: int = 10
    woe_smoothing: float = 0.5
    proxy_threshold: float = 0.2   # association strength (Cramer's V / eta / |Spearman|) that flags a proxy


@dataclass(frozen=True)
class VariableStrength:
    column: str
    iv: float
    band: str
    ks: float | None
    woe: list[dict]   # per bin: {"bin", "count", "bad_rate", "woe"}

    def to_dict(self) -> dict:
        return {"column": self.column, "iv": round(self.iv, 4), "band": self.band,
                "ks": None if self.ks is None else round(self.ks, 4), "woe": self.woe}


@dataclass(frozen=True)
class FintechReport:
    findings: FindingsReport                 # findings after domain rules
    variable_strength: list[VariableStrength]
    lender_decision_columns: list[str]
    protected_attributes: dict[str, str]     # column -> protected category
    proxies: list[dict]                      # {"feature", "protected", "strength", "measure"}
    notes: list[dict]

    def to_dict(self) -> dict:
        return {
            "findings": self.findings.to_dict(),
            "variable_strength": [v.to_dict() for v in self.variable_strength],
            "lender_decision_columns": self.lender_decision_columns,
            "protected_attributes": self.protected_attributes,
            "proxies": self.proxies,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Information Value, WoE, KS
# ---------------------------------------------------------------------------

def iv_band(iv: float) -> str:
    return next(label for limit, label in IV_BANDS if iv < limit)


def _bins(x: pd.Series, role: ColumnRole, n_bins: int) -> pd.Series:
    """Quantile bins for numbers, categories as they are; missing is its own bin."""
    if role == ColumnRole.NUMERIC and x.nunique() > n_bins:
        binned = pd.qcut(x, q=n_bins, duplicates="drop").astype(str)
    else:
        binned = x.astype(str)
    return binned.where(x.notna(), "missing")


def information_value(x: pd.Series, y: pd.Series, role: ColumnRole, cfg: FintechConfig) -> tuple[float, list[dict]]:
    """IV = sum over bins of (share of goods - share of bads) x WoE,
    WoE = ln(share of goods / share of bads). 'Bad' = target 1 (e.g. default)."""
    codes, labels = pd.factorize(_bins(x, role, cfg.iv_bins), sort=True)
    yv = np.asarray(y, dtype=int)
    total = np.bincount(codes, minlength=len(labels))
    bad_count = np.bincount(codes, weights=yv, minlength=len(labels))
    good_count = total - bad_count
    goods, bads = good_count + cfg.woe_smoothing, bad_count + cfg.woe_smoothing
    good_share, bad_share = goods / goods.sum(), bads / bads.sum()
    woe = np.log(good_share / bad_share)
    iv = float(((good_share - bad_share) * woe).sum())
    detail = [
        {"bin": str(label), "count": int(total[i]),
         "bad_rate": round(float(bad_count[i] / max(1, total[i])), 4), "woe": round(float(woe[i]), 4)}
        for i, label in enumerate(labels)
    ]
    return iv, detail


def ks_statistic(x: pd.Series, y: pd.Series) -> float | None:
    mask = x.notna()
    pos, neg = x[mask & (y == 1)], x[mask & (y == 0)]
    if len(pos) < 2 or len(neg) < 2:
        return None
    return float(sps.ks_2samp(pos, neg).statistic)


# ---------------------------------------------------------------------------
# Domain rules
# ---------------------------------------------------------------------------

def _matches(column: str, patterns) -> bool:
    name = column.lower()
    return any(re.search(p, name) for p in patterns)


def lender_decision_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if _matches(c, LENDER_DECISION_PATTERNS)]


def protected_attributes(columns: list[str]) -> dict[str, str]:
    found = {}
    for col in columns:
        for category, pattern in PROTECTED_PATTERNS.items():
            if re.search(pattern, col.lower()):
                found[col] = category
                break
    return found


def _demote_lender_decisions(report: FindingsReport, decisions: list[str]) -> FindingsReport:
    """Re-label lender decisions and promote the applicant drivers they shadowed."""
    decision_set = set(decisions)
    adjusted: list[Finding] = []
    for f in report.findings:
        if f.column in decision_set:
            adjusted.append(replace(
                f, kind="lender_decision", related_to=None, correlation=None,
                summary=f.summary + " (set by the lender from its own risk assessment; "
                                    "describes the lender's view, not the applicant)",
            ))
        elif f.kind == "secondary_driver" and f.related_to in decision_set:
            adjusted.append(replace(f, kind="driver", related_to=None, correlation=None))
        else:
            adjusted.append(f)
    return replace(report, findings=adjusted)


# ---------------------------------------------------------------------------
# Fair lending: proxies for protected attributes
# ---------------------------------------------------------------------------

def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = sps.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: how much of a number's variation is explained by a category (0..1)."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(x) & categories.notna().to_numpy()
    codes, labels = pd.factorize(categories[mask])
    if len(labels) < 2:
        return 0.0
    x = x[mask]
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=x) / counts
    between = float((counts * (means - x.mean()) ** 2).sum())
    total = float(((x - x.mean()) ** 2).sum())
    return float(np.sqrt(between / total)) if total > 0 else 0.0


def _association(df: pd.DataFrame, a: str, a_numeric: bool, b: str, b_numeric: bool) -> tuple[float, str]:
    if a_numeric and b_numeric:
        return abs(float(df[[a, b]].rank().corr().iloc[0, 1])), "spearman"
    if a_numeric:
        return _correlation_ratio(df[b].astype(str), df[a]), "eta"
    if b_numeric:
        return _correlation_ratio(df[a].astype(str), df[b]), "eta"
    return _cramers_v(df[a].astype(str), df[b].astype(str)), "cramers_v"


def find_proxies(prep: PreprocessResult, protected: dict[str, str], cfg: FintechConfig) -> list[dict]:
    schema, df = prep.schema, prep.data
    usable = {c.name: c.role == ColumnRole.NUMERIC for c in schema.columns
              if c.role in (ColumnRole.NUMERIC, ColumnRole.BINARY, ColumnRole.CATEGORICAL)
              and c.name != schema.target}
    proxies = []
    for p_col, category in protected.items():
        if p_col not in usable:
            continue
        for feature, f_numeric in usable.items():
            if feature in protected or feature.split("__")[0] == p_col:
                continue
            strength, measure = _association(df, feature, f_numeric, p_col, usable[p_col])
            if strength >= cfg.proxy_threshold:
                proxies.append({"feature": feature, "protected": p_col, "category": category,
                                "strength": round(strength, 3), "measure": measure})
    return sorted(proxies, key=lambda d: -d["strength"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fintech_pack(prep: PreprocessResult, findings: FindingsReport,
                     config: FintechConfig | None = None) -> FintechReport:
    cfg = config or FintechConfig()
    schema, df, target = prep.schema, prep.data, prep.schema.target
    columns = [c.name for c in schema.columns if c.name != target]
    decisions = lender_decision_columns(columns)
    protected = protected_attributes(columns)
    notes: list[dict] = []

    strengths: list[VariableStrength] = []
    binary_target = target is not None and set(df[target].dropna().unique()) <= {0, 1}
    if binary_target:
        y = df[target].astype(int)
        for info in schema.columns:
            if info.name == target or info.role not in (ColumnRole.NUMERIC, ColumnRole.BINARY, ColumnRole.CATEGORICAL):
                continue
            iv, woe = information_value(df[info.name], y, info.role, cfg)
            ks = ks_statistic(df[info.name].astype(float), y) if info.role == ColumnRole.NUMERIC else None
            strengths.append(VariableStrength(info.name, iv, iv_band(iv), ks, woe))
        strengths.sort(key=lambda v: (-v.iv, v.column))
        # A warning, not critical: very strong applicant variables (credit scores)
        # can legitimately exceed 0.5. Confirmed leakage was already excluded upstream.
        for v in strengths:
            if v.band == "suspicious":
                notes.append({"severity": "warning", "column": v.column,
                              "message": f"IV {v.iv:.2f} is unusually high; confirm it is known at application time"})
    else:
        notes.append({"severity": "info", "column": None,
                      "message": "IV/WoE need a yes/no target; skipped"})

    adjusted = _demote_lender_decisions(findings, decisions)
    reported = {f.column for f in adjusted.findings}
    for col in decisions:
        notes.append({"severity": "info", "column": col,
                      "message": "lender decision variable: reflects the lender's risk assessment, not the applicant"})
    for col, category in protected.items():
        severity = "warning" if col in reported else "info"
        notes.append({"severity": severity, "column": col,
                      "message": f"protected attribute ({category}): must not be used in credit decisions"
                                 + ("; it is associated with the target" if col in reported else "")})

    proxies = find_proxies(prep, protected, cfg)
    for p in proxies:
        notes.append({"severity": "warning", "column": p["feature"],
                      "message": f"possible proxy for {p['protected']} ({p['measure']} {p['strength']:.2f}); "
                                 "using it could indirectly discriminate"})

    return FintechReport(adjusted, strengths, decisions, protected, proxies, notes)
