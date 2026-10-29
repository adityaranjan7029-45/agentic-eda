"""Preprocessing as one node: a loop over plain, deterministic step functions.

v1 wired ten preprocessing steps as ten LangGraph nodes and let an LLM pick
any subset, including skipping cleaning. It also ran ML-style transforms
(scaling, PCA, 45 automatic ratio features) *before* analysis, so findings and
charts came out in z-score units nobody can read ("credit score = -0.8") and
the ratio explosion manufactured spurious correlations.

v2 splits steps into two kinds:
  - HYGIENE steps always run when the schema says they're needed (dedupe,
    placeholder nulls, spelling, type parsing, excluding unusable columns).
    No plan can skip them.
  - OPTIONAL steps come from a fixed vocabulary. `recommend_plan` picks them
    deterministically from the schema; a Planner agent may later override the
    choice, but only within the vocabulary (unknown names are rejected).

The output stays in original units, so every finding and chart is readable
by a credit analyst.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eda_engine.schema import (
    ColumnRole,
    DatasetSchema,
    Flag,
    clean_values,
    datetime_values,
    infer_schema,
    numeric_values,
)

POSITIVE_LABELS = frozenset({
    "1", "yes", "y", "true", "t", "default", "defaulted", "churn", "churned",
    "bad", "fraud", "positive", "fail", "failed",
})

# Columns that can't be analysed statistically as they are.
_UNUSABLE_ROLES = {
    ColumnRole.ID: "identifier, not a feature",
    ColumnRole.EMPTY: "no values",
    ColumnRole.CONSTANT: "a single value carries no information",
    ColumnRole.HIGH_CARDINALITY: "too many categories for reliable statistics",
    ColumnRole.FREE_TEXT: "free text is out of scope for tabular analysis",
}


@dataclass
class StepReport:
    name: str
    kind: str        # "hygiene" | "optional"
    detail: str      # what actually changed, in plain words


@dataclass
class PreprocessResult:
    data: pd.DataFrame
    schema: DatasetSchema                       # roles of the *output* columns
    steps: list[StepReport]
    excluded: dict[str, str]                    # column -> reason
    added_columns: list[str]
    target_mapping: dict[str, int] | None = None
    rejected_plan_steps: list[str] = field(default_factory=list)


@dataclass
class _State:
    df: pd.DataFrame
    schema: DatasetSchema
    excluded: dict[str, str] = field(default_factory=dict)
    added: list[str] = field(default_factory=list)
    target_mapping: dict[str, int] | None = None

    def roles(self, *roles: ColumnRole) -> list[str]:
        return [c.name for c in self.schema.columns if c.role in roles and c.name in self.df.columns]


StepFn = Callable[[_State], str]


# ---------------------------------------------------------------------------
# Hygiene steps (always run)
# ---------------------------------------------------------------------------

def _deduplicate(s: _State) -> str:
    before = len(s.df)
    s.df = s.df.drop_duplicates().reset_index(drop=True)
    return f"removed {before - len(s.df)} exact duplicate rows"


def _drop_missing_target(s: _State) -> str:
    target = s.schema.target
    if target is None:
        return "no target selected"
    before = len(s.df)
    s.df = s.df[clean_values(s.df[target]).notna()].reset_index(drop=True)
    return f"removed {before - len(s.df)} rows with a missing target"


def _canonical_spelling(values: pd.Series) -> pd.Series:
    """Merge case variants into the most common original spelling, so
    'Rent' / 'rent' / 'RENT' all become 'rent' if that's the usual form,
    while 'PA-09' and 'Month-to-month' keep their natural casing."""
    key = values.str.lower()
    present = values.notna()
    usual = values[present].groupby(key[present]).agg(lambda v: v.value_counts().idxmax())
    return key.map(usual)


def _clean_text(s: _State) -> str:
    """Placeholder tokens -> NaN, whitespace stripped, and case variants of a
    category merged into one spelling."""
    changed = []
    for col in s.roles(ColumnRole.CATEGORICAL, ColumnRole.BINARY, ColumnRole.TARGET):
        series = s.df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        cleaned = _canonical_spelling(clean_values(series))
        if not cleaned.equals(series.astype("string")):
            changed.append(col)
        s.df[col] = cleaned.astype(object).where(cleaned.notna(), np.nan)
    return f"normalized text in {len(changed)} columns" + (f" ({', '.join(changed[:5])}{'...' if len(changed) > 5 else ''})" if changed else "")


def _parse_types(s: _State) -> str:
    """Numbers and dates stored as text become real numbers and dates."""
    parsed = []
    for info in s.schema.columns:
        if info.name not in s.df.columns or Flag.STORED_AS_TEXT not in info.flags:
            continue
        if info.role == ColumnRole.NUMERIC:
            s.df[info.name] = numeric_values(s.df[info.name], info)
        elif info.role == ColumnRole.DATETIME:
            s.df[info.name] = datetime_values(s.df[info.name])
        parsed.append(info.name)
    return f"parsed {len(parsed)} text columns" + (f" ({', '.join(parsed)})" if parsed else "")


def _encode_target(s: _State) -> str:
    """A two-valued target becomes 0/1, with 1 meaning the outcome of interest
    (a recognised positive word like 'yes'/'default', else the rarer class)."""
    target = s.schema.target
    if target is None or s.schema[target].n_unique != 2:
        return "target is not binary; left as is"
    values = s.df[target]
    labels = sorted(values.dropna().unique(), key=str)
    positive = next((v for v in labels if str(v).lower() in POSITIVE_LABELS), None)
    if positive is None:
        positive = values.value_counts().idxmin()
    s.target_mapping = {str(v): int(v == positive) for v in labels}
    s.df[target] = values.map(lambda v: int(v == positive)).astype("int64")
    return f"encoded target as 0/1 with 1 = '{positive}' ({s.df[target].mean():.1%} of rows)"


def _exclude_unusable(s: _State) -> str:
    for info in s.schema.columns:
        if info.name not in s.df.columns:
            continue
        reason = _UNUSABLE_ROLES.get(info.role)
        if Flag.LEAKAGE_SUSPECT in info.flags:
            reason = "suspected target leakage (predicts the target near-perfectly)"
        if reason:
            s.excluded[info.name] = reason
    s.df = s.df.drop(columns=list(s.excluded))
    return f"excluded {len(s.excluded)} columns from analysis"


# ---------------------------------------------------------------------------
# Optional steps (chosen by the plan)
# ---------------------------------------------------------------------------

def _missing_indicators(s: _State, min_missing: float = 0.05) -> str:
    """Missingness can be the signal: for credit data, 'months since last
    delinquency' is missing exactly when there was no delinquency."""
    added = []
    for col in list(s.df.columns):
        if col == s.schema.target:
            continue
        share = s.df[col].isna().mean()
        if share >= min_missing:
            name = f"{col}__missing"
            s.df[name] = s.df[col].isna().astype("int64")
            added.append(name)
    s.added += added
    return f"added {len(added)} missing-value indicator columns"


def _group_rare_categories(s: _State, min_share: float = 0.01) -> str:
    grouped = 0
    for col in s.roles(ColumnRole.CATEGORICAL):
        shares = s.df[col].value_counts(normalize=True)
        rare = shares[shares < min_share].index
        if len(rare) > 1:
            s.df[col] = s.df[col].where(~s.df[col].isin(rare), "other")
            grouped += 1
    return f"grouped rare categories (<{min_share:.0%}) in {grouped} columns"


def _cap_outliers(s: _State, quantile: float = 0.01) -> str:
    """Winsorize heavy-tailed numeric columns at the 1st/99th percentiles."""
    capped = []
    for col in s.roles(ColumnRole.NUMERIC):
        x = s.df[col]
        if abs(x.skew()) > 2:
            lo, hi = x.quantile([quantile, 1 - quantile])
            s.df[col] = x.clip(lo, hi)
            capped.append(col)
    return f"capped outliers in {len(capped)} heavy-tailed columns"


def _date_features(s: _State) -> str:
    added = []
    for col in s.roles(ColumnRole.DATETIME):
        dates = datetime_values(s.df[col])
        for part, values in (("month", dates.dt.month), ("dayofweek", dates.dt.dayofweek)):
            name = f"{col}__{part}"
            s.df[name] = values.astype("Int64")
            added.append(name)
    s.added += added
    return f"added {len(added)} calendar features"


def _impute(s: _State) -> str:
    """Median/mode fill. Off by default: imputation distorts the very
    relationships EDA is trying to measure, so findings use complete pairs."""
    filled = 0
    for col in s.df.columns:
        if col == s.schema.target or not s.df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(s.df[col]):
            s.df[col] = s.df[col].fillna(s.df[col].median())
        else:
            mode = s.df[col].mode()
            if not mode.empty:
                s.df[col] = s.df[col].fillna(mode.iloc[0])
        filled += 1
    return f"imputed missing values in {filled} columns"


# ---------------------------------------------------------------------------
# Registry, plan, runner
# ---------------------------------------------------------------------------

HYGIENE_STEPS: dict[str, StepFn] = {
    "deduplicate": _deduplicate,
    "drop_missing_target": _drop_missing_target,
    "clean_text": _clean_text,
    "parse_types": _parse_types,
    "encode_target": _encode_target,
    "exclude_unusable": _exclude_unusable,
}

OPTIONAL_STEPS: dict[str, StepFn] = {
    "missing_indicators": _missing_indicators,
    "group_rare_categories": _group_rare_categories,
    "cap_outliers": _cap_outliers,
    "date_features": _date_features,
    "impute": _impute,
}


def recommend_plan(schema: DatasetSchema) -> list[str]:
    """Deterministic default choice of optional steps, based on the schema."""
    plan = []
    if any(c.missing_fraction >= 0.05 and c.role != ColumnRole.TARGET for c in schema.columns):
        plan.append("missing_indicators")
    if schema.by_role(ColumnRole.CATEGORICAL):
        plan.append("group_rare_categories")
    if schema.by_role(ColumnRole.DATETIME):
        plan.append("date_features")
    return plan


def run_preprocessing(
    df: pd.DataFrame, schema: DatasetSchema, plan: list[str] | None = None
) -> PreprocessResult:
    """Run hygiene steps, then the optional steps in `plan` (default: recommended).

    Never modifies `df`. Unknown plan entries are reported, not executed."""
    plan = recommend_plan(schema) if plan is None else plan
    rejected = [p for p in plan if p not in OPTIONAL_STEPS]
    chosen = [p for p in OPTIONAL_STEPS if p in plan]  # fixed order, duplicates ignored

    state = _State(df=df.copy(), schema=schema)
    reports = [StepReport(name, "hygiene", fn(state)) for name, fn in HYGIENE_STEPS.items()]
    reports += [StepReport(name, "optional", OPTIONAL_STEPS[name](state)) for name in chosen]

    out_schema = infer_schema(state.df, target=schema.target)
    return PreprocessResult(
        data=state.df,
        schema=out_schema,
        steps=reports,
        excluded=state.excluded,
        added_columns=state.added,
        target_mapping=state.target_mapping,
        rejected_plan_steps=rejected,
    )
