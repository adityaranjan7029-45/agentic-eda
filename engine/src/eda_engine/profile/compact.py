"""Compact dataset profile: the dataset description an LLM actually sees.

v1 sent `df.info()` plus five raw sample rows: ~4,100 tokens for 100 columns,
half of Groq's per-minute budget before any agent had run, and real customer
rows leaving the server. This module replaces that with:

  - One line per column: name|role|missing%|distinct|summary|flags.
  - Families of numbered columns (bureau_attr_001..064) collapsed into one line.
  - A hard token budget. If the profile doesn't fit, detail is reduced in
    steps (full -> brief -> truncated), never silently overflowing.

Privacy by design: no raw rows are ever included. Actual values appear only
as aggregate shares of low-cardinality categories ("rent:42%"). IDs, high-
cardinality columns (e.g. employer names) and free text never show values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from eda_engine.schema import (
    ColumnInfo,
    ColumnRole,
    DatasetSchema,
    Flag,
    clean_values,
    datetime_values,
    numeric_values,
)
from eda_engine.tokens import TokenCounter, estimate_tokens

ROLE_CODES = {
    ColumnRole.TARGET: "target",
    ColumnRole.ID: "id",
    ColumnRole.EMPTY: "empty",
    ColumnRole.CONSTANT: "const",
    ColumnRole.BINARY: "bin",
    ColumnRole.NUMERIC: "num",
    ColumnRole.DATETIME: "date",
    ColumnRole.CATEGORICAL: "cat",
    ColumnRole.HIGH_CARDINALITY: "hicard",
    ColumnRole.FREE_TEXT: "text",
}

FLAG_CODES = {
    Flag.LEAKAGE_SUSPECT: "LEAK",
    Flag.MOSTLY_MISSING: "sparse",
    Flag.HIDDEN_NULLS: "nulltok",
    Flag.INCONSISTENT_SPELLING: "spell",
    Flag.STORED_AS_TEXT: "astext",
    Flag.NON_PARSEABLE_VALUES: "badvals",
}

LEGEND = (
    "format: name|role|missing%|distinct|summary|flags\n"
    "roles: num bin cat hicard date text id const empty target\n"
    "flags: LEAK=predicts target near-perfectly (likely leakage) sparse=>=60% missing "
    "nulltok=placeholder nulls spell=inconsistent spelling astext=stored as text "
    "badvals=some unparseable values"
)

_FAMILY_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")
_FAMILY_ROLES = {ColumnRole.NUMERIC, ColumnRole.BINARY, ColumnRole.CATEGORICAL}
DETAIL_LEVELS = ("full", "brief", "truncated")


@dataclass(frozen=True)
class CompactProfile:
    text: str
    token_estimate: int
    detail_level: str              # "full" | "brief" | "truncated"
    n_columns: int
    n_columns_omitted: int = 0
    families: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_number(x: float) -> str:
    """Short, human-readable numbers: 1234567 -> 1.2M, 48210 -> 48k, 0.1234 -> 0.12."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "na"
    x = float(x)
    a = abs(x)
    if a >= 1e6:
        s = f"{x / 1e6:.1f}M"
    elif a >= 1e4:
        s = f"{x / 1e3:.0f}k"
    elif a >= 1e3:
        s = f"{x / 1e3:.1f}k"
    elif x.is_integer():
        s = str(int(x))
    elif a >= 1:
        s = f"{x:.3g}"
    else:
        s = f"{x:.2g}"
    return s.replace(".0k", "k").replace(".0M", "M")


def _pct(x: float) -> str:
    return str(int(round(100 * x)))


def _shares(values: pd.Series, k: int) -> str:
    counts = values.value_counts(normalize=True)
    shown = ",".join(f"{str(v)[:20]}:{_pct(p)}%" for v, p in counts.head(k).items())
    rest = len(counts) - min(k, len(counts))
    return shown + (f" +{rest} more" if rest else "")


# ---------------------------------------------------------------------------
# Per-column summaries
# ---------------------------------------------------------------------------

def _summary(series: pd.Series, info: ColumnInfo, level: str) -> str:
    role = info.role
    if role in (ColumnRole.ID, ColumnRole.EMPTY, ColumnRole.HIGH_CARDINALITY):
        return "-"  # never show identifiers or free-form values
    if role == ColumnRole.FREE_TEXT:
        words = clean_values(series).dropna().astype(str).str.split().str.len().mean()
        return f"~{fmt_number(words)} words"
    if role == ColumnRole.CONSTANT:
        value = clean_values(series).dropna()
        return f"={str(value.iloc[0])[:20]}" if len(value) else "-"
    if role == ColumnRole.DATETIME:
        dates = datetime_values(series).dropna()
        return f"{dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}" if len(dates) else "-"

    is_numeric_like = role == ColumnRole.NUMERIC or (
        role == ColumnRole.TARGET and info.n_unique > 2 and not pd.api.types.is_object_dtype(series)
    )
    if is_numeric_like:
        x = numeric_values(series, info).dropna()
        if x.empty:
            return "-"
        if level == "full":
            q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
            skew = x.skew()
            skew_s = f" skew {0.0 if abs(skew) < 0.05 else skew:.1f}" if pd.notna(skew) else ""
            return f"med {fmt_number(med)} iqr {fmt_number(q1)}..{fmt_number(q3)}{skew_s}"
        return f"med {fmt_number(x.median())}"

    # binary, categorical, or a two-valued target: aggregate shares only
    values = clean_values(series).dropna()
    if pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
        values = values.astype(str).str.lower()
    k = 3 if level == "full" else 2
    return _shares(values, k)


def _column_line(df: pd.DataFrame, info: ColumnInfo, level: str) -> str:
    flags = " ".join(FLAG_CODES[f] for f in FLAG_CODES if f in info.flags)
    return "|".join([
        info.name,
        ROLE_CODES[info.role],
        _pct(info.missing_fraction),
        str(info.n_unique),
        _summary(df[info.name], info, level),
        flags,
    ])


# ---------------------------------------------------------------------------
# Column families (bureau_attr_001 ... bureau_attr_064 -> one line)
# ---------------------------------------------------------------------------

def _find_families(schema: DatasetSchema, min_size: int) -> dict[str, list[ColumnInfo]]:
    groups: dict[tuple[str, ColumnRole], list[ColumnInfo]] = {}
    for col in schema.columns:
        m = _FAMILY_RE.match(col.name)
        if not m or col.role not in _FAMILY_ROLES or Flag.LEAKAGE_SUSPECT in col.flags:
            continue  # leakage suspects are always listed on their own line
        groups.setdefault((m["prefix"], col.role), []).append(col)
    return {prefix: cols for (prefix, _role), cols in groups.items() if len(cols) >= min_size}


def _family_line(prefix: str, cols: list[ColumnInfo]) -> str:
    first = _FAMILY_RE.match(cols[0].name)["num"]
    last = _FAMILY_RE.match(cols[-1].name)["num"]
    missing = [c.missing_fraction for c in cols]
    flag_counts = []
    for flag, code in FLAG_CODES.items():
        n = sum(flag in c.flags for c in cols)
        if n:
            flag_counts.append(f"{code}:{n}")
    return "|".join([
        f"{prefix}{first}..{prefix}{last} [{len(cols)} cols]",
        ROLE_CODES[cols[0].role],
        f"{_pct(min(missing))}..{_pct(max(missing))}",
        "-",
        "-",
        " ".join(flag_counts),
    ])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _header(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    lines = [f"rows={len(df)} cols={df.shape[1]} duplicate_rows={int(df.duplicated().sum())}"]
    if schema.target:
        info = schema[schema.target]
        lines.append(f"target={schema.target} {_summary(df[schema.target], info, 'full')}")
    return lines


def build_compact_profile(
    df: pd.DataFrame,
    schema: DatasetSchema,
    max_tokens: int = 2000,
    token_counter: TokenCounter = estimate_tokens,
    min_family_size: int = 4,
) -> CompactProfile:
    """Describe `df` in at most `max_tokens` tokens (as measured by `token_counter`)."""
    families = _find_families(schema, min_family_size)
    in_family = {c.name for cols in families.values() for c in cols}
    header = _header(df, schema)

    for level in DETAIL_LEVELS[:2]:
        body: list[str] = []
        emitted: set[str] = set()
        for col in schema.columns:
            if col.role == ColumnRole.TARGET:
                continue  # already described in the header
            if col.name in in_family:
                prefix = next(p for p, cols in families.items() if col in cols)
                if prefix not in emitted:
                    body.append(_family_line(prefix, families[prefix]))
                    emitted.add(prefix)
            else:
                body.append(_column_line(df, col, level))
        text = "\n".join([*header, LEGEND, *body])
        tokens = token_counter(text)
        if tokens <= max_tokens:
            return CompactProfile(text, tokens, level, df.shape[1],
                                  families={p: [c.name for c in cols] for p, cols in families.items()})

    # Still too big: keep the most important lines first, then cut.
    def priority(col: ColumnInfo) -> int:
        return 0 if col.flags else 1  # flagged columns (leakage, sparse...) are kept first

    ordered = sorted(
        (c for c in schema.columns if c.name not in in_family and c.role != ColumnRole.TARGET),
        key=priority,
    )
    lines = [*header, LEGEND]
    lines += [_family_line(p, cols) for p, cols in families.items()]
    kept = 0
    for col in ordered:
        candidate = _column_line(df, col, "brief")
        note = f"... {len(ordered) - kept - 1} more columns omitted to fit the token budget"
        if token_counter("\n".join([*lines, candidate, note])) > max_tokens:
            break
        lines.append(candidate)
        kept += 1
    omitted = len(ordered) - kept
    if omitted:
        lines.append(f"... {omitted} more columns omitted to fit the token budget")
    text = "\n".join(lines)
    return CompactProfile(text, token_counter(text), "truncated", df.shape[1], omitted,
                          {p: [c.name for c in cols] for p, cols in families.items()})
