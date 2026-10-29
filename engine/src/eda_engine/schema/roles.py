"""Column role detection: decide what each column *is* before any analysis runs.

Every later stage depends on this. Profiling, preprocessing and the findings
engine all treat an ID, a date, a money amount stored as text and a free-text
note differently, and getting a role wrong silently corrupts everything after
it (v1 label-encoded numbers stored as text and "found" patterns in IDs).

Design rules:
  - Deterministic and explainable: every column gets a `reason` saying which
    rule fired, so a user (or an interviewer) can see why.
  - Read-only: the input DataFrame is never modified. Fixing problems is the
    preprocessing step's job; this step only describes them via `flags`.
  - No LLM involved. This runs locally and costs zero tokens.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from eda_engine.stats import auc_strength, binary_auc


class ColumnRole(StrEnum):
    TARGET = "target"
    ID = "id"
    EMPTY = "empty"
    CONSTANT = "constant"
    BINARY = "binary"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    HIGH_CARDINALITY = "high_cardinality"
    FREE_TEXT = "free_text"


class Flag(StrEnum):
    HIDDEN_NULLS = "hidden_nulls"                    # placeholder strings like "N/A" meaning missing
    INCONSISTENT_SPELLING = "inconsistent_spelling"  # "Rent" / " rent" / "RENT"
    MOSTLY_MISSING = "mostly_missing"
    STORED_AS_TEXT = "stored_as_text"                # numbers or dates stored as strings
    NON_PARSEABLE_VALUES = "non_parseable_values"    # a few values didn't parse as the column's type
    LEAKAGE_SUSPECT = "leakage_suspect"              # predicts the target too well to be real


DEFAULT_NULL_TOKENS = frozenset({
    "", "na", "n/a", "nan", "null", "none", "nil", "missing", "unknown",
    "undefined", "?", "-", "--", "#n/a",
})

_ID_NAME_TOKENS = frozenset({"id", "uuid", "guid", "identifier"})
_CURRENCY_RE = re.compile(r"[$€£₹¥]")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_DATE_LIKE_RE = re.compile(
    r"\d[-/.:]\d|\d{4}-\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.IGNORECASE
)


@dataclass(frozen=True)
class RoleConfig:
    """Thresholds for role detection. Defaults suit typical tabular business data."""

    null_tokens: frozenset[str] = DEFAULT_NULL_TOKENS
    id_named_min_uniqueness: float = 0.5   # 'customer_id' is an ID even when repeated (a foreign key)
    id_unnamed_min_uniqueness: float = 0.95
    id_max_code_length: int = 40
    mostly_missing_threshold: float = 0.6
    max_categories: int = 50
    parse_success_threshold: float = 0.99
    date_parse_threshold: float = 0.9
    free_text_min_words: float = 3.0
    leakage_min_auc_strength: float = 0.45  # flags AUC >= 0.95 or <= 0.05
    category_smoothing: float = 20.0
    sample_size: int = 20_000
    seed: int = 0


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    role: ColumnRole
    dtype: str
    missing_fraction: float
    n_unique: int
    reason: str
    parse_hint: str | None = None  # "currency" | "percent" | "numeric_string" | "date_string"
    flags: tuple[Flag, ...] = ()

    def has(self, flag: Flag) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role.value,
            "dtype": self.dtype,
            "missing_fraction": round(self.missing_fraction, 4),
            "n_unique": self.n_unique,
            "reason": self.reason,
            "parse_hint": self.parse_hint,
            "flags": [f.value for f in self.flags],
        }


@dataclass(frozen=True)
class DatasetSchema:
    n_rows: int
    target: str | None
    columns: tuple[ColumnInfo, ...]

    def __getitem__(self, name: str) -> ColumnInfo:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def by_role(self, role: ColumnRole) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def with_flag(self, flag: Flag) -> list[str]:
        return [c.name for c in self.columns if flag in c.flags]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "target": self.target,
            "columns": [c.to_dict() for c in self.columns],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_tokens(name: str) -> set[str]:
    """'customerID' -> {'customer', 'id'}; 'loan_ID-2' -> {'loan', 'id', '2'}.
    Token matching avoids v1's substring bug where 'paid' and 'valid' contained 'id'."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t}


def _is_text(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _sample(values: pd.Series, cfg: RoleConfig) -> pd.Series:
    if len(values) > cfg.sample_size:
        return values.sample(cfg.sample_size, random_state=cfg.seed)
    return values


def _clean_text(series: pd.Series, cfg: RoleConfig) -> tuple[pd.Series, int]:
    """Strip whitespace and turn placeholder tokens into real missing values.
    Returns the cleaned strings and how many hidden nulls were found."""
    text = series.astype("string").str.strip()
    hidden = text.str.lower().isin(cfg.null_tokens) & series.notna()
    return text.mask(hidden), int(hidden.sum())


def _parse_numeric(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse strings like '$1,234.50', '12%' or '42'. Returns (numbers, hint)."""
    has_currency = values.str.contains(_CURRENCY_RE).mean() > 0.5
    has_percent = values.str.endswith("%").mean() > 0.5
    cleaned = values.str.replace(_CURRENCY_RE, "", regex=True).str.replace("%", "", regex=False).str.strip()
    # Only strip commas that are thousands separators ("1,234"), never decimal
    # commas ("1,5" stays unparseable rather than silently becoming 15).
    is_thousands = cleaned.str.match(_THOUSANDS_RE).fillna(False).astype(bool)
    cleaned = cleaned.where(~is_thousands, cleaned.str.replace(",", "", regex=False))
    numbers = pd.to_numeric(cleaned, errors="coerce")
    hint = "currency" if has_currency else "percent" if has_percent else "numeric_string"
    return numbers, hint


def _date_parse_fraction(values: pd.Series) -> float:
    looks_like_date = values.str.contains(_DATE_LIKE_RE).mean()
    if looks_like_date < 0.9:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _binary_codes(values: pd.Series) -> pd.Series:
    """Map a two-valued series to 0/1 (sorted order, so 'no'->0, 'yes'->1, 0->0, 1->1)."""
    present = sorted(values.dropna().unique(), key=str)
    return values.map({present[0]: 0, present[1]: 1}).astype("Float64")


def _numeric_values(series: pd.Series, info: ColumnInfo, cfg: RoleConfig) -> pd.Series:
    if info.parse_hint in ("currency", "percent", "numeric_string"):
        cleaned, _ = _clean_text(series, cfg)
        return _parse_numeric(cleaned)[0]
    return pd.to_numeric(series, errors="coerce")


def numeric_values(series: pd.Series, info: ColumnInfo, config: RoleConfig | None = None) -> pd.Series:
    """The column as floats, parsed the same way role detection read it
    ('$1,234.50' -> 1234.5, '12%' -> 12.0). Unparseable values become NaN."""
    return _numeric_values(series, info, config or RoleConfig()).astype("float64")


def clean_values(series: pd.Series, config: RoleConfig | None = None) -> pd.Series:
    """Text with whitespace stripped and placeholder tokens ('N/A', 'unknown') turned into NaN."""
    if not _is_text(series):
        return series
    return _clean_text(series, config or RoleConfig())[0]


def datetime_values(series: pd.Series) -> pd.Series:
    """The column as datetimes; unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


# ---------------------------------------------------------------------------
# Per-column classification
# ---------------------------------------------------------------------------

def _classify(name: str, series: pd.Series, n_rows: int, target: str | None, cfg: RoleConfig) -> ColumnInfo:
    dtype = str(series.dtype)
    flags: list[Flag] = []

    if _is_text(series):
        values, n_hidden = _clean_text(series, cfg)
        if n_hidden:
            flags.append(Flag.HIDDEN_NULLS)
    else:
        values, n_hidden = series, 0

    missing = float(values.isna().mean()) if n_rows else 1.0
    present = values.dropna()
    n_present = len(present)
    if missing >= cfg.mostly_missing_threshold and missing < 1.0:
        flags.append(Flag.MOSTLY_MISSING)

    def info(role: ColumnRole, reason: str, n_unique: int, hint: str | None = None) -> ColumnInfo:
        return ColumnInfo(name, role, dtype, missing, n_unique, reason, hint, tuple(flags))

    if n_present == 0:
        return info(ColumnRole.EMPTY, "no non-missing values", 0)

    if name == target:
        return info(ColumnRole.TARGET, "chosen as the target column", int(present.nunique()))

    if pd.api.types.is_datetime64_any_dtype(series):
        return info(ColumnRole.DATETIME, "stored as a datetime type", int(present.nunique()))

    # Count categories on canonical spelling so 'Rent'/' rent'/'RENT' count once.
    canonical = present.str.lower() if _is_text(series) else present
    n_unique = int(canonical.nunique())
    uniqueness = n_unique / n_present

    if n_unique == 1:
        return info(ColumnRole.CONSTANT, "a single distinct value", n_unique)

    named_like_id = bool(_name_tokens(name) & _ID_NAME_TOKENS)
    if named_like_id and uniqueness >= cfg.id_named_min_uniqueness:
        return info(ColumnRole.ID, f"named like an identifier and {uniqueness:.0%} unique", n_unique)

    if pd.api.types.is_bool_dtype(series):
        return info(ColumnRole.BINARY, "boolean type", n_unique)

    if not _is_text(series):
        if n_unique == 2:
            return info(ColumnRole.BINARY, "numeric with exactly two distinct values", n_unique)
        is_row_counter = (
            pd.api.types.is_integer_dtype(series)
            and uniqueness == 1.0
            and present.is_monotonic_increasing
            and (present.diff().dropna() == 1).all()
        )
        if is_row_counter:
            return info(ColumnRole.ID, "consecutive integers, one per row (a row counter)", n_unique)
        return info(ColumnRole.NUMERIC, "numeric type", n_unique)

    # ---- Text columns from here on ----
    # Compare raw spellings (before stripping) against canonical ones, so both
    # case variants ('Rent'/'RENT') and whitespace variants ('rent'/' rent') count.
    raw_spellings = series[values.notna()].astype("string").nunique()
    if n_unique < raw_spellings:
        flags.append(Flag.INCONSISTENT_SPELLING)
    if n_unique == 2:
        return info(ColumnRole.BINARY, "text with exactly two distinct values", n_unique)

    sample = _sample(present, cfg)
    numbers, hint = _parse_numeric(sample)
    parsed_fraction = float(numbers.notna().mean())
    if parsed_fraction >= cfg.parse_success_threshold:
        flags.append(Flag.STORED_AS_TEXT)
        if parsed_fraction < 1.0:
            flags.append(Flag.NON_PARSEABLE_VALUES)
        return info(ColumnRole.NUMERIC, f"text that parses as numbers ({hint})", n_unique, hint)

    date_fraction = _date_parse_fraction(sample)
    if date_fraction >= cfg.date_parse_threshold:
        flags.append(Flag.STORED_AS_TEXT)
        if date_fraction < 1.0:
            flags.append(Flag.NON_PARSEABLE_VALUES)
        return info(ColumnRole.DATETIME, "text that parses as dates", n_unique, "date_string")

    no_spaces = (~sample.str.contains(r"\s", regex=True)).mean() >= 0.99
    short = sample.str.len().mean() <= cfg.id_max_code_length
    if uniqueness >= cfg.id_unnamed_min_uniqueness and no_spaces and short:
        return info(ColumnRole.ID, f"short codes without spaces, {uniqueness:.0%} unique", n_unique)

    if n_unique <= cfg.max_categories:
        return info(ColumnRole.CATEGORICAL, f"{n_unique} distinct values", n_unique)

    mean_words = sample.str.split().str.len().mean()
    if mean_words >= cfg.free_text_min_words:
        return info(ColumnRole.FREE_TEXT, f"long text, {mean_words:.1f} words on average", n_unique)
    return info(ColumnRole.HIGH_CARDINALITY, f"{n_unique} distinct short values", n_unique)


# ---------------------------------------------------------------------------
# Leakage screen
# ---------------------------------------------------------------------------

def _leakage_scores(df: pd.DataFrame, schema_cols: list[ColumnInfo], target: str, cfg: RoleConfig) -> dict[str, float]:
    """Single-column AUC against a binary target, for columns that could leak.

    A real-world feature almost never predicts the outcome near-perfectly on
    its own. When one does, it's usually recorded *after* the outcome
    (e.g. 'days past due' for a default target). Returns {column: auc}."""
    target_info = next(c for c in schema_cols if c.name == target)
    if target_info.n_unique != 2:
        return {}

    y_raw = df[target]
    if _is_text(y_raw):
        y_raw = _clean_text(y_raw, cfg)[0].str.lower()
    y = _binary_codes(y_raw)

    scores: dict[str, float] = {}
    for col in schema_cols:
        if col.name == target:
            continue
        series = df[col.name]
        if col.role == ColumnRole.NUMERIC:
            x = _numeric_values(series, col, cfg)
        elif col.role == ColumnRole.BINARY:
            x = series if not _is_text(series) else _clean_text(series, cfg)[0].str.lower()
            x = _binary_codes(x)
        elif col.role == ColumnRole.CATEGORICAL:
            cats = _clean_text(series, cfg)[0].str.lower()
            frame = pd.DataFrame({"c": cats, "y": y}).dropna()
            prior = frame["y"].mean()
            stats = frame.groupby("c")["y"].agg(["sum", "count"])
            m = cfg.category_smoothing
            rate = (stats["sum"] + prior * m) / (stats["count"] + m)
            x = cats.map(rate)
        else:
            continue
        scores[col.name] = binary_auc(x.astype("float64"), y.astype("float64"))
    return scores


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame, target: str | None = None, config: RoleConfig | None = None) -> DatasetSchema:
    """Classify every column of `df`. Never modifies `df`."""
    cfg = config or RoleConfig()
    if target is not None and target not in df.columns:
        raise ValueError(f"target column {target!r} not found")
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"duplicate column names: {dupes}")

    n_rows = len(df)
    columns = [_classify(str(name), df[name], n_rows, target, cfg) for name in df.columns]

    if target is not None:
        scores = _leakage_scores(df, columns, target, cfg)
        for i, col in enumerate(columns):
            auc = scores.get(col.name)
            if auc is not None and auc_strength(auc) >= cfg.leakage_min_auc_strength:
                columns[i] = ColumnInfo(
                    col.name, col.role, col.dtype, col.missing_fraction, col.n_unique,
                    f"{col.reason}; predicts '{target}' almost perfectly on its own (AUC {auc:.3f})",
                    col.parse_hint, col.flags + (Flag.LEAKAGE_SUSPECT,),
                )

    return DatasetSchema(n_rows=n_rows, target=target, columns=tuple(columns))
