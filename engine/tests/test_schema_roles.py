"""Tests for column role detection.

Two layers:
  1. Small hand-built tables pinning down each rule, including the false
     positives that broke v1 (e.g. 'paid_amount' treated as an ID).
  2. The synthetic credit dataset: every role and data-quality problem its
     manifest documents must be detected, and nothing else may be flagged.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from eda_engine.schema import ColumnRole, Flag, infer_schema
from eda_engine.stats import binary_auc
from eda_engine.synth.credit import generate_credit_applications

R = ColumnRole


def role_of(values, name="col", **kwargs):
    return infer_schema(pd.DataFrame({name: values}), **kwargs)[name]


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def test_id_detected_by_name_token_including_camel_case():
    assert role_of([f"C{i}" for i in range(100)], name="customerID").role == R.ID


def test_repeated_foreign_key_named_id_is_still_an_id():
    values = [i // 2 for i in range(100)]  # each id appears twice: 50% unique
    assert role_of(values, name="order_id").role == R.ID


def test_id_substring_inside_a_word_is_not_an_id():
    """v1 matched 'id' as a substring, so 'paid' and 'valid' looked like IDs."""
    rng = np.random.default_rng(0)
    assert role_of(rng.normal(1000, 50, 200), name="paid_amount").role == R.NUMERIC
    assert role_of(rng.normal(0, 1, 200), name="valid_score").role == R.NUMERIC


def test_unnamed_unique_short_codes_are_ids():
    codes = [f"{i:04d}-VHVEG" for i in range(200)]
    assert role_of(codes, name="ref").role == R.ID


def test_row_counter_is_an_id_but_unique_floats_are_not():
    assert role_of(list(range(1, 201)), name="row").role == R.ID
    assert role_of(np.linspace(0, 1, 200), name="score").role == R.NUMERIC


# ---------------------------------------------------------------------------
# Binary, numeric-as-text, dates, text
# ---------------------------------------------------------------------------

def test_two_valued_columns_are_binary_whatever_their_type():
    assert role_of(["yes", "no"] * 50).role == R.BINARY
    assert role_of([0, 1] * 50).role == R.BINARY
    assert role_of([36, 60] * 50).role == R.BINARY
    assert role_of([True, False] * 50).role == R.BINARY


def test_case_and_whitespace_variants_are_flagged_but_counted_once():
    info = role_of(["Yes", "yes", "YES", "no", " no"] * 20)
    assert info.role == R.BINARY
    assert info.n_unique == 2
    assert info.has(Flag.INCONSISTENT_SPELLING)


def test_whitespace_only_variants_are_flagged():
    """'electronic check' vs 'electronic check ' (trailing space) is a real
    pattern in the v1 churn sample. Case is identical, so only a raw-vs-stripped
    comparison catches it."""
    info = role_of(["electronic check", "electronic check ", "mailed check", "bank transfer"] * 25)
    assert info.n_unique == 3
    assert info.has(Flag.INCONSISTENT_SPELLING)


@pytest.mark.parametrize(
    "values,hint",
    [
        (["42", "7", "13", "-2.5"] * 25, "numeric_string"),
        (["$1,234.50", "$99.00", "$12,000.00", "$5.25"] * 25, "currency"),
        (["12%", "7.5%", "100%", "0%"] * 25, "percent"),
    ],
)
def test_numbers_stored_as_text(values, hint):
    info = role_of(values)
    assert info.role == R.NUMERIC
    assert info.parse_hint == hint
    assert info.has(Flag.STORED_AS_TEXT)


def test_decimal_commas_are_not_silently_misread_as_thousands():
    """'1,5' must not become 15; it stays text rather than corrupting numbers."""
    info = role_of(["1,5", "2,25", "3,75", "4,5"] * 25)
    assert info.role != R.NUMERIC


def test_a_few_unparseable_values_are_tolerated_and_flagged():
    values = [str(i) for i in range(199)] + ["abc"]
    info = role_of(values, name="amount")
    assert info.role == R.NUMERIC
    assert info.has(Flag.NON_PARSEABLE_VALUES)


def test_dates_as_text_and_as_datetime_type():
    text_dates = [f"2024-01-{d:02d}" for d in range(1, 29)] * 4
    assert role_of(text_dates).role == R.DATETIME
    assert role_of(text_dates).parse_hint == "date_string"
    assert role_of(pd.to_datetime(text_dates)).role == R.DATETIME


def test_free_text_versus_high_cardinality_codes():
    sentences = [f"customer called about invoice number {i} again today" for i in range(120)]
    assert role_of(sentences, name="notes").role == R.FREE_TEXT
    employers = [f"employer {i % 80}" for i in range(400)]
    assert role_of(employers, name="employer").role == R.HIGH_CARDINALITY


def test_few_distinct_text_values_are_categorical():
    assert role_of(["rent", "own", "mortgage"] * 40).role == R.CATEGORICAL


# ---------------------------------------------------------------------------
# Missing, empty, constant
# ---------------------------------------------------------------------------

def test_hidden_null_tokens_count_as_missing():
    info = role_of(["a", "b", "c", "N/A", "unknown", "?", " "] * 20)
    assert info.has(Flag.HIDDEN_NULLS)
    assert info.missing_fraction == pytest.approx(4 / 7)


def test_empty_and_constant_columns():
    assert role_of([np.nan] * 50).role == R.EMPTY
    assert role_of(["N/A", "unknown", ""] * 20).role == R.EMPTY
    assert role_of(["v3"] * 50).role == R.CONSTANT


def test_mostly_missing_is_a_flag_not_a_role():
    """A mostly-missing column still has a type, and its missingness can be
    informative, so it must not be thrown away just for being sparse."""
    values = [1.5, 2.5, 3.5] + [np.nan] * 17
    info = role_of(values * 5)
    assert info.role == R.NUMERIC
    assert info.has(Flag.MOSTLY_MISSING)


# ---------------------------------------------------------------------------
# Target and leakage
# ---------------------------------------------------------------------------

@pytest.fixture
def leaky_frame():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    return pd.DataFrame({
        "y": y,
        "honest": rng.normal(size=2000) + 0.5 * y,
        "leak_numeric": y * 100 + rng.integers(0, 5, 2000),
        "leak_category": np.where(y == 1, "sent_to_collections", "current"),
    })


def test_leakage_suspects_are_flagged_and_honest_features_are_not(leaky_frame):
    schema = infer_schema(leaky_frame, target="y")
    assert set(schema.with_flag(Flag.LEAKAGE_SUSPECT)) == {"leak_numeric", "leak_category"}
    assert "AUC" in schema["leak_numeric"].reason
    assert schema["y"].role == R.TARGET


def test_no_leakage_screen_without_a_binary_target(leaky_frame):
    assert infer_schema(leaky_frame).with_flag(Flag.LEAKAGE_SUSPECT) == []
    continuous = leaky_frame.assign(y=np.arange(len(leaky_frame)) * 1.5)
    assert infer_schema(continuous, target="y").with_flag(Flag.LEAKAGE_SUSPECT) == []


def test_text_target_is_supported(leaky_frame):
    frame = leaky_frame.assign(y=np.where(leaky_frame["y"] == 1, "Yes", "No"))
    assert "leak_numeric" in infer_schema(frame, target="y").with_flag(Flag.LEAKAGE_SUSPECT)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="not found"):
        infer_schema(pd.DataFrame({"a": [1, 2]}), target="missing")
    with pytest.raises(ValueError, match="duplicate"):
        infer_schema(pd.DataFrame([[1, 2]], columns=["a", "a"]))


# ---------------------------------------------------------------------------
# The synthetic credit dataset: manifest is the answer key
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def credit():
    dataset = generate_credit_applications(n_rows=5000, seed=7)
    original = dataset.data.copy()
    schema = infer_schema(dataset.data, target=dataset.manifest["target"])
    return dataset, schema, original


def test_credit_schema_does_not_modify_input(credit):
    dataset, _, original = credit
    pd.testing.assert_frame_equal(dataset.data, original)


def test_credit_roles_match_the_manifest(credit):
    dataset, schema, _ = credit
    m = dataset.manifest
    assert schema[m["target"]].role == R.TARGET
    for col in m["id_columns"]:
        assert schema[col].role == R.ID, col
    for col in m["constant_columns"]:
        assert schema[col].role == R.CONSTANT, col
    for col in m["date_columns"]:
        assert schema[col].role == R.DATETIME, col
    for col in m["high_cardinality_columns"]:
        assert schema[col].role == R.HIGH_CARDINALITY, col
    for col in m["data_quality_issues"]["currency_strings"]:
        assert schema[col].role == R.NUMERIC and schema[col].parse_hint == "currency", col
    for col in m["noise_columns"]:
        assert schema[col].role == R.NUMERIC, col


def test_credit_data_quality_flags_match_the_manifest(credit):
    dataset, schema, _ = credit
    issues = dataset.manifest["data_quality_issues"]
    assert set(issues["hidden_null_tokens"]) <= set(schema.with_flag(Flag.HIDDEN_NULLS))
    assert set(issues["inconsistent_category_spellings"]) == set(schema.with_flag(Flag.INCONSISTENT_SPELLING))
    assert set(dataset.manifest["mostly_missing_columns"]) <= set(schema.with_flag(Flag.MOSTLY_MISSING))


def test_credit_leakage_flags_exactly_the_planted_leaks(credit):
    """Both leaks caught; none of the real drivers or noise falsely accused."""
    dataset, schema, _ = credit
    assert set(schema.with_flag(Flag.LEAKAGE_SUSPECT)) == set(dataset.manifest["leakage_columns"])


def test_clean_credit_data_has_no_quality_flags():
    dataset = generate_credit_applications(n_rows=2000, seed=5, messy=False)
    schema = infer_schema(dataset.data)
    assert schema.with_flag(Flag.HIDDEN_NULLS) == []
    assert schema.with_flag(Flag.INCONSISTENT_SPELLING) == []
    # Dates are always text in a CSV; only the messy dataset stores money as text.
    assert schema.with_flag(Flag.STORED_AS_TEXT) == dataset.manifest["date_columns"]


def test_schema_serializes_to_json(credit):
    _, schema, _ = credit
    restored = json.loads(json.dumps(schema.to_dict()))
    assert len(restored["columns"]) == len(schema.columns)
    assert restored["target"] == "default_flag"


# ---------------------------------------------------------------------------
# binary_auc
# ---------------------------------------------------------------------------

def test_binary_auc_edge_cases():
    y = pd.Series([0, 0, 1, 1])
    assert binary_auc(pd.Series([1, 2, 3, 4]), y) == 1.0
    assert binary_auc(pd.Series([4, 3, 2, 1]), y) == 0.0
    assert binary_auc(pd.Series([5, 5, 5, 5]), y) == 0.5
    assert np.isnan(binary_auc(pd.Series([1, 2, 3]), pd.Series([1, 1, 1])))


def test_binary_auc_matches_sklearn_with_ties_and_missing_values():
    rng = np.random.default_rng(3)
    y = pd.Series(rng.integers(0, 2, 500))
    scores = pd.Series(rng.integers(0, 20, 500).astype(float))
    scores[::17] = np.nan
    ok = scores.notna()
    assert binary_auc(scores, y) == pytest.approx(roc_auc_score(y[ok], scores[ok]))
