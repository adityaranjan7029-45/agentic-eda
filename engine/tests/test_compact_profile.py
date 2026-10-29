"""Tests for the compact dataset profile sent to the LLM."""

import io

import numpy as np
import pandas as pd
import pytest

from eda_engine.profile import build_compact_profile, fmt_number
from eda_engine.schema import infer_schema
from eda_engine.synth.credit import generate_credit_applications
from eda_engine.tokens import estimate_tokens


@pytest.fixture(scope="module")
def credit():
    dataset = generate_credit_applications(n_rows=5000, seed=42)
    schema = infer_schema(dataset.data, target="default_flag")
    return dataset, schema


def _v1_style_profile(df: pd.DataFrame) -> str:
    """What v1 sent the Planner: df.info() + null counts + 5 raw rows."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue() + str(df.isnull().sum()) + df.head().to_string()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_120_column_dataset_fits_the_default_budget_at_full_detail(credit):
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema)
    assert profile.detail_level == "full"
    assert profile.token_estimate <= 2000
    assert profile.n_columns == 122


def test_far_smaller_than_the_v1_profile(credit):
    dataset, schema = credit
    v2 = build_compact_profile(dataset.data, schema).token_estimate
    v1 = estimate_tokens(_v1_style_profile(dataset.data))
    assert v2 < 0.6 * v1


@pytest.mark.parametrize("budget", [300, 600, 1000, 1500, 2500])
def test_any_budget_is_respected(credit, budget):
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema, max_tokens=budget)
    assert profile.token_estimate <= budget
    assert estimate_tokens(profile.text) == profile.token_estimate


def test_detail_degrades_in_order(credit):
    dataset, schema = credit
    levels = [build_compact_profile(dataset.data, schema, max_tokens=b).detail_level for b in (5000, 1700, 600)]
    assert levels == ["full", "brief", "truncated"]


def test_truncation_keeps_flagged_columns_first(credit):
    """Under a tight budget, leakage suspects must still be visible."""
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema, max_tokens=500)
    assert profile.detail_level == "truncated"
    assert profile.n_columns_omitted > 0
    for col in dataset.manifest["leakage_columns"]:
        assert f"{col}|" in profile.text
    assert "omitted to fit the token budget" in profile.text


def test_custom_exact_counter_is_used(credit):
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema, max_tokens=50, token_counter=len)
    assert profile.token_estimate == len(profile.text)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def test_every_column_is_represented_once(credit):
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema)
    lines = profile.text.splitlines()
    in_families = {c for cols in profile.families.values() for c in cols}
    listed = {line.split("|")[0] for line in lines if "|" in line}
    for col in dataset.data.columns:
        if col == "default_flag":
            assert lines[1].startswith("target=default_flag")
        elif col in in_families:
            assert col not in listed
        else:
            assert col in listed, col


def test_numbered_columns_collapse_into_one_family_line(credit):
    dataset, schema = credit
    profile = build_compact_profile(dataset.data, schema)
    assert len(profile.families["bureau_attr_"]) == 64
    assert "bureau_attr_001..bureau_attr_064 [64 cols]" in profile.text


def test_leakage_suspects_are_never_hidden_inside_a_family():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    frame = pd.DataFrame({f"f_{i}": rng.normal(size=1000) for i in range(6)} | {"y": y})
    frame["f_6"] = y * 10 + rng.normal(size=1000)  # leaky family member
    profile = build_compact_profile(frame, infer_schema(frame, target="y"))
    assert "f_6" not in profile.families["f_"]
    assert any(line.startswith("f_6|") and "LEAK" in line for line in profile.text.splitlines())


def test_flags_and_parsed_values_appear(credit):
    dataset, schema = credit
    text = build_compact_profile(dataset.data, schema).text
    lines = {line.split("|")[0]: line for line in text.splitlines() if "|" in line}
    assert "LEAK" in lines["days_past_due_current"]
    assert "sparse" in lines["months_since_last_delinquency"]
    assert "nulltok" in lines["employment_type"]
    assert "spell" in lines["home_ownership"]
    assert "astext" in lines["savings_balance"] and "med " in lines["savings_balance"]  # '$1,234' parsed
    assert "2024-" in lines["application_date"]


def test_profile_is_deterministic(credit):
    dataset, schema = credit
    assert build_compact_profile(dataset.data, schema).text == build_compact_profile(dataset.data, schema).text


# ---------------------------------------------------------------------------
# Privacy by design
# ---------------------------------------------------------------------------

def test_no_raw_values_from_identifying_columns(credit):
    """IDs, employer names (high cardinality) and free text never appear;
    only aggregate shares of low-cardinality categories do."""
    dataset, schema = credit
    text = build_compact_profile(dataset.data, schema).text
    for col in ("application_id", "customer_id", "employer_name"):
        for value in dataset.data[col].dropna().astype(str).unique()[:50]:
            assert value not in text, (col, value)


def test_free_text_values_never_appear():
    notes = [f"customer {i} disputed the late fee on account ending {1000 + i}" for i in range(200)]
    frame = pd.DataFrame({"notes": notes, "x": range(200)})
    text = build_compact_profile(frame, infer_schema(frame)).text
    assert "disputed" not in text
    assert "notes|text|" in text


# ---------------------------------------------------------------------------
# Edge cases and formatting
# ---------------------------------------------------------------------------

def test_handles_edge_case_columns_and_no_target():
    frame = pd.DataFrame({
        "empty": [np.nan] * 50,
        "constant": ["v3"] * 50,
        "when": pd.date_range("2024-01-01", periods=50),
        "flag": [True, False] * 25,
        "amount": np.linspace(0, 1e6, 50),
    })
    profile = build_compact_profile(frame, infer_schema(frame))
    assert "target=" not in profile.text
    assert "constant|const|0|1|=v3|" in profile.text
    assert "when|date|0|50|2024-01-01..2024-02-19|" in profile.text


def test_v1_churn_sample_profiles_cleanly():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_customer_churn.csv"
    if not path.exists():
        pytest.skip("v1 sample not present")
    frame = pd.read_csv(path)
    profile = build_compact_profile(frame, infer_schema(frame, target="Churn"))
    assert profile.detail_level == "full"
    assert profile.token_estimate < 600


@pytest.mark.parametrize(
    "value,expected",
    [
        (1_234_567, "1.2M"), (48_210, "48k"), (2_000, "2k"), (1_500, "1.5k"),
        (42, "42"), (12.25, "12.2"), (0.1234, "0.12"), (-3, "-3"), (float("nan"), "na"),
    ],
)
def test_fmt_number(value, expected):
    assert fmt_number(value) == expected
