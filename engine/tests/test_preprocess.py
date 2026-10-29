"""Tests for preprocessing: hygiene always runs, optional steps follow the plan,
and the output stays in original, readable units."""

import numpy as np
import pandas as pd
import pytest

from eda_engine.preprocess import OPTIONAL_STEPS, recommend_plan, run_preprocessing
from eda_engine.schema import ColumnRole, infer_schema
from eda_engine.synth.credit import generate_credit_applications


@pytest.fixture(scope="module")
def credit():
    dataset = generate_credit_applications(n_rows=5000, seed=42)
    original = dataset.data.copy()
    schema = infer_schema(dataset.data, target="default_flag")
    result = run_preprocessing(dataset.data, schema)
    return dataset, original, result


# ---------------------------------------------------------------------------
# Hygiene on the messy credit dataset (manifest = answer key)
# ---------------------------------------------------------------------------

def test_input_is_never_modified(credit):
    dataset, original, _ = credit
    pd.testing.assert_frame_equal(dataset.data, original)


def test_duplicates_removed(credit):
    dataset, _, result = credit
    assert len(result.data) == len(dataset.data) - dataset.manifest["data_quality_issues"]["duplicate_rows"]
    assert result.data.duplicated().sum() == 0


def test_placeholder_nulls_and_spelling_variants_are_cleaned(credit):
    _, _, result = credit
    assert set(result.data["home_ownership"].dropna()) == {"rent", "mortgage", "own"}
    assert not result.data["employment_type"].isin(["N/A", "unknown", "?", " "]).any()
    assert result.data["employment_type"].isna().sum() > 0


def test_numbers_and_dates_stored_as_text_are_parsed(credit):
    _, _, result = credit
    assert pd.api.types.is_float_dtype(result.data["savings_balance"])
    assert pd.api.types.is_datetime64_any_dtype(result.data["application_date"])


def test_unusable_and_leaky_columns_are_excluded_with_reasons(credit):
    dataset, _, result = credit
    m = dataset.manifest
    expected = set(m["id_columns"] + m["constant_columns"] + m["leakage_columns"] + m["high_cardinality_columns"])
    assert set(result.excluded) == expected
    assert "leakage" in result.excluded["days_past_due_current"]
    assert not expected & set(result.data.columns)


def test_target_is_encoded_as_zero_one(credit):
    _, _, result = credit
    assert set(result.data["default_flag"].unique()) == {0, 1}
    assert result.target_mapping == {"0": 0, "1": 1}


def test_values_stay_in_original_units(credit):
    """v1 z-scored everything before analysis ('credit score = -0.8').
    v2 must leave credit scores looking like credit scores."""
    dataset, _, result = credit
    assert result.data["credit_score"].median() == pytest.approx(dataset.data["credit_score"].median(), abs=2)
    assert 600 < result.data["credit_score"].median() < 760


def test_no_column_explosion(credit):
    """v1's feature engineering added 45 ratio columns from the top-10
    variance columns. v2 adds only a handful of explainable columns."""
    dataset, _, result = credit
    assert result.data.shape[1] <= dataset.data.shape[1]
    assert len(result.added_columns) <= 5


def test_informative_missingness_is_kept_as_a_feature(credit):
    _, _, result = credit
    assert "months_since_last_delinquency" in result.data.columns
    assert "months_since_last_delinquency__missing" in result.added_columns


def test_output_schema_describes_the_output(credit):
    _, _, result = credit
    assert set(c.name for c in result.schema.columns) == set(result.data.columns)
    assert result.schema["default_flag"].role == ColumnRole.TARGET


def test_every_step_reports_what_it_did(credit):
    _, _, result = credit
    names = [s.name for s in result.steps]
    assert names[:6] == ["deduplicate", "drop_missing_target", "clean_text", "parse_types", "encode_target", "exclude_unusable"]
    assert all(s.detail for s in result.steps)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_hygiene_cannot_be_skipped_by_the_plan(credit):
    dataset, _, _ = credit
    schema = infer_schema(dataset.data, target="default_flag")
    result = run_preprocessing(dataset.data, schema, plan=[])
    assert result.data.duplicated().sum() == 0
    assert "days_past_due_current" in result.excluded
    assert [s.kind for s in result.steps] == ["hygiene"] * 6


def test_unknown_plan_steps_are_rejected_not_run(credit):
    dataset, _, _ = credit
    schema = infer_schema(dataset.data, target="default_flag")
    result = run_preprocessing(dataset.data, schema, plan=["impute", "delete_everything", "pca"])
    assert result.rejected_plan_steps == ["delete_everything", "pca"]
    assert [s.name for s in result.steps if s.kind == "optional"] == ["impute"]


def test_recommended_plan_follows_the_schema():
    rng = np.random.default_rng(0)
    plain = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    assert recommend_plan(infer_schema(plain)) == []
    richer = plain.assign(
        when=pd.date_range("2024-01-01", periods=100),
        grade=["a", "b", "c", "d"] * 25,
        sparse=[1.0] * 10 + [np.nan] * 90,
    )
    assert recommend_plan(infer_schema(richer)) == ["missing_indicators", "group_rare_categories", "date_features"]
    assert set(recommend_plan(infer_schema(richer))) <= set(OPTIONAL_STEPS)


# ---------------------------------------------------------------------------
# Individual optional steps
# ---------------------------------------------------------------------------

def _run(frame, plan, target=None):
    return run_preprocessing(frame, infer_schema(frame, target=target), plan=plan)


def test_cap_outliers_clips_only_heavy_tailed_columns():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"tail": rng.lognormal(0, 2, 2000), "normal": rng.normal(size=2000)})
    out = _run(frame, ["cap_outliers"]).data
    assert out["tail"].max() <= frame["tail"].quantile(0.99) + 1e-9
    pd.testing.assert_series_equal(out["normal"], frame["normal"])


def test_rare_categories_are_grouped():
    rng = np.random.default_rng(2)
    segments = ["retail"] * 500 + ["sme"] * 480 + ["x1"] * 5 + ["x2"] * 5 + ["x3"] * 5 + ["x4"] * 5
    # A unique numeric column keeps deduplication from collapsing the rows.
    frame = pd.DataFrame({"segment": segments, "balance": rng.normal(size=1000)})
    out = _run(frame, ["group_rare_categories"]).data
    assert set(out["segment"]) == {"retail", "sme", "other"}


def test_impute_fills_everything_but_the_target():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "x": [1.0, np.nan, 3.0] * 30, "c": ["a", None, "a"] * 30, "y": [0, 1, 0] * 30,
        "balance": rng.normal(size=90),  # unique values, so no row is a duplicate
    })
    out = _run(frame, ["impute"], target="y").data
    assert len(out) == 90
    assert out.drop(columns="y").isna().sum().sum() == 0


def test_text_target_maps_the_positive_word_to_one():
    frame = pd.DataFrame({"x": range(100), "churn": ["No", "Yes", "no", "No"] * 25})
    result = _run(frame, [], target="churn")
    assert result.target_mapping == {"No": 0, "Yes": 1}
    assert result.data["churn"].mean() == pytest.approx(0.25)


def test_rows_with_missing_target_are_dropped():
    frame = pd.DataFrame({"x": range(10), "y": [0, 1, None, "N/A", 0, 1, 0, 1, 0, 1]})
    assert len(_run(frame, [], target="y").data) == 8


def test_case_variants_merge_to_the_most_common_spelling():
    frame = pd.DataFrame({"contract": ["Month-to-month"] * 6 + ["month-to-month"] * 2 + ["One year"] * 4 + ["ONE YEAR"]})
    out = _run(frame, []).data
    assert set(out["contract"]) == {"Month-to-month", "One year"}
