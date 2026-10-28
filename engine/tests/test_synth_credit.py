"""Tests for the synthetic credit-application generator.

This dataset is the evaluation suite's answer key, so these tests check that
the ground truth the manifest claims actually holds in the generated data.
If the manifest says a column is a strong driver, the data must show it;
if it says a column is noise, the data must show nothing.
"""

import json

import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import pytest
from sklearn.metrics import roc_auc_score

from eda_engine.synth.credit import (
    HIDDEN_NULL_TOKENS,
    generate_credit_applications,
    main,
)

SEEDS = (3, 7, 11)
MIN_AUC = {"strong": 0.62, "moderate": 0.55, "weak": 0.545}
MAX_NOISE_AUC_DEVIATION = 0.045


@pytest.fixture(scope="module", params=SEEDS)
def dataset(request):
    return generate_credit_applications(n_rows=5000, seed=request.param)


@pytest.fixture(scope="module")
def clean_dataset():
    return generate_credit_applications(n_rows=2000, seed=5, messy=False)


def _deduped(dataset):
    return dataset.data.drop_duplicates()


def _oriented_auc(df, column, direction, target):
    x = (df[column] == 60).astype(float) if column == "loan_term_months" else df[column]
    ok = x.notna()
    scores = x[ok] if direction == "+" else -x[ok]
    return roc_auc_score(df.loc[ok, target], scores)


# ---------------------------------------------------------------------------
# Shape, determinism, manifest integrity
# ---------------------------------------------------------------------------

def test_is_wide_enough_for_the_100_column_target(dataset):
    assert dataset.data.shape[1] >= 120
    assert dataset.manifest["n_columns"] == dataset.data.shape[1]


def test_same_seed_is_identical_and_different_seed_differs():
    a = generate_credit_applications(500, seed=1).data
    b = generate_credit_applications(500, seed=1).data
    c = generate_credit_applications(500, seed=2).data
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_every_column_named_in_the_manifest_exists(dataset):
    m, cols = dataset.manifest, set(dataset.data.columns)
    named = (
        [m["target"]]
        + [d["column"] for d in m["planted_drivers"]]
        + [d["column"] for d in m["confounded_columns"]]
        + m["noise_columns"]
        + [c for cluster in m["correlated_noise_clusters"] for c in cluster]
        + m["leakage_columns"]
        + [d["column"] for d in m["informative_missingness"]]
        + m["protected_attributes"]
        + [d["column"] for d in m["proxy_features"]]
        + [c for pair in m["trivial_derived_pairs"] + m["redundant_pairs"] for c in pair]
        + m["id_columns"] + m["constant_columns"] + m["mostly_missing_columns"]
        + m["high_cardinality_columns"] + m["date_columns"]
    )
    assert set(named) <= cols, set(named) - cols


def test_planted_drivers_are_never_also_listed_as_noise(dataset):
    planted = {d["column"] for d in dataset.manifest["planted_drivers"]}
    assert planted.isdisjoint(dataset.manifest["noise_columns"])


def test_target_rate_matches_request(dataset):
    assert 0.10 <= dataset.data[dataset.manifest["target"]].mean() <= 0.14


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        generate_credit_applications(n_rows=10)
    with pytest.raises(ValueError):
        generate_credit_applications(target_rate=0.9)


# ---------------------------------------------------------------------------
# The planted ground truth must actually be present in the data
# ---------------------------------------------------------------------------

def test_planted_drivers_have_the_claimed_direction_and_strength(dataset):
    df, m = _deduped(dataset), dataset.manifest
    for driver in m["planted_drivers"]:
        auc = _oriented_auc(df, driver["column"], driver["direction"], m["target"])
        assert auc >= MIN_AUC[driver["strength"]], (driver, round(auc, 3))


def test_noise_columns_carry_no_signal(dataset):
    df, m = _deduped(dataset), dataset.manifest
    for column in m["noise_columns"]:
        auc = _oriented_auc(df, column, "+", m["target"])
        assert abs(auc - 0.5) <= MAX_NOISE_AUC_DEVIATION, (column, round(auc, 3))


def test_weakest_driver_beats_strongest_noise(dataset):
    """The answer key is only fair if real signal is separable from noise."""
    df, m = _deduped(dataset), dataset.manifest
    weakest = min(_oriented_auc(df, d["column"], d["direction"], m["target"]) for d in m["planted_drivers"])
    noisiest = max(abs(_oriented_auc(df, c, "+", m["target"]) - 0.5) + 0.5 for c in m["noise_columns"])
    assert weakest > noisiest


def test_leakage_columns_almost_perfectly_predict_the_target(dataset):
    df, m = _deduped(dataset), dataset.manifest
    for column in m["leakage_columns"]:
        assert roc_auc_score(df[m["target"]], df[column]) >= 0.95


def test_missingness_is_informative(dataset):
    df, m = _deduped(dataset), dataset.manifest
    col = m["informative_missingness"][0]["column"]
    missing = df[col].isna()
    assert df.loc[missing, m["target"]].mean() < df.loc[~missing, m["target"]].mean()


def test_postal_area_is_a_proxy_for_age(dataset):
    df = _deduped(dataset)
    senior_share = df.assign(senior=df["age"] >= 60).groupby("postal_area")["senior"].mean()
    assert senior_share.max() > 4 * senior_share.min()


def test_gender_has_no_planted_effect(dataset):
    df, m = _deduped(dataset), dataset.manifest
    rates = df.groupby("gender")[m["target"]].mean()
    assert abs(rates["female"] - rates["male"]) < 0.03


def test_confounded_column_tracks_its_source(dataset):
    df = _deduped(dataset)
    assert df["interest_rate"].corr(df["credit_score"]) < -0.8


def test_correlated_noise_clusters_are_actually_correlated(dataset):
    df = _deduped(dataset)
    for cluster in dataset.manifest["correlated_noise_clusters"]:
        corr = df[cluster].corr(method="spearman").to_numpy()
        off_diagonal = corr[~np.eye(len(cluster), dtype=bool)]
        assert off_diagonal.mean() > 0.5


def test_trivial_and_redundant_pairs(dataset):
    df = _deduped(dataset)
    for a, b in dataset.manifest["trivial_derived_pairs"] + dataset.manifest["redundant_pairs"]:
        assert df[a].corr(df[b]) > 0.95


# ---------------------------------------------------------------------------
# Data-quality problems the preprocessing must handle
# ---------------------------------------------------------------------------

def test_messy_mode_contains_the_documented_problems(dataset):
    df, m = dataset.data, dataset.manifest
    issues = m["data_quality_issues"]
    assert df.duplicated().sum() == issues["duplicate_rows"] > 0
    assert df["employment_type"].isin(HIDDEN_NULL_TOKENS).any()
    assert df["savings_balance"].str.startswith("$").all()
    raw = df["home_ownership"].nunique()
    normalized = df["home_ownership"].str.strip().str.lower().nunique()
    assert raw > normalized == 3
    assert df["data_source_version"].nunique() == 1
    assert df["secondary_applicant_income"].isna().mean() > 0.8
    assert df["application_id"].nunique() / len(df) > 0.99


def test_clean_mode_has_no_quality_problems(clean_dataset):
    df = clean_dataset.data
    assert df.duplicated().sum() == 0
    assert not df["employment_type"].isin(HIDDEN_NULL_TOKENS).any()
    assert pd.api.types.is_numeric_dtype(df["savings_balance"])
    assert clean_dataset.manifest["data_quality_issues"]["duplicate_rows"] == 0


def test_cli_writes_csv_and_manifest(tmp_path):
    main(["--rows", "300", "--seed", "9", "--out", str(tmp_path)])
    df = pd.read_csv(tmp_path / "credit_applications.csv")
    manifest = json.loads((tmp_path / "credit_applications.manifest.json").read_text())
    assert df.shape[0] == manifest["n_rows"]
    assert df.shape[1] == manifest["n_columns"]