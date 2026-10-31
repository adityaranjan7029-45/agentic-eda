"""Tests for the fintech / credit-risk pack, checked against the synthetic manifest."""

import json

import numpy as np
import pandas as pd
import pytest

from eda_engine.findings import discover_findings
from eda_engine.packs import (
    PACKS,
    FintechConfig,
    information_value,
    iv_band,
    ks_statistic,
    lender_decision_columns,
    protected_attributes,
    run_fintech_pack,
)
from eda_engine.preprocess import run_preprocessing
from eda_engine.schema import ColumnRole, infer_schema
from eda_engine.synth.credit import generate_credit_applications


def _run(frame, target=None, config=None):
    prep = run_preprocessing(frame, infer_schema(frame, target=target))
    return run_fintech_pack(prep, discover_findings(prep), config)


@pytest.fixture(scope="module", params=[7, 42])
def credit(request):
    dataset = generate_credit_applications(n_rows=5000, seed=request.param)
    return dataset, _run(dataset.data, target="default_flag")


# ---------------------------------------------------------------------------
# IV / WoE / KS
# ---------------------------------------------------------------------------

def test_planted_driver_iv_matches_planted_strength(credit):
    dataset, report = credit
    iv = {v.column: v.iv for v in report.variable_strength}
    minimum = {"strong": 0.3, "moderate": 0.1, "weak": 0.02}   # strong / medium / weak band
    for d in dataset.manifest["planted_drivers"]:
        assert iv[d["column"]] >= minimum[d["strength"]], (d, iv[d["column"]])


def test_noise_columns_have_weak_or_useless_iv(credit):
    dataset, report = credit
    for v in report.variable_strength:
        if v.column in dataset.manifest["noise_columns"]:
            assert v.band in ("useless", "weak"), (v.column, v.iv)


def test_variables_are_ranked_by_iv(credit):
    _, report = credit
    ivs = [v.iv for v in report.variable_strength]
    assert ivs == sorted(ivs, reverse=True)


def test_iv_bands_follow_industry_cutoffs():
    assert [iv_band(x) for x in (0.0, 0.019, 0.02, 0.09, 0.1, 0.29, 0.3, 0.49, 0.5, 3.0)] == [
        "useless", "useless", "weak", "weak", "medium", "medium", "strong", "strong", "suspicious", "suspicious"]


def test_iv_matches_hand_computed_value():
    # bin a: 80 goods, 20 bads; bin b: 20 goods, 80 bads (smoothing off)
    x = pd.Series(["a"] * 100 + ["b"] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    iv, woe = information_value(x, y, ColumnRole.CATEGORICAL, FintechConfig(woe_smoothing=0.0))
    expected = 2 * (0.8 - 0.2) * np.log(0.8 / 0.2)
    assert iv == pytest.approx(expected)
    by_bin = {w["bin"]: w for w in woe}
    assert by_bin["a"]["woe"] == pytest.approx(np.log(4), abs=1e-4)   # more goods -> positive WoE
    assert by_bin["b"]["bad_rate"] == 0.8


def test_missing_values_get_their_own_bin():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=1000))
    y = pd.Series(rng.integers(0, 2, 1000))
    x[:100] = np.nan
    _, woe = information_value(x, y, ColumnRole.NUMERIC, FintechConfig())
    missing = [w for w in woe if w["bin"] == "missing"]
    assert len(missing) == 1 and missing[0]["count"] == 100
    assert sum(w["count"] for w in woe) == 1000


def test_ks_is_one_for_perfect_separation_and_small_for_noise():
    y = pd.Series([0] * 50 + [1] * 50)
    assert ks_statistic(pd.Series(range(100), dtype=float), y) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    noise_y = pd.Series(rng.integers(0, 2, 4000))
    assert ks_statistic(pd.Series(rng.normal(size=4000)), noise_y) < 0.06
    assert ks_statistic(pd.Series([1.0, 2.0]), pd.Series([0, 1])) is None


def test_suspiciously_high_iv_raises_a_warning():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    # noise keeps single-column AUC below the 0.95 leakage screen, so it reaches the pack
    frame = pd.DataFrame({"recovery_score": y + rng.normal(0, 0.6, 2000),
                          "income_value": rng.normal(size=2000), "default_flag": y})
    report = _run(frame, target="default_flag")
    iv = {v.column: v for v in report.variable_strength}
    assert iv["recovery_score"].band == "suspicious"
    assert any(n["column"] == "recovery_score" and n["severity"] == "warning" for n in report.notes)


# ---------------------------------------------------------------------------
# Lender-decision rule
# ---------------------------------------------------------------------------

def test_lender_decision_names_are_recognised():
    cols = ["interest_rate", "apr", "loan_grade", "sub_grade", "approved_amount", "offered_limit",
            "credit_score", "grade_point", "upgrade_count", "debt_to_income"]
    assert lender_decision_columns(cols) == [
        "interest_rate", "apr", "loan_grade", "sub_grade", "approved_amount", "offered_limit", "grade_point"]


def test_interest_rate_is_demoted_and_credit_score_becomes_the_driver(credit):
    """The general engine can only say interest_rate and credit_score are the
    same signal. The pack knows the lender sets the rate from the score."""
    _, report = credit
    by_column = {f.column: f for f in report.findings.findings}
    assert by_column["interest_rate"].kind == "lender_decision"
    assert "lender" in by_column["interest_rate"].summary
    assert by_column["credit_score"].kind == "driver"
    assert by_column["credit_score"].related_to is None
    assert "interest_rate" not in {f.column for f in report.findings.drivers()}


def test_pack_does_not_lose_or_invent_findings(credit):
    dataset, report = credit
    base = discover_findings(run_preprocessing(dataset.data, infer_schema(dataset.data, target="default_flag")))
    assert [f.column for f in report.findings.findings] == [f.column for f in base.findings]
    planted = {d["column"] for d in dataset.manifest["planted_drivers"]}
    assert planted <= {f.column for f in report.findings.drivers()}


# ---------------------------------------------------------------------------
# Fair lending
# ---------------------------------------------------------------------------

def test_protected_attribute_names_are_recognised():
    cols = ["age", "applicant_age", "gender", "sex", "marital_status", "race", "nationality",
            "disability_flag", "page_views", "usage_days", "sextant", "credit_score"]
    assert protected_attributes(cols) == {
        "age": "age", "applicant_age": "age", "gender": "gender", "sex": "gender",
        "marital_status": "marital_status", "race": "race_ethnicity",
        "nationality": "national_origin", "disability_flag": "disability"}


def test_planted_age_proxy_is_detected(credit):
    dataset, report = credit
    flagged = {(p["feature"], p["protected"]) for p in report.proxies}
    for proxy in dataset.manifest["proxy_features"]:
        assert (proxy["column"], proxy["proxies_for"]) in flagged
    for p in report.proxies:
        assert p["strength"] >= 0.2
        assert p["feature"] not in report.protected_attributes


def test_proxies_are_only_plausible_columns(credit):
    """Only postal_area (planted) and employment_type ('retired' is age-related) are associated with age."""
    _, report = credit
    assert {p["feature"] for p in report.proxies} <= {"postal_area", "employment_type"}


def test_protected_attributes_are_noted(credit):
    _, report = credit
    noted = {n["column"] for n in report.notes if "protected attribute" in n["message"]}
    assert {"age", "gender"} <= noted


def test_proxy_threshold_is_configurable():
    dataset = generate_credit_applications(n_rows=3000, seed=5)
    strict = _run(dataset.data, "default_flag", FintechConfig(proxy_threshold=0.99))
    assert strict.proxies == []


def test_proxy_measures_cover_every_type_pair():
    rng = np.random.default_rng(4)
    age = rng.integers(20, 70, 3000)
    frame = pd.DataFrame({
        "age": age,
        "years_at_address": age - 18 + rng.normal(0, 3, 3000),                       # numeric-numeric
        "life_stage": np.where(age > 55, "senior", np.where(age > 35, "mid", "young")),  # categorical-numeric
        "gender": rng.choice(["f", "m"], 3000),
        "title_code": None,
        "y": rng.integers(0, 2, 3000),
    })
    frame["title_code"] = np.where(frame["gender"] == "f", rng.choice(["a", "b"], 3000, p=[0.9, 0.1]), "c")
    report = _run(frame, target="y")
    measures = {(p["feature"], p["protected"]): p["measure"] for p in report.proxies}
    assert measures[("years_at_address", "age")] == "spearman"
    assert measures[("life_stage", "age")] == "eta"
    assert measures[("title_code", "gender")] == "cramers_v"


# ---------------------------------------------------------------------------
# Edge cases and plumbing
# ---------------------------------------------------------------------------

def test_no_target_skips_iv_but_still_checks_fairness():
    rng = np.random.default_rng(5)
    age = rng.integers(20, 70, 1000)
    frame = pd.DataFrame({"age": age, "tenure_years": age / 2 + rng.normal(0, 1, 1000),
                          "spend": rng.normal(size=1000)})
    report = _run(frame)
    assert report.variable_strength == []
    assert any("skipped" in n["message"] for n in report.notes)
    assert {p["feature"] for p in report.proxies} == {"tenure_years"}


def test_report_serializes_to_json(credit):
    _, report = credit
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["lender_decision_columns"] == ["interest_rate"]
    assert restored["variable_strength"][0]["woe"]


def test_pack_is_registered():
    assert PACKS["fintech"] is run_fintech_pack


def test_protected_attributes_are_not_reported_as_proxies_of_each_other():
    """age and marital_status are related, but both are already protected:
    a proxy warning is for ordinary features that smuggle protected information in."""
    rng = np.random.default_rng(6)
    age = rng.integers(20, 70, 3000)
    frame = pd.DataFrame({
        "age": age,
        "marital_status": np.where(age + rng.normal(0, 5, 3000) > 40, "married", "single"),
        "spend": rng.normal(size=3000),
        "y": rng.integers(0, 2, 3000),
    })
    report = _run(frame, target="y")
    assert set(report.protected_attributes) == {"age", "marital_status"}
    assert report.proxies == []
