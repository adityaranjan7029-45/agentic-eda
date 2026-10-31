"""Tests for the findings engine.

The synthetic credit dataset's manifest is the answer key: every planted
driver must be found, no noise column may be reported, and the known traps
(leakage, near-duplicate columns, a confounder) must be handled correctly.
"""

import json

import numpy as np
import pandas as pd
import pytest

from eda_engine.findings import FindingsConfig, benjamini_hochberg, discover_findings
from eda_engine.preprocess import run_preprocessing
from eda_engine.schema import infer_schema
from eda_engine.synth.credit import generate_credit_applications


def _analyse(frame, target=None, config=None):
    return discover_findings(run_preprocessing(frame, infer_schema(frame, target=target)), config)


@pytest.fixture(scope="module", params=[3, 11, 42])
def credit(request):
    dataset = generate_credit_applications(n_rows=5000, seed=request.param)
    return dataset, _analyse(dataset.data, target="default_flag")


# ---------------------------------------------------------------------------
# Answer key: recall, precision, traps
# ---------------------------------------------------------------------------

def test_every_planted_driver_is_found(credit):
    dataset, report = credit
    found = {f.column for f in report.drivers()}
    planted = {d["column"] for d in dataset.manifest["planted_drivers"]}
    assert planted <= found, planted - found


def test_no_noise_column_is_reported(credit):
    dataset, report = credit
    reported = {f.column for f in report.findings}
    assert not reported & set(dataset.manifest["noise_columns"])


def test_planted_directions_are_correct(credit):
    dataset, report = credit
    by_column = {f.column: f for f in report.findings}
    for driver in dataset.manifest["planted_drivers"]:
        assert by_column[driver["column"]].direction == driver["direction"], driver


def test_leakage_columns_never_become_findings(credit):
    dataset, report = credit
    reported = {f.column for f in report.findings}
    assert not reported & set(dataset.manifest["leakage_columns"])
    critical = [n for n in report.quality_notes if n["severity"] == "critical"]
    assert {n["column"] for n in critical} == set(dataset.manifest["leakage_columns"])


def test_confounder_is_linked_not_reported_as_an_independent_cause(credit):
    """interest_rate is priced from credit_score. Statistics alone can't say
    which one is the cause, but they must be flagged as the same signal."""
    _, report = credit
    by_column = {f.column: f for f in report.findings}
    pair = {"interest_rate", "credit_score"}
    secondary = [by_column[c] for c in pair if by_column[c].kind == "secondary_driver"]
    assert len(secondary) == 1
    assert secondary[0].related_to in pair
    assert abs(secondary[0].correlation) > 0.85


def test_protected_attribute_with_no_planted_effect_is_not_a_finding(credit):
    _, report = credit
    assert "gender" not in {f.column for f in report.findings}


def test_near_duplicate_columns_are_reported_as_structure_not_insight(credit):
    dataset, report = credit
    pairs = {frozenset(p["columns"]) for p in report.redundant_pairs}
    for a, b in dataset.manifest["trivial_derived_pairs"] + dataset.manifest["redundant_pairs"]:
        assert frozenset((a, b)) in pairs


def test_informative_missingness_is_found_and_linked_to_its_source(credit):
    _, report = credit
    by_column = {f.column: f for f in report.findings}
    flag = by_column["months_since_last_delinquency__missing"]
    assert flag.direction == "-"  # missing = never delinquent = lower risk
    assert flag.related_to == "num_delinquencies_2y"


# ---------------------------------------------------------------------------
# Finding contents
# ---------------------------------------------------------------------------

def test_findings_are_ranked_by_effect_size(credit):
    _, report = credit
    ginis = [f.gini for f in report.findings]
    assert ginis == sorted(ginis, reverse=True)


def test_every_finding_is_significant_after_fdr_and_meaningful(credit):
    _, report = credit
    for f in report.findings:
        assert f.q_value <= 0.05 and f.p_value <= f.q_value
        assert f.gini >= 0.10


def test_summaries_carry_real_numbers_consistent_with_direction(credit):
    _, report = credit
    for f in report.findings:
        if "top_quintile_rate" in f.evidence:
            top, bottom = f.evidence["top_quintile_rate"], f.evidence["bottom_quintile_rate"]
            assert (top > bottom) == (f.direction == "+"), f.column
            assert f"{100 * top:.1f}%" in f.summary and f"{100 * bottom:.1f}%" in f.summary
        elif "rates" in f.evidence:
            for rate in (max(f.evidence["rates"].values()), min(f.evidence["rates"].values())):
                assert f"{100 * rate:.1f}%" in f.summary, f.column


def test_report_accounts_for_every_tested_column(credit):
    _, report = credit
    assert report.n_tested == len(report.findings) + report.not_reported["not_significant"] + report.not_reported["effect_too_small"]


def test_report_serializes_to_json(credit):
    _, report = credit
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["target"] == "default_flag"
    assert len(restored["findings"]) == len(report.findings)


# ---------------------------------------------------------------------------
# False discovery control
# ---------------------------------------------------------------------------

def test_pure_noise_produces_no_findings():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({f"x{i}_value": rng.normal(size=3000) for i in range(100)})
    frame["y"] = rng.integers(0, 2, 3000)
    report = _analyse(frame, target="y")
    assert report.n_tested == 100
    assert report.findings == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_false_discovery_control_matters_on_small_wide_data(seed):
    """300 rows x 150 pure-noise columns: by chance, several columns show
    Gini > 0.10 with p < 0.05 (4-11 per dataset when uncorrected). The
    Benjamini-Hochberg correction is what keeps them out."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i}_value": rng.normal(size=300) for i in range(150)})
    frame["y"] = rng.integers(0, 2, 300)
    assert _analyse(frame, target="y").findings == []


def test_benjamini_hochberg_matches_hand_computed_values():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005, 0.01, 0.03, 0.04 -> raw adj: 0.02, 0.02, 0.04, 0.04
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg([]) == []


def test_min_gini_is_configurable():
    dataset = generate_credit_applications(n_rows=3000, seed=5)
    loose = _analyse(dataset.data, "default_flag", FindingsConfig(min_gini=0.0))
    strict = _analyse(dataset.data, "default_flag", FindingsConfig(min_gini=0.3))
    assert len(strict.findings) < len(loose.findings)
    assert all(f.gini >= 0.3 for f in strict.findings)


# ---------------------------------------------------------------------------
# Categorical drivers, small classes, no-target mode
# ---------------------------------------------------------------------------

def test_categorical_driver_reports_best_and_worst_groups():
    rng = np.random.default_rng(1)
    contract = rng.choice(["month-to-month", "one year", "two year"], 3000)
    rate = pd.Series(contract).map({"month-to-month": 0.4, "one year": 0.15, "two year": 0.05})
    frame = pd.DataFrame({"contract": contract, "noise": rng.normal(size=3000),
                          "churn": (rng.random(3000) < rate).astype(int)})
    report = _analyse(frame, target="churn")
    assert [f.column for f in report.findings] == ["contract"]
    finding = report.findings[0]
    assert finding.direction == "mixed"
    assert "month-to-month" in finding.summary and "two year" in finding.summary


def test_too_few_positives_is_untestable_not_a_crash():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x_value": rng.normal(size=500), "y": [1] * 5 + [0] * 495})
    report = _analyse(frame, target="y")
    assert report.findings == []
    assert report.not_reported["untestable"] == 1


def test_no_target_mode_reports_real_correlations_not_duplicates():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    frame = pd.DataFrame({
        "spend": a,
        "spend_copy": a * 2 + 1,                           # near-duplicate: structure, not insight
        "income": 0.6 * a + 0.8 * rng.normal(size=2000),   # real relationship
        "unrelated": rng.normal(size=2000),
    })
    report = _analyse(frame)
    pairs = {frozenset(f.evidence["columns"]) for f in report.findings}
    assert frozenset({"spend", "income"}) in pairs
    assert frozenset({"spend", "spend_copy"}) not in pairs
    assert {"spend", "spend_copy"} in [set(p["columns"]) for p in report.redundant_pairs]
    assert not any("unrelated" in p for p in pairs)


# ---------------------------------------------------------------------------
# Confidence labels (added in step 8, from the benchmark)
# ---------------------------------------------------------------------------

def test_strong_findings_are_high_confidence(credit):
    _, report = credit
    by_column = {f.column: f for f in report.findings}
    assert by_column["credit_score"].confidence == "high"
    assert by_column["debt_to_income"].confidence == "high"
    for f in report.findings:
        assert f.confidence == ("high" if f.q_value <= 0.001 else "tentative")
        assert f.to_dict()["confidence"] == f.confidence


def test_confidence_threshold_is_configurable():
    dataset = generate_credit_applications(n_rows=3000, seed=5)
    strict = _analyse(dataset.data, "default_flag", FindingsConfig(high_confidence_q=0.0))
    assert strict.findings and all(f.confidence == "tentative" for f in strict.findings)


def test_borderline_findings_on_small_data_are_tentative():
    """On 1,000 rows, weak effects only just pass the FDR gate: they are reported, but labelled."""
    dataset = generate_credit_applications(n_rows=1000, seed=0)
    report = _analyse(dataset.data, "default_flag")
    assert {f.confidence for f in report.findings} == {"high", "tentative"}
