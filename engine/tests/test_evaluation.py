"""Tests for the evaluation harness.

Two things must hold: a healthy engine scores well, and a broken engine is
caught. The second matters more: a harness that always passes proves nothing.
"""

import dataclasses
import json

import pytest

import eda_engine.evaluation.harness as harness
from eda_engine.evaluation import BenchmarkReport, run_benchmark, score_dataset
from eda_engine.findings import FindingsConfig
from eda_engine.synth.credit import generate_credit_applications


@pytest.fixture(scope="module")
def dataset():
    return generate_credit_applications(n_rows=5000, seed=42)


@pytest.fixture(scope="module")
def healthy(dataset):
    return score_dataset(dataset)


@pytest.fixture(scope="module")
def small_benchmark():
    return run_benchmark(sizes=(2500, 5000), seeds=range(2))


# ---------------------------------------------------------------------------
# A healthy engine
# ---------------------------------------------------------------------------

def test_healthy_engine_passes_every_check(healthy):
    assert healthy.passed, [c for c in healthy.failed()]
    assert healthy.recall == 1.0
    assert healthy.noise_reported == {}


def test_every_stage_is_checked(healthy):
    assert {c.stage for c in healthy.checks} == {"schema", "preprocess", "profile", "findings", "pack"}
    assert len(healthy.checks) >= 20


def test_recall_is_reported_per_strength(healthy, dataset):
    strengths = {d["strength"] for d in dataset.manifest["planted_drivers"]}
    assert set(healthy.recall_by_strength) == strengths
    assert sum(healthy.planted_counts.values()) == len(dataset.manifest["planted_drivers"])


def test_benchmark_summary_and_gates(small_benchmark):
    rows = small_benchmark.summary()
    assert [r["n_rows"] for r in rows] == sorted(r["n_rows"] for r in rows)
    assert all(r["runs"] == 2 for r in rows)
    gates = small_benchmark.gates()
    assert all(g.passed for g in gates), [g for g in gates if not g.passed]


def test_benchmark_report_renders_and_serializes(small_benchmark):
    md = small_benchmark.to_markdown()
    assert md.startswith("| Rows |") and "false-discovery share" in md
    restored = json.loads(json.dumps(small_benchmark.to_dict()))
    assert len(restored["runs"]) == 4
    assert {g["name"] for g in restored["gates"]} == {g.name for g in small_benchmark.gates()}


# ---------------------------------------------------------------------------
# A broken engine must be caught
# ---------------------------------------------------------------------------

def test_no_false_discovery_control_is_caught(dataset):
    """Accepting everything (no FDR, no effect floor) floods the report with noise."""
    broken = BenchmarkReport([score_dataset(dataset, FindingsConfig(fdr=1.0, min_gini=0.0))])
    assert len(broken.runs[0].noise_reported) > 10
    failed = {g.name for g in broken.gates() if not g.passed}
    assert any("false-discovery" in name for name in failed)


def test_too_strict_engine_is_caught(dataset):
    """An engine that reports almost nothing has no false discoveries, but misses drivers."""
    broken = score_dataset(dataset, FindingsConfig(min_gini=0.5))
    assert broken.recall < 0.5
    assert "strong planted drivers found" in {c.name for c in broken.failed()}
    assert not all(g.passed for g in BenchmarkReport([broken]).gates())


def test_missing_domain_pack_is_caught(dataset, monkeypatch):
    def no_pack(prep, findings, config=None):
        from eda_engine.packs.fintech import FintechReport
        return FintechReport(findings, [], [], {}, [], [])
    monkeypatch.setattr(harness, "run_fintech_pack", no_pack)
    failed = {c.name for c in score_dataset(dataset).failed()}
    assert {"confounder labelled as lender decision", "planted proxy detected"} <= failed


def _over_budget_profiler(monkeypatch):
    real = harness.build_compact_profile

    def bloated(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), token_estimate=harness.PROFILE_BUDGET + 1)
    monkeypatch.setattr(harness, "build_compact_profile", bloated)


def test_profile_respects_tiny_budgets(dataset, monkeypatch):
    monkeypatch.setattr(harness, "PROFILE_BUDGET", 300)
    score = score_dataset(dataset)
    assert score.profile_tokens <= 300 and score.passed


def test_profile_over_budget_is_caught(dataset, monkeypatch):
    _over_budget_profiler(monkeypatch)
    failed = {c.stage for c in score_dataset(dataset).failed()}
    assert failed == {"profile"}


def test_gates_judge_rates_not_single_runs(healthy):
    """One noise finding among many is within a 5% false-discovery target; the
    same finding in a report of five is not."""
    many = BenchmarkReport([healthy])
    many.runs[0] = harness.RunScore(**{**healthy.__dict__, "n_findings": 40, "n_high_confidence": 30,
                                       "noise_reported": {"bureau_attr_001": "tentative"}})
    assert all(g.passed for g in many.gates())
    few = BenchmarkReport([harness.RunScore(**{**healthy.__dict__, "n_findings": 5, "n_high_confidence": 5,
                                               "noise_reported": {"bureau_attr_001": "high"}})])
    failed = {g.name for g in few.gates() if not g.passed}
    assert any("high-confidence" in name for name in failed)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_writes_reports_and_exits_cleanly(tmp_path, capsys):
    harness.main(["--sizes", "5000", "--seeds", "1", "--first-seed", "42", "--out", str(tmp_path)])
    assert (tmp_path / "benchmark.md").read_text(encoding="utf-8").startswith("| Rows |")
    data = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert data["runs"][0]["seed"] == 42
    assert "| Rows |" in capsys.readouterr().out


def test_cli_exits_nonzero_when_a_gate_fails(tmp_path, monkeypatch):
    _over_budget_profiler(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        harness.main(["--sizes", "5000", "--seeds", "1", "--out", str(tmp_path)])
    assert exit_info.value.code == 1


def test_missing_a_weak_driver_on_large_data_fails_the_recall_gate(healthy):
    """Every hard check can pass while a weak driver is missed; on 5,000+ rows that is still a failure."""
    missed = harness.RunScore(**{**healthy.__dict__, "recall_by_strength": {**healthy.recall_by_strength, "weak": 0.75}})
    assert missed.passed and missed.recall < 1.0
    failed = {g.name for g in BenchmarkReport([missed]).gates() if not g.passed}
    assert failed == {"every planted driver found at >= 5,000 rows"}
    small = harness.RunScore(**{**missed.__dict__, "n_rows": 1000})   # on small data it's expected
    assert all(g.passed for g in BenchmarkReport([small]).gates())
