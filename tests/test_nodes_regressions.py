"""Regression tests for bugs found in the preprocessing nodes."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes import PreprocessingPlan, type_conversion_node  # noqa: E402


def _run_type_conversion(df):
    plan = PreprocessingPlan(steps=["type_conversion"], reasoning="test")
    return type_conversion_node({"df": df, "plan": plan})["df"]


def test_numeric_strings_are_converted():
    """pandas 3 made to_numeric(errors='ignore') raise; the old bare except
    swallowed it, so numeric strings were silently never converted and later
    got label-encoded as categories."""
    out = _run_type_conversion(pd.DataFrame({"amount": ["1000", "250.5", "75"]}))
    assert pd.api.types.is_numeric_dtype(out["amount"])
    assert out["amount"].tolist() == [1000.0, 250.5, 75.0]


def test_real_text_columns_are_left_alone():
    out = _run_type_conversion(pd.DataFrame({"name": ["alice", "bob", "carol"]}))
    assert out["name"].tolist() == ["alice", "bob", "carol"]


def test_mixed_column_is_not_partially_converted():
    """All-or-nothing: one non-numeric value keeps the whole column as text,
    rather than coercing it to NaN and losing data."""
    out = _run_type_conversion(pd.DataFrame({"code": ["12", "34", "ab"]}))
    assert out["code"].tolist() == ["12", "34", "ab"]
