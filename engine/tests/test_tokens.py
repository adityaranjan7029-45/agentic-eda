"""Tests for the token estimator.

The exact counts below were measured with the o200k_base tokenizer (the
encoding family the gpt-oss models use). CI can't run that tokenizer, so the
measurements are recorded here as fixed reference values.
"""

import math

import pytest

from eda_engine.tokens import estimate_tokens

# (text, exact o200k token count)
MEASURED = [
    ("rows=5025 cols=122 duplicate_rows=25", 11),
    ("format: name|role|missing%|distinct|summary|flags", 14),
    ("flags: LEAK=predicts target near-perfectly (likely leakage) sparse=>=60% missing "
     "nulltok=placeholder nulls spell=inconsistent spelling astext=stored as text "
     "badvals=some unparseable values", 45),
    ("bureau_attr_001..bureau_attr_064 [64 cols]|num|0..5|-|-|", 21),
    ("employment_type|cat|3|5|salaried:61%,self_employed:17%,contract:14% +2 more|nulltok", 30),
    ("months_since_last_delinquency|num|71|25|med 11 iqr 6..19 skew 0.2|sparse", 30),
    ("txn_cash_withdrawal_trend_3m|num|0|3327|med -0.0011 iqr -0.1..0.1 skew 0.0|", 40),
    ("savings_balance|num|0|4994|med 4.9k iqr 2.1k..12k skew 6.8|astext", 35),
    ("application_date|date|0|729|2024-01-01..2025-12-30|astext", 25),
    ("FeatureQ16Value|num|0|2000|med 20.3 iqr 10.2..39.1 skew 3.9|", 33),
    ("PaymentMethod|cat|3|4|electronic check:34%,mailed check:23%,"
     "bank transfer (automatic):22% +1 more|spell", 33),
    # Whitespace-padded table text, like v1's df.head().to_string():
    ("  application_id    customer_id  age  gender marital_status  dependents\n"
     "0    APP-0002536  CUST-00012345   42    male        married           1", 40),
    ("The quick brown fox jumps over the lazy dog while the analyst reviews credit default drivers.", 17),
]


@pytest.mark.parametrize("text,exact", MEASURED)
def test_estimate_never_undercounts(text, exact):
    """Undercounting is the dangerous direction: it lets a prompt exceed the
    provider's token limit. Over-estimating only wastes a little budget."""
    assert estimate_tokens(text) >= exact


def test_estimate_is_not_wildly_pessimistic_on_profile_text():
    profile_lines = MEASURED[:-2]
    estimated = estimate_tokens("\n".join(t for t, _ in profile_lines))
    exact = sum(n for _, n in profile_lines)
    assert estimated <= 1.35 * exact


def test_the_chars_divided_by_four_rule_would_undercount():
    """Documents why we don't use the common rule of thumb: on compact
    tabular text it misses roughly a third of the tokens."""
    profile_lines = MEASURED[:-2]
    text = "\n".join(t for t, _ in profile_lines)
    exact = sum(n for _, n in profile_lines)
    assert math.ceil(len(text) / 4) < 0.75 * exact


def test_empty_text_is_zero_tokens():
    assert estimate_tokens("") == 0
