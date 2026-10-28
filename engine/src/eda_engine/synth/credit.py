"""Synthetic credit-application dataset with planted, documented ground truth.

Why this exists:
  1. It's the demo dataset: wide (120+ columns), realistic, and safe to commit
     (Kaggle competition data like Home Credit can't be redistributed).
  2. It's the evaluation suite's answer key. Every pattern is planted on
     purpose and recorded in a manifest, so we can measure how many real
     drivers the pipeline finds and how many false findings it reports.

What gets planted (all listed in the manifest):
  - Real default drivers of known direction and strength.
  - Pure-noise columns, including correlated clusters (multicollinearity).
  - Leakage columns that are only known AFTER the outcome.
  - Informative missingness, a proxy for a protected attribute, trivial and
    redundant column pairs, and typical data-quality problems (duplicates,
    hidden null tokens, currency strings, inconsistent category spellings).

Usage:
    eda-synth-credit --rows 5000 --seed 42 --out data/samples
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default_flag"
N_BUREAU = 64
BUREAU_CLUSTER_SIZE = 8
N_BUREAU_CLUSTERS = 4  # bureau_attr_001..032 form 4 correlated clusters of 8
TXN_CATEGORIES = ("groceries", "travel", "utilities", "entertainment", "cash_withdrawal")
TXN_STATS = ("avg_monthly", "max_monthly", "count_monthly", "trend_3m")
N_POSTAL_AREAS = 12
HIDDEN_NULL_TOKENS = ("N/A", "unknown", "?", " ")

# (column, direction, strength, logit weight). Direction "+" means higher
# values raise default risk. Weights apply to the z-scored column (or to the
# 0/1 indicator for loan_term_months == 60).
PLANTED_DRIVERS = (
    ("credit_score", "-", "strong", 1.10),
    ("debt_to_income", "+", "strong", 0.90),
    ("revolving_utilization", "+", "moderate", 0.65),
    ("num_delinquencies_2y", "+", "moderate", 0.65),
    ("txn_cash_withdrawal_trend_3m", "+", "moderate", 0.65),
    ("loan_term_months", "+", "moderate", 1.00),
    ("num_credit_inquiries_6m", "+", "weak", 0.55),
    ("employment_years", "-", "weak", 0.55),
    # Needles in the wide bureau block. Both are deliberately OUTSIDE the
    # correlated clusters (001-032): a needle inside a cluster leaks its
    # signal into its cluster-mates, which would then be mislabeled as noise.
    ("bureau_attr_051", "+", "weak", 0.55),
    ("bureau_attr_042", "-", "weak", 0.55),
)


@dataclass
class GeneratedDataset:
    data: pd.DataFrame
    manifest: dict


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logit_without_intercept: np.ndarray, target_rate: float) -> float:
    """Bisection on the intercept so the mean default probability hits target_rate."""
    lo, hi = -15.0, 15.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _sigmoid(logit_without_intercept + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _demographics(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    age = np.clip(rng.normal(42, 12, n), 21, 75).round().astype(int)
    gender = rng.choice(["female", "male", "non_binary"], n, p=[0.48, 0.49, 0.03])
    marital_status = rng.choice(["single", "married", "divorced", "widowed"], n, p=[0.38, 0.45, 0.13, 0.04])
    dependents = np.clip(rng.poisson(1.0, n), 0, 6)
    education = rng.choice(
        ["high_school", "bachelors", "masters", "doctorate", "other"], n, p=[0.35, 0.38, 0.17, 0.03, 0.07]
    )

    # Proxy for a protected attribute: applicants aged 60+ cluster in three
    # postal areas. Postal area itself has no planted effect on default, but
    # a fairness review must flag it as a stand-in for age.
    areas = np.array([f"PA-{i:02d}" for i in range(1, N_POSTAL_AREAS + 1)])
    p_general = np.full(N_POSTAL_AREAS, 1.0)
    p_general[-3:] = 0.3
    p_senior = np.full(N_POSTAL_AREAS, 0.2)
    p_senior[-3:] = 6.0
    senior = age >= 60
    postal_area = np.where(
        senior,
        rng.choice(areas, n, p=p_senior / p_senior.sum()),
        rng.choice(areas, n, p=p_general / p_general.sum()),
    )

    employment_type = rng.choice(
        ["salaried", "self_employed", "contract", "unemployed"], n, p=[0.62, 0.18, 0.15, 0.05]
    )
    employment_type = np.where((age >= 62) & (rng.random(n) < 0.7), "retired", employment_type)
    employment_years = np.minimum(age - 20, rng.gamma(2.0, 4.0, n)).round(1)
    employment_years = np.where(np.isin(employment_type, ["retired", "unemployed"]), 0.0, employment_years)
    home_ownership = rng.choice(["rent", "mortgage", "own"], n, p=[0.42, 0.40, 0.18])

    return {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education": education,
        "postal_area": postal_area,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
    }


def _financials(rng: np.random.Generator, n: int, demo: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    education_uplift = pd.Series(demo["education"]).map(
        {"high_school": 0.0, "bachelors": 0.25, "masters": 0.4, "doctorate": 0.55, "other": 0.05}
    ).to_numpy()
    annual_income = np.exp(rng.normal(10.75 + education_uplift, 0.45, n)).round(-2)

    credit_score = np.clip(
        rng.normal(680, 65, n) + 1.2 * (demo["employment_years"] - demo["employment_years"].mean()), 300, 850
    ).round().astype(int)

    debt_to_income = np.clip(rng.beta(2.0, 5.0, n) * 0.9, 0.01, 0.85).round(4)
    monthly_debt = (debt_to_income * annual_income / 12).round(2)

    loan_amount = np.clip(np.exp(rng.normal(9.6, 0.6, n)) * (annual_income / 60000) ** 0.5, 1000, 100000).round(-2)
    loan_term_months = rng.choice([36, 60], n, p=[0.65, 0.35])
    # Priced off the credit score, so it correlates with default only
    # *through* credit_score -- a realistic confounder, not a planted driver.
    interest_rate = np.clip(
        4.0 + (850 - credit_score) / 22 + 1.5 * (loan_term_months == 60) + rng.normal(0, 0.8, n), 3.0, 36.0
    ).round(2)

    num_delinquencies_2y = rng.poisson(0.35, n)
    months_since_last_delinquency = np.where(
        num_delinquencies_2y > 0, rng.integers(1, 25, n) // np.maximum(num_delinquencies_2y, 1) + 1, np.nan
    )
    total_credit_limit = np.exp(rng.normal(10.2, 0.7, n)).round(-2)

    return {
        "annual_income": annual_income,
        "monthly_income": (annual_income / 12).round(2),
        "monthly_debt": monthly_debt,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "num_open_accounts": rng.poisson(6, n),
        "num_credit_inquiries_6m": rng.poisson(1.2, n),
        "revolving_utilization": np.clip(rng.beta(2.0, 3.0, n) * 1.1, 0, 1.2).round(3),
        "num_delinquencies_2y": num_delinquencies_2y,
        "months_since_last_delinquency": months_since_last_delinquency,
        "bankruptcies": rng.binomial(1, 0.04, n),
        "total_credit_limit": total_credit_limit,
        "total_revolving_limit": (total_credit_limit * 0.97 + rng.normal(0, 300, n)).round(-1),
        "savings_balance": np.exp(rng.normal(8.5, 1.3, n)).round(2),
        "checking_balance": np.exp(rng.normal(7.4, 1.0, n)).round(2),
        "secondary_applicant_income": np.where(
            rng.random(n) < 0.15, np.exp(rng.normal(10.5, 0.5, n)).round(-2), np.nan
        ),
    }


def _application_meta(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    start = np.datetime64("2024-01-01")
    application_date = (start + rng.integers(0, 730, n).astype("timedelta64[D]")).astype(str)
    employer_ids = np.minimum(rng.zipf(1.6, n), 400)
    return {
        "application_date": application_date,
        "channel": rng.choice(["online", "branch", "broker"], n, p=[0.6, 0.25, 0.15]),
        "product_type": rng.choice(
            ["personal_loan", "auto_loan", "debt_consolidation", "home_improvement"], n, p=[0.4, 0.2, 0.3, 0.1]
        ),
        "employer_name": np.array([f"employer_{i:04d}" for i in employer_ids]),
        "data_source_version": np.full(n, "v3"),
    }


def _bureau_attributes(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """64 credit-bureau style attributes: mostly noise, with correlated
    clusters (to exercise multicollinearity handling) and two planted needles
    hidden among them (to exercise wide-dataset search)."""
    cols: dict[str, np.ndarray] = {}
    factors = rng.normal(size=(N_BUREAU_CLUSTERS, n))
    for i in range(1, N_BUREAU + 1):
        name = f"bureau_attr_{i:03d}"
        cluster = (i - 1) // BUREAU_CLUSTER_SIZE
        base = rng.normal(size=n)
        if cluster < N_BUREAU_CLUSTERS:
            base = 0.85 * factors[cluster] + np.sqrt(1 - 0.85**2) * base
        kind = i % 3
        if kind == 0:
            cols[name] = np.exp(0.6 * base + 7).round(2)  # balance-like
        elif kind == 1:
            cols[name] = np.clip(np.round(2 + 1.5 * base), 0, None)  # count-like
        else:
            cols[name] = _sigmoid(base).round(4)  # ratio-like
    return cols


def _transaction_features(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {}
    for category in TXN_CATEGORIES:
        level = rng.normal(5.5, 0.6, n)
        cols[f"txn_{category}_avg_monthly"] = np.exp(level).round(2)
        cols[f"txn_{category}_max_monthly"] = np.exp(level + np.abs(rng.normal(0.5, 0.2, n))).round(2)
        cols[f"txn_{category}_count_monthly"] = rng.poisson(np.exp(level - 3.5))
        cols[f"txn_{category}_trend_3m"] = rng.normal(0, 0.15, n).round(4)
    return cols


def _add_target(rng: np.random.Generator, df: pd.DataFrame, target_rate: float) -> np.ndarray:
    logit = np.zeros(len(df))
    for column, direction, _strength, weight in PLANTED_DRIVERS:
        if column == "loan_term_months":
            signal = (df[column].to_numpy() == 60).astype(float)
        else:
            signal = _zscore(df[column].to_numpy(dtype=float))
        logit += (weight if direction == "+" else -weight) * signal
    intercept = _calibrate_intercept(logit, target_rate)
    return (rng.random(len(df)) < _sigmoid(logit + intercept)).astype(int)


def _add_leakage(rng: np.random.Generator, y: np.ndarray) -> dict[str, np.ndarray]:
    """Columns only knowable after the outcome. Any model or 'insight' built
    on them is cheating -- the pipeline must flag them, not celebrate them."""
    n = len(y)
    dpd_good = np.where(rng.random(n) < 0.05, rng.integers(1, 30, n), 0)
    days_past_due_current = np.where(y == 1, rng.integers(31, 181, n), dpd_good)
    collections_flag = np.where(rng.random(n) < 0.03, 1 - y, y)
    return {"days_past_due_current": days_past_due_current, "collections_flag": collections_flag}


def _make_messy(rng: np.random.Generator, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = len(df)
    df = df.copy()

    mask = rng.random(n) < 0.03
    df.loc[mask, "employment_type"] = rng.choice(HIDDEN_NULL_TOKENS, mask.sum())

    variants = rng.random(n)
    ho = df["home_ownership"].astype(object)
    ho = np.where(variants < 0.04, ho.str.upper(), ho)
    ho = np.where((variants >= 0.04) & (variants < 0.08), " " + pd.Series(ho).str.title(), ho)
    df["home_ownership"] = ho

    df["savings_balance"] = df["savings_balance"].map(lambda v: f"${v:,.2f}")

    df.loc[rng.random(n) < 0.02, "annual_income"] = np.nan
    for col in [c for c in df.columns if c.startswith("bureau_attr_")]:
        df.loc[rng.random(n) < rng.uniform(0.0, 0.05), col] = np.nan

    n_dupes = max(1, int(0.005 * n))
    dupes = df.iloc[rng.choice(n, n_dupes, replace=False)]
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df, n_dupes


def _build_manifest(df: pd.DataFrame, seed: int, n_dupes: int, messy: bool) -> dict:
    planted = {c for c, *_ in PLANTED_DRIVERS}
    bureau = [f"bureau_attr_{i:03d}" for i in range(1, N_BUREAU + 1)]
    txn = [f"txn_{c}_{s}" for c in TXN_CATEGORIES for s in TXN_STATS]
    clusters = [
        bureau[k * BUREAU_CLUSTER_SIZE:(k + 1) * BUREAU_CLUSTER_SIZE] for k in range(N_BUREAU_CLUSTERS)
    ]
    return {
        "dataset": "synthetic_credit_applications",
        "generator_version": 1,
        "seed": seed,
        "messy": messy,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target": TARGET,
        "target_rate": round(float(df[TARGET].mean()), 4),
        "planted_drivers": [
            {"column": c, "direction": d, "strength": s} for c, d, s, _w in PLANTED_DRIVERS
        ],
        "confounded_columns": [
            {"column": "interest_rate", "via": "credit_score",
             "note": "priced from credit_score; correlates with default but is not an independent driver"},
        ],
        "noise_columns": [c for c in bureau + txn if c not in planted],
        "correlated_noise_clusters": clusters,
        "leakage_columns": ["days_past_due_current", "collections_flag"],
        "informative_missingness": [
            {"column": "months_since_last_delinquency",
             "meaning": "missing means no delinquency in 2 years, which lowers default risk"},
        ],
        "protected_attributes": ["age", "gender"],
        "proxy_features": [{"column": "postal_area", "proxies_for": "age"}],
        "null_effect_columns": ["gender"],
        "trivial_derived_pairs": [["annual_income", "monthly_income"]],
        "redundant_pairs": [["total_credit_limit", "total_revolving_limit"]],
        "id_columns": ["application_id", "customer_id"],
        "constant_columns": ["data_source_version"],
        "mostly_missing_columns": ["secondary_applicant_income"],
        "high_cardinality_columns": ["employer_name"],
        "date_columns": ["application_date"],
        "data_quality_issues": {
            "duplicate_rows": n_dupes if messy else 0,
            "hidden_null_tokens": {"employment_type": list(HIDDEN_NULL_TOKENS)} if messy else {},
            "currency_strings": ["savings_balance"] if messy else [],
            "inconsistent_category_spellings": ["home_ownership"] if messy else [],
        },
    }


def generate_credit_applications(
    n_rows: int = 5000, seed: int = 42, target_rate: float = 0.12, messy: bool = True
) -> GeneratedDataset:
    """Generate the dataset and its ground-truth manifest. Same seed -> identical output."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100")
    if not 0.01 <= target_rate <= 0.5:
        raise ValueError("target_rate must be between 0.01 and 0.5")

    rng = np.random.default_rng(seed)
    customer_numbers = rng.choice(10 * n_rows, n_rows, replace=False)

    columns: dict[str, np.ndarray] = {
        "application_id": np.array([f"APP-{i:07d}" for i in range(1, n_rows + 1)]),
        "customer_id": np.array([f"CUST-{c:08d}" for c in customer_numbers]),
    }
    demo = _demographics(rng, n_rows)
    columns |= demo
    columns |= _financials(rng, n_rows, demo)
    columns |= _application_meta(rng, n_rows)
    columns |= _bureau_attributes(rng, n_rows)
    columns |= _transaction_features(rng, n_rows)
    df = pd.DataFrame(columns)

    y = _add_target(rng, df, target_rate)
    for name, values in _add_leakage(rng, y).items():
        df[name] = values
    df[TARGET] = y

    n_dupes = 0
    if messy:
        df, n_dupes = _make_messy(rng, df)

    return GeneratedDataset(data=df, manifest=_build_manifest(df, seed, n_dupes, messy))


def write_dataset(dataset: GeneratedDataset, out_dir: Path, name: str = "credit_applications") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    manifest_path = out_dir / f"{name}.manifest.json"
    dataset.data.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(dataset.manifest, indent=2))
    return csv_path, manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic credit-application dataset.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rate", type=float, default=0.12)
    parser.add_argument("--clean", action="store_true", help="skip data-quality problems")
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    args = parser.parse_args(argv)

    dataset = generate_credit_applications(args.rows, args.seed, args.target_rate, messy=not args.clean)
    csv_path, manifest_path = write_dataset(dataset, args.out)
    print(f"Wrote {csv_path} ({dataset.data.shape[0]} rows x {dataset.data.shape[1]} columns)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()