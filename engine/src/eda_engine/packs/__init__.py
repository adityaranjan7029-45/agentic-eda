"""Industry packs: domain knowledge layered on top of the general engine."""

from eda_engine.packs.fintech import (
    FintechConfig,
    FintechReport,
    VariableStrength,
    information_value,
    iv_band,
    ks_statistic,
    lender_decision_columns,
    protected_attributes,
    run_fintech_pack,
)

PACKS = {"fintech": run_fintech_pack}

__all__ = [
    "PACKS",
    "FintechConfig",
    "FintechReport",
    "VariableStrength",
    "information_value",
    "iv_band",
    "ks_statistic",
    "lender_decision_columns",
    "protected_attributes",
    "run_fintech_pack",
]
