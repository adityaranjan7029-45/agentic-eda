"""Column role detection: what each column is, before any analysis runs."""

from eda_engine.schema.roles import (
    ColumnInfo,
    ColumnRole,
    DatasetSchema,
    Flag,
    RoleConfig,
    infer_schema,
)

__all__ = ["ColumnInfo", "ColumnRole", "DatasetSchema", "Flag", "RoleConfig", "infer_schema"]
