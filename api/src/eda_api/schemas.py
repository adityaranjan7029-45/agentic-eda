"""Response shapes (what the API promises the frontend)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    status: str
    filename: str
    size_bytes: int
    n_rows: int | None
    n_columns: int | None
    target: str | None
    narrative_source: str | None
    verification_status: str | None
    tokens_used: int
    key_source: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RunDetail(RunSummary):
    report: dict | None = None


class RunList(BaseModel):
    runs: list[RunSummary]
    total: int
    limit: int
    offset: int


class Health(BaseModel):
    status: str
    database: str
    version: str
