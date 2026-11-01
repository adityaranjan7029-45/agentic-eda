"""Runs: validate the upload, analyse it, store the report, delete the upload.

Kept separate from the HTTP layer so step 12 can call `execute_run` from a
background worker instead of inside the request.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from typing import BinaryIO

from sqlalchemy.orm import Session

from eda_api.config import Settings
from eda_api.db import Report, Run, User
from eda_api.uploads import UploadRejected, read_csv, temporary_upload
from eda_engine.agents import run_analysis
from eda_engine.llm import ApiKey, KeyRing, LLMClient, TokenBudget

log = logging.getLogger(__name__)


def json_safe(value):
    """Postgres JSONB rejects NaN/Infinity; replace them with null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def build_llm(settings: Settings, user_keys: dict[str, str]) -> LLMClient | None:
    ring = KeyRing(server={p: ApiKey(k) for p, k in settings.server_llm_keys().items()})
    ring = ring.with_user_keys({p: k for p, k in user_keys.items() if k and k.strip()})
    if not any(ring.get(p) for p in ("groq", "cerebras", "openrouter")):
        return None
    return LLMClient(ring, budget=TokenBudget(settings.run_token_budget))


def _key_source(report_tokens: dict) -> str:
    sources = {c.get("key_source") for c in report_tokens.get("detail", [])}
    return "user" if "user" in sources else "server" if "server" in sources else "none"


def execute_run(session: Session, user: User, source: BinaryIO, filename: str, target: str | None,
                settings: Settings, user_keys: dict[str, str] | None = None) -> Run:
    """Create a run and process it synchronously. Validation problems raise UploadRejected
    (no run is stored); analysis failures are stored as a failed run."""
    with temporary_upload(source, settings.upload_dir, settings.max_upload_bytes) as (path, size):
        df = read_csv(path, settings.max_rows, settings.max_columns)
        target = (target or "").strip() or None
        if target is not None and target not in df.columns:
            raise UploadRejected(f"target column '{target}' is not in the file")

        run = Run(user_id=user.id, status="running", filename=filename[:255], size_bytes=size,
                  n_rows=len(df), n_columns=df.shape[1], target=target, started_at=datetime.now(UTC))
        session.add(run)
        session.commit()
        log.info("run %s started: %s rows x %s columns", run.id, run.n_rows, run.n_columns)

        llm = build_llm(settings, user_keys or {})
        try:
            report = run_analysis(df, target, llm=llm)
            data = json_safe(json.loads(json.dumps(report.to_dict(), default=str)))
            run.report = Report(data=data)
            run.status = "succeeded"
            run.narrative_source = report.narrative_source
            run.verification_status = report.verification["status"]
            run.tokens_used = report.tokens.get("used", 0)
            run.key_source = _key_source(report.tokens)
        except Exception as exc:  # noqa: BLE001 - any analysis failure becomes a failed run, never a 500
            log.exception("run %s failed", run.id)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: analysis failed"
        finally:
            if llm is not None:
                llm.close()
            run.finished_at = datetime.now(UTC)
            session.commit()
        log.info("run %s %s", run.id, run.status)
        return run
