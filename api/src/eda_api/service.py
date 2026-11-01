"""Runs, in two halves:

  accept_run   (API)     stream the upload to disk, validate it, store the run as `queued`
  process_run  (worker)  analyse, store the report, delete the upload, publish progress

The upload lives on disk between the two (owner-only permissions). Whoever runs
the job deletes it in `finally`; anything orphaned by a crash is purged on startup.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from eda_api.config import Settings
from eda_api.db import Report, Run, User
from eda_api.jobs import Job, event
from eda_api.uploads import UploadRejected, read_csv, save_upload
from eda_engine.agents import run_analysis
from eda_engine.llm import ApiKey, KeyRing, LLMClient, TokenBudget

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running")


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


# ---------------------------------------------------------------------------
# API side
# ---------------------------------------------------------------------------

def accept_run(session: Session, user: User, source: BinaryIO, filename: str, target: str | None,
               settings: Settings) -> tuple[Run, Job]:
    """Validate the upload and store a queued run. Raises UploadRejected (nothing stored, file deleted)."""
    active = session.scalar(select(func.count()).select_from(Run)
                            .where(Run.user_id == user.id, Run.status.in_(ACTIVE)))
    if active >= settings.max_active_runs:
        raise UploadRejected(f"you already have {active} runs in progress; wait for one to finish", 429)

    run_id = str(uuid.uuid4())
    path, size = save_upload(source, settings.upload_dir, settings.max_upload_bytes, name=run_id)
    try:
        df = read_csv(path, settings.max_rows, settings.max_columns)
        target = (target or "").strip() or None
        if target is not None and target not in df.columns:
            raise UploadRejected(f"target column '{target}' is not in the file")
        run = Run(id=run_id, user_id=user.id, status="queued", stage="queued", filename=filename[:255],
                  size_bytes=size, n_rows=len(df), n_columns=df.shape[1], target=target)
        session.add(run)
        session.commit()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return run, Job(run_id=run.id, user_id=user.id, path=str(path), target=target)


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

def process_run(sessions: sessionmaker[Session], settings: Settings, job: Job, user_keys: dict[str, str],
                publish: Callable[[dict], None]) -> str:
    """Run one job end to end. Never raises: failures become a failed run. Returns the final status."""
    path = Path(job.path)
    status, llm = "failed", None
    with sessions() as session:
        run = session.get(Run, job.run_id)
        try:
            if run is None or run.status != "queued":
                log.warning("job %s skipped: run %s", job.run_id, "missing" if run is None else run.status)
                return run.status if run else "missing"
            run.status, run.stage, run.started_at = "running", "started", datetime.now(UTC)
            session.commit()
            publish(event("started", "worker picked up the job"))

            df = read_csv(path, settings.max_rows, settings.max_columns)
            path.unlink(missing_ok=True)            # data is in memory now; the file isn't needed

            def on_event(stage: str, detail: str) -> None:
                run.stage = stage
                session.commit()
                publish(event(stage, detail))

            llm = build_llm(settings, user_keys)
            report = run_analysis(df, job.target, llm=llm, on_event=on_event)
            run.report = Report(data=json_safe(json.loads(json.dumps(report.to_dict(), default=str))))
            run.narrative_source = report.narrative_source
            run.verification_status = report.verification["status"]
            run.tokens_used = report.tokens.get("used", 0)
            run.key_source = _key_source(report.tokens)
            status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - any failure becomes a failed run
            log.exception("run %s failed", job.run_id)
            if run is not None:
                run.error = f"{type(exc).__name__}: analysis failed"
        finally:
            path.unlink(missing_ok=True)
            if llm is not None:
                llm.close()
            if run is not None and run.status == "running":
                run.status, run.stage, run.finished_at = status, "done", datetime.now(UTC)
                session.commit()
                publish(event("done", run.error or run.verification_status or "", status))
    log.info("run %s %s", job.run_id, status)
    return status


def fail_stale_run(sessions: sessionmaker[Session], job: Job, publish: Callable[[dict], None]) -> None:
    """A worker died mid-job: mark the run failed (its upload and keys may be gone) and clean up."""
    Path(job.path).unlink(missing_ok=True)
    with sessions() as session:
        run = session.get(Run, job.run_id)
        if run is not None and run.status in ACTIVE:
            run.status, run.stage, run.finished_at = "failed", "done", datetime.now(UTC)
            run.error = "WorkerLost: the worker stopped before finishing; please run it again"
            session.commit()
            publish(event("done", run.error, "failed"))


