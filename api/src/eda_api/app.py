"""FastAPI app factory. `eda_api.main` builds the app from your settings.

Endpoints:
    GET    /health            API, database and queue status
    POST   /runs              upload a CSV (+ optional target) -> 202, run queued
    GET    /runs              your run history, newest first
    GET    /runs/{id}         one run (status, current stage) with its report when done
    GET    /runs/{id}/events  live progress (Server-Sent Events), replayable from the start
    DELETE /runs/{id}         delete a run and its report
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from eda_api import __version__
from eda_api.auth import current_user, get_session
from eda_api.config import Settings, get_settings
from eda_api.db import Base, Run, User, make_engine, make_sessionmaker
from eda_api.schemas import Health, RunDetail, RunList, RunSummary
from eda_api.jobs import FINAL_STATUSES, InlineBackend, RedisBackend, event
from eda_api.service import ACTIVE, accept_run, process_run
from eda_api.uploads import UploadRejected, purge_stale_uploads

log = logging.getLogger("eda_api")


SSE_WAIT_MS = 15_000     # how long one read waits for new events before sending a keep-alive


def make_backend(settings: Settings, sessions):
    if settings.queue_mode == "redis":
        import redis
        return RedisBackend(redis.Redis.from_url(settings.redis_url), settings.secret_key.get_secret_value(),
                            settings.key_handover_seconds)
    return InlineBackend(lambda job, secrets, publish: process_run(sessions, settings, job, secrets, publish),
                         workers=max(1, settings.inline_workers), synchronous=settings.inline_workers == 0)


def create_app(settings: Settings | None = None, backend=None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    sessions = make_sessionmaker(engine)
    backend = backend or make_backend(settings, sessions)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        removed = purge_stale_uploads(settings.upload_dir, settings.stale_upload_minutes)
        if removed:
            log.warning("purged %d stale upload(s) left by a previous crash", removed)
        if settings.database_url.startswith("sqlite"):
            Base.metadata.create_all(engine)        # local convenience; Postgres uses Alembic migrations
        if backend.mode == "inline":
            _fail_orphaned_runs(sessions)           # inline jobs die with the process that ran them
        yield
        if hasattr(backend, "close"):
            backend.close()
        engine.dispose()

    app = FastAPI(title="Agentic EDA API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.sessionmaker = sessions
    app.state.backend = backend
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    @app.exception_handler(UploadRejected)
    async def _rejected(_, exc: UploadRejected):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.get("/health", response_model=Health)
    def health(response: Response, session: Session = Depends(get_session)):
        try:
            session.execute(text("SELECT 1"))
            database = "ok"
        except Exception:  # noqa: BLE001
            database = "unavailable"
            response.status_code = 503
        queue = backend.mode
        if backend.mode == "redis":
            try:
                backend.redis.ping()
            except Exception:  # noqa: BLE001
                queue, response.status_code = "redis unavailable", 503
        ok = database == "ok" and response.status_code != 503
        return Health(status="ok" if ok else "degraded", database=database, queue=queue, version=__version__)

    @app.post("/runs", response_model=RunDetail, status_code=202)
    def create_run(
        file: UploadFile = File(..., description="CSV file"),
        target: str | None = Form(None, description="outcome column, e.g. default_flag"),
        x_groq_key: str | None = Header(default=None),
        x_cerebras_key: str | None = Header(default=None),
        x_openrouter_key: str | None = Header(default=None),
        user: User = Depends(current_user),
        session: Session = Depends(get_session),
    ):
        name = file.filename or "upload.csv"
        if not name.lower().endswith((".csv", ".txt")):
            raise UploadRejected("only .csv files are supported", 415)
        user_keys = {"groq": x_groq_key, "cerebras": x_cerebras_key, "openrouter": x_openrouter_key}
        run, job = accept_run(session, user, file.file, name, target, settings)
        try:
            backend.enqueue(job, {p: k for p, k in user_keys.items() if k})
        except Exception:
            log.exception("could not queue run %s", run.id)
            run.status, run.stage, run.error = "failed", "done", "QueueUnavailable: could not start the analysis"
            run.finished_at = datetime.now(UTC)
            session.commit()
            Path(job.path).unlink(missing_ok=True)
            raise HTTPException(503, "the analysis queue is unavailable; please try again shortly") from None
        session.refresh(run)
        return _detail(run)

    @app.get("/runs", response_model=RunList)
    def list_runs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  user: User = Depends(current_user), session: Session = Depends(get_session)):
        total = session.scalar(select(func.count()).select_from(Run).where(Run.user_id == user.id))
        runs = session.scalars(select(Run).where(Run.user_id == user.id)
                               .order_by(Run.created_at.desc(), Run.id).limit(limit).offset(offset)).all()
        return RunList(runs=[RunSummary.model_validate(r) for r in runs], total=total, limit=limit, offset=offset)

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str, user: User = Depends(current_user), session: Session = Depends(get_session)):
        return _detail(_owned_run(session, user, run_id))

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str, user: User = Depends(current_user), session: Session = Depends(get_session),
                   last_event_id: str | None = Header(default=None)):
        """Server-Sent Events: every progress event so far, then new ones live, until the run ends."""
        run = _owned_run(session, user, run_id)
        final = None if run.status in ACTIVE else event("done", run.error or run.verification_status or "",
                                                        run.status)
        session.close()                             # don't hold a DB connection for a long-lived stream

        def stream():
            position = last_event_id
            yield "retry: 3000\n\n"
            while True:
                batch = backend.events(run_id, position, block_ms=0 if final else SSE_WAIT_MS)
                for event_id, ev in batch:
                    position = event_id
                    yield f"id: {event_id}\nevent: {ev['stage']}\ndata: {json.dumps(ev)}\n\n"
                    if ev["status"] in FINAL_STATUSES:
                        return
                if final is not None:               # finished before we connected and the stream expired
                    yield f"event: done\ndata: {json.dumps(final)}\n\n"
                    return
                if not batch:
                    # Quiet for a while: make sure the run hasn't ended without a final event
                    # (e.g. a worker died and the clean-up marked it failed).
                    with sessions() as check:
                        current = check.get(Run, run_id)
                        if current is None or current.status not in ACTIVE:
                            done = event("done", (current.error or "") if current else "deleted",
                                         current.status if current else "failed")
                            yield f"event: done\ndata: {json.dumps(done)}\n\n"
                            return
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/runs/{run_id}", status_code=204)
    def delete_run(run_id: str, user: User = Depends(current_user), session: Session = Depends(get_session)):
        run = _owned_run(session, user, run_id)
        if run.status in ACTIVE:
            raise HTTPException(409, "the run is still in progress; delete it when it has finished")
        session.delete(run)
        session.commit()
        return Response(status_code=204)

    return app


def _owned_run(session: Session, user: User, run_id: str) -> Run:
    run = session.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        # 404 (not 403) for someone else's run: don't reveal that it exists
        raise HTTPException(404, "run not found")
    return run


def _fail_orphaned_runs(sessions) -> None:
    with sessions() as session:
        orphans = session.scalars(select(Run).where(Run.status.in_(ACTIVE))).all()
        for run in orphans:
            run.status, run.stage, run.finished_at = "failed", "done", datetime.now(UTC)
            run.error = "ServerRestarted: the server restarted before this run finished; please run it again"
        session.commit()
        if orphans:
            log.warning("marked %d unfinished inline run(s) as failed after restart", len(orphans))


def _detail(run: Run) -> RunDetail:
    summary = RunSummary.model_validate(run).model_dump()
    return RunDetail(**summary, report=run.report.data if run.report else None)

