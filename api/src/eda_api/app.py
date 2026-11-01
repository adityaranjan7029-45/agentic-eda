"""FastAPI app factory. `eda_api.main` builds the app from your settings.

Endpoints:
    GET    /health          API + database status
    POST   /runs            upload a CSV (+ optional target) -> analysed run with report
    GET    /runs            your run history, newest first
    GET    /runs/{id}       one run with its full report
    DELETE /runs/{id}       delete a run and its report
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from eda_api import __version__
from eda_api.auth import current_user, get_session
from eda_api.config import Settings, get_settings
from eda_api.db import Base, Run, User, make_engine, make_sessionmaker
from eda_api.schemas import Health, RunDetail, RunList, RunSummary
from eda_api.service import execute_run
from eda_api.uploads import UploadRejected, purge_stale_uploads

log = logging.getLogger("eda_api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        removed = purge_stale_uploads(settings.upload_dir, settings.stale_upload_minutes)
        if removed:
            log.warning("purged %d stale upload(s) left by a previous crash", removed)
        if settings.database_url.startswith("sqlite"):
            Base.metadata.create_all(engine)        # local convenience; Postgres uses Alembic migrations
        yield
        engine.dispose()

    app = FastAPI(title="Agentic EDA API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.sessionmaker = make_sessionmaker(engine)
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
        return Health(status="ok" if database == "ok" else "degraded", database=database, version=__version__)

    @app.post("/runs", response_model=RunDetail, status_code=201)
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
        run = execute_run(session, user, file.file, name, target, settings,
                          {p: k for p, k in user_keys.items() if k})
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

    @app.delete("/runs/{run_id}", status_code=204)
    def delete_run(run_id: str, user: User = Depends(current_user), session: Session = Depends(get_session)):
        session.delete(_owned_run(session, user, run_id))
        session.commit()
        return Response(status_code=204)

    return app


def _owned_run(session: Session, user: User, run_id: str) -> Run:
    run = session.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        # 404 (not 403) for someone else's run: don't reveal that it exists
        raise HTTPException(404, "run not found")
    return run


def _detail(run: Run) -> RunDetail:
    summary = RunSummary.model_validate(run).model_dump()
    return RunDetail(**summary, report=run.report.data if run.report else None)

