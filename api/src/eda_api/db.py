"""Database engine, session and table models (SQLAlchemy 2)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

JSONType = JSON().with_variant(JSONB(), "postgresql")   # JSONB on Postgres, JSON elsewhere


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    runs: Mapped[list[Run]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")   # queued | running | succeeded | failed
    filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    n_rows: Mapped[int | None] = mapped_column(Integer)
    n_columns: Mapped[int | None] = mapped_column(Integer)
    target: Mapped[str | None] = mapped_column(String(255))
    narrative_source: Mapped[str | None] = mapped_column(String(16))   # llm | template
    verification_status: Mapped[str | None] = mapped_column(String(32))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    key_source: Mapped[str | None] = mapped_column(String(16))         # user | server | none
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user: Mapped[User] = relationship(back_populates="runs")
    report: Mapped[Report | None] = relationship(back_populates="run", cascade="all, delete-orphan",
                                                 uselist=False)


class Report(Base):
    __tablename__ = "reports"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSONType)
    run: Mapped[Run] = relationship(back_populates="report")


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url.rstrip("/").endswith("sqlite:"):
            kwargs["poolclass"] = StaticPool       # one shared in-memory database (tests)
        return create_engine(url, **kwargs)
    # Neon closes idle connections; pre_ping replaces dead ones transparently.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
