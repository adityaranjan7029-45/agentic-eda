"""The same API against a real Postgres (JSONB, psycopg, migrations, cascades).

Skipped unless TEST_DATABASE_URL is set. CI starts a Postgres 16 service for
it; locally:  $env:TEST_DATABASE_URL="postgresql://user:pass@localhost:5432/eda_test"
Never point it at a database you care about: it drops and recreates the tables.
"""

import io
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def migrated():
    from alembic import command
    from alembic.config import Config

    from eda_api.config import Settings

    url = Settings(_env_file=None, database_url=URL).database_url
    api_dir = os.path.join(os.path.dirname(__file__), "..")
    cfg = Config(os.path.join(api_dir, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))
    cfg.attributes["database_url"] = url
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield url, cfg
    command.downgrade(cfg, "base")


def test_migration_matches_models_on_postgres(migrated):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from eda_api.db import Base

    engine = create_engine(migrated[0])
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata) == []
        column = next(c for c in inspect(conn).get_columns("reports") if c["name"] == "data")
        assert str(column["type"]) == "JSONB"
    engine.dispose()


def test_full_run_on_postgres_stores_jsonb_and_cascades(migrated, tmp_path, monkeypatch):
    from eda_api.app import create_app
    from eda_api.config import Settings
    from eda_engine.synth.credit import generate_credit_applications

    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None, app_env="test", database_url=migrated[0], upload_dir=tmp_path)
    csv = generate_credit_applications(n_rows=2000, seed=4).data.to_csv(index=False).encode()
    headers = {"X-User-Email": "pg@example.com"}
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["database"] == "ok"
        run = client.post("/runs", files={"file": ("c.csv", io.BytesIO(csv), "text/csv")},
                          data={"target": "default_flag"}, headers=headers).json()
        assert run["status"] == "succeeded"

        engine = create_engine(migrated[0])
        with engine.connect() as conn:   # query inside the JSONB document, as the history page will
            headline = conn.scalar(text("SELECT data->'narrative'->>'headline' FROM reports WHERE run_id = :id"),
                                   {"id": run["id"]})
            assert headline == run["report"]["narrative"]["headline"]

        assert client.delete(f"/runs/{run['id']}", headers=headers).status_code == 204
        with engine.connect() as conn:
            assert conn.scalar(text("SELECT count(*) FROM reports")) == 0
        engine.dispose()
