"""API tests: in-memory SQLite, no network, no real keys."""

import io
import json
import logging
import os
import time

import httpx
import pytest
from pydantic import SecretStr
from fastapi.testclient import TestClient
from sqlalchemy import select

import eda_api.service as service
from eda_api.app import create_app
from eda_api.config import Settings
from eda_api.db import Report, Run
from eda_api.service import json_safe
from eda_api.uploads import purge_stale_uploads
from eda_engine.synth.credit import generate_credit_applications

KEY_VARS = ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL", "APP_ENV")
ALICE = {"X-User-Email": "alice@example.com"}
BOB = {"X-User-Email": "bob@example.com"}


@pytest.fixture(scope="module")
def credit_csv() -> bytes:
    return generate_credit_applications(n_rows=2000, seed=3).data.to_csv(index=False).encode()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, app_env="test", database_url="sqlite:///:memory:",
                    upload_dir=tmp_path / "uploads", inline_workers=0)   # jobs finish inside the request


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def upload(client, csv: bytes, target="default_flag", headers=ALICE, name="credit.csv"):
    data = {"target": target} if target is not None else {}
    return client.post("/runs", files={"file": (name, io.BytesIO(csv), "text/csv")}, data=data, headers=headers)


def uploads_left(settings) -> list:
    return list(settings.upload_dir.glob("*")) if settings.upload_dir.exists() else []


class FakeReport:
    """Stands in for the engine when a test only cares about the HTTP layer."""
    narrative_source = "template"
    verification = {"status": "passed"}
    tokens = {"used": 0, "detail": []}

    def to_dict(self):
        return {"narrative": {"headline": "fake"}, "score": float("nan")}


@pytest.fixture
def fast_engine(monkeypatch):
    monkeypatch.setattr(service, "run_analysis", lambda df, target, llm=None, on_event=None: FakeReport())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "database": "ok", "queue": "inline", "version": "0.1.0"}


def test_upload_analyse_list_get_delete(client, credit_csv, settings):
    created = upload(client, credit_csv)
    assert created.status_code == 202, created.text
    run = created.json()
    assert run["status"] == "succeeded"
    assert (run["n_rows"], run["target"], run["filename"]) == (2010, "default_flag", "credit.csv")  # incl. 10 planted duplicates
    assert run["narrative_source"] == "template" and run["verification_status"] == "passed"
    assert run["key_source"] == "none" and run["tokens_used"] == 0
    report = run["report"]
    assert report["narrative"]["headline"] and report["charts"] and report["findings"]["findings"]
    assert uploads_left(settings) == []                      # Privacy-by-Design: raw file gone

    history = client.get("/runs", headers=ALICE).json()
    assert history["total"] == 1 and history["runs"][0]["id"] == run["id"]
    assert "report" not in history["runs"][0]                # list stays light

    fetched = client.get(f"/runs/{run['id']}", headers=ALICE).json()
    assert fetched["report"] == report

    assert client.delete(f"/runs/{run['id']}", headers=ALICE).status_code == 204
    assert client.get(f"/runs/{run['id']}", headers=ALICE).status_code == 404
    assert client.get("/runs", headers=ALICE).json()["total"] == 0


def test_report_contains_no_raw_rows(client, credit_csv):
    run = upload(client, credit_csv).json()
    stored = json.dumps(run["report"])
    first_ids = generate_credit_applications(n_rows=2000, seed=3).data["application_id"].astype(str)[:20]
    assert not any(i in stored for i in first_ids)           # identifiers never reach the stored report


def test_no_target_runs_in_correlation_mode(client, credit_csv):
    run = upload(client, credit_csv, target=None).json()
    assert run["status"] == "succeeded" and run["target"] is None


def test_history_is_newest_first_and_paginated(client, credit_csv, fast_engine):
    ids = [upload(client, credit_csv).json()["id"] for _ in range(3)]
    page = client.get("/runs?limit=2&offset=0", headers=ALICE).json()
    assert page["total"] == 3 and [r["id"] for r in page["runs"]] == ids[::-1][:2]
    assert [r["id"] for r in client.get("/runs?limit=2&offset=2", headers=ALICE).json()["runs"]] == [ids[0]]
    assert client.get("/runs?limit=500", headers=ALICE).status_code == 422


def test_nan_in_a_report_is_stored_as_null(client, credit_csv, fast_engine):
    run = upload(client, credit_csv).json()
    assert run["report"]["score"] is None
    assert json_safe({"a": [1.0, float("inf"), {"b": float("-inf")}]}) == {"a": [1.0, None, {"b": None}]}


# ---------------------------------------------------------------------------
# Privacy and failure handling
# ---------------------------------------------------------------------------

def test_analysis_failure_is_stored_as_failed_run_and_upload_is_deleted(client, credit_csv, settings, monkeypatch):
    def explode(df, target, llm=None, on_event=None):
        raise RuntimeError(f"boom with data {df.iloc[0, 0]}")
    monkeypatch.setattr(service, "run_analysis", explode)
    response = upload(client, credit_csv)
    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "failed" and run["report"] is None
    assert run["error"] == "RuntimeError: analysis failed"    # no data leaks into the error message
    assert run["finished_at"] is not None
    assert uploads_left(settings) == []


def test_validation_failure_stores_nothing_and_deletes_upload(client, credit_csv, settings):
    response = upload(client, credit_csv, target="no_such_column")
    assert response.status_code == 422 and "no_such_column" in response.json()["detail"]
    assert client.get("/runs", headers=ALICE).json()["total"] == 0
    assert uploads_left(settings) == []


def test_users_cannot_see_each_others_runs(client, credit_csv, fast_engine):
    run_id = upload(client, credit_csv, headers=ALICE).json()["id"]
    assert client.get(f"/runs/{run_id}", headers=BOB).status_code == 404
    assert client.delete(f"/runs/{run_id}", headers=BOB).status_code == 404
    bob_history = client.get("/runs", headers=BOB).json()
    assert bob_history["total"] == 0 and bob_history["runs"] == []
    assert client.get(f"/runs/{run_id}", headers=ALICE).status_code == 200


def test_user_key_is_used_but_never_stored_or_logged(client, credit_csv, settings, monkeypatch, caplog):
    user_key = "gsk_users_own_secret_key_123"
    seen_auth = []
    narrative = {"headline": "Report", "caveats": [],
                 "sections": [{"title": "Summary", "text": "Several variables are associated with default.",
                               "findings": [], "charts": []}]}

    def fake_provider(request):
        seen_auth.append(request.headers["authorization"])
        system = json.loads(request.content)["messages"][0]["content"]
        content = {"steps": []} if system.startswith("You choose") else narrative
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 200}})

    real_client = service.LLMClient

    class MockedClient(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(fake_provider), **kwargs)

    monkeypatch.setattr(service, "LLMClient", MockedClient)
    with caplog.at_level(logging.DEBUG):
        run = client.post("/runs", files={"file": ("c.csv", io.BytesIO(credit_csv), "text/csv")},
                          data={"target": "default_flag"}, headers={**ALICE, "X-Groq-Key": user_key}).json()

    assert seen_auth and all(a == f"Bearer {user_key}" for a in seen_auth)
    assert run["key_source"] == "user" and run["narrative_source"] == "llm" and run["tokens_used"] == 2200
    with client.app.state.sessionmaker() as session:
        stored = [json.dumps(r.data) for r in session.scalars(select(Report))]
        stored += [json.dumps({c.name: str(getattr(r, c.name)) for c in Run.__table__.columns})
                   for r in session.scalars(select(Run))]
    assert not any(user_key in s for s in stored)
    assert user_key not in caplog.text


def test_server_keys_from_settings_are_used_when_user_brings_none(settings):
    configured = settings.model_copy(update={"groq_api_key": SecretStr("gsk_server"),
                                             "cerebras_api_key": SecretStr("  ")})
    assert configured.server_llm_keys() == {"groq": "gsk_server"}
    llm = service.build_llm(configured, {})
    assert llm.keys.source("groq") == "server" and llm.keys.get("cerebras") is None
    assert llm.budget.limit == 10_000
    assert service.build_llm(settings, {}) is None             # no keys anywhere -> template report


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def test_too_large_upload_is_rejected_while_streaming(settings, credit_csv):
    small = settings.model_copy(update={"max_upload_mb": 1})
    with TestClient(create_app(small)) as c:
        response = upload(c, credit_csv * 2)                  # ~1.6 MB
    assert response.status_code == 413 and "1 MB" in response.json()["detail"]
    assert uploads_left(small) == []


@pytest.mark.parametrize("name, body, status, message", [
    ("data.xlsx", b"a,b\n1,2\n", 415, "only .csv"),
    ("data.csv", b"", 422, "empty"),
    ("data.csv", b"just_one_column\n1\n2\n", 422, "at least 2 columns"),
    ("data.csv", b'a,b\n"unterminated,2\n', 422, "could not read"),
])
def test_bad_files_are_rejected_with_a_clear_message(client, name, body, status, message):
    response = upload(client, body, target=None, name=name)
    assert response.status_code == status and message in response.json()["detail"]


def test_row_and_column_limits(settings, credit_csv):
    with TestClient(create_app(settings.model_copy(update={"max_rows": 500}))) as c:
        assert "more than 500 rows" in upload(c, credit_csv).json()["detail"]
    with TestClient(create_app(settings.model_copy(update={"max_columns": 50}))) as c:
        assert "limit is 50" in upload(c, credit_csv).json()["detail"]


# ---------------------------------------------------------------------------
# Identity (dev-only until step 13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers", [{}, {"X-User-Email": "not-an-email"}])
def test_requests_without_a_valid_identity_are_refused(client, headers):
    assert client.get("/runs", headers=headers).status_code == 401


def test_dev_identity_header_is_disabled_in_production(settings):
    prod = settings.model_copy(update={"app_env": "prod"})
    with TestClient(create_app(prod)) as c:
        assert c.get("/runs", headers=ALICE).status_code == 401


def test_email_identity_is_case_insensitive(client, credit_csv, fast_engine):
    upload(client, credit_csv, headers={"X-User-Email": "Alice@Example.com"})
    assert client.get("/runs", headers=ALICE).json()["total"] == 1


# ---------------------------------------------------------------------------
# Settings and housekeeping
# ---------------------------------------------------------------------------

def test_settings_normalise_neon_urls_and_enforce_prod_rules(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    neon = "postgresql://u:p@ep-x.neon.tech/db?sslmode=require"
    assert Settings(_env_file=None, database_url=neon).database_url.startswith("postgresql+psycopg://u:p@ep-x")
    assert Settings(_env_file=None, database_url="postgres://u:p@h/db").database_url == "postgresql+psycopg://u:p@h/db"
    with pytest.raises(ValueError, match="requires a Postgres"):
        Settings(_env_file=None, app_env="prod", database_url="sqlite:///x.db")


def test_settings_read_dotenv_and_ignore_v1_variables(tmp_path, monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=groq\nGROQ_API_KEY=gsk_from_file\nGROQ_MODEL=x\nMAX_UPLOAD_MB=5\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.server_llm_keys() == {"groq": "gsk_from_file"} and s.max_upload_mb == 5
    assert "gsk_from_file" not in repr(s)                     # SecretStr hides it


def test_stale_uploads_are_purged_at_startup(settings):
    settings.upload_dir.mkdir(parents=True)
    old, fresh = settings.upload_dir / "old.upload", settings.upload_dir / "fresh.upload"
    old.write_bytes(b"x")
    fresh.write_bytes(b"y")
    past = time.time() - 2 * 3600
    os.utime(old, (past, past))
    with TestClient(create_app(settings)):
        pass
    assert not old.exists() and fresh.exists()
    assert purge_stale_uploads(settings.upload_dir / "missing", 60) == 0


def test_alembic_migration_matches_the_models(tmp_path, monkeypatch):
    """The migration must create exactly what the models describe, and downgrade cleanly."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine, inspect

    from eda_api.db import Base

    api_dir = os.path.join(os.path.dirname(__file__), "..")
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    cfg = Config(os.path.join(api_dir, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")

    engine = create_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    assert diff == []
    assert set(inspect(engine).get_table_names()) >= {"users", "runs", "reports", "alembic_version"}

    command.downgrade(cfg, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()
