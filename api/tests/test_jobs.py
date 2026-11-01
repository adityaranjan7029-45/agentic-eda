"""Job queue, worker and live progress, on both backends.

Redis mode uses fakeredis here (and real Redis in test_redis.py / CI). The same
contract tests run against inline and Redis mode, so the two can't drift apart.
"""

import io
import json
import threading
import time

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import eda_api.service as service
from eda_api.app import create_app
from eda_api.config import Settings
from eda_api.db import Run, make_engine, make_sessionmaker
from eda_api.jobs import LEASES, PROCESSING, QUEUE, InlineBackend, Job, RedisBackend, fernet_for
from eda_api.worker import Worker
from eda_engine.synth.credit import generate_credit_applications

SECRET = "test-secret-key-that-is-long-enough-0123456789"
ALICE = {"X-User-Email": "alice@example.com"}
STAGES = ["queued", "started", "plan", "analyse", "narrate", "critic", "finalize", "done"]


@pytest.fixture(scope="module")
def credit_csv() -> bytes:
    return generate_credit_applications(n_rows=1500, seed=5).data.to_csv(index=False).encode()


@pytest.fixture
def base_settings(tmp_path, monkeypatch):
    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL", "REDIS_URL", "APP_ENV"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, app_env="test", database_url=f"sqlite:///{tmp_path / 'jobs.db'}",
                    upload_dir=tmp_path / "uploads")


def post(client, csv, headers=ALICE, target="default_flag"):
    return client.post("/runs", files={"file": ("credit.csv", io.BytesIO(csv), "text/csv")},
                       data={"target": target}, headers=headers)


def read_sse(client, run_id, headers=ALICE, last_event_id=None) -> list[tuple[str | None, dict]]:
    h = {**headers, **({"Last-Event-ID": last_event_id} if last_event_id else {})}
    with client.stream("GET", f"/runs/{run_id}/events", headers=h) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events, current_id = [], None
        for line in response.iter_lines():
            if line.startswith("id: "):
                current_id = line[4:]
            elif line.startswith("data: "):
                events.append((current_id, json.loads(line[6:])))
                current_id = None
        return events


def wait_for(client, run_id, headers=ALICE, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}", headers=headers).json()
        if run["status"] in ("succeeded", "failed"):
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


class RedisSetup:
    """API + worker sharing one fake Redis, one database and one upload folder."""

    def __init__(self, settings):
        self.redis = fakeredis.FakeRedis()
        self.settings = settings.model_copy(update={"redis_url": "redis://fake", "secret_key": SECRET})
        self.backend = RedisBackend(self.redis, SECRET, key_ttl_seconds=600)
        self.app = create_app(self.settings, backend=self.backend)
        self.worker = Worker(self.settings, self.backend,
                             make_sessionmaker(make_engine(self.settings.database_url)))


@pytest.fixture
def redis_mode(base_settings):
    setup = RedisSetup(base_settings)
    with TestClient(setup.app) as client:
        setup.client = client
        yield setup


@pytest.fixture
def inline_mode(base_settings):
    with TestClient(create_app(base_settings.model_copy(update={"inline_workers": 2}))) as client:
        yield client


# ---------------------------------------------------------------------------
# Contract: both modes behave the same
# ---------------------------------------------------------------------------

def test_inline_mode_runs_in_the_background_with_live_events(inline_mode, credit_csv, base_settings):
    created = post(inline_mode, credit_csv)
    assert created.status_code == 202
    assert created.json()["status"] in ("queued", "running", "succeeded")
    events = read_sse(inline_mode, created.json()["id"])          # streams until the run ends
    assert [e["stage"] for _, e in events] == STAGES
    assert events[-1][1]["status"] == "succeeded"
    run = inline_mode.get(f"/runs/{created.json()['id']}", headers=ALICE).json()
    assert run["status"] == "succeeded" and run["stage"] == "done" and run["report"]
    assert list(base_settings.upload_dir.glob("*")) == []


def test_redis_mode_queues_until_a_worker_runs_it(redis_mode, credit_csv):
    created = post(redis_mode.client, credit_csv).json()
    assert created["status"] == "queued" and created["report"] is None
    assert redis_mode.redis.llen(QUEUE) == 1

    assert redis_mode.worker.run_once(timeout_seconds=1) == "succeeded"
    assert redis_mode.redis.llen(QUEUE) == 0 and redis_mode.redis.llen(PROCESSING) == 0
    assert redis_mode.redis.hlen(LEASES) == 0

    events = read_sse(redis_mode.client, created["id"])
    assert [e["stage"] for _, e in events] == STAGES
    run = redis_mode.client.get(f"/runs/{created['id']}", headers=ALICE).json()
    assert run["status"] == "succeeded" and run["report"]["narrative"]["headline"]
    assert list(redis_mode.settings.upload_dir.glob("*")) == []


def test_late_or_reconnecting_clients_get_the_events_they_missed(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    redis_mode.worker.run_once(timeout_seconds=1)
    everything = read_sse(redis_mode.client, run_id)
    resume_from = everything[2][0]                                   # client saw 3 events, then dropped
    rest = read_sse(redis_mode.client, run_id, last_event_id=resume_from)
    assert [e for _, e in rest] == [e for _, e in everything[3:]]


def test_stream_of_a_finished_run_whose_events_expired_still_ends(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    redis_mode.worker.run_once(timeout_seconds=1)
    redis_mode.redis.delete(f"eda:events:{run_id}")
    events = read_sse(redis_mode.client, run_id)
    assert [(e["stage"], e["status"]) for _, e in events] == [("done", "succeeded")]


def test_stream_waits_for_a_worker_and_delivers_live(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    threading.Timer(0.5, lambda: redis_mode.worker.run_once(timeout_seconds=1)).start()
    events = read_sse(redis_mode.client, run_id)                      # starts before the worker
    assert [e["stage"] for _, e in events] == STAGES


# ---------------------------------------------------------------------------
# Users' own keys: encrypted handover in Redis
# ---------------------------------------------------------------------------

def test_user_key_is_encrypted_in_redis_expires_and_is_deleted_on_pickup(redis_mode, credit_csv, monkeypatch):
    user_key = "gsk_users_own_secret_key_456"
    run_id = redis_mode.client.post(
        "/runs", files={"file": ("c.csv", io.BytesIO(credit_csv), "text/csv")},
        data={"target": "default_flag"}, headers={**ALICE, "X-Groq-Key": user_key}).json()["id"]

    stored = redis_mode.redis.get(f"eda:secrets:{run_id}")
    assert stored and user_key.encode() not in stored                     # ciphertext, not the key
    assert 0 < redis_mode.redis.ttl(f"eda:secrets:{run_id}") <= 600        # expires on its own
    assert all(user_key.encode() not in redis_mode.redis.dump(k) for k in redis_mode.redis.keys()
               if not k.startswith(b"eda:secrets:"))                      # nowhere else in Redis
    assert json.loads(fernet_for(SECRET).decrypt(stored)) == {"groq": user_key}

    seen = []

    def fake_provider(request):
        seen.append(request.headers["authorization"])
        system = json.loads(request.content)["messages"][0]["content"]
        content = {"steps": []} if system.startswith("You choose") else {
            "headline": "Report", "caveats": [],
            "sections": [{"title": "Summary", "text": "Several variables are associated with default.",
                          "findings": [], "charts": []}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 200}})

    real = service.LLMClient
    monkeypatch.setattr(service, "LLMClient",
                        lambda *a, **k: real(*a, transport=httpx.MockTransport(fake_provider), **k))
    assert redis_mode.worker.run_once(timeout_seconds=1) == "succeeded"
    assert seen and all(a == f"Bearer {user_key}" for a in seen)
    assert redis_mode.redis.get(f"eda:secrets:{run_id}") is None           # gone after pickup
    run = redis_mode.client.get(f"/runs/{run_id}", headers=ALICE).json()
    assert run["key_source"] == "user" and user_key not in json.dumps(run)


def test_tampered_or_expired_secrets_mean_no_user_key():
    backend = RedisBackend(fakeredis.FakeRedis(), SECRET)
    backend.redis.set("eda:secrets:r1", b"not-a-valid-token")
    assert backend.claim_secrets("r1") == {}
    assert backend.claim_secrets("never-set") == {}
    other = RedisBackend(backend.redis, "a-completely-different-secret-key-0123456789")
    other.enqueue(Job("r2", "u", "/nope", None), {"groq": "gsk_x"})
    assert backend.claim_secrets("r2") == {}                                # wrong server secret


def test_redis_mode_requires_a_long_secret_key(base_settings):
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(_env_file=None, redis_url="redis://localhost:6379/0")
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(_env_file=None, redis_url="redis://localhost:6379/0", secret_key="short")
    assert Settings(_env_file=None, redis_url="redis://x", secret_key=SECRET).queue_mode == "redis"
    assert base_settings.queue_mode == "inline"


# ---------------------------------------------------------------------------
# Failures: worker crash, queue down, analysis error
# ---------------------------------------------------------------------------

def test_jobs_left_by_a_dead_worker_are_failed_and_cleaned_up(redis_mode, credit_csv):
    run_id = redis_mode.client.post(
        "/runs", files={"file": ("c.csv", io.BytesIO(credit_csv), "text/csv")},
        data={"target": "default_flag"}, headers={**ALICE, "X-Groq-Key": "gsk_k"}).json()["id"]
    job, _ = redis_mode.backend.take(timeout_seconds=1)                 # a worker takes it... and dies
    assert redis_mode.redis.llen(PROCESSING) == 1
    assert redis_mode.worker.recover() == 0                              # still within the timeout

    redis_mode.redis.hset(LEASES, run_id, time.time() - 3600)           # an hour later
    assert redis_mode.worker.recover() == 1
    run = redis_mode.client.get(f"/runs/{run_id}", headers=ALICE).json()
    assert run["status"] == "failed" and run["error"].startswith("WorkerLost")
    assert redis_mode.redis.llen(PROCESSING) == 0
    assert redis_mode.redis.get(f"eda:secrets:{run_id}") is None
    assert list(redis_mode.settings.upload_dir.glob("*")) == []
    assert read_sse(redis_mode.client, run_id)[-1][1]["status"] == "failed"


def test_queue_down_fails_the_run_cleanly(base_settings, credit_csv):
    class DownRedis(fakeredis.FakeRedis):
        def lpush(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    setup = RedisSetup(base_settings)
    setup.backend.redis = DownRedis()
    with TestClient(setup.app) as client:
        response = post(client, credit_csv)
        assert response.status_code == 503
        runs = client.get("/runs", headers=ALICE).json()["runs"]
        assert runs[0]["status"] == "failed" and runs[0]["error"].startswith("QueueUnavailable")
    assert list(setup.settings.upload_dir.glob("*")) == []


def test_health_reports_redis_status(base_settings):
    setup = RedisSetup(base_settings)
    with TestClient(setup.app) as client:
        assert client.get("/health").json()["queue"] == "redis"
        setup.backend.redis.ping = lambda: (_ for _ in ()).throw(ConnectionError("down"))
        response = client.get("/health")
        assert response.status_code == 503 and response.json()["queue"] == "redis unavailable"


def test_analysis_error_in_the_worker_becomes_a_failed_run(redis_mode, credit_csv, monkeypatch):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    monkeypatch.setattr(service, "run_analysis",
                        lambda *a, **k: (_ for _ in ()).throw(MemoryError("too big")))
    assert redis_mode.worker.run_once(timeout_seconds=1) == "failed"
    run = redis_mode.client.get(f"/runs/{run_id}", headers=ALICE).json()
    assert run["status"] == "failed" and run["error"] == "MemoryError: analysis failed"
    assert list(redis_mode.settings.upload_dir.glob("*")) == []
    assert redis_mode.redis.llen(PROCESSING) == 0


def test_a_job_delivered_twice_is_only_run_once(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    raw = redis_mode.redis.lindex(QUEUE, 0)
    redis_mode.redis.lpush(QUEUE, raw)                                   # duplicate delivery
    assert redis_mode.worker.run_once(timeout_seconds=1) == "succeeded"
    assert redis_mode.worker.run_once(timeout_seconds=1) == "succeeded"  # skipped: already finished
    stages = [e["stage"] for _, e in read_sse(redis_mode.client, run_id)]
    assert stages.count("started") == 1


def test_inline_runs_interrupted_by_a_restart_are_marked_failed(base_settings):
    engine = make_engine(base_settings.database_url)
    from eda_api.db import Base, User
    Base.metadata.create_all(engine)
    sessions = make_sessionmaker(engine)
    with sessions() as s:
        user = User(email="alice@example.com")
        s.add(user)
        s.flush()
        s.add(Run(user_id=user.id, status="running", stage="analyse", filename="x.csv", size_bytes=1))
        s.commit()
    with TestClient(create_app(base_settings)) as client:
        run = client.get("/runs", headers=ALICE).json()["runs"][0]
    assert run["status"] == "failed" and run["error"].startswith("ServerRestarted")


# ---------------------------------------------------------------------------
# Limits and housekeeping
# ---------------------------------------------------------------------------

def test_at_most_two_active_runs_per_user(redis_mode, credit_csv):
    assert post(redis_mode.client, credit_csv).status_code == 202
    assert post(redis_mode.client, credit_csv).status_code == 202
    third = post(redis_mode.client, credit_csv)
    assert third.status_code == 429 and "2 runs in progress" in third.json()["detail"]
    assert post(redis_mode.client, credit_csv, headers={"X-User-Email": "bob@example.com"}).status_code == 202
    redis_mode.worker.run_once(timeout_seconds=1)
    assert post(redis_mode.client, credit_csv).status_code == 202        # a slot freed up
    assert len(list(redis_mode.settings.upload_dir.glob("*"))) == 3      # only queued uploads remain


def test_runs_in_progress_cannot_be_deleted(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    assert redis_mode.client.delete(f"/runs/{run_id}", headers=ALICE).status_code == 409
    redis_mode.worker.run_once(timeout_seconds=1)
    assert redis_mode.client.delete(f"/runs/{run_id}", headers=ALICE).status_code == 204


def test_events_are_private(redis_mode, credit_csv):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    response = redis_mode.client.get(f"/runs/{run_id}/events", headers={"X-User-Email": "bob@example.com"})
    assert response.status_code == 404


def test_progress_stage_is_stored_on_the_run(redis_mode, credit_csv, monkeypatch):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    seen = []
    real = service.run_analysis

    def spy(df, target, llm=None, on_event=None):
        def wrapped(stage, detail):
            on_event(stage, detail)
            with redis_mode.app.state.sessionmaker() as s:
                seen.append(s.scalar(select(Run.stage).where(Run.id == run_id)))
        return real(df, target, llm=llm, on_event=wrapped)

    monkeypatch.setattr(service, "run_analysis", spy)
    redis_mode.worker.run_once(timeout_seconds=1)
    assert seen == ["plan", "analyse", "narrate", "critic", "finalize"]   # visible to GET /runs/{id} live


def test_worker_refuses_to_start_without_redis(monkeypatch):
    from eda_api import worker
    monkeypatch.setattr(worker, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(SystemExit, match="needs REDIS_URL"):
        worker.main()


def test_inline_backend_event_reads_wait_and_time_out():
    backend = InlineBackend(lambda *a: None, synchronous=True)
    started = time.monotonic()
    assert backend.events("r", None, block_ms=200) == []
    assert 0.15 < time.monotonic() - started < 2
    threading.Timer(0.1, lambda: backend.publish("r", {"stage": "x", "status": "running"})).start()
    got = backend.events("r", None, block_ms=5000)
    assert [e["stage"] for _, e in got] == ["x"]
    assert backend.events("r", got[-1][0], block_ms=0) == []


def test_upload_is_deleted_even_if_the_worker_cannot_read_it(redis_mode, credit_csv, monkeypatch):
    run_id = post(redis_mode.client, credit_csv).json()["id"]
    monkeypatch.setattr(service, "read_csv", lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")))
    assert redis_mode.worker.run_once(timeout_seconds=1) == "failed"
    assert list(redis_mode.settings.upload_dir.glob("*")) == []
    assert redis_mode.client.get(f"/runs/{run_id}", headers=ALICE).json()["error"] == "OSError: analysis failed"


def test_stream_ends_when_the_run_ends_without_a_final_event(redis_mode, credit_csv, monkeypatch):
    """E.g. a worker died and the clean-up marked the run failed before publishing."""
    import eda_api.app as app_module
    monkeypatch.setattr(app_module, "SSE_WAIT_MS", 200)
    run_id = post(redis_mode.client, credit_csv).json()["id"]

    def mark_failed():                               # happens while the client is already listening
        with redis_mode.app.state.sessionmaker() as s:
            s.get(Run, run_id).status = "failed"
            s.commit()

    threading.Timer(0.5, mark_failed).start()
    events = read_sse(redis_mode.client, run_id)
    assert [(e["stage"], e["status"]) for _, e in events] == [("queued", "queued"), ("done", "failed")]
