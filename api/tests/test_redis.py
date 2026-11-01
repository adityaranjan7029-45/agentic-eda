"""API + a real worker thread against a real Redis server.

Skipped unless TEST_REDIS_URL is set (CI starts a Redis service). Locally:
    $env:TEST_REDIS_URL="redis://localhost:6379/15"
It flushes that database, so use a spare one.
"""

import io
import json
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

URL = os.environ.get("TEST_REDIS_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_REDIS_URL not set")
SECRET = "ci-secret-key-that-is-long-enough-0123456789"


def test_real_redis_worker_runs_jobs_with_live_events(tmp_path, monkeypatch):
    import redis

    from eda_api.app import create_app
    from eda_api.config import Settings
    from eda_api.db import make_engine, make_sessionmaker
    from eda_api.jobs import RedisBackend
    from eda_api.worker import Worker
    from eda_engine.synth.credit import generate_credit_applications

    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    client_redis = redis.Redis.from_url(URL)
    client_redis.flushdb()
    settings = Settings(_env_file=None, app_env="test", database_url=f"sqlite:///{tmp_path / 'r.db'}",
                        upload_dir=tmp_path / "uploads", redis_url=URL, secret_key=SECRET)
    backend = RedisBackend(client_redis, SECRET)
    csv = generate_credit_applications(n_rows=1500, seed=8).data.to_csv(index=False).encode()
    stop = threading.Event()

    with TestClient(create_app(settings, backend=backend)) as client:
        assert client.get("/health").json()["queue"] == "redis"
        worker = Worker(settings, RedisBackend(redis.Redis.from_url(URL), SECRET),
                        make_sessionmaker(make_engine(settings.database_url)))
        thread = threading.Thread(target=worker.run_forever, args=(stop,), daemon=True)
        thread.start()
        try:
            run = client.post("/runs", files={"file": ("c.csv", io.BytesIO(csv), "text/csv")},
                              data={"target": "default_flag"},
                              headers={"X-User-Email": "ci@example.com", "X-Groq-Key": "gsk_ci_only"}).json()
            with client.stream("GET", f"/runs/{run['id']}/events",
                               headers={"X-User-Email": "ci@example.com"}) as stream:
                stages = [json.loads(line[6:])["stage"] for line in stream.iter_lines() if line.startswith("data: ")]
            assert stages[0] == "queued" and stages[-1] == "done" and "critic" in stages
            final = client.get(f"/runs/{run['id']}", headers={"X-User-Email": "ci@example.com"}).json()
            assert final["status"] == "succeeded"
            assert client_redis.get(f"eda:secrets:{run['id']}") is None
            assert client_redis.llen("eda:processing") == 0
        finally:
            stop.set()
            thread.join(timeout=10)
    assert list((tmp_path / "uploads").glob("*")) == []
    client_redis.flushdb()
    time.sleep(0)
