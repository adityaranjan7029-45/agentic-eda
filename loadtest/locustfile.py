"""Load test for the API, the queue and the workers.

Simulates signed-in analysts hitting the API the way the web app does (each with a
short-lived HS256 token, like the Next.js proxy mints). Most traffic is reading
(history, reports); some users start analyses and follow the live progress stream
until the report is ready.

    $env:LOADTEST_JWT_SECRET = "<API_JWT_SECRET from .env>"
    .venv-load\\Scripts\\locust -f loadtest/locustfile.py --host http://localhost:8000 `
        --headless -u 40 -r 5 -t 3m --csv reports/load

Pass/fail (the run exits with code 1 if any is missed):
    errors                       < 1% of requests (429 "too many runs" is expected, not an error)
    p95 GET /runs                < 500 ms
    p95 GET /runs/{id}           < 800 ms
    p95 analysis, queued->done   < 60 s
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid

import jwt
from locust import HttpUser, between, events, task

SECRET = os.environ.get("LOADTEST_JWT_SECRET", "")
TOKEN_TTL = 300
THRESHOLDS_MS = {"GET /runs": 500, "GET /runs/{id}": 800, "analysis: queued -> done": 60_000}
MAX_ERROR_RATE = 0.01


@events.init.add_listener
def _check_config(environment, **_):
    if len(SECRET) < 32:
        raise SystemExit("Set LOADTEST_JWT_SECRET to the API's API_JWT_SECRET (32+ characters)")


class Analyst(HttpUser):
    """One signed-in person. Reads a lot, starts an analysis now and then."""

    wait_time = between(1, 4)

    def on_start(self):
        self.account = str(random.randint(10**8, 10**9 - 1))
        self.token, self.token_expires = "", 0.0
        self.run_ids: list[str] = []

    # --- helpers -------------------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        now = time.time()
        if now > self.token_expires - 30:   # renew before it lapses, like the proxy does per request
            claims = {"sub": f"github:{self.account}", "email": f"load-{self.account}@example.test",
                      "name": "Load Test", "iss": "agentic-eda-web", "aud": "agentic-eda-api",
                      "iat": int(now), "exp": int(now) + TOKEN_TTL, "jti": uuid.uuid4().hex}
            self.token, self.token_expires = jwt.encode(claims, SECRET, algorithm="HS256"), now + TOKEN_TTL
        return {"Authorization": f"Bearer {self.token}"}

    # --- reading (most traffic) ----------------------------------------------------------
    @task(6)
    def history(self):
        with self.client.get("/runs?limit=50", headers=self.headers(), name="GET /runs",
                             catch_response=True) as r:
            if r.status_code == 200:
                self.run_ids = [x["id"] for x in r.json()["runs"] if x["status"] == "succeeded"]
                r.success()
            else:
                r.failure(f"status {r.status_code}")

    @task(3)
    def open_report(self):
        if not self.run_ids:
            return
        self.client.get(f"/runs/{random.choice(self.run_ids)}", headers=self.headers(),
                        name="GET /runs/{id}")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")

    # --- writing: start an analysis and follow it live -----------------------------------
    @task(1)
    def analyse_sample(self):
        started = time.perf_counter()
        with self.client.post("/runs/sample", headers=self.headers(), name="POST /runs/sample",
                              catch_response=True) as r:
            if r.status_code == 429:          # two already in progress: the limit working as designed
                r.success()
                return
            if r.status_code != 202:
                r.failure(f"status {r.status_code}")
                return
            run_id = r.json()["id"]
        self.follow(run_id, started)

    def follow(self, run_id: str, started: float):
        """Read the SSE stream until the run finishes; report queued->done as its own metric."""
        outcome, error = None, None
        try:
            with self.client.get(f"/runs/{run_id}/events", headers=self.headers(), stream=True,
                                 name="GET /runs/{id}/events (stream)", timeout=180,
                                 catch_response=True) as stream:
                for line in stream.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("status") in ("succeeded", "failed"):
                            outcome = data["status"]
                            break
                stream.success()   # the stream's own time is just the first byte; the run metric below covers it
        except Exception as exc:  # noqa: BLE001 - any stream failure counts against the run
            error = exc
        if outcome != "succeeded" and error is None:
            error = RuntimeError(f"run ended as {outcome or 'nothing (stream closed early)'}")
        events.request.fire(request_type="RUN", name="analysis: queued -> done",
                            response_time=(time.perf_counter() - started) * 1000, response_length=0,
                            exception=error, context={})


@events.quitting.add_listener
def _verdict(environment, **_):
    stats = environment.stats
    problems = []
    total = stats.total
    if total.num_requests and total.num_failures / total.num_requests > MAX_ERROR_RATE:
        problems.append(f"error rate {total.num_failures / total.num_requests:.1%} > {MAX_ERROR_RATE:.0%}")
    for name, limit in THRESHOLDS_MS.items():
        entry = next((e for (n, _m), e in stats.entries.items() if n == name), None)
        if entry is None or entry.num_requests == 0:
            problems.append(f"no data for {name}")
            continue
        p95 = entry.get_response_time_percentile(0.95)
        if p95 > limit:
            problems.append(f"p95 {name} = {p95:.0f} ms > {limit} ms")
    if problems:
        print("LOAD TEST FAILED: " + "; ".join(problems))
        environment.process_exit_code = 1
    else:
        print("LOAD TEST PASSED: all thresholds met")
