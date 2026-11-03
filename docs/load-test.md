# Load test

`loadtest/locustfile.py` simulates signed-in analysts using the API the way the web app
does: each virtual user gets its own GitHub-style identity and a 5-minute HS256 token
(like the Next.js proxy mints), then mostly reads (history, reports, health) and now and
then starts an analysis of the sample dataset and follows its live progress stream until
the report is ready. "analysis: queued -> done" is measured end to end: from the upload
request to the final `done` event.

The run fails (exit code 1) unless:

| Check | Limit |
|---|---|
| failed requests | < 1% (a 429 "two runs already in progress" is the limit working, not a failure) |
| p95 `GET /runs` | < 500 ms |
| p95 `GET /runs/{id}` | < 800 ms |
| p95 analysis, queued -> done | < 60 s |

## Results (step 15)

**Setup:** the production configuration (API with `APP_ENV=prod` and Redis mode,
2 `eda-worker` processes, Postgres 16, Redis 7) on a 2-vCPU, 8 GB machine; the load
generator ran on the same machine. 40 users, ramped up at 5 per second, 3 minutes:
`locust --headless -u 40 -r 5 -t 3m`.

| Endpoint | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | req/s |
|---|---:|---:|---:|---:|---:|---:|
| `GET /runs` (history) | 775 | 0 | 13 | 180 | 310 | 4.32 |
| `GET /runs/{id}` (report) | 243 | 0 | 23 | 210 | 300 | 1.36 |
| `GET /health` | 146 | 0 | 8 | 160 | 240 | 0.81 |
| `POST /runs/sample` (generate + queue) | 127 | 0 | 630 | 1,300 | 2,000 | 0.71 |
| `GET /runs/{id}/events` (first byte) | 107 | 0 | 8 | 160 | 190 | 0.60 |
| **analysis: queued -> done** | 107 | 0 | 31,000 | 40,000 | 41,000 | 0.60 |
| **all** | 1,505 | **0** | 18 | | | 8.39 |

Verdict: **passed**. Afterwards, all 405 runs in the database (from the e2e and load
test sessions) had succeeded, the
queue and the in-progress list were empty, and the upload folder was empty (every file
deleted after reading). No errors in the API or worker logs.

### What the numbers say

- **The API stays fast while the workers are saturated.** Reads stay at a 13–23 ms
  median even when the queue is long, because analyses never run inside the API in
  Redis mode. This is the reason for the queue (step 12).
- **Throughput is set by the workers.** One analysis takes about 3 seconds on its own;
  two workers on 2 vCPUs finish about 0.6 per second (~36 per minute). This test offered
  0.71 per second, so the queue grew, and the waiting time (median 31 s) is almost all of
  the end-to-end time. More workers on more cores raise this linearly
  (`docker compose up -d --scale worker=4`); on 2 vCPUs they would only compete for CPU.
- **`POST /runs/sample` is the slowest request** because it generates a 3,000-row
  synthetic dataset in the API before queueing it. Real uploads skip that step.

## Problems this test and the end-to-end tests found (fixed in step 15)

1. **Live progress broke whenever a run waited more than 5 seconds in the queue.**
   redis-py 8 changed its default socket timeout to 5 s, but the progress stream waits up
   to 15 s for the next event. The read failed, the API's stream crashed, and the browser
   showed `ERR_INCOMPLETE_CHUNKED_ENCODING`. Only visible under load, when runs queue.
   Fix: all Redis clients are made by `connect_redis()` with a 30 s socket timeout (longer
   than any blocking read). Tests: a real-Redis test that blocks for 7 s, and a unit test
   that the timeout outlasts every blocking read.
2. **The "two runs in progress" limit could be bypassed by simultaneous requests.**
   Three uploads sent at the same instant all counted "0 in progress" before any had
   saved, so all three started. Fix: `accept_run` locks the user's row
   (`SELECT ... FOR UPDATE`) until the new run is saved, so a user's uploads are counted
   one after another. Test: six simultaneous uploads against real Postgres, exactly two
   accepted.
3. **API memory reached ~935 MB under load** (glibc keeps a memory arena per thread).
   `MALLOC_ARENA_MAX=2` in the image brought it to ~500 MB with the same latency
   (re-run above).

Each fix was checked by undoing it and confirming its test fails.

## Running it

See `docs/docker.md` section 4. Point `--host` at any running API; set
`LOADTEST_JWT_SECRET` to that API's `API_JWT_SECRET`. Use `--csv reports/load` to save
the raw numbers. CI runs a 45-second, 10-user version against the Docker stack on every push.
