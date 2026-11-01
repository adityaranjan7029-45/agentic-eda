"""Job queue and progress events: one interface, two backends.

    InlineBackend   jobs run in background threads inside the API process (dev, tests, Windows)
    RedisBackend    jobs go on a Redis list; separate `eda-worker` processes run them (production)

Both do the same three things:
    enqueue(job, secrets)      hand a job (and a user's own LLM keys, if any) to whoever runs it
    publish(run_id, event)     record a progress event
    events(run_id, after, ..)  read events after a position, waiting briefly for new ones

Redis layout:
    eda:queue                  list of waiting jobs (JSON)
    eda:processing             list of jobs a worker has taken (BLMOVE: never lost in between)
    eda:leases                 hash run_id -> time the worker started it (crash detection)
    eda:secrets:<run_id>       the user's keys, Fernet-encrypted, expires after a few minutes
    eda:events:<run_id>        Redis Stream of progress events (kept for an hour, replayable)
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from cryptography.fernet import Fernet, InvalidToken

QUEUE, PROCESSING, LEASES = "eda:queue", "eda:processing", "eda:leases"
EVENTS_TTL_SECONDS = 3600
FINAL_STATUSES = ("succeeded", "failed")


@dataclass(frozen=True)
class Job:
    run_id: str
    user_id: str
    path: str               # the uploaded file, waiting on disk (deleted by whoever runs the job)
    target: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> Job:
        return cls(**json.loads(raw))


def event(stage: str, detail: str = "", status: str = "running") -> dict:
    return {"stage": stage, "detail": detail, "status": status, "at": time.time()}


Runner = Callable[[Job, dict[str, str], Callable[[dict], None]], None]


# ---------------------------------------------------------------------------
# Inline backend
# ---------------------------------------------------------------------------

class InlineBackend:
    """Runs jobs in a thread pool. Users' keys stay in this process's memory only.

    `synchronous=True` runs the job inside `enqueue` (deterministic tests)."""

    mode = "inline"

    def __init__(self, runner: Runner, workers: int = 2, synchronous: bool = False):
        self._runner, self._synchronous = runner, synchronous
        self._pool = None if synchronous else ThreadPoolExecutor(workers, thread_name_prefix="eda-job")
        self._events: dict[str, list[tuple[str, dict]]] = {}
        self._changed = threading.Condition()

    def enqueue(self, job: Job, secrets: dict[str, str]) -> None:
        self.publish(job.run_id, event("queued", "waiting for a worker", "queued"))
        task = lambda: self._runner(job, dict(secrets), lambda e: self.publish(job.run_id, e))  # noqa: E731
        if self._synchronous:
            task()
        else:
            self._pool.submit(task)

    def publish(self, run_id: str, ev: dict) -> None:
        with self._changed:
            stream = self._events.setdefault(run_id, [])
            stream.append((str(len(stream) + 1), ev))
            self._changed.notify_all()

    def events(self, run_id: str, after: str | None = None, block_ms: int = 0) -> list[tuple[str, dict]]:
        position = int(after or 0)
        deadline = time.monotonic() + block_ms / 1000
        with self._changed:
            while True:
                stream = self._events.get(run_id, [])
                if len(stream) > position or time.monotonic() >= deadline:
                    return stream[position:]
                self._changed.wait(timeout=max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

def fernet_for(secret: str) -> Fernet:
    """Any long secret string -> a valid Fernet key (32 bytes, base64)."""
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


class RedisBackend:
    mode = "redis"

    def __init__(self, redis, secret_key: str, key_ttl_seconds: int = 600):
        self.redis, self._fernet, self._key_ttl = redis, fernet_for(secret_key), key_ttl_seconds

    # -- API side ----------------------------------------------------------

    def enqueue(self, job: Job, secrets: dict[str, str]) -> None:
        if secrets:
            token = self._fernet.encrypt(json.dumps(secrets).encode())
            self.redis.set(f"eda:secrets:{job.run_id}", token, ex=self._key_ttl)
        self.publish(job.run_id, event("queued", "waiting for a worker", "queued"))
        self.redis.lpush(QUEUE, job.to_json())

    def publish(self, run_id: str, ev: dict) -> None:
        key = f"eda:events:{run_id}"
        self.redis.xadd(key, {"data": json.dumps(ev)}, maxlen=500, approximate=True)
        self.redis.expire(key, EVENTS_TTL_SECONDS)

    def events(self, run_id: str, after: str | None = None, block_ms: int = 0) -> list[tuple[str, dict]]:
        key = f"eda:events:{run_id}"
        reply = self.redis.xread({key: after or "0"}, block=block_ms or None, count=100)
        out = []
        for _, entries in reply or []:
            for entry_id, fields in entries:
                entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                data = fields.get(b"data") or fields.get("data")
                out.append((entry_id, json.loads(data)))
        return out

    def queue_length(self) -> int:
        return self.redis.llen(QUEUE)

    # -- worker side -------------------------------------------------------

    def take(self, timeout_seconds: int = 5) -> tuple[Job, str] | None:
        """Move the next job to the processing list atomically (a crash can't lose it)."""
        raw = self.redis.blmove(QUEUE, PROCESSING, timeout_seconds, "RIGHT", "LEFT")
        if raw is None:
            return None
        job = Job.from_json(raw)
        self.redis.hset(LEASES, job.run_id, time.time())
        return job, raw if isinstance(raw, str) else raw.decode()

    def claim_secrets(self, run_id: str) -> dict[str, str]:
        """Read and delete the user's keys in one step. Expired or tampered -> no keys."""
        token = self.redis.getdel(f"eda:secrets:{run_id}")
        if not token:
            return {}
        try:
            return json.loads(self._fernet.decrypt(token))
        except (InvalidToken, ValueError):
            return {}

    def finish(self, job: Job, raw: str) -> None:
        self.redis.lrem(PROCESSING, 1, raw)
        self.redis.hdel(LEASES, job.run_id)

    def stale_jobs(self, timeout_seconds: float) -> list[tuple[Job, str]]:
        """Jobs a worker took but didn't finish within the timeout (the worker probably died)."""
        now = time.time()
        stale = []
        for raw in self.redis.lrange(PROCESSING, 0, -1):
            raw = raw if isinstance(raw, str) else raw.decode()
            job = Job.from_json(raw)
            started = self.redis.hget(LEASES, job.run_id)
            if started is None or now - float(started) > timeout_seconds:
                stale.append((job, raw))
        return stale
