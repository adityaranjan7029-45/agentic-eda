"""`eda-worker`: runs queued jobs from Redis (Redis mode only).

    eda-worker               run until stopped (Ctrl+C / SIGTERM finish the current job first)

Start one or more next to the API; they share the upload folder and the database.
On start-up and every minute it looks for jobs a dead worker left unfinished
and marks them failed, so no run stays "running" forever.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

import redis

from eda_api.config import Settings, get_settings
from eda_api.db import make_engine, make_sessionmaker
from eda_api.jobs import RedisBackend
from eda_api.service import fail_stale_run, process_run
from eda_api.uploads import purge_stale_uploads

log = logging.getLogger("eda_api.worker")
RECOVERY_INTERVAL_SECONDS = 60


class Worker:
    def __init__(self, settings: Settings, backend: RedisBackend, sessions):
        self.settings, self.backend, self.sessions = settings, backend, sessions

    def recover(self) -> int:
        stale = self.backend.stale_jobs(self.settings.job_timeout_minutes * 60)
        for job, raw in stale:
            log.warning("run %s: worker lost, marking failed", job.run_id)
            fail_stale_run(self.sessions, job, lambda e, rid=job.run_id: self.backend.publish(rid, e))
            self.backend.claim_secrets(job.run_id)          # make sure no key is left behind
            self.backend.finish(job, raw)
        return len(stale)

    def run_once(self, timeout_seconds: int = 5) -> str | None:
        """Take one job, run it, return its final status (None if the queue stayed empty)."""
        taken = self.backend.take(timeout_seconds)
        if taken is None:
            return None
        job, raw = taken
        secrets = self.backend.claim_secrets(job.run_id)    # read + delete: the key leaves Redis now
        try:
            return process_run(self.sessions, self.settings, job, secrets,
                               lambda e: self.backend.publish(job.run_id, e))
        finally:
            secrets.clear()
            self.backend.finish(job, raw)

    def run_forever(self, stop: threading.Event) -> None:
        purge_stale_uploads(self.settings.upload_dir, self.settings.stale_upload_minutes)
        next_recovery = 0.0
        while not stop.is_set():
            if time.monotonic() >= next_recovery:
                self.recover()
                next_recovery = time.monotonic() + RECOVERY_INTERVAL_SECONDS
            self.run_once(timeout_seconds=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if settings.queue_mode != "redis":
        raise SystemExit("eda-worker needs REDIS_URL (without it, the API runs jobs itself: inline mode)")
    backend = RedisBackend(redis.Redis.from_url(settings.redis_url), settings.secret_key.get_secret_value(),
                           settings.key_handover_seconds)
    worker = Worker(settings, backend, make_sessionmaker(make_engine(settings.database_url)))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (log.info("stopping after the current job"), stop.set()))
    log.info("worker ready (queue: %d waiting)", backend.queue_length())
    worker.run_forever(stop)
