# API + worker image (same code, different command).
#   docker compose runs it as: migrate (alembic), api (uvicorn), worker (eda-worker)
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Exact dependency versions first (cached unless the lock file changes).
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

# 2. Our two packages, without re-resolving dependencies.
COPY engine/pyproject.toml engine/pyproject.toml
COPY engine/src engine/src
COPY api/pyproject.toml api/pyproject.toml
COPY api/src api/src
COPY api/alembic.ini api/alembic.ini
COPY api/alembic api/alembic
RUN pip install --no-deps ./engine ./api

# 3. Run as an unprivileged user. The upload folder is created here so the
#    named volume mounted on it starts with this owner (API and workers share it).
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/uploads \
    && chown app:app /data/uploads \
    && chmod 700 /data/uploads
USER app

# MALLOC_ARENA_MAX=2: fewer glibc memory arenas. Under the step-15 load test this cut the
# API's memory from ~935 MB to ~500 MB with no change in latency.
ENV UPLOAD_DIR=/data/uploads \
    MALLOC_ARENA_MAX=2
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=5 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# --proxy-headers: behind Caddy (step 16) the client IP and scheme come from X-Forwarded-*.
CMD ["uvicorn", "eda_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
