#!/usr/bin/env bash
# Deploy or update Agentic EDA on this server, or switch size.
#
#   ./deploy/deploy.sh small    free 1 GB VM: 1 worker, 10 MB uploads   (default)
#   ./deploy/deploy.sh large    4 GB VM: 2 workers, 50 MB uploads
#   ./deploy/deploy.sh          redeploy in the current mode (e.g. after a merge to main)
#   ./deploy/deploy.sh status   show what's running and the current mode
#
# Rollback: IMAGE_TAG=<commit sha> ./deploy/deploy.sh   (images are tagged with each commit)
# First run asks a few questions and writes .env (secrets, mode 600). Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=.env
MODE_FILE=.deploy-mode
COMPOSE=(docker compose -f deploy/compose.yml --env-file "$ENV_FILE")

say()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
secret() { openssl rand -hex 32; }

mode="${1:-$(cat "$MODE_FILE" 2>/dev/null || echo small)}"

if [ "$mode" = "status" ]; then
  echo "mode: $(cat "$MODE_FILE" 2>/dev/null || echo 'not deployed yet')"
  "${COMPOSE[@]}" --env-file "deploy/$(cat "$MODE_FILE" 2>/dev/null || echo small).env" ps
  free -h | head -2
  exit 0
fi
case "$mode" in small | large) ;; *) fail "unknown mode '$mode' (use small, large or status)" ;; esac

# --- 1. settings and secrets (first run only) ---------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  say "First deployment: a few questions (answers are saved in .env, readable only by you)"
  read -rp "Domain for this app (e.g. eda.example.dev): " domain
  read -rp "Email for HTTPS certificate notices: " email
  read -rp "Live GitHub OAuth app - Client ID: " gh_id
  read -rsp "Live GitHub OAuth app - Client secret (hidden): " gh_secret; echo
  read -rsp "Server Groq API key (optional, Enter to skip; hidden): " groq; echo
  umask 077
  cat > "$ENV_FILE" <<ENV
# Agentic EDA production settings. Never commit or share this file.
DOMAIN=$domain
ACME_EMAIL=$email
IMAGE_REPO=ghcr.io/adityaranjan7029-45
AUTH_GITHUB_ID=$gh_id
AUTH_GITHUB_SECRET=$gh_secret
GROQ_API_KEY=$groq
AUTH_SECRET=$(secret)
API_JWT_SECRET=$(secret)
SECRET_KEY=$(secret)
POSTGRES_PASSWORD=$(secret)
REDIS_PASSWORD=$(secret)
ENV
  say "Saved .env"
fi
chmod 600 "$ENV_FILE"
for key in DOMAIN ACME_EMAIL IMAGE_REPO AUTH_GITHUB_ID AUTH_GITHUB_SECRET AUTH_SECRET API_JWT_SECRET SECRET_KEY POSTGRES_PASSWORD REDIS_PASSWORD; do
  grep -qE "^$key=.+" "$ENV_FILE" || fail "$key is empty in .env"
done

# --- 2. sanity checks ------------------------------------------------------------------
ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$mode" = large ] && [ "$ram_mb" -lt 3000 ]; then
  fail "large mode needs a 4 GB VM, this one has ${ram_mb} MB. Resize the VM in Azure first."
fi
domain=$(grep -E '^DOMAIN=' "$ENV_FILE" | cut -d= -f2-)

# --- 3. code, images, start ----------------------------------------------------------
say "Updating code"
git pull --ff-only --quiet || say "git pull skipped (local changes or no network); deploying the current checkout"

COMPOSE+=(--env-file "deploy/$mode.env")
say "Pulling images (${IMAGE_TAG:-latest})"
"${COMPOSE[@]}" pull --quiet

say "Starting in $mode mode"
"${COMPOSE[@]}" up -d --no-build --remove-orphans --wait --wait-timeout 300
echo "$mode" > "$MODE_FILE"
docker image prune -f >/dev/null

# --- 4. check it from the outside ------------------------------------------------------
say "Checking https://$domain"
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null --max-time 10 "https://$domain/"; then
    say "Live: https://$domain  (mode: $mode)"
    "${COMPOSE[@]}" ps --format 'table {{.Service}}\t{{.Status}}'
    exit 0
  fi
  sleep 5
done
fail "The site didn't answer over HTTPS. Check DNS (A record -> this VM's IP) and: docker compose -f deploy/compose.yml logs caddy"
