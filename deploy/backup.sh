#!/usr/bin/env bash
# Nightly Postgres backup (cron from server-setup.sh). Keeps the newest 7.
# Restore: gunzip -c ~/backups/<file>.sql.gz | docker compose -f deploy/compose.yml --env-file .env \
#            --env-file deploy/small.env exec -T postgres psql -U eda -d eda
set -euo pipefail
cd "$(dirname "$0")/.."
mode=$(cat .deploy-mode 2>/dev/null || echo small)
dir="$HOME/backups"
mkdir -p "$dir"
file="$dir/eda-$(date +%Y%m%d-%H%M).sql.gz"
docker compose -f deploy/compose.yml --env-file .env --env-file "deploy/$mode.env" \
  exec -T postgres pg_dump -U eda -d eda --clean --if-exists | gzip > "$file"
chmod 600 "$file"
find "$dir" -maxdepth 1 -name 'eda-*.sql.gz' -printf '%T@ %p\n' | sort -rn | tail -n +8 | cut -d' ' -f2- | xargs -r rm --
echo "$(date -Is) backup ok: $file ($(du -h "$file" | cut -f1))"
