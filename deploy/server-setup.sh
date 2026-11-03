#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server (Azure B2ats_v2 or bigger).
# Run once as the normal user (azureuser):  bash server-setup.sh
#
#   - Docker Engine + Compose, with log rotation
#   - firewall: only SSH, HTTP, HTTPS
#   - 2 GB swap (the free VM has 1 GB of RAM)
#   - automatic security updates
#   - the app's code in ~/agentic-eda and a nightly database backup
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/adityaranjan7029-45/agentic-eda.git}"
APP_DIR="$HOME/agentic-eda"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

step "System updates"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -yq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq ca-certificates curl git ufw unattended-upgrades openssl

step "Docker"
if ! command -v docker >/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091
  codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $codename stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
# keep container logs from filling the disk
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }
JSON
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"

step "Firewall (SSH, HTTP, HTTPS only)"
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw --force enable

step "Swap (2 GB)"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-agentic-eda.conf >/dev/null
sudo sysctl -q --system

step "Automatic security updates"
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

step "App code"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi

step "Nightly database backup (03:30, keeps 7)"
mkdir -p "$HOME/backups"
( crontab -l 2>/dev/null | grep -v 'deploy/backup.sh' ; echo "30 3 * * * bash $APP_DIR/deploy/backup.sh >> $HOME/backups/backup.log 2>&1" ) | crontab -

step "Done"
echo "Log out and back in (so the docker group applies), then:"
echo "  cd ~/agentic-eda && ./deploy/deploy.sh small"
