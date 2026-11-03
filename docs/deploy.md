# Deploying to a server (Azure for Students)

The live site runs on one small Linux VM with Docker. It normally uses Azure's **free
1 GB VM** ("small" mode) and can be resized to **4 GB** ("large" mode, paid from the
student credit) when more capacity is needed.

```
internet ──HTTPS──> Caddy ──> web (Next.js) ──> api ──> postgres
                    :443                         │  └──> redis ──> worker(s)
```

Only Caddy is reachable from the internet. GitHub Actions builds the images after every
merge to `main` (once all tests pass) and publishes them to GitHub Container Registry;
the server only downloads them, so it never needs memory to build anything.

| Mode | VM size | Workers | Max upload | Max rows / columns | Cost |
|---|---|---|---|---|---|
| `small` | B2ats_v2 (2 vCPU, 1 GB) + 2 GB swap | 1 | 10 MB | 100,000 / 100 | free tier |
| `large` | B2als_v2 or B2s (2 vCPU, 4 GB) | 2 | 50 MB | 500,000 / 500 | credit, only while resized |

Measured: a 10 MB file peaks at about 360 MB of memory, a 49 MB file at about 1 GB.

## One-time setup

### 1. Make the images public (once, after the first publish)

GitHub → your profile → **Packages** → `agentic-eda-api` → **Package settings** →
**Change visibility → Public**. Repeat for `agentic-eda-web`. (The code is public
already; this lets the server download the images without a password.)

### 2. Create the VM (Azure portal)

**Virtual machines → Create → Azure virtual machine:**

| Setting | Value |
|---|---|
| Resource group | new: `agentic-eda` |
| Name | `agentic-eda` |
| Region | Central India (or South India / East US if the size is unavailable) |
| Image | Ubuntu Server 24.04 LTS - x64 Gen2 |
| Size | **B2ats_v2** ("free services eligible") |
| Authentication | SSH public key, username `azureuser`, your `azure_eda.pub` |
| Inbound ports | SSH (22), HTTP (80), HTTPS (443) |
| OS disk | 64 GiB, Premium SSD (P6), delete with VM |
| Public IP | new, **Static** |

Also create a budget: **Cost Management → Budgets** → $10/month, alerts at 50% and 100%.

### 3. DNS

At your domain provider, add one record:

| Type | Host | Value | TTL |
|---|---|---|---|
| A | `eda` | the VM's public IP | 300 (or default) |

Check from your PC: `nslookup eda.<your-domain>` should print the VM's IP.

### 4. A GitHub login app for the live site

GitHub → **Settings → Developer settings → OAuth Apps → New OAuth App**:

- Application name: `Agentic EDA`
- Homepage URL: `https://eda.<your-domain>`
- Authorization callback URL: `https://eda.<your-domain>/api/auth/callback/github`

Copy the **Client ID**, then **Generate a new client secret** and copy it (shown once).
Keep the local app from step 13 for `localhost`.

### 5. Server setup and first deploy

```powershell
ssh -i $HOME\.ssh\azure_eda azureuser@<VM-IP>
```

```bash
curl -fsSL https://raw.githubusercontent.com/adityaranjan7029-45/agentic-eda/main/deploy/server-setup.sh -o server-setup.sh
bash server-setup.sh          # Docker, firewall, swap, updates, code, nightly backup (~5 min)
exit                          # log out once so the docker group applies
```

```bash
ssh -i ~/.ssh/azure_eda azureuser@<VM-IP>
cd ~/agentic-eda
./deploy/deploy.sh small      # asks: domain, email, GitHub client ID + secret, optional Groq key
```

The first run writes `~/agentic-eda/.env` (all secrets generated on the server, file
readable only by you), downloads the images, starts everything, obtains the HTTPS
certificate and checks `https://eda.<your-domain>` answers.

## Everyday operations

| Task | Command (on the server, in `~/agentic-eda`) |
|---|---|
| Update after a merge to `main` | `./deploy/deploy.sh` (keeps the current mode) |
| Status, mode, memory | `./deploy/deploy.sh status` |
| Logs | `docker compose -f deploy/compose.yml logs -f --tail 100 api worker web caddy` |
| Roll back to an earlier version | `IMAGE_TAG=<commit sha> ./deploy/deploy.sh` |
| Backup now | `./deploy/backup.sh` (also runs nightly at 03:30, keeps 7 in `~/backups`) |

## Switching between free and 4 GB

**Up (an interview, a demo day):**
1. Azure portal → VM → **Size** → `B2als_v2` (or `B2s`) → **Resize**; wait for *Running*.
2. `./deploy/deploy.sh large`

**Back down (afterwards):**
1. `./deploy/deploy.sh small` first, so the app already fits in 1 GB.
2. Portal → VM → **Size** → `B2ats_v2` → **Resize**.
3. Next day: **Cost Management → Cost analysis** should be back to about $0 a day.

`deploy.sh large` refuses to run on a VM with less than ~3 GB of RAM, so the order can't
go wrong in that direction. The IP, disk, database and HTTPS certificate survive a resize.

## Hosting another project on the same server

Point another A record (e.g. `chat`) at the same IP, add the app's service to the
compose file and one block to `deploy/Caddyfile`:

```
chat.<your-domain> {
	reverse_proxy chat-app:8080
}
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `deploy.sh` stops at "didn't answer over HTTPS" | DNS not pointing at the VM yet (`nslookup`), or port 80/443 closed in the Azure network settings; `docker compose -f deploy/compose.yml logs caddy` |
| Pull fails: `denied` / `unauthorized` | the GHCR packages are still private (step 1) |
| GitHub login says "redirect_uri is not associated" | the OAuth app's callback URL doesn't match `https://eda.<your-domain>/api/auth/callback/github` exactly |
| A run fails with "the server restarted" | the worker ran out of memory: use a smaller file, or switch to large mode |
| Site slow after a period of heavy use | burstable CPU credits spent; they refill while idle, or resize up |
