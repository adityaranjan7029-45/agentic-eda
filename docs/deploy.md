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
| `small` | B2ats_v2 (2 vCPU, 1 GB) + 2 GB swap | 1 | 10 MB | 100,000 / 200 | free tier |
| `large` | B2als_v2 or B2s (2 vCPU, 4 GB) | 2 | 50 MB | 500,000 / 500 | credit, only while resized |

Measured: a 10 MB file peaks at about 360 MB of memory (100,000 rows × 12 columns, or 7,000 rows × 200 columns), a 49 MB file at about 1 GB. The sample credit dataset (122 columns) must fit every mode; `api/tests/test_deploy_modes.py` checks it.

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

### 4b. Google sign-in (optional, recommended for business users)

The "Sign in with Google" button appears (first) as soon as the server has a Google OAuth client;
without one, only GitHub is shown. Users are stored as (provider, account id), so the same person
signing in with GitHub and with Google gets two separate accounts.

1. [console.cloud.google.com](https://console.cloud.google.com) → create a project (e.g. `agentic-eda`).
2. **APIs & Services → OAuth consent screen** (also called *Google Auth Platform → Branding*):
   User type **External**; app name `Agentic EDA`; your support email; app home page `https://eda.<your-domain>`;
   scopes: only the defaults (`openid`, `email`, `profile`). Don't upload a logo (that triggers Google's verification).
3. **Audience** → **Publish app** (status *In production*). While it says *Testing*, only the test users
   you list can sign in.
4. **Credentials → Create credentials → OAuth client ID** → type **Web application**:
   - Authorized JavaScript origins: `https://eda.<your-domain>`
   - Authorized redirect URIs: `https://eda.<your-domain>/api/auth/callback/google`
   - (for local dev, add `http://localhost:3000` and `http://localhost:3000/api/auth/callback/google`)
5. Copy the **Client ID** and **Client secret**. On a new server `deploy.sh` asks for them; on an
   existing one, add them to `~/agentic-eda/.env`:
   ```bash
   cd ~/agentic-eda
   read -rp "Google client ID: " gid; read -rsp "Google client secret (hidden): " gs; echo
   sed -i '/^AUTH_GOOGLE_/d' .env
   printf 'AUTH_GOOGLE_ID=%s\nAUTH_GOOGLE_SECRET=%s\n' "$(echo "$gid" | tr -d '[:space:]')" "$(echo "$gs" | tr -d '[:space:]')" >> .env
   unset gid gs
   ./deploy/deploy.sh
   ```

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
| Logs (Ctrl+C to stop) | `./deploy/deploy.sh logs` or e.g. `./deploy/deploy.sh logs web caddy` |
| Roll back to an earlier version | `IMAGE_TAG=<commit sha> ./deploy/deploy.sh` |
| Backup now | `./deploy/backup.sh` (also runs nightly at 03:30, keeps 7 in `~/backups`) |

## Where each command runs

| Where | Prompt looks like | Used for |
|---|---|---|
| **PowerShell on your PC** | `PS E:\agentic-eda>` | git, pushing code, SSH into the VM |
| **Bash on the VM** (after SSH) | `azureuser@agentic-eda:~$` | `server-setup.sh`, `deploy.sh`, logs, backups |
| **Azure Cloud Shell** (`>_` icon in the portal, choose Bash) | `you@Azure:~$` | managing Azure itself, e.g. resizing the VM (same as the portal clicks) |

Log in to the VM from your PC:

```powershell
ssh -i $HOME\.ssh\azure_eda azureuser@<VM-IP>
```

Your VM: resource group `agentic-eda`, name `agentic-eda`, region India South Central.

## Switching between free and 4 GB

What changes is in `deploy/small.env` and `deploy/large.env` (workers, upload limits, memory caps);
`deploy/compose.yml` reads them and `deploy.sh small|large` picks one. The VM size itself is changed in Azure.

| Mode | VM size | Workers | Max upload |
|---|---|---|---|
| small (normal, free) | `Standard_B2ats_v2` (1 GB) | 1 | 10 MB |
| large (busy days, credit) | `Standard_B2als_v2` (4 GB), or `Standard_B2s` | 2 | 50 MB |

**Why two places?** They control two different things:

| Where | What it changes | Why it's separate |
|---|---|---|
| **Azure** (portal or Cloud Shell) | the **hardware**: 1 GB ⇄ 4 GB RAM | only Azure can change the machine's size |
| **The VM** (over SSH, `deploy.sh`) | the **app's settings**: workers, upload limit, memory caps | only the server knows how the app runs |

Order matters: going **up**, resize first (more room), then `deploy.sh large`. Going **down**, `deploy.sh small`
first (the app shrinks), then resize. `deploy.sh large` refuses to run on a VM with less than ~3 GB, so it can't
overload the free server by mistake.

**Up (an interview, a demo day):**

1. Resize the VM, in **Cloud Shell (Bash)**:
   ```bash
   az vm resize -g agentic-eda -n agentic-eda --size Standard_B2als_v2
   ```
   or in the portal: VM → **Size** → `B2als_v2` → **Resize**. The VM restarts (~2 minutes).
2. On your **PC**: `ssh -i $HOME\.ssh\azure_eda azureuser@<VM-IP>`
3. On the **VM**:
   ```bash
   cd ~/agentic-eda && ./deploy/deploy.sh large
   ```
4. Set a reminder to switch back.

**Back down (afterwards), app first, then the VM:**

1. On the **VM**:
   ```bash
   cd ~/agentic-eda && ./deploy/deploy.sh small
   ```
2. In **Cloud Shell (Bash)**:
   ```bash
   az vm resize -g agentic-eda -n agentic-eda --size Standard_B2ats_v2
   ```
   or in the portal: VM → **Size** → `B2ats_v2` → **Resize**.
3. Next day: **Cost Management → Cost analysis** should be back to about $0 a day.

Check the current mode any time on the VM: `./deploy/deploy.sh status`.

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
| First deploy prints `curl: (35) ... tlsv1 alert internal error` once or twice, then "Live" | normal: Caddy was still getting the HTTPS certificate |
| `deploy.sh` stops at "didn't answer over HTTPS" | DNS not pointing at the VM yet (`nslookup`), or port 80/443 closed in the Azure network settings; `./deploy/deploy.sh logs caddy` |
| Sign-in shows "Server error: problem with the server configuration" | `./deploy/deploy.sh logs web`. `incorrect_client_credentials` = wrong GitHub client secret: generate a new one, fix `AUTH_GITHUB_SECRET` in `.env` (no spaces), run `./deploy/deploy.sh` |
| Pull fails: `denied` / `unauthorized` | the GHCR packages are still private (step 1) |
| Google says "Error 400: redirect_uri_mismatch" | the redirect URI in the Google client must be exactly `https://eda.<your-domain>/api/auth/callback/google` |
| Google says "Access blocked: app can only be used within its organization" or only some people can sign in | the consent screen is still in *Testing* or *Internal*: set it to External and **Publish app** |
| GitHub says "redirect_uri is not associated" | the OAuth app's callback URL doesn't match `https://eda.<your-domain>/api/auth/callback/github` exactly |
| A run fails with "the server restarted" | the worker ran out of memory: use a smaller file, or switch to large mode |
| Site slow after a period of heavy use | burstable CPU credits spent; they refill while idle, or resize up |
