# ☯ CF-DaoHub — 道法自然

**三进程。六文件。全球可达。**

一台机器跑中枢，其余机器跑 Agent。云端 Agent 通过 Cloudflare 公网直连所有设备。

---

## 架构

```
                        Internet                          LAN
  ┌──────────┐     ┌──────────────┐     ┌────────────────────────┐
  │ Cloud    │────▶│ cloudflared  │────▶│ ps_agent_server.py     │
  │ Agent    │     │ Quick Tunnel │     │ :9910                  │
  │ (anywhere)│    │ (HTTPS/HTTP2)│     │                        │
  └──────────┘     └──────────────┘     │ ┌────────────────────┐ │
                                        │ │ agent_dao.py       │ │
                                        │ │ -> localhost:9910  │ │<- Hub machine
                                        │ └────────────────────┘ │
                                        │                        │
                                        │ ┌────────────────────┐ │
                                        │ │ agent_dao.py       │ │
                                        │ │ -> 192.168.x.x:9910│ │<- LAN machines
                                        │ └────────────────────┘ │
                                        └────────────────────────┘
```

- **Hub**: One machine runs `hub.ps1` -> launches Server + local Agent + cloudflared tunnel
- **Agent**: `agent_dao.py` -> long-polls hub, runs commands, reports results
- **Cloud Client**: Anyone connects via CF URL + Token from anywhere

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ (stdlib only, zero pip deps) |
| cloudflared | [Download](https://github.com/cloudflare/cloudflared/releases), put in PATH or anywhere |
| WinRM | Only needed for `deploy.ps1`, optional |
| Firewall | Port 9910 open on LAN only (for agent LAN connections) |

---

## Files (6)

| File | Role | Runs On |
|------|------|---------|
| `README.md` | This doc — universal entry point | Readme |
| `hub.ps1` | One-click hub launcher (Server + Agent + Tunnel) | Hub machine |
| `agent_dao.py` | Universal agent daemon (LAN/CF dual mode) | All managed machines |
| `deploy.ps1` | Remote deploy agent to another Windows machine | Hub machine |
| `CLOUD_AGENT_GUIDE.md` | Cloud agent guide (SDK + API reference) | Cloud agents |
| `cf_cloud_agent.py` | CLI client (health/exec/broadcast) | Anywhere |

---

## Quick Start (3 steps)

### Step 1: Launch Hub

On the hub machine:

```powershell
.\hub.ps1
```

Automatically:
1. Starts `ps_agent_server.py` on port 9910
2. Starts `agent_dao.py` registering as local machine
3. Starts `cloudflared tunnel` generating public URL
4. Writes `~/.dao/cf-hub-conn.json`

Example output:
```
=== CF-DaoHub Running ===
  Local:  http://localhost:9910
  Public: https://xxx-yyy-zzz.trycloudflare.com
  Token:  dao-ps-agent-2026
```

> **Custom port/token**: Edit `$PORT` and `$TOKEN` at top of `hub.ps1`.
>
> **cloudflared auto-discovery**: `hub.ps1` searches PATH, winget, scoop, choco, and current directory. No manual path config.

---

### Step 2: Deploy Agent to other machines

#### Option A: Remote deploy (WinRM)

```powershell
.\deploy.ps1 -TargetHost 10.0.0.5 -ServerHost 10.0.0.1
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `-TargetHost` | ✅ | — | Target machine IP |
| `-ServerHost` | ✅ | — | Hub machine IP |
| `-ServerPort` | — | `9910` | Hub port |
| `-AgentPath` | — | `C:\dao\agent_dao.py` | Agent path on target |

#### Option B: Manual

On the target machine:

```powershell
python agent_dao.py --server http://<hub-IP>:9910
```

---

### Step 3: Connect from Cloud

Copy the Python SDK from `CLOUD_AGENT_GUIDE.md`, or use CLI:

```bash
# Health check
python cf_cloud_agent.py --url https://xxx.trycloudflare.com --health

# List agents
python cf_cloud_agent.py --url https://xxx.trycloudflare.com --agents

# Execute command
python cf_cloud_agent.py --url https://xxx.trycloudflare.com --exec hostname

# Broadcast to all
python cf_cloud_agent.py --url https://xxx.trycloudflare.com --broadcast "whoami"
```

---

## Configuration

### Change hub port

Edit `hub.ps1`:
```powershell
$PORT = 8080
```

### Change master token

Edit `hub.ps1`:
```powershell
$TOKEN = 'your-custom-token'
```

All API requests must carry `Authorization: Bearer your-custom-token`.

### Custom agent hostname

```powershell
python agent_dao.py --server http://localhost:9910 --hostname MY-ALIAS
```

Defaults to system hostname.

---

## API Reference

All endpoints require `Authorization: Bearer <TOKEN>` header.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Hub health + online agent count |
| GET | `/api/agents` | All agent list |
| POST | `/api/exec-sync` | Sync execute `{"agent_id":"...","cmd":"..."}` |
| POST | `/api/exec` | Async execute (returns cmd_id) |
| POST | `/api/broadcast` | Broadcast to all online agents |
| GET | `/api/agent/<id>/screenshot` | Screenshot |
| GET | `/api/agent/<id>/download?path=C:\...` | Download file |

Full reference in `CLOUD_AGENT_GUIDE.md`.

---

## Adding a third machine

```powershell
# On the new machine
python agent_dao.py --server http://<hub-IP>:9910
```

Hub auto-discovers new agents. Cloud agents can reach it immediately.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| cloudflared not found | Not in PATH | `winget install Cloudflare.cloudflared` or download to current dir |
| No tunnel URL | QUIC blocked | `hub.ps1` uses `--protocol http2`; if still failing check network |
| Agent can't reach hub | Firewall/wrong IP | Hub machine allow inbound 9910; use correct LAN IP |
| Deploy agent dies | Start-Process unreliable in WinRM | `deploy.ps1` uses `Invoke-WmiMethod Win32_Process` |

---

## Migrating from legacy

If you used the old version (`cf_start_hub.ps1`, `cf_watchdog.ps1`, scheduled tasks):

```powershell
Get-Process python, cloudflared -EA SilentlyContinue | Stop-Process -Force
Unregister-ScheduledTask -TaskName "CFDaoHub*" -Confirm:$false -EA SilentlyContinue
.\hub.ps1
```

---

☯ The best leader is one whose existence is barely known. Water benefits all things without contention. The Way follows its own nature.
