# ☯ CF-DaoHub — Cloud Agent Guide

**One document to rule them all. Any cloud agent reads this, connects to all machines.**

---

## Access Point

```
CF URL:   https://<your-hub>.trycloudflare.com     (from ~/.dao/cf-hub-conn.json)
Token:    <your-master-token>                       (set in hub.ps1)
```

The CF URL changes on every cloudflared restart. Always read `~/.dao/cf-hub-conn.json` on the hub machine.

---

## Prerequisites

**Every target machine must have `agent_dao.py` running.** A cloud agent cannot wake a machine without an agent.

On the hub machine, `hub.ps1` starts the agent automatically. For other machines, deploy manually or via `deploy.ps1`.

Verify with `health()` — all target machines should show `online`.

---

## Machine Identities

Each machine registers with its hostname (auto-detected) or custom `--hostname`. Example:

| Agent ID | Alias | Meaning |
|----------|-------|---------|
| `MY-LAPTOP` | `laptop` / `10` | Primary machine |
| `MY-DESKTOP` | `desktop` / `20` | Secondary machine |

Agent IDs are case-insensitive. The `alias` table in `ps_agent_server.py` maps nicknames to full hostnames.

---

## Python SDK (copy-paste ready)

```python
import urllib.request, json, ssl, time, os

# === Proxy immunity + SSL skip (for trycloudflare self-signed certs) ===
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ctx),
)
urllib.request.install_opener(opener)

# === Hub config — CHANGE THESE ===
URL   = "https://<your-hub>.trycloudflare.com"
TOKEN = "<your-master-token>"

def api(method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{URL}{path}", data=data,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Content-Type": "application/json"},
        method=method)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def health():
    """Hub health check"""
    return api("GET", "/api/health")

def dao(cmd, agent="<primary-hostname>", timeout=30, retries=3):
    """
    Execute a command on the specified agent, return stdout string.

    agent: hostname or alias (e.g. 'MY-LAPTOP', 'laptop', 'desktop')
    """
    for i in range(retries):
        r = api("POST", "/api/exec-sync", {
            "agent_id": agent, "cmd": cmd, "timeout": timeout
        }, timeout=timeout + 15)
        if r.get("status") == "completed":
            return r["result"]["stdout"]
        if "not found" in str(r.get("error", "")):
            time.sleep(5)
    return f"[dao] unreachable: {agent}"

def dao_raw(cmd, agent="<primary-hostname>", timeout=30):
    """Execute command, return full result dict {stdout, stderr, exit_code}"""
    r = api("POST", "/api/exec-sync", {
        "agent_id": agent, "cmd": cmd, "timeout": timeout
    }, timeout=timeout + 15)
    return r.get("result", {}) if r.get("status") == "completed" else r
```

### Usage

```python
health()                              # {'status': 'ok', 'agents_online': 2, ...}

dao("hostname")                       # "MY-LAPTOP\n"
dao("hostname", "MY-DESKTOP")         # "MY-DESKTOP\n"
dao("whoami", "desktop")              # "desktop\\user\n"
dao("ipconfig", "laptop")

# Full result
r = dao_raw("dir C:\\", "MY-DESKTOP")
print(r["stdout"], r["stderr"], r["exit_code"])
```

---

## CLI Client (cf_cloud_agent.py)

No code needed — one-line commands:

```bash
# Health check
python cf_cloud_agent.py --url https://<hub>.trycloudflare.com --token <token> --health

# List agents
python cf_cloud_agent.py --url ... --token ... --agents

# Execute command
python cf_cloud_agent.py --url ... --token ... --exec "hostname"
python cf_cloud_agent.py --url ... --token ... --exec "dir C:\" --agent-id MY-DESKTOP

# Broadcast to all machines
python cf_cloud_agent.py --url ... --token ... --broadcast "echo ok"

# Auto-discover (reads ~/.dao/cf-hub-conn.json from current machine)
python cf_cloud_agent.py --auto --health
```

---

## API Reference

All requests require header: `Authorization: Bearer <token>`

### Queries

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/health` | `{"status":"ok","version":"3.4","agents_online":N}` |
| GET | `/api/agents` | `{"agents":[{"id","hostname","status","pending_commands",...}]}` |

### Command Execution

| Method | Path | Body | Note |
|--------|------|------|------|
| POST | `/api/exec-sync` | `{"agent_id":"...","cmd":"...","timeout":30}` | Sync, blocks until result |
| POST | `/api/exec` | `{"agent_id":"...","cmd":"..."}` | Async, returns `cmd_id` |
| POST | `/api/broadcast` | `{"type":"shell","payload":{"command":"..."}}` | Broadcast to all online agents |

### Extended Operations (via exec-sync type field)

| type | payload | Description |
|------|---------|-------------|
| `shell` | `{"command":"..."}` | Default, shell command |
| `screenshot` | `{}` | Screenshot, returns base64 PNG |
| `file_read` | `{"path":"C:\\..."}` | Read file, returns base64 |
| `file_write` | `{"path":"C:\\...","content_base64":"..."}` | Write file |

---

## Architecture

```
Cloud Agent ──CF HTTPS──> cloudflared ──> localhost:9910 (Server)
                                                ^
                        LAN Agent ──LAN direct──> <hub-IP>:9910
```

- **External clients**: connect via CF Tunnel public URL, globally reachable
- **LAN agents**: connect directly to hub IP, 1ms latency, no 408 timeout
- **Hub-local agent**: localhost connection, same code

Agent script: `agent_dao.py --server <hub-address>` — same code for LAN and CF.

---

## Auto-Discovery

Hub writes `~/.dao/cf-hub-conn.json` on startup:

```json
{
    "url": "https://<hub>.trycloudflare.com",
    "token": "<master-token>",
    "local_url": "http://localhost:9910",
    "port": 9910
}
```

Cloud agents can read this file for the latest URL instead of hardcoding.

---

*The Way follows its own nature. The best leader is barely known to exist.*
