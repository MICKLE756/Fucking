# ════════════════════════════════════════════════════════════
# ☰ Deploy Agent to Windows machine via WinRM
#
# Usage:
#   .\deploy.ps1 -TargetHost 10.0.0.5 -ServerHost 10.0.0.1
#   .\deploy.ps1 -TargetHost 192.168.1.100 -ServerHost 192.168.1.1 -ServerPort 8080
#
# Params:
#   -TargetHost   Target machine IP (required)
#   -ServerHost   Hub machine IP (required)
#   -ServerPort   Hub port (default 9910)
#   -AgentPath    Path on target for agent_dao.py (default C:\dao\agent_dao.py)
#
# Requires: WinRM enabled on target (Enable-PSRemoting -Force)
# ════════════════════════════════════════════════════════════

param(
    [Parameter(Mandatory=$true)]
    [string]$TargetHost,

    [Parameter(Mandatory=$true)]
    [string]$ServerHost,

    [int]   $ServerPort = 9910,
    [string]$AgentPath  = 'C:\dao\agent_dao.py'
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$AGENT_PY_FILE = Join-Path $PSScriptRoot 'agent_dao.py'

if (-not (Test-Path $AGENT_PY_FILE)) {
    Write-Host "[!] agent_dao.py not found in script directory" -ForegroundColor Red
    exit 1
}

# Base64 encode to survive WinRM serialization (all ASCII)
$bytes = [System.IO.File]::ReadAllBytes($AGENT_PY_FILE)
$b64 = [System.Convert]::ToBase64String($bytes)

Write-Host "=== Deploy Agent ===" -ForegroundColor Cyan
Write-Host "  Target: $TargetHost" -ForegroundColor White
Write-Host "  Server: ${ServerHost}:$ServerPort" -ForegroundColor White
Write-Host "  Path:   $AgentPath" -ForegroundColor White
Write-Host ""

# 1. WinRM check
Write-Host "[1/3] Testing WinRM..." -ForegroundColor Yellow
try {
    $test = Invoke-Command -ComputerName $TargetHost -ScriptBlock { hostname } -ErrorAction Stop
    Write-Host "  [+] Connected: $test" -ForegroundColor Green
} catch {
    Write-Host "  [!] WinRM unreachable: $_" -ForegroundColor Red
    Write-Host "      On target run as admin: Enable-PSRemoting -Force" -ForegroundColor Yellow
    exit 1
}

# 2. Deploy
Write-Host "[2/3] Deploying agent_dao.py..." -ForegroundColor Yellow

$result = Invoke-Command -ComputerName $TargetHost -ScriptBlock {
    param($b64content, $remotePath, $serverHost, $serverPort)

    Get-Process python,pythonw -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep 2

    $dir = Split-Path $remotePath -Parent
    if (-not (Test-Path $dir)) { New-Item -Path $dir -ItemType Directory -Force | Out-Null }
    $bytes = [System.Convert]::FromBase64String($b64content)
    [System.IO.File]::WriteAllBytes($remotePath, $bytes)

    $proc = Invoke-WmiMethod -Class Win32_Process -Name Create `
        -ArgumentList "pythonw `"$remotePath`" --server http://${serverHost}:${serverPort}"

    return "PID=$($proc.ProcessId)  ReturnValue=$($proc.ReturnValue)"
} -ArgumentList $b64, $AgentPath, $ServerHost, $ServerPort

Write-Host "  [+] $result" -ForegroundColor Green

# 3. Verify
Write-Host "[3/3] Verifying agent online..." -ForegroundColor Yellow
Start-Sleep 6

try {
    $agents = Invoke-RestMethod "http://localhost:$ServerPort/api/agents" `
        -Headers @{Authorization="Bearer dao-ps-agent-2026"} `
        -TimeoutSec 5
    $online = $agents.agents | Where-Object { $_.status -eq 'online' }
    Write-Host "  [+] Online agents: $($online.Count)" -ForegroundColor Green
    foreach ($a in $online) {
        Write-Host "      $($a.hostname) — $($a.ip) — $($a.user)" -ForegroundColor White
    }
} catch {
    Write-Host "  [~] Cannot query hub: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Deploy complete." -ForegroundColor Green
Write-Host ""
