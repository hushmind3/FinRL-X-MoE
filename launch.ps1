$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$vite = Join-Path $root 'node_modules\vite\bin\vite.js'
$runtime = Join-Path $root 'runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $vite)) {
  throw '먼저 실행.cmd를 눌러 설치와 실행을 진행하세요.'
}

function Test-Service([string]$url, [string]$expected) {
  try {
    $response = Invoke-RestMethod -Uri $url -TimeoutSec 2
    if ($expected -eq 'api') {
      return $response.service -eq 'finrlx-backend' -and
        $response.project_id -eq 'finrlx-trading-lab' -and
        $response.runtime_contract -eq 2
    }
    $page = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
    return $page.Content -match '<title>FinRL-X Trading Lab</title>'
  } catch { return $false }
}

function Stop-StaleWorkspaceListener([int]$port, [string]$marker) {
  $listeners = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
  foreach ($listener in $listeners) {
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($null -eq $processInfo) { continue }
    $sameWorkspace = $processInfo.CommandLine -like "*$root*" -and $processInfo.CommandLine -like "*$marker*"
    if (!$sameWorkspace) {
      throw "Port $port is occupied by another process (PID $($listener.OwningProcess)); stop it or change the configured port."
    }
    Stop-Process -Id $listener.OwningProcess -Force
  }
  for ($i = 0; $i -lt 20 -and (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue); $i++) {
    Start-Sleep -Milliseconds 250
  }
}

$healthUrl = 'http://127.0.0.1:8878/api/v1/health'
$uiUrl = 'http://127.0.0.1:5178'
if (!(Test-Service $healthUrl 'api')) {
  Stop-StaleWorkspaceListener 8878 'uvicorn finrlx.api:app'
  $apiPid = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','finrlx.api:app','--app-dir','backend','--host','127.0.0.1','--port','8878') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'backend.log') -RedirectStandardError (Join-Path $runtime 'backend-error.log') -PassThru
  Set-Content -LiteralPath (Join-Path $runtime 'backend.pid') -Value $apiPid.Id -Encoding ascii
}
$node = (Get-Command node.exe -ErrorAction Stop).Source
try { $nodeProcessId = [int](Get-Content -LiteralPath (Join-Path $runtime 'frontend.pid') -ErrorAction Stop) } catch { $nodeProcessId = 0 }
$frontendAlive = $false
if ($nodeProcessId) {
  $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $nodeProcessId" -ErrorAction SilentlyContinue
  $frontendAlive = $null -ne $processInfo -and $processInfo.CommandLine -like "*$root*" -and $processInfo.CommandLine -like '*vite.js*5178*'
}
if (!(Test-Service $uiUrl 'ui') -and !$frontendAlive) {
  Stop-StaleWorkspaceListener 5178 'vite.js'
  $uiPid = Start-Process -FilePath $node -ArgumentList @($vite,'--host','127.0.0.1','--port','5178','--strictPort') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'frontend.log') -RedirectStandardError (Join-Path $runtime 'frontend-error.log') -PassThru
  Set-Content -LiteralPath (Join-Path $runtime 'frontend.pid') -Value $uiPid.Id -Encoding ascii
}

$apiReady = $false
$uiReady = $false
for ($i = 0; $i -lt 60; $i++) {
  if (!$apiReady) { $apiReady = Test-Service $healthUrl 'api' }
  if (!$uiReady) { $uiReady = Test-Service $uiUrl 'ui' }
  if ($apiReady -and $uiReady) { break }
  Start-Sleep -Seconds 1
}
if (!$apiReady -or !$uiReady) {
  throw "Service startup failed. API=$apiReady UI=$uiReady. Check runtime logs."
}
Start-Process $uiUrl
Write-Host 'FinRL-X is running'
Write-Host "Dashboard: $uiUrl"
Write-Host 'API docs: http://127.0.0.1:8878/docs'
