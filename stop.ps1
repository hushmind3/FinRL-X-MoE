$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$runtime = Join-Path $root 'runtime'
foreach ($name in @('frontend','backend')) {
  $pidFile = Join-Path $runtime "$name.pid"
  if (!(Test-Path -LiteralPath $pidFile)) { continue }
  $processId = 0
  try { $processId = [int](Get-Content -LiteralPath $pidFile -ErrorAction Stop) } catch {}
  if (!$processId) { continue }
  $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
  if (!$processInfo) { Remove-Item -LiteralPath $pidFile -Force; continue }
  $command = [string]$processInfo.CommandLine
  $belongsToApp = if ($name -eq 'backend') {
    $command -like '*finrlx.api:app*' -and $command -like '*8878*'
  } else {
    $command -like "*$root*node_modules\vite\bin\vite.js*" -and $command -like '*5178*'
  }
  if ($belongsToApp) { Stop-Process -Id $processId -Force }
  Remove-Item -LiteralPath $pidFile -Force
}
Write-Host 'FinRL-X 프로세스를 종료했습니다.'
