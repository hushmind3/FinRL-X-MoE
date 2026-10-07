@echo off
setlocal
cd /d "%~dp0"
"%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
if errorlevel 1 (
  echo.
  echo Launch failed. See runtime\backend.log and runtime\frontend.log.
  pause
  exit /b 1
)
