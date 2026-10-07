@echo off
setlocal
cd /d "%~dp0"
py -3.13 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.13 was not found. Install Python 3.13 and retry.
  goto :error
)
node --version >nul 2>nul
if errorlevel 1 (
  echo Node.js was not found. Install Node.js 22 or newer and retry.
  goto :error
)
npm --version >nul 2>nul
if errorlevel 1 (
  echo npm was not found. Reinstall Node.js with npm and retry.
  goto :error
)
py -3.13 -m venv .venv
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install --no-deps "finrl-trading @ git+https://github.com/AI4Finance-Foundation/FinRL-Trading.git@4409abe925c904e570be78ebfb5e77ac3491dff8"
if errorlevel 1 goto :error
npm ci
if errorlevel 1 goto :error
echo.
echo Installation complete. Start.cmd will open the trading dashboard.
pause
exit /b 0
:error
echo Installation failed. Review the error above.
pause
exit /b 1
