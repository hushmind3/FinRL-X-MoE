@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: installing Python and frontend packages. This can take several minutes.
  call "%~dp0install.cmd"
  if errorlevel 1 exit /b 1
)
call "%~dp0start.cmd"
