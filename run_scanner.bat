@echo off
title Portable USB Bounty Scanner
cd /d %~dp0
echo Setting up environment...

:: Authenticated token to bypass GitHub rate limits (5,000 req/hr)
set GITHUB_TOKEN=

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :run
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :run
)

C:\Python314\python.exe --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=C:\Python314\python.exe
    goto :run
)

echo [!] Error: Python was not found or is not registered in the system path.
pause
exit

:run
echo Launching Bounty Scanner using: %PYTHON_CMD%
%PYTHON_CMD% scanner.py
pause