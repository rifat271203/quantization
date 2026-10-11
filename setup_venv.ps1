# Minimal venv setup for Phase 1 tokenizer fertility check (CPU-only).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Creating virtual environment in .venv ..."
python -m venv .venv

Write-Host "Activating .venv ..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing requirements (CPU torch is fine) ..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate later with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Run fertility check:   python scripts\tokenizer_fertility_check.py"
