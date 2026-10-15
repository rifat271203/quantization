# PowerShell script to run the 7-tier benchmark
# Usage:
#   .\scripts\run_parallel_jobs.ps1            # Runs with balanced 2 concurrent workers
#   .\scripts\run_parallel_jobs.ps1 -Workers 7 # Runs all 7 simultaneously

param (
    [int]$Workers = 2,
    [int]$Limit = 0
)

$VENV_PYTHON = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Error "Virtual environment python not found at $VENV_PYTHON"
    exit 1
}

$argsList = @("scripts\run_7tiers_parallel.py", "--concurrency", $Workers)
if ($Limit -gt 0) {
    $argsList += @("--limit", $Limit)
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Gemma 3 1B IT 7-Tier Quantization Study" -ForegroundColor Green
Write-Host "Concurrency: $Workers | Limit: $(if ($Limit -gt 0) { $Limit } else { 'All 300' })" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

& $VENV_PYTHON @argsList
