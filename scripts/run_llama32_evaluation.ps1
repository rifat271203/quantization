# Full 900-Item Belebele Evaluation Runner for Llama 3.2 1B Instruct (All 7 Tiers)
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting Llama 3.2 1B Full 900-Item Evaluation (12,600 Predictions)" -ForegroundColor Cyan
Write-Host "Predictions will be saved to: results\llama32_logprob_predictions.jsonl" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# Run main evaluation harness
.\.venv\Scripts\python.exe audit\evaluate_llama32_logprobs.py

# Check progress upon exit
.\.venv\Scripts\python.exe audit\check_llama32_progress.py
