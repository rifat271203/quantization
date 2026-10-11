#!/usr/bin/env bash
# Minimal venv setup for Phase 1 tokenizer fertility check (CPU-only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Creating virtual environment in .venv ..."
python3 -m venv .venv

echo "Activating .venv ..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo "Installing requirements (CPU torch is fine) ..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate later with:  source .venv/bin/activate"
echo "Run fertility check:   python scripts/tokenizer_fertility_check.py"
