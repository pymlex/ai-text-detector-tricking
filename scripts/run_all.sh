#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
fi

set -a
source .env
set +a

python scripts/setup_gh_auth.py
python main.py --step all
python scripts/push_dataset_hf.py
python scripts/push_model_hf.py
python scripts/push_results_github.py --message "Ubuntu Jupyter: DPO detector evasion pipeline"
