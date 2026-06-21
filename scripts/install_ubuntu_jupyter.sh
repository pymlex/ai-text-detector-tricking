#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git build-essential
fi

pip install -q -r requirements.txt
pip install -q git+https://github.com/pymlex/githublex.git

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
