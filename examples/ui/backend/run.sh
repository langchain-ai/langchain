#!/usr/bin/env bash
# Start the FastAPI backend that wraps the four LangChain v1 example agents.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   examples/ui/backend/run.sh
#
# Serves on http://localhost:8000 (the Vite dev server proxies /api to it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../examples/ui/backend
REPO="$(cd "$HERE/../../.." && pwd)"                    # repo root

cd "$REPO/libs/langchain_v1"
exec uv run \
  --with langchain-openai \
  --with fastapi \
  --with "uvicorn[standard]" \
  python -m uvicorn main:app --app-dir "$HERE" --port 8000 --reload
