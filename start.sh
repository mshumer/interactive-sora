#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of invocation location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="$ROOT_DIR/.venv"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"

# Create virtual environment if missing and install backend dependencies.
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$ROOT_DIR/requirements.txt"

deactivate

# Install frontend dependencies (npm ci if lockfile exists, otherwise npm install)
if [ -d "$ROOT_DIR/frontend" ]; then
  pushd "$ROOT_DIR/frontend" >/dev/null
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
  popd >/dev/null
fi

# Launch backend (uvicorn) and frontend (Vite dev server) together; ensure clean shutdown.
trap 'kill 0' EXIT

source "$VENV_DIR/bin/activate"
uvicorn app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

pushd "$ROOT_DIR/frontend" >/dev/null
npm run dev -- --host &
FRONTEND_PID=$!
popd >/dev/null

wait $BACKEND_PID $FRONTEND_PID
