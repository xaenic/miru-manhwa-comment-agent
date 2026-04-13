#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  if "${VENV_PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "Bootstrapped pip with ensurepip"
  else
    BOOTSTRAP_SCRIPT="$(mktemp)"
    trap 'rm -f "${BOOTSTRAP_SCRIPT}"' EXIT

    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${BOOTSTRAP_SCRIPT}"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "${BOOTSTRAP_SCRIPT}" https://bootstrap.pypa.io/get-pip.py
    else
      echo "pip is missing and neither ensurepip, curl, nor wget is available." >&2
      exit 1
    fi

    "${VENV_PYTHON}" "${BOOTSTRAP_SCRIPT}"
    echo "Bootstrapped pip with get-pip.py"
  fi
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r requirements.txt

if [ ! -f "${ROOT_DIR}/.env" ]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "Created .env from .env.example"
fi

cat <<'EOF'

Setup complete.

Required env vars in .env before posting:
- XAI_API_KEY

One-cycle dry run:
  ./.venv/bin/python simulate_manhwa_comment_agents.py \
    --api-base-url http://127.0.0.1:5000 \
    --series-slug YOUR_MANHWA_SLUG \
    --language en \
    --run-once \
    --dry-run

Continuous run every 10 minutes:
  ./.venv/bin/python simulate_manhwa_comment_agents.py \
    --api-base-url http://127.0.0.1:5000 \
    --series-slug YOUR_MANHWA_SLUG \
    --language en \
    --interval-minutes 10
EOF
