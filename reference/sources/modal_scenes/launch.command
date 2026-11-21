#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BST="${BST_PYTHON:-python3}"
if ! command -v "$PYTHON_BST" >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required. Set BST_PYTHON to its executable."
  exit 1
fi
"$PYTHON_BST" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10 minimum"'
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BST" -m venv "$ROOT/.venv"
fi
if ! "$ROOT/.venv/bin/python" -c 'import numpy; from PIL import Image' >/dev/null 2>&1; then
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$ROOT/.venv/bin/python" "$ROOT/generer_gifs.py" "$@"
