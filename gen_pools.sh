#!/usr/bin/env bash
set -euo pipefail

python MAW_reranker.py --mode build-pools --include-secondary "$@"
