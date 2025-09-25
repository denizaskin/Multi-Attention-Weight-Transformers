#!/usr/bin/env bash
set -euo pipefail

python MAW_reranker.py --mode suite --variants all --with-dev-sweep --include-secondary --datasets "$@"
