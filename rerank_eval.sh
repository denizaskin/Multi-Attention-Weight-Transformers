#!/usr/bin/env bash
set -euo pipefail

if [ "${#}" -eq 0 ]; then
  python MAW_reranker.py --mode suite --variants default
else
  python MAW_reranker.py --mode suite --variants default --datasets "$@"
fi
