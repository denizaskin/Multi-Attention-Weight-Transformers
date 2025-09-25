#!/usr/bin/env bash
set -euo pipefail

if [ "${#}" -eq 0 ]; then
  python MAW_reranker.py --mode build-pools
else
  python MAW_reranker.py --mode build-pools --datasets "$@"
fi
