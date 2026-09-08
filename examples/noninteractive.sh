#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to a GGUF inside your configured MODEL_ROOT}"
exec aag-llama-server-start --model "$MODEL" --context 8192 --performance medium --mmproj off --mtp off --dry-run
