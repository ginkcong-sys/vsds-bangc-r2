#!/usr/bin/env bash
set -e
CMD=${1:-infer}
shift || true
case "$CMD" in
  infer) python3 src/infer.py "$@" ;;
  verify_repro) python3 src/verify_repro.py "$@" ;;
  *) python3 src/infer.py "$CMD" "$@" ;;
esac
