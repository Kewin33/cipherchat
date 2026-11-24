#!/usr/bin/env bash
# Sync only the v3 parts of the repo to the compute machines (home is shared NFS across at-10-1-1..4).
# Does not touch other files in ~/cipherchat on the remote.
set -e
cd "$(dirname "$0")/.."
HOST=${1:-at-10-1-2}
tar czf - --exclude=__pycache__ --exclude='*.pt' cipherchat/v3 cipherchat/__init__.py scripts/v3_* train_v3.py \
  docs/V3.md 2>/dev/null | ssh -o BatchMode=yes "$HOST" 'mkdir -p ~/cipherchat && tar xzf - -C ~/cipherchat'
echo "synced v3 -> $HOST:~/cipherchat"
