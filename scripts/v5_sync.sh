#!/usr/bin/env bash
# Sync this checkout to its own directory on a compute machine (default ~/cipherchat_v5; home is shared NFS across
# at-10-1-1..4). A separate directory keeps other work in ~/cipherchat (e.g. uncommitted v3 changes) untouched.
# The venv and the trained v3 checkpoints are linked from ~/cipherchat.
#   bash scripts/v5_sync.sh at-10-1-4 [remote_dir]
set -e
cd "$(dirname "$0")/.."
HOST=${1:-at-10-1-4}
DIR=${2:-cipherchat_v5}
tar czf - --exclude=__pycache__ --exclude='*.pt' --exclude=graphify-out --exclude=.git --exclude='build*' \
  --exclude=node_modules . | ssh -o BatchMode=yes "$HOST" "mkdir -p ~/$DIR && tar xzf - -C ~/$DIR && cd ~/$DIR && \
  ln -sfn ~/cipherchat/.venv .venv && mkdir -p weights/v3 && \
  for f in ~/cipherchat/weights/v3/*.pt; do ln -sfn \$f weights/v3/; done"
echo "synced -> $HOST:~/$DIR"
