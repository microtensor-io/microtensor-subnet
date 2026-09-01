#!/bin/sh
set -e

ROOT="${MT_HOME:-/var/lib/microtensor}/evaluation-env"
if [ ! -f "$ROOT/lock.txt" ]; then
    echo "pinned evaluation environment missing at $ROOT; building it now"
    mt validator env-setup
fi

exec mt validator run "$@"
