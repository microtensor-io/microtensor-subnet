#!/bin/sh
set -e

if ! docker info >/dev/null 2>&1; then
    echo "the docker socket is not reachable; mount /var/run/docker.sock and set DOCKER_GID" >&2
    exit 1
fi

if [ "${MT_SANDBOX_RUNTIME:-runsc}" = "runsc" ] && ! docker info --format '{{json .Runtimes}}' | grep -q runsc; then
    echo "gVisor (runsc) is not registered with Docker; set MT_SANDBOX_RUNTIME=runc to run miner code in plain containers" >&2
    exit 1
fi

if [ "$(pwd)" != "/opt/microtensor" ]; then
    cd /opt/microtensor
fi

exec mt "$@"
