#!/usr/bin/env bash

export PATH=$HOME/.local/bin:$PATH

if [ -z "$1" ]; then
    cat /etc/motd
    exec /bin/bash
fi

exec "$@"
