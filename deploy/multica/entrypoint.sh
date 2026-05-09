#!/bin/bash
set -e

multica config set server_url "${MULTICA_SERVER_URL}"
multica config set app_url "${MULTICA_APP_URL}"

if [ -n "${MULTICA_TOKEN}" ]; then
    multica auth set-token "${MULTICA_TOKEN}"
fi

exec multica daemon start --foreground
