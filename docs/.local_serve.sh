#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "${SCRIPT_DIR}"

yarn run chokidar "{extras,docs_skeleton,snippets}/**" -c "bash ./.local_sync.sh '{path}' '{event}'" & cd _dist/docs_skeleton && yarn run start & wait