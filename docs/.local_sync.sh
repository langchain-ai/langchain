#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "${SCRIPT_DIR}"

echo $1:$2

if [[ $2 != "change" ]]; then
  exit 0
fi

RAW_FILEPATH=$1
FILEPATH=""
NEWPATH=""

mkdir -p _dist/docs_skeleton
if [[ "$RAW_FILEPATH" =~ ^"docs_skeleton/docs/" ]]; then
  FILEPATH="${RAW_FILEPATH:19}"
  NEWPATH=_dist/docs_skeleton/docs/$FILEPATH
elif [[ "$RAW_FILEPATH" =~ ^"snippets/" ]]; then
  FILEPATH="${RAW_FILEPATH:9}"
  NEWPATH=_dist/snippets/$FILEPATH
elif [[ "$RAW_FILEPATH" =~ ^"extras/" ]]; then
  FILEPATH="${RAW_FILEPATH:7}"
  NEWPATH=_dist/docs_skeleton/docs/$FILEPATH
else
  exit 0
fi

cp $RAW_FILEPATH $NEWPATH

# TODO: Recompile notebooks
# if [[ $RAW_FILEPATH == *.ipynb ]]
# then
#   echo "Rebuilding $NEWPATH"
#   cd _dist/docs_skeleton
#   NEWPATH="${NEWPATH:20}"
#   rm -f "${NEWPATH%.ipynb}.md"
#   NEWDIR=$(echo $NEWPATH | sed 's|\(.*\)/.*|\1|')
#   poetry run nbdoc_build --srcdir $NEWDIR/
# fi

echo "Synced $FILEPATH"
