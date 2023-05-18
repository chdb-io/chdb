#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

rm -f MANIFEST.in

echo "include README.md" >> MANIFEST.in
echo "include LICENSE.txt" >> MANIFEST.in
echo "graft chdb" >> MANIFEST.in
echo "global-exclude *.py[cod]" >> MANIFEST.in
echo "global-exclude __pycache__" >> MANIFEST.in
echo "global-exclude .DS_Store" >> MANIFEST.in
echo "global-exclude .git*" >> MANIFEST.in
echo "global-exclude ~*" >> MANIFEST.in
export SO_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "include chdb/_chdb${SO_SUFFIX}" >> MANIFEST.in