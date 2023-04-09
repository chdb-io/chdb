#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

echo "include chdb/*.py" > MANIFEST.in
export SO_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "include chdb/_chdb${SO_SUFFIX}" >> MANIFEST.in