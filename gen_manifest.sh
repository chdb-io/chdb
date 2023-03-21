#!/bin/bash
echo "include chdb/*.py" > MANIFEST.in
export SO_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "include chdb/_chdb${SO_SUFFIX}" >> MANIFEST.in