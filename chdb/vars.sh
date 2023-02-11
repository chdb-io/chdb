# get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR="${DIR}/.." # project root directory
BUILD_DIR="$PROJ_DIR/buildlib" # build directory
CHDB_DIR="$PROJ_DIR/chdb" # chdb directory
CHDB_PY_MOD="_chdb"
CHDB_PY_MODULE="${CHDB_PY_MOD}$(python3-config --extension-suffix)" # python module name

# check current os type, and make ldd command
if [ "$(uname)" == "Darwin" ]; then
    LDD="otool -L"
elif [ "$(uname)" == "Linux" ]; then
    LDD="ldd"
else
    echo "OS not supported"
    exit 1
fi