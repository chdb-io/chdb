# get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR="${DIR}/.." # project root directory
BUILD_DIR="$PROJ_DIR/buildlib" # build directory
CHDB_DIR="$PROJ_DIR/chdb" # chdb directory
CHDB_PY_MOD="_chdb"
CHDB_PY_MODULE="${CHDB_PY_MOD}.abi3.so"
pushd ${PROJ_DIR}
CHDB_VERSION=$(python3 -c 'import setup; print(setup.get_latest_git_tag())')
popd

if [ "$1" == "cross-compile" ]; then
    return
fi

# try to use largest llvm-strip version
# if none of them are found, use llvm-strip or strip
if [ -z "$STRIP" ]; then
    STRIP=$(ls -1 /usr/bin/llvm-strip* | sort -V | tail -n 1)
fi
if [ -z "$STRIP" ]; then
    STRIP=$(ls -1 /usr/local/bin/llvm-strip* | sort -V | tail -n 1)
fi
# on macOS
if [ -z "$STRIP" ]; then
    STRIP=$(ls -1 /usr/local/Cellar/llvm/*/bin/llvm-strip* | sort -V | tail -n 1)
fi
if [ -z "$STRIP" ]; then
    STRIP=$(ls -1 /usr/local/opt/llvm/bin/llvm-strip* | sort -V | tail -n 1)
fi

# if none of them are found, use llvm-strip or strip
if [ -z "$STRIP" ]; then
    STRIP=$(which llvm-strip)
fi
if [ -z "$STRIP" ]; then
    STRIP=$(which strip)
fi

# check current os type, and make ldd command
if [ "$(uname)" == "Darwin" ]; then
    LDD="otool -L"
    AR="llvm-ar"
    NM="llvm-nm"
elif [ "$(uname)" == "Linux" ]; then
    LDD="ldd"
    AR="ar"
    NM="nm"
else
    echo "OS not supported"
    exit 1
fi
