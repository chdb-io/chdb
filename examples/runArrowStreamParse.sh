#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

NANOARROW_VER="0.8.0"
NANOARROW_DIR="nanoarrow-${NANOARROW_VER}"
NANOARROW_TAR="apache-arrow-nanoarrow-${NANOARROW_VER}.tar.gz"
NANOARROW_URL="https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-${NANOARROW_VER}/${NANOARROW_TAR}"

# ---- Step 1: Download & extract nanoarrow if needed ----
if [ ! -d "$NANOARROW_DIR" ]; then
    echo "Downloading nanoarrow ${NANOARROW_VER}..."
    curl -fSL -o "$NANOARROW_TAR" "$NANOARROW_URL"
    mkdir -p "$NANOARROW_DIR"
    tar xzf "$NANOARROW_TAR" -C "$NANOARROW_DIR" --strip-components=1
    rm -f "$NANOARROW_TAR"
fi

NA_SRC="$NANOARROW_DIR/src"
FLATCC_DIR="$NANOARROW_DIR/thirdparty/flatcc"

# ---- Step 2: Generate nanoarrow_config.h from template ----
CONFIG_H="$NA_SRC/nanoarrow/nanoarrow_config.h"
if [ ! -f "$CONFIG_H" ]; then
    echo "Generating nanoarrow_config.h..."
    sed -e 's/@NANOARROW_VERSION_MAJOR@/0/g' \
        -e 's/@NANOARROW_VERSION_MINOR@/8/g' \
        -e 's/@NANOARROW_VERSION_PATCH@/0/g' \
        -e 's/@NANOARROW_VERSION@/0.8.0/g' \
        -e 's/@NANOARROW_NAMESPACE_DEFINE@//g' \
        "$NA_SRC/nanoarrow/nanoarrow_config.h.in" > "$CONFIG_H"
fi

# ---- Step 3: Compile ----
echo "Compiling chdbArrowStreamParse..."

CC="${CC:-cc}"
CFLAGS="-std=c99 -O2 -Wall -Wno-unused-function"

if [ "$(uname)" = "Darwin" ]; then
    LDD="otool -L"
    LIB_PATH_VAR="DYLD_LIBRARY_PATH"
else
    LDD="ldd"
    LIB_PATH_VAR="LD_LIBRARY_PATH"
fi

# ---- LZ4 detection (ClickHouse ArrowStream defaults to lz4_frame compression) ----
LZ4_CFLAGS=""
LZ4_LIBS=""
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists liblz4 2>/dev/null; then
    LZ4_CFLAGS="$(pkg-config --cflags liblz4) -DNANOARROW_IPC_WITH_LZ4"
    LZ4_LIBS="$(pkg-config --libs liblz4)"
    echo "LZ4 support enabled (via pkg-config)"
else
    echo "WARNING: liblz4 not found, ArrowStream with LZ4 compression will fail at runtime"
fi

$CC $CFLAGS $LZ4_CFLAGS \
    -I"$NA_SRC" \
    -I"$FLATCC_DIR/include" \
    -I../programs/local/ \
    -DFLATCC_PORTABLE \
    chdbArrowStreamParse.c \
    "$NA_SRC/nanoarrow/common/array.c" \
    "$NA_SRC/nanoarrow/common/schema.c" \
    "$NA_SRC/nanoarrow/common/utils.c" \
    "$NA_SRC/nanoarrow/common/array_stream.c" \
    "$NA_SRC/nanoarrow/ipc/decoder.c" \
    "$NA_SRC/nanoarrow/ipc/encoder.c" \
    "$NA_SRC/nanoarrow/ipc/reader.c" \
    "$NA_SRC/nanoarrow/ipc/writer.c" \
    "$NA_SRC/nanoarrow/ipc/codecs.c" \
    "$FLATCC_DIR/src/runtime/builder.c" \
    "$FLATCC_DIR/src/runtime/emitter.c" \
    "$FLATCC_DIR/src/runtime/verifier.c" \
    "$FLATCC_DIR/src/runtime/refmap.c" \
    -L.. -lchdb $LZ4_LIBS \
    -o chdbArrowStreamParse

echo "Build OK"
export ${LIB_PATH_VAR}="$DIR/.."
$LDD ./chdbArrowStreamParse

echo ""
echo "=== Run ==="
./chdbArrowStreamParse
