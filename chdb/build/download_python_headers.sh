#!/bin/bash

set -e

TARGET_DIR="${HOME}/python_include"
TEMP_DIR="${TARGET_DIR}/tmp"

VERSIONS=(
    "3.8.10:3.8:3.8"
    "3.9.13:3.9:3.9"
    "3.10.11:3.10:3.10"
    "3.11.9:3.11:3.11"
    "3.12.10:3.12:3.12"
    "3.13.9:3.13:3.13"
    "3.14.0:3.14:3.14"
)

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"

for entry in "${VERSIONS[@]}"; do
    IFS=':' read -r FULL_VER SUBDIR MINOR_VER <<< "$entry"

    echo "=========================================="
    echo "Processing Python ${FULL_VER}..."
    echo "=========================================="

    # 检查目标目录是否已存在
    DEST_DIR="${TARGET_DIR}/${SUBDIR}"
    if [ -d "$DEST_DIR" ] && [ -f "${DEST_DIR}/Python.h" ]; then
        echo "✓ Python ${FULL_VER} headers already installed at ${DEST_DIR}"
        echo "  Skipping..."
        continue
    fi

    WORK_DIR="${TEMP_DIR}/${SUBDIR}"
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"

    PKG_URL="https://www.python.org/ftp/python/${FULL_VER}/python-${FULL_VER}-macos11.pkg"

    echo "Downloading: $PKG_URL"
    if wget -q --spider "$PKG_URL" 2>/dev/null; then
        wget -q --show-progress -O python.pkg "$PKG_URL"
    else
        echo "ERROR: Failed to download Python ${FULL_VER}"
        exit 1
    fi

    echo "Extracting pkg with 7z..."
    7z x -y python.pkg > /dev/null

    PAYLOAD_DIR=""
    for dir in Python_Framework.pkg PythonFramework-*.pkg; do
        if [ -d "$dir" ] || [ -f "$dir/Payload" ]; then
            PAYLOAD_DIR="$dir"
            break
        fi
    done

    if [ -z "$PAYLOAD_DIR" ]; then
        PAYLOAD_DIR=$(find . -name "Payload" -type f | head -1 | xargs dirname)
    fi

    if [ -z "$PAYLOAD_DIR" ] || [ ! -f "${PAYLOAD_DIR}/Payload" ]; then
        echo "ERROR: Cannot find Payload for Python ${FULL_VER}"
        exit 1
    fi

    echo "Extracting Payload from ${PAYLOAD_DIR}..."
    cd "$PAYLOAD_DIR"
    7z x -y Payload -so 2>/dev/null | cpio -id 2>/dev/null || true

    HEADER_SRC=""
    for path in \
        "Versions/${MINOR_VER}/Headers" \
        "Headers"
    do
        if [ -d "$path" ] && [ -f "$path/Python.h" ]; then
            HEADER_SRC="$path"
            break
        fi
    done

    if [ -z "$HEADER_SRC" ]; then
        PYTHON_H=$(find . -name "Python.h" -type f | head -1)
        if [ -n "$PYTHON_H" ]; then
            HEADER_SRC=$(dirname "$PYTHON_H")
        fi
    fi

    if [ -z "$HEADER_SRC" ] || [ ! -f "${HEADER_SRC}/Python.h" ]; then
        echo "ERROR: Cannot find headers for Python ${FULL_VER}"
        exit 1
    fi

    mkdir -p "$DEST_DIR"
    cp -r "${HEADER_SRC}/"* "$DEST_DIR/"

    echo "✓ Python ${FULL_VER} headers installed to ${DEST_DIR}"
    echo "  Files: $(ls "$DEST_DIR" | wc -l | tr -d ' ') items"
done

echo ""
echo "=========================================="
echo "Done! Headers installed to: ${TARGET_DIR}"
echo "=========================================="
ls -la "$TARGET_DIR"