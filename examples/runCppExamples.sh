#!/bin/bash

set -e

# check current os type, and make ldd command
if [ "$(uname)" == "Darwin" ]; then
    LDD="otool -L"
    LIB_PATH="DYLD_LIBRARY_PATH"
elif [ "$(uname)" == "Linux" ]; then
    LDD="ldd"
    LIB_PATH="LD_LIBRARY_PATH"
else
    echo "OS not supported"
    exit 1
fi

# cd to the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR


EXAMPLES=(
    "chdbCppBasic"
    "chdbCppAdvanced" 
    "chdbCppErrorHandling"
    "chdbCppStreaming"
)

# Set library path for runtime
export ${LIB_PATH}=..

for example in "${EXAMPLES[@]}"; do
    echo ""
    echo "Building $example..."
    
    if [ -f "${example}.cpp" ]; then
        clang++ -std=c++20 -I../programs/local/ -L.. \
            "${example}.cpp" -lchdb -o "$example" \
            -Wno-unused-parameter -O2
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully built $example"
            echo "Library dependencies:"
            ${LDD} "$example"
            
            if [ "$1" == "--run" ]; then
                echo "Running $example:"
                echo "---"
                "./$example"
                echo "---"
            fi
        else
            echo "✗ Failed to build $example"
        fi
    else
        echo "✗ Source file ${example}.cpp not found"
    fi
done
