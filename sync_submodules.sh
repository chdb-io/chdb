#!/bin/bash

# Script to sync submodule versions from ClickHouse repository
CLICKHOUSE_PATH="/Users/victor/code/clickhouse"
CURRENT_PATH=$(pwd)

if [ ! -d "$CLICKHOUSE_PATH" ]; then
    echo "ClickHouse repository not found at $CLICKHOUSE_PATH"
    exit 1
fi

echo "Comparing submodule versions..."
echo "================================"

# Get submodule list from current project
git submodule status | while read line; do
    # Extract commit hash and path
    commit=$(echo "$line" | awk '{print $1}' | sed 's/^[+ -]//')
    path=$(echo "$line" | awk '{print $2}')
    
    # Get corresponding submodule from ClickHouse
    if [ -d "$CLICKHOUSE_PATH/$path" ]; then
        cd "$CLICKHOUSE_PATH"
        ch_commit=$(git ls-tree HEAD "$path" | awk '{print $3}')
        cd "$CURRENT_PATH"
        
        if [ "$commit" != "$ch_commit" ]; then
            echo "DIFF: $path"
            echo "  Current: $commit"
            echo "  ClickHouse: $ch_commit"
            
            # Update to ClickHouse version
            cd "$path"
            git fetch origin
            git checkout "$ch_commit"
            cd "$CURRENT_PATH"
            
            echo "  Updated to ClickHouse version"
        else
            echo "OK: $path (already in sync)"
        fi
    else
        echo "SKIP: $path (not found in ClickHouse)"
    fi
    echo ""
done

echo "Submodule sync complete!"