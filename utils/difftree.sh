#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 dir1 dir2"
    exit 1
fi

# 获取参数
dir1=$1
dir2=$2

# 遍历 dir1 中的所有文件
find "$dir1" -type f | while read file; do
  # 将文件路径转换为相对路径
  relative_file="${file#$dir1/}"
  # 如果相应的文件也在 dir2 中存在
  if [ -e "$dir2/$relative_file" ]; then
    # 计算 diff 的行数并打印
    echo "$relative_file: $(diff "$dir1/$relative_file" "$dir2/$relative_file" | wc -l) line(s)"
  else
    # 如果文件只在 dir1 中存在，打印该文件的行数
    echo "$relative_file: $(wc -l < "$dir1/$relative_file") line(s) (only in dir1)"
  fi
done

