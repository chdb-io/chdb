#!python3

import filecmp
import sys
import os
import difflib


def strip_comments(line):
    line = line.strip()
    if line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('#'):
        return ''
    return line


def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [line for line in f1.readlines() if len(strip_comments(line)) > 0]
        lines2 = [line for line in f2.readlines() if len(strip_comments(line)) > 0]
        diff = difflib.unified_diff(lines1, lines2, n=0)
        diff_count = sum(1 for _ in diff if not _.startswith('---') and not _.startswith('+++'))
        print(f"Differences in {file1} {file2}: {diff_count} lines")
    return diff_count


def compare_folders(folder1, folder2):
    diff_count = 0

    if not os.path.isdir(folder1):
        print(f"Folder '{folder1}' does not exist.")
        return -1
    if not os.path.isdir(folder2):
        print(f"Folder '{folder2}' does not exist.")
        return -1
    # Use filecmp module's dircmp function to compare two folders
    dir_cmp = filecmp.dircmp(folder1, folder2)

    # Iterate over subfolders that exist in both folders
    for subfolder in dir_cmp.common_dirs:
        subfolder1 = os.path.join(folder1, subfolder)
        subfolder2 = os.path.join(folder2, subfolder)
        # Recursively compare subfolders
        diff_count += compare_folders(subfolder1, subfolder2)

    # Iterate over files that exist in both folders
    for file in dir_cmp.common_files:
        file1 = os.path.join(folder1, file)
        file2 = os.path.join(folder2, file)
        # Compare the two files with git like diff
        diff_count += compare_files(file1, file2)

    # Iterate over files or folders that exist only in one folder
    for file in dir_cmp.left_only + dir_cmp.right_only:
        file_path = os.path.join(folder1, file) if file in dir_cmp.left_only else os.path.join(folder2, file)

        # Check if the file is a folder
        if os.path.isdir(file_path):
            # If it's a folder, recursively compare subfolders
            print(f"Differences in subfolder {file_path}:")
            # Print all the files that exist only in one folder and their line count without comments
            for file in os.listdir(file_path):
                file_path2 = os.path.join(file_path, file)
                if os.path.isfile(file_path2):
                    with open(file_path2, 'r') as file:
                        lines = [line for line in file.readlines() if len(strip_comments(line)) > 0]
                        diff_count += len(lines)
                        print(f"Only in {file_path2}: {len(lines)} lines")
        else:
            # If it's a file, count the number of lines
            with open(file_path, 'r') as file:
                lines = [line for line in file.readlines() if len(strip_comments(line)) > 0]
                diff_count += len(lines)
                print(f"Only in {file.name}: {len(lines)} lines")

    return diff_count


# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Please provide two folder paths as command-line arguments.")
    sys.exit(1)

# Get the folder paths from the command-line arguments
folder1 = sys.argv[1]
folder2 = sys.argv[2]

# Compare the two folders
total_diff_count = compare_folders(folder1, folder2)

print(f"Total differences: {total_diff_count} lines")
