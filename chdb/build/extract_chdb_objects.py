#!/usr/bin/env python3

"""
Precisely extract target files from libchdb.a in chdb_example.map file
Specially handles ./libchdb.a(object.o) format and issues warnings for abnormal formats
"""

import os
import sys
import re

def extract_chdb_objects(map_file_path):
    """Extract chdb-related target files from map file"""

    if not os.path.exists(map_file_path):
        print(f"❌ Map file does not exist: {map_file_path}")
        return None

    print(f"Analyzing map file: {map_file_path}")

    # Try to read with UTF-8, fallback to latin-1 if that fails
    try:
        with open(map_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("Warning: UTF-8 decode failed, trying with latin-1 encoding...")
        with open(map_file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    # Store results
    extracted_objects = set()
    warnings = []
    line_stats = {
        "total_lines": len(lines),
        "chdb_lines": 0,
        "matched_lines": 0,
        "warning_lines": 0
    }

    # Precise regex pattern matching ./libchdb.a(object.o) and ./libchdb.a[anything](object.o) formats
    chdb_pattern = re.compile(r'\./libchdb\.a(?:\[[^\]]*\])?\(([^)]+\.o)\)')

    print("Analyzing map file line by line...")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if contains libchdb.a
        if 'libchdb.a' in line:
            line_stats["chdb_lines"] += 1

            # Try to match standard format
            matches = chdb_pattern.findall(line)

            if matches:
                # Found matching object files
                line_stats["matched_lines"] += 1
                for obj in matches:
                    extracted_objects.add(obj)

            else:
                # Contains libchdb.a but format doesn't match, issue warning
                line_stats["warning_lines"] += 1
                warning_msg = f"Line {line_num:5d}: Contains 'libchdb.a' but format doesn't match"
                warnings.append({
                    "line_number": line_num,
                    "content": line,
                    "message": warning_msg
                })
                print(f"   ⚠️  {warning_msg}")
                print(f"       Content: {line[:100]}{'...' if len(line) > 100 else ''}")

    print(f"\n📈 Analysis statistics:")
    print(f"   Total lines: {line_stats['total_lines']}")
    print(f"   Lines containing 'chdb': {line_stats['chdb_lines']}")
    print(f"   Successfully matched lines: {line_stats['matched_lines']}")
    print(f"   Warning lines: {line_stats['warning_lines']}")
    print(f"   Extracted object files: {len(extracted_objects)}")

    return {
        "objects": extracted_objects,
        "warnings": warnings,
        "stats": line_stats
    }


def save_results(result, output_prefix="chdb_objects"):
    """Save analysis results to files"""

    objects = result["objects"]
    warnings = result["warnings"]
    stats = result["stats"]

    # Save object file list
    objects_file = f"{output_prefix}.txt"
    with open(objects_file, 'w') as f:
        for obj in sorted(objects):
            f.write(f"{obj}\n")

    print(f"\n💾 Object file list saved: {objects_file} ({len(objects)} files)")

    # Save warning information
    if warnings:
        warnings_file = f"{output_prefix}_warnings.txt"
        with open(warnings_file, 'w') as f:
            f.write(f"Map File Analysis Warning Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total warnings: {len(warnings)}\n\n")

            for warning in warnings:
                f.write(f"Line number: {warning['line_number']}\n")
                f.write(f"Message: {warning['message']}\n")
                f.write(f"Content: {warning['content']}\n")
                f.write("-" * 50 + "\n")

        print(f"⚠️  Warning information saved: {warnings_file} ({len(warnings)} warnings)")

    # Save detailed statistics
    stats_file = f"{output_prefix}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Map File Analysis Statistics Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total lines: {stats['total_lines']}\n")
        f.write(f"Lines containing 'chdb': {stats['chdb_lines']}\n")
        f.write(f"Successfully matched lines: {stats['matched_lines']}\n")
        f.write(f"Warning lines: {stats['warning_lines']}\n")
        f.write(f"Extracted object files: {len(objects)}\n")
        f.write(f"Match success rate: {stats['matched_lines']/stats['chdb_lines']*100:.1f}%\n" if stats['chdb_lines'] > 0 else "Match success rate: N/A\n")

        f.write(f"\nExtracted object file list:\n")
        f.write("-" * 30 + "\n")
        for i, obj in enumerate(sorted(objects), 1):
            f.write(f"{i:3d}. {obj}\n")

    print(f"📊 Statistics report saved: {stats_file}")


def main():
    print("🚀 chdb Object File Extraction Tool")
    print("=" * 60)

    # Determine map file path
    map_file = "chdb_example.map"
    if len(sys.argv) > 1:
        map_file = sys.argv[1]

    # Extract object files
    result = extract_chdb_objects(map_file)

    if not result:
        sys.exit(1)

    objects = result["objects"]
    warnings = result["warnings"]

    if not objects:
        print("\n❌ No object files extracted")
        if warnings:
            print("💡 But some warnings were found, please check if the format is correct")
        sys.exit(1)

    # Save results
    save_results(result)

    print(f"\nExtraction completed!")
    print(f"Generated files:")
    print(f"   - chdb_objects.txt           (extracted object file list)")
    if warnings:
        print(f"   - chdb_objects_warnings.txt  (format warning information)")

    # Display summary
    print(f"\nExtraction summary:")
    print(f"   Successfully extracted: {len(objects)} object files")
    if warnings:
        print(f"   Format warnings: {len(warnings)}")

if __name__ == "__main__":
    main()
