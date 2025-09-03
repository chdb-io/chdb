#!/usr/bin/env python3

"""
Create minimized libchdb.a based on chdb_objects.txt
"""

import os
import platform
import sys
import subprocess

IS_MACOS_X86 = (platform.system() == "Darwin" and platform.machine() in ["x86_64", "i386"])
AR_CMD = ""

if IS_MACOS_X86:
    AR_CMD = "llvm-ar"
    print(f"Using llvm-ar for macOS x86 platform to avoid archive corruption issues")
else:
    AR_CMD = "ar"
    print(f"Using standard ar command for platform: {platform.system()} {platform.machine()}")

print(f"Selected ar command: {AR_CMD}")

def read_required_objects(objects_file="chdb_objects.txt"):
    """Read list of required target files"""

    if not os.path.exists(objects_file):
        print(f"❌ Object file list does not exist: {objects_file}")
        return None

    print(f"Reading object file list: {objects_file}")

    required_objects = set()

    with open(objects_file, 'r') as f:
        for line in f:
            obj = line.strip()
            if obj:
                required_objects.add(obj)

    print(f"   Required object files: {len(required_objects)} files")

    return required_objects


def extract_objects_from_archive(archive_path, required_objects, temp_dir):
    """Extract required object files from static library"""

    if not os.path.exists(archive_path):
        print(f"❌ Static library does not exist: {archive_path}")
        return []

    print(f"Extracting object files from {archive_path}...")

    # Get all object files in the archive
    result = subprocess.run([AR_CMD, 't', archive_path],
                             capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to read archive: {result.stderr}")
        return []

    available_objects = set(obj.strip() for obj in result.stdout.split('\n') if obj.strip())
    print(f"   Total object files in archive: {len(available_objects)} files")

    # Check if temp directory already exists and contains files
    if os.path.exists(temp_dir) and os.listdir(temp_dir):
        print(f"Reusing existing temp directory: {temp_dir}")
        # Count all .o files in temp directory
        existing_files = [f for f in os.listdir(temp_dir) if f.endswith('.o')]
        print(f"   Found already extracted object files: {len(existing_files)} files")
        return existing_files

    # Find object files that need to be extracted
    objects_to_extract = required_objects.intersection(available_objects)

    # Force exclude any object files containing ASTSQLSecurity
    objects_to_extract = {obj for obj in objects_to_extract if 'ASTSQLSecurity' not in obj}

    # Add all object files starting with "chdb_func" from the archive, but exclude those containing specific keywords
    # exclude_keywords = ['h3', 'H3', 'convertCharset', 'lowerUTF8', 'normalizeString', 'upperUTF8']

    # chdb_func_objects = {
    #     obj for obj in available_objects
    #     if obj.startswith('chdb_func') and not any(keyword in obj for keyword in exclude_keywords)
    # }
    # objects_to_extract.update(chdb_func_objects)

    # Force include all objects containing specific keywords
    # force_include_keywords = [
    #     'TargetSpecific',
    #     'SimdJSONParser',
    #     'FunctionsConversion_impl',
    #     'SipHash',
    #     'farmhash',
    #     'metrohash',
    #     'MurmurHash',
    #     'DummyJSONParser',
    #     'wide_integer_to_string',
    #     'Base58',
    #     'ArgumentExtractor',
    #     'int8_to_string',
    #     'getMostSubtype',
    #     'sha3iuf'
    # ]

    # force_include_objects = {
    #     obj for obj in available_objects
    #     if any(keyword in obj for keyword in force_include_keywords)
    # }
    # objects_to_extract.update(force_include_objects)

    # if chdb_func_objects:
    #     print(f"   Added chdb_func* object files (excluding specified keywords): {len(chdb_func_objects)} files")

    # if force_include_objects:
    #     print(f"   Force added critical object files: {len(force_include_objects)} files")

    missing_objects = required_objects - available_objects

    print(f"   Found required object files: {len(objects_to_extract)} files")
    if missing_objects:
        print(f"   Missing object files: {len(missing_objects)} files")
        for obj in sorted(missing_objects):
            print(f"     - {obj}")
        return []

    if not objects_to_extract:
        print("❌ No required object files found")
        return []

    # Extract object files to temp directory
    print(f"Extracting object files to temp directory...")

    extracted_files = []

    # Switch to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Extract object files in batches to avoid command line argument length limits
        batch_size = 500  # Process 100 files per batch
        objects_list = list(objects_to_extract)

        for i in range(0, len(objects_list), batch_size):
            batch = objects_list[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}: {len(batch)} files")

            extract_cmd = [AR_CMD, 'x', '../libchdb.a'] + batch
            result = subprocess.run(extract_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"   ❌ Batch {i//batch_size + 1} extraction failed: {result.stderr}")
                return []

            # Add successfully extracted files to the list
            extracted_files.extend(batch)
    finally:
        os.chdir(original_dir)

    print(f"Extraction result: {len(extracted_files)} / {len(objects_to_extract)} files")

    return extracted_files


def create_minimal_library(extracted_files, temp_dir, output_lib="libchdb_minimal.a"):
    """Create minimized static library"""

    if not extracted_files:
        print("❌ No object files extracted")
        return False

    print(f"Creating minimized static library: {output_lib}")

    # Remove old output library
    if os.path.exists(output_lib):
        os.remove(output_lib)
        print(f"   Removed old library file: {output_lib}")

    # Filter out system library files
    filtered_files = []
    system_lib_patterns = []

    for obj_file in extracted_files:
        is_system = any(pattern in obj_file for pattern in system_lib_patterns)
        if not is_system:
            filtered_files.append(obj_file)
        else:
            print(f"   Skipping system library file: {obj_file}")
            continue

    valid_obj_paths = []
    for obj in filtered_files:
        obj_path = os.path.join(temp_dir, obj)
        if os.path.exists(obj_path):
            valid_obj_paths.append(obj_path)
        else:
            print(f"   Error: Non-existent file: {obj}")
            return False

    if not valid_obj_paths:
        print("❌ No valid object files found")
        return False

    print(f"   Valid object files: {len(valid_obj_paths)} files")

    # Create static library in batches, 1000 files per batch
    batch_size = 1000
    total_batches = (len(valid_obj_paths) + batch_size - 1) // batch_size

    print(f"   Will process {len(valid_obj_paths)} files in {total_batches} batches")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(valid_obj_paths))
        batch_files = valid_obj_paths[start_idx:end_idx]

        print(f"   Processing batch {batch_idx + 1}/{total_batches}: {len(batch_files)} files")

        if batch_idx == 0:
            # First batch uses 'rcs' to create a new library
            ar_cmd = [AR_CMD, 'rcs', os.path.abspath(output_lib)] + batch_files
        else:
            # Subsequent batches use 'rs' to append to the existing library
            ar_cmd = [AR_CMD, 'rs', os.path.abspath(output_lib)] + batch_files

        result = subprocess.run(ar_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Batch {batch_idx + 1} creation failed: {result.stderr}")
            return False

    if not os.path.exists(output_lib):
        print(f"❌ Output file does not exist: {output_lib}")
        return False

    print(f"✅ Successfully created minimized static library: {output_lib}")
    print(f"   File size: {os.path.getsize(output_lib):,} bytes")

    return True


def main():
    print("Starting creation of minimized libchdb.a")
    print("=" * 50)

    # Path configuration
    chdb_objects_file = "chdb_objects.txt"
    original_lib = "libchdb.a"
    temp_dir = "libchdb_objects_tmp_dir"
    output_lib = "libchdb_minimal.a"

    if len(sys.argv) > 1:
        chdb_objects_file = sys.argv[1]
    if len(sys.argv) > 2:
        original_lib = sys.argv[2]
    if len(sys.argv) > 3:
        output_lib = sys.argv[3]

    # Read required object files
    required_objects = read_required_objects(chdb_objects_file)
    if not required_objects:
        print("❌ Empty chdb_objects.txt")
        sys.exit(1)

    # Check if temp directory already exists and contains files
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print(f"Using temp directory: {temp_dir}")

    try:
        # Extract object files from original library
        extracted_files = extract_objects_from_archive(original_lib, required_objects, temp_dir)

        if not extracted_files:
            print("❌ No object files extracted")
            sys.exit(1)

        # Create minimized library
        success = create_minimal_library(extracted_files, temp_dir, output_lib)

        if not success:
            print("❌ Failed to create minimized library")
            sys.exit(1)

    finally:
        pass

    print(f"\nMinimized library creation completed!")
    print(f"Generated files:")
    print(f"   - {output_lib}              (minimized static library)")

if __name__ == "__main__":
    main()
