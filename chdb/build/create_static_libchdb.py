#!/usr/bin/env python3

"""
Script to create libchdb.a static library
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import shutil

# Global variables (will be set based on arguments)
IS_MACOS_X86 = False
IS_MACOS = False
CROSS_COMPILE = False
AR_CMD = ""
BUILD_DIR = ""

def setup_platform(cross_compile=False, ar_cmd=None):
    """Setup platform-specific variables"""
    global IS_MACOS_X86, IS_MACOS, CROSS_COMPILE, AR_CMD

    if cross_compile:
        # Cross-compiling for macOS on Linux
        IS_MACOS = True
        CROSS_COMPILE = True
        if ar_cmd:
            AR_CMD = ar_cmd
        else:
            AR_CMD = "ar"
        print(f"Cross-compile mode: targeting macOS")
    else:
        # Native build
        IS_MACOS_X86 = (platform.system() == "Darwin" and platform.machine() in ["x86_64", "i386"])
        IS_MACOS = platform.system() == "Darwin"
        if IS_MACOS_X86:
            AR_CMD = "llvm-ar"
            print(f"Using llvm-ar for macOS x86 platform to avoid archive corruption issues")
        else:
            AR_CMD = "ar"
            print(f"Using standard ar command for platform: {platform.system()} {platform.machine()}")

    print(f"Selected ar command: {AR_CMD}")
    print(f"CROSS_COMPILE: {CROSS_COMPILE}, IS_MACOS: {IS_MACOS}")

def parse_libchdb_cmd(build_dir_override=None):
    """Extract object files and static libraries"""
    global BUILD_DIR

    # Get the directory containing this script, then go up two levels
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if build_dir_override:
        build_dir = build_dir_override
    else:
        build_dir = 'buildlib'

    BUILD_DIR = build_dir
    print(f"Using build directory: {build_dir}")

    # First, check build.log to see if it contains @CMakeFiles/clickhouse.rsp
    build_log_path = os.path.join(project_root, build_dir, 'build.log')
    rsp_file_path = os.path.join(project_root, build_dir, 'CMakeFiles', 'clickhouse.rsp')

    command = ""
    use_rsp_file = False

    if os.path.exists(build_log_path):
        with open(build_log_path, 'r') as f:
            build_log_content = f.read()
            if '@CMakeFiles/clickhouse.rsp' in build_log_content:
                use_rsp_file = True
                print("Found @CMakeFiles/clickhouse.rsp in build.log, using response file")
            else:
                print("No response file reference found in build.log, using build.log directly")

    if use_rsp_file:
        # Read from response file
        with open(rsp_file_path, 'r') as f:
            command = f.read().strip()
        print(f"Reading from response file: {rsp_file_path}")
    elif os.path.exists(build_log_path):
        # Extract linking command from build.log
        with open(build_log_path, 'r') as f:
            command = f.read().strip()
        print(f"Extracted linking command from build.log")
    else:
        raise FileNotFoundError(f"Neither {rsp_file_path} nor {build_log_path} exists")

    # Print the command for debugging
    print("\n=== COMMAND ===")
    print(command)
    print("=== END COMMAND ===\n")

    # Common prefix for absolute paths
    base_path = os.path.join(project_root, build_dir)

    # Extract all .o files and .a files from the command
    # Pattern for .o files (must be followed by space or end of string)
    obj_pattern = r'[^\s]+\.cpp\.o(?=\s|$)|[^\s]+\.c\.o(?=\s|$)'
    obj_files = re.findall(obj_pattern, command)

    # Pattern for .a files (must be followed by space or end of string)
    lib_pattern = r'[^\s]+\.a(?=\s|$)'
    lib_files = re.findall(lib_pattern, command)

    # Filter out main.cpp.o and convert to absolute paths
    filtered_obj_files = []
    for obj in obj_files:
        if 'programs/CMakeFiles/clickhouse.dir/main.cpp.o' not in obj:
            # Convert to absolute path
            abs_path = os.path.join(base_path, obj)
            filtered_obj_files.append(abs_path)
            # print(f"Found object file: {abs_path}")
        else:
            print(f"Excluding: {obj}")

    # Convert library files to absolute paths
    abs_lib_files = []
    for lib in lib_files:
        abs_path = os.path.join(base_path, lib)
        abs_lib_files.append(abs_path)
        # print(f"Found library file: {abs_path}")

    print(f"Found {len(filtered_obj_files)} object files (excluding main.cpp.o)")
    print(f"Found {len(abs_lib_files)} static libraries")
    print(f"Using base path: {base_path}")

    return filtered_obj_files, abs_lib_files

def create_static_library(obj_files, lib_files):
    """Create libchdb.a static library using ar command"""

    print("\nCreating static library using ar...")

    # Output library name
    output_lib = "libchdb.a"

    # Remove existing library if it exists
    if os.path.exists(output_lib):
        os.remove(output_lib)
        print(f"Removed existing {output_lib}")

    # Check if ar is available (skip on macOS)
    if not IS_MACOS:
        try:
            subprocess.run([AR_CMD, "--version"],
                           capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: ar not found. Please install ar.")
            return False

    # Create temporary directory for extracted objects
    temp_dir = "create_static_lib_tmp_dir"

    need_extract = True

    # Always create fresh tmp directory to ensure extraction
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Removed existing temporary directory: {os.path.abspath(temp_dir)}")

    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temporary directory: {os.path.abspath(temp_dir)}")

    print(f"Using temporary directory: {os.path.abspath(temp_dir)}")

    try:

        # Extract objects from static libraries
        extracted_objects = []

        # Add libiconv.a to the list of libraries to extract on macOS
        if not CROSS_COMPILE and IS_MACOS:
            libiconv_path = "/opt/homebrew/opt/libiconv/lib/libiconv.a"
            if os.path.exists(libiconv_path):
                lib_files.append(libiconv_path)
                print(f"Added libiconv.a for static linking: {libiconv_path}")
            else:
                print(f"Warning: libiconv.a not found at {libiconv_path}")

        for lib_file in lib_files:
            if not need_extract:
                continue

            if not os.path.exists(lib_file):
                print(f"Warning: Library file not found: {lib_file}")
                return False

            lib_basename = os.path.basename(lib_file)
            lib_name = lib_basename.replace('.a', '').replace('lib_', '')

            # Check if lib_file path contains src/Functions
            if 'src/Functions' in lib_file:
                lib_prefix = f"chdb_func_{lib_name}"
                print(f"Functions library detected: {lib_basename} -> prefix: {lib_prefix}")
            # Special handling for clickhouse-local-lib.a which contains chdb.cpp.o
            elif 'clickhouse-local-lib.a' in lib_file:
                lib_prefix = "chdb_local"
                print(f"Local library detected: {lib_basename} -> prefix: {lib_prefix}")
            # Special handling for libiconv.a
            elif 'libiconv.a' in lib_file:
                lib_prefix = "chdb_iconv"
                print(f"iconv library detected: {lib_basename} -> prefix: {lib_prefix}")
            else:
                lib_prefix = lib_name

            # Create subdirectory for this library
            lib_temp_dir = os.path.join(temp_dir, lib_basename.replace('.', '_'))
            os.makedirs(lib_temp_dir, exist_ok=True)

            # Optimized extraction: analyze for duplicates first, then choose strategy
            try:
                # First, get list of all object files in the archive
                list_result = subprocess.run([AR_CMD, "t", lib_file],
                                             capture_output=True,
                                             text=True,
                                             check=True)

                all_files_in_archive = [f.strip() for f in list_result.stdout.split('\n') if f.strip().endswith('.o')]
                print(f"Found {len(all_files_in_archive)} object files in {lib_basename}")

                # Analyze for duplicate filenames (case-insensitive)
                # Group files by lowercase filename to detect case-insensitive duplicates
                case_insensitive_groups = {}
                for filename in all_files_in_archive:
                    lower_name = filename.lower()
                    if lower_name not in case_insensitive_groups:
                        case_insensitive_groups[lower_name] = []
                    case_insensitive_groups[lower_name].append(filename)

                # Separate unique files from duplicates (case-insensitive)
                unique_files = []
                duplicate_files = []

                for lower_name, actual_filenames in case_insensitive_groups.items():
                    if len(actual_filenames) == 1:
                        unique_files.extend(actual_filenames)
                    else:
                        duplicate_files.extend(actual_filenames)

                if duplicate_files:
                    print(f"Found {len(duplicate_files)} filenames with duplicates (case-insensitive):")
                    # Report duplicate groups
                    for lower_name, actual_filenames in case_insensitive_groups.items():
                        if len(actual_filenames) > 1:
                            print(f"     {lower_name} -> {', '.join(actual_filenames)} ({len(actual_filenames)} files)")

                print(f"Strategy: {len(unique_files)} unique files (fast extraction), {len(duplicate_files)} duplicate types (iterative extraction)")

                extracted_count = 0

                # FAST PATH: Bulk extract unique files
                if unique_files:
                    print(f"Bulk extracting {len(unique_files)} unique files...")

                    # Use ar x for bulk extraction of unique files
                    extract_result = subprocess.run([AR_CMD, "x", lib_file] + unique_files,
                                                    cwd=lib_temp_dir,
                                                    capture_output=True,
                                                    text=True)

                    if extract_result.returncode != 0:
                        print(f"Warning: Bulk extraction failed: {extract_result.stderr}")
                        return False
                    else:
                        # Rename extracted unique files with library prefix
                        for filename in unique_files:
                            original_path = os.path.join(lib_temp_dir, filename)
                            if os.path.exists(original_path):
                                # Special handling for chdb.cpp.o - keep original name
                                if filename == "chdb.cpp.o" and 'clickhouse-local-lib' in lib_file:
                                    unique_filename = filename
                                    print(f"Special handling for chdb.cpp.o - keeping original name")
                                # Generate prefixed filename for other files
                                elif filename.startswith(f"{lib_prefix}__"):
                                    unique_filename = filename
                                else:
                                    unique_filename = f"{lib_prefix}__{filename}"

                                unique_path = os.path.join(lib_temp_dir, unique_filename)
                                os.rename(original_path, unique_path)
                                extracted_objects.append(unique_path)
                                extracted_count += 1

                                print(f"Bulk extracted {filename} → {unique_filename}")

                # SLOW PATH: Iterative extract for duplicate files only
                if duplicate_files:
                    print(f"Iteratively extracting duplicate files...")

                    # Create working copy for iterative extraction
                    working_archive = os.path.join(lib_temp_dir, f"working_{lib_basename}")
                    shutil.copy2(lib_file, working_archive)

                    for lower_name, actual_filenames in case_insensitive_groups.items():
                        if len(actual_filenames) <= 1:
                            continue  # Skip unique files

                        print(f"Processing case-insensitive group: {lower_name} ({len(actual_filenames)} files)")

                        # Process each file in this group with sequential numbering
                        for file_index, target_filename in enumerate(actual_filenames, 1):
                            # Extract this occurrence
                            extract_result = subprocess.run([AR_CMD, "p", working_archive, target_filename],
                                                            capture_output=True)

                            if extract_result.returncode != 0:
                                print(f"Failed to extract {target_filename}")
                                print(f"STDERR: {extract_result.stderr.decode() if extract_result.stderr else 'No error message'}")
                                return False

                            # Generate unique filename: originalname_X.o format
                            # Special handling for chdb.cpp.o - keep original name for first occurrence
                            if target_filename == "chdb.cpp.o" and 'clickhouse-local-lib' in lib_file and file_index == 1:
                                unique_filename = target_filename
                                print(f"Special handling for chdb.cpp.o - keeping original name")
                            else:
                                name_part = target_filename.replace('.o', '')
                                unique_filename = f"{lib_prefix}__{name_part}_{file_index}.o"

                            # Write extracted content
                            unique_path = os.path.join(lib_temp_dir, unique_filename)
                            with open(unique_path, 'wb') as f:
                                f.write(extract_result.stdout)

                            extracted_objects.append(unique_path)
                            extracted_count += 1

                            print(f"Extracted {target_filename} → {unique_filename} (group #{file_index})")

                            # Delete this occurrence from working archive
                            delete_result = subprocess.run([AR_CMD, "d", working_archive, target_filename],
                                                            capture_output=True)

                            if delete_result.returncode != 0:
                                print(f"Warning: Failed to delete {target_filename} from working archive")
                                print(f"STDERR: {delete_result.stderr.decode() if delete_result.stderr else 'No error message'}")
                                return False

                    # Clean up working archive
                    try:
                        os.remove(working_archive)
                    except:
                        pass

                print(f"Extracted and renamed {extracted_count} objects from {lib_basename}")

            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to extract {lib_file}: {e}")
                return False

        # Collect all object files (direct .o files + extracted from .a files)
        all_objects = []

        # Add direct object files
        for obj_file in obj_files:
            # Check if obj_file path contains src/Functions
            if 'src/Functions' in obj_file:
                # Copy the file to temp_dir with clickhouse_chdb_functions prefix
                obj_basename = os.path.basename(obj_file)
                new_name = f"chdb_func_{obj_basename}"
                dest_path = os.path.join(temp_dir, new_name)

                try:
                    shutil.copy2(obj_file, dest_path)
                    print(f"Copied Functions object: {obj_basename} -> {new_name}")
                    all_objects.append(dest_path)  # Use the copied file
                except Exception as e:
                    print(f"ERROR: Failed to copy {obj_file}: {e}")
                    return False
            else:
                all_objects.append(obj_file)

        # Add extracted object files
        all_objects.extend(extracted_objects)

        # Filter out libcrypto__fipsprov objects
        original_count = len(all_objects)
        all_objects = [obj for obj in all_objects if 'libcrypto__fipsprov' not in obj]
        filtered_count = original_count - len(all_objects)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} libcrypto__fipsprov objects")

        print(f"Total objects to include: {len(all_objects)}")
        print(f"   - Direct .o files: {len([o for o in all_objects if o in obj_files])}")
        print(f"   - Extracted from .a files: {len(extracted_objects)}")

        if not all_objects:
            print("ERROR: No object files to include in library")
            return False

        # Create static library using ar
        print(f"\nCreating {output_lib} with ar...")

        try:
            # Use ar to create static library
            # rcs: create archive, add files, create symbol table, suppress warnings

            # For very large command lines, we need to create the library incrementally
            # ar has a limit on command line length
            batch_size = 1000  # Process objects in batches

            # Create empty archive first (skip on macOS as it doesn't support empty archives)
            if not IS_MACOS:
                result = subprocess.run([AR_CMD, "rcs", output_lib],
                                        capture_output=True,
                                        text=True)

                if result.returncode != 0:
                    print(f"ar failed to create empty archive: {result.returncode}")
                    print(f"STDERR: {result.stderr}")
                    return False

            # Add objects in batches
            for i in range(0, len(all_objects), batch_size):
                batch = all_objects[i:i + batch_size]
                print(f"Adding batch {i//batch_size + 1}/{(len(all_objects) + batch_size - 1)//batch_size} ({len(batch)} objects)...")

                # On macOS, use 'rcs' for first batch to create archive, 'rs' for subsequent batches
                # On Linux, always use 'rs' since empty archive was created above
                if IS_MACOS and i == 0:
                    cmd = [AR_CMD, "rcs", output_lib] + batch
                else:
                    cmd = [AR_CMD, "rs", output_lib] + batch

                result = subprocess.run(cmd,
                                        capture_output=True,
                                        text=True)

                if result.returncode != 0:
                    print(f"ar failed with return code {result.returncode}")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    return False

            # Check if library was created
            if os.path.exists(output_lib):
                lib_size = os.path.getsize(output_lib)
                print(f"✅ Successfully created {output_lib} ({lib_size:,} bytes)")

                # Show library info
                try:
                    ar_result = subprocess.run([AR_CMD, "t", output_lib],
                                              capture_output=True,
                                              text=True)
                    if ar_result.returncode == 0:
                        object_count = len(ar_result.stdout.strip().split('\n'))
                        print(f"📊 Library contains {object_count} object files")
                except:
                    print(f"ERROR: {output_lib} cannot be read")
                    return False

                # Show updated library info after post-processing
                try:
                    ar_result = subprocess.run([AR_CMD, "t", output_lib],
                                              capture_output=True,
                                              text=True)
                    if ar_result.returncode == 0:
                        final_object_count = len(ar_result.stdout.strip().split('\n'))
                        final_lib_size = os.path.getsize(output_lib)
                        print(f"Final library: {final_object_count} object files, {final_lib_size:,} bytes")
                except:
                    print("Warning: Could not read final library info")

                return True
            else:
                print(f"ERROR: {output_lib} was not created")
                return False

        except subprocess.CalledProcessError as e:
            print(f"ERROR running libtool: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    finally:
        pass

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create libchdb.a static library')
    parser.add_argument('--cross-compile', '-c', action='store_true',
                        help='Cross-compile mode (targeting macOS from Linux)')
    parser.add_argument('--build-dir', '-b', type=str, default=None,
                        help='Build directory path (relative to project root or absolute)')
    parser.add_argument('--ar-cmd', type=str, default=None,
                        help='Path to ar command (for cross-compilation)')
    return parser.parse_args()

def main():
    print("Creating libchdb.a static library...")

    # Parse arguments
    args = parse_args()

    # Setup platform based on arguments
    setup_platform(cross_compile=args.cross_compile, ar_cmd=args.ar_cmd)

    try:
        # Parse the command file
        obj_files, lib_files = parse_libchdb_cmd(build_dir_override=args.build_dir)

        # Create static library
        success = create_static_library(obj_files, lib_files)

        if success:
            print("\nSUCCESS! libchdb.a created successfully!")
        else:
            print("\nFAILED to create static library")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
