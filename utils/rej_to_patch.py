import re
import sys
import os


def convert_rej_to_patch(rej_file_path, output_patch_path=None):
    # Check if the .rej file exists
    if not os.path.exists(rej_file_path):
        print(f"Error: File {rej_file_path} not found.")
        sys.exit(1)

    # Set the output file path
    if not output_patch_path:
        output_patch_path = rej_file_path.replace(".rej", ".patch")

    # Regular expressions to match diff and hunk lines
    hunk_header_regex = re.compile(r"^@@\s")

    # Process the .rej file and convert to .patch
    with open(rej_file_path, "r") as rej_file, open(
        output_patch_path, "w"
    ) as patch_file:
        inside_hunk = False
        file_a, file_b = None, None

        for line in rej_file:
            # Look for the first diff header
            if line.startswith("diff"):
                # Extract the file names (assuming `diff a/file b/file`)
                parts = line.split()
                file_a = parts[1]
                file_b = parts[2]
                inside_hunk = False  # Reset flag
                patch_file.write(line)

            # Detect hunk headers (@@ -start,length +start,length @@)
            elif hunk_header_regex.match(line):
                inside_hunk = True
                # Add the diff header if it's missing
                if file_a and file_b:
                    patch_file.write(f"--- {file_a}\n")
                    patch_file.write(f"+++ {file_b}\n")
                    file_a, file_b = None, None
                patch_file.write(line)
            # Write the patch lines that follow the hunk headers (+, -)
            elif inside_hunk:
                patch_file.write(line)

    print(f"Conversion complete: {output_patch_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rej_to_patch.py <path_to_rej_file>")
        sys.exit(1)

    rej_file_path = sys.argv[1]
    convert_rej_to_patch(rej_file_path)
