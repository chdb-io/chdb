import re
import os
import platform
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "../tests/data/alltypes_dictionary.parquet")

# reset elapsed time to 0.0 from output, since it will be different each time
# eg: "elapsed": 0.001015,
def reset_elapsed(input):
    try:
        if not isinstance(input, str):
            input = input.decode()
        input = re.sub(r'("elapsed": )\d+\.\d+', r'\g<1>0.0', input)
        input = re.sub(r'(<elapsed>)\d+\.\d+(</elapsed>)', r'\g<1>0.0\g<2>', input)
        input = re.sub(r'(tz=).*]', r'\g<1>Etc/UTC]', input)
        input = input.replace('08:', '00:')
    except UnicodeDecodeError:
        pass
    return input


def is_musl_linux():
    """Check if running on musl Linux"""
    if platform.system() != "Linux":
        return False
    try:
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
        print(f"stdout: {result.stdout.lower()}")
        print(f"stderr: {result.stderr.lower()}")
        # Check both stdout and stderr for musl
        output_text = (result.stdout + result.stderr).lower()
        return 'musl' in output_text
    except Exception as e:
        print(f"Exception in is_musl_linux: {e}")
        return False
