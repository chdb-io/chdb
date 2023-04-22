import re
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(
    current_dir, "../contrib/arrow/cpp/submodules/parquet-testing/data/alltypes_dictionary.parquet")

# reset elapsed time to 0.0 from output, since it will be different each time
# eg: "elapsed": 0.001015,


def reset_elapsed(input):
    try:
        input = re.sub(r'("elapsed": )\d+\.\d+', r'\g<1>0.0', input.decode())
        input = re.sub(r'(<elapsed>)\d+\.\d+(</elapsed>)', r'\g<1>0.0\g<2>', input)
    except UnicodeDecodeError:
        pass
    return input
