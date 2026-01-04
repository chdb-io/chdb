# GDB command script for catching mutex exceptions
# Usage: gdb -batch -x tests/gdb_catch_throw.gdb --args python test_script.py

set pagination off
set print thread-events off
set python print-stack full

# Load the Python exception catcher
source tests/gdb_catch_mutex_error.py

run
quit 0
