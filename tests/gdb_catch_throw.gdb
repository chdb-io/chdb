# GDB command script for catching mutex exceptions
# Usage: gdb -batch -x tests/gdb_catch_throw.gdb --args python test_script.py

set pagination off
set print thread-events off
set python print-stack full

# Allow pending breakpoints (for symbols not yet loaded)
set breakpoint pending on

# Use GDB's built-in catch throw instead of breakpoint on __cxa_throw
catch throw

# Define commands to run when exception is caught
commands
  silent
  echo \n=== C++ EXCEPTION CAUGHT ===\n
  bt full
  echo \n=== Thread info ===\n
  info threads
  echo \n=== FORCING CRASH (SIGABRT) ===\n
  signal SIGABRT
end

run
quit 0
