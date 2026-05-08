---
name: Bug report
about: Report unexpected behavior, errors, or crashes in chDB
labels: bug
---

## What happened

<!-- A clear, concise description of what went wrong. -->

## How to reproduce

<!-- The smallest code sample or query that triggers the issue. -->

```python
import chdb
# minimal repro here
```

If the bug only reproduces with specific data, attach a small file
(or describe the schema and a couple of rows).

## Expected behavior

<!-- What did you expect to happen instead? -->

## Environment

- chDB version: <!-- python -c "import chdb; print(chdb.__version__)" -->
- chdb-core version: <!-- python -c "import chdb; print(getattr(chdb, 'core_version', 'unknown'))" -->
- pandas version: <!-- python -c "import pandas; print(pandas.__version__)" -->
- Python version: <!-- python --version -->
- OS / architecture: <!-- e.g. macOS arm64, Ubuntu 22.04 x86_64 -->

## Stack trace or error output

<!-- Paste the full traceback. For crashes (SIGSEGV / SIGABRT / SIGBUS),
     please capture a native stack trace as well — see AGENTS.md §2.6
     for the gdb/lldb/faulthandler flow. -->

```
<paste here>
```

## Additional context

<!-- Anything else: data shape, file format, related issues, performance
     numbers if relevant. -->
