"""Subprocess entry point: run one benchmark case and print its result.

Before timing, the worker imports the libraries every case may use and opens
one throwaway session so engine start-up is not charged to the case. The
result carries ``rss_start_mb`` (RSS after this warm-up) next to
``peak_rss_mb`` so the two can be compared.
"""

import importlib
import json
import sys

from benchmark.perf_guide import common


def warm_up(payload):
    import chdb  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    from chdb import session as chs

    chs.Session(":memory:").close()
    if payload["suite"] == "bench_ml":
        try:
            import torch  # noqa: F401
        except ImportError:
            pass


def main():
    payload = json.loads(sys.argv[1])
    module = importlib.import_module(f"benchmark.perf_guide.{payload['suite']}")
    case = getattr(module, payload["case"])
    if payload["case"] != "baseline_import":
        warm_up(payload)
    result = common.timed(lambda: case(**payload["args"]))
    print("@@RESULT@@" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
