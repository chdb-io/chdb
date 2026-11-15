#!/usr/bin/env python3
"""Stream a chDB query into a PyTorch DataLoader.

The loader is created with batch_size=None because each dataset item is already
one Arrow batch converted to tensors. With --workers 2 the scan is split by
cityHash64(id) and each worker opens its own session (spawn context).
"""

import argparse


SQL = """
SELECT
    number AS id,
    toFloat32(number) AS x,
    toFloat32(number % 2) AS y
FROM numbers(1000)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch is not installed; skipping. Install it with: pip install torch")
        return

    from chdb.pipeline import TorchIterableDataset

    dataset = TorchIterableDataset(
        SQL,
        features=["x"],
        label="y",
        batch_size=128,
        key="id",
    )
    loader_kwargs = {"batch_size": None}
    if args.workers > 0:
        loader_kwargs["num_workers"] = args.workers
        loader_kwargs["multiprocessing_context"] = "spawn"

    total = 0
    batches = 0
    for features, label in DataLoader(dataset, **loader_kwargs):
        if features.shape[0] != label.shape[0]:
            raise SystemExit(f"mismatched batch: {features.shape} vs {label.shape}")
        total += features.shape[0]
        batches += 1

    if total != 1000:
        raise SystemExit(f"expected 1000 rows, got {total}")
    print(f"rows={total} batches={batches} workers={args.workers}")


if __name__ == "__main__":
    main()
