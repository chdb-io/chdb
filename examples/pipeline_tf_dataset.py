#!/usr/bin/env python3
"""Stream a chDB query into a tf.data.Dataset.

Each element is one Arrow batch stacked into ``(features, label)`` tensors.
The dataset reruns the query when iterated again, which is what an epoch is.
"""

SQL = """
SELECT
    toFloat32(number) AS x0,
    toFloat32(number % 10) AS x1,
    toInt64(number % 2) AS y
FROM numbers(500)
"""


def main():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("TensorFlow is not installed; skipping. Install it with: pip install tensorflow")
        return

    from chdb.pipeline import tf_dataset

    dataset = tf_dataset(SQL, features=["x0", "x1"], label="y", batch_size=128)
    total = 0
    batches = 0
    for features, label in dataset:
        if int(features.shape[0]) != int(label.shape[0]):
            raise SystemExit(f"mismatched batch: {features.shape} vs {label.shape}")
        if int(features.shape[1]) != 2:
            raise SystemExit(f"expected 2 features, got {features.shape}")
        total += int(features.shape[0])
        batches += 1

    if total != 500:
        raise SystemExit(f"expected 500 rows, got {total}")
    print(f"rows={total} batches={batches}")


if __name__ == "__main__":
    main()
