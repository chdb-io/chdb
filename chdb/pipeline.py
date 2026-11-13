"""Stream chDB query results into training loops.

``iter_batches`` pulls one Arrow batch at a time from ``Session.send_query``.
``TorchIterableDataset`` and ``tf_dataset`` adapt that stream to PyTorch and
TensorFlow. TensorFlow is imported only by ``tf_dataset``. PyTorch is imported
when this module loads if it is already installed, because DataLoader workers
must pickle ``TorchIterableDataset``; if PyTorch is absent the module still
imports and the dataset raises when constructed.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SETTINGS = re.compile(r"\bSETTINGS\b", re.IGNORECASE)

try:
    from torch.utils.data import IterableDataset as _TorchIterableDatasetBase
    from torch.utils.data import get_worker_info as _get_worker_info
except ImportError:  # torch is optional; the dataset raises if it is missing
    class _TorchIterableDatasetBase:  # type: ignore[no-redef]
        """Stand-in so the class body imports without PyTorch."""

    def _get_worker_info():
        return None

    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


def iter_batches(
    sql: str,
    *,
    batch_size: int = 65536,
    conn: Any = None,
    output: str = "arrow",
) -> Iterator[Any]:
    """Yield successive pieces of ``sql`` without holding the full result.

    ``batch_size`` is applied as ``max_block_size`` when ``sql`` has no
    ``SETTINGS`` clause, and as ``record_batch(rows_per_batch=...)``. A block
    the engine has already formed is not split, so an existing ``SETTINGS``
    clause keeps whatever ``max_block_size`` it sets.

    ``output`` is ``"arrow"`` (a ``pyarrow.RecordBatch``), ``"numpy"`` (a dict
    of column name to a copied ``ndarray``), or ``"pandas"``. Arrow batches are
    valid until the next batch is pulled. Numpy and pandas outputs own their
    memory.

    ``conn`` is an object with ``send_query``, typically ``chdb.session.Session``.
    When it is omitted, a session is opened for this stream and closed when
    the generator finishes or is closed.
    """
    if output not in ("arrow", "numpy", "pandas"):
        raise ValueError('output must be "arrow", "numpy", or "pandas"')
    batch_size = _positive_int(batch_size, "batch_size")
    statement = _prepare_sql(sql, batch_size)

    session, owned = _open_session(conn)
    stream = None
    try:
        stream = session.send_query(statement, "Arrow")
        for batch in stream.record_batch(rows_per_batch=batch_size):
            if output == "arrow":
                yield batch
            elif output == "numpy":
                yield _batch_to_numpy(batch)
            else:
                yield batch.to_pandas()
    finally:
        if stream is not None:
            stream.close()
        if owned:
            session.close()


def shard_sql(sql: str, shard: int, num_shards: int, key: str) -> str:
    """Restrict ``sql`` to one deterministic shard of ``key``.

    Rows are kept when ``cityHash64(key) % num_shards = shard``. ``key`` is a
    single SQL identifier. The same ``(key, num_shards)`` pair assigns every
    row to exactly one shard, so workers can split a scan without coordinating.
    """
    shard = int(shard)
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard < num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    if not isinstance(key, str) or _IDENTIFIER.fullmatch(key) is None:
        raise ValueError("key must be a simple SQL identifier")
    body = _single_statement(sql)
    return (
        f"SELECT * FROM ({body}) "
        f"WHERE cityHash64({key}) % {num_shards} = {shard}"
    )


def tf_dataset(
    sql: str,
    features: list,
    label: Optional[str] = None,
    *,
    batch_size: int = 65536,
    output_signature: Any = None,
    conn: Any = None,
):
    """Return a ``tf.data.Dataset`` that reruns ``sql`` for each epoch.

    Each element is a ``(features, label)`` pair of tensors, or just the
    feature tensor when ``label`` is omitted. Feature columns are stacked on
    axis 1. When ``output_signature`` is omitted, dtypes are taken from a
    one-row probe, so ``sql`` must be safe to run twice. To give one worker a
    slice of the scan, pass :func:`shard_sql` as ``sql``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for tf_dataset. Install it with: pip install tensorflow"
        ) from exc

    feature_names = _column_names(features, "features")
    if label is not None:
        _column_names([label], "label")

    if output_signature is None:
        output_signature = _infer_tf_signature(
            tf, sql, feature_names, label, batch_size, conn
        )

    def generator():
        for columns in iter_batches(sql, batch_size=batch_size, conn=conn, output="numpy"):
            features_array = _stack_columns(columns, feature_names)
            if label is None:
                yield features_array
            else:
                yield features_array, columns[label]

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


class TorchIterableDataset(_TorchIterableDatasetBase):
    """Iterable dataset that yields one tensor batch per Arrow batch.

    Use it with ``DataLoader(..., batch_size=None)`` so PyTorch does not stack
    these batches again. With ``num_workers > 0``, pass ``key`` and start the
    loader with ``multiprocessing_context="spawn"``. Each worker opens its own
    session inside ``__iter__`` and reads
    ``cityHash64(key) % num_workers = worker_id``. Do not pass ``conn`` into a
    multi-worker loader.
    """

    def __init__(
        self,
        sql: str,
        features: list,
        label: Optional[str] = None,
        *,
        batch_size: int = 65536,
        transform: Any = None,
        conn: Any = None,
        key: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TorchIterableDataset. Install it with: pip install torch"
            )
        self.sql = sql
        self.features = _column_names(features, "features")
        self.label = None if label is None else _column_names([label], "label")[0]
        self.batch_size = batch_size
        self.transform = transform
        self.conn = conn
        self.key = key

    def __iter__(self):
        import torch

        info = _get_worker_info()
        sql = self.sql
        conn = self.conn
        if info is not None and info.num_workers > 0:
            if self.key is None:
                raise ValueError("key is required when DataLoader num_workers > 0")
            if conn is not None:
                raise ValueError(
                    "conn cannot be shared with DataLoader workers; leave conn=None"
                )
            sql = shard_sql(sql, info.id, info.num_workers, self.key)

        for batch in iter_batches(
            sql, batch_size=self.batch_size, conn=conn, output="arrow"
        ):
            columns = [_column_tensor(torch, batch, name) for name in self.features]
            features = columns[0] if len(columns) == 1 else torch.stack(columns, dim=1)
            if self.label is None:
                item = features
            else:
                item = (features, _column_tensor(torch, batch, self.label))
            if self.transform is not None:
                item = self.transform(item)
            yield item


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _single_statement(sql: str) -> str:
    if not isinstance(sql, str):
        raise TypeError("sql must be a string")
    body = sql.strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if not body:
        raise ValueError("sql is empty")
    if ";" in body:
        raise ValueError("sql must be a single statement")
    return body


def _prepare_sql(sql: str, batch_size: int) -> str:
    body = _single_statement(sql)
    if _SETTINGS.search(body):
        return body
    return f"{body} SETTINGS max_block_size={batch_size}"


def _open_session(conn: Any):
    if conn is not None:
        if not hasattr(conn, "send_query"):
            raise TypeError("conn must provide send_query()")
        return conn, False
    from chdb import session

    return session.Session(), True


def _column_names(names: list, what: str) -> list:
    if isinstance(names, str) or not names:
        raise ValueError(f"{what} must be a non-empty list of column names")
    cleaned = []
    for name in names:
        if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None:
            raise ValueError(f"{what} entries must be simple SQL identifiers")
        cleaned.append(name)
    return cleaned


def _batch_to_numpy(batch):
    import numpy as np

    columns = {}
    for name in batch.schema.names:
        columns[name] = np.array(batch.column(name).to_numpy(zero_copy_only=False), copy=True)
    return columns


def _column_tensor(torch, batch, name: str):
    import numpy as np

    array = np.array(batch.column(name).to_numpy(zero_copy_only=False), copy=True)
    return torch.from_numpy(array)


def _stack_columns(columns: dict, names: list):
    import numpy as np

    arrays = [columns[name] for name in names]
    if len(arrays) == 1:
        return arrays[0]
    return np.column_stack(arrays)


def _infer_tf_signature(tf, sql, features, label, batch_size, conn):
    probe_sql = f"SELECT * FROM ({_single_statement(sql)}) LIMIT 1"
    try:
        probe = next(
            iter_batches(probe_sql, batch_size=batch_size, conn=conn, output="numpy")
        )
    except StopIteration as exc:
        raise ValueError("sql returned no rows; pass output_signature explicitly") from exc

    features_array = _stack_columns(probe, features)
    feature_spec = tf.TensorSpec(
        shape=(None,) + tuple(features_array.shape[1:]),
        dtype=tf.as_dtype(features_array.dtype),
    )
    if label is None:
        return feature_spec
    labels = probe[label]
    label_spec = tf.TensorSpec(
        shape=(None,) + tuple(labels.shape[1:]),
        dtype=tf.as_dtype(labels.dtype),
    )
    return (feature_spec, label_spec)
