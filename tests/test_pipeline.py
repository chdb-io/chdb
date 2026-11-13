"""Streaming pipeline helpers. Torch is optional; TensorFlow is optional."""

import pytest

from chdb.pipeline import iter_batches, shard_sql


def test_iter_batches_covers_every_row_within_batch_size():
    seen = []
    sizes = []
    for batch in iter_batches("SELECT number FROM numbers(1000)", batch_size=300):
        sizes.append(batch.num_rows)
        assert batch.num_rows <= 300
        seen.extend(batch.column("number").to_pylist())
    assert sizes == [300, 300, 300, 100]
    assert seen == list(range(1000))


def test_iter_batches_numpy_and_pandas_outputs():
    rows = []
    for columns in iter_batches(
        "SELECT number FROM numbers(10)", batch_size=4, output="numpy"
    ):
        rows.extend(columns["number"].tolist())
    assert rows == list(range(10))

    frames = list(
        iter_batches("SELECT number FROM numbers(5)", batch_size=2, output="pandas")
    )
    assert sum(len(frame) for frame in frames) == 5
    assert list(frames[0].columns) == ["number"]


def test_iter_batches_leaves_caller_session_open():
    from chdb import session

    conn = session.Session()
    try:
        consumed = list(iter_batches("SELECT 1 AS n", conn=conn, batch_size=10))
        assert consumed[0].column("n")[0].as_py() == 1
        assert conn.query("SELECT 2", "CSV").bytes().strip() == b"2"
    finally:
        conn.close()


def test_iter_batches_rejects_bad_arguments():
    with pytest.raises(ValueError):
        list(iter_batches("SELECT 1", batch_size=0))
    with pytest.raises(ValueError):
        list(iter_batches("SELECT 1", output="csv"))
    with pytest.raises(ValueError):
        list(iter_batches("SELECT 1; SELECT 2"))


def test_existing_settings_clause_is_not_rewritten():
    sizes = [
        batch.num_rows
        for batch in iter_batches(
            "SELECT number FROM numbers(100) SETTINGS max_block_size=100",
            batch_size=10,
        )
    ]
    assert sizes == [100]


def test_shard_sql_assigns_each_row_once():
    import chdb

    seen = []
    for shard in range(4):
        sql = shard_sql("SELECT number FROM numbers(200)", shard, 4, "number")
        text = chdb.query(sql, "CSV").bytes().decode().strip()
        if text:
            seen.extend(int(line) for line in text.splitlines())
    assert sorted(seen) == list(range(200))


def test_shard_sql_rejects_bad_arguments():
    with pytest.raises(ValueError):
        shard_sql("SELECT number FROM numbers(10)", 2, 2, "number")
    with pytest.raises(ValueError):
        shard_sql("SELECT number FROM numbers(10)", 0, 2, "num ber")
    with pytest.raises(ValueError):
        shard_sql("SELECT 1; SELECT 2", 0, 1, "number")


def test_torch_iterable_dataset_yields_every_row():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from chdb.pipeline import TorchIterableDataset

    dataset = TorchIterableDataset(
        "SELECT toFloat32(number) AS x, toInt64(number % 2) AS y FROM numbers(50)",
        features=["x"],
        label="y",
        batch_size=16,
        key="x",
    )
    total = 0
    last_x = None
    for features, label in DataLoader(dataset, batch_size=None):
        assert features.shape[0] == label.shape[0]
        assert features.dtype == torch.float32
        assert label.dtype == torch.int64
        total += features.shape[0]
        last_x = features
    assert total == 50
    assert last_x is not None


def test_torch_workers_cover_each_row_once():
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from chdb.pipeline import TorchIterableDataset

    dataset = TorchIterableDataset(
        "SELECT number AS id, toFloat32(number) AS x FROM numbers(40)",
        features=["x"],
        label="id",
        batch_size=10,
        key="id",
    )
    seen = []
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    for _features, ids in loader:
        seen.extend(int(value) for value in ids.tolist())
    assert sorted(seen) == list(range(40))
