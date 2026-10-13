# 机器学习：数据读取管道

训练数据的过滤、类型、归一化统计量、训练/验证切分放在 SQL 里。PyTorch 和 TensorFlow 只按批取已经裁好的列。对应实现是 [`chdb/pipeline.py`](../../chdb/pipeline.py)，示例是 [`examples/pipeline_torch_dataloader.py`](../../examples/pipeline_torch_dataloader.py) 和 [`examples/pipeline_tf_dataset.py`](../../examples/pipeline_tf_dataset.py)。

这一页的数字来自 [`benchmark/perf_guide/bench_ml.py`](../../benchmark/perf_guide/bench_ml.py)：出租车数据 2024 年 1 到 3 月（955 万行）和全年（4117 万行），7 个 `Float32` 特征加 1 个标签。机器和方法见 [总览](index.md)。

## 先说结论

| 场景 | 建议 |
| --- | --- |
| 整个 epoch 的特征放得进内存（几百万行、十几列） | `query(..., "DataFrame")` 一次拿完再切张量最快。流式在这个规模上没有内存优势 |
| 放不进、或者不想让训练进程常驻 8 GB 以上 | `TorchIterableDataset` / `iter_batches`，并把会话的 `max_threads` 调到 1 到 2 |
| DataLoader 的 `num_workers` | 读数据不需要。只有 Python 侧每批有重的变换时才开 |
| 特征里有窗口函数、JOIN 这类重计算，且 epoch ≥ 3 | 先 `INSERT INTO FUNCTION file(...)` 落成 Parquet，每个 epoch 只扫描 |
| 训练/验证切分 | `cityHash64(key) % 10 < 8`，一次扫描，开销可忽略 |

## 一次拿完 vs 按批流式

一个季度 955 万行，特征 7 列 + 标签：

| 做法 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| `pd.read_parquet(columns=...)` + `fillna` + `torch.from_numpy` | 0.40 s | 2,677 MB |
| chDB `query(..., "DataFrame")` + `torch.from_numpy` | 0.44 s | 2,232 MB |
| `TorchIterableDataset` + `DataLoader(batch_size=None)`，16 线程 | 1.00 s | 1,921 MB |

955 万行 × 8 列 `Float32` 只有 300 MB，一次拿完是对的。流式路径多出来的 0.6 秒是逐批做 Arrow → 张量的拷贝，省下的内存也不多，因为引擎读 Parquet 本身就要 1.5 GB 左右的工作集（见 [总览](index.md#物化整份结果还是流式拿)）。

全年 4117 万行：

| 做法 | 耗时 | 峰值 RSS（父进程） | 每个 worker 峰值 |
| --- | --- | --- | --- |
| `pd.read_parquet` + `torch.from_numpy` | 1.59 s | 8,513 MB | |
| chDB `"DataFrame"` + `torch.from_numpy` | 1.04 s | 7,569 MB | |
| `TorchIterableDataset`，16 线程 | 3.93 s | 4,137 MB | |
| `TorchIterableDataset`，`max_threads = 2` | 3.81 s | 2,058 MB | |
| `TorchIterableDataset`，`max_threads = 2`，`num_workers = 4` | 3.55 s | 526 MB | 1,663 MB |

- 一次拿完的峰值是最终张量（1.3 GB）的 6 倍：引擎输出、Arrow 中间表、pandas DataFrame、`to_numpy` 拷贝在某个时刻同时存在。数据再大一倍这个数就是 15 GB。
- 流式把峰值钉在 2 GB，和数据量无关。代价是耗时 3 到 4 倍，也就是每秒 1000 万行左右的吞吐，对多数训练循环来说已经比模型的一步快。
- `max_threads` 从 16 降到 2，内存减半，速度反而略快。消费端是单线程 Python，引擎多线程只是在堆积排队的 block。
- 4 个 worker 的那一行，父进程只剩 526 MB，是因为读数据的工作搬到了子进程里；加上 4 个 worker 的 1.6 GB，总量并没有少。

## 吞吐和批大小

`iter_batches` 在同一个季度数据上：

| 设置 | 吞吐（行/秒） | 峰值 RSS |
| --- | --- | --- |
| 批 4,096，arrow 输出 | 806 万 | 1,879 MB |
| 批 65,536，arrow | 786 万 | 1,899 MB |
| 批 1,048,576，arrow | 791 万 | 2,058 MB |
| 批 65,536，numpy 输出 | 749 万 | 1,932 MB |
| 批 65,536，pandas 输出 | 769 万 | 1,939 MB |
| 批 65,536，arrow，`max_threads = 1` | 1,190 万 | 1,263 MB |
| 批 65,536，arrow，`max_threads = 2` | 845 万 | 1,677 MB |

批大小从 4 千到 100 万对吞吐没有影响，输出格式的差别也在 5% 以内。真正起作用的是 `max_threads`：1 线程比 16 线程快 50%，还少用 600 MB。按模型步长定批就好，不要为了吞吐把批调大。

## DataLoader 的 worker 不会让读取变快

| `num_workers` | 耗时 | 吞吐 | 父进程 RSS | 每个 worker RSS |
| --- | --- | --- | --- | --- |
| 0 | 1.00 s | 1,082 万行/秒 | 1,921 MB | |
| 2（spawn，`key="rk"`） | 2.57 s | 371 万行/秒 | 521 MB | 1,791 MB |
| 4 | 2.38 s | 411 万行/秒 | 515 MB | 1,489 MB |

每个 worker 都要把整份数据扫一遍、算一遍 `cityHash64`，只留下属于自己的那份。引擎本来就是多线程的，再拆进程只是重复读。worker 的价值在 Python 侧：每批要做图像解码、分词、复杂增强这种比读取慢得多的事时，让 4 个进程各做四分之一才划算。

多 worker 时必须同时满足：

- 传 `key`，worker `i` 读 `cityHash64(key) % num_workers = i`。`key` 要是一个标识符，复合键先在 SQL 里 `cityHash64(a, b, c) AS rk` 再传 `"rk"`
- `multiprocessing_context="spawn"`。`fork` 会带上父进程里已经初始化的引擎
- `conn=None`。会话不能跨进程共用，每个 worker 在 `__iter__` 里自己打开

```python
loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=4,
    multiprocessing_context="spawn",
)
```

`tests/test_pipeline.py` 用两个 worker 核对过：并集仍是每一行恰好一次。

## 每个 epoch 重算，还是先落盘

同一个季度跑 3 个 epoch：

| 特征 SQL | 每个 epoch 重算（3 轮） | 一次性落盘 | 从落盘文件读 3 轮 |
| --- | --- | --- | --- |
| 轻：`log1p`、`toHour`、`dateDiff`、除法（6 个表达式） | 3.73 s | 0.45 s，122 MB | 3.90 s |
| 重：按 `PULocationID` 分区、按时间排序的 200 行窗口均值和计数 | 5.85 s | 2.33 s，119 MB | 3.15 s |

轻特征落盘没有收益：标量表达式的计算量比把数据搬到 Python 还小。重特征（窗口函数、JOIN、分组统计量）从第 3 个 epoch 起回本，并且每轮省下的时间固定。写法与 [归档](archiving-parquet.md) 相同：

```sql
INSERT INTO FUNCTION file('features.parquet', Parquet)
SELECT
    toFloat32(fare_amount) AS fare_amount,
    toFloat32(avg(fare_amount) OVER w) AS zone_fare_ma,
    toFloat32(count() OVER w) AS zone_recent_trips,
    toFloat32(total_amount) AS total_amount
FROM file('taxi/*.parquet', Parquet)
WINDOW w AS (PARTITION BY PULocationID ORDER BY tpep_pickup_datetime ROWS BETWEEN 200 PRECEDING AND CURRENT ROW)
```

## 切分和分片的开销

955 万行上：

| 查询 | 行数 | SelectedBytes | 耗时 |
| --- | --- | --- | --- |
| 全量 `count(), sum(total_amount)` | 9,554,778 | 15.8 MB | 0.19 s |
| `WHERE cityHash64(...) % 10 < 8`（训练集） | 7,642,549 | 126 MB | 0.16 s |
| `WHERE cityHash64(...) % 10 >= 8`（验证集） | 1,912,229 | 126 MB | 0.27 s |
| `shard_sql(sql, 0, 4, "rk")` | 2,388,262 | 126 MB | 0.21 s |

多读的 110 MB 是哈希要用到的 6 个键列。耗时的差别在底噪里。同一个键每次落在同一边，不需要先物化一个 id 列表。

```sql
SELECT *
FROM features
WHERE cityHash64(id) % 10 < 8   -- 训练

-- 验证集：cityHash64(id) % 10 >= 8
```

## 特征留在查询里

统计量算一次，写进查询，而不是先 `to_df()` 再在 Python 里 `fit`：

```sql
WITH stats AS (
    SELECT avg(fare_amount) AS mean_fare, stddevPop(fare_amount) AS std_fare
    FROM file('taxi/*.parquet', Parquet)
)
SELECT
    (fare_amount - mean_fare) / nullIf(std_fare, 0) AS fare_z,
    toFloat32(trip_distance) AS trip_distance
FROM file('taxi/*.parquet', Parquet)
CROSS JOIN stats
```

已经在内存里的 `pyarrow.Table`、`RecordBatch` 或 pandas DataFrame 可以用 `FROM Python(变量名)` 直接查，不必再写盘。26.9 起 RecordBatch 和 `pyarrow.dataset.Dataset` 都能这样用。这份数据本来就在堆上，省内存要靠不要把它再复制成一份 DataFrame，而不是靠 `Python()`。

## chdb.pipeline

```python
from chdb.pipeline import iter_batches, shard_sql, TorchIterableDataset, tf_dataset
```

`import chdb.pipeline` 不要求安装 TensorFlow。装了 PyTorch 时，导入这个模块会加载它，因为 DataLoader 的 worker 要 pickle `TorchIterableDataset`。没装 PyTorch 时模块仍能导入，构造数据集时才报错。

### iter_batches

```python
for batch in iter_batches(
    "SELECT id, toFloat32(x) AS x FROM file('train.parquet', Parquet)",
    batch_size=65536,
    output="arrow",  # 或 "numpy" / "pandas"
    conn=session.Session(":memory:?max_threads=2"),
):
    ...
```

查询没有 `SETTINGS` 时，helper 会加上 `max_block_size`。已有 `SETTINGS` 则原样保留，批的大小由那条子句决定。`arrow` 批在下一次迭代之前有效。`numpy` 和 `pandas` 会拷贝，调用方拿住也没关系。

`conn` 传入已有的 `Session` 时，读完不会关掉它。不传则 helper 自己开一个会话，生成器结束或被关闭时关掉。想限制线程数，就把带 `max_threads` 的会话传进来，或者在 SQL 末尾写 `SETTINGS max_threads = 2, max_block_size = 65536`。

### 分片

```python
sql = shard_sql(
    "SELECT id, x, y FROM file('train.parquet', Parquet)",
    shard=0,
    num_shards=4,
    key="id",
)
```

生成的条件是 `cityHash64(id) % 4 = 0`。`key` 只能是一个标识符。同一个 `(key, num_shards)` 下每一行恰好属于一个 shard。

### PyTorch

`TorchIterableDataset` 每次 yield 已经是一个 batch 的张量，所以 DataLoader 要用 `batch_size=None`，否则 PyTorch 会把这些 batch 再堆一维。

```python
from torch.utils.data import DataLoader
from chdb.pipeline import TorchIterableDataset

dataset = TorchIterableDataset(
    """
    SELECT toFloat32(trip_distance) AS trip_distance,
           toFloat32(fare_amount) AS fare_amount,
           toFloat32(total_amount) AS total_amount,
           cityHash64(tpep_pickup_datetime, PULocationID, fare_amount) AS rk
    FROM file('taxi/*.parquet', Parquet)
    SETTINGS max_threads = 2, max_block_size = 65536
    """,
    features=["trip_distance", "fare_amount"],
    label="total_amount",
    batch_size=65536,
    key="rk",
)
loader = DataLoader(dataset, batch_size=None)
for x, y in loader:
    ...
```

列需要是数值类型。字符串列转不成张量。想要 `float32` 就在 SQL 里 `toFloat32`。

### TensorFlow

```python
from chdb.pipeline import tf_dataset

dataset = tf_dataset(
    """
    SELECT toFloat32(number) AS x0, toFloat32(number % 10) AS x1, toInt64(number % 2) AS y
    FROM numbers(500)
    """,
    features=["x0", "x1"],
    label="y",
    batch_size=128,
)
```

没传 `output_signature` 时，helper 先跑 `SELECT * FROM (sql) LIMIT 1` 推断 dtype，所以这条 SQL 必须能安全地执行两遍。每个 epoch 重新迭代 dataset 都会再跑一遍查询。`tf.data` 的 worker 切分不在这个函数里，要切就先用 `shard_sql` 包好再传进来。
