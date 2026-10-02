# 机器学习：数据读取管道

训练数据的过滤、类型、归一化统计量、训练/验证切分放在 SQL 里。PyTorch 和 TensorFlow 只按批取已经裁好的列。对应实现是 [`chdb/pipeline.py`](../../chdb/pipeline.py)，示例是 [`examples/pipeline_torch_dataloader.py`](../../examples/pipeline_torch_dataloader.py) 和 [`examples/pipeline_tf_dataset.py`](../../examples/pipeline_tf_dataset.py)。

## 特征留在查询里

统计量算一次，写进查询，而不是先 `to_df()` 再在 Python 里 `fit`：

```sql
WITH stats AS (
    SELECT avg(px) AS mean_px, stddevPop(px) AS std_px
    FROM file('ticks.parquet', Parquet)
)
SELECT
    symbol,
    (px - mean_px) / nullIf(std_px, 0) AS px_z,
    qty
FROM file('ticks.parquet', Parquet)
CROSS JOIN stats
```

训练/验证切分用 `cityHash64`，同一个键每次落在同一边：

```sql
SELECT *
FROM features
WHERE cityHash64(id) % 10 < 8   -- 训练

-- 验证集：cityHash64(id) % 10 >= 8
```

预处理结果如果会被多个 epoch 重复读，先落成 Parquet，后面只做扫描。写法与 [归档](archiving-parquet.md) 相同：`INSERT INTO FUNCTION file(..., 'Parquet') SELECT ...`。

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
):
    ...
```

查询没有 `SETTINGS` 时，helper 会加上 `max_block_size`。已有 `SETTINGS` 则原样保留，批的大小由那条子句决定。`arrow` 批在下一次迭代之前有效。`numpy` 和 `pandas` 会拷贝，调用方拿住也没关系。

`conn` 传入已有的 `Session` 时，读完不会关掉它。不传则 helper 自己开一个会话，生成器结束或被关闭时关掉。

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
    SELECT number AS id, toFloat32(number) AS x, toFloat32(number % 2) AS y
    FROM numbers(1000)
    """,
    features=["x"],
    label="y",
    batch_size=128,
    key="id",
)
loader = DataLoader(dataset, batch_size=None)
for x, y in loader:
    ...
```

多个 worker 时必须同时满足：

- 传 `key`，worker `i` 读 `cityHash64(key) % num_workers = i`
- `multiprocessing_context="spawn"`。`fork` 会带上父进程里已经初始化的引擎
- `conn=None`。会话不能跨进程共用，每个 worker 在 `__iter__` 里自己打开

```python
loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=2,
    multiprocessing_context="spawn",
)
```

`examples/pipeline_torch_dataloader.py --workers 2` 用 1000 行核对过：两个 worker 的并集仍是每一行恰好一次。列需要是数值类型。字符串列转不成张量。想要 `float32` 就在 SQL 里 `toFloat32`。

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

## 通用做法

- 批大小按模型步长和 `max_block_size` 一起定。太大，单批峰值上去；太小，引擎往返变多。示例用 128 是为了跑通，训练可以用 8192 或 65536 再量。
- SELECT 列表只留特征和标签。Parquet 上没被选中的列不会进 batch。
- 不要为了省事 `to_df()` 再 `DataLoader`。那一步把整个 epoch 放进了 Python 堆。
- 多个 epoch 读同一份预处理结果时，把结果写成 Parquet，而不是每个 epoch 都重算窗口。
- 归一化用的均值和方差在 SQL 里算完再广播，避免先把全表拉进 sklearn。
