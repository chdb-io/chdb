# 性能与内存

chDB 把 ClickHouse 放在当前进程里。计算可以留在引擎里，只把需要的结果按批拿出来。量化行情、训练样本如果先整表变成 pandas DataFrame，占用的是解压后的全表，再加上引擎自己的基线。

这份说明覆盖内存放在哪、怎么限内存、怎么流式取结果。后面几页是具体场景：

- [量化交易](quant.md)
- [机器学习数据管道](machine-learning.md)
- [资源监控](monitoring.md)
- [把历史数据归档成 Parquet](archiving-parquet.md)
- [查询时用得上的 SQL](sql-tricks.md)

需要 `chdb-core>=26.9.0`。下面的 `system.metrics`、`system.events` 和 `ChdbError` 都是在这个版本的嵌入式会话里核对过的。

## 数据实际住在哪

| 放法 | 内存里有什么 |
| --- | --- |
| `file()` 读 Parquet / CSV | 引擎按 block 读。文件本身不进 Python 堆 |
| `Session(目录)` 上的 MergeTree | 数据在该目录的磁盘上，查询按索引读 |
| `ENGINE = Memory`，或连接串 `:memory:` 里建的小表 | 在进程内存里 |
| `query(..., "DataFrame")`、`to_df()` | 整份结果变成 pandas，进 Python 堆 |
| `send_query(..., "Arrow")` | 一次一个 RecordBatch，用完即可丢掉 |

引擎有一份固定基线。空会话里 `system.metrics` 的 `MemoryTracking` 就是几百 MB。文件远小于这个基线时，少建一个 DataFrame 在 RSS 上看不出来。文件变大，而且 SQL 只读用到的列时，差距才出现。[量化](quant.md) 里的 `examples/example_quant_bars.py` 用 200 万行、约 58MB 的 Parquet（含两列训练用不到的信号）跑过：SQL 聚合的进程峰值 RSS 低于 `pandas.read_parquet` 再 `groupby`。数字随机器变，示例会把当次的 KiB 打出来。

## 限制这一次查询的内存

连接串可以带设置。下面把线程数打进会话：

```python
from chdb import session

conn = session.Session(":memory:?max_threads=2")
print(conn.query("SELECT getSetting('max_threads')", "CSV"))
```

单条查询用 `SETTINGS` 收紧。`max_memory_usage = 0` 表示不限制。`max_bytes_before_external_group_by` 和 `max_bytes_before_external_sort` 默认也是 0，表示聚合和排序不溢写到磁盘。设成字节阈值之后，超过阈值的 `GROUP BY` / `ORDER BY` 可以落到临时文件。

```sql
SELECT symbol, sum(qty)
FROM file('ticks.parquet', Parquet)
GROUP BY symbol
SETTINGS
    max_memory_usage = 4000000000,
    max_bytes_before_external_group_by = 2000000000,
    max_bytes_before_external_sort = 2000000000,
    max_threads = 2,
    max_block_size = 65536
```

超过 `max_memory_usage` 时引擎抛 `MEMORY_LIMIT_EXCEEDED`（Code 241）。Python 里这是 `chdb.ChdbError`。

`max_block_size` 决定引擎一次吐出多大的 block。流式读取要靠它把批切小，见下一节。

## 不要一次物化整份结果

`query(..., "DataFrame")` 会把结果全部放进 pandas。大结果用 `send_query`：

```python
stream = conn.send_query(
    "SELECT number FROM numbers(100000) SETTINGS max_block_size=65536",
    "Arrow",
)
for batch in stream.record_batch(rows_per_batch=65536):
    handle(batch)  # pyarrow.RecordBatch
stream.close()
```

`record_batch(rows_per_batch=...)` 会把多个 block 攒到这个行数，但不会把一个已经形成的 block 再切开。block 比 `rows_per_batch` 大时，你拿到的就是那一整块。所以批大小要同时写进 `max_block_size`。查询里已经有 `SETTINGS` 时，`chdb.pipeline.iter_batches` 不会再改写它。

DataStore 默认懒执行：`read_parquet`、过滤、聚合先记成计划，`to_df()` / `print()` 才跑。聚合很重、又不需要和 pandas 逐行对齐时，打开 performance 模式，去掉保序和兼容包装：

```python
import chdb.datastore as pd
from chdb.datastore.config import config

config.use_performance_mode()
ds = pd.read_parquet("ticks.parquet")
result = ds[ds["px"] > 100]
```

布尔索引会下推。在 20 万行的 Parquet 上，`ds[ds["id"] == 1].to_df()` 的 `SelectedRows` 增量是几百行，不是整表。`ds.sql("SELECT ... FROM __df__ WHERE id = 1")` 结果正确，但读行数是整表量级。省内存的过滤走上面的布尔索引，或者直接写 `file()` SQL。`sql()` 的差别写在 [SQL 技巧](sql-tricks.md)。

performance 模式的行为差异（行序、`first`/`last`、分组键放在列上）见文档站的 [Performance mode](https://clickhouse.com/docs/chdb/configuration/performance-mode)。DataStore 和 pandas 的耗时对比见 [Performance guide](https://clickhouse.com/docs/chdb/guides/pandas-performance)。

## 怎么选

| 数据和使用方式 | 放法 |
| --- | --- |
| 扫描很多、结果很小（K 线、特征、报表） | Parquet 或 MergeTree，SQL 里聚合，只把结果取出来 |
| 要反复按 `(symbol, ts)` 查同一份历史 | `Session(目录)` + MergeTree，`ORDER BY (symbol, ts)` |
| 训练时要按批喂模型 | [pipeline](machine-learning.md)，`send_query` 的 Arrow 批 |
| 结果本来就小，下一步是 pandas / 画图 | `query(..., "DataFrame")` 或 `to_df()` |
| 临时表，跑完就丢，而且放得进内存 | `ENGINE = Memory` 或 `:memory:` |

列式文件优先 Parquet。CSV 每次都要解析整行。压缩默认是 zstd，过滤下推默认开着，细节在 [归档](archiving-parquet.md) 和 [SQL 技巧](sql-tricks.md)。
