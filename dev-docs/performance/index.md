# 性能与内存

chDB 把 ClickHouse 放在当前进程里。计算可以留在引擎里，只把需要的结果按批拿出来。行情、训练样本如果先整表变成 pandas DataFrame，占用的是解压后的全表，再加上引擎自己的基线。

这份说明覆盖内存放在哪、怎么限内存、怎么流式取结果，所有数字都来自同一套可复现的基准。后面几页是具体场景：

- [量化交易](quant.md)
- [机器学习数据管道](machine-learning.md)
- [资源监控](monitoring.md)
- [把历史数据归档成 Parquet](archiving-parquet.md)
- [查询时用得上的 SQL](sql-tricks.md)

需要 `chdb-core>=26.9.0`。

## 基准怎么跑的

代码在 [`benchmark/perf_guide/`](../../benchmark/perf_guide/)，原始结果在 [`benchmark/perf_guide/results/`](../../benchmark/perf_guide/results/)，`python -m benchmark.perf_guide.report` 会把结果渲染成表格。本页和后面几页的表都是从那里截出来的。

| 项目 | 取值 |
| --- | --- |
| 机器 | AMD EPYC 9654（96 核 192 线程），377 GB 内存，NVMe，Ubuntu 24.04，Linux 6.8 |
| 软件 | chdb-core 26.9.0（引擎 26.9.2.1），Python 3.11.9，pandas 2.3.3，pyarrow 26.0.0，torch 2.14 CPU |
| 线程 | 除非写明，所有会话用 `max_threads=16` |
| 重复 | 每个 case 跑 3 次（SQL 技巧和量化两页是 5 次），耗时取中位数，峰值内存取最大值 |
| 隔离 | 每个 case 在新的子进程里跑。先 `import chdb / pandas / pyarrow`、开一个会话再关掉，然后才计时 |
| 耗时 | `wall_s`：case 函数本身的时间，不含进程启动和导入 |
| 峰值内存 | `peak_rss_mb`：进程的 `VmHWM`，也就是整个子进程生命周期的最高驻留 |
| 引擎读了多少 | `system.events` 里 `SelectedRows` / `SelectedBytes` 在 case 前后的差。对 Parquet，`SelectedBytes` 是真正从文件里选出的字节；`SelectedRows` 接近过滤后的行数，不适合拿来看裁剪效果 |

数据是公开数据：

| 数据集 | 内容 | 行数 | 大小 |
| --- | --- | --- | --- |
| taxi | NYC 黄色出租车 2024 年 12 个月，每月一个 Parquet，19 列 | 4117 万 | 693 MB Parquet；1 月单月 50 MB Parquet / 335 MB CSV |
| ticks | Binance 现货 aggTrades，BTCUSDT + ETHUSDT，2024-01-01 到 01-31，Hive 目录 `day=…/symbol=…` | 6302 万（BTC 4196 万） | 527 MB Parquet（zstd） |
| klines | Binance BTCUSDT 1 秒 K 线，同一个月 | 268 万 | 57 MB |

`python -m benchmark.perf_guide.prepare_data` 会下载并转换它们。这台机器空闲时的负载在 5 到 20 之间（其他人也在用），所以 0.2 秒以下的耗时差别不要当真：那已经是这套工具的底噪。

## 数据实际住在哪

| 放法 | 内存里有什么 |
| --- | --- |
| `file()` 读 Parquet / CSV | 引擎按 block 读。文件本身不进 Python 堆 |
| `Session(目录)` 上的 MergeTree | 数据在该目录的磁盘上，查询按主键读 |
| `ENGINE = Memory`，或连接串 `:memory:` 里建的小表 | 在进程内存里 |
| `query(..., "DataFrame")`、`to_df()` | 整份结果变成 pandas，进 Python 堆 |
| `send_query(..., "Arrow")` | 一次一个 RecordBatch，用完即可丢掉 |

进程的固定开销：

| 阶段 | 峰值 RSS | 说明 |
| --- | --- | --- |
| `import chdb` | 233 MB | 共享库本身 |
| 再开一个 `Session(":memory:")` | 328 MB | `system.metrics` 里 `MemoryTracking` 只有 72 MB，其余是库和分配器 |

也就是说，任何用 chDB 的进程起步就是 300 多 MB。下面所有的峰值内存都包含这一块。

## 一次聚合：4117 万行出租车，按 (passenger_count, payment_type) 分组

```sql
SELECT passenger_count, payment_type, count() AS trips, sum(total_amount) AS revenue
FROM file('taxi/*.parquet', Parquet)
GROUP BY passenger_count, payment_type
```

| 做法 | 耗时 | 峰值 RSS | 备注 |
| --- | --- | --- | --- |
| `pd.read_parquet` 全列 → `concat` → `groupby` | 8.04 s | 14,364 MB | DataFrame 本身 7,683 MB |
| `pd.read_parquet(columns=[3 列])` → `groupby` | 1.73 s | 4,178 MB | DataFrame 988 MB |
| chDB SQL，`"DataFrame"` 输出 | 0.15 s | 1,079 MB | 结果 41 行 |
| DataStore `read_parquet(...).groupby(...).agg(...)`，pandas 兼容模式 | 0.24 s | 1,308 MB | |
| DataStore，performance 模式 | 0.27 s | 1,313 MB | |

pandas 这边的 14 GB 不是 pandas 的错，是“先把整表变成 DataFrame”这一步的代价：693 MB 的 Parquet 解压成 7.7 GB 的 DataFrame，中间还有 Arrow 到 pandas 的转换副本。只选三列后降到 4 GB。SQL 那条只把需要的两列读出来、在引擎里聚合，Python 侧只拿到 41 行。

## 物化整份结果，还是流式拿

同样一条 `SELECT *`，取 2024 年 1 到 3 月（955 万行、19 列）：

| 做法 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| `query(..., "DataFrame")` | 2.14 s | 6,290 MB（DataFrame 1,809 MB） |
| `query(..., "ArrowTable")` | 1.53 s | 5,055 MB（Arrow 表 1,309 MB） |
| `send_query(..., "Arrow")` 流式，批 65,536 行 | 2.69 s | 2,609 MB |
| 流式，批 8,192 行 | 2.53 s | 2,530 MB |
| 流式，批 1,048,576 行 | 2.76 s | 2,952 MB |
| 流式，`max_threads=1` | 2.11 s | 1,980 MB |
| 流式，`max_threads=1, max_parsing_threads=1, max_download_threads=1` | 2.11 s | 950 MB |

物化 DataFrame 的峰值是最终 DataFrame 大小的 3.5 倍：引擎的输出 block、Arrow 中间表、pandas 的最终拷贝在某一刻同时存在。流式读把这一项换成了一个批的大小。

流式路径的内存不随数据量增长。把范围放大到全年 4117 万行：

| 做法 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| `query(..., "DataFrame")` 全年 | 8.23 s | 24,155 MB（DataFrame 7,766 MB） |
| 流式，默认 16 线程 | 12.53 s | 3,946 MB |
| 流式，`max_threads=1, max_parsing_threads=1, max_download_threads=1` | 9.09 s | 1,118 MB |

两个结论：

- 流式读的时候，消费方是单线程的 Python。引擎线程多只会让更多解压好的 Parquet 数据排队等着被取走，既占内存又不更快。这个场景把 `max_threads` 和 `max_parsing_threads` 一起设成 1，内存从 3.9 GB 降到 1.1 GB，还快了 27%。
- 批大小对速度几乎没影响。8,192 到 1,048,576 之间，耗时在 2.5 到 2.8 秒之间波动；批越大峰值越高。按模型需要的步长定批就行。

`chdb.pipeline.iter_batches` 是在 `send_query` 上封装的，开销在同一量级（arrow 输出 3.47 s，numpy 4.21 s，pandas 3.81 s，峰值都在 2.8 GB 左右）。

`record_batch(rows_per_batch=...)` 会把多个 block 攒到这个行数，但不会把一个已经形成的 block 再切开。所以批大小要同时写进 `max_block_size`。查询里已经有 `SETTINGS` 时，`iter_batches` 不会再改写它。

## 限制这一次查询的内存

连接串可以带设置，会话里的所有查询都生效：

```python
from chdb import session

conn = session.Session(":memory:?max_threads=2&max_memory_usage=4000000000")
```

单条查询用 `SETTINGS` 收紧。`max_memory_usage = 0` 表示不限制。`max_bytes_before_external_group_by` 和 `max_bytes_before_external_sort` 默认也是 0，表示聚合和排序不溢写到磁盘。

下面是一条真正吃内存的聚合：按 `(pickup_datetime, PULocationID, DOLocationID)` 分组，4117 万行里有 4058 万个不同的组，几乎每行一个组。

| 设置 | 结果 | 耗时 | 引擎峰值（MemoryTracking） | 进程峰值 RSS |
| --- | --- | --- | --- | --- |
| 默认 | 成功 | 0.93 s | 8,461 MB | 10,933 MB |
| `max_bytes_before_external_group_by = 1 GiB` | 成功 | 1.59 s | 1,223 MB | 4,339 MB |
| `max_memory_usage = 4 GiB` | `MEMORY_LIMIT_EXCEEDED`（Code 241） | 0.57 s | 4,085 MB | 6,265 MB |
| `max_memory_usage = 4 GiB` + 外部聚合 1 GiB | 成功 | 1.33 s | 1,256 MB | 4,213 MB |
| `max_memory_usage = 1 GiB` + 外部聚合 256 MiB | `MEMORY_LIMIT_EXCEEDED` | 1.07 s | 1,021 MB | 2,564 MB |
| `max_memory_usage = 4 GiB` + 外部聚合 1 GiB + `max_threads = 4` | 成功 | 3.79 s | 1,792 MB | 3,890 MB |

怎么读这张表：

- 外部聚合把引擎的记账峰值从 8.5 GB 压到 1.2 GB，代价是耗时从 0.93 秒到 1.59 秒。溢写出去的临时文件放在引擎的临时目录里，查询结束就删。
- 只设 `max_memory_usage` 不设外部聚合，引擎到了上限就直接抛错，不会自己变慢换安全。两个要一起设。
- 上限也不能设得比外部聚合阈值高太少：1 GiB 上限配 256 MiB 阈值仍然失败，因为溢写后合并各线程的哈希表那一步本身就要超过 1 GiB。经验上限 ≥ 阈值的 3 到 4 倍。
- 进程 RSS 比引擎记账高 2 到 3 GB。多出来的是分配器拿到手但还没还给系统的页，加上 Python 和共享库。给容器定内存上限时看 RSS 那一列。
- `max_threads` 降到 4 并没有再省多少内存，反而慢了近 3 倍。省内存先用外部聚合，不要先砍线程。

超过 `max_memory_usage` 时引擎抛 `MEMORY_LIMIT_EXCEEDED`，Python 里这是 `chdb.ChdbError`，可以捕获后换更保守的设置重试。

## 线程数

同一条 41 行结果的 GROUP BY（4117 万行输入）：

| `max_threads` | 耗时 | 峰值 RSS |
| --- | --- | --- |
| 1 | 0.88 s | 553 MB |
| 2 | 0.49 s | 691 MB |
| 4 | 0.34 s | 872 MB |
| 8 | 0.30 s | 1,012 MB |
| 16 | 0.29 s | 1,076 MB |
| 32 | 0.13 s | 916 MB |
| 64 | 0.13 s | 867 MB |

从 1 到 4 线程接近线性，之后收益递减。前几个线程每个大约多 100 MB 的工作内存（每个线程一份聚合状态和读缓冲），再往上趋平。不加设置时 chDB 用机器所有核；在这台 192 线程的机器上，意味着一条简单聚合可能起 192 个线程。与别人共用的机器、或者容器里给了 2 个核却看到 192 个线程时，把 `max_threads` 写进连接串。

## 文件格式和列裁剪

| 对比 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| 1 月 Parquet（50 MB），按 passenger_count 聚合 | 0.13 s | 374 MB |
| 同一个月的 CSV（335 MB） | 0.57 s | 964 MB |
| 全年 4117 万行只读 2 列：`avg(total_amount), avg(trip_distance)` | 0.25 s | 694 MB |
| 一个季度 955 万行 `SELECT *` 成 DataFrame | 0.97 s | 6,314 MB |

CSV 每次都要解析整行，没有列裁剪也没有统计信息。只要文件会被读两次以上，先转成 Parquet（写法在 [归档](archiving-parquet.md)）。Parquet 上没被选中的列不会被读：全年两列的 `SelectedBytes` 是 126 MB，季度 `SELECT *` 是 160 MB，而后者只有前者四分之一的行。

## DataStore 里的下推

DataStore 默认懒执行：`read_parquet`、过滤、聚合先记成计划，`to_df()` / `print()` / `len()` 才跑。布尔索引和列选择会合成一条带过滤的 `file()` 查询。

全年出租车数据上取 `total_amount > 200 AND passenger_count >= 4` 的两列（2,282 行）：

| 写法 | 耗时 | 峰值 RSS | SelectedRows | SelectedBytes |
| --- | --- | --- | --- | --- |
| `ds[(ds.total_amount > 200) & (ds.passenger_count >= 4)][["total_amount", "trip_distance"]]` | 0.25 s | 609 MB | 68,194 | 64 MB |
| `ds.sql("SELECT total_amount, trip_distance FROM __df__ WHERE ...")` | 7.23 s | 13,730 MB | 41,172,975 | 693 MB |

`ds.sql()` 先把 `__df__` 物化成一个 DataFrame 再在上面跑 SQL，所以读了整表、占了 13.7 GB。要写 SQL 就直接 `Session().query` 对 `file()` 写，不要经过 `__df__`。

聚合很重、又不需要和 pandas 逐行对齐时，`config.use_performance_mode()` 去掉保序和兼容包装。上面的分组测试里两种模式耗时相同，差别在行序和 `first`/`last` 的语义，见 [Performance mode](https://clickhouse.com/docs/chdb/configuration/performance-mode)。

## 怎么选

| 数据和使用方式 | 放法 |
| --- | --- |
| 扫描很多、结果很小（K 线、特征、报表） | Parquet 或 MergeTree，SQL 里聚合，只把结果取出来 |
| 要反复按 `(symbol, ts)` 查同一份历史 | `Session(目录)` + MergeTree，`ORDER BY (symbol, ts)` |
| 训练时要按批喂模型 | [pipeline](machine-learning.md)，`send_query` 的 Arrow 批，`max_threads` 调小 |
| 结果本来就小，下一步是 pandas / 画图 | `query(..., "DataFrame")` 或 `to_df()` |
| 临时表，跑完就丢，而且放得进内存 | `ENGINE = Memory` 或 `:memory:` |
| 一条 GROUP BY 可能超出内存 | `max_memory_usage` 和 `max_bytes_before_external_group_by` 一起设 |
| 和别人共用机器、或在小容器里 | 连接串写 `max_threads` |

## 读数时的提醒

- 峰值 RSS 不会随查询结束回落。分配器会把页留着复用，所以一个进程里跑过一次大查询之后，RSS 基本停在那个水位。要看“这一次查询用了多少”，看 `MemoryTracking` 的差，或者像基准那样用子进程隔离。
- 这台机器很快，缓存也热。0.2 秒以下的差别是底噪；相差 3 倍以上的才值得当结论。
- 其他机器上绝对值会变，相对关系（物化 vs 流式、CSV vs Parquet、精确 vs 近似）不会。
