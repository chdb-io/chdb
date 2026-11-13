# 用 SQL 把历史数据归档成 Parquet

热数据留在 MergeTree，按天分区。过期的分区先写成 Parquet，再 `DROP PARTITION`。查询侧对两边用同一套 SQL。

这一页的数字来自 [`benchmark/perf_guide/bench_archive.py`](../../benchmark/perf_guide/bench_archive.py)：写入测试用一周两个合约的 aggTrades（1348 万行，源文件 113 MB zstd Parquet），分区写入测试用一个月出租车数据（296 万行，50 MB）。机器和方法见 [总览](index.md)。可直接运行的小例子是 [`examples/example_archive_parquet.py`](../../examples/example_archive_parquet.py)。

## 写 Parquet 的三个开关

写法只有一条语句，`SETTINGS` 决定文件长什么样：

```sql
INSERT INTO FUNCTION file('archive/week.parquet', Parquet)
SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker
FROM ticks
ORDER BY symbol, ts, agg_trade_id
SETTINGS
    output_format_parquet_compression_method = 'zstd',
    output_format_parquet_row_group_size = 1000000
```

26.9 上相关设置的默认值：

| 设置 | 默认 |
| --- | --- |
| `output_format_parquet_compression_method` | `zstd` |
| `output_format_parquet_row_group_size` | `1000000` |
| `use_hive_partitioning` | `1` |
| `input_format_parquet_filter_push_down` | `1` |
| `input_format_parquet_bloom_filter_push_down` | `1` |

### 压缩算法

1348 万行、6 列，写出再整表 `count(), sum(qty)`：

| 算法 | 文件大小 | 写耗时 | 整表读耗时 |
| --- | --- | --- | --- |
| none | 329 MB | 0.78 s | 0.11 s |
| snappy | 190 MB | 0.62 s | 0.14 s |
| lz4 | 188 MB | 0.66 s | 0.15 s |
| zstd（默认） | 113 MB | 0.66 s | 0.16 s |
| gzip | 123 MB | 0.59 s | 0.24 s |

zstd 比 lz4/snappy 小 40%，写和读的耗时都在同一量级（读的差别 0.05 秒，在底噪里）。没有理由改默认值。gzip 文件更大、读更慢，不要选。

### row group 大小和排序

同一周数据，zstd，改 row group 行数，然后查一小时 BTCUSDT 的范围、再按 `agg_trade_id` 点查一行：

| 写法 | 文件大小 | 一小时范围：SelectedBytes | 点查一行：SelectedBytes |
| --- | --- | --- | --- |
| 100,000 行一组，按 `(symbol, ts)` 排序 | 168 MB | 317 KB | 211 KB |
| 1,000,000 行一组（默认），排序 | 113 MB | 176 KB | 52 KB |
| 10,000,000 行一组，排序 | 114 MB | 419 KB | 76 KB |
| 1,000,000 行一组，**随机顺序** | 253 MB | 88,707 KB | 269 KB |

三件事：

- **排序比什么都重要。** 同样的 1,000,000 行一组，排好序的范围查询读 176 KB，随机顺序读 88 MB，差 500 倍；文件本身也大了 2.2 倍，因为相邻值不相似压不动。归档时一定 `ORDER BY` 查询最常用的列。
- row group 太小会让文件变大（100,000 行一组比默认大 50%，每组的字典和统计信息是固定开销），对裁剪精度帮助有限。
- row group 太大裁剪变粗（10,000,000 行一组范围查询多读 1.4 倍）。默认 1,000,000 对日内逐笔是合适的；一天 100 万到 300 万笔的合约，一天就是 1 到 3 个 row group。

点查一行只读 52 KB，是因为 `agg_trade_id` 列上同时有 min/max 统计和 bloom filter 在起作用（两个下推开关默认都开）。

## 按天写出：`PARTITION BY`

路径里的 `{_partition_id}` 会被换成 `PARTITION BY` 表达式的值，目录自然就是 Hive 风格的 `day=2024-01-01/`：

```sql
INSERT INTO FUNCTION file('archive/day={_partition_id}/trips.parquet', Parquet)
PARTITION BY toDate(tpep_pickup_datetime)
SELECT * FROM file('taxi/yellow_tripdata_2024-01.parquet', Parquet)
WHERE toYYYYMM(tpep_pickup_datetime) = 202401
SETTINGS output_format_parquet_compression_method = 'zstd'
```

一个月 296 万行写成 31 个文件：

| 做法 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| 一条 `INSERT ... PARTITION BY` | 1.86 s | 1,633 MB |
| Python 循环 31 天，每天一条 `INSERT ... WHERE toDate(...) = day` | 3.15 s | 1,079 MB |

一条语句扫一遍源文件；循环扫 31 遍，源文件在页缓存里时也慢 70%。`PARTITION BY` 版本的峰值高一些，多半是 31 个输出文件的写缓冲同时打开。分区数上千时（比如按小时又按合约）改成分批循环，每批几十个分区。

## MergeTree 侧：导出、丢分区

```sql
CREATE TABLE ticks (
    symbol LowCardinality(String),
    ts DateTime64(3),
    agg_trade_id Int64,
    price Float64,
    qty Float64,
    is_buyer_maker Bool
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (symbol, ts, agg_trade_id)
```

`PARTITION BY toYYYYMMDD(ts)` 时分区名是 `20240101` 这样的整数。没有分区键的表只有一块，`DROP PARTITION tuple()` 会把整张表丢掉，不适合滚动历史。

一周数据（1348 万行、78 个 part、184 MB）上：

| 步骤 | 耗时 | 结果 |
| --- | --- | --- |
| 把 2024-01-01 分区写成 Parquet（71 万行） | 0.32 s | 5.7 MB 文件 |
| `ALTER TABLE ticks DROP PARTITION 20240101` | 0.22 s | 表剩 1277 万行 |
| 对照：`ALTER TABLE ticks DELETE WHERE toYYYYMMDD(ts) = 20240102`（`mutations_sync = 2`） | 2.26 s | 表剩 1044 万行 |

`DROP PARTITION` 是改目录，和分区里有多少行无关；`DELETE` 是 mutation，要把涉及的 part 全部重写一遍，这里慢 10 倍，表大了慢得更多。按天分区的意义就在这里。

导出先于删除。确认写出去了再删：

```sql
SELECT day, count()
FROM file('archive/**/*.parquet', Parquet)
GROUP BY day
ORDER BY day
```

## 对象存储是同一条语句

`s3` 表函数在引擎里。有桶的时候把 `file(...)` 换成 `s3(...)`，`PARTITION BY` 和 `{_partition_id}` 照旧：

```sql
INSERT INTO FUNCTION s3(
    'https://bucket.s3.amazonaws.com/ticks/day={_partition_id}/ticks.parquet',
    'Parquet'
)
PARTITION BY toString(toDate(ts))
SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker
FROM ticks
WHERE toYYYYMMDD(ts) = 20240101
ORDER BY symbol, ts
```

密钥用环境变量或 `s3` 参数传入，不要写进仓库。读回来是 `FROM s3('https://...', 'Parquet')`。本地没有桶时，示例只演示 `file()`。

## 读的时候带上虚拟列

`_file` 和 `_path` 是 `file()` 提供的，不在 Parquet 文件里。用来核对哪一天来自哪个文件：

```sql
SELECT _file, count(), min(ts), max(ts)
FROM file('archive/**/*.parquet', Parquet)
GROUP BY _file
ORDER BY _file
```

Hive 目录已经带了 `day=` 时，日期列会自动出现，过滤写在这个列上就能在打开文件前跳过其他目录（数字见 [量化](quant.md#只读一天目录裁剪)）。

热数据和归档要一起查时，用 `UNION ALL`：一边是 MergeTree，一边是 `file()`。列名和类型对齐即可。不要为了这个把归档再载回一张 Memory 表。
