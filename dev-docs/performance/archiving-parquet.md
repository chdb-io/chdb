# 用 SQL 把历史数据归档成 Parquet

热数据留在 MergeTree，按天分区。过期的分区先写成 Parquet，再 `DROP PARTITION`。查询侧对两边用同一套 SQL。可运行的流程是 [`examples/example_archive_parquet.py`](../../examples/example_archive_parquet.py)：250 行、三天的表，归档 `2024-01-01` 之后 MergeTree 剩 150 行，Parquet 里仍是那天的 100 行。

## 表按天分区

```sql
CREATE TABLE ticks (
    day Date,
    symbol LowCardinality(String),
    ts DateTime64(3),
    px Float64,
    qty UInt32
)
ENGINE = MergeTree
PARTITION BY day
ORDER BY (symbol, ts)
```

`PARTITION BY day` 时，`system.parts` 里的分区名就是 `2024-01-01` 这种日期。没有分区键的表，`DROP PARTITION tuple()` 会把仅有的那一块丢掉，不适合用来滚动历史。

## 写出一个分区

路径里的 `{_partition_id}` 会被换成 `PARTITION BY` 表达式的值，于是目录自然是 Hive 风格的 `day=2024-01-01/`。

```sql
INSERT INTO FUNCTION file('archive/day={_partition_id}/ticks.parquet', 'Parquet')
PARTITION BY toString(day)
SELECT day, symbol, ts, px, qty
FROM ticks
WHERE day = toDate('2024-01-01')
ORDER BY symbol, ts
SETTINGS output_format_parquet_compression_method = 'zstd'
```

26.9 上这几项的默认值：

| 设置 | 默认 |
| --- | --- |
| `output_format_parquet_compression_method` | `zstd` |
| `output_format_parquet_row_group_size` | `1000000` |
| `use_hive_partitioning` | `1` |
| `input_format_parquet_filter_push_down` | `1` |
| `input_format_parquet_bloom_filter_push_down` | `1` |

`ORDER BY symbol, ts` 让一个 row group 里的统计信息（min/max）跟查询条件同序，后面的过滤才剪得掉 row group。row group 太大，剪枝变粗；太小，文件碎片变多。默认 100 万行对日内逐笔通常够用，要改就在这条 `INSERT` 的 `SETTINGS` 里写 `output_format_parquet_row_group_size`。

确认写出去了再删：

```sql
SELECT day, count()
FROM file('archive/**/*.parquet', Parquet)
GROUP BY day
ORDER BY day
SETTINGS use_hive_partitioning = 1
```

```sql
ALTER TABLE ticks DROP PARTITION '2024-01-01'
```

`DROP PARTITION` 只删 MergeTree 里的那一天。Parquet 文件还在。

## 对象存储是同一条语句

`s3` 表函数在引擎里。有桶的时候把 `file(...)` 换成 `s3(...)`，`PARTITION BY` 和 `{_partition_id}` 照旧：

```sql
INSERT INTO FUNCTION s3(
    'https://bucket.s3.amazonaws.com/ticks/day={_partition_id}/ticks.parquet',
    'Parquet'
)
PARTITION BY toString(day)
SELECT day, symbol, ts, px, qty
FROM ticks
WHERE day = toDate('2024-01-01')
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

Hive 目录已经带了 `day=` 时，日期列会自动出现，不必再从 `_path` 里解析。过滤写在这个列上：

```sql
SELECT count()
FROM file('archive/**/*.parquet', Parquet)
WHERE day = toDate('2024-01-02')
SETTINGS use_hive_partitioning = 1
```

热数据和归档要一起查时，用 `UNION ALL`：一边是 MergeTree，一边是 `file()`。列名和类型对齐即可。不要为了这个把归档再载回一张 Memory 表。
