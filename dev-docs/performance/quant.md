# 量化：行情不要整表放进内存

逐笔数据按天、按合约堆在 Parquet 或 MergeTree 里。K 线、滚动特征、成交对报价的对齐都在 SQL 里做完，Python 只接收最后那张小表。

这一页的数字来自 [`benchmark/perf_guide/bench_quant.py`](../../benchmark/perf_guide/bench_quant.py)，数据是 Binance 现货 aggTrades（BTCUSDT 2024 年 1 月 31 天 4196 万笔、ETHUSDT 2106 万笔，Hive 目录 `ticks/day=YYYY-MM-DD/symbol=XXX/trades.parquet`，共 527 MB zstd Parquet）和同月 BTCUSDT 的 1 秒 K 线。机器和方法见 [总览](index.md)。可直接运行的小例子是 [`examples/example_quant_bars.py`](../../examples/example_quant_bars.py)。

## 分钟线：31 天 4196 万笔 → 44,640 根 bar

```sql
SELECT
    toStartOfMinute(ts) AS minute,
    argMin(price, (ts, agg_trade_id)) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, (ts, agg_trade_id)) AS close,
    sum(qty) AS volume,
    sum(price * qty) / sum(qty) AS vwap,
    count() AS trades
FROM file('ticks/day=*/symbol=BTCUSDT/trades.parquet', Parquet)
GROUP BY minute
ORDER BY minute
```

| 做法 | 耗时 | 峰值 RSS | 说明 |
| --- | --- | --- | --- |
| SQL，`"DataFrame"` 输出 | 1.49 s | 1,637 MB | 读 4 列，Python 只拿到 44,640 行 |
| SQL，`iter_batches` 流式接收 bar | 2.30 s | 1,919 MB | 结果本来就小，流式在这里没有意义 |
| SQL，数据在 MergeTree（见下） | 1.15 s | 458 MB | |
| pandas：31 个文件 `read_table` → `concat` → `sort_values` → `resample("1min")` | 13.93 s | 8,486 MB | DataFrame 1,678 MB，排序和 resample 的中间量占了其余 |
| pandas：一天一天读、算、拼 | 7.81 s | 1,014 MB | 内存回到合理水平，耗时仍是 SQL 的 5 倍 |

`argMin(price, (ts, agg_trade_id))` 用元组做比较键，同一毫秒多笔成交时按 `agg_trade_id` 决定先后，开盘收盘才是确定的。pandas 那边要先 `sort_values(["ts", "agg_trade_id"])` 才能让 `first`/`last` 等价；排序、`notional` 临时列和 `resample` 的分组中间量叠在 1.7 GB 的 DataFrame 之上，才有 8.5 GB 的峰值。

四种做法算出来的 `sum(volume)` 一致（1,403,408.85），都已核对。

## 只读一天：目录裁剪

一天一个目录，目录名要是 `day=YYYY-MM-DD`。`use_hive_partitioning` 在 26.9 里默认就是 1，目录名里的 `day`、`symbol` 会成为虚拟列。

```sql
INSERT INTO FUNCTION file('ticks/day={_partition_id}/trades.parquet', 'Parquet')
PARTITION BY toString(toDate(ts))
SELECT ...
```

取 2024-01-15 这一天 BTCUSDT 的 `count(), sum(qty)`：

| 写法 | SelectedBytes | 耗时 |
| --- | --- | --- |
| 读全月 31 天（对照）：`WHERE symbol = 'BTCUSDT'` | 68.4 MB | 0.17 s |
| 直接给路径 `file('ticks/day=2024-01-15/symbol=BTCUSDT/trades.parquet')` | 2.2 MB | 0.14 s |
| 通配 `file('ticks/**/*.parquet') WHERE day = '2024-01-15' AND symbol = 'BTCUSDT'` | 2.2 MB | 0.14 s |
| 通配，按时间列过滤 `WHERE ts >= '2024-01-15' AND ts < '2024-01-16'` | 3.3 MB | 0.21 s |
| 同上，关掉 `input_format_parquet_filter_push_down` | 3.3 MB | 0.22 s |
| 通配，`WHERE toDate(ts) = '2024-01-15'`（函数包着列） | 3.4 MB | 0.14 s |

按目录列过滤和直接写路径一样省：引擎在打开文件之前就把其余 30 天跳过了。按 `ts` 过滤要多读一列，并且是打开每个文件后靠 row group 的 min/max 统计把其他天的 row group 跳掉（文件是按 `ts` 排过序写出去的，统计信息才有用）。两种都远小于全月的 68 MB。

一个顺带的发现：`WHERE qty > 50` 这种和排序无关的条件也只读了 3 MB——写 Parquet 时每个 row group 都带 `qty` 的 min/max，绝大多数 row group 的最大 `qty` 都到不了 50。只要条件是简单的列比较，Parquet 的统计信息都会被利用。

## 同一份历史要反复查：MergeTree

Parquet 适合按天扫描。同一份数据要按合约和时间反复点查时，放进带目录的会话：

```python
from chdb import session

conn = session.Session("market_db")   # 目录留在磁盘上
conn.query("""
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
""")
conn.query("INSERT INTO ticks SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker FROM file('ticks/**/*.parquet', Parquet)")
```

| 步骤 | 数字 |
| --- | --- |
| `INSERT ... SELECT` 导入 6302 万行 | 1.6 s，峰值 3.6 GB |
| 导入后的 parts 数 | 548 个（每个分区一批一个） |
| `OPTIMIZE TABLE ticks FINAL` 合并到每个分区 1 个 part | 7.9 s，剩 32 个 part |
| 磁盘占用 | 844 MB；同一份数据 zstd Parquet 是 527 MB |

MergeTree 默认 LZ4 压缩，所以比 zstd Parquet 大 60%。列上可以写 `CODEC(ZSTD)`，这是空间和解压速度的取舍。

取 2024-01-15 12:00 到 13:00 这一小时的 BTCUSDT（28,489 笔）：

| 来源 | SelectedRows | SelectedBytes | 耗时 |
| --- | --- | --- | --- |
| Parquet，目录列 + `ts` 范围 | 28,541 | 227 KB | 0.17 s |
| Parquet，只用 `ts` 范围（不写 `day =`） | 28,541 | 227 KB | 0.19 s |
| MergeTree，主键 `(symbol, ts)` | 41,036 | 924 KB | 0.19 s |

两边都只碰了几万行。MergeTree 多读的 12,500 行是索引粒度（8,192 行一个 mark）造成的边角；`SelectedBytes` 是解压后的字节，和 Parquet 的压缩字节没有可比性。这个数据量下耗时都在工具的底噪里，差别出现在更大的表、更高的查询频率上：MergeTree 不用每次解析 Parquet 的 footer 和 schema，而且分区可以 `DROP PARTITION`（见 [归档](archiving-parquet.md)）。

全月分钟线在 MergeTree 上是 1.15 s / 458 MB，比 Parquet 的 1.49 s / 1,637 MB 省内存，因为 MergeTree 按 mark 读、不用整个 row group 解压到内存。

## 成交对齐报价：ASOF JOIN

每笔成交带上不晚于该时间的最近一根 1 秒 K 线。ASOF JOIN 需要至少一个等值条件，再加上一个不等式：

```sql
SELECT t.ts, t.price, k.close
FROM (SELECT ts, price, 'BTCUSDT' AS symbol FROM file('ticks/day=2024-01-15/symbol=BTCUSDT/trades.parquet', Parquet)) AS t
ASOF LEFT JOIN (SELECT ts, close, 'BTCUSDT' AS symbol FROM file('klines/day=2024-01-15/klines.parquet', Parquet)) AS k
    ON t.symbol = k.symbol AND t.ts >= k.ts
```

一天 136 万笔成交对 86,400 根 K 线：

| 做法 | 耗时 | 峰值 RSS |
| --- | --- | --- |
| ASOF JOIN，输出 `count(), avg(price - close)` | 0.18 s | 437 MB |
| `pd.merge_asof` | 0.12 s | 508 MB |

一天的量 pandas 也放得下，两者耗时相当。差别在于 SQL 这条不要求两边都先排好序、也不要求同一天的数据都在内存里：把文件路径换成 31 天的通配、`symbol` 换成真实列，语句不变，内存不变。

## 滚动窗口

```sql
SELECT ts, price,
       avg(price) OVER (ORDER BY ts, agg_trade_id ROWS BETWEEN 999 PRECEDING AND CURRENT ROW) AS ma
FROM file('ticks/day=2024-01-15/symbol=BTCUSDT/trades.parquet', Parquet)
```

| 窗口 | SQL 窗口函数 | pandas `rolling().mean()` |
| --- | --- | --- |
| 100 笔 | 0.21 s / 404 MB | 0.21 s / 529 MB |
| 1000 笔 | 0.30 s / 406 MB | 0.21 s / 527 MB |

逐笔级别的滚动均值，pandas 和 SQL 打平，窗口变大时 pandas 的滑动算法还略占优。SQL 的意义不在单日：

- 31 天分钟线上叠 20/60 根均线和 1 分钟收益率，从 4196 万笔原始成交一条语句算到 44,640 行结果：0.95 s，峰值 1,275 MB。pandas 要先把 31 天拼出来（上文 8.5 GB）。
- 窗口函数的输出行数等于输入行数。只要均线、不要逐笔时，外面再套一层 `GROUP BY`，不要把逐笔级别的窗口结果收成 DataFrame。

```sql
WITH bars AS (
    SELECT toStartOfMinute(ts) AS minute, argMax(price, (ts, agg_trade_id)) AS close
    FROM file('ticks/day=*/symbol=BTCUSDT/trades.parquet', Parquet)
    GROUP BY minute
)
SELECT
    minute, close,
    avg(close) OVER (ORDER BY minute ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
    avg(close) OVER (ORDER BY minute ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
    close / lagInFrame(close, 1) OVER (ORDER BY minute) - 1 AS ret
FROM bars
```

## 回测里的内存上限

回测查询加上内存上限，避免一次 `GROUP BY` 把进程打满。阈值按机器改，`max_memory_usage` 和 `max_bytes_before_external_group_by` 要一起设，原因和数字见 [总览](index.md#限制这一次查询的内存)：

```sql
SELECT symbol, toStartOfMinute(ts) AS minute, sum(qty) AS volume
FROM ticks
WHERE symbol = 'BTCUSDT' AND ts >= '2024-01-15' AND ts < '2024-01-16'
GROUP BY symbol, minute
SETTINGS max_memory_usage = 4000000000, max_bytes_before_external_group_by = 1000000000
```

## 最后才变成数组

特征矩阵进模型之前再转。前面的聚合如果已经把行数降下来，`query(..., "Arrow")` 或 `"DataFrame"` 都可以。行数还是逐笔级别时，用 [pipeline](machine-learning.md) 按批拿，不要 `to_df()`。
