# 查询时用得上的 SQL

目标是少读、少排序、少把中间结果拉进 Python。每一条都附上在 26.9 上量到的数字，来自 [`benchmark/perf_guide/bench_sql.py`](../../benchmark/perf_guide/bench_sql.py)。数据是 4117 万行出租车 Parquet，以及一周 1348 万笔 aggTrades 导入的 MergeTree 表（`ORDER BY (symbol, ts, agg_trade_id)`）。机器和方法见 [总览](index.md)；0.2 秒以下的耗时差别是底噪，重点看 `SelectedBytes` 和峰值内存。

## 精确还是近似：差 3 到 6 倍

4117 万行上数不同的上车时间（真值 20,049,544）：

| 写法 | 结果 | 误差 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- | --- |
| `uniqExact(col)` | 20,049,544 | 0 | 0.56 s | 3,030 MB |
| `count(DISTINCT col)` | 20,049,544 | 0 | 0.55 s | 3,064 MB |
| `uniq(col)` | 20,230,981 | +0.9% | 0.27 s | 854 MB |
| `uniqCombined(col)` | 20,053,162 | +0.02% | 0.14 s | 856 MB |

分位数（`total_amount` 的 p50 / p90 / p99，真值 21 / 56.76 / 104.88）：

| 写法 | 结果 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- |
| `quantilesExact(0.5, 0.9, 0.99)(x)` | 21 / 56.76 / 104.88 | 0.92 s | 2,229 MB |
| `quantiles(...)(x)`（水塘抽样） | 21.02 / 56.93 / 104.88 | 0.17 s | 585 MB |
| `quantilesTDigest(...)(x)` | 21.005 / 56.78 / 104.905 | 0.16 s | 759 MB |

精确版本要把所有不同值（或所有值）放进内存，2 到 3 GB 就是这么来的。报表、监控、特征统计用 `uniqCombined` 和 `quantileTDigest`；对账、去重才用精确版本。

## 每组取最后一行：argMax

一周两个合约的逐笔，取每个 `(symbol, minute)` 的最后成交价（20,160 组）：

| 写法 | 耗时 | 峰值 RSS | 结果 |
| --- | --- | --- | --- |
| `argMax(price, (ts, agg_trade_id))` + `GROUP BY` | 0.61 s | 692 MB | 20,160 行 |
| `row_number() OVER (PARTITION BY ... ORDER BY ts DESC) = 1` | 0.64 s | 1,238 MB | 20,160 行 |
| `ORDER BY ts DESC LIMIT 1 BY symbol, minute` | 2.03 s | 1,159 MB | 20,160 行 |
| 先 `max(ts)` 再 JOIN 回原表 | 0.45 s | 734 MB | **38,765 行**：同一毫秒多笔成交全被带回来了 |

- `argMax` 是一次扫描加一张聚合表，内存最省。元组做比较键可以处理时间戳相同的情况。
- 窗口函数内存多一倍，因为每个分区的行要先攒起来排序。
- `LIMIT 1 BY` 要先对 1348 万行做一次全局 `ORDER BY`，这里最慢。它适合“已经有序、或者数据很少、或者每组要前 N 行”的情况，不是去重的通用写法。
- 自连接看起来最快，但结果是错的（多了 92%）。要用它，JOIN 键必须唯一。

```sql
SELECT symbol, toStartOfMinute(ts) AS minute,
       argMax(price, (ts, agg_trade_id)) AS last_price
FROM ticks
GROUP BY symbol, minute
```

## PREWHERE

MergeTree 上 `WHERE qty > 5` 然后 `sum(price * qty)`，1348 万行里 11 万行命中：

| 写法 | SelectedBytes | 耗时 |
| --- | --- | --- |
| `optimize_move_to_prewhere = 0` | 216 MB | 0.20 s |
| 默认（自动移到 PREWHERE） | 175 MB | 0.21 s |
| 手写 `PREWHERE qty > 5` | 175 MB | 0.31 s |

引擎默认会把过滤条件挪到 `PREWHERE`：先只读 `qty` 这一列，命中的行再去读 `price`。这里少读 19%；命中率越低、其余列越宽，省得越多。手写 `PREWHERE` 和自动的结果一样，一般不必写。Parquet 没有 `PREWHERE` 语法，但过滤下推到 reader 之后效果类似，并且 row group 统计还能把整组跳过（见 [归档](archiving-parquet.md)）。

## 按主键顺序读

取 BTCUSDT 最新的 10 笔：

```sql
SELECT ts, price FROM ticks
WHERE symbol = 'BTCUSDT'
ORDER BY ts DESC
LIMIT 10
```

| 设置 | SelectedRows | SelectedBytes |
| --- | --- | --- |
| `optimize_read_in_order = 1`（默认） | 289,550 | 4.9 MB |
| `optimize_read_in_order = 0` | 9,456,409 | 86 MB |

表是 `ORDER BY (symbol, ts, ...)`，查询按 `ts` 排序时引擎从有序的 part 尾部开始读，读够 10 行就停。关掉之后要把 946 万行全读出来排序。前提是查询的 `ORDER BY` 和表的 `ORDER BY` 前缀一致。

## count() 几乎免费

| 写法 | SelectedBytes |
| --- | --- |
| `SELECT count() FROM file('taxi/*.parquet', Parquet)` | 8 KB |
| `SELECT count(total_amount) FROM ...` | 68.7 MB |
| `SELECT count() FROM ... WHERE total_amount > 20` | 68.7 MB |

`count()` 不带列、不带条件时直接读 Parquet 的元数据。只要加了列或条件，就要读那一列。

## 类型：LowCardinality 和 Nullable

把 4117 万行出租车导进两张 MergeTree 表，一张原样（`vendor String, payment String, passenger_count Nullable(Float64)`），一张调过（`LowCardinality(String)`，`passenger_count Float64` 用 `ifNull` 补 0）：

| | 原样 | 调过 |
| --- | --- | --- |
| 磁盘大小 | 496 MB | 441 MB |
| 按 `(vendor, payment)` 分组：SelectedBytes | 1,482 MB | 727 MB |
| `WHERE payment = 'pay_1' AND passenger_count > 2`：SelectedBytes | 964 MB | 637 MB |

`LowCardinality` 把字符串换成字典编号，分组和比较都在编号上做，读的字节减半。`Nullable` 多一列空值标记，过滤和聚合都要看它。两张表的耗时在这台机器上都在 0.4 到 0.6 秒，差别在底噪里；读字节减半在慢盘和大表上才会变成时间。

- 取值很少的字符串用 `LowCardinality(String)`（合约、交易所、买卖方向）。
- 列实际上没有空值时不要声明 `Nullable`。
- 训练要用的浮点在 SQL 里 `toFloat32`，避免默认 `Float64` 进张量后再转一次。

## 组合子

- `sumIf(qty, px > 100)`、`countIf(symbol = 'AAPL')`：条件聚合，一次扫描出多个口径。
- `argMin / argMax`：开盘、收盘、最新值。
- `-State / -Merge`：预聚合落盘，之后合并（`uniqCombinedState` 存进表，查询时 `uniqCombinedMerge`）。

## SAMPLE 在这个版本上没有减少读取

MergeTree 可以 `ORDER BY k SAMPLE BY k`，查询写 `SAMPLE 0.1`。在 26.9 上分析器是强制打开的（`enable_analyzer` 不能关）。对一张 200 万行、`SAMPLE BY k` 的表，`SELECT count(k) FROM sm SAMPLE 0.1` 和 `SELECT sum(v) FROM sm SAMPLE 0.1` 返回的都是全表的数，`EXPLAIN` 仍然是 61/61 个 granule。不要把 `SAMPLE` 当成这里的抽样手段。要抽一部分行，用 `cityHash64(key) % N = shard`，和 [机器学习](machine-learning.md) 里的分片是同一种条件。

## DataStore 里哪条路径会下推

布尔索引会下推到 `file()`，`ds.sql()` 不会。全年数据上取两列、2,282 行：

| 写法 | 耗时 | 峰值 RSS | SelectedRows | SelectedBytes |
| --- | --- | --- | --- | --- |
| `ds[(ds["total_amount"] > 200) & (ds["passenger_count"] >= 4)][["total_amount", "trip_distance"]]` | 0.25 s | 609 MB | 68,194 | 64 MB |
| `ds.sql("SELECT total_amount, trip_distance FROM __df__ WHERE ...")` | 7.23 s | 13,730 MB | 41,172,975 | 693 MB |

`sql()` 先把 `__df__` 物化成一个 DataFrame 再在上面跑 SQL。省内存的过滤走布尔索引，要控制 `PREWHERE`、`LIMIT BY`、`SETTINGS` 时直接用 `session.Session().query` 或 `send_query` 对 `file()` 写。`explain()` 用来确认数据源还是 `file(...)`、计划是不是还懒着：

```python
lazy = ds[ds["px"] > 100]
lazy.explain()   # 先看计划
lazy.to_df()     # 再取结果
```

performance 模式会拿掉保序和一部分 pandas 包装，聚合更容易合成一条 SQL。模式本身的差异见 [Performance mode](https://clickhouse.com/docs/chdb/configuration/performance-mode)。
