# 查询时用得上的 SQL

这些写法都在 chdb-core 26.9.0（引擎 26.9.2.1）的嵌入式会话里跑过。目标是少读、少排序、少把中间结果拉进 Python。

## PREWHERE

MergeTree 上把最能剪行的条件放进 `PREWHERE`，先读这一列，再读其余列：

```sql
SELECT count()
FROM ticks
PREWHERE symbol = 'AAPL'
WHERE px > 10
```

`symbol` 用 `LowCardinality(String)`，并且在 `ORDER BY (symbol, ts)` 的前面，这个条件才剪得动主键。

Parquet 没有 `PREWHERE` 语法，但过滤会下推。下面这条的 `EXPLAIN` 里能看到 `Prewhere filter column`：

```sql
EXPLAIN indexes = 1
SELECT number
FROM file('flat.parquet', Parquet)
WHERE k = 0
SETTINGS input_format_parquet_filter_push_down = 1
```

`input_format_parquet_filter_push_down` 和 `input_format_parquet_bloom_filter_push_down` 默认就是 1。写出 Parquet 时如果按过滤列排过序（见 [归档](archiving-parquet.md)），row group 的 min/max 才能把整组跳掉。

## LIMIT BY，而不是窗口去重

每个合约只要最新的一行：

```sql
SELECT symbol, ts, px
FROM ticks
ORDER BY symbol, ts DESC
LIMIT 1 BY symbol
```

`LIMIT n BY key` 在排序后的流上截断，不必 `row_number() OVER (PARTITION BY symbol) = 1` 再套一层子查询。

## argMax 代替自连接

同一分组里“时间最大的那一行的价格”就是 `argMax(px, ts)`，开盘是 `argMin(px, ts)`。K 线的完整写法在 [量化](quant.md)。不要先找 `max(ts)` 再 JOIN 回原表，除非还要带回很多没有聚合函数能表达的列。

## 组合子和近似聚合

- `sumIf(qty, px > 100)`、`countIf(symbol = 'AAPL')`：条件聚合，少一次扫描。
- `uniq(symbol)`：基数估计，比 `count(DISTINCT symbol)` 省内存。要精确就用 `uniqExact`。
- `quantileTDigest(0.5)(px)`：分位数的 t-digest 近似。要精确用 `quantileExact`，它会吃更多内存。

```sql
SELECT uniq(symbol), quantileTDigest(0.5)(px)
FROM ticks
```

## 按主键顺序读

`optimize_read_in_order` 默认是 1。`ORDER BY` 和表的 `ORDER BY` 一致时，引擎可以按已经有序的部分读，而不是再排一遍。表是 `ORDER BY (symbol, ts)` 时，查询就写成 `ORDER BY symbol, ts`，不要写成相反的表达式再指望它命中。

## 类型

- 取值很少的字符串用 `LowCardinality(String)`（合约、交易所、买卖方向）。
- 列实际上没有空值时不要声明 `Nullable`。`Nullable` 让过滤和聚合都去看一张额外的标记。
- 训练要用的浮点在 SQL 里 `toFloat32`，避免默认 `Float64` 进张量后再转一次。

## SAMPLE 在这个版本上没有减少读取

MergeTree 可以 `ORDER BY k SAMPLE BY k`，查询写 `SAMPLE 0.1`。在 26.9 上分析器是强制打开的（`enable_analyzer` 不能关）。对一张 200 万行、`SAMPLE BY k` 的表，`SELECT count(k) FROM sm SAMPLE 0.1` 和 `SELECT sum(v) FROM sm SAMPLE 0.1` 返回的都是全表的数，`EXPLAIN` 仍然是 61/61 个 granule。不要把 `SAMPLE` 当成这里的抽样手段。要抽一部分行，用 `cityHash64(key) % N = shard`，和 [机器学习](machine-learning.md) 里的分片是同一种条件。

## DataStore 里哪条路径会下推

布尔索引会下推到 `file()`。20 万行 Parquet 上，`ds[ds["id"] == 1].to_df()` 的 `system.events` `SelectedRows` 增量是几百行。

```python
import chdb.datastore as pd

ds = pd.read_parquet("ticks.parquet")
print(ds[ds["id"] == 1].to_df())
```

`ds.sql("SELECT count() FROM __df__ WHERE id = 1")` 结果是对的，但同一次对比里读行数是整表量级（大约两遍文件），生成的计划里用户这条 SQL 也没有并进 `file()` 的 `SELECT`。省内存不要走 `sql()` 做过滤。`explain()` 用来确认数据源还是 `file(...)`、计划是不是还懒着：

```python
lazy = ds[ds["px"] > 100]
lazy.explain()   # 先看计划
lazy.to_df()     # 再取结果
```

要控制 `PREWHERE`、`LIMIT BY`、`SETTINGS` 时，直接用 `session.Session().query` 或 `send_query`。那条 SQL 就是发给引擎的 SQL。

performance 模式会拿掉保序和一部分 pandas 包装，聚合更容易合成一条 SQL。模式本身的差异见 [Performance mode](https://clickhouse.com/docs/chdb/configuration/performance-mode)。
