# 量化：行情不要整表放进内存

逐笔数据按天、按合约堆在 Parquet 或 MergeTree 里。K 线、滚动特征、成交对报价的对齐都在 SQL 里做完，Python 只接收最后那张小表。

可运行的对照是 [`examples/example_quant_bars.py`](../../examples/example_quant_bars.py)。它用 SQL 写出 200 万行 tick（约 58MB Parquet，另有两列信号列是聚合用不到的），再分别用 SQL 和 pandas 做分钟线，并打印两次进程的峰值 RSS。

## 按天落盘

一天一个目录，文件名不重要，目录名要是 `day=YYYY-MM-DD`。`use_hive_partitioning` 在 26.9 里默认就是 1，查询带上日期过滤时引擎可以只打开对应目录。

```sql
INSERT INTO FUNCTION file('ticks/day={_partition_id}/ticks.parquet', 'Parquet')
PARTITION BY toString(day)
SELECT
    toDate('2024-01-02') AS day,
    ['AAPL', 'MSFT', 'NVDA', 'AMZN'][1 + (number % 4)] AS symbol,
    toDateTime64('2024-01-02 09:30:00', 3) + (number % 23400) AS ts,
    100 + (number % 50) + randCanonical() AS px,
    1 + (number % 20) AS qty
FROM numbers(2000000)
```

`{_partition_id}` 会换成 `PARTITION BY` 的值。同一条路径在 [归档](archiving-parquet.md) 里用来把 MergeTree 的分区搬出去。

只读一天：

```sql
SELECT count()
FROM file('ticks/day=2024-01-02/*.parquet', Parquet)
```

多天一起读时用 `file('ticks/**/*.parquet', Parquet)`，日期列来自目录名。

## 分钟线留在 SQL 里

`toStartOfInterval` 切 bar，`argMin` / `argMax` 按时间取开盘和收盘，不用把逐笔拉进 Python 再排序。

```sql
SELECT
    symbol,
    toStartOfInterval(ts, INTERVAL 1 MINUTE) AS bar,
    argMin(px, ts) AS open,
    max(px) AS high,
    min(px) AS low,
    argMax(px, ts) AS close,
    sum(qty) AS volume
FROM file('ticks.parquet', Parquet)
GROUP BY symbol, bar
ORDER BY symbol, bar
```

滚动均价用窗口函数，仍然只返回计算后的列：

```sql
SELECT
    symbol,
    ts,
    px,
    avg(px) OVER (
        PARTITION BY symbol
        ORDER BY ts
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS ma
FROM file('ticks.parquet', Parquet)
```

窗口的输出行数等于输入行数。只要均价、不要逐笔时，继续在外面 `GROUP BY`，不要把窗口结果收成 DataFrame。

## 成交对齐报价：ASOF JOIN

每笔成交带上不晚于该时间的最近一笔报价。两边都要能按 `(symbol, ts)` 有序比较。

```sql
SELECT t.symbol, t.ts, t.px, q.bid
FROM file('ticks.parquet', Parquet) AS t
ASOF LEFT JOIN file('quotes.parquet', Parquet) AS q
    ON t.symbol = q.symbol AND t.ts >= q.ts
ORDER BY t.symbol, t.ts
```

这是示例里核对过的写法。报价表远小于逐笔表时，JOIN 的输出仍然是逐笔粒度；回测只需要 bar 的话，先聚合成 K 线再 JOIN。

## 同一份历史要反复查：MergeTree

Parquet 适合按天扫描。同一份数据要按合约和时间反复点查时，放进带目录的会话，让主键就是查询条件：

```python
from chdb import session

conn = session.Session("market_db")  # 目录留在磁盘上
conn.query("""
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
""")
```

`LowCardinality(String)` 适合合约代码这种取值很少的列。`PARTITION BY day` 之后，过期的一天可以用 `ALTER TABLE ticks DROP PARTITION '2024-01-01'` 丢掉，不必重写整张表。把丢掉的那天先写成 Parquet 的步骤在 [归档](archiving-parquet.md)。

回测查询加上内存上限，避免一次 `GROUP BY` 把进程打满。阈值按机器改：

```sql
SELECT symbol, sum(qty) AS volume
FROM ticks
WHERE symbol = 'AAPL'
  AND day = toDate('2024-01-02')
GROUP BY symbol
SETTINGS max_memory_usage = 4000000000
```

## 最后才变成数组

特征矩阵进模型之前再转。前面的聚合如果已经把行数降下来，`query(..., "Arrow")` 或 `"DataFrame"` 都可以。行数还是逐笔级别时，用 [pipeline](machine-learning.md) 按批拿，不要 `to_df()`。
