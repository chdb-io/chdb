# 一个小的资源监控

嵌入式会话没有独立的服务进程，监控就是在同一个进程里采样。可运行的版本是 [`examples/example_resource_monitor.py`](../../examples/example_resource_monitor.py)。

## 读什么

| 来源 | 含义 |
| --- | --- |
| `system.metrics`，`metric = 'MemoryTracking'` | 引擎当前记账的分配，单位字节 |
| `system.events` 的 `Query`、`SelectQuery`、`SelectedRows`、`SelectedBytes` | 进程启动以来的累计计数。做差才是这一段的增量 |
| `/proc/self/status` 的 `VmRSS`（macOS 用 `resource.getrusage` 的 `ru_maxrss`） | 进程实际驻留。和 `MemoryTracking` 不是同一个数 |

`MemoryTracking` 把引擎分配器里的块都算进去，RSS 只算驻留页。示例里一次 `groupArray` 期间，引擎记账大约 590MB，RSS 大约 320MB。用它们看趋势，不要要求两个数相等。

`system.query_log` 在 26.9 的嵌入式会话里不存在。`system.user_query_log` 表在，`DESCRIBE` 能看到 `memory_usage` 和 `query_duration_ms`，但执行查询再 `SYSTEM FLUSH LOGS` 之后行数仍是 0。不要靠这张表做历史查询。`system.asynchronous_metrics` 也不在。

## 用另一个会话采样

同一个 `Session` 上，重查询会把后面的 `query()` 排在它后面。采样线程如果共用那个会话，往往要等重查询结束才拿到一个点。另开一个 `Session()`，采样可以在重查询进行时返回。两个会话看到的 `system.metrics` 是这个进程的数。

```python
import threading
import time
from chdb import session

workload = session.Session()
monitor = session.Session()
samples = []
stop = threading.Event()

def loop():
    while not stop.wait(0.05):
        text = monitor.query(
            """
            SELECT
                (SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'),
                (SELECT value FROM system.events WHERE event = 'SelectedRows')
            """,
            "TSV",
        ).bytes().decode().strip()
        memory_tracking, selected_rows = text.split("\t")
        samples.append((int(memory_tracking), int(selected_rows)))

thread = threading.Thread(target=loop, daemon=True)
thread.start()
workload.query(
    """
    SELECT sleep(0.4), length(groupArray(number))
    FROM numbers(1500000)
    SETTINGS max_threads = 1
    """,
    "Null",
)
stop.set()
thread.join()
```

示例在这条约 0.4 秒的查询里采到 7 个点。查询比采样间隔还短时，点数会变成 1，那只说明查询结束得快。

## 把样本写成 MergeTree

采样先放在 Python 列表里，查询结束后一次性 `INSERT`。采样线程不要往正在被重查询占用的表里并发写。

```sql
CREATE TABLE resource_samples (
    ts DateTime64(3),
    memory_tracking UInt64,
    rss UInt64,
    queries UInt64,
    selected_rows UInt64
)
ENGINE = MergeTree
ORDER BY ts
```

之后可以用 SQL 看峰值，而不必把监控本身再做成一套服务：

```sql
SELECT
    count() AS samples,
    max(memory_tracking) AS max_engine_bytes,
    max(rss) AS max_rss_bytes,
    max(selected_rows) AS selected_rows
FROM resource_samples
```

会话带目录（`Session("monitor_db")`）时这张表会留下来，下次进程还能查。`:memory:` 会话结束，表就没了。

## DataStore 自己的计时

只想看 DataStore 哪一步慢，用它自带的 profiler，不必再采样 RSS：

```python
from chdb.datastore.config import config, get_profiler

config.enable_profiling()
# ... 跑管道 ...
get_profiler().report()
```

文档站上的说明：[Profiling](https://clickhouse.com/docs/chdb/debugging/profiling)。单条 SQL 的计划用 `EXPLAIN`，见 [SQL 技巧](sql-tricks.md)。
