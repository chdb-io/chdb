# 一个小的资源监控

嵌入式会话没有独立的服务进程，监控就是在同一个进程里采样。可运行的版本是 [`examples/example_resource_monitor.py`](../../examples/example_resource_monitor.py)；基准里的 `MemoryTrackingSampler`（[`benchmark/perf_guide/common.py`](../../benchmark/perf_guide/common.py)）是同一个思路，本页所有"引擎峰值"都是它采出来的。

## 读什么

| 来源 | 含义 |
| --- | --- |
| `system.metrics`，`metric = 'MemoryTracking'` | 引擎当前记账的分配，单位字节 |
| `system.events` 的 `Query`、`SelectQuery`、`SelectedRows`、`SelectedBytes` | 进程启动以来的累计计数。做差才是这一段的增量 |
| `/proc/self/status` 的 `VmRSS` / `VmHWM`（macOS 用 `resource.getrusage` 的 `ru_maxrss`） | 进程实际驻留 / 历史最高驻留。和 `MemoryTracking` 不是同一个数 |

`system.query_log` 在 26.9 的嵌入式会话里不存在。`system.user_query_log` 表在，`DESCRIBE` 能看到 `memory_usage` 和 `query_duration_ms`，但执行查询再 `SYSTEM FLUSH LOGS` 之后行数仍是 0。不要靠这张表做历史查询。`system.asynchronous_metrics` 也不在。

## 两个数为什么不一样

一条 4117 万行、4058 万组的 `GROUP BY`，每 50 毫秒采一次 `MemoryTracking` 和 `VmRSS`（单位 MB）：

| 时间 | MemoryTracking | VmRSS |
| --- | --- | --- |
| 0.25 s | 140 | 984 |
| 0.41 s | 2,693 | 3,719 |
| 0.56 s | 3,778 | 5,410 |
| 0.72 s | 6,800 | 8,632 |
| 0.83 s | **8,620** | 10,138 |
| 0.93 s | 6,022 | 10,704 |
| 1.03 s（查询结束） | 1,564 | **10,783** |

- 查询进行中 RSS 比记账高 1.5 到 2 GB：分配器的碎片和缓存，加上 Python、共享库。
- 查询结束后记账回落到 1.5 GB，RSS 停在 10.8 GB 不动。分配器把页留着给下一次用。所以看 RSS 只能知道"这个进程曾经用到多少"，看 `MemoryTracking` 才知道"现在正在用多少"。
- 容器的内存上限是按 RSS 杀的。给上限时用 RSS 的峰值加余量，不要用 `MemoryTracking`。

同一条查询加上 `max_bytes_before_external_group_by = 1 GiB`：

| 时间 | MemoryTracking | VmRSS |
| --- | --- | --- |
| 0.44 s | 1,491 | 2,356 |
| 0.65 s | 1,081 | 3,221 |
| 0.86 s | 453 | 3,625 |
| 1.07 s | 974 | 4,198 |
| 1.39 s | 959 | 4,175 |
| 1.50 s（查询结束） | 129 | 4,176 |

记账在 0.5 到 1.5 GB 之间锯齿：每到阈值就把一块聚合状态写到磁盘再清空。RSS 的峰值从 10.8 GB 降到 4.2 GB，耗时从 1.06 秒到 1.54 秒。[总览](index.md#限制这一次查询的内存)里有这组设置的完整对照表。

## 采样本身花多少

| 项目 | 数字 |
| --- | --- |
| 一次 `SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'` | 0.94 ms |
| 一次读 4 个 `system.events` 计数 | 1.97 ms |
| 20 条 4117 万行的聚合，不采样 | 1.47 s |
| 同上，每 100 ms 采一次（采到 13 个点） | 1.45 s |
| 同上，每 10 ms 采一次（99 个点） | 1.48 s |

采样对被测查询没有可测量的影响。10 ms 一次也只是用掉一个线程的十分之一。要在意的只有一点：采样必须在**另一个** `Session` 上。

## 用另一个会话采样

同一个 `Session` 上，重查询会把后面的 `query()` 排在它后面。采样线程如果共用那个会话，往往要等重查询结束才拿到一个点。另开一个 `Session()`，采样可以在重查询进行时返回。两个会话看到的 `system.metrics` 是这个进程的数。

```python
import threading
import time
from chdb import session

workload = session.Session()
monitor = session.Session(":memory:?max_threads=1")
samples = []
stop = threading.Event()


def read_rss_kb():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])


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
        samples.append((time.time(), int(memory_tracking), read_rss_kb() * 1024, int(selected_rows)))


thread = threading.Thread(target=loop, daemon=True)
thread.start()
workload.query("SELECT ... FROM file('taxi/*.parquet', Parquet) GROUP BY ...", "Null")
stop.set()
thread.join()
```

第一次采样会晚 0.2 秒左右，是监控会话自己的启动时间。要从 0 开始采，就在启动工作查询之前先让采样线程跑一轮。

查询比采样间隔还短时，点数会变成 1，那只说明查询结束得快。

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
    max(selected_rows) - min(selected_rows) AS selected_rows
FROM resource_samples
```

会话带目录（`Session("monitor_db")`）时这张表会留下来，下次进程还能查。`:memory:` 会话结束，表就没了。

## 给查询加上保险

监控告诉你峰值在哪，`SETTINGS` 让它别再到那里。两者配合的做法：

1. 第一次跑新查询时采样，记下 `MemoryTracking` 的峰值 P。
2. 之后给这条查询写 `max_memory_usage = 1.5 P`，并把 `max_bytes_before_external_group_by` 设在 P / 3 左右。
3. 捕获 `chdb.ChdbError`（Code 241 `MEMORY_LIMIT_EXCEEDED`），日志里记下这条 SQL，然后用更低的外部聚合阈值重试一次。

## DataStore 自己的计时

只想看 DataStore 哪一步慢，用它自带的 profiler，不必再采样 RSS：

```python
from chdb.datastore.config import config, get_profiler

config.enable_profiling()
# ... 跑管道 ...
get_profiler().report()
```

文档站上的说明：[Profiling](https://clickhouse.com/docs/chdb/debugging/profiling)。单条 SQL 的计划用 `EXPLAIN`，见 [SQL 技巧](sql-tricks.md)。
