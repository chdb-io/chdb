import pyarrow as pa
import example

# 获取 arrow::Table 对象
table = example.queryToArrow("SELECT * FROM file('/home/Clickhouse/bench/result.parquet', Parquet) LIMIT 10")

# 使用 pyarrow.lib.Table.from_batches 方法将其转换为 pyarrow.lib.Table
table = pa.lib.Table.from_batches([table])