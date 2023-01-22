import pyarrow as pa
f1 = pa.OSFile("/home/Clickhouse/bench/result.arrow")
af = pa.ipc.open_file(f1, 23922).read_all()
print(type(af)) # pyarrow.lib.Table
print(af)
